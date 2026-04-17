"""Head-tracking driver for the vision thread.

Same motion profile as `vision/face_recognition/track_and_turn_head.py`:
the servo is stepped at ~30 Hz in ≤6-tick micro-moves, which gives the
smooth "following my face" feel. In WALL-E the face *detection* runs at
the vision service's slow rate (1-2 Hz on Jetson because the Coral
subprocess IPC is the bottleneck), so we split the job into two layers:

1. VisionService calls `update_from_face(x, frame_width)` whenever it
   has a fresh detection. That call only recomputes the *target* tick
   — no serial I/O.
2. A background thread inside HeadTracker ticks at ~30 Hz and walks
   `current_tick` toward `target_tick` in the same 6-tick chunks the
   standalone script uses, sending one `head step N` per tick. Serial
   I/O stays under the firmware's per-command delta cap.

Body-assist spin turns from the standalone are intentionally dropped —
the voice pipeline drives the wheels, and having the tracker also spin
the base would fight the drive tool.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

_log = logging.getLogger("walle.vision.head_tracker")

# Servo tick map — same as the reference script. tick=150 is the
# robot's physical right, tick=550 is its physical left.
HEAD_RIGHT_TICK = 150
HEAD_CENTER_TICK = 350
HEAD_LEFT_TICK = 550
HEAD_MIN_TICK = min(HEAD_RIGHT_TICK, HEAD_LEFT_TICK)
HEAD_MAX_TICK = max(HEAD_RIGHT_TICK, HEAD_LEFT_TICK)
HEAD_CENTER_TRIM = 0
HEAD_OPTICAL_CENTER_TICK = HEAD_CENTER_TICK + HEAD_CENTER_TRIM

# Smoothing / deadband — match the standalone script.
MEDIAN_WINDOW = 3
SMOOTH_ALPHA = 0.45
DEADBAND_PX = 30
REVERSE_HYSTERESIS_PX = 15
EDGE_MARGIN_RATIO = 0.18

# Motion parameters — copied from the reference's HeadController. The
# tick-thread runs at TICK_HZ and moves at most MAX_STEP_CAP ticks each
# iteration. At 30 Hz × 6 ticks = 180 ticks/sec, a full-range swing
# takes ~2.2 s — visibly smooth, same as standalone.
TICK_HZ = 30
TICK_INTERVAL_SEC = 1.0 / TICK_HZ
MAX_STEP_BASE = 1
MAX_STEP_EXTRA = 6
MAX_STEP_CAP = 6
MIN_SEND_DELTA = 1

# Firmware's per-command tick-delta safety ceiling. Even a single step
# is well below this; kept here as a hard cap for `notify_manual_tick`
# resync and for clarity.
HEAD_MAX_STEP_PER_CMD = 60

# After this many seconds with no face, drift back to centre.
LOST_TARGET_TIMEOUT_SEC = 1.8


class HeadTracker:
    """Minimal head-only tracker driven by face coordinates.

    Call `update_from_face()` whenever new face data is available (any
    rate — 1 Hz on Jetson is fine). The internal 30 Hz thread walks the
    servo toward the most recent target in firmware-legal micro-steps,
    identical to the standalone script.
    """

    def __init__(
        self,
        send_command: Callable[[str], object],
        *,
        manual_override_sec: float = 4.0,
    ) -> None:
        self._send = send_command
        self._manual_override_sec = manual_override_sec

        import numpy as np  # noqa: PLC0415 — lazy for text-mode hosts
        self._np = np

        self._lock = threading.Lock()
        self._enabled = True
        self._running = False

        self.current_tick = HEAD_OPTICAL_CENTER_TICK
        self._target_tick = HEAD_OPTICAL_CENTER_TICK
        self._last_sent_tick: Optional[int] = None
        self._last_send_monotonic = 0.0

        self._recent_face_x: deque = deque(maxlen=MEDIAN_WINDOW)
        self._smoothed_x: Optional[float] = None
        self._last_face_seen_at = 0.0
        self._last_command_dir = 0
        self._error_ratio = 0.0  # |err| / half_frame, set in update_from_face

        # Suspend window for explicit voice "look left" / head_pan calls.
        self._suspend_until = 0.0

        # Send an initial absolute centre so the firmware's internal
        # position matches our `current_tick` state.
        try:
            self._send(f"head tick {self.current_tick}")
        except Exception:
            _log.debug("initial head-center send failed", exc_info=True)

        self._running = True
        self._thread = threading.Thread(
            target=self._tick_loop, name="head-tracker-tick", daemon=True
        )
        self._thread.start()

    # ----- public API ------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled
            if not enabled:
                self._reset_tracking_state_locked()

    def suspend(self, duration_sec: Optional[float] = None) -> None:
        """Pause tracking for N seconds (e.g. after a manual head_pan)."""
        dur = (
            duration_sec if duration_sec is not None else self._manual_override_sec
        )
        with self._lock:
            self._suspend_until = time.monotonic() + dur
            self._reset_tracking_state_locked()

    def notify_manual_tick(self, tick: int) -> None:
        """External code (voice head_pan) just moved the servo — resync."""
        with self._lock:
            self.current_tick = _clamp_tick(tick)
            self._target_tick = self.current_tick
            self._suspend_until = time.monotonic() + self._manual_override_sec
            self._reset_tracking_state_locked()

    def update_from_face(self, face_center_x: float, frame_width: int) -> None:
        """Recompute the target tick from a new face observation.

        Cheap — only updates state, no serial I/O. The tick thread picks
        up the new target on its next iteration.
        """
        with self._lock:
            if not self._enabled:
                return
            if time.monotonic() < self._suspend_until:
                return
            self._update_target_locked(face_center_x, frame_width)

    def on_no_face(self) -> None:
        """Called when a vision frame contained no face."""
        now = time.monotonic()
        with self._lock:
            if not self._enabled:
                return
            if now < self._suspend_until:
                return
            if self._last_face_seen_at == 0.0:
                # Never seen a face — stay centred.
                self._target_tick = HEAD_OPTICAL_CENTER_TICK
                return
            if now - self._last_face_seen_at >= LOST_TARGET_TIMEOUT_SEC:
                self._reset_tracking_state_locked()
                self._target_tick = HEAD_OPTICAL_CENTER_TICK

    def close(self) -> None:
        self._running = False
        t = getattr(self, "_thread", None)
        if t is not None:
            t.join(timeout=1.0)
        try:
            self._send(f"head tick {HEAD_OPTICAL_CENTER_TICK}")
        except Exception:
            _log.debug("close head-center send failed", exc_info=True)

    # ----- internals -------------------------------------------------

    def _reset_tracking_state_locked(self) -> None:
        self._recent_face_x.clear()
        self._smoothed_x = None
        self._last_command_dir = 0
        self._error_ratio = 0.0

    def _update_target_locked(
        self, face_center_x: float, frame_width: int
    ) -> None:
        np = self._np
        frame_center_x = frame_width / 2.0
        self._recent_face_x.append(face_center_x)
        filtered_x = float(np.median(self._recent_face_x))

        if self._smoothed_x is None:
            self._smoothed_x = filtered_x
        else:
            self._smoothed_x = (
                (1.0 - SMOOTH_ALPHA) * self._smoothed_x
                + SMOOTH_ALPHA * filtered_x
            )

        raw_error_px = self._smoothed_x - frame_center_x

        desired_dir = 0
        if raw_error_px < -DEADBAND_PX:
            desired_dir = 1
        elif raw_error_px > DEADBAND_PX:
            desired_dir = -1

        # Reverse-direction hysteresis — stay put on small oscillations.
        if (
            self._last_command_dir != 0
            and desired_dir != 0
            and desired_dir != self._last_command_dir
            and abs(raw_error_px) < (DEADBAND_PX + REVERSE_HYSTERESIS_PX)
        ):
            desired_dir = 0

        if abs(raw_error_px) < DEADBAND_PX:
            desired_dir = 0

        self._last_face_seen_at = time.monotonic()

        if desired_dir == 0:
            # Inside deadband — hold target at current position so the
            # tick thread stops stepping.
            self._target_tick = self.current_tick
            self._error_ratio = 0.0
            return

        edge_margin_px = frame_width * EDGE_MARGIN_RATIO
        left_x = edge_margin_px
        right_x = frame_width - edge_margin_px
        mapped_x = min(max(self._smoothed_x, left_x), right_x)

        target_tick = float(
            np.interp(
                mapped_x, [left_x, right_x], [HEAD_LEFT_TICK, HEAD_RIGHT_TICK]
            )
        )
        self._target_tick = _clamp_tick(target_tick)
        self._error_ratio = abs(raw_error_px) / max(1.0, frame_center_x)

    # --- 30 Hz tick thread ---

    def _tick_loop(self) -> None:
        next_tick = time.monotonic()
        while self._running:
            next_tick += TICK_INTERVAL_SEC
            self._do_tick()
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # We fell behind — reset the schedule so drift doesn't
                # compound into runaway sends.
                next_tick = time.monotonic()

    def _do_tick(self) -> None:
        with self._lock:
            if not self._enabled:
                return
            if time.monotonic() < self._suspend_until:
                return
            delta = self._target_tick - self.current_tick
            if delta == 0:
                return
            dynamic_max_step = min(
                MAX_STEP_BASE + int(self._error_ratio * MAX_STEP_EXTRA),
                MAX_STEP_CAP,
            )
            if delta > dynamic_max_step:
                step = dynamic_max_step
            elif delta < -dynamic_max_step:
                step = -dynamic_max_step
            else:
                step = int(delta)
            new_tick = _clamp_tick(self.current_tick + step)
            actual_step = new_tick - self.current_tick
            if actual_step == 0:
                return
            if (
                self._last_sent_tick is not None
                and abs(new_tick - self._last_sent_tick) < MIN_SEND_DELTA
            ):
                return
            # Firmware cap — our step is always ≤ MAX_STEP_CAP ≤ 60,
            # so this is a safety check, not the common path.
            if abs(actual_step) > HEAD_MAX_STEP_PER_CMD:
                actual_step = (
                    HEAD_MAX_STEP_PER_CMD
                    if actual_step > 0
                    else -HEAD_MAX_STEP_PER_CMD
                )
                new_tick = _clamp_tick(self.current_tick + actual_step)
            self.current_tick = new_tick
            if actual_step > 0:
                self._last_command_dir = 1
            elif actual_step < 0:
                self._last_command_dir = -1
            self._last_sent_tick = new_tick
            self._last_send_monotonic = time.monotonic()

        # Serial I/O OUTSIDE the lock — firmware ack is 10-50 ms and we
        # don't want face-update callers to block on it.
        try:
            self._send(f"head step {actual_step}")
        except Exception:
            _log.debug("head step send failed", exc_info=True)


def _clamp_tick(tick) -> int:
    return max(HEAD_MIN_TICK, min(HEAD_MAX_TICK, int(round(tick))))

"""Head-tracking driver for the vision thread.

Ports the tracking math from `vision/face_recognition/track_and_turn_head.py`
into a form VisionService can feed at frame rate, using the shared
`SerialManager` instead of opening its own serial port.

Only head tracking — body-assist turns from the standalone script are
dropped so the tracker never fights with voice-driven drive commands.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

_log = logging.getLogger("walle.vision.head_tracker")

# Servo tick map matches the reference script and the firmware's head
# step-walking bounds. Optical center trim is zero — adjust here if the
# mounted camera is biased off-axis relative to the servo.
HEAD_RIGHT_TICK = 150
HEAD_CENTER_TICK = 350
HEAD_LEFT_TICK = 550
HEAD_MIN_TICK = min(HEAD_RIGHT_TICK, HEAD_LEFT_TICK)
HEAD_MAX_TICK = max(HEAD_RIGHT_TICK, HEAD_LEFT_TICK)
HEAD_CENTER_TRIM = 0
HEAD_OPTICAL_CENTER_TICK = HEAD_CENTER_TICK + HEAD_CENTER_TRIM

# Tracking parameters. The reference script ran the loop at ~30 FPS and
# clamped each update to a ≤6-tick micro-step for visual smoothness. In
# voice mode VisionService ticks at ~1 FPS (Jetson CPU budget), so a
# per-update cap of a few ticks means the servo can traverse at most
# ~6 ticks/second — 60+ seconds to cross the 400-tick range. Instead
# we walk straight to the target per update, splitting the move into
# firmware-legal 60-tick `head step` chunks inside _send_tick so the
# firmware's per-command rate limit never rejects it.
MEDIAN_WINDOW = 3
SMOOTH_ALPHA = 0.55
DEADBAND_PX = 30
REVERSE_HYSTERESIS_PX = 15
SEND_INTERVAL_SEC = 0.0  # vision loop paces us — no extra throttle
MIN_SEND_DELTA = 2
LOST_TARGET_TIMEOUT_SEC = 1.8
RETURN_TO_CENTER_STEP = 40
EDGE_MARGIN_RATIO = 0.18

# Firmware safety: each `head step N` command may move at most this many
# ticks (motor_control_cli.ino enforces). Larger requested deltas are
# step-walked.
HEAD_MAX_STEP_PER_CMD = 60
STEP_WALK_DELAY_SEC = 0.02


class HeadTracker:
    """Minimal head-only tracker driven by face coordinates.

    Caller must supply a `send_command(cmd: str)` callable that writes
    one CLI line (without trailing newline) to the firmware — typically
    bound to `SerialManager.send_command`.
    """

    def __init__(
        self,
        send_command: Callable[[str], object],
        *,
        manual_override_sec: float = 4.0,
    ) -> None:
        self._send = send_command
        self._manual_override_sec = manual_override_sec

        # NumPy is imported lazily so this module can be imported in
        # text-mode/test paths that don't pull the vision stack.
        import numpy as np  # noqa: PLC0415

        self._np = np

        self._lock = threading.Lock()
        self._enabled = True

        self.current_tick = HEAD_OPTICAL_CENTER_TICK
        self._last_sent_tick: Optional[int] = None
        self._last_send_time = 0.0

        self._recent_face_x: deque = deque(maxlen=MEDIAN_WINDOW)
        self._smoothed_x: Optional[float] = None
        self._last_target_time = 0.0
        self._last_command_dir = 0

        # When the voice pipeline's head_pan tool runs, suspend vision
        # tracking briefly so the user's explicit "look left" is honoured
        # instead of being overwritten by the next face update.
        self._suspend_until = 0.0

        # Initialize firmware position so subsequent deltas land in
        # a known place. Best-effort — a simulation-mode SerialManager
        # will just log the command. We emit `head tick N` here (not
        # `head step`) because there's no cached firmware position yet;
        # firmware accepts a full-range absolute set at boot.
        try:
            self._send(f"head tick {self.current_tick}")
        except Exception:
            _log.debug("initial head-center send failed", exc_info=True)

    # ---- public API ------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled
            if not enabled:
                self._reset_tracking_state()

    def suspend(self, duration_sec: Optional[float] = None) -> None:
        """Block updates for `duration_sec` seconds after a manual command."""
        dur = (
            duration_sec if duration_sec is not None else self._manual_override_sec
        )
        with self._lock:
            self._suspend_until = time.monotonic() + dur
            self._reset_tracking_state()

    def notify_manual_tick(self, tick: int) -> None:
        """Tell the tracker that an external actor (voice tool) just moved
        the head. Resets the cached `current_tick` so we don't fight it
        and suspends tracking briefly."""
        with self._lock:
            self.current_tick = _clamp_tick(tick)
            self._suspend_until = time.monotonic() + self._manual_override_sec
            self._reset_tracking_state()

    def update_from_face(self, face_center_x: float, frame_width: int) -> None:
        """Feed one face's center-x (pixels, original-frame coords)."""
        with self._lock:
            if not self._enabled:
                return
            if time.monotonic() < self._suspend_until:
                return
            self._update_locked(face_center_x, frame_width)

    def on_no_face(self) -> None:
        """Drift back toward center after `LOST_TARGET_TIMEOUT_SEC` of no faces."""
        with self._lock:
            if not self._enabled:
                return
            if time.monotonic() < self._suspend_until:
                return
            self._on_lost_locked()

    def close(self) -> None:
        try:
            self._send(f"head tick {HEAD_OPTICAL_CENTER_TICK}")
        except Exception:
            _log.debug("close head-center send failed", exc_info=True)

    # ---- internals -------------------------------------------------

    def _reset_tracking_state(self) -> None:
        self._recent_face_x.clear()
        self._smoothed_x = None
        self._last_command_dir = 0

    def _send_tick(self, tick: int, force: bool = False) -> bool:
        """Move the servo to absolute `tick`, step-walking through any
        delta larger than the firmware's per-command limit.

        At 1-2 Hz vision updates a single tracking step can span 300+
        ticks (face jumps left-to-right), so sending one `head tick N`
        with delta > 60 gets clamped by the firmware and the servo
        stalls halfway. We split the move into ≤60-tick `head step`
        commands with a short delay between so the firmware processes
        each one cleanly.
        """
        tick = _clamp_tick(tick)
        now = time.monotonic()
        delta_total = tick - self.current_tick
        if not force:
            if abs(delta_total) < MIN_SEND_DELTA:
                return False
            if now - self._last_send_time < SEND_INTERVAL_SEC:
                return False
        old_tick = self.current_tick

        remaining = delta_total
        while abs(remaining) > HEAD_MAX_STEP_PER_CMD:
            step = (
                HEAD_MAX_STEP_PER_CMD if remaining > 0 else -HEAD_MAX_STEP_PER_CMD
            )
            try:
                self._send(f"head step {step}")
            except Exception:
                _log.debug("head step send failed", exc_info=True)
            self.current_tick += step
            remaining -= step
            if STEP_WALK_DELAY_SEC > 0:
                time.sleep(STEP_WALK_DELAY_SEC)
        if remaining != 0:
            try:
                self._send(f"head step {remaining}")
            except Exception:
                _log.debug("head step send failed", exc_info=True)
            self.current_tick += remaining

        if self.current_tick > old_tick:
            self._last_command_dir = 1
        elif self.current_tick < old_tick:
            self._last_command_dir = -1
        self._last_sent_tick = self.current_tick
        self._last_send_time = time.monotonic()
        return True

    def _update_locked(self, face_center_x: float, frame_width: int) -> None:
        np = self._np
        frame_center_x = frame_width / 2.0
        self._recent_face_x.append(face_center_x)
        filtered_x = float(np.median(self._recent_face_x))

        if self._smoothed_x is None:
            self._smoothed_x = filtered_x
        else:
            self._smoothed_x = (
                (1.0 - SMOOTH_ALPHA) * self._smoothed_x + SMOOTH_ALPHA * filtered_x
            )

        raw_error_px = self._smoothed_x - frame_center_x
        effective_error_px = raw_error_px

        desired_dir = 0
        if raw_error_px < -DEADBAND_PX:
            desired_dir = 1
        elif raw_error_px > DEADBAND_PX:
            desired_dir = -1

        # Hysteresis — don't reverse direction on small oscillations.
        if (
            self._last_command_dir != 0
            and desired_dir != 0
            and desired_dir != self._last_command_dir
            and abs(raw_error_px) < (DEADBAND_PX + REVERSE_HYSTERESIS_PX)
        ):
            desired_dir = 0
            effective_error_px = 0.0

        if abs(raw_error_px) < DEADBAND_PX:
            desired_dir = 0
            effective_error_px = 0.0

        self._last_target_time = time.monotonic()

        edge_margin_px = frame_width * EDGE_MARGIN_RATIO
        left_x = edge_margin_px
        right_x = frame_width - edge_margin_px
        mapped_x = min(max(self._smoothed_x, left_x), right_x)

        target_tick = float(
            np.interp(mapped_x, [left_x, right_x], [HEAD_LEFT_TICK, HEAD_RIGHT_TICK])
        )

        if desired_dir == 0:
            target_tick = self.current_tick

        target_tick = _clamp_tick(target_tick)
        if desired_dir == 0:
            # Inside deadband — hold position rather than drift.
            return
        # Jump straight to target; _send_tick step-walks the delta into
        # firmware-legal 60-tick chunks. At 1-2 Hz vision frames there
        # is no point spreading the move across N update cycles — the
        # face may well have moved again by then.
        self._send_tick(int(round(target_tick)))

    def _on_lost_locked(self) -> None:
        now = time.monotonic()
        if now - self._last_target_time < LOST_TARGET_TIMEOUT_SEC:
            return
        self._reset_tracking_state()
        if self.current_tick < HEAD_OPTICAL_CENTER_TICK:
            new_tick = min(
                self.current_tick + RETURN_TO_CENTER_STEP, HEAD_OPTICAL_CENTER_TICK
            )
        elif self.current_tick > HEAD_OPTICAL_CENTER_TICK:
            new_tick = max(
                self.current_tick - RETURN_TO_CENTER_STEP, HEAD_OPTICAL_CENTER_TICK
            )
        else:
            new_tick = self.current_tick
        self._send_tick(new_tick)


def _clamp_tick(tick) -> int:
    return max(HEAD_MIN_TICK, min(HEAD_MAX_TICK, int(round(tick))))

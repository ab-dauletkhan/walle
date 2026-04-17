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

# Tracking parameters: same as the reference script, tuned at 30 FPS.
# At the voice-mode VISION_FPS (~1-2 Hz) updates are much less frequent,
# so the per-update tick step can actually be larger — but keep it
# conservative so the firmware's 60-tick/command rate limit is never hit.
MEDIAN_WINDOW = 3
SMOOTH_ALPHA = 0.45
DEADBAND_PX = 30
REVERSE_HYSTERESIS_PX = 15
SEND_INTERVAL_SEC = 0.03
MIN_SEND_DELTA = 1
MAX_STEP_BASE = 4
MAX_STEP_EXTRA = 18
MAX_STEP_CAP = 24
LOST_TARGET_TIMEOUT_SEC = 1.8
RETURN_TO_CENTER_STEP = 3
EDGE_MARGIN_RATIO = 0.18


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
        # will just log the command.
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
        tick = _clamp_tick(tick)
        now = time.monotonic()
        if not force:
            if (
                self._last_sent_tick is not None
                and abs(tick - self._last_sent_tick) < MIN_SEND_DELTA
            ):
                return False
            if now - self._last_send_time < SEND_INTERVAL_SEC:
                return False
        old_tick = self.current_tick
        self.current_tick = tick
        try:
            self._send(f"head tick {tick}")
        except Exception:
            _log.debug("head tick send failed", exc_info=True)
        if tick > old_tick:
            self._last_command_dir = 1
        elif tick < old_tick:
            self._last_command_dir = -1
        self._last_sent_tick = tick
        self._last_send_time = now
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
        error_ratio = abs(effective_error_px) / max(1.0, frame_center_x)
        dynamic_max_step = min(
            MAX_STEP_BASE + int(error_ratio * MAX_STEP_EXTRA), MAX_STEP_CAP
        )
        if desired_dir == 0:
            dynamic_max_step = 0

        tick_error = target_tick - self.current_tick
        if tick_error > dynamic_max_step:
            tick_error = dynamic_max_step
        elif tick_error < -dynamic_max_step:
            tick_error = -dynamic_max_step
        else:
            tick_error = int(round(tick_error))

        self._send_tick(int(round(self.current_tick + tick_error)))

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

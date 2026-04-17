"""30 FPS face tracking for the head servo.

The recognition subprocess (SubprocessCoralVisionBackend) runs at
~5-10 FPS because every frame goes through JPEG + base64 + JSON +
detect + embed in the py3.9 Coral worker. That's fine for context and
the greeter, but too slow for the head — the servo visibly lags the
face and jitters when the embedder's match confidence flickers across
the threshold.

FastFaceTracker reuses the *same* Coral worker subprocess (Edge TPU is
single-owner — you can't run two py3.9 workers against it), sending a
lightweight `detect_only` op over the shared CoralWorkerClient. The
caller pre-resizes the frame to the detector's native input on the
main side, so the worker skips its PIL/cv2 resize step. With just
detection (no embedder), the round-trip is ~25-35 ms on Jetson Orin
Nano, enough for ~30 FPS head tracking.

Thread layout:
    VisionService._camera_reader_loop  — reads camera at native FPS
    VisionService._loop                — recognition, ~2-10 FPS, op=detect_recognize
    FastFaceTracker._loop              — detection + head target, ~30 FPS, op=detect_only
    HeadTracker._tick_loop             — servo stepper, 30 Hz
Both the recognition loop and this tracker send ops through the
shared CoralWorkerClient; its lock serializes them so frames don't
interleave on the worker's stdin/stdout.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Optional

import numpy as np

_log = logging.getLogger("walle.vision.fast_tracker")

# 15 Hz is enough for smooth head tracking once HeadTracker's median +
# EMA + deadband smoothing is applied, and halves the CPU / GIL load of
# the detect-only op versus 30 Hz. Moonshine's audio thread needs that
# headroom on the Jetson — at 30 Hz this loop starved the STT pipeline.
_TARGET_FPS = 15
_TARGET_INTERVAL_SEC = 1.0 / _TARGET_FPS
_DETECTION_THRESHOLD = 0.80  # matches track_and_turn_head.FACE_SCORE_THRESHOLD
_FPS_LOG_INTERVAL_SEC = 5.0


class FastFaceTracker:
    """30 Hz loop that drives HeadTracker.update_from_face via a shared
    Coral worker running a detect-only op."""

    def __init__(
        self,
        vision_service,
        head_tracker,
        worker,
        input_w: int,
        input_h: int,
    ) -> None:
        self._vision_service = vision_service
        self._head_tracker = head_tracker
        self._worker = worker
        self._input_w = int(input_w)
        self._input_h = int(input_h)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._cv2 = None  # lazy — set at start() so import errors happen there
        self._frames_since_report = 0
        self._last_fps_report = 0.0

    def start(self) -> None:
        import cv2  # noqa: PLC0415 — optional dep matches VisionService

        self._cv2 = cv2
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="fast-face-tracker", daemon=True
        )
        self._thread.start()
        _log.info(
            "FastFaceTracker started (target %d FPS, input=%dx%d)",
            _TARGET_FPS,
            self._input_w,
            self._input_h,
        )

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None
        _log.info("FastFaceTracker stopped")

    def _loop(self) -> None:
        cv2 = self._cv2
        input_w = self._input_w
        input_h = self._input_h
        next_tick = time.monotonic()
        self._last_fps_report = next_tick
        errors = 0
        while self._running:
            next_tick += _TARGET_INTERVAL_SEC
            try:
                frame_bgr = self._vision_service.get_latest_frame()
                if frame_bgr is None:
                    self._sleep_until(next_tick)
                    continue
                frame_h, frame_w = frame_bgr.shape[:2]
                resized = cv2.resize(
                    frame_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR
                )
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                rgb_b64 = base64.b64encode(rgb.tobytes()).decode("ascii")
                resp = self._worker.request(
                    {
                        "op": "detect_only",
                        "frame_rgb_b64": rgb_b64,
                        "frame_w": int(frame_w),
                        "frame_h": int(frame_h),
                    }
                )
                detections = resp.get("detections") or []
                self._apply_detections(detections, frame_w)
                errors = 0
            except Exception:
                errors += 1
                _log.exception(
                    "FastFaceTracker frame failed (%d consecutive)", errors
                )
                if errors >= 5:
                    _log.error(
                        "FastFaceTracker giving up after %d consecutive errors",
                        errors,
                    )
                    self._running = False
                    break
                time.sleep(0.2)
                continue

            self._tick_fps_report()
            self._sleep_until(next_tick)

    def _apply_detections(
        self, detections: list[dict], frame_width: int
    ) -> None:
        best: Optional[dict] = None
        best_score = -1.0
        for det in detections:
            score = float(det.get("score") or 0.0)
            if score < _DETECTION_THRESHOLD:
                continue
            if score <= best_score:
                continue
            best = det
            best_score = score

        if best is None:
            self._head_tracker.on_no_face()
            return

        x1 = int(best.get("xmin") or 0)
        x2 = int(best.get("xmax") or 0)
        center_x = (x1 + x2) / 2.0
        self._head_tracker.update_from_face(center_x, frame_width)

    def _sleep_until(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    def _tick_fps_report(self) -> None:
        self._frames_since_report += 1
        now = time.monotonic()
        elapsed = now - self._last_fps_report
        if elapsed >= _FPS_LOG_INTERVAL_SEC:
            fps = self._frames_since_report / elapsed
            _log.info("FastFaceTracker fps=%.1f", fps)
            self._frames_since_report = 0
            self._last_fps_report = now

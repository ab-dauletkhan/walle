"""30 FPS face tracking for the head servo.

The recognition subprocess (SubprocessCoralVisionBackend) is the
bottleneck in the main vision loop: JPEG encode + base64 + JSON wrap +
pipe + decode + PIL resize + detect + embed pushes each round-trip to
80-120 ms, capping it at ~10 FPS. That's enough for context updates and
the greeter but too slow for the head servo, which at 10 FPS visibly
lags the face and jitters when the embedder's match confidence
flickers.

FastFaceTracker runs a dedicated Python-3.9 Coral worker doing
*detection only*, fed raw resized frames (no JPEG, no base64) over a
binary-framed pipe. Per-frame budget ~25-35 ms → ~30 FPS on Jetson
Orin Nano, matching the standalone track_and_turn_head.py script.

Thread layout:
    VisionService._camera_reader_loop  — reads camera at native FPS
    VisionService._loop                — recognition path, ~2-10 FPS
    FastFaceTracker._loop              — detection + head target, ~30 FPS
    HeadTracker._tick_loop             — servo stepper, 30 Hz
"""

from __future__ import annotations

import json
import logging
import struct
import subprocess
import threading
import time
from typing import Optional

import numpy as np

from vision.face_recognition.coral_runtime import (
    build_reexec_env,
    resolve_coral_python,
)

_log = logging.getLogger("walle.vision.fast_tracker")

_LENGTH_PREFIX = struct.Struct("<I")  # payload length
_PAYLOAD_HEADER = struct.Struct("<II")  # frame_w, frame_h

_TARGET_FPS = 30
_TARGET_INTERVAL_SEC = 1.0 / _TARGET_FPS
_DETECTION_THRESHOLD = 0.80  # matches track_and_turn_head.FACE_SCORE_THRESHOLD
_FPS_LOG_INTERVAL_SEC = 10.0


class DetectOnlyWorkerClient:
    """Thin client for the py3.9 detect-only Coral worker subprocess."""

    def __init__(self) -> None:
        coral_python = resolve_coral_python()
        if coral_python is None:
            raise RuntimeError(
                "Detect-only Coral worker requires a Python 3.9 runtime. "
                "Set WALLE_CORAL_PYTHON39 or create .venv-coral39 first."
            )

        # bufsize=0 on stdin/stdout so raw-bytes reads/writes don't sit
        # in a text-mode line buffer. We write framed binary on stdin;
        # the worker still writes line-delimited JSON on stdout.
        self._proc = subprocess.Popen(
            [
                coral_python,
                "-m",
                "vision.face_recognition.coral_worker",
                "serve-detect-only",
            ],
            env=build_reexec_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        handshake_line = self._proc.stdout.readline()
        if not handshake_line:
            err = b""
            if self._proc.stderr is not None:
                err = self._proc.stderr.read()
            raise RuntimeError(
                f"Detect-only worker exited before handshake: {err.decode(errors='replace')}"
            )
        hs = json.loads(handshake_line.decode("utf-8"))
        if not hs.get("ok"):
            raise RuntimeError(f"Detect-only worker handshake failed: {hs}")
        self.input_w = int(hs["input_w"])
        self.input_h = int(hs["input_h"])
        _log.info(
            "detect-only worker ready (input=%dx%d, pid=%s)",
            self.input_w,
            self.input_h,
            self._proc.pid,
        )

    def detect(
        self, rgb_input: np.ndarray, frame_w: int, frame_h: int
    ) -> list[dict]:
        """Send one framed request; return parsed detections."""
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"Detect-only worker exited with code {self._proc.returncode}"
            )
        if (
            rgb_input.shape != (self.input_h, self.input_w, 3)
            or rgb_input.dtype != np.uint8
        ):
            raise ValueError(
                f"expected uint8 RGB array of shape "
                f"({self.input_h},{self.input_w},3), got {rgb_input.shape} {rgb_input.dtype}"
            )
        raw = rgb_input.tobytes()
        header = _PAYLOAD_HEADER.pack(int(frame_w), int(frame_h))
        payload = header + raw
        self._proc.stdin.write(_LENGTH_PREFIX.pack(len(payload)))
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()

        response_line = self._proc.stdout.readline()
        if not response_line:
            raise RuntimeError("detect-only worker closed stdout mid-request")
        resp = json.loads(response_line.decode("utf-8"))
        if not resp.get("ok"):
            raise RuntimeError(f"detect-only worker error: {resp.get('error')}")
        return resp.get("detections") or []

    def close(self) -> None:
        proc = self._proc
        if proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()


class FastFaceTracker:
    """Runs a 30 Hz detection loop that drives HeadTracker.update_from_face."""

    def __init__(self, vision_service, head_tracker) -> None:
        self._vision_service = vision_service
        self._head_tracker = head_tracker
        self._worker: Optional[DetectOnlyWorkerClient] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._cv2 = None  # lazy import — stays None until start()
        # FPS instrumentation
        self._frames_since_report = 0
        self._last_fps_report = 0.0

    def start(self) -> None:
        try:
            import cv2  # noqa: PLC0415 — optional dep matches VisionService
        except ImportError as exc:
            raise RuntimeError(
                "FastFaceTracker requires OpenCV in the main runtime."
            ) from exc
        self._cv2 = cv2
        self._worker = DetectOnlyWorkerClient()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="fast-face-tracker", daemon=True
        )
        self._thread.start()
        _log.info("FastFaceTracker started (target %d FPS)", _TARGET_FPS)

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        _log.info("FastFaceTracker stopped")

    def _loop(self) -> None:
        assert self._worker is not None
        assert self._cv2 is not None
        cv2 = self._cv2
        input_w = self._worker.input_w
        input_h = self._worker.input_h
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
                detections = self._worker.detect(rgb, frame_w, frame_h)
                self._apply_detections(detections, frame_w)
                errors = 0
            except Exception:
                errors += 1
                _log.exception(
                    "FastFaceTracker frame failed (%d consecutive)", errors
                )
                if errors >= 5:
                    _log.error(
                        "FastFaceTracker giving up after %d consecutive errors", errors
                    )
                    self._running = False
                    break
                time.sleep(0.2)
                continue

            self._tick_fps_report()
            self._sleep_until(next_tick)

    def _apply_detections(self, detections: list[dict], frame_width: int) -> None:
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

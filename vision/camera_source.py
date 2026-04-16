from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30
DEFAULT_FIRST_FRAME_TIMEOUT_SEC = 3.0


class CameraSource(Protocol):
    backend_name: str

    def read(self) -> tuple[bool, Optional[np.ndarray]]: ...

    def release(self) -> None: ...


@dataclass(frozen=True)
class CameraOpenResult:
    source: Optional[CameraSource]
    first_frame: Optional[np.ndarray]
    backend_name: str
    error: Optional[str] = None


def default_headless_mode(display: Optional[str] = None) -> bool:
    """Default to headless when no GUI display is available."""
    if display is None:
        display = os.environ.get("DISPLAY")
    return not bool(display)


def open_camera_source(
    camera_index: int,
    *,
    width: int = DEFAULT_CAMERA_WIDTH,
    height: int = DEFAULT_CAMERA_HEIGHT,
    fps: int = DEFAULT_CAMERA_FPS,
    first_frame_timeout_sec: float = DEFAULT_FIRST_FRAME_TIMEOUT_SEC,
):
    errors: list[str] = []

    candidates = [
        (
            "opencv-v4l2",
            lambda: OpenCVCameraSource(
                camera_index,
                width=width,
                height=height,
                fps=fps,
                prefer_v4l2=True,
            ),
        ),
        (
            "opencv-default",
            lambda: OpenCVCameraSource(
                camera_index,
                width=width,
                height=height,
                fps=fps,
                prefer_v4l2=False,
            ),
        ),
        (
            "gstreamer",
            lambda: GStreamerCameraSource(
                camera_index,
                width=width,
                height=height,
                fps=fps,
            ),
        ),
    ]

    for backend_name, factory in candidates:
        try:
            source = factory()
        except Exception as exc:
            errors.append(f"{backend_name}: {exc}")
            continue

        ok, frame = _read_first_frame(source, timeout_sec=first_frame_timeout_sec)
        if ok and frame is not None:
            return CameraOpenResult(
                source=source,
                first_frame=frame,
                backend_name=backend_name,
            )

        try:
            source.release()
        except Exception:
            pass
        errors.append(
            f"{backend_name}: no frame received within {first_frame_timeout_sec:.1f}s"
        )

    error = "; ".join(errors) if errors else "no camera backend available"
    return CameraOpenResult(
        source=None,
        first_frame=None,
        backend_name="none",
        error=error,
    )


class OpenCVCameraSource:
    def __init__(
        self,
        camera_index: int,
        *,
        width: int,
        height: int,
        fps: int,
        prefer_v4l2: bool,
    ):
        import cv2

        self._cv2 = cv2
        if prefer_v4l2 and hasattr(cv2, "CAP_V4L2"):
            self.backend_name = "opencv-v4l2"
            self._cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        else:
            self.backend_name = "opencv-default"
            self._cap = cv2.VideoCapture(camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera index {camera_index}")

        self._configure_capture(width=width, height=height, fps=fps)

    def _configure_capture(self, *, width: int, height: int, fps: int) -> None:
        # Prefer MJPG over raw YUYV for USB cameras; it reduces bandwidth and
        # helps avoid V4L2 select() timeouts on Jetson + hub setups.
        if hasattr(self._cv2, "VideoWriter_fourcc") and hasattr(
            self._cv2, "CAP_PROP_FOURCC"
        ):
            fourcc = self._cv2.VideoWriter_fourcc(*"MJPG")
            self._cap.set(self._cv2.CAP_PROP_FOURCC, fourcc)
        if hasattr(self._cv2, "CAP_PROP_FRAME_WIDTH"):
            self._cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, width)
        if hasattr(self._cv2, "CAP_PROP_FRAME_HEIGHT"):
            self._cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, height)
        if hasattr(self._cv2, "CAP_PROP_FPS"):
            self._cap.set(self._cv2.CAP_PROP_FPS, fps)
        if hasattr(self._cv2, "CAP_PROP_BUFFERSIZE"):
            self._cap.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(self._cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            self._cap.set(self._cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self._cap.release()


class GStreamerCameraSource:
    def __init__(self, camera_index: int, *, width: int, height: int, fps: int):
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        self._Gst = Gst
        self.backend_name = "gstreamer"
        Gst.init(None)

        device = f"/dev/video{camera_index}"
        pipeline_candidates = [
            (
                f"v4l2src device={device} do-timestamp=true ! "
                f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1 ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
            ),
            (
                f"v4l2src device={device} do-timestamp=true ! "
                f"video/x-raw,format=YUY2,width={width},height={height} ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
            ),
            (
                f"v4l2src device={device} do-timestamp=true ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
            ),
        ]

        self._pipeline = None
        self._appsink = None
        self._last_error = None

        for pipeline_str in pipeline_candidates:
            try:
                pipeline = Gst.parse_launch(pipeline_str)
                appsink = pipeline.get_by_name("sink")
                if appsink is None:
                    pipeline.set_state(Gst.State.NULL)
                    continue
                pipeline.set_state(Gst.State.PLAYING)
                self._pipeline = pipeline
                self._appsink = appsink
                return
            except Exception as exc:
                self._last_error = str(exc)
                if self._pipeline is not None:
                    self._pipeline.set_state(Gst.State.NULL)
                self._pipeline = None
                self._appsink = None

        raise RuntimeError(self._last_error or "could not create GStreamer pipeline")

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._appsink is None:
            return False, None

        sample = self._appsink.emit("try-pull-sample", 1 * self._Gst.SECOND)
        if sample is None:
            return False, None

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if buffer is None or caps is None:
            return False, None

        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        size = buffer.get_size()
        expected = int(width) * int(height) * 3
        if size < expected:
            return False, None

        raw = buffer.extract_dup(0, expected)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((int(height), int(width), 3))
        return True, frame.copy()

    def release(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(self._Gst.State.NULL)
            self._pipeline = None
            self._appsink = None


def _read_first_frame(source: CameraSource, *, timeout_sec: float):
    result: dict[str, tuple[bool, Optional[np.ndarray]]] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            result["frame"] = source.read()
        finally:
            done.set()

    thread = threading.Thread(target=_worker, daemon=True, name="camera-first-frame")
    thread.start()
    done.wait(timeout_sec)
    if not done.is_set():
        return False, None
    return result.get("frame", (False, None))

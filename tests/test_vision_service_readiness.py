import time

import numpy as np

from vision.camera_source import CameraOpenResult
from walle.memory.context_manager import ContextManager
from walle.vision import service


class DummyBackend:
    def process_frame(self, frame):
        return []

    def close(self):
        return None


class ReadyVisionService(service.VisionService):
    def _create_backend(self):
        return DummyBackend()


class FakeCameraSource:
    backend_name = "gstreamer"

    def __init__(self, frame):
        self._frame = frame
        self.released = False

    def read(self):
        return True, self._frame

    def release(self):
        self.released = True


def test_vision_service_requires_first_frame(monkeypatch):
    monkeypatch.setattr(
        service,
        "open_camera_source",
        lambda *args, **kwargs: CameraOpenResult(
            source=None,
            first_frame=None,
            backend_name="none",
            error="camera timeout",
        ),
    )

    vision = ReadyVisionService(ContextManager(), camera_index=0, fps=1)
    vision.start()

    assert vision.is_active is False
    assert vision.last_error == "camera timeout"
    assert vision.get_latest_frame() is None


def test_vision_service_marks_ready_after_first_frame(monkeypatch):
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    source = FakeCameraSource(frame)
    monkeypatch.setattr(
        service,
        "open_camera_source",
        lambda *args, **kwargs: CameraOpenResult(
            source=source,
            first_frame=frame,
            backend_name="gstreamer",
            error=None,
        ),
    )

    vision = ReadyVisionService(ContextManager(), camera_index=0, fps=5)
    vision.start()
    time.sleep(0.05)

    latest = vision.get_latest_frame()
    assert latest is not None
    assert latest.shape == frame.shape
    assert vision.camera_backend_name == "gstreamer"
    assert vision.is_active is True

    vision.stop()
    assert source.released is True

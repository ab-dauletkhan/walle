import numpy as np

from vision import camera_source


def test_default_headless_mode_follows_display():
    assert camera_source.default_headless_mode("") is True
    assert camera_source.default_headless_mode(":0") is False


def test_default_headless_mode_uses_environment(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    assert camera_source.default_headless_mode("") is True
    assert camera_source.default_headless_mode(None) is True


def test_open_camera_source_falls_back_to_gstreamer(monkeypatch):
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    class FailingOpenCV:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("opencv unavailable")

    class FakeGStreamer:
        def __init__(self, *args, **kwargs):
            self.backend_name = "gstreamer"

        def read(self):
            return True, frame

        def release(self):
            return None

    monkeypatch.setattr(camera_source, "OpenCVCameraSource", FailingOpenCV)
    monkeypatch.setattr(camera_source, "GStreamerCameraSource", FakeGStreamer)

    result = camera_source.open_camera_source(0, first_frame_timeout_sec=0.1)

    assert result.source is not None
    assert result.backend_name == "gstreamer"
    assert result.first_frame is not None
    assert result.first_frame.shape == (4, 4, 3)

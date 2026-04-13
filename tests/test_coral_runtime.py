from vision.face_recognition import coral_runtime
from walle.vision import service


def test_resolve_coral_python_prefers_explicit_env(monkeypatch, tmp_path):
    fake_python = tmp_path / "python3.9"
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_python.chmod(0o755)

    monkeypatch.setenv(coral_runtime.PYTHON39_ENV, str(fake_python))

    assert coral_runtime.resolve_coral_python() == str(fake_python)


def test_should_delegate_edge_tpu_respects_local_flag(monkeypatch):
    monkeypatch.setattr(coral_runtime, "running_on_python39_or_lower", lambda: False)
    monkeypatch.setenv(coral_runtime.LOCAL_ENV_FLAG, "1")

    assert coral_runtime.should_delegate_edge_tpu(True) is False


def test_build_default_vision_registry_prefers_subprocess_backend(monkeypatch):
    monkeypatch.setattr(service.conf, "VISION_CORAL_ENABLED", True)
    monkeypatch.setattr(service, "running_on_python39_or_lower", lambda: False)

    registry = service.build_default_vision_registry()

    assert registry._factories[0][0] == "coral-subprocess"
    assert registry._factories[-1][0] == "cpu"

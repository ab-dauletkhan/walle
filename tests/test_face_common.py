from vision.face_recognition import common
from vision.face_recognition.errors import FaceRecognitionError


def test_create_interpreter_falls_back_to_libedgetpu_soversion(monkeypatch, tmp_path):
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"fake")

    attempted = []

    class DummyInterpreter:
        def __init__(self, model_path, experimental_delegates=None):
            self.model_path = model_path
            self.experimental_delegates = experimental_delegates

    def fake_load_delegate(name):
        attempted.append(name)
        if name == "libedgetpu.so.1.0":
            raise ValueError("missing")
        return f"delegate:{name}"

    monkeypatch.setattr(
        common,
        "require_tflite_runtime",
        lambda: (DummyInterpreter, fake_load_delegate),
    )

    interpreter = common.create_interpreter(model_path, edge_tpu=True)

    assert attempted == ["libedgetpu.so.1.0", "libedgetpu.so.1"]
    assert interpreter.experimental_delegates == ["delegate:libedgetpu.so.1"]


def test_create_interpreter_raises_clear_error_when_all_delegate_names_fail(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"fake")

    class DummyInterpreter:
        def __init__(self, model_path, experimental_delegates=None):
            self.model_path = model_path
            self.experimental_delegates = experimental_delegates

    def fake_load_delegate(name):
        raise ValueError(f"{name} missing")

    monkeypatch.setattr(
        common,
        "require_tflite_runtime",
        lambda: (DummyInterpreter, fake_load_delegate),
    )

    try:
        common.create_interpreter(model_path, edge_tpu=True)
    except FaceRecognitionError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected FaceRecognitionError")

    assert "libedgetpu.so.1.0" in message
    assert "libedgetpu.so.1" in message

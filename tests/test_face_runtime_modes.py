from vision.face_recognition import common


def test_live_edge_tpu_defaults_to_mixed_mode_on_linux_arm(monkeypatch):
    monkeypatch.setattr(common.platform, "system", lambda: "Linux")
    monkeypatch.setattr(common.platform, "machine", lambda: "aarch64")

    assert common.recommended_live_edge_tpu_modes(True) == (True, False)


def test_live_edge_tpu_defaults_to_dual_tpu_off_linux_arm(monkeypatch):
    monkeypatch.setattr(common.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(common.platform, "machine", lambda: "arm64")

    assert common.recommended_live_edge_tpu_modes(True) == (True, True)
    assert common.recommended_live_edge_tpu_modes(False) == (False, False)

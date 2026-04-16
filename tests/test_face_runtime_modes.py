from vision.face_recognition import common
from vision.face_recognition import track_and_turn_head


def test_live_edge_tpu_defaults_to_mixed_mode_on_linux_arm(monkeypatch):
    monkeypatch.setattr(common.platform, "system", lambda: "Linux")
    monkeypatch.setattr(common.platform, "machine", lambda: "aarch64")

    assert common.recommended_live_edge_tpu_modes(True) == (True, False)


def test_live_edge_tpu_defaults_to_dual_tpu_off_linux_arm(monkeypatch):
    monkeypatch.setattr(common.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(common.platform, "machine", lambda: "arm64")

    assert common.recommended_live_edge_tpu_modes(True) == (True, True)
    assert common.recommended_live_edge_tpu_modes(False) == (False, False)


def test_track_head_parser_supports_headless_mode():
    parser = track_and_turn_head.build_parser()

    assert parser.parse_args(["--headless"]).headless is True
    assert parser.parse_args(["--window"]).headless is False

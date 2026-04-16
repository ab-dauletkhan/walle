from walle import startup


def test_parse_args_accepts_audio_and_listening_flags():
    args = startup.parse_args(
        [
            "--mic-device",
            "USB Composite",
            "--speaker-device",
            "UACDemo",
            "--speaker-rate",
            "48000",
            "--speaker-channels",
            "2",
            "--listen-timeout-long",
            "1.5",
            "--max-utterance",
            "20",
            "--stt-model",
            "small-streaming",
        ]
    )

    assert args.mic_device == "USB Composite"
    assert args.speaker_device == "UACDemo"
    assert args.speaker_rate == 48000
    assert args.speaker_channels == 2
    assert args.listen_timeout_long == 1.5
    assert args.max_utterance == 20
    assert args.stt_model == "small-streaming"
    assert args._explicit_stt_model is True


def test_apply_runtime_overrides_sets_jetson_stt_default(monkeypatch):
    monkeypatch.setattr(startup.conf, "OLLAMA_MODEL", "orig-model")
    monkeypatch.setattr(startup.conf, "EMBEDDING_DEVICE", "cuda")
    monkeypatch.setattr(startup.conf, "EMBEDDING_BATCH_SIZE", 8)
    monkeypatch.setattr(startup.conf, "USE_FAISS", False)
    monkeypatch.setattr(startup.conf, "USE_SEMANTIC_SEARCH", False)
    monkeypatch.setattr(startup.conf, "VISION_FPS", 2)

    args = startup.parse_args(["--jetson"])
    startup.apply_runtime_overrides(args)

    assert args.stt_model == "tiny-streaming"
    assert startup.conf.OLLAMA_MODEL == "qwen2.5:3b"
    assert startup.conf.EMBEDDING_DEVICE == "cpu"
    assert startup.conf.VISION_FPS == 1


def test_apply_runtime_overrides_respects_explicit_stt_model(monkeypatch):
    monkeypatch.setattr(startup.conf, "VISION_FPS", 2)

    args = startup.parse_args(["--jetson", "--stt-model", "base"])
    startup.apply_runtime_overrides(args)

    assert args.stt_model == "base"

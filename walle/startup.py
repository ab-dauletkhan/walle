"""
WALL-E Startup — Composition Root and entry point.

All object creation happens here. No other module creates its own dependencies.
Pattern: Composition Root / Dependency Injection.
"""

import argparse
import logging
import os
import signal
import sys

from walle.chat_loop import ChatLoop, ChatSession, LLMStreamer
from walle.lifecycle import LifecycleManager
from walle.memory.config import conf, setup_logging, validate_ollama
from walle.memory.context_manager import ContextManager
from walle.memory.embeddings import get_embedding_model
from walle.memory.heartbeat import HeartbeatManager
from walle.memory.memory_system import ArchivalMemory, Memory, RecallMemory
from walle.serial_manager import SerialManager
from walle.tools.communication import CommunicationExecutor
from walle.tools.knowledge import KnowledgeToolExecutor
from walle.tools.memory_tools import MemoryToolExecutor
from walle.tools.personality import (
    PersonalityEngine,
    PersonalityToolExecutor,
)
from walle.tools.registry import (
    CommunicationToolProvider,
    RelevantMemoryProvider,
    SchemaExecutorToolProvider,
    SystemPromptBuilder,
    ToolRegistry,
    ToolSuiteFacade,
)
from walle.tools.robot.executor import RobotControlExecutor, get_robot_control_tools
from walle.vision.service import (
    CaptureImageExecutor,
    VisionService,
    get_capture_image_tools,
)

_log = logging.getLogger("walle.startup")
_BASE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Composition Root — builds the full object graph
# ---------------------------------------------------------------------------
def build_app(args) -> tuple:
    """Create all subsystems and wire them together.

    Returns:
        (orchestrator, lifecycle, context_manager, serial_manager,
         vision_service, robot_exec)
    """
    from openai import OpenAI

    lifecycle = LifecycleManager()

    # -- Serial --
    serial_mgr = SerialManager(port=args.serial_port, baud_rate=args.baud_rate)
    lifecycle.register("serial", serial_mgr.close)

    # -- Memory --
    get_embedding_model()  # warm up embeddings before REPL starts
    core_mem = Memory()
    recall_mem = RecallMemory(use_semantic=conf.USE_SEMANTIC_SEARCH)
    archival_mem = ArchivalMemory(use_semantic=conf.USE_SEMANTIC_SEARCH)
    mem_exec = MemoryToolExecutor(core_mem, recall_mem, archival_mem)

    # -- Personality --
    personality = PersonalityEngine.load()
    personality_exec = PersonalityToolExecutor(personality)
    lifecycle.register("personality", personality.save)

    # -- Robot --
    robot_exec = RobotControlExecutor(serial_manager=serial_mgr)

    # -- Other executors --
    heartbeat = HeartbeatManager()
    knowledge_exec = KnowledgeToolExecutor()
    context_manager = ContextManager()
    comm_exec = CommunicationExecutor()

    # -- Tool registry (Microkernel: each provider registers independently) --
    # Memory / personality / knowledge tools are intentionally not registered
    # with the LLM to keep the tool schema budget small on Jetson. The
    # executors are kept alive — the memory loop still auto-injects relevant
    # recall/archival hits via RelevantMemoryProvider without needing the
    # LLM to self-edit blocks.
    _ = mem_exec, personality_exec, knowledge_exec  # retained for future re-enable
    registry = ToolRegistry()
    registry.register(CommunicationToolProvider(comm_exec))
    registry.register(SchemaExecutorToolProvider(get_robot_control_tools, robot_exec))
    tool_suite = ToolSuiteFacade(registry)

    # -- LLM client --
    client = OpenAI(
        base_url=f"{conf.OLLAMA_BASE_URL}/v1", api_key="ollama", timeout=120.0
    )
    llm_streamer = LLMStreamer(client, conf.OLLAMA_MODEL, num_ctx=conf.OLLAMA_NUM_CTX)

    # -- Prompt & memory providers --
    prompt_builder = SystemPromptBuilder(
        context_manager=context_manager,
        personality_engine=personality,
        core_memory=core_mem,
    )
    memory_provider = RelevantMemoryProvider(
        recall_memory=recall_mem,
        archival_memory=archival_mem,
    )

    # -- Chat loop --
    session = ChatSession()
    chat_loop = ChatLoop(
        llm_streamer=llm_streamer,
        tool_suite=tool_suite,
        session=session,
        prompt_builder=prompt_builder,
        memory_provider=memory_provider,
        comm_exec=comm_exec,
        recall_mem=recall_mem,
        archival_mem=archival_mem,
        heartbeat=heartbeat,
        context_manager=context_manager,
        on_turn_start=None,
        on_turn_end=None,
    )

    # -- Register shutdown hooks --
    # LifecycleManager executes in REVERSE order, so register in reverse
    # of desired shutdown sequence:
    #   api → vision → compression-wait → robot-neutral → faiss → embeddings → personality → serial
    # Serial and personality are already registered above (first), so they run last.
    # The remaining hooks from last-to-run to first-to-run:
    lifecycle.register("embeddings", recall_mem.shutdown)
    lifecycle.register("faiss-save", lambda: _save_faiss(recall_mem, archival_mem))
    lifecycle.register(
        "robot-neutral", lambda: robot_exec.execute("reset_to_neutral", {})
    )
    lifecycle.register("memory-compression", chat_loop.wait_for_compression)

    # -- Build orchestrator (thin facade) --
    # Import here to avoid circular — walle_main imports from startup
    from walle.orchestrator import WallEOrchestrator

    orchestrator = WallEOrchestrator(
        chat_loop=chat_loop,
        comm_exec=comm_exec,
        recall_mem=recall_mem,
        context_manager=context_manager,
        vision_service=None,
        serial_manager=serial_mgr,
        tool_suite=tool_suite,
    )

    _log.info("Orchestrator initialized: %s via Ollama", conf.OLLAMA_MODEL)

    return orchestrator, lifecycle, context_manager, serial_mgr, robot_exec


def attach_vision(orchestrator, context_manager, lifecycle, args, vision_fps: int = 2):
    """Optionally create and attach VisionService."""
    if args.no_vision:
        return None
    vision = VisionService(context_manager, camera_index=args.camera, fps=vision_fps)
    capture_exec = CaptureImageExecutor(vision, conf.OLLAMA_BASE_URL)
    orchestrator.set_vision_service(vision)
    orchestrator.tool_suite.register(
        SchemaExecutorToolProvider(
            get_capture_image_tools, capture_exec, request_heartbeat_after=True
        )
    )
    vision.start()
    lifecycle.register("vision", vision.stop)
    return vision


def _save_faiss(recall_mem, archival_mem):
    for mem in (recall_mem, archival_mem):
        if hasattr(mem, "faiss_manager") and mem.faiss_manager is not None:
            try:
                mem.faiss_manager.save()
            except Exception as e:
                _log.warning("FAISS save failed: %s", e)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------
def _status_line(label: str, detail: str, ok: bool) -> str:
    tag = "[OK]" if ok else "[--]"
    return f"  {label:<10} {detail:<28} {tag}"


def _test_tts(url: str) -> bool:
    try:
        import requests

        r = requests.get(url, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _test_piper(model_path: str) -> bool:
    """Verify a Piper voice is usable without actually loading the ONNX.

    Loading takes a few hundred ms on Jetson, so we only check file
    presence here — the engine constructor does the real load and will
    raise if the model is corrupt.
    """
    return os.path.exists(model_path) and os.path.exists(model_path + ".json")


def _parse_audio_device(value: str):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _query_audio_devices():
    import sounddevice as sd  # noqa: PLC0415

    return sd.query_devices()


def _resolve_audio_device(value, kind: str):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    needle = value.lower()
    try:
        devs = _query_audio_devices()
    except Exception as exc:
        _log.warning("Audio device query failed: %s", exc)
        return None
    want_input = kind == "input"
    for idx, dev in enumerate(devs):
        channels = (
            dev["max_input_channels"] if want_input else dev["max_output_channels"]
        )
        if channels <= 0:
            continue
        if needle in dev["name"].lower():
            return idx
    _log.warning("No %s device matching '%s' — falling back to default", kind, value)
    return None


def _describe_audio_device(device, kind: str) -> str:
    if device is None:
        return f"default {kind}"
    try:
        info = _query_audio_devices()[device]
    except Exception:
        return f"{kind} #{device}"
    return f"{kind} #{device} {info['name']}"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="WALL-E Unified Orchestrator")

    # Mode
    parser.add_argument(
        "--text-mode", action="store_true", help="Text REPL instead of voice"
    )
    parser.add_argument(
        "--no-vision", action="store_true", help="Disable vision processing"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default 0)"
    )
    parser.add_argument(
        "--jetson", action="store_true", help="Apply Jetson Orin Nano optimizations"
    )

    # Serial
    parser.add_argument(
        "--serial-port", default=conf.SERIAL_PORT, help="Arduino serial port"
    )
    parser.add_argument(
        "--baud-rate", type=int, default=conf.BAUD_RATE, help="Serial baud rate"
    )

    # Web API
    parser.add_argument(
        "--web-port", type=int, default=5001, help="API server port (0 to disable)"
    )

    # TTS
    parser.add_argument("--no-tts", action="store_true", help="Console TTS (no audio)")
    parser.add_argument(
        "--tts-engine",
        default="piper",
        choices=["piper", "mimic3"],
        help="TTS engine (default: piper — local, natural, no HTTP server)",
    )
    parser.add_argument(
        "--piper-model",
        default="voices/en_US-lessac-medium.onnx",
        help="Path to Piper .onnx voice (needs matching .onnx.json)",
    )
    parser.add_argument(
        "--piper-speaker",
        type=int,
        default=None,
        help="Piper speaker id for multi-speaker voices",
    )
    parser.add_argument(
        "--tts-url",
        default="http://localhost:59125",
        help="Mimic3 TTS URL (only when --tts-engine=mimic3)",
    )
    parser.add_argument(
        "--tts-voice",
        default="en_UK/apope_low",
        help="Mimic3 voice (only when --tts-engine=mimic3)",
    )

    # Voice / STT
    parser.add_argument("--wake-word", default="hey", help="Wake word phrase")
    parser.add_argument(
        "--listen-timeout", type=float, default=3.0, help="Silence timeout (seconds)"
    )
    parser.add_argument(
        "--listen-timeout-long",
        type=float,
        default=1.0,
        help="Conversational silence timeout once the user is mid-sentence",
    )
    parser.add_argument(
        "--max-utterance",
        type=float,
        default=15.0,
        help="Hard upper bound on a single listen window in seconds",
    )
    parser.add_argument("--language", default="en", help="STT language")
    parser.add_argument(
        "--stt-model",
        default="medium-streaming",
        choices=[
            "tiny-streaming",
            "small-streaming",
            "medium-streaming",
            "tiny",
            "base",
        ],
        help="Moonshine STT model",
    )
    parser.add_argument(
        "--intent-threshold", type=float, default=0.65, help="Intent match threshold"
    )
    parser.add_argument(
        "--embedding-model",
        default="embeddinggemma-300m",
        help="Intent embedding model",
    )
    parser.add_argument(
        "--embedding-quantization", default="q4", help="Embedding quantization"
    )
    parser.add_argument(
        "--mic-device",
        type=_parse_audio_device,
        default=None,
        help="PortAudio input device index or name substring",
    )
    parser.add_argument(
        "--mic-channels",
        type=int,
        default=1,
        help="Microphone input channels (default: 1)",
    )
    parser.add_argument(
        "--mic-blocksize",
        type=int,
        default=8192,
        help="PortAudio blocksize in samples (default: 8192 = 512 ms @ 16 kHz)",
    )
    parser.add_argument(
        "--speaker-device",
        type=_parse_audio_device,
        default=None,
        help="PortAudio output device index or name substring",
    )
    parser.add_argument(
        "--speaker-rate",
        type=int,
        default=None,
        help="Resample playback to this Hz before sending to the speaker device",
    )
    parser.add_argument(
        "--speaker-channels",
        type=int,
        default=None,
        help="Expand mono playback to this many channels before output",
    )

    args = parser.parse_args(argv)
    args._explicit_stt_model = any(
        arg == "--stt-model" or str(arg).startswith("--stt-model=") for arg in argv
    )
    return args


def apply_runtime_overrides(args):
    if args.jetson:
        conf.OLLAMA_MODEL = "qwen2.5:3b"
        conf.EMBEDDING_DEVICE = "cpu"
        conf.EMBEDDING_BATCH_SIZE = 4
        conf.USE_FAISS = True
        conf.USE_SEMANTIC_SEARCH = True
        conf.VISION_FPS = 1
        if not getattr(args, "_explicit_stt_model", False):
            args.stt_model = "small-streaming"
        _log.info(
            "Jetson mode: model=%s, embedding=%s, fps=%s, stt=%s",
            conf.OLLAMA_MODEL,
            conf.EMBEDDING_DEVICE,
            conf.VISION_FPS,
            args.stt_model,
        )
    return args


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Voice mode is the live Jetson deployment — suppress debug chatter
    # (LLM token metrics, stale-turn retries) so only real output is spoken/shown.
    # Text mode stays verbose for development.
    run_mode = "debug" if args.text_mode else "test"
    conf.RUN_MODE = run_mode
    setup_logging(run_mode)

    # --- Jetson overrides ---
    args = apply_runtime_overrides(args)

    args.mic_device = _resolve_audio_device(args.mic_device, "input")
    args.speaker_device = _resolve_audio_device(args.speaker_device, "output")
    args._mic_label = _describe_audio_device(args.mic_device, "input")
    args._speaker_label = _describe_audio_device(args.speaker_device, "output")

    from walle.voice.assistant import set_output_device

    set_output_device(
        args.speaker_device,
        rate=args.speaker_rate,
        channels=args.speaker_channels,
    )
    _log.info("Audio input: %s", args._mic_label)
    if args.speaker_rate is not None or args.speaker_channels is not None:
        extras = []
        if args.speaker_rate is not None:
            extras.append(f"{args.speaker_rate} Hz")
        if args.speaker_channels is not None:
            extras.append(f"{args.speaker_channels}ch")
        _log.info("Audio output: %s (%s)", args._speaker_label, ", ".join(extras))
    else:
        _log.info("Audio output: %s", args._speaker_label)

    # --- Build the app ---
    status_lines = []

    _log.info("[1/7] Initializing memory system...")
    orchestrator, lifecycle, context_manager, serial_mgr, robot_exec = build_app(args)
    mem_mode = "FAISS" if conf.USE_FAISS else "text"
    status_lines.append(_status_line("Memory:", f"3-tier + {mem_mode}", True))

    # Validate Ollama
    _log.info("[2/7] Validating Ollama (%s)...", conf.OLLAMA_MODEL)
    ollama_ok, ollama_proc = validate_ollama(conf)
    status_lines.append(
        _status_line("LLM:", f"{conf.OLLAMA_MODEL} via Ollama", ollama_ok)
    )
    if ollama_proc:
        lifecycle.register("ollama", ollama_proc.terminate)
    if not ollama_ok:
        _log.error("Ollama not available. Cannot start without LLM.")
        return

    # Vision
    _log.info("[3/7] Setting up vision...")
    vision = attach_vision(
        orchestrator, context_manager, lifecycle, args, conf.VISION_FPS
    )
    if vision is not None:
        status_lines.append(
            _status_line(
                "Vision:",
                f"{vision.status_detail} (cam={args.camera})",
                vision.is_active,
            )
        )
    else:
        status_lines.append(_status_line("Vision:", "disabled", False))

    # Serial
    _log.info("[4/7] Serial connection...")
    status_lines.append(
        _status_line(
            "Arduino:", args.serial_port or "simulation", serial_mgr.is_connected()
        )
    )

    # API server
    _log.info("[5/7] API server...")
    api_srv = None
    if args.web_port > 0:
        from walle.api_server import APIServer

        api_srv = APIServer(
            serial_mgr,
            llm_client=orchestrator,
            vision_service=vision,
            port=args.web_port,
        )
        api_srv.start()
        lifecycle.register("api-server", api_srv.stop)
        status_lines.append(
            _status_line("API:", f"http://localhost:{args.web_port}", True)
        )
    else:
        status_lines.append(_status_line("API:", "disabled", False))

    # TTS
    _log.info("[6/7] TTS engine...")
    tts_ok = False
    if not args.no_tts:
        if args.tts_engine == "piper":
            tts_ok = _test_piper(args.piper_model)
            if not tts_ok:
                _log.warning(
                    "Piper voice not found at %s — falling back to console TTS. "
                    "Download: curl -L -o %s "
                    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                    "en/en_US/lessac/medium/en_US-lessac-medium.onnx "
                    "(and the same URL + .json)",
                    args.piper_model,
                    args.piper_model,
                )
        else:
            tts_ok = _test_tts(args.tts_url)
    if args.no_tts:
        tts_label = "Console"
    elif tts_ok:
        tts_label = "Piper" if args.tts_engine == "piper" else "Mimic3"
    else:
        tts_label = "Console (fallback)"
    if not args.no_tts and not tts_ok:
        args.no_tts = True
    tts_detail = tts_label if args.no_tts else f"{tts_label} -> {args._speaker_label}"
    if not args.no_tts and (
        args.speaker_rate is not None or args.speaker_channels is not None
    ):
        extras = []
        if args.speaker_rate is not None:
            extras.append(f"{args.speaker_rate}Hz")
        if args.speaker_channels is not None:
            extras.append(f"{args.speaker_channels}ch")
        tts_detail += f" ({', '.join(extras)})"
    status_lines.append(_status_line("TTS:", tts_detail, True))

    # STT
    if not args.text_mode:
        _log.info("[7/7] Loading STT model...")
        stt_label = f"Moonshine {args.stt_model} -> {args._mic_label}"
        status_lines.append(_status_line("STT:", stt_label, True))
    else:
        status_lines.append(_status_line("STT:", "text mode (skipped)", False))

    # Startup banner
    banner = f"\n{'=' * 42}\n  WALL-E System Status\n{'=' * 42}"
    for line in status_lines:
        banner += f"\n{line}"
    banner += f"\n{'=' * 42}"
    _log.info(banner)

    # --- Run ---
    if args.text_mode:
        print('  Type your message. "exit" to quit.\n')
        try:
            _text_mode_repl(orchestrator, args)
        finally:
            lifecycle.shutdown()
        return

    # Voice mode — custom signal handler because VoiceAssistant.run()
    # may not propagate KeyboardInterrupt cleanly through audio threads.
    def _halt(*_):
        # Restore default handler so a second Ctrl+C force-kills immediately.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        lifecycle.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, _halt)
    signal.signal(signal.SIGTERM, _halt)
    print(f'  Say "{args.wake_word}" to begin!\n')
    _run_voice_mode(orchestrator, robot_exec, args, lifecycle)


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------
def _text_mode_repl(orchestrator, args):
    from walle.voice.assistant import (
        ConsoleTTSEngine,
        Mimic3TTSEngine,
        PiperTTSEngine,
    )

    if args.no_tts:
        tts = ConsoleTTSEngine()
    elif args.tts_engine == "piper":
        tts = PiperTTSEngine(
            model_path=args.piper_model, speaker_id=args.piper_speaker
        )
    else:
        tts = Mimic3TTSEngine(url=args.tts_url, voice=args.tts_voice)

    print(f"\n{'=' * 50}")
    print(f"  WALL-E Text Mode | {conf.OLLAMA_MODEL} via Ollama")
    print("  Type 'exit' or 'quit' to stop.")
    print(f"{'=' * 50}\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            response = orchestrator.chat(user_input)
            if response:
                print(f"WALL-E: {response}\n")
                try:
                    tts.speak(response)
                except Exception as exc:
                    _log.warning("TTS failed: %s", exc)
            else:
                print("WALL-E: (no response)\n")
    except KeyboardInterrupt:
        print("\n")


class _RobotBridge:
    """Maps voice intent actions to RobotControlExecutor tool calls.

    Defined here (not in walle_main.py) because it needs BaseRobotController
    from walle.voice.assistant which pulls heavy audio deps. This class is only
    instantiated in voice mode after those deps are lazy-loaded.
    """

    # Voice intents map directly to the unified `drive` tool. The old
    # wave/dance intents targeted arm/neck servos that no longer exist in
    # the CLI firmware, so they're dropped rather than silently no-oped.
    ACTION_MAP = {
        "forward": (
            "drive",
            {"direction": "forward", "speed": 85, "duration_ms": 2000},
        ),
        "backward": (
            "drive",
            {"direction": "backward", "speed": 85, "duration_ms": 2000},
        ),
        "left": ("drive", {"direction": "left", "speed": 75, "duration_ms": 900}),
        "right": ("drive", {"direction": "right", "speed": 75, "duration_ms": 900}),
        "stop": ("stop_movement", {}),
    }

    def __init__(self, robot_exec):
        self._robot = robot_exec

    def execute(self, action: str, utterance: str, confidence: float) -> None:
        mapping = self.ACTION_MAP.get(action)
        if mapping:
            tool_name, args = mapping
            _log.debug(
                "RobotBridge: %s -> %s (confidence=%.0f%%)",
                action,
                tool_name,
                confidence * 100,
            )
            self._robot.execute(tool_name, args)
        else:
            _log.warning("RobotBridge: unknown action '%s'", action)

    def close(self) -> None:
        pass


def _run_voice_mode(orchestrator, robot_exec, args, lifecycle: LifecycleManager):
    from walle.voice.assistant import (
        ROBOT_INTENTS,
        ConsoleTTSEngine,
        Mimic3TTSEngine,
        PiperTTSEngine,
        VoiceAssistant,
        _stt_model_choices,
    )

    robot_bridge = _RobotBridge(robot_exec)
    if args.no_tts:
        tts = ConsoleTTSEngine()
    elif args.tts_engine == "piper":
        tts = PiperTTSEngine(
            model_path=args.piper_model, speaker_id=args.piper_speaker
        )
    else:
        tts = Mimic3TTSEngine(url=args.tts_url, voice=args.tts_voice)

    # ModelArch is lazy-imported inside assistant.py to keep text-mode free
    # of the moonshine_voice dependency. Resolve the string choice here.
    stt_arch_map = _stt_model_choices()
    stt_arch = stt_arch_map.get(args.stt_model, stt_arch_map["medium-streaming"])

    wake_sounds_dir = os.path.join(_BASE, "voice", "wake_up_sounds")
    if not os.path.isdir(wake_sounds_dir):
        wake_sounds_dir = None

    assistant = VoiceAssistant(
        llm=orchestrator,
        robot=robot_bridge,
        tts=tts,
        language=args.language,
        stt_model_arch=stt_arch,
        embedding_model=args.embedding_model,
        embedding_quantization=args.embedding_quantization,
        intent_threshold=args.intent_threshold,
        intents=ROBOT_INTENTS,
        wake_word=args.wake_word,
        listen_timeout=args.listen_timeout,
        listen_timeout_long=args.listen_timeout_long,
        max_utterance=args.max_utterance,
        wake_sounds_dir=wake_sounds_dir,
        mic_device=args.mic_device,
        mic_channels=args.mic_channels,
        mic_blocksize=args.mic_blocksize,
        system_prompt="WALL-E voice mode active.",
    )

    try:
        assistant.run()
    except KeyboardInterrupt:
        pass
    finally:
        lifecycle.shutdown()

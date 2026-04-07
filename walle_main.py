"""
WALL-E Unified Orchestrator
Wires together STT/TTS voice pipeline with the memory-augmented LLM brain.
"""
import logging
import os
import json
import time
import signal
import argparse
import threading
from dataclasses import dataclass
from collections import deque
from datetime import datetime
from typing import Iterator, Optional

# --- Path setup ---
_BASE = os.path.dirname(os.path.abspath(__file__))

# --- Memory subsystem imports ---
from memory.config import conf, setup_logging
from memory.memory_system import Memory, RecallMemory, ArchivalMemory
from memory.memory_tools import MemoryToolExecutor
from memory.personality_system import PersonalityEngine, PersonalityToolExecutor
from memory.robot_tools import RobotControlExecutor
from memory.heartbeat import HeartbeatManager
from memory.knowledge_tools import KnowledgeToolExecutor
from memory.context_manager import ContextManager, EnvironmentContext, InteractionContext, SensorSimulator
from memory.communication_tools import CommunicationExecutor

# --- Vision imports ---
from vision_service import VisionService, CaptureImageExecutor

# --- Integration support ---
from orchestrator_support import RelevantMemoryProvider, SystemPromptBuilder, ToolSuiteFacade

# --- Serial & API imports ---
from serial_manager import SerialManager

# --- STT/TTS imports ---
# LLMClient is lightweight — safe to import at module level.
# Voice-heavy deps (VoiceAssistant, Mimic3TTSEngine, etc.) are lazy-loaded in main().
from stt_tts.mock_llm import LLMClient

try:
    from stt_tts.main import BaseRobotController
except ImportError:
    # Fallback ABC if stt_tts deps (sounddevice, moonshine) are not installed.
    # RobotBridge still works; voice mode will fail with a clear error.
    from abc import ABC, abstractmethod
    class BaseRobotController(ABC):
        @abstractmethod
        def execute(self, action: str, utterance: str, confidence: float) -> None: ...
        @abstractmethod
        def close(self) -> None: ...

_log = logging.getLogger("walle.orchestrator")


@dataclass(frozen=True)
class ToolFunctionCall:
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCallEnvelope:
    id: str
    function: ToolFunctionCall
    type: str = "function"


# ---------------------------------------------------------------------------
# ChatSession — lightweight conversation buffer for the LLM tool loop
# ---------------------------------------------------------------------------
class ChatSession:
    def __init__(self):
        self.history = deque(maxlen=conf.MAX_CONTEXT_MESSAGES)

    def add(self, role, content, tool_calls=None):
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.history.append(msg)

    def add_tool_result(self, tool_id, name, content):
        self.history.append({"role": "tool", "tool_call_id": tool_id, "name": name, "content": content})

    def get_messages(self, system_prompt: str):
        return [{"role": "system", "content": system_prompt}] + list(self.history)


# ---------------------------------------------------------------------------
# WallELLMClient — full memory-augmented chat as a streaming LLM client
# ---------------------------------------------------------------------------
class WallELLMClient(LLMClient):
    """Adapts the MemGPT-style chat loop into the LLMClient.stream_chat interface."""

    def __init__(self, serial_port=None, baud_rate=115200, serial_manager=None):
        from openai import OpenAI

        self.client = OpenAI(base_url=f"{conf.OLLAMA_BASE_URL}/v1", api_key="ollama")
        self.model = conf.OLLAMA_MODEL

        # Core systems
        self.core_mem = Memory()
        self.recall_mem = RecallMemory(use_semantic=conf.USE_SEMANTIC_SEARCH)
        self.archival_mem = ArchivalMemory(use_semantic=conf.USE_SEMANTIC_SEARCH)
        self.mem_exec = MemoryToolExecutor(self.core_mem, self.recall_mem, self.archival_mem)
        self.personality = PersonalityEngine.load()
        self.personality_exec = PersonalityToolExecutor(self.personality)
        self.serial_manager = serial_manager
        if serial_manager is not None:
            self.robot_exec = RobotControlExecutor(serial_manager=serial_manager)
        else:
            self.robot_exec = RobotControlExecutor(serial_port=serial_port, baud_rate=baud_rate)
        self.heartbeat = HeartbeatManager()
        self.knowledge_exec = KnowledgeToolExecutor()
        self.context_manager = ContextManager()
        self.comm_exec = CommunicationExecutor()
        self.session = ChatSession()
        self.prompt_builder = SystemPromptBuilder(
            context_manager=self.context_manager,
            personality_engine=self.personality,
            core_memory=self.core_mem,
        )
        self.memory_provider = RelevantMemoryProvider(
            recall_memory=self.recall_mem,
            archival_memory=self.archival_mem,
        )

        # Vision (initialized later via set_vision_service)
        self.vision_service: Optional[VisionService] = None
        self.capture_image_exec: Optional[CaptureImageExecutor] = None
        self.tool_suite = ToolSuiteFacade(
            communication_executor=self.comm_exec,
            robot_executor=self.robot_exec,
            memory_executor=self.mem_exec,
            personality_executor=self.personality_exec,
            knowledge_executor=self.knowledge_exec,
        )
        self._compression_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = False

        _log.info("Orchestrator initialized: %s via Ollama", self.model)

    def set_vision_service(self, vision_service: VisionService):
        """Attach vision service and create capture_image executor."""
        self.vision_service = vision_service
        self.capture_image_exec = CaptureImageExecutor(
            vision_service, conf.OLLAMA_BASE_URL
        )
        self.tool_suite.set_capture_image_executor(self.capture_image_exec)

    # -- helpers --

    def _get_system_prompt(self, relevant_memories: str = "") -> str:
        return self.prompt_builder.build(relevant_memories)

    def _retrieve_relevant_context(self, query: str) -> str:
        return self.memory_provider.build_context(query)

    def chat(self, user_input: str) -> Optional[str]:
        """Public chat entry point for API, REPL, and voice mode."""
        return self._run_chat_loop(user_input)

    def _summarize_text(self, text: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"Summarize this concisely:\n{text}"}],
                max_tokens=200,
            )
            return resp.choices[0].message.content
        except Exception:
            return text[:500] + "..."

    def _stream_llm(self, messages, tools, max_retries=2):
        """Stream response from Ollama, return (content, tool_calls_list)."""
        for attempt in range(max_retries):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                )

                full_content = ""
                tool_calls_map = {}
                inner_thought_started = False

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_content += delta.content
                        if _log.isEnabledFor(logging.DEBUG):
                            if not inner_thought_started:
                                print("   \U0001f4ad ", end="", flush=True)
                                inner_thought_started = True
                            print(delta.content, end="", flush=True)

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_map:
                                tool_calls_map[idx] = {"id": "", "func": {"name": "", "args": ""}}
                            if tc.id:
                                tool_calls_map[idx]["id"] += tc.id
                            if tc.function.name:
                                tool_calls_map[idx]["func"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_map[idx]["func"]["args"] += tc.function.arguments

                if inner_thought_started:
                    print()

                tool_calls_list = []
                for idx in sorted(tool_calls_map.keys()):
                    t = tool_calls_map[idx]
                    tool_calls_list.append(ToolCallEnvelope(
                        id=t["id"] or f"call_{idx}",
                        function=ToolFunctionCall(
                            name=t["func"]["name"],
                            arguments=t["func"]["args"],
                        ),
                        type="function",
                    ))

                return full_content, tool_calls_list

            except Exception as e:
                _log.warning("Stream error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                time.sleep(1)

        _log.error("Failed to generate LLM response after %d retries", max_retries)
        return "I'm having trouble reaching my brain right now. Please check that Ollama is running.", []

    # -- core chat loop --

    def _run_chat_loop(self, user_input: str) -> Optional[str]:
        """Run full MemGPT-style chat loop. Returns the message for the user."""
        # Drop any stale message from a prior interrupted turn.
        self.comm_exec.get_last_message()

        # 1. Context update
        interaction_count = self.recall_mem.get_count()
        self.context_manager.update_interaction(InteractionContext(
            last_interaction=datetime.now(),
            interaction_count=interaction_count,
        ))
        if interaction_count % 5 == 0:
            self.context_manager.update_environment(
                SensorSimulator.simulate_environment_context(
                    battery=max(0, 80 - interaction_count)
                )
            )

        # 2. Retrieve relevant memories
        memory_context = self._retrieve_relevant_context(user_input)

        # 3. Insert user message (deferred embedding)
        self.recall_mem.insert("user", user_input, defer_embedding=True)
        self.session.add("user", user_input)
        self.heartbeat.reset()

        # 4. Compress old memories if needed (non-blocking, guarded against re-entry)
        self._start_memory_compression_if_needed()

        # 5. Tool execution loop
        iteration = 0
        user_received_message = False

        while iteration < 10:
            iteration += 1
            tools = self.tool_suite.build_schemas()

            content, tool_calls = self._stream_llm(
                self.session.get_messages(self._get_system_prompt(memory_context)), tools
            )

            if not tool_calls:
                if content:
                    self.session.add("assistant", content)
                    if not user_received_message:
                        _log.debug("No send_message called, prompting...")
                        continue
                break

            self.session.add("assistant", content or "", tool_calls)
            hb_req = False

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    _log.warning("JSON error in tool %s: %s", name, e)
                    self.session.add_tool_result(tc.id, name, f"Error: Invalid JSON - {e}")
                    continue

                execution = self.tool_suite.execute_tool(
                    name=name,
                    args=args,
                    user_message_already_sent=user_received_message,
                )

                if execution.user_message and not user_received_message:
                    _log.debug("WALL-E: %s", execution.user_message)
                    self.recall_mem.insert("assistant", execution.user_message)
                    user_received_message = True

                hb_req = hb_req or execution.heartbeat_requested
                self.session.add_tool_result(tc.id, name, execution.tool_result)

            if user_received_message:
                break

            if hb_req and self.heartbeat.can_heartbeat():
                self.heartbeat.request_heartbeat()
                _log.debug("Heartbeat — continuing...")
                continue

            _log.debug("Continuing for response...")

        return self.comm_exec.get_last_message()

    def _start_memory_compression_if_needed(self) -> None:
        if self.recall_mem.get_count() <= conf.RECALL_MEMORY_LIMIT:
            return
        if not self._compression_lock.acquire(blocking=False):
            return

        def _compress_then_unlock():
            try:
                self.recall_mem.compress_old_memories(
                    self._summarize_text,
                    self.archival_mem,
                )
            finally:
                self._compression_lock.release()

        compress_thread = threading.Thread(
            target=_compress_then_unlock,
            daemon=False,
            name="memory-compress",
        )
        compress_thread.start()
        self._compress_thread = compress_thread

    def wait_for_background_tasks(self, timeout: float = 15.0) -> None:
        compress_thread = getattr(self, "_compress_thread", None)
        if compress_thread is not None and compress_thread.is_alive():
            _log.info("Waiting for memory compression to finish...")
            compress_thread.join(timeout=timeout)
            if compress_thread.is_alive():
                _log.warning("Compression thread did not finish in time")

    def reset_robot_to_neutral(self) -> None:
        try:
            self.robot_exec.execute("reset_to_neutral", {})
        except Exception as exc:
            _log.warning("Robot neutral reset failed: %s", exc)

    def save_runtime_state(self) -> None:
        for mem in (self.recall_mem, self.archival_mem):
            if hasattr(mem, "faiss_manager") and mem.faiss_manager is not None:
                try:
                    mem.faiss_manager.save()
                except Exception as exc:
                    _log.warning("FAISS save failed: %s", exc)
        self.recall_mem.shutdown()
        self.personality.save()

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_complete:
                return

            _log.info("Shutting down WALL-E...")
            if self.vision_service is not None:
                self.vision_service.stop()
            self.wait_for_background_tasks()
            self.reset_robot_to_neutral()
            self.save_runtime_state()
            self.robot_exec.close()
            if self.serial_manager is not None:
                self.serial_manager.close()
            self._shutdown_complete = True
            _log.info("WALL-E offline. Goodbye!")

    # -- LLMClient interface --

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """Implements LLMClient.stream_chat for the SpeechRouter."""
        # Extract latest user message (SpeechRouter maintains its own history,
        # but we use our internal ChatSession for the full memory context)
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break

        if not user_text:
            yield "I didn't catch that."
            return

        response = self.chat(user_text)
        if not response:
            response = "I'm not sure what to say."

        # Yield word-by-word for streaming TTS feel
        words = response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.02)

    def close(self):
        self.shutdown()


# ---------------------------------------------------------------------------
# RobotBridge — adapts RobotControlExecutor to BaseRobotController interface
# ---------------------------------------------------------------------------
class RobotBridge(BaseRobotController):
    """Maps voice intent actions to RobotControlExecutor tool calls."""

    ACTION_MAP = {
        "forward":  ("drive_forward",   {"speed": 50, "duration_ms": 1000}),
        "backward": ("drive_backward",  {"speed": 50, "duration_ms": 1000}),
        "left":     ("turn_left",       {"speed": 50, "duration_ms": 500}),
        "right":    ("turn_right",      {"speed": 50, "duration_ms": 500}),
        "stop":     ("stop_movement",   {}),
        "wave":     ("wave_hello",      {}),
        "dance":    ("express_emotion", {"emotion": "happy"}),
    }

    def __init__(self, robot_exec: RobotControlExecutor):
        self._robot = robot_exec

    def execute(self, action: str, utterance: str, confidence: float) -> None:
        mapping = self.ACTION_MAP.get(action)
        if mapping:
            tool_name, args = mapping
            _log.debug("RobotBridge: %s -> %s (confidence=%.0f%%)", action, tool_name, confidence * 100)
            self._robot.execute(tool_name, args)
        else:
            _log.warning("RobotBridge: unknown action '%s'", action)

    def close(self) -> None:
        pass  # WallELLMClient owns serial cleanup


# ---------------------------------------------------------------------------
# Text-mode REPL (for debugging without audio hardware)
# ---------------------------------------------------------------------------
def text_mode_repl(llm_client: WallELLMClient, shutdown_event: threading.Event = None):
    print(f"\n{'=' * 50}")
    print(f"  WALL-E Text Mode | {conf.OLLAMA_MODEL} via Ollama")
    print(f"  Type 'exit' or 'quit' to stop.")
    print(f"{'=' * 50}\n")

    try:
        while not (shutdown_event and shutdown_event.is_set()):
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            response = llm_client.chat(user_input)
            if response:
                print(f"WALL-E: {response}\n")
            else:
                print("WALL-E: (no response)\n")
    except KeyboardInterrupt:
        print("\n")


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------
def _status_line(label: str, detail: str, ok: bool) -> str:
    tag = "[OK]" if ok else "[--]"
    return f"  {label:<10} {detail:<28} {tag}"


def _validate_ollama(model: str) -> bool:
    """Check Ollama is reachable and model is available."""
    try:
        import requests
        r = requests.get(f"{conf.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        return model in models or f"{model}:latest" in models
    except Exception:
        return False


def _test_tts(url: str) -> bool:
    """Check if Mimic3 TTS server is reachable."""
    try:
        import requests
        r = requests.get(url, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
def _shutdown(llm_client: WallELLMClient, api_srv=None):
    """Orderly shutdown of all subsystems."""
    if api_srv is not None:
        api_srv.stop()
    llm_client.shutdown()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="WALL-E Unified Orchestrator")

    # Mode
    parser.add_argument("--text-mode", action="store_true", help="Text REPL instead of voice")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision processing")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--jetson", action="store_true", help="Apply Jetson Orin Nano optimizations")

    # Serial
    parser.add_argument("--serial-port", default=conf.SERIAL_PORT, help="Arduino serial port")
    parser.add_argument("--baud-rate", type=int, default=conf.BAUD_RATE, help="Serial baud rate")

    # Web API
    parser.add_argument("--web-port", type=int, default=5001, help="API server port (0 to disable)")

    # TTS
    parser.add_argument("--no-tts", action="store_true", help="Console TTS (no audio)")
    parser.add_argument("--tts-url", default="http://localhost:59125", help="Mimic3 TTS URL")
    parser.add_argument("--tts-voice", default="en_UK/apope_low", help="Mimic3 voice")

    # Voice / STT
    parser.add_argument("--wake-word", default="hey robot", help="Wake word phrase")
    parser.add_argument("--listen-timeout", type=float, default=3.0, help="Silence timeout (seconds)")
    parser.add_argument("--language", default="en", help="STT language")
    parser.add_argument("--stt-model", default="small-streaming",
                        choices=["tiny-streaming", "small-streaming", "medium-streaming", "tiny", "base"],
                        help="Moonshine STT model")
    parser.add_argument("--intent-threshold", type=float, default=0.65, help="Intent match threshold")
    parser.add_argument("--embedding-model", default="embeddinggemma-300m", help="Intent embedding model")
    parser.add_argument("--embedding-quantization", default="q4", help="Embedding quantization")

    args = parser.parse_args()

    # --- Apply Jetson overrides if requested ---
    vision_fps = 2
    if args.jetson:
        import config_jetson
        vision_fps = config_jetson.VISION_FPS

    # All data-file paths are now absolute (via MEMORY_DIR in config.py),
    # so there is no need to os.chdir() — keeps CWD predictable for the
    # vision service, web API, and any other module using relative paths.

    # --- Sequential startup with progress reporting ---
    status_lines = []

    # Step 1: Memory system
    _log.info("[1/7] Initializing memory system...")
    serial_mgr = SerialManager(port=args.serial_port, baud_rate=args.baud_rate)
    llm_client = WallELLMClient(serial_manager=serial_mgr)
    mem_mode = "FAISS" if conf.USE_FAISS else "text"
    status_lines.append(_status_line("Memory:", f"3-tier + {mem_mode}", True))

    # Step 2: Validate Ollama
    _log.info("[2/7] Validating Ollama (%s)...", conf.OLLAMA_MODEL)
    ollama_ok = _validate_ollama(conf.OLLAMA_MODEL)
    status_lines.append(_status_line("LLM:", f"{conf.OLLAMA_MODEL} via Ollama", ollama_ok))
    if not ollama_ok:
        _log.warning("Ollama not reachable or model '%s' not found.", conf.OLLAMA_MODEL)
        _log.warning("Make sure Ollama is running: ollama serve")
        _log.warning("And model is pulled: ollama pull %s", conf.OLLAMA_MODEL)

    # Step 3: Vision
    _log.info("[3/7] Setting up vision...")
    if not args.no_vision:
        vision = VisionService(llm_client.context_manager, camera_index=args.camera, fps=vision_fps)
        llm_client.set_vision_service(vision)
        vision.start()
        status_lines.append(_status_line("Vision:", f"{vision.backend_name} (cam={args.camera})", vision.is_active))
    else:
        status_lines.append(_status_line("Vision:", "disabled", False))

    # Step 4: Serial / Arduino
    _log.info("[4/7] Serial connection...")
    status_lines.append(_status_line("Arduino:", args.serial_port or "simulation", serial_mgr.is_connected()))

    # Step 5: API server
    _log.info("[5/7] API server...")
    api_srv = None
    if args.web_port > 0:
        from api_server import APIServer
        api_srv = APIServer(serial_mgr, llm_client=llm_client, vision_service=llm_client.vision_service, port=args.web_port)
        api_srv.start()
        status_lines.append(_status_line("API:", f"http://localhost:{args.web_port}", True))
    else:
        status_lines.append(_status_line("API:", "disabled", False))

    # Step 6: TTS
    _log.info("[6/7] TTS engine...")
    tts_ok = False
    if not args.no_tts:
        tts_ok = _test_tts(args.tts_url)
    tts_label = "Console" if args.no_tts else ("Mimic3" if tts_ok else "Console (fallback)")
    if not args.no_tts and not tts_ok:
        args.no_tts = True  # fall back to console
    status_lines.append(_status_line("TTS:", tts_label, True))

    # Step 7: STT (only in voice mode)
    if not args.text_mode:
        _log.info("[7/7] Loading STT model...")
        stt_label = f"Moonshine {args.stt_model}"
        status_lines.append(_status_line("STT:", stt_label, True))
    else:
        status_lines.append(_status_line("STT:", "text mode (skipped)", False))

    # --- Startup summary ---
    banner = f"\n{'=' * 42}\n  WALL-E System Status\n{'=' * 42}"
    for line in status_lines:
        banner += f"\n{line}"
    banner += f"\n{'=' * 42}"
    _log.info(banner)

    # --- Register signal handlers for graceful shutdown ---
    # The handler only sets a flag — actual cleanup happens in the main thread
    # to avoid calling non-async-signal-safe functions (locks, I/O) from a
    # signal context, which could deadlock.
    _shutdown_requested = threading.Event()

    def _signal_handler(sig, frame):
        _shutdown_requested.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # --- Run ---
    if args.text_mode:
        print('  Type your message. "exit" to quit.\n')
        text_mode_repl(llm_client, _shutdown_requested)
        _shutdown(llm_client, api_srv)
        return

    # Voice mode — lazy-load heavy STT/TTS dependencies
    print(f'  Say "{args.wake_word}" to begin!\n')

    from stt_tts.main import VoiceAssistant, ROBOT_INTENTS, Mimic3TTSEngine, ConsoleTTSEngine, ModelArch

    robot_bridge = RobotBridge(llm_client.robot_exec)
    tts = ConsoleTTSEngine() if args.no_tts else Mimic3TTSEngine(url=args.tts_url, voice=args.tts_voice)

    stt_arch_map = {
        "tiny-streaming": ModelArch.TINY_STREAMING,
        "small-streaming": ModelArch.SMALL_STREAMING,
        "medium-streaming": ModelArch.MEDIUM_STREAMING,
        "tiny": ModelArch.TINY,
        "base": ModelArch.BASE,
    }
    stt_arch = stt_arch_map.get(args.stt_model, ModelArch.SMALL_STREAMING)

    wake_sounds_dir = os.path.join(_BASE, "stt_tts", "wake_up_sounds")
    if not os.path.isdir(wake_sounds_dir):
        wake_sounds_dir = None

    assistant = VoiceAssistant(
        llm=llm_client,
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
        wake_sounds_dir=wake_sounds_dir,
        system_prompt="WALL-E voice mode active.",
    )

    try:
        # Voice mode blocks in assistant.run(); the shutdown event lets the
        # signal handler break out cleanly if SIGINT/SIGTERM arrives.
        assistant.run()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown(llm_client, api_srv)


if __name__ == "__main__":
    main()

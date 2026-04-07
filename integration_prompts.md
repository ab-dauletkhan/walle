# WALL-E Integration Prompts for Claude Code

Execute these batches in order. Each batch assumes the previous one is complete.

---

## BATCH 1: Unified Orchestrator — Wire STT/TTS into the Memory-Aware LLM

### Prompt:

```
I need you to create a unified orchestrator for my WALL-E robot project at C:/Users/Meiras/Desktop/memory/walle-main/.

**Problem**: Right now the STT/TTS system (`stt_tts/main.py`) and the memory-aware LLM system (`memory/walle_enhanced.py`) are completely disconnected. The STT/TTS has its own mock LLM backend, and the memory system is a text-only REPL. I need them wired together.

**What to create**: A new file `walle-main/walle_main.py` (the unified entry point) that:

1. **Initializes all subsystems from `memory/`**:
   - Add `memory/` to sys.path
   - Import and initialize exactly as `walle_enhanced.py` does (lines 26-43): `Memory`, `RecallMemory`, `ArchivalMemory`, `MemoryToolExecutor`, `PersonalityEngine`, `PersonalityToolExecutor`, `RobotControlExecutor`, `HeartbeatManager`, `KnowledgeToolExecutor`, `ContextManager`, `CommunicationExecutor`, `ChatSession`
   - Import `OpenAI` client pointed at Ollama (`conf.OLLAMA_BASE_URL`)

2. **Creates a new LLM client class that implements `stt_tts/main.py`'s `LLMClient` interface**:
   - `LLMClient` (from `stt_tts/mock_llm.py`) has method: `def chat(self, messages: list[dict]) -> str`
   - The new class `WallELLMClient(LLMClient)` should:
     - Call `chat()` from `walle_enhanced.py` logic (lines 208-320) — context retrieval, memory compression, tool execution loop, streaming
     - After the LLM calls `send_message` tool, capture the message via `comm_exec.get_last_message()` and return it as the string response
     - Handle all tool routing (robot, memory, personality, knowledge) internally
     - Store conversation in recall memory and update context manager

3. **Initializes STT/TTS components from `stt_tts/main.py`**:
   - Import `VoiceAssistant`, `SpeechRouter`, `Mimic3TTSEngine`, `ConsoleTTSEngine`, `BaseRobotController`, `ROBOT_INTENTS` from `stt_tts/main.py`
   - Import `MicTranscriber`, `ModelArch`, `IntentRecognizer` from moonshine_voice
   - Use the existing `RobotControlExecutor` from `memory/robot_tools.py` but wrap it to implement `BaseRobotController` interface (the abstract class needs `execute(action, utterance, confidence)` and `close()`)

4. **Wire them together in main()**:
   - Create `WallELLMClient` (wrapping the full memory+tools chat pipeline)
   - Create `RobotBridge` (wrapping `RobotControlExecutor` to match `BaseRobotController` interface — map actions like "forward" to `drive_forward`, "wave" to `wave_hello`, etc.)
   - Create TTS engine (Mimic3 or Console based on availability)
   - Create `VoiceAssistant(llm=walle_llm, robot=robot_bridge, tts=tts_engine, ...)`
   - Call `assistant.run()`

5. **Also support a text-mode fallback** (for debugging without audio hardware):
   - Add `--text-mode` CLI argument
   - In text mode: skip VoiceAssistant, run a simple REPL that calls `WallELLMClient.chat()` and prints + speaks the response

**Key constraints**:
- Do NOT modify existing files in `memory/` or `stt_tts/` — only create the new `walle_main.py`
- The `CommunicationExecutor.get_last_message()` returns the message and clears it — this is the bridge between LLM output and TTS
- `SpeechRouter._handle_llm_query(text)` calls `self._llm.chat(messages)` and then `self._speak(response)` — so TTS happens automatically through the existing SpeechRouter
- Serial port for Arduino should be configurable via CLI arg `--serial-port` (defaults to `conf.SERIAL_PORT` from config.py)
- Import paths: `stt_tts/` files need to be importable — add to sys.path

**Config reference** (from `memory/config.py`):
- `conf.OLLAMA_BASE_URL` = "http://localhost:11434"
- `conf.OLLAMA_MODEL` = "qwen2.5:3b"
- `conf.SERIAL_PORT` = None
- `conf.BAUD_RATE` = 9600
- `conf.USE_SEMANTIC_SEARCH` = True
- `conf.USE_FAISS` = True

After creating the file, verify it has no syntax errors by running: `cd /c/Users/Meiras/Desktop/memory/walle-main && python -c "import ast; ast.parse(open('walle_main.py').read()); print('Syntax OK')"`
```

---

## BATCH 2: Vision Pipeline Integration with ContextManager

### Prompt:

```
I need you to integrate the vision pipeline into the WALL-E orchestrator at C:/Users/Meiras/Desktop/memory/walle-main/.

**Current state**: `walle_main.py` exists (from previous batch) and wires STT/TTS to the memory-aware LLM. Vision is not connected yet.

**Vision components that exist**:
- `walle_vision/face_recognition_coral/recognize_face.py` — Google Coral face recognition
- `walle_vision/face_recognition_coral/scan_people.py` — Face scanning/enrollment
- `comp_vision_diplomka/pipeline.py` — YOLOv8 face detection + InsightFace ArcFace embedding (512-dim)
- `comp_vision_diplomka/realtime_person_track_face_recog.py` — DeepSORT tracking + face recognition

**Context system that exists** (in `memory/context_manager.py`):
- `VisualContext` dataclass with fields: `faces_detected: List[Dict]` (each dict has name, confidence, location), `objects_detected: List[Dict]`, `scene_description: str`, `timestamp: datetime`
- `ContextManager.update_visual(visual_context: VisualContext)` — updates the visual context
- `ContextManager.get_context_string()` — returns formatted string injected into LLM system prompt
- Already used by `walle_enhanced.py` in `get_system_prompt()` and `chat()`

**What to create**: A new file `walle-main/vision_service.py` that:

1. **Runs as a background thread** that continuously processes camera frames
2. **Abstracts over two backends**:
   - `CoralVisionBackend` — uses Google Coral TPU for face recognition (import from `walle_vision/face_recognition_coral/`)
   - `CPUVisionBackend` — uses the `comp_vision_diplomka/pipeline.py` (YOLOv8 + InsightFace) as fallback
   - Auto-detect which is available (try importing pycoral, fall back to CPU)
3. **Produces `VisualContext` objects** from each frame:
   - Detected faces → `faces_detected` list with name and confidence
   - Scene description from detected objects (if available)
4. **Feeds into ContextManager** via callback:
   ```python
   class VisionService:
       def __init__(self, context_manager: ContextManager, camera_index=0, fps=2):
           ...
       def start(self) -> None:  # Starts background thread
       def stop(self) -> None:   # Stops background thread
       def get_latest_frame(self) -> Optional[np.ndarray]:  # For on-demand capture
   ```
5. **Rate-limits** updates to the ContextManager (max 2 FPS to save resources on Jetson)
6. **Handles camera via OpenCV** (`cv2.VideoCapture`) — works with both USB cameras and CSI cameras on Jetson

**Then modify `walle_main.py`** to:
1. Import `VisionService`
2. Initialize it with the shared `context_manager` instance
3. Start it before the main loop: `vision.start()`
4. Stop it on shutdown: `vision.stop()`
5. Add CLI arg `--no-vision` to disable vision (for testing without camera)
6. Add CLI arg `--camera` for camera index (default 0)

**Also create a `capture_image` tool** that the LLM can call:
- Add a new tool to the tools list in the chat function
- When called, it grabs `vision_service.get_latest_frame()`, saves to temp file, and runs it through Ollama's vision model (moondream) for a description
- Return the description to the LLM
- This mirrors be-more-agent's `capture_image` action but uses our existing vision pipeline

**Key constraints**:
- Keep it lightweight for Jetson Orin Nano (8GB total RAM+VRAM)
- Coral TPU runs face recognition — don't load face models on GPU
- The LLM (qwen2.5:3b) needs most of the GPU memory — vision should use Coral or CPU
- Camera capture must be thread-safe (use a lock for frame access)
- Handle gracefully when no camera is available (log warning, continue without vision)

After creating the files, verify syntax of both modified files.
```

---

## BATCH 3: Unify Arduino Control & Add Web Interface Bridge

### Prompt:

```
I need to unify Arduino serial control in the WALL-E project at C:/Users/Meiras/Desktop/memory/walle-main/.

**Problem**: Two components can control the Arduino:
1. `memory/robot_tools.py` — `RobotControlExecutor` (used by LLM tool calls)
2. `web_interface/app.py` — `ArduinoDevice` class (used by web dashboard)
Both open their own serial connections. Only one can use the serial port at a time.

**What to do**:

### Step 1: Create a shared serial manager

Create `walle-main/serial_manager.py`:
```python
class SerialManager:
    """Thread-safe singleton serial connection to Arduino."""
    def __init__(self, port: str, baud_rate: int = 9600):
        # Opens serial connection
        # Uses threading.Lock for thread safety

    def send_command(self, cmd: str) -> str:
        # Thread-safe write + optional read

    def close(self) -> None:
        # Clean shutdown

    def is_connected(self) -> bool:
        ...
```

### Step 2: Modify `memory/robot_tools.py`

Update `RobotControlExecutor` to accept an external serial connection instead of creating its own:
- Change `__init__` to accept optional `serial_manager: SerialManager = None`
- If provided, use `serial_manager.send_command()` instead of internal `self.serial_connection`
- If not provided, fall back to current behavior (create own connection or simulate)
- This is a minimal change — keep all existing tool handlers exactly as they are

### Step 3: Create a lightweight API bridge for the web interface

Create `walle-main/api_server.py`:
- A simple Flask or FastAPI server running on a separate port (e.g., 5001)
- Endpoints:
  - `POST /command` — send a serial command (used by web interface)
  - `GET /status` — return robot status (battery, connected, etc.)
  - `POST /chat` — send text to the LLM chat pipeline, return response
  - `GET /vision` — return latest vision context as JSON
- All endpoints use the shared `SerialManager` and `WallELLMClient`
- This replaces the web interface's direct serial access

### Step 4: Update `walle_main.py`

- Create `SerialManager` once at startup
- Pass it to `RobotControlExecutor`
- Pass it to `api_server` (if web interface is enabled)
- Add CLI arg `--web-port` (default 5001, 0 to disable)
- Start API server in background thread if enabled

**Key constraints**:
- The `web_interface/` folder already exists with Flask app, static files, and templates — do NOT modify it. The new API server is a separate lightweight bridge.
- Serial commands are simple strings like "Y50" (forward), "X50" (turn), "G50" (head), "q" (stop) — see `robot_tools.py` and `wall-e/AI_Control_Documentation.md`
- The Arduino serial runs at 9600 baud (configurable via `conf.BAUD_RATE`)
- Keep the simulation mode: if no serial port specified, all commands are logged but not sent

After making changes, verify:
1. Syntax check on all modified/created files
2. `robot_tools.py` still passes: `cd memory && python -c "from robot_tools import RobotControlExecutor; r = RobotControlExecutor(); print('OK')"`
```

---

## BATCH 4: Resource Optimization & Startup Sequence for Jetson

### Prompt:

```
I need to optimize the WALL-E system at C:/Users/Meiras/Desktop/memory/walle-main/ for the Jetson Orin Nano (8GB shared RAM+VRAM) and create a proper startup sequence.

**Current resource concerns**:
1. **Ollama LLM (qwen2.5:3b)**: ~2-3GB VRAM — this is the heaviest component
2. **Sentence-transformers (all-MiniLM-L6-v2)**: ~80MB — used for memory embeddings
3. **Moonshine STT**: ~200MB — speech-to-text model
4. **Mimic3 TTS**: runs as HTTP server, moderate memory
5. **Vision (if CPU mode)**: YOLOv8 + InsightFace can be heavy
6. **FAISS**: CPU-only, ~1.5MB per 10K vectors — lightweight

**What to do**:

### Step 1: Create a startup script `walle-main/start.sh`

A bash script that:
1. Checks if Ollama is running, starts it if not (`ollama serve &`)
2. Waits for Ollama to be ready (poll `curl http://localhost:11434/api/tags`)
3. Checks if the required model is pulled (`ollama list | grep qwen`)
4. Checks if Mimic3 TTS server is running, starts it if not
5. Activates the Python virtual environment if it exists
6. Launches `python walle_main.py` with appropriate args
7. Handles Ctrl+C gracefully (kills child processes)

### Step 2: Create `walle-main/config_jetson.py`

An override config for Jetson-specific settings:
- `OLLAMA_MODEL = "qwen3:4b"` (recommended for tool calling)
- `EMBEDDING_DEVICE = "cpu"` (keep GPU free for LLM)
- `EMBEDDING_BATCH_SIZE = 4` (lower for RAM constraint)
- `USE_FAISS = True` (CPU-based, very efficient)
- Vision FPS = 1 (reduce to save CPU)
- Add `JETSON_MODE = True` flag

### Step 3: Add lazy loading to `walle_main.py`

Modify the initialization in `walle_main.py` to load components sequentially with progress reporting:
1. First: Memory system (fast, ~100ms)
2. Second: Embedding model (if semantic search enabled, ~2s)
3. Third: Validate Ollama connection and model availability
4. Fourth: STT model (Moonshine, ~3s)
5. Fifth: TTS engine connection test
6. Sixth: Vision service (if enabled)
7. Seventh: Serial connection (if port specified)
8. Last: Start the main loop

Print a startup summary:
```
=== WALL-E System Status ===
LLM:    qwen3:4b via Ollama     [OK]
Memory: 3-tier + FAISS          [OK]
STT:    Moonshine small         [OK]
TTS:    Mimic3                  [OK]
Vision: Coral TPU               [OK]
Arduino: /dev/ttyUSB0           [OK]
API:    http://localhost:5001   [OK]
================================
Say "hey robot" to begin!
```

### Step 4: Add graceful shutdown to `walle_main.py`

Handle SIGINT/SIGTERM:
1. Stop vision service
2. Stop API server
3. Send neutral position to Arduino (`reset_to_neutral`)
4. Close serial connection
5. Save any pending memory (recall/archival flush)
6. Print shutdown message

### Step 5: Create `walle-main/requirements_jetson.txt`

A minimal requirements file for Jetson deployment (not the full 250+ package list):
- Core: openai, sentence-transformers, faiss-cpu, numpy, pyserial
- STT: moonshine-voice
- TTS: requests (for Mimic3 HTTP), sounddevice, scipy
- Vision: opencv-python (Jetson has its own build), pillow
- Web: flask or fastapi + uvicorn
- No PyTorch (Jetson has its own CUDA-aware build pre-installed)
- No ultralytics or insightface if using Coral

After creating all files, verify syntax and run: `python walle_main.py --help` to confirm CLI args work.
```

---

## BATCH 5: End-to-End Testing & Bug Fixes

### Prompt:

```
I need to test the integrated WALL-E system at C:/Users/Meiras/Desktop/memory/walle-main/ and fix any issues.

**Run these tests in order**:

### Test 1: Text-mode memory test
```bash
cd /c/Users/Meiras/Desktop/memory/walle-main
python walle_main.py --text-mode --no-vision
```
- Type "Hello, my name is Meiras"
- Verify the LLM responds via `send_message` tool (not raw output)
- Type "What is my name?"
- Verify it recalls from core memory or conversation
- Type "Remember that I like robotics"
- Verify it calls `archival_memory_insert` or `core_memory_append`
- Check that no Python errors/tracebacks occur

### Test 2: Robot tools test
```bash
python walle_main.py --text-mode --no-vision
```
- Type "Wave hello to me"
- Verify it calls `wave_hello` tool (should print simulation output)
- Type "Look around"
- Verify it calls `scan_surroundings` tool
- Type "Move forward slowly"
- Verify it calls `drive_forward` with low speed

### Test 3: Import validation
```bash
python -c "
import sys
sys.path.insert(0, 'memory')
sys.path.insert(0, 'stt_tts')
from walle_main import WallELLMClient, RobotBridge
print('Imports OK')
"
```

### Test 4: Vision service isolation test
```bash
python -c "
import sys
sys.path.insert(0, 'memory')
from vision_service import VisionService
from context_manager import ContextManager
cm = ContextManager()
vs = VisionService(cm, camera_index=-1)  # -1 = no camera, should handle gracefully
print('Vision service created OK')
print('Context:', cm.get_context_string())
"
```

### Test 5: Serial manager test
```bash
python -c "
from serial_manager import SerialManager
sm = SerialManager(port=None)  # No port = simulation
result = sm.send_command('Y50')
print('Serial result:', result)
sm.close()
print('Serial manager OK')
"
```

**For each test**:
1. Run it
2. If it fails, read the error, fix the root cause in the relevant file
3. Re-run to confirm the fix
4. Move to the next test

**Common issues to watch for**:
- Import path issues (memory/ and stt_tts/ need to be on sys.path)
- Ollama not running (the text-mode tests need Ollama — if unavailable, mock it)
- Missing dependencies (check what's installed, suggest pip install for missing ones)
- Thread cleanup on Ctrl+C (ensure no zombie threads)
- The `chat()` function in walle_enhanced.py uses global variables — the new orchestrator must either use those globals or create its own instances

After all tests pass, create a brief INTEGRATION_STATUS.md in the project root documenting what works, what needs hardware to test, and any known limitations.
```

---

## Summary of Batches

| Batch | Creates/Modifies | Purpose |
|-------|-----------------|---------|
| 1 | `walle_main.py` (new) | Wire STT/TTS ↔ Memory-aware LLM |
| 2 | `vision_service.py` (new), `walle_main.py` (modify) | Background vision → ContextManager |
| 3 | `serial_manager.py` (new), `api_server.py` (new), `robot_tools.py` (modify), `walle_main.py` (modify) | Unified Arduino access + web API |
| 4 | `start.sh` (new), `config_jetson.py` (new), `requirements_jetson.txt` (new), `walle_main.py` (modify) | Jetson optimization + startup |
| 5 | Various fixes | End-to-end testing |

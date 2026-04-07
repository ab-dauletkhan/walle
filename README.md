# WALL-E Robot Technical Specification

**Platform:** NVIDIA Jetson Orin Nano (8 GB RAM)
**LLM:** Ollama (qwen3:4b) via OpenAI-compatible API
**Memory Architecture:** MemGPT-inspired 3-tier system (Core / Recall / Archival)
**Vision:** Google Coral Edge TPU (primary), CPU YOLOv8+ArcFace (fallback)
**Voice:** Moonshine STT (streaming), Mimic3 TTS (HTTP)
**Hardware:** Arduino + PCA9685 (7 servos) + L298N (2 DC motors)

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Module 1: Memory & LLM Brain](#2-module-1-memory--llm-brain)
3. [Module 2: Computer Vision](#3-module-2-computer-vision)
4. [Module 3: Speech (STT & TTS)](#4-module-3-speech-stt--tts)
5. [Module 4: Robot Hardware & Serial (ROS-like Module)](#5-module-4-robot-hardware--serial-ros-like-module)
6. [Module 5: Web Interface & API](#6-module-5-web-interface--api)
7. [Configuration Reference](#7-configuration-reference)
8. [Startup & Shutdown Sequences](#8-startup--shutdown-sequences)
9. [Dependencies](#9-dependencies)
10. [Resource Budget (8 GB Jetson)](#10-resource-budget-8-gb-jetson)

---

## 1. System Architecture

```
                          +---------------------+
                          |     User (Voice)     |
                          +----------+----------+
                                     |
                     +---------------v---------------+
                     |   Moonshine STT (streaming)   |
                     |   IntentRecognizer + Router    |
                     +-------+---------------+-------+
                             |               |
                    robot intent        wake word detected
                             |               |
                  +----------v----+   +------v-----------+
                  | RobotBridge   |   | WallELLMClient   |
                  | (direct cmds) |   | (MemGPT chat)    |
                  +----------+----+   +------+-----------+
                             |               |
                             |    +----------v----------+
                             |    | Tool Execution Loop |
                             |    | (max 10 iterations) |
                             |    +---+---+---+---+----+
                             |        |   |   |   |
              +--------------+---+    |   |   |   +--------+
              |                  |    |   |   |            |
     +--------v------+  +-------v-+  |   |  +v--------+  +v-----------+
     | SerialManager  |  | Memory  |  |   |  |Knowledge|  |Personality |
     | -> Arduino     |  | Tools   |  |   |  |  Tools  |  |  Engine    |
     +--------+-------+  +---------+  |   |  +---------+  +------------+
              |                        |   |
     +--------v-------+    +----------v-+ +v-----------+
     | PCA9685 Servos  |    | send_msg   | | capture_   |
     | L298N DC Motors |    | -> TTS     | | image      |
     +----------------+    +------------+ +---+--------+
                                              |
                                    +---------v----------+
                                    |   VisionService    |
                                    | Coral TPU / CPU    |
                                    +--------------------+
```

### File Layout

```
walle-main/
+-- walle_main.py              # Unified orchestrator (WallELLMClient, main loop)
+-- serial_manager.py          # Thread-safe Arduino serial I/O
+-- vision_service.py          # Background vision (Coral / CPU backends)
+-- api_server.py              # REST API (Flask, port 5001)
+-- config_jetson.py           # Jetson Orin Nano overrides
+-- start.sh                   # Service startup script
|
+-- memory/
|   +-- config.py              # Central configuration dataclass
|   +-- memory_system.py       # 3-tier memory + FAISS vector search
|   +-- memory_tools.py        # LLM tool schemas for memory ops
|   +-- robot_tools.py         # LLM tool schemas for motor/servo
|   +-- communication_tools.py # send_message tool
|   +-- knowledge_tools.py     # DuckDuckGo web search tool
|   +-- personality_system.py  # Personality traits + persistence
|   +-- context_manager.py     # Multi-modal context aggregation
|   +-- heartbeat.py           # Multi-step reasoning continuation
|   +-- base_executor.py       # Abstract tool executor pattern
|
+-- stt_tts/
|   +-- main.py                # Voice assistant (STT + TTS + state machine)
|   +-- mock_llm.py            # LLMClient interface + mock/Ollama impl
|
+-- walle_vision/
|   +-- face_recognition_coral/
|       +-- recognize_face.py  # Coral face detection + embedding
|       +-- create_embeddings.py
|       +-- scan_people.py     # Face enrollment utility
|
+-- comp_vision_diplomka/
|   +-- pipeline.py            # CPU fallback (YOLOv8 + ArcFace ONNX)
|
+-- wall-e/
|   +-- wall-e.ino             # Arduino firmware (main)
|   +-- animations.ino         # Predefined servo animations
|   +-- display.ino            # OLED battery display
|   +-- L298NMotorController.hpp
|   +-- Queue.hpp              # Ring buffer for animation waypoints
|
+-- web_interface/
    +-- app.py                 # Flask web dashboard (port 5000)
    +-- config.py              # Web config (password, ports)
    +-- picamera2_stream.py    # MJPEG camera streaming
    +-- static/js/main.js      # Frontend JS
    +-- templates/             # Jinja2 HTML templates
```

---

## 2. Module 1: Memory & LLM Brain

### 2.1 Three-Tier Memory Architecture

```
+-------------------+    Always in LLM context window
| CORE MEMORY       |    persona (2000 chars), human (2000 chars), system (1000 chars, read-only)
| (in-context)      |    Editable via core_memory_append / core_memory_replace tools
+-------------------+
         |
+-------------------+    Last 40 messages, FAISS-indexed
| RECALL MEMORY     |    Auto-compressed to archival when limit exceeded
| (recent history)  |    Semantic search via all-MiniLM-L6-v2 embeddings (384-dim)
+-------------------+
         |
+-------------------+    Facts, preferences, summaries
| ARCHIVAL MEMORY   |    Importance-weighted (1-10) with exponential recency decay
| (long-term)       |    effective = importance * 0.7 + recency_score * 0.3
+-------------------+    recency_score = exp(-age_days / 30)
```

### 2.2 Core Memory

**Storage:** SQLite (`walle_core_memory.db`)

```sql
CREATE TABLE core_memory (
  label       TEXT PRIMARY KEY,
  value       TEXT,
  limit_chars INTEGER,
  description TEXT,
  read_only   INTEGER,
  metadata    TEXT  -- JSON: {created_at, last_modified}
);
```

| Block   | Limit  | Writable | Purpose                              |
|---------|--------|----------|--------------------------------------|
| persona | 2000   | Yes      | WALL-E's self-knowledge              |
| human   | 2000   | Yes      | Facts about current user             |
| system  | 1000   | No       | Base system context                  |

**API:** `Block.append(content)`, `Block.replace(old, new)`, `Memory.compile() -> XML string`

### 2.3 Recall Memory

**Storage:** SQLite (`walle_recall_memory.db`) + FAISS index

```sql
CREATE TABLE recall_memory (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  role      TEXT,        -- 'user' | 'assistant'
  content   TEXT,
  tools_used TEXT,       -- JSON array
  metadata   TEXT,       -- JSON
  embedding  BLOB        -- float32[384] bytes
);
CREATE INDEX idx_timestamp ON recall_memory(timestamp DESC);
```

| Parameter                  | Value | Description                        |
|----------------------------|-------|------------------------------------|
| RECALL_MEMORY_LIMIT        | 40    | Triggers compression when exceeded |
| MAX_PENDING_EMBEDDINGS     | 50    | Backpressure threshold             |
| Embedding workers          | 2     | ThreadPoolExecutor max_workers     |
| Deferred embedding         | Yes   | Background thread for TTFT         |

**Compression:** When count > 40, a background thread summarizes old messages via the LLM, inserts the summary into archival memory, and deletes the originals.

### 2.4 Archival Memory

**Storage:** SQLite (`walle_archival_memory.db`) + FAISS index

```sql
CREATE TABLE archival_memory (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP,
  category   TEXT,       -- 'fact', 'preference', 'conversation_summary'
  content    TEXT,
  importance INTEGER,    -- 1-10
  metadata   TEXT,       -- JSON
  embedding  BLOB        -- float32[384] bytes
);
```

**Importance Decay Formula:**
```
effective = importance * 0.7 + recency_score * 10 * 0.3
recency_score = exp(-age_days / 30.0)
```

### 2.5 FAISS Vector Search

| Parameter       | Value                   |
|-----------------|-------------------------|
| Index type      | `IndexFlatIP`           |
| Dimension       | 384                     |
| Similarity      | Inner Product (cosine)  |
| Embeddings      | L2-normalized float32   |
| Persistence     | Binary .index + .ids JSON |
| Async save      | After every 10 insertions |

**Search pipeline:** Query embedding -> FAISS k-NN -> map internal IDs to DB row IDs -> fetch full records -> apply importance decay ranking (archival only).

### 2.6 Embedding Model

| Parameter        | Value                                     |
|------------------|-------------------------------------------|
| Model            | `sentence-transformers/all-MiniLM-L6-v2`  |
| Dimension        | 384                                       |
| Size             | ~80 MB                                    |
| Device (Jetson)  | CPU (GPU reserved for LLM)                |
| Batch size       | 4 (Jetson) / 8 (default)                  |
| Normalization    | L2 (for cosine similarity via dot product)|

### 2.7 LLM Chat Loop

**Model:** `qwen3:4b` (Jetson) or `qwen2.5:3b` (default) via Ollama

**System Prompt Structure:**
```
[Base instruction: You are WALL-E. Reply ONLY via send_message tool.]
[Personality: humor=50%, honesty=90%, sass=20%]
[Tool usage instructions]
[CURRENT CONTEXT: visual + environment + interaction]
[Core memory blocks compiled as XML]
```

**Chat Loop (`_run_chat_loop`):**
1. Update context (interaction timestamp, environment every 5 turns)
2. Parallel memory retrieval: recall (3 results) + archival (2 results)
3. Insert user message with deferred embedding
4. Check compression threshold (>40 messages -> background thread)
5. **Tool execution loop** (max 10 iterations):
   - Stream LLM response with all tool schemas
   - Execute tool calls via dispatchers
   - `send_message` -> break loop
   - `request_heartbeat` -> continue (max 5 heartbeats)
6. Return final message

### 2.8 LLM Tool Schemas

**Memory Tools:**
- `core_memory_append(label, content)` — Append to persona or human block
- `core_memory_replace(label, old_content, new_content)` — Edit block text
- `archival_memory_insert(category, content, importance)` — Store long-term fact
- `recall_memory_search(query, limit)` — Search recent conversation

**Communication:**
- `send_message(message, request_heartbeat)` — ONLY way to talk to user

**Robot Control:**
- `drive_forward(speed, duration_ms)` — Y{speed}
- `drive_backward(speed, duration_ms)` — Y{-speed}
- `turn_left(speed, duration_ms)` — X{-speed}
- `turn_right(speed, duration_ms)` — X{speed}
- `stop_movement()` — q
- `set_head_rotation(position)` — G{0-100}
- `set_neck_position(position)` — Adapter: T{position} + B{100-position}
- `set_both_arms(left, right)` — L{val}\nR{val}
- `express_emotion(emotion)` — Arm + dual-neck presets
- `scan_surroundings(speed)` — Head sweep G20->G50->G80->G50
- `wave_hello()` — Right-arm wave sequence
- `reset_to_neutral()` — A0 neutral animation

**Knowledge:**
- `consult_internet_for_facts(query)` — DuckDuckGo search, English filtered, multi-region fallback

**Personality:**
- `set_personality(trait, value)` — humor/honesty/sass (0-100)

**Vision:**
- `capture_image()` — Grab frame from vision service

---

## 3. Module 2: Computer Vision

### 3.1 Dual-Backend Architecture

```
VisionService (background thread, 1-2 FPS)
    |
    +-- try: CoralVisionBackend (Edge TPU)
    |        Face det: SSD MobileNet v2 (quantized)
    |        Face emb: MobileNet triplet (96x96)
    |
    +-- fallback: CPUVisionBackend
             Face det: YOLOv8-nano
             Face emb: ArcFace ResNet-50 (ONNX, 112x112)
```

### 3.2 Coral TPU Backend

| Component       | Model File                                                      | Input       | Output           |
|-----------------|-----------------------------------------------------------------|-------------|------------------|
| Face Detection  | `ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite`        | Variable    | Boxes, scores    |
| Face Embedding  | `Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite`     | 96x96 RGB   | Embedding vector |
| TPU Delegate    | `libedgetpu.so.1.0`                                            | -           | -                |

**Pipeline per frame:**
1. Capture BGR frame from OpenCV
2. Convert to RGB PIL image
3. Resize to model input dimensions
4. SSD MobileNet inference -> bounding boxes + scores
5. Filter by `VISION_FACE_DETECTION_THRESHOLD` (0.5)
6. For detections above `VISION_FACE_RECOGNITION_MIN` (0.8):
   - Crop face, resize to 96x96
   - Quantize input: `uint8(img / scale + zero_point)`
   - Generate embedding, dequantize output
   - Cosine similarity against stored embeddings (max 20 per person)
   - Match threshold: 0.5
7. Return `[{name, confidence, location}]`

### 3.3 CPU Fallback Backend

| Component       | Model File              | Input       | Output            |
|-----------------|-------------------------|-------------|-------------------|
| Face Detection  | `yolov8n-face.pt`       | BGR frame   | Boxes in xyxy     |
| Face Embedding  | `w600k_r50.onnx`        | 112x112 RGB | 512-dim embedding |
| Runtime         | ONNX Runtime (CPU)      | -           | -                 |

**ArcFace preprocessing:** `(img / 255.0 - 0.5) / 0.5`, NCHW layout, L2-normalized output.
**Match threshold:** 0.6 (higher than Coral due to different embedding space).

### 3.4 Face Enrollment

**Utility:** `scan_people.py`
- Camera: 1280x960 @ 30 FPS
- Detection threshold: 0.9 (conservative for clean enrollment)
- Captures 20+ images per person
- Stores cropped 96x96 faces as `.npy` arrays
- `create_embeddings.py` converts crops to embedding vectors

**Storage:**
```
scanned_people/{person_name}/
    embeddings/    # .npy embedding vectors (max 20 loaded at runtime)
    npy/           # Raw face crops
    png/           # Visual verification images
```

### 3.5 Vision Service Threading

| Parameter                 | Value      |
|---------------------------|------------|
| Target FPS (Jetson)       | 1          |
| Target FPS (default)      | 2          |
| Max consecutive errors    | 5          |
| Backoff base              | 0.5 s      |
| Backoff max               | 30.0 s     |
| Frame access              | Lock-protected `.copy()` |
| Thread type               | Daemon     |
| Join timeout              | 5.0 s      |

**Output:** Updates `ContextManager.visual` with `VisualContext(faces_detected, objects_detected, scene_description, timestamp)`.

---

## 4. Module 3: Speech (STT & TTS)

### 4.1 STT: Moonshine (Streaming)

| Parameter        | Value                              |
|------------------|------------------------------------|
| Library          | `moonshine-voice`                  |
| Model            | `SMALL_STREAMING` (~80 MB)         |
| Embedding model  | `embeddinggemma-300m` (q4)         |
| Input            | Microphone (default audio device)  |
| Output           | Streaming text via callbacks       |

**Intent Recognition:**
- Embedding-based semantic matching
- Threshold: 0.65 cosine similarity
- Default intents: forward, backward, left, right, stop, wave, dance
- Matched intents routed directly to RobotBridge (bypasses LLM)

### 4.2 TTS: Mimic3 (HTTP)

| Parameter    | Value                        |
|--------------|------------------------------|
| Server       | `http://localhost:59125`     |
| Voice        | `en_UK/apope_low`           |
| Format       | WAV (server-side synthesis)  |
| Playback     | `sounddevice` (blocking)     |
| Blocksize    | 4096 samples                 |
| Latency      | "high" (prevent underrun)    |
| Timeout      | 30 seconds                   |
| Fallback     | Console print on failure     |

### 4.3 Voice State Machine

```
          wake word detected
IDLE ──────────────────────────> LISTENING
  ^                                  |
  |    silence timeout fires         |
  +----------------------------------+
           (send to LLM)
```

**States:**
- **IDLE:** Background speech ignored. Only robot intents (via IntentRecognizer) are active.
- **LISTENING:** Collecting speech for LLM query. Silence timeout starts on each new utterance.

### 4.4 Wake Word Detection

**Algorithm:** Word-boundary-aware sequential token matching.

```python
# Normalize both sides (strip punctuation, lowercase)
# Match wake tokens in sequence within text tokens
# "Hey, robot! What time is it?" with wake "hey robot"
#   -> tokens: [hey, robot, what, time, is, it]
#   -> match: hey(0), robot(1) -> return "what time is it"
```

No substring matching (prevents false positives like "rocket launcher" triggering "rocket").

### 4.5 Echo Suppression

**Two-layer system:**

| Layer          | Mechanism                  | Duration | Threshold |
|----------------|----------------------------|----------|-----------|
| Time-based     | `_speaking` flag           | During playback + cooldown | - |
| Content-based  | Word overlap matching      | 4.0 s after TTS | 60% overlap |

**Post-speak timeout:** `max(listen_timeout * 3, ECHO_WINDOW + 1)` ensures the listen timeout never expires while echo suppression is still active.

### 4.6 Audio Parameters

| Parameter        | Value     | Purpose                     |
|------------------|-----------|-----------------------------|
| AUDIO_BLOCKSIZE  | 4096      | Samples per block           |
| AUDIO_LATENCY    | "high"    | Prevents buffer underrun    |
| ECHO_COOLDOWN    | 2.0 s     | Post-TTS silence period     |
| ECHO_WINDOW      | 4.0 s     | Content-based echo window   |
| WAKE_DEBOUNCE    | 1.5 s     | Prevent rapid re-trigger    |
| Conversation max | 20 msgs   | Rolling window (trimmed)    |

---

## 5. Module 4: Robot Hardware & Serial (ROS-like Module)

### 5.1 Hardware Overview

**Servo Motors (PCA9685, I2C 0x40, 60 Hz):**

| Ch | Joint         | Low PWM | High PWM | Max Vel (u/s) | Accel (u/s^2) |
|----|---------------|---------|----------|----------------|----------------|
| 0  | Head Rotation | 410     | 120      | 500            | 350            |
| 1  | Neck Top      | 532     | 178      | 400            | 300            |
| 2  | Neck Bottom   | 120     | 310      | 500            | 480            |
| 3  | Eye Right     | 465     | 271      | 2400           | 1800           |
| 4  | Eye Left      | 278     | 479      | 2400           | 1800           |
| 5  | Arm Left      | 340     | 135      | 600            | 500            |
| 6  | Arm Right     | 150     | 360      | 600            | 500            |

**DC Motors (L298N):**

| Motor | IN1 | IN2 | ENA/ENB | Max Vel | Accel    |
|-------|-----|-----|---------|---------|----------|
| Left  | D2  | D4  | D3 (PWM)| 255     | 800 u/s^2|
| Right | D5  | D7  | D6 (PWM)| 255     | 800 u/s^2|

**Other pins:** Servo enable = D11 (LOW=enabled), Battery = A2 (optional).

### 5.2 Serial Protocol

**Baud:** 115200, **Terminator:** `\n`, **ACK:** `"OK\n"`, **Buffer:** 6 chars max.

**Command Reference:**

| Cmd | Format       | Range       | Function                           |
|-----|--------------|-------------|------------------------------------|
| Y   | Y{-100..100} | -255..255 PWM | Forward/reverse drive            |
| X   | X{-100..100} | -255..255 PWM | Left/right turn                  |
| S   | S{-100..100} | raw          | Steering trim offset              |
| O   | O{0..250}    | raw          | Motor deadzone                    |
| G   | G{0..100}    | % of range   | Head rotation                     |
| T   | T{0..100}    | % of range   | Neck top                          |
| B   | B{0..100}    | % of range   | Neck bottom                       |
| E   | E{0..100}    | % of range   | Left eye                          |
| U   | U{0..100}    | % of range   | Right eye                         |
| L   | L{0..100}    | % of range   | Left arm                          |
| R   | R{0..100}    | % of range   | Right arm                         |
| A   | A{0..2}      | animation #  | Play predefined animation         |
| M   | M{0\|1}      | boolean      | Autonomous mode toggle            |
| w/a/s/d/q | single char | -       | WASD movement + stop              |
| ?   | ?            | -            | Heartbeat (returns STATUS line)   |

**Servo percentage to PWM:** `setpos[i] = number * 0.01 * (preset[i][1] - preset[i][0]) + preset[i][0]`

### 5.3 Physics-Based Servo Motion Control

**Update rate:** Every 10 ms (`SERVO_UPDATE_TIME`)

```
posError = setpos - curpos

IF |posError| > 1 (CONTROLLER_THRESHOLD):
    stoppingDistance = curvel^2 / (2 * accel)
    IF stoppingDistance > |posError|:
        acceleration = -accel    # Decelerate
    ELSE:
        acceleration = accel     # Accelerate

    curvel += acceleration * dt / 1000
    curvel = clamp(curvel, -maxvel, maxvel)

    dP = curvel * dt / 1000
    curpos += min(|dP|, |posError|) * sign(dP)
```

**Safety features:**
- **Servo stuck detection:** If position unchanged for 3000 ms, servo is disabled (PWM set to 0) and warning sent. Re-enabled when new command targets that servo.
- **Servo auto-off:** All servos disabled after 6000 ms of no movement (prevents overheating).
- **Motor trim snapshot:** `moveValue`, `turnValue`, `turnOffset` captured at start of `manageMotors()` to prevent mid-update changes from serial reads.

### 5.4 Predefined Animations

| # | Name       | Frames | Duration | Description                     |
|---|------------|--------|----------|---------------------------------|
| 0 | Reset      | 1      | 1.0 s    | Neutral startup position        |
| 1 | Bootup     | 9      | ~9.0 s   | Eye blink sequence              |
| 2 | Inquisitive| 10     | ~19.5 s  | Curious head/arm movements      |

**Autonomous mode:** When queue is empty and `autoMode=true`, randomly generates joint positions every 500-3000 ms. Eyes are coordinated (move together or in opposite directions).

### 5.5 Python Serial Manager

**Thread-safe** with `threading.Lock()`. Validates commands before sending. Falls back to simulation mode if no port or pyserial unavailable.

| Method           | Purpose                              | Timeout |
|------------------|--------------------------------------|---------|
| `send_command()` | Validate + send + wait ACK           | 2.0 s   |
| `heartbeat()`    | Send `?`, return STATUS line         | 1.0 s   |
| `read_line()`    | Non-blocking serial read             | 0.5 s   |

**Heartbeat response format:** `STATUS 248,560,140,475,270,250,290 M0,0`
(7 servo positions, then left/right motor velocities)

---

## 6. Module 5: Web Interface & API

### 6.1 REST API (port 5001)

| Method | Endpoint   | Request Body             | Response                        |
|--------|------------|--------------------------|---------------------------------|
| POST   | /command   | `{"command": "G50"}`     | `{"status": "OK"}`             |
| GET    | /status    | -                        | `{"connected": bool, "simulation": bool}` |
| POST   | /chat      | `{"text": "..."}`        | `{"response": "..."}`          |
| GET    | /vision    | -                        | `{"faces": [...], "objects": [...], ...}` |

**Validation on /command:** `len(cmd) <= 20`, matches `^[A-Za-z0-9\n\s?-]+$`.
**Chat endpoint:** Serialized with 60s lock timeout. Returns generic error message (no internal leak).

### 6.2 Web Dashboard (port 5000)

**Authentication:** Session-based login. Password from `WALLE_PASSWORD` env var (default: `"walle"`).
**Secret key:** From `WALLE_SECRET_KEY` env var (random generated if unset).

**Features:**
- Joystick motor control (POST /motor with stickX/stickY)
- Individual servo sliders (POST /servoControl)
- Animation playback (POST /animate)
- Audio file playback with volume control
- Text-to-speech via espeak-ng + rubberband pitch shifting
- Arduino serial port selection and connection
- Battery level monitoring
- Camera stream toggle (MJPEG via picamera2 on port 8080)
- System restart / shutdown

---

## 7. Configuration Reference

### 7.1 Core Config (`memory/config.py`)

| Parameter                    | Default                              | Jetson Override  |
|------------------------------|--------------------------------------|------------------|
| OLLAMA_BASE_URL              | `http://localhost:11434`             | -                |
| OLLAMA_MODEL                 | `qwen2.5:3b`                        | `qwen3:4b`       |
| EMBEDDING_MODEL              | `all-MiniLM-L6-v2`                  | -                |
| EMBEDDING_DEVICE             | `cuda`                               | `cpu`            |
| EMBEDDING_BATCH_SIZE         | 8                                    | 4                |
| USE_FAISS                    | True                                 | True             |
| FAISS_DIMENSION              | 384                                  | -                |
| USE_SEMANTIC_SEARCH          | True                                 | True             |
| RECALL_MEMORY_LIMIT          | 40                                   | -                |
| MAX_CONTEXT_MESSAGES         | 10                                   | -                |
| IMPORTANCE_DECAY_HALF_LIFE   | 30.0 days                            | -                |
| VISION_FACE_DETECTION_THRESHOLD | 0.5                               | -                |
| VISION_FACE_MATCH_THRESHOLD  | 0.5 (Coral) / 0.6 (CPU)             | -                |
| SERIAL_PORT                  | None (simulation)                    | -                |
| BAUD_RATE                    | 115200                               | -                |
| VISION_FPS                   | 2                                    | 1                |

### 7.2 Voice Config (CLI flags)

| Flag              | Default              | Description               |
|-------------------|----------------------|---------------------------|
| --wake-word       | "hey robot"          | Wake phrase                |
| --listen-timeout  | 3.0                  | Silence timeout (seconds)  |
| --tts-voice       | en_UK/apope_low      | Mimic3 voice ID            |
| --stt-model       | small-streaming      | Moonshine model size       |

---

## 8. Startup & Shutdown Sequences

### 8.1 Startup (`start.sh` + `walle_main.py`)

```
1. Start Ollama LLM server (wait up to 30s)
2. Pull model if missing (qwen3:4b on Jetson)
3. Check Mimic3 TTS server (optional)
4. Activate Python venv
5. Initialize memory system (Core + Recall + Archival)
6. Validate Ollama reachability + model availability
7. Start VisionService (background daemon thread)
8. Connect Arduino serial (or enter simulation mode)
9. Start API server (Flask daemon thread, port 5001)
10. Test TTS health
11. Load STT model (if voice mode)
12. Register signal handlers (SIGINT, SIGTERM)
13. Enter text REPL or voice mode
```

### 8.2 Shutdown (`_shutdown()`)

```
1. Stop VisionService (join thread, release camera)
2. Stop API server
3. Wait for memory compression thread (max 15s)
4. Reset robot to neutral position (A0 neutral animation)
5. Close serial connection
6. Sync-save FAISS indices to disk
7. Shutdown embedding thread pool (wait for in-flight futures)
8. Save personality state
```

---

## 9. Dependencies

### 9.1 Jetson Minimal (`requirements_jetson.txt`)

| Package              | Purpose                    |
|----------------------|----------------------------|
| openai >= 1.0        | Ollama client              |
| sentence-transformers | Embeddings (all-MiniLM)   |
| faiss-cpu            | Vector search              |
| pyserial             | Arduino serial             |
| moonshine-voice      | STT                        |
| sounddevice, scipy   | Audio playback             |
| opencv-python        | Camera + vision            |
| pillow               | Image processing           |
| flask                | API + web server           |
| pydantic             | Data validation            |
| requests             | HTTP (TTS, Ollama)         |
| numpy                | Numerical computing        |

**Optional:** `pycoral` + `tflite-runtime` (Coral TPU), `ultralytics` + `onnxruntime` (CPU vision), `duckduckgo_search` (knowledge tool).

---

## 10. Resource Budget (8 GB Jetson)

| Component                     | RAM Estimate  |
|-------------------------------|---------------|
| Ollama qwen3:4b (quantized)   | 2.0-3.0 GB   |
| Sentence-transformers (CPU)    | ~350 MB       |
| FAISS indices + SQLite         | ~50 MB        |
| Python process + libraries     | ~100 MB       |
| Vision backend (Coral or YOLO) | 100-200 MB    |
| Frame buffers (640x480)        | ~10-50 MB     |
| Moonshine STT model            | ~80 MB        |
| **Total**                      | **~3-4 GB**   |
| **Headroom**                   | **~4-5 GB**   |

**Key optimizations for 8 GB:**
- Embeddings on CPU (GPU reserved for LLM inference)
- Vision FPS capped at 1 (reduces CPU load)
- Embedding batch size reduced to 4
- Deferred embeddings (background thread, not blocking chat)
- Memory compression (keeps recall under 40 messages)
- Embedding queue backpressure (max 50 pending, prevents OOM)
- Conversation history capped at 20 messages (voice mode)

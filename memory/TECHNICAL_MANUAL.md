## This document covers the internal architecture, memory logic, and robot control protocols.

## 1. System Architecture

The architecture treats the LLM as an operating system that manages its own context window rather than just a text generator.

### A. The Cognitive Loop (`walle_enhanced.py`)
The main loop constructs a dynamic `System Prompt` on every turn:
1.  **Context Injection**: `ContextManager` injects "sensory" data (Vision, Battery, Interaction history) into the prompt.
2.  **Prompt Construction**: Combines Persona + Protocol + Context + Core Memory Blocks.
3.  **Execution**: The LLM generates a response or a tool call.

### B. Heartbeat Mechanism (`heartbeat.py`)
Standard LLMs reply once and stop. WALL-E uses a **Heartbeat** to chain actions:
* If a tool returns `request_heartbeat=True` (e.g., after a search), the system feeds the result back to the LLM with the message *"Heartbeat requested. Continue..."*.
* This allows complex chains: `Search` -> `Read Result` -> `Save to Memory` -> `Reply`.

---

## 2. Memory System Guide

WALL-E implements a tiered memory architecture designed to solve the "Finite Context Window" problem.

### Tier 1: Core Memory (RAM)
* **Location**: Inside the System Prompt (Always Visible).
* **Storage**: `walle_core_memory.db`
* **Tools**: `core_memory_append`, `core_memory_replace`.
* **Usage**: Holds the agent's identity (`persona` block) and user profile (`human` block).

### Tier 2: Recall Memory (History)
* **Location**: Vector Database (SQLite + Embeddings).
* **Storage**: `walle_recall_memory.db`
* **Logic**: 
    * Ingests every message using `nomic-embed-text`.
    * Performs Cosine Similarity search when the user asks questions.
    * Auto-compresses old messages into summaries when limits are reached.

### Tier 3: Archival Memory (Hard Drive)
* **Location**: SQLite Database.
* **Storage**: `walle_archival_memory.db`
* **Tools**: `archival_memory_insert`, `archival_memory_search`.
* **Usage**: Infinite storage for facts found online or deep historical summaries.

---

## 3. Robot Control Tools & Protocol

The system communicates with hardware via Serial (USB) using a custom G-Code-like protocol. If no serial port is configured, these run in **Simulation Mode**.

### Servo Tools
* `set_head_rotation(position)`: 0 (Left) - 100 (Right).
* `set_neck_position(top, bottom)`: Control neck height and tilt.
* `set_both_eyes(position)`: Control eye tilt (Expression).
* `look_at(horizontal, vertical)`: Coordinated head+neck movement.

### Emotion Macros
* `express_emotion(emotion)`: Executes preset servo positions.
    * **Happy**: Eyes Up, Head Up.
    * **Sad**: Eyes Down, Head Down.
    * **Curious**: Head Tilted.
* `wave_hello`: Executes an arm animation sequence.

### Movement Tools (Blocking)
* `drive_forward(speed, duration_ms)`
* `turn_left(speed, duration_ms)`
* `stop_movement()`: Emergency stop.
* **Note**: These functions block execution until movement is complete, ensuring the robot doesn't speak while its motors are loud.

### Hardware Protocol Reference
If building your own hardware, your Arduino sketch must accept these newline-terminated strings:

| Command | Description | Range |
|---------|-------------|-------|
| `G<val>` | Head Pan | 0-100 |
| `N<val>` | Neck Top | 0-100 |
| `L<val>` | Left Eye | 0-100 |
| `R<val>` | Right Eye | 0-100 |
| `A<val>` | Left Arm | 0-100 |
| `B<val>` | Right Arm | 0-100 |
| `Y<val>` | Drive Fwd/Rev| -100 to 100 |
| `X<val>` | Turn L/R | -100 to 100 |
| `q` | Stop Motors | N/A |

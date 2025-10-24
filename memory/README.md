# WALL-E: A MemGPT-Powered Robot Companion

This project is a Python-based robot companion, WALL-E, featuring an advanced memory system inspired by the MemGPT paper and a configurable personality engine.

## Core Features

*   **Three-Tier Memory System:**
    *   **Core Memory:** Always-on context for self-identity and user information.
    *   **Recall Memory:** Searchable database of recent conversation history.
    *   **Archival Memory:** Long-term storage for important facts and user preferences.
*   **Autonomous Memory Management:** The AI uses a toolkit of functions to manage its own memory in real-time.
*   **Configurable Personality:** Traits like Humor, Honesty, Sass, and Curiosity can be adjusted on-the-fly, directly influencing the robot's behavior.
*   **Multi-Step Reasoning:** A "heartbeat" mechanism allows the robot to chain multiple actions to solve complex requests without user interruption.
*   **Persistent State:** Core memory and personality settings are saved and loaded across sessions.

## System Architecture

*   `walle_enhanced.py`: Main application entry point.
*   `memory_system.py`: Implements the Core, Recall, and Archival memory tiers.
*   `memory_tools.py`: Defines the functions the AI uses to interact with its memory.
*   `personality_system.py`: Manages the TARS-inspired personality traits.
*   `heartbeat.py`: Enables multi-step "chain of thought" reasoning.
*   `context_manager.py`: (Future Integration) Designed for handling real-world sensor data.

## Setup & Installation

1.  **Install Ollama:** Follow the instructions at [https://ollama.ai](https://ollama.ai).

2.  **Pull an AI Model:** This project is optimized for smaller models.
    ```bash
    ollama pull qwen3:1.7b
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install "openai>=1,<2"
    ```
    *Note: For optional semantic search, also run `pip install sentence-transformers`.*

4.  **Configure the Model:** Open `walle_enhanced.py` and ensure the `MODEL_NAME` variable matches your pulled model (e.g., `"qwen3:1.7b"`).

## Running the Robot

Ensure the Ollama service is running in the background, then execute the main script:

```bash
python walle_enhanced.py
```

## Usage

*   **Chat Naturally:** Interact with WALL-E using natural language.
*   **Adjust Personality:**
    *   `set humor to 90`
    *   `set sass to 10`
*   **View Settings:**
    *   `show personality`
*   **Exit:**
    *   `exit` or `quit` (this will save the current memory and personality state).

## Utilities

*   **Check Environment:** Run this script to verify your Ollama connection and see available models.
    ```bash
    python ollama_diagnostics.py
    ```
*   **Inspect Memory:** Run this script to view the contents of WALL-E's core, recall, and archival memory databases.
    ```bash
    python memory_inspector.py
    ```
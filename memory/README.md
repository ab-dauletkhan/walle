# WALL-E: MemGPT-Inspired Robot Brain 🤖

**WALL-E** is an autonomous agent framework designed for local execution. It transforms a standard LLM into a persistent personality with a "physical" body, featuring a three-tiered memory system (Core, Recall, Archival), a multi-modal context manager (simulating vision/sensors), and direct hardware control.

## 🌟 Key Features

* **🧠 Three-Tier Memory System**:
    * **Core Memory**: Persistent instructions and persona residing *inside* the context window. Editable by the LLM.
    * **Recall Memory**: Vector-backed conversation history (using `nomic-embed-text`).
    * **Archival Memory**: Long-term storage for facts and deep history.
* **💓 Heartbeat / Chain-of-Thought**: A mechanism allowing the robot to "think" or perform multiple actions (e.g., "Look left" -> "Scan" -> "Speak") without user interruption.
* **👀 Context Awareness**: Simulates sensory inputs (Vision, Battery, Environment) injected dynamically into the prompt.
* **🦾 Robotics Control**: Direct serial port integration for motor control, head rotation, and emotional expression.
* **🌍 Internet Connected**: Tools to fetch real-time data (weather, news) using DuckDuckGo.

## 🛠️ Prerequisites

* **Python 3.10+**
* **Ollama** running locally ([Download](https://ollama.ai)).
* **Hardware (Optional)**: An Arduino/Serial robot. The system defaults to "Simulation Mode" if no device is found.

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/ab-dauletkhan/walle](https://github.com/ab-dauletkhan/walle)
    cd walle
    ```

2.  **Install Dependencies**:
    ```bash
    pip install openai requests duckduckgo-search numpy
    ```

3.  **Setup Ollama Models**:
    You need the base text model and the embedding model for memory search.
    ```bash
    ollama pull qwen3:4b
    ollama pull nomic-embed-text
    ```

4.  **Create the Custom Brain**:
    Create a file named `Modelfile` (no extension) in your project folder:
    ```dockerfile
    FROM qwen3:4b
    # Set context window to 4096 tokens (Low VRAM usage)
    PARAMETER num_ctx 4096
    # Set the system prompt permanently
    SYSTEM "You are WALL-E, a helpful robot companion."
    ```
    Then build the model:
    ```bash
    ollama create walle-brain -f Modelfile
    ```

## 🏃 Usage

### Start the Brain
```bash
python walle_enhanced.py
````

*This initializes the cognitive loop, connects to databases, and starts the simulation.*

### Memory Inspector

To see what is actually stored in the database (Core blocks, Recall vectors, etc.):

```bash
python memory_inspector.py
```

## ⚙️ Configuration

Edit `config.py` to adjust:

  * **`SERIAL_PORT`**: Set to your Arduino port (e.g., `COM3` or `/dev/ttyUSB0`) for hardware control.
  * **`SEARCH_REGIONS`**: Customize DuckDuckGo search regions (default: `["wt-wt", "us-en"]`).
  * **`RECALL_MEMORY_LIMIT`**: How many messages to keep before summarizing to archival.

## 🧪 Quick System Tests

Run these steps to verify your installation:

1.  **Configuration Check**:
    Run `python config.py`. Expect no errors.

2.  **Memory Write Test**:

      * Start the bot: `python walle_enhanced.py`
      * Say: *"My name is [Your Name]. I am a Python developer."*
      * Exit and run `python memory_inspector.py`. Check the **Core Memory** section; the `<human>` block should now contain your name.

3.  **Tool Use Test**:

      * Ask: *"What is the price of Bitcoin?"* -\> Should trigger `consult_internet_for_facts`.
      * Command: *"Look left and wave."* -\> Should trigger `set_head_rotation` and `wave_hello`.
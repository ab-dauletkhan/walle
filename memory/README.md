# WALL-E: Memory-Augmented Robot Brain

**WALL-E v3.1** is an autonomous agent framework designed for local execution on NVIDIA Jetson devices. It transforms a standard LLM into a persistent personality with a "physical" body, featuring a three-tiered memory system with FAISS-accelerated search, importance decay for fact prioritization, and direct hardware control.

**New in v3.1**:
- FAISS integration for 5-10x faster semantic search
- Importance decay for intelligent fact prioritization
- Ollama with Qwen3-4B as recommended backend (easier setup, excellent tool calling)

## Key Features

* **Ollama Backend**: Simple setup with Qwen3-4B for excellent tool calling
* **Three-Tier Memory System**:
    * **Core Memory**: Persistent persona and user profile in context window (LLM-editable)
    * **Recall Memory**: FAISS-accelerated conversation history with semantic search
    * **Archival Memory**: Long-term facts with importance decay
* **FAISS Vector Search**: O(log n) semantic search instead of O(n) naive cosine similarity
* **Importance Decay**: Recent facts prioritized over older ones using exponential decay
* **Heartbeat / Chain-of-Thought**: Multi-step reasoning without user interruption
* **Context Awareness**: Simulated sensory inputs (Vision, Battery, Environment)
* **Robotics Control**: Serial port integration for motor control and expressions
* **Internet Connected**: Real-time data fetching via DuckDuckGo

## Quick Start (Recommended: Ollama)

### 1. Clone and Install

```bash
git clone <your-repo-url> memory
cd memory
pip install -r requirements.txt
```

### 2. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Pull Qwen3-4B Model

```bash
ollama pull qwen3:4b
```

### 4. Run WALL-E

```bash
python walle_enhanced.py
```

That's it! The system uses Ollama with Qwen3-4B by default.

## Configuration

Edit `config.py` to customize:

### Ollama Settings
```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:4b"    # Best tool calling among small models
```

### Memory Settings
```python
USE_SEMANTIC_SEARCH = True   # Enable FAISS-accelerated vector search
USE_FAISS = True             # Enable FAISS (falls back to naive cosine if False)
RECALL_MEMORY_LIMIT = 40     # Messages before compression to archival
```

### Importance Decay
```python
IMPORTANCE_DECAY_HALF_LIFE = 30.0  # Days until recency score halves
IMPORTANCE_STATIC_WEIGHT = 0.7     # Weight for static importance
IMPORTANCE_RECENCY_WEIGHT = 0.3    # Weight for recency (newer = higher)
```

### Robot Control
```python
SERIAL_PORT = None           # e.g., "/dev/ttyUSB0" or None for simulation
BAUD_RATE = 9600
```

## Architecture

```
+------------------------------------------------------------------+
|                      WALL-E Memory System                         |
+------------------------------------------------------------------+
|                                                                   |
|  +----------------+  +------------------+  +-------------------+  |
|  |  Core Memory   |  |  Recall Memory   |  |  Archival Memory  |  |
|  |   (SQLite)     |  | (SQLite + FAISS) |  | (SQLite + FAISS)  |  |
|  |                |  |                  |  |                   |  |
|  | - persona      |  | - 40 messages    |  | - Facts/summaries |  |
|  | - human        |  | - FAISS index    |  | - Importance decay|  |
|  | - system       |  | - Semantic search|  | - Categories      |  |
|  +-------+--------+  +--------+---------+  +---------+---------+  |
|          |                    |                      |            |
|          | Always in          | O(log n)             | Decay-     |
|          | context            | search               | weighted   |
|          +--------------------+----------------------+            |
|                               |                                   |
|                    +----------v-----------+                       |
|                    |    Ollama / LLM      |                       |
|                    |    (qwen3:4b)        |                       |
|                    +----------------------+                       |
+-------------------------------------------------------------------+
```

## Performance

### Search Performance (FAISS vs Naive)

| Method | Complexity | 100 msgs | 1000 msgs |
|--------|------------|----------|-----------|
| Naive Cosine | O(n) | ~30ms | ~300ms |
| FAISS | O(log n) | ~5ms | ~15ms |

### Importance Decay Example

```
Fact: "User likes coffee" (importance=8, age=0 days)
  -> effective_importance = 8 * 0.7 + 1.0 * 10 * 0.3 = 8.6

Fact: "User liked tea" (importance=9, age=60 days)
  -> recency_score = exp(-60/30) = 0.135
  -> effective_importance = 9 * 0.7 + 0.135 * 10 * 0.3 = 6.7

Result: Recent "coffee" fact ranks higher than older "tea" fact
```

## Project Structure

```
memory/
├── walle_enhanced.py      # Main cognitive loop
├── config.py              # Configuration settings
├── memory_system.py       # 3-tier memory + FAISS
├── memory_tools.py        # LLM memory tools
├── memory_inspector.py    # Debug/inspect memory
├── context_manager.py     # Sensor simulation
├── robot_tools.py         # Hardware control
├── knowledge_tools.py     # Internet search
├── personality_system.py  # Personality traits
├── heartbeat.py           # Multi-step reasoning
├── test_memory_system.py  # Comprehensive test suite
└── requirements.txt       # Dependencies
```

## Memory Inspector

View stored memories:

```bash
python memory_inspector.py
```

## Testing

### Run Automated Tests

```bash
# Run with pytest (verbose)
pytest test_memory_system.py -v

# Or run directly
python test_memory_system.py
```

### Test Coverage

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestCoreMemory` | 9 | Blocks, append, replace, limits, persistence |
| `TestRecallMemory` | 6 | Insert, semantic search, FAISS integration |
| `TestArchivalMemory` | 6 | Facts, categories, importance decay |
| `TestFAISSManager` | 4 | Add, search, save/load, performance |
| `TestToolCalling` | 5 | Memory tools, schema format |
| `TestSearchPerformance` | 1 | FAISS vs naive benchmark |
| `TestIntegration` | 2 | Full workflow, persistence |

### Manual Tests

1. **Memory Write Test**:
   - Start: `python walle_enhanced.py`
   - Say: "My name is Alex. I'm a Python developer."
   - Exit and run: `python memory_inspector.py`
   - Check: `<human>` block should contain your info

2. **Tool Use Test**:
   - Ask: "What's the weather in Tokyo?"
   - Should trigger: `consult_internet_for_facts`

## Rollback Options

Disable features without code changes:

```python
# In config.py:

# Disable FAISS (use naive cosine similarity)
USE_FAISS = False

# Disable importance decay
IMPORTANCE_RECENCY_WEIGHT = 0.0

# Change model
OLLAMA_MODEL = "llama3.2:3b"  # Or any other Ollama model
```

## Requirements

- Python 3.10+
- Ollama with Qwen3-4B model
- ~4-5GB RAM for Qwen3-4B
- Optional: Arduino/Serial robot for hardware control

## Acknowledgments

- **MemGPT** - Inspiration for memory architecture
- **FAISS** - Fast similarity search
- **Ollama** - Easy local LLM deployment
- **Qwen Team** - Qwen3 models with excellent tool calling

## License

This project is open source. See LICENSE for details.

---

**WALL-E v3.1** - Memory-augmented AI agent with FAISS search and importance decay.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

# Absolute path to the memory/ directory — used for all data files so that
# code works regardless of the process's current working directory.
MEMORY_DIR: str = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT: str = os.path.dirname(MEMORY_DIR)

@dataclass
class Config:
    # --- Run Mode ---
    # "debug" = verbose output (inner thoughts, TTFT, latency, token speed, tool calls)
    # "test"  = clean output only (input/output for TTS pipeline)
    RUN_MODE: str = "debug"

    # --- Ollama Settings (Recommended Backend) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"  # Best tool calling among small models

    # --- Embedding Model (Lightweight for Jetson) ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # 80MB model
    EMBEDDING_DEVICE: str = "cuda"  # Use GPU for embeddings, falls back to CPU
    EMBEDDING_BATCH_SIZE: int = 8

    # --- Memory Settings ---
    MAX_CONTEXT_MESSAGES: int = 10  # Rolling context window
    USE_SEMANTIC_SEARCH: bool = True  # Enable with lightweight embeddings
    RECALL_MEMORY_LIMIT: int = 40     # Compress to archival after this limit

    # --- FAISS Settings (Fast Vector Search) ---
    USE_FAISS: bool = True                       # Enable FAISS for O(log n) search
    FAISS_INDEX_PATH: str = "walle_faiss.index"  # Persistent index file
    FAISS_DIMENSION: int = 384                   # all-MiniLM-L6-v2 output dimension
    FAISS_REBUILD_THRESHOLD: int = 100           # Rebuild index after N insertions

    # --- Importance Decay Settings ---
    IMPORTANCE_DECAY_HALF_LIFE: float = 30.0     # Days until recency score halves
    IMPORTANCE_STATIC_WEIGHT: float = 0.7        # Weight for static importance (0-1)
    IMPORTANCE_RECENCY_WEIGHT: float = 0.3       # Weight for recency score (0-1)

    # --- Search Settings ---
    MAX_SEARCH_RESULTS: int = 5
    SEARCH_REGIONS: List[str] = field(default_factory=lambda: ["wt-wt", "us-en"])
    SEARCH_FALLBACK_LEXICON_PATH: str = field(
        default_factory=lambda: os.path.join(MEMORY_DIR, "search_fallback_lexicon.json")
    )
    SEARCH_FALLBACK_EXACT_MATCH_BOOST: float = 5.0
    SEARCH_FALLBACK_OVERLAP_WEIGHT: float = 2.0
    SEARCH_FALLBACK_COVERAGE_WEIGHT: float = 1.0

    # --- Vision Thresholds ---
    VISION_FACE_DETECTION_THRESHOLD: float = 0.5   # Min score from face detector
    VISION_FACE_RECOGNITION_MIN: float = 0.8       # Min detection score to attempt recognition
    VISION_FACE_MATCH_THRESHOLD: float = 0.5       # Cosine similarity for Coral backend
    VISION_CPU_FACE_MATCH_THRESHOLD: float = 0.6   # Cosine similarity for CPU backend

    # --- STT / Voice Thresholds ---
    INTENT_MATCH_THRESHOLD: float = 0.65           # Semantic intent recognition
    ECHO_OVERLAP_THRESHOLD: float = 0.4            # Echo suppression word overlap
    ECHO_SUPPRESS_WINDOW: float = 8.0              # Echo suppression window (seconds)
    ECHO_COOLDOWN: float = 2.0                     # Post-TTS silence (seconds)

    # --- Robot Settings ---
    SERIAL_PORT: str = None
    BAUD_RATE: int = 115200

    def validate(self) -> bool:
        """Validates Ollama configuration."""
        log = logging.getLogger("walle.config")
        try:
            import requests
            res = requests.get(self.OLLAMA_BASE_URL, timeout=5)
            if res.status_code != 200:
                log.warning("Ollama server not responding at %s", self.OLLAMA_BASE_URL)
                return False

            res = requests.get(f"{self.OLLAMA_BASE_URL}/api/tags")
            models = [m['name'] for m in res.json().get('models', [])]
            if self.OLLAMA_MODEL not in models and f"{self.OLLAMA_MODEL}:latest" not in models:
                log.warning("Model '%s' not found. Available: %s", self.OLLAMA_MODEL, models)
                log.warning("Run: ollama pull %s", self.OLLAMA_MODEL)
                return False

            log.info("Ollama validation passed - %s", self.OLLAMA_MODEL)
            return True
        except Exception as e:
            log.warning("Ollama validation failed: %s", e)
            log.warning("Make sure Ollama is running: ollama serve")
            return False

    @classmethod
    def load(cls):
        return cls()

conf = Config.load()


# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------
def setup_logging(run_mode: str = None) -> None:
    """Configure the root 'walle' logger.

    Call once at startup (in main()).  All modules should obtain their own
    child logger via ``logging.getLogger("walle.<module>")``.

    - "debug" mode  -> DEBUG to console + file
    - "test"  mode  -> INFO  to console, DEBUG to file
    """
    mode = run_mode or conf.RUN_MODE
    root_logger = logging.getLogger("walle")

    # Avoid adding duplicate handlers if called twice
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.DEBUG)

    # --- Console handler ---
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if mode == "debug" else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console)

    # --- File handler (always DEBUG) ---
    log_path = os.path.join(_PROJECT_ROOT, "walle.log")
    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root_logger.addHandler(fh)
    except OSError:
        root_logger.warning("Could not open log file %s", log_path)


# Convenience: call at import time so early log calls don't vanish.
# main() can call setup_logging() again to reconfigure after arg parsing.
setup_logging()

# Module-level logger for this file
_log = logging.getLogger("walle.config")


def debug_print(*args, **kwargs):
    """Legacy helper kept for backward-compat. Prefer logger.debug()."""
    if conf.RUN_MODE == "debug":
        print(*args, **kwargs)

import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields
from typing import List, Optional

# Project root (two levels up from walle/memory/)
_PROJECT_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# Data directory for runtime files (databases, indices, personality)
MEMORY_DIR: str = os.path.join(_PROJECT_ROOT, "data")


def _default_vision_embedder_edge_tpu() -> bool:
    machine = platform.machine().lower()
    return not (platform.system() == "Linux" and machine in {"aarch64", "arm64"})


@dataclass
class Config:
    # --- Run Mode ---
    # "debug" = verbose output (inner thoughts, TTFT, latency, token speed, tool calls)
    # "test"  = clean output only (input/output for TTS pipeline)
    RUN_MODE: str = "debug"

    # --- Ollama Settings (Recommended Backend) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma4:31b-cloud"  # Best tool calling among small models

    # --- Embedding Model (Lightweight for Jetson) ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # 80MB model
    EMBEDDING_DEVICE: str = "cuda"  # Use GPU for embeddings, falls back to CPU
    EMBEDDING_BATCH_SIZE: int = 8

    # --- Memory Settings ---
    MAX_CONTEXT_MESSAGES: int = 10  # Rolling context window
    USE_SEMANTIC_SEARCH: bool = True  # Enable with lightweight embeddings
    RECALL_MEMORY_LIMIT: int = 40  # Compress to archival after this limit

    # --- FAISS Settings (Fast Vector Search) ---
    USE_FAISS: bool = True  # Enable FAISS for O(log n) search
    FAISS_INDEX_PATH: str = "walle_faiss.index"  # Persistent index file
    FAISS_DIMENSION: int = 384  # all-MiniLM-L6-v2 output dimension
    FAISS_REBUILD_THRESHOLD: int = 100  # Rebuild index after N insertions

    # --- Importance Decay Settings ---
    IMPORTANCE_DECAY_HALF_LIFE: float = 30.0  # Days until recency score halves
    IMPORTANCE_STATIC_WEIGHT: float = 0.7  # Weight for static importance (0-1)
    IMPORTANCE_RECENCY_WEIGHT: float = 0.3  # Weight for recency score (0-1)

    # --- Search Settings ---
    MAX_SEARCH_RESULTS: int = 5
    SEARCH_REGIONS: List[str] = field(default_factory=lambda: ["wt-wt", "us-en"])
    # --- Vision Thresholds ---
    VISION_FACE_DETECTION_THRESHOLD: float = 0.5  # Min score from face detector
    VISION_FACE_RECOGNITION_MIN: float = (
        0.8  # Min detection score to attempt recognition
    )
    VISION_FACE_MATCH_THRESHOLD: float = 0.5  # Cosine similarity for Coral backend
    VISION_CPU_FACE_MATCH_THRESHOLD: float = 0.6  # Cosine similarity for CPU backend
    VISION_DETECTOR_EDGE_TPU: bool = True
    VISION_EMBEDDER_EDGE_TPU: bool = _default_vision_embedder_edge_tpu()

    # --- STT / Voice Thresholds ---
    INTENT_MATCH_THRESHOLD: float = 0.65  # Semantic intent recognition
    ECHO_OVERLAP_THRESHOLD: float = 0.4  # Echo suppression word overlap
    ECHO_SUPPRESS_WINDOW: float = 8.0  # Echo suppression window (seconds)
    ECHO_COOLDOWN: float = 2.0  # Post-TTS silence (seconds)

    # --- Vision Processing ---
    VISION_FPS: int = 2  # Camera processing rate

    # --- Robot Settings ---
    SERIAL_PORT: str = None
    BAUD_RATE: int = 115200

    @classmethod
    def load(cls) -> "Config":
        """Create Config with defaults, then override from WALLE_* env vars."""
        instance = cls()
        for f in fields(instance):
            env_key = f"WALLE_{f.name}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                try:
                    origin = f.type
                    if origin is bool or origin == "bool":
                        parsed = env_val.lower() in ("1", "true", "yes")
                    elif origin is int or origin == "int":
                        parsed = int(env_val)
                    elif origin is float or origin == "float":
                        parsed = float(env_val)
                    else:
                        parsed = env_val
                    object.__setattr__(instance, f.name, parsed)
                except (ValueError, TypeError):
                    pass  # Keep default if env value can't be cast
        return instance


def validate_ollama(config: "Config") -> tuple[bool, Optional[subprocess.Popen]]:
    """Ensure Ollama is running and the configured model is available.

    1. Try to connect.  If unreachable, start ``ollama serve`` and retry.
    2. If the model is missing, pull it automatically.

    Returns (success, process) — *process* is non-None only if we spawned
    ``ollama serve`` ourselves; the caller is responsible for terminating it.
    """
    import requests

    log = logging.getLogger("walle.config")
    process = None

    def _is_reachable() -> bool:
        try:
            return requests.get(config.OLLAMA_BASE_URL, timeout=3).ok
        except Exception:
            return False

    def _has_model() -> bool:
        try:
            models = [
                m["name"]
                for m in requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
                .json()
                .get("models", [])
            ]
            return (
                config.OLLAMA_MODEL in models
                or f"{config.OLLAMA_MODEL}:latest" in models
            )
        except Exception:
            return False

    # -- 1. Ensure server is running --
    if not _is_reachable():
        binary = shutil.which("ollama")
        if not binary:
            log.error("Ollama binary not found in PATH")
            return False, None

        log.info("Starting ollama serve...")
        process = subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for _ in range(10):
            time.sleep(1)
            if _is_reachable():
                break
        else:
            process.terminate()
            log.error("Ollama did not start within 10 seconds")
            return False, None

    # -- 2. Ensure model is available --
    if not _has_model():
        log.info("Pulling model '%s'...", config.OLLAMA_MODEL)
        try:
            subprocess.run(
                [shutil.which("ollama") or "ollama", "pull", config.OLLAMA_MODEL],
                check=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.error("Failed to pull model '%s': %s", config.OLLAMA_MODEL, e)
            if process:
                process.terminate()
            return False, None

    log.info("Ollama ready — %s", config.OLLAMA_MODEL)
    return True, process


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
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(fh)
    except OSError:
        root_logger.warning("Could not open log file %s", log_path)


# Module-level logger for this file
_log = logging.getLogger("walle.config")

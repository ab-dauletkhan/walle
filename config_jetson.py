"""
Jetson Orin Nano configuration overrides for WALL-E.
Import this BEFORE initializing subsystems to apply Jetson-specific settings.

Usage:
    import config_jetson  # applies overrides to conf
    from config import conf  # now has Jetson values
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory"))

from config import conf

# ---------- Jetson mode flag ----------
JETSON_MODE = True

# ---------- LLM ----------
# qwen3:4b recommended for better tool calling on Jetson
conf.OLLAMA_MODEL = "qwen3:4b"

# ---------- Embedding ----------
# Keep GPU free for LLM — embeddings on CPU
conf.EMBEDDING_DEVICE = "cpu"
conf.EMBEDDING_BATCH_SIZE = 4  # lower for 8GB RAM constraint

# ---------- Memory ----------
conf.USE_FAISS = True  # CPU-based, very efficient
conf.USE_SEMANTIC_SEARCH = True

# ---------- Vision ----------
# Reduced FPS to save CPU cycles on Jetson
VISION_FPS = 1

# ---------- Summary ----------
import logging as _logging
_logging.getLogger("walle.config_jetson").info(
    "Model=%s, Embedding=%s, FAISS=%s, VisionFPS=%s",
    conf.OLLAMA_MODEL, conf.EMBEDDING_DEVICE, conf.USE_FAISS, VISION_FPS
)

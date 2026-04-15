"""Coral face recognition utilities for WALL-E."""

from pathlib import Path

FACE_DIR = Path(__file__).resolve().parent
MODELS_DIR = FACE_DIR / "models"
DEFAULT_DATA_DIR = FACE_DIR / "scanned_people"
COCO_LABELS_PATH = FACE_DIR / "coco_labels.txt"
PEOPLE_LABELS_PATH = FACE_DIR / "people_labels.txt"

__all__ = [
    "DEFAULT_DATA_DIR",
    "FACE_DIR",
    "MODELS_DIR",
    "PEOPLE_LABELS_PATH",
    "COCO_LABELS_PATH",
]

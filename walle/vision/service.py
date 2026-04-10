"""
WALL-E Vision Service
Background thread that processes camera frames and feeds VisualContext into ContextManager.
Supports two backends: Google Coral TPU (preferred) and CPU (YOLOv8 + InsightFace fallback).
"""
import logging
import os
import time
import threading
import tempfile
import base64
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np

_log = logging.getLogger("walle.vision")

_BASE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_BASE))  # walle/ → project root

from walle.memory.config import conf
from walle.memory.context_manager import ContextManager, VisualContext
from vision.face_recognition.common import (
    DEFAULT_DATA_DIR,
    PEOPLE_LABELS_PATH,
    create_detection_interpreter,
    create_embedding_interpreter,
    detect_faces,
    input_size,
    load_labels,
    load_people_embeddings,
    recognize_detection,
    require_pil_image,
)


# ---------------------------------------------------------------------------
# Vision Backend Interface
# ---------------------------------------------------------------------------
class VisionBackend(ABC):
    """Abstract backend for face detection and recognition."""

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Process a BGR frame, return list of face dicts: [{name, confidence, location}]."""
        ...

    @abstractmethod
    def close(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Google Coral TPU Backend
# ---------------------------------------------------------------------------
class CoralVisionBackend(VisionBackend):
    """Face detection + recognition using Google Coral Edge TPU."""

    def __init__(self):
        self._Image = require_pil_image()

        self._det_interpreter = create_detection_interpreter(edge_tpu=True)
        self._input_w, self._input_h = input_size(self._det_interpreter)

        self._emb_interpreter = create_embedding_interpreter(edge_tpu=True)
        self._people_labels = load_labels(PEOPLE_LABELS_PATH)
        self._people_embeddings = load_people_embeddings(DEFAULT_DATA_DIR, required=False)
        if not self._people_embeddings:
            _log.warning(
                "No Coral face embeddings found at %s; detected faces will be Unknown",
                DEFAULT_DATA_DIR,
            )

        _log.info("Coral TPU backend initialized")

    def _detect_faces(self, image_pil):
        """Run face detection, return list of (ymin, xmin, ymax, xmax, score) in pixel coords."""
        resized = image_pil.convert("RGB").resize((self._input_w, self._input_h), self._Image.LANCZOS)
        w, h = image_pil.size
        return detect_faces(
            self._det_interpreter,
            resized,
            conf.VISION_FACE_DETECTION_THRESHOLD,
            w,
            h,
        )

    def _recognize_face(self, image_rgb_array, detection):
        """Crop face, compute embedding, match against known people."""
        match = recognize_detection(
            self._emb_interpreter,
            image_rgb_array,
            detection,
            self._people_labels,
            self._people_embeddings,
            conf.VISION_FACE_MATCH_THRESHOLD,
        )
        return match.name, match.confidence

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        from PIL import Image
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        detections = self._detect_faces(pil_image)
        rgb_array = np.array(pil_image)

        faces = []
        for detection in detections:
            if detection.score < conf.VISION_FACE_RECOGNITION_MIN:
                continue
            name, confidence = self._recognize_face(rgb_array, detection)
            faces.append({
                "name": name or "Unknown",
                "confidence": round(confidence if name else detection.score, 2),
                "location": f"({detection.xmin},{detection.ymin})-({detection.xmax},{detection.ymax})",
            })
        return faces

    def close(self):
        pass


# ---------------------------------------------------------------------------
# CPU Backend (YOLOv8 + InsightFace)
# ---------------------------------------------------------------------------
class CPUVisionBackend(VisionBackend):
    """Face detection + recognition using YOLOv8 and InsightFace ArcFace on CPU."""

    def __init__(self):
        cv_dir = os.path.join(_PROJECT_ROOT, "models", "comp_vision")

        from ultralytics import YOLO
        import onnxruntime as ort

        # Face detector
        model_path = os.path.join(cv_dir, "yolov8n-face.pt")
        self._yolo = YOLO(model_path)

        # ArcFace embedder (CPU only to save GPU for LLM)
        onnx_path = os.path.join(cv_dir, "models", "w600k_r50.onnx")
        if os.path.exists(onnx_path):
            self._emb_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self._emb_input_name = self._emb_session.get_inputs()[0].name
        else:
            self._emb_session = None
            _log.warning("ArcFace model not found, face recognition disabled")

        # Simple in-memory face database (name -> list of embeddings)
        self._known_faces: Dict[str, List[np.ndarray]] = {}
        self._load_known_faces(cv_dir)

        _log.info("CPU backend initialized (YOLOv8 + InsightFace)")

    def _load_known_faces(self, cv_dir):
        """Load any pre-computed face embeddings from disk."""
        db_dir = os.path.join(cv_dir, "face_db")
        if not os.path.isdir(db_dir):
            return
        for name_dir in os.listdir(db_dir):
            emb_path = os.path.join(db_dir, name_dir)
            if not os.path.isdir(emb_path):
                continue
            embeddings = []
            for f in os.listdir(emb_path):
                if f.endswith(".npy"):
                    embeddings.append(np.load(os.path.join(emb_path, f)))
            if embeddings:
                self._known_faces[name_dir] = embeddings

    def _get_embedding(self, face_rgb_112: np.ndarray) -> Optional[np.ndarray]:
        if self._emb_session is None:
            return None
        img = face_rgb_112.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # NCHW
        emb = self._emb_session.run(None, {self._emb_input_name: img})[0][0]
        norm = np.linalg.norm(emb)
        return emb / max(norm, 1e-12)

    def _match_face(self, embedding: np.ndarray) -> tuple:
        """Match embedding against known faces. Returns (name, similarity)."""
        best_name = None
        best_sim = 0.0
        for name, refs in self._known_faces.items():
            sims = [float(np.dot(embedding, ref)) for ref in refs]
            avg_sim = np.mean(sims) if sims else 0.0
            if avg_sim > best_sim:
                best_sim = avg_sim
                best_name = name
        if best_sim < conf.VISION_CPU_FACE_MATCH_THRESHOLD:
            return None, best_sim
        return best_name, best_sim

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        import cv2

        results = self._yolo.predict(source=frame, device="cpu", verbose=False)
        faces = []

        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy.cpu().numpy()[0].astype(int)
                x1, y1, x2, y2 = xyxy
                det_conf = float(box.conf.cpu().numpy()[0])

                if det_conf < conf.VISION_FACE_DETECTION_THRESHOLD:
                    continue

                # Crop and recognize
                name = None
                similarity = det_conf
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0 and self._emb_session is not None:
                    crop_rgb = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2RGB)
                    emb = self._get_embedding(crop_rgb)
                    if emb is not None:
                        name, similarity = self._match_face(emb)

                faces.append({
                    "name": name or "Unknown",
                    "confidence": round(similarity, 2),
                    "location": f"({x1},{y1})-({x2},{y2})",
                })

        return faces

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Vision Backend Registry (Strategy + Registry pattern)
# ---------------------------------------------------------------------------
class VisionBackendRegistry:
    """Registry of backend factories, tried in order until one succeeds."""

    def __init__(self):
        self._factories: list[tuple[str, type]] = []

    def register(self, name: str, factory) -> None:
        self._factories.append((name, factory))

    def create_first_available(self) -> Optional[VisionBackend]:
        for name, factory in self._factories:
            try:
                return factory()
            except Exception as e:
                _log.warning("Backend '%s' unavailable: %s", name, e)
        _log.warning("No vision backend available — running without face recognition")
        return None


def build_default_vision_registry() -> VisionBackendRegistry:
    """Default priority: Coral TPU → CPU (YOLO + InsightFace)."""
    registry = VisionBackendRegistry()
    registry.register("coral", CoralVisionBackend)
    registry.register("cpu", CPUVisionBackend)
    return registry


# ---------------------------------------------------------------------------
# VisionService — background thread feeding ContextManager
# ---------------------------------------------------------------------------
class VisionService:
    """Continuously processes camera frames and updates ContextManager with visual context."""

    def __init__(self, context_manager: ContextManager, camera_index: int = 0, fps: int = 2):
        self._context_manager = context_manager
        self._camera_index = camera_index
        self._target_interval = 1.0 / max(fps, 1)

        self._backend: Optional[VisionBackend] = None
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        # Auto-detect backend
        self._backend = self._create_backend()

    @property
    def is_active(self) -> bool:
        return self._backend is not None and self._running and self._cap is not None

    @property
    def backend_name(self) -> str:
        if self._backend is None:
            return "none"
        return self._backend.__class__.__name__.replace("VisionBackend", "").lower()

    def _create_backend(self) -> Optional[VisionBackend]:
        """Use the default backend registry to find the first available backend."""
        return build_default_vision_registry().create_first_available()

    def start(self) -> None:
        """Start the background vision processing thread."""
        if self._backend is None:
            _log.warning("No backend available, skipping start")
            return

        if self._camera_index < 0:
            _log.warning("Camera disabled (index < 0)")
            return

        import cv2
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            _log.warning("Could not open camera %s, continuing without vision", self._camera_index)
            self._cap.release()
            self._cap = None
            return

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vision-service")
        self._thread.start()
        _log.info("Started (camera=%s, target_fps=%s)", self._camera_index, int(1 / self._target_interval))

    def stop(self) -> None:
        """Stop the background thread and release camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._backend is not None:
            self._backend.close()
        _log.info("Stopped")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Thread-safe access to the most recent camera frame."""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    # Error backoff parameters
    _MAX_CONSECUTIVE_ERRORS = 5
    _BACKOFF_BASE = 0.5        # seconds, doubles each consecutive error
    _BACKOFF_MAX = 30.0        # cap

    def _loop(self) -> None:
        """Background loop: capture frame -> detect faces -> update context."""
        consecutive_errors = 0

        while self._running:
            loop_start = time.monotonic()

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Thread-safe write (matches locked read in get_latest_frame)
            with self._frame_lock:
                self._latest_frame = frame

            try:
                faces = self._backend.process_frame(frame)
                visual = VisualContext(
                    faces_detected=faces,
                    timestamp=datetime.now(),
                )
                self._context_manager.update_visual(visual)
                consecutive_errors = 0  # reset on success
            except Exception as e:
                consecutive_errors += 1
                backoff = min(
                    self._BACKOFF_BASE * (2 ** (consecutive_errors - 1)),
                    self._BACKOFF_MAX,
                )
                if consecutive_errors <= self._MAX_CONSECUTIVE_ERRORS:
                    _log.warning("Frame processing error (%s): %s", consecutive_errors, e)
                elif consecutive_errors == self._MAX_CONSECUTIVE_ERRORS + 1:
                    _log.error("Repeated failures (%sx), throttling to %.1fs intervals. Suppressing further logs.", consecutive_errors, backoff)
                time.sleep(backoff)
                continue

            # Rate-limit + frame drop detection
            elapsed = time.monotonic() - loop_start
            if elapsed > self._target_interval * 2:
                _log.debug("Frame processing took %.0fms (target %.0fms) — frame drop likely",
                           elapsed * 1000, self._target_interval * 1000)
            sleep_time = self._target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# capture_image tool — lets the LLM request an image description on demand
# ---------------------------------------------------------------------------
def get_capture_image_tools() -> list:
    """Tool schema for the capture_image tool."""
    return [
        {
            "type": "function",
            "function": {
                "name": "capture_image",
                "description": "Capture a photo from the robot's camera and get a description of what is visible. Use when the user asks what you can see or you need visual information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "detail": {
                            "type": "string",
                            "description": "What to focus on: 'general' for overall scene, 'faces' for people, 'objects' for items",
                            "enum": ["general", "faces", "objects"],
                        },
                    },
                    "required": [],
                },
            },
        }
    ]


class CaptureImageExecutor:
    """Handles the capture_image tool call: grabs a frame and describes it via a vision LLM."""

    def __init__(self, vision_service: VisionService, ollama_base_url: str, vision_model: str = "moondream"):
        self._vision_service = vision_service
        self._ollama_url = ollama_base_url
        self._vision_model = vision_model

    def execute(self, fn_name: str, args: dict) -> str:
        if fn_name != "capture_image":
            return f"Unknown tool: {fn_name}"

        frame = self._vision_service.get_latest_frame()
        if frame is None:
            return "No camera available — cannot capture image."

        detail = args.get("detail", "general")
        prompts = {
            "general": "Describe what you see in this image in 1-2 sentences.",
            "faces": "Describe the people visible in this image. How many? What are they doing?",
            "objects": "List the main objects visible in this image.",
        }
        prompt = prompts.get(detail, prompts["general"])

        try:
            return self._describe_frame(frame, prompt)
        except Exception as e:
            return f"Image capture failed: {e}"

    def _describe_frame(self, frame: np.ndarray, prompt: str) -> str:
        """Send frame to Ollama vision model for description."""
        import cv2
        import requests

        # Encode frame as JPEG
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_image = base64.b64encode(buf.tobytes()).decode("utf-8")

        # Call Ollama generate API with image
        resp = requests.post(
            f"{self._ollama_url}/api/generate",
            json={
                "model": self._vision_model,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("response", "Could not describe the image.")

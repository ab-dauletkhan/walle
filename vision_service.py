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

from memory.config import conf
from memory.context_manager import ContextManager, VisualContext


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
        from tflite_runtime.interpreter import Interpreter, load_delegate
        from PIL import Image

        self._Image = Image
        coral_dir = os.path.join(_BASE, "walle_vision", "face_recognition_coral")

        # Face detection model (SSD MobileNet v2)
        self._det_interpreter = Interpreter(
            model_path=os.path.join(coral_dir, "models", "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"),
            experimental_delegates=[load_delegate("libedgetpu.so.1.0")],
        )
        self._det_interpreter.allocate_tensors()
        _, self._input_h, self._input_w, _ = self._det_interpreter.get_input_details()[0]["shape"]

        # Face embedding model
        self._emb_interpreter = Interpreter(
            model_path=os.path.join(coral_dir, "models", "Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite"),
            experimental_delegates=[load_delegate("libedgetpu.so.1.0")],
        )
        self._emb_interpreter.allocate_tensors()

        # Load people labels and embeddings
        self._people_labels = self._load_labels(os.path.join(coral_dir, "people_labels.txt"))
        self._scanned_dir = os.path.join(coral_dir, "scanned_people")
        self._people_embeddings = self._load_people_embeddings()

        _log.info("Coral TPU backend initialized")

    def _load_labels(self, path):
        import re
        labels = {}
        if not os.path.exists(path):
            return labels
        with open(path, "r", encoding="utf-8") as f:
            for row, line in enumerate(f):
                pair = re.split(r"[:\s]+", line.strip(), maxsplit=1)
                if len(pair) == 2 and pair[0].strip().isdigit():
                    labels[int(pair[0])] = pair[1].strip()
                else:
                    labels[row] = pair[0].strip()
        return labels

    def _load_people_embeddings(self):
        """Pre-load embeddings for each scanned person."""
        people = {}
        if not os.path.isdir(self._scanned_dir):
            return people
        for folder in sorted(os.listdir(self._scanned_dir)):
            emb_dir = os.path.join(self._scanned_dir, folder, "embeddings")
            if not os.path.isdir(emb_dir):
                continue
            embeddings = []
            for f in sorted(os.listdir(emb_dir))[:20]:
                if f.endswith(".npy"):
                    embeddings.append(np.load(os.path.join(emb_dir, f)))
            if embeddings:
                people[folder] = embeddings
        return people

    def _detect_faces(self, image_pil):
        """Run face detection, return list of (ymin, xmin, ymax, xmax, score) in pixel coords."""
        resized = image_pil.convert("RGB").resize((self._input_w, self._input_h), self._Image.LANCZOS)
        # Set input tensor
        input_details = self._det_interpreter.get_input_details()[0]
        self._det_interpreter.tensor(input_details["index"])()[0][:, :] = resized
        self._det_interpreter.invoke()

        # Parse outputs
        boxes = np.squeeze(self._det_interpreter.get_tensor(self._det_interpreter.get_output_details()[0]["index"]))
        classes = np.squeeze(self._det_interpreter.get_tensor(self._det_interpreter.get_output_details()[1]["index"]))
        scores = np.squeeze(self._det_interpreter.get_tensor(self._det_interpreter.get_output_details()[2]["index"]))
        count = int(np.squeeze(self._det_interpreter.get_tensor(self._det_interpreter.get_output_details()[3]["index"])))

        w, h = image_pil.size
        results = []
        for i in range(min(count, len(scores))):
            if scores[i] < conf.VISION_FACE_DETECTION_THRESHOLD:
                continue
            ymin = int(max(0, boxes[i][0] * h))
            xmin = int(max(0, boxes[i][1] * w))
            ymax = int(min(h, boxes[i][2] * h))
            xmax = int(min(w, boxes[i][3] * w))
            results.append((ymin, xmin, ymax, xmax, float(scores[i])))
        return results

    def _recognize_face(self, image_rgb_array, ymin, xmin, ymax, xmax):
        """Crop face, compute embedding, match against known people."""
        import cv2
        crop = image_rgb_array[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            return None, 0.0
        crop = cv2.resize(crop, (96, 96)).astype("uint8")
        crop_input = crop.reshape(1, 96, 96, 3) / 255.0

        # Compute embedding
        input_details = self._emb_interpreter.get_input_details()[0]
        scale, zero_point = input_details["quantization"]
        self._emb_interpreter.tensor(input_details["index"])()[0][:, :] = np.clip(crop_input / scale + zero_point, 0, 255).astype(np.uint8)
        self._emb_interpreter.invoke()
        output_details = self._emb_interpreter.get_output_details()[0]
        emb = self._emb_interpreter.get_tensor(output_details["index"])
        o_scale, o_zp = output_details["quantization"]
        emb = o_scale * (emb - o_zp)

        # Match against known people using cosine similarity
        best_name = None
        best_sim = -1.0
        emb_flat = emb.flatten()
        emb_norm = np.linalg.norm(emb_flat)
        if emb_norm < 1e-12:
            return None, 0.0
        emb_unit = emb_flat / emb_norm

        for folder, embeddings in self._people_embeddings.items():
            sims = []
            for ref in embeddings[:20]:
                ref_flat = ref.flatten()
                ref_norm = np.linalg.norm(ref_flat)
                if ref_norm < 1e-12:
                    continue
                sims.append(float(np.dot(emb_unit, ref_flat / ref_norm)))
            if not sims:
                continue
            avg_sim = np.mean(sims)
            if avg_sim > conf.VISION_FACE_MATCH_THRESHOLD and avg_sim > best_sim:
                best_sim = avg_sim
                try:
                    idx = int(folder)
                    best_name = self._people_labels.get(idx, folder)
                except ValueError:
                    best_name = folder

        confidence = max(0.0, best_sim) if best_name else 0.0
        return best_name, confidence

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        from PIL import Image
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        detections = self._detect_faces(pil_image)
        rgb_array = np.array(pil_image)

        faces = []
        for ymin, xmin, ymax, xmax, score in detections:
            if score < conf.VISION_FACE_RECOGNITION_MIN:
                continue
            name, confidence = self._recognize_face(rgb_array, ymin, xmin, ymax, xmax)
            faces.append({
                "name": name or "Unknown",
                "confidence": round(confidence if name else score, 2),
                "location": f"({xmin},{ymin})-({xmax},{ymax})",
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
        cv_dir = os.path.join(_BASE, "comp_vision_diplomka")

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

    def _create_backend(self) -> Optional[VisionBackend]:
        """Try Coral first, fall back to CPU."""
        try:
            return CoralVisionBackend()
        except Exception as e:
            _log.warning("Coral not available (%s), trying CPU backend...", e)

        try:
            return CPUVisionBackend()
        except Exception as e:
            _log.warning("CPU backend not available (%s)", e)
            _log.warning("Running without vision — no face recognition")
            return None

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

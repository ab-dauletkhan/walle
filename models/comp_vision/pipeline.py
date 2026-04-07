# pipeline.py
import cv2
import numpy as np
from typing import List, Tuple
import onnxruntime as ort
from numpy.linalg import norm

from ultralytics import YOLO

yolo_face_model = YOLO("yolov8n-face.pt")
# --- Detector Option B: YOLOv8 face (recommended) -----------------
def detect_faces_yolo8(img_bgr) -> List[Tuple[int,int,int,int]]:
    
    results = yolo_face_model.predict(source=img_bgr, device="mps", verbose=False)
    boxes_out = []
    for r in results:
        for b in r.boxes.xyxy.cpu().numpy():
            x1,y1,x2,y2 = b[:4].astype(int)
            boxes_out.append((x1, y1, x2 - x1, y2 - y1))
    return boxes_out

# --- Alignment utilities ------------------------------------------
def crop_and_align(img_bgr, bbox) -> np.ndarray:
    """
    Minimal crop + resize to 112x112.
    For best accuracy, use 5-point landmarks & affine align (InsightFace).
    """
    x,y,w,h = bbox
    crop = img_bgr[max(0,y):y+h, max(0,x):x+w]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (112,112), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

def l2_normalize(v, eps=1e-12):
    return v / max(norm(v), eps)

# --- Embedding model (InsightFace) --------------------------------
class ArcFaceEmbedder:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        # Replace with "CUDAExecutionProvider" if onnxruntime-gpu installed
        input_name = self.session.get_inputs()[0].name
        self.input_name = input_name

    def preprocess(self, img_rgb: np.ndarray) -> np.ndarray:
        # InsightFace expects NCHW float32 normalized
        img = img_rgb.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2,0,1))  # CHW
        return img[np.newaxis, ...]       # NCHW

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        x = self.preprocess(img_rgb)
        emb = self.session.run(None, {self.input_name: x})[0][0]
        return l2_normalize(emb)  # unit length for cosine

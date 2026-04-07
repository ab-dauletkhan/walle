# reid_tracking.py
import time
import cv2
import numpy as np
import torch
import onnxruntime as ort
from typing import Dict, List, Tuple
from ultralytics import YOLO

# ================== Device ==================
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = 0
else:
    DEVICE = "cpu"

# ================== ReID embedder (ONNX) ==================
class ReIDEmbedder:
    """
    Generic person Re-ID embedder (ONNX).
    Expects 256x128 RGB, ImageNet mean/std, NCHW float32.
    Returns L2-normalized feature (float32).
    """
    def __init__(self, onnx_path: str):
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    @staticmethod
    def _letterbox(img, new_h=256, new_w=128):
        h, w = img.shape[:2]
        scale = min(new_w / w, new_h / h)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((new_h, new_w, 3), dtype=img.dtype)
        top = (new_h - nh) // 2
        left = (new_w - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized
        return canvas

    @staticmethod
    def _l2n(v: np.ndarray, eps=1e-12) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / max(n, eps)

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        # rgb = self._letterbox(rgb, 256, 128).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_AREA)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        chw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # NCHW
        return chw

    def __call__(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = self.preprocess(crop_bgr)
        feat = self.sess.run(None, {self.input_name: x})[0][0]
        return self._l2n(feat.astype(np.float32))

# ================== Online gallery (no DB) ==================
class OnlineGallery:
    """
    Keeps global identities in-memory.
    - Each identity stores a prototype embedding (moving average) + count.
    - Matches new embeddings by cosine similarity.
    - If best_sim >= SIM_THRESH => assign to that global_id, else create new one.
    """
    def __init__(self, sim_thresh: float = 0.7, ema: float = 1):
        self.sim_thresh = sim_thresh
        self.ema = ema  # exponential moving average factor
        self._next_gid = 1
        self.protos: Dict[int, np.ndarray] = {}  # gid -> embedding
        self.counts: Dict[int, int] = {}         # gid -> obs count

    @staticmethod
    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def add_or_match(self, emb: np.ndarray) -> Tuple[int, float]:
        if not self.protos:
            gid = self._next_gid; self._next_gid += 1
            self.protos[gid] = emb.copy()
            self.counts[gid] = 1
            return gid, 1.0

        # find best gid by cosine sim
        best_gid, best_sim = -1, -1.0
        for gid, proto in self.protos.items():
            sim = self.cos_sim(emb, proto)
            if sim > best_sim:
                best_sim, best_gid = sim, gid

        if best_sim >= self.sim_thresh:
            # update prototype with EMA
            proto = self.protos[best_gid]
            self.protos[best_gid] = self._ema_update(proto, emb)
            self.counts[best_gid] += 1
            return best_gid, best_sim
        else:
            gid = self._next_gid; self._next_gid += 1
            self.protos[gid] = emb.copy()
            self.counts[gid] = 1
            return gid, best_sim

    def _ema_update(self, proto: np.ndarray, new_emb: np.ndarray) -> np.ndarray:
        out = self.ema * proto + (1.0 - self.ema) * new_emb
        # keep unit norm
        return out / max(np.linalg.norm(out), 1e-12)

# ================== Main loop ==================
def main():
    # 1) YOLO person detector + built-in tracker
    yolo = YOLO("yolov8n.pt")  # class 0 = person

    # 2) ReID embedder and online gallery
    reid = ReIDEmbedder("models/reid.onnx")  # e.g., OSNet exported to ONNX
    gallery = OnlineGallery(sim_thresh=0.97, ema=1.0)

    # 3) Cache per track: track_id -> {"gid": int, "t": float}
    #    We don’t embed every frame; we throttle per track.
    track_cache: Dict[int, Dict[str, float]] = {}
    REEMBED_COOLDOWN = 1.5  # seconds

    stream = yolo.track(
        source=0, stream=True, persist=True, tracker="bytetrack.yaml",
        classes=[0], imgsz=640, device=DEVICE, verbose=False, show=False
    )

    try:
        for res in stream:
            frame = res.orig_img
            if res.boxes is None or res.boxes.xyxy is None:
                cv2.imshow("Online ReID (No DB)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            boxes = res.boxes.xyxy.cpu().numpy()
            ids   = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else []

            now = time.time()
            active_tids = set()

            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box[:4])
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue

                active_tids.add(tid)
                need = (tid not in track_cache) or ((now - track_cache[tid]["t"]) >= REEMBED_COOLDOWN)

                if need:
                    emb = reid(crop)  # full-body embedding (unit vector)
                    gid, sim = gallery.add_or_match(emb)
                    track_cache[tid] = {"gid": gid, "t": now, "sim": sim}
                else:
                    gid = int(track_cache[tid]["gid"])

                # Draw
                color = (0, 200, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {tid} | GID {track_cache[tid]['gid']}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # # Optional: clean cache for disappeared tracks to keep memory small
            # for tid in list(track_cache.keys()):
            #     if tid not in active_tids and (now - track_cache[tid]["t"]) > 10.0:
            #         del track_cache[tid]

            cv2.imshow("Online ReID (No DB)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

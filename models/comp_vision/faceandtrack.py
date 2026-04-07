# multi_track_face.py
import time, cv2, numpy as np, torch, psycopg2
from typing import Dict, List, Tuple
from ultralytics import YOLO

# --- your pipeline pieces ---
from pipeline import detect_faces_yolo8, crop_and_align, ArcFaceEmbedder
# your previous face search fn (pgvector)
from recognize import search_topk, DSN   # or paste search_topk + DSN here

# =============== Device =================
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = 0
else:
    DEVICE = "cpu"

# =============== IoU helpers ===============
def iou_xyxy(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-9)

def to_xyxy_from_xywh(b: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    x, y, w, h = b
    return (x, y, x + w, y + h)

def to_xyxy_from_ultralytics(b: np.ndarray) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = map(int, b[:4])
    return (x1, y1, x2, y2)

def assign_faces_to_persons(
    person_boxes_xyxy: List[Tuple[int,int,int,int]],
    face_boxes_xywh: List[Tuple[int,int,int,int]],
    iou_thresh: float = 0.15
) -> Dict[int, int]:
    """
    Greedy: match each person (index) to best face (index) by IoU if above threshold.
    Returns dict person_index -> face_index
    """
    mapping, used_faces = {}, set()
    for pi, pbox in enumerate(person_boxes_xyxy):
        best_iou, best_fi = 0.0, -1
        for fi, fbox in enumerate(face_boxes_xywh):
            if fi in used_faces:
                continue
            iou = iou_xyxy(pbox, to_xyxy_from_xywh(fbox))
            if iou > best_iou:
                best_iou, best_fi = iou, fi
        if best_fi >= 0 and best_iou >= iou_thresh:
            mapping[pi] = best_fi
            used_faces.add(best_fi)
    return mapping

# =============== Main ===============
def main():
    # 1) Person tracker (ByteTrack via YOLO)
    yolo = YOLO("yolov8n.pt")  # person class=0

    # 2) Face embedder (ArcFace) + DB
    embedder = ArcFaceEmbedder("models/w600k_r50.onnx")
    try:
        conn = psycopg2.connect(DSN)
        db_ok = True
    except Exception as e:
        print("[WARN] DB connect failed:", e)
        conn, db_ok = None, False

    # Per-track NAME cache: tid -> {"name": str, "sim": float, "t": float}
    name_cache: Dict[int, Dict[str, float]] = {}
    NAME_COOLDOWN = 2.0               # seconds between re-queries for one track
    FACE_SIM_USE = 0.60               # cosine similarity threshold to accept best hit
    # (If you used your previous "1 - threshold" scheme, adapt accordingly.)

    stream = yolo.track(
        source=0, stream=True, persist=True, tracker="bytetrack.yaml",
        classes=[0], imgsz=640, device=DEVICE, verbose=False, show=False
    )

    try:
        for res in stream:
            frame = res.orig_img

            # --- 1) PERSONS from YOLO tracker ---
            if res.boxes is None or res.boxes.xyxy is None:
                cv2.imshow("Tracking + Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            person_xyxy = [to_xyxy_from_ultralytics(b) for b in res.boxes.xyxy.cpu().numpy()]
            tids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else []

            # --- 2) FACES on same frame ---
            # For speed, you can detect inside each person crop (optional).
            face_xywh = detect_faces_yolo8(frame)  # returns [(x,y,w,h), ...]

            # --- 3) ASSOCIATE faces -> person tracks (by IoU) ---
            mapping = assign_faces_to_persons(person_xyxy, face_xywh, iou_thresh=0.15)

            now = time.time()
            for i, tid in enumerate(tids):
                x1, y1, x2, y2 = person_xyxy[i]

                # If this person has a face matched, (re)recognize on cooldown
                if i in mapping and db_ok:
                    fx, fy, fw, fh = face_xywh[mapping[i]]

                    do_query = (
                        tid not in name_cache or
                        (now - name_cache[tid]["t"]) >= NAME_COOLDOWN or
                        name_cache[tid]["name"] == "Unknown"
                    )

                    if do_query:
                        face_rgb = crop_and_align(frame, (fx, fy, fw, fh))
                        if face_rgb is not None:
                            emb = embedder(face_rgb).astype(np.float32)
                            # DB nearest neighbors
                            hits = search_topk(conn, emb, k=5)
                            if hits:
                                best = max(hits, key=lambda r: r[3])  # cosine_sim
                                sim  = float(best[3])
                                name = best[2] if sim >= FACE_SIM_USE else "Unknown"
                            else:
                                sim, name = 0.0, "Unknown"
                            name_cache[tid] = {"name": name, "sim": sim, "t": now}

                    # Draw FACE box (green known / red unknown)
                    name = name_cache.get(tid, {}).get("name", "Unknown")
                    face_color = (200, 200, 200) if name == "Unknown" else (0, 200, 0)
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), face_color, 2)
                    cv2.putText(frame, name, (fx, fy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

                # --- 4) Draw PERSON box + ID + optional name ---
                label = f"ID {tid}"
                if tid in name_cache:
                    label += f" | {name_cache[tid]['name']}"
                color = (255, 255, 255)  # dull white
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Tracking + Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if conn is not None:
            conn.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

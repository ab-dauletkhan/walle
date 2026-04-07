import time
import cv2
import numpy as np
import psycopg2
from typing import Dict, Tuple, List

from ultralytics import YOLO
import torch

# --- your modules ---
from pipeline import detect_faces_yolo8, crop_and_align, ArcFaceEmbedder
# reuse your exact query function
def search_topk(conn, emb, k=5):
    # cosine distance: smaller is better. Ensure emb is unit-normalized!
    with conn.cursor() as cur:
        cur.execute("""
        SELECT p.person_id, p.external_id, p.full_name,
               1 - (fe.embedding <#> %s::vector) AS cosine_sim,  -- cosine similarity
               fe.emb_id
        FROM face_embeddings fe
        JOIN persons p USING(person_id)
        ORDER BY fe.embedding <=> %s::vector    -- cosine distance
        LIMIT %s;
        """, (emb.tolist(), emb.tolist(), k))
        return cur.fetchall()


DSN = "postgresql://faceuser:facepass@localhost:5432/facedb"
EMB_ONNX = "models/w600k_r50.onnx"

# ==========================
# Utility: IoU + assignment
# ==========================
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
    iou_thresh: float = 0.2
) -> Dict[int, int]:
    """
    Returns mapping: person_index -> face_index
    Greedy by max IoU above threshold.
    """
    mapping = {}
    used_faces = set()
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

# ==========================
# Device selection
# ==========================
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = 0
else:
    DEVICE = "cpu"

def main():
    # 1) YOLO person model with built-in tracker (ByteTrack)
    person_model = YOLO("yolov8n.pt")  # small & fast; class 0 = person

    # 2) Face pipeline pieces (load once)
    embedder = ArcFaceEmbedder(EMB_ONNX)
    conn = psycopg2.connect(DSN)

    # 3) Per-track cache to throttle recognition
    #    track_id -> {"name": str, "sim": float, "t": float}
    cache: Dict[int, Dict[str, float]] = {}
    RECOG_COOLDOWN = 2.0   # seconds

    # 4) Start tracking stream (webcam source=0)
    results_stream = person_model.track(
        source=0,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],           # only 'person'
        imgsz=640,
        device=DEVICE,
        verbose=False,
        show=False             # we'll draw ourselves
    )

    try:
        for result in results_stream:
            frame = result.orig_img
            # Person boxes + IDs from YOLO tracker
            if result.boxes is None or result.boxes.xyxy is None:
                cv2.imshow("Person Track + Face Recog", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            person_boxes_xyxy = [to_xyxy_from_ultralytics(b) for b in result.boxes.xyxy.cpu().numpy()]
            track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

            # 5) Face detections on the same frame (your detector)
            face_boxes_xywh = detect_faces_yolo8(frame)  # [(x,y,w,h), ...]

            # 6) Assign faces → person tracks (via IoU)
            assignment = assign_faces_to_persons(person_boxes_xyxy, face_boxes_xywh, iou_thresh=0.2)

            now = time.time()

            # 7) Iterate each tracked person box
            for i, pid in enumerate(track_ids):
                (x1, y1, x2, y2) = person_boxes_xyxy[i]
                color = (64, 192, 255)  # person box color
                name_to_show = cache.get(pid, {}).get("name", "Unknown")

                # If a face is associated, (re)recognize on cooldown/unknown
                if i in assignment:
                    face_idx = assignment[i]
                    fx, fy, fw, fh = face_boxes_xywh[face_idx]
                    do_recognize = (
                        pid not in cache or
                        (now - cache[pid]["t"]) >= RECOG_COOLDOWN or
                        cache[pid]["name"] == "Unknown"
                    )
                    if do_recognize:
                        face_rgb = crop_and_align(frame, (fx, fy, fw, fh))
                        if face_rgb is not None:
                            emb = embedder(face_rgb).astype(np.float32)
                            hits = search_topk(conn, emb, k=5)
                            if hits:
                                best = max(hits, key=lambda r: r[3])  # by cosine_sim
                                sim = float(best[3])
                                # adjust the threshold as you used earlier: label if sim >= (1 - threshold)
                                # Here we use a direct sim cut: e.g., 0.6
                                name = best[2] if sim >= 0.6 else "Unknown"
                            else:
                                sim, name = 0.0, "Unknown"
                            cache[pid] = {"name": name, "sim": sim, "t": now}
                            name_to_show = name

                    # Draw the face box as well (green known / red unknown)
                    known = name_to_show != "Unknown"
                    face_color = (0, 200, 0) if known else (0, 0, 200)
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), face_color, 2)
                    cv2.putText(frame, name_to_show, (fx, fy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

                # Draw the person track box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {pid}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Person Track + Face Recog", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        conn.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

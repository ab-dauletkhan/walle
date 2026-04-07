from ultralytics import YOLO
import cv2
from pipeline import crop_and_align, ArcFaceEmbedder
import psycopg2
import numpy as np
import time




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
    
    
# Initialize models
model = YOLO("yolov8m-face.pt")
embedder = ArcFaceEmbedder("models/w600k_r50.onnx")
conn = psycopg2.connect("postgresql://faceuser:facepass@localhost:5432/facedb")

# Run YOLO tracker
results = model.track(source=0, tracker='bytetrack.yaml', show=True, stream=True, persist=True)

# Cache to avoid re-embedding every frame
cache = {}  # track_id -> {"name": str, "t": float}

for frame_result in results:  # frame_result per frame
    frame = frame_result.orig_img
    boxes = frame_result.boxes.xyxy.cpu().numpy()
    ids = frame_result.boxes.id.cpu().numpy() if frame_result.boxes.id is not None else []

    now = time.time()
    for box, tid in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box[:4])
        bbox = (x1, y1, x2 - x1, y2 - y1)

        # Re-recognize only every 2s or if unknown
        do_recognize = (
            tid not in cache or
            (now - cache[tid]["t"]) > 2 or
            cache[tid]["name"] == "Unknown"
        )

        if do_recognize:
            face_rgb = crop_and_align(frame, bbox)
            if face_rgb is not None:
                emb = embedder(face_rgb).astype(np.float32)
                hits = search_topk(conn, emb, k=5)
                if hits:
                    best = max(hits, key=lambda r: r[3])
                    name = best[2] if best[3] >= 0.6 else "Unknown"
                else:
                    name = "Unknown"
                cache[tid] = {"name": name, "t": now}

        label = cache.get(tid, {}).get("name", "Unknown")
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {int(tid)}: {label}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLO Face Tracking + Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
conn.close()

# enroll.py
import glob, psycopg2, cv2, numpy as np
from psycopg2.extras import Json
from pipeline import detect_faces_yolo8, crop_and_align, ArcFaceEmbedder

DSN = "postgresql://faceuser:facepass@localhost:5432/facedb"

def insert_person(conn, external_id, full_name, meta=None):
    with conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO persons (external_id, full_name, meta) VALUES (%s,%s,%s) "
            "ON CONFLICT (external_id) DO UPDATE SET full_name=EXCLUDED.full_name "
            "RETURNING person_id;",
            (external_id, full_name, Json(meta or {}))
            
        )
        return cur.fetchone()[0]

def insert_embedding(conn, person_id, emb, quality=1.0):
    with conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO face_embeddings (person_id, embedding, quality) VALUES (%s,%s,%s)",
            (person_id, emb.tolist(), quality)
        )

def main(folder_glob, external_id, full_name, onnx_path):
    det = detect_faces_yolo8  # or detect_faces_yoloface
    embedder = ArcFaceEmbedder(onnx_path)

    conn = psycopg2.connect(DSN)
    pid = insert_person(conn, external_id, full_name)

    for path in glob.glob(folder_glob):
        img = cv2.imread(path)
        if img is None: 
            continue
        boxes = det(img)
        if not boxes:
            continue
        # take largest face
        x,y,w,h = max(boxes, key=lambda b: b[2]*b[3])
        face = crop_and_align(img, (x,y,w,h))
        if face is None: 
            continue
        emb = embedder(face).astype(np.float32)
        insert_embedding(conn, pid, emb, quality=1.0)

    conn.close()

if __name__ == "__main__":
    # Example: enroll all JPGs of Alice
    # Provide your ArcFace/InsightFace ONNX path below
    main("data/edige/*.jpg", external_id="edige_001", full_name="Edige Akimali",
         onnx_path="models/w600k_r50.onnx")

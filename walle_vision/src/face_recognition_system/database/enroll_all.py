import os, glob
import psycopg2, cv2, numpy as np
from psycopg2.extras import Json

from pgvector.psycopg2 import register_vector
from pgvector import Vector

from src.face_recognition_system.recognize.pipeline import (
    detect_faces_yolo8, crop_and_align, ArcFaceEmbedder
)

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
            (person_id, Vector(emb.astype(np.float32).tolist()), float(quality))
        )

def enroll_person_folder(conn, det, embedder, person_dir):
    name = os.path.basename(person_dir.rstrip("/"))
    external_id = name
    full_name = name

    pid = insert_person(conn, external_id, full_name)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths += glob.glob(os.path.join(person_dir, ext))

    if not image_paths:
        print(f"[skip] {name}: no images")
        return

    inserted = 0
    for path in sorted(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        boxes = det(img)
        if not boxes:
            continue

        # assumes boxes are (x,y,w,h) like your code
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        face = crop_and_align(img, (x, y, w, h))
        if face is None:
            continue

        emb = embedder(face)
        if emb is None:
            continue

        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        insert_embedding(conn, pid, emb, quality=1.0)
        inserted += 1

    print(f"[done] {name}: embeddings inserted = {inserted}")

def main(dataset_root="data", onnx_path="models/w600k_r50.onnx"):
    det = detect_faces_yolo8
    embedder = ArcFaceEmbedder(onnx_path)

    conn = psycopg2.connect(DSN)
    register_vector(conn)

    for name in sorted(os.listdir(dataset_root)):
        person_dir = os.path.join(dataset_root, name)
        if os.path.isdir(person_dir):
            enroll_person_folder(conn, det, embedder, person_dir)

    conn.close()

if __name__ == "__main__":
    main("data", "models/w600k_r50.onnx")

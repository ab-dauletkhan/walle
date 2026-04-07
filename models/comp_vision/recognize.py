import psycopg2, cv2, numpy as np
from pipeline import detect_faces_yolo8, crop_and_align, ArcFaceEmbedder

DSN = "postgresql://faceuser:facepass@localhost:5432/facedb"

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

def recognize_image(img_bgr, onnx_path, threshold=0.8):
    det = detect_faces_yolo8
    embedder = ArcFaceEmbedder(onnx_path)

    boxes = det(img_bgr)
    results = []
    conn = psycopg2.connect(DSN)

    for (x, y, w, h) in boxes:
        face = crop_and_align(img_bgr, (x, y, w, h))
        if face is None:
            continue
        emb = embedder(face).astype(np.float32)

        # Search neighbors
        hits = search_topk(conn, emb, k=5)
        # hits: [(person_id, external_id, full_name, cosine_sim, emb_id), ...]

        if hits:
            best = max(hits, key=lambda r: r[3])  # highest cosine similarity
            sim = best[3]
            label = best[2] if sim >= (1 - threshold) else "Unknown"
        else:
            label = "Unknown"
            sim = 0.0

        # ✅ Draw rectangle and label on image
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)  # green if known, red if unknown
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            img_bgr,
            f"{label}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        results.append({
            "bbox": (x, y, w, h),
            "best_name": label,
            "best_similarity": float(sim),
            "candidates": [
                {"name": r[2], "sim": float(r[3]), "external_id": r[1]} for r in hits
            ]
        })

    conn.close()
    # return img_bgr, results
    return results

if __name__ == "__main__":
    # img = cv2.imread("samples/5.jpg")
    # out = recognize_image(img, "models/w600k_r50.onnx", threshold=0.999)
    # print(out)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        out = recognize_image(frame, "models/w600k_r50.onnx",threshold=0.4
                              )

        # Display output text on the frame
        # if len(out) > 0:
        #     cv2.putText(frame, str(out[0]['best_name']), (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
        #                 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Real-time Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
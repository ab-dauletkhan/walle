from src.tracking.deepsort_pytorch.deep_sort_pytorch.deep_sort.sort.tracker import Tracker as DeepSortTracker
from src.tracking.deepsort_pytorch.deep_sort_pytorch.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric as knnd
from src.tracking.deepsort_pytorch.deep_sort_pytorch.deep_sort.sort.detection import Detection
from src.tracking.deepsort_pytorch.deep_sort_pytorch.deep_sort.deep.feature_extractor import Extractor
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import psycopg2
import time
from src.face_recognition_system.recognize.recognize import recognize_image
from src.face_recognition_system.recognize.pipeline import ArcFaceEmbedder
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, cuda_available={torch.cuda.is_available()}")

DSN = "postgresql://faceuser:facepass@localhost:5432/facedb"



model_yolo = YOLO('models/yolov8n.pt')
model_yolo.to(device)
model_yolo_face = YOLO('models/yolov8n-face.pt')
model_yolo_face.to(device)
extractor = Extractor(model_path='models/ckpt.t7', use_cuda=True) #re-id model

metric = knnd("cosine", matching_threshold=0.5)
tracker = DeepSortTracker(metric,  max_age = 30000, n_init = 2) #saves only peole from last 10 seconds (300 frames, 30 fps)


def get_features_for_tracker(frame):    
    result_yolo = model_yolo.predict(frame, device=0, verbose=False)
    detections = []

    for box in result_yolo[0].boxes:
        x1_yolo, y1_yolo, x2_yolo, y2_yolo, conf_yolo, cls_yolo = box.data[0]
        if cls_yolo != 0 or conf_yolo < 0.6:
            continue

        x1_yolo = int(x1_yolo)
        y1_yolo = int(y1_yolo)
        w1_yolo = int(x2_yolo - x1_yolo)
        h1_yolo = int(y2_yolo - y1_yolo)
        cls_yolo = int(cls_yolo)

        crop_yolo = frame[y1_yolo:y1_yolo+h1_yolo, x1_yolo:x1_yolo+w1_yolo]

        if crop_yolo.size == 0:
            continue  # skip bad crops

        crop_rgb_yolo = cv2.cvtColor(crop_yolo, cv2.COLOR_BGR2RGB)
        feature_yolo = extractor([crop_rgb_yolo])[0] 

        detection = Detection(np.array([x1_yolo, y1_yolo, w1_yolo, h1_yolo]), float(conf_yolo), cls_yolo, feature_yolo)
        detections.append(detection)
    return detections

def person_bbx(track, frame, cx=None, cy=None, label=None):
    x1, y1, w1, h1 = track.to_tlwh()
    x2 = int(x1 + w1)
    y2 = int(y1 + h1)
    track_id = str(track.track_id)

    cv2.rectangle(frame, (int(x1), int(y1)), (x2, y2), (100, 255, 150), 2)
    cv2.putText(frame, track_id, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # text = label + ' ' + f'ID: {str(track_id)}'
    # if x <= cx <= x + w and y <= cy <= y + h:
    #     cv2.putText(frame, track_id, (int(x1), int(y1) - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    
cap = cv2.VideoCapture("/home/c504/Desktop/walle_vision/videos_for_inference/Yerzhan.mp4")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
    

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_vid = cv2.VideoWriter(
    "/home/c504/Desktop/walle_vision/inference_results_tracking/Yerzhan_annotated_1.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),
)	 

flaggg = 0
image = []
conn = psycopg2.connect(DSN)
embedder = ArcFaceEmbedder("models/w600k_r50.onnx")

t_start = time.perf_counter()
frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  
    frames += 1
    detections = get_features_for_tracker(frame)
    tracker.predict()
    tracker.update(detections)
    frame_height, frame_width = frame.shape[:2]

    for track in tracker.tracks:
        if not track.is_confirmed():
            continue
        if track.time_since_update == 0:
            out = recognize_image(frame, embedder, conn=conn, threshold=0.8)


	    # --- Whole person bbox from DeepSort ---
            x1p, y1p, w1p, h1p = track.to_tlwh()
            x2p = x1p + w1p
            y2p = y1p + h1p
            area_person = w1p * h1p
            print(
		f"frame_width: {frame_width}, frame_height: {frame_height}, "
                f"x_left_whole_person: {int(x1p)}, x_right_whole_person: {int(x2p)}, "
                f"y_top_whole_person: {int(y1p)}, y_bottom_whole_person: {int(y2p)}, "
                f"area_of_whole_person: {int(area_person)}"
            )
            if len(out) == 0:
                print(f"frame: {frames}, track_id: {track.track_id}, face: NOT DETECTED")
            else:
                # --- Face bboxes from recognize_image ---
                for face_result in out:
                    fx, fy, fw, fh = face_result['bbox']
                    x_right_face = fx + fw
                    y_right_face = fy + fh
                    area_face = fw * fh
                    print(
                        f"x_left_face: {fx}, x_right_face: {x_right_face}, "
                        f"y_top_face: {fy}, y_bottom_face: {y_right_face}, "
                        f"area_of_face: {area_face}"
                    )


            # if len(out) != 0:
            #     x, y, w, h = out[-1]['bbox']
            #     cx = (x + w) / 2
            #     cy = (y + h) / 2


            person_bbx(track, frame)
            # draw_faces(frame)
    print()

    out_vid.write(frame)
t_end = time.perf_counter()
elapsed = t_end - t_start
fps_proc = frames / elapsed if elapsed > 0 else 0.0

#print(f"Processed {frames} frames in {elapsed:.3f} seconds")
#print(f"Processing speed: {fps_proc:.2f} FPS")


cap.release()
out_vid.release()
conn.close()
if len(image) != 0:
    cv2.imshow("face", image)

from deepsort_pytorch.deep_sort_pytorch.deep_sort.sort.tracker import Tracker as DeepSortTracker
from deepsort_pytorch.deep_sort_pytorch.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric as knnd
from deepsort_pytorch.deep_sort_pytorch.deep_sort.sort.detection import Detection
from deepsort_pytorch.deep_sort_pytorch.deep_sort.deep.feature_extractor import Extractor
from ultralytics import YOLO
import numpy as np
import cv2
import torch


from recognize import recognize_image

# Choose MPS if available, else fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DSN = "postgresql://faceuser:facepass@localhost:5432/facedb"
print(f"Using device: {device}")


model_yolo = YOLO('yolov8n.pt')
model_yolo.to(device)
model_yolo_face = YOLO('yolov8n-face.pt')
model_yolo_face.to(device)
extractor = Extractor(model_path='re-id_model/ckpt.t7', use_cuda=False) #re-id model

metric = knnd("cosine", matching_threshold=0.5)
tracker = DeepSortTracker(metric,  max_age = 30000, n_init = 2) #saves only peole from last 10 seconds (300 frames, 30 fps)


def get_features_for_tracker(frame):    
    result_yolo = model_yolo(frame)
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

    
video_paths = "tracker_test_videos/Screen Recording 2025-11-06 at 9.49.42 AM.mov"
cap = cv2.VideoCapture(video_paths) 

flaggg = 0
image = []


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  
    detections = get_features_for_tracker(frame)
    tracker.predict()
    tracker.update(detections)
        
    for track in tracker.tracks:
        if not track.is_confirmed():
            continue
        if track.time_since_update == 0:
            out = recognize_image(frame, "models/w600k_r50.onnx",threshold=0.8)
            # if len(out) != 0:
            #     x, y, w, h = out[-1]['bbox']
            #     cx = (x + w) / 2
            #     cy = (y + h) / 2


            person_bbx(track, frame)
            # draw_faces(frame)


    cv2.imshow("Video_1", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cv2.waitKey(1)
cap.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

if len(image) != 0:
    cv2.imshow("face", image)
if cv2.waitKey(10000) & 0xFF == ord('q'):
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)


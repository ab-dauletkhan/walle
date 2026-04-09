import time

import cv2
import numpy as np
import serial
import torch
from src.tracking.deepsort_pytorch.deep_sort_pytorch.deep_sort.deep.feature_extractor import (
    Extractor,
)
from src.tracking.deepsort_pytorch.deep_sort_pytorch.deep_sort.sort.detection import (
    Detection,
)
from src.tracking.deepsort_pytorch.deep_sort_pytorch.deep_sort.sort.nn_matching import (
    NearestNeighborDistanceMetric as knnd,
)
from src.tracking.deepsort_pytorch.deep_sort_pytorch.deep_sort.sort.tracker import (
    Tracker as DeepSortTracker,
)
from ultralytics import YOLO

# =========================
# Config
# =========================

SERIAL_PORT = "/dev/ttyACM0"  # change if needed
SERIAL_BAUD = 115200

HEAD_CHANNEL = 2
HEAD_RIGHT_TICK = 150
HEAD_CENTER_TICK = 350
HEAD_LEFT_TICK = 550
HEAD_MIN_TICK = HEAD_RIGHT_TICK
HEAD_MAX_TICK = HEAD_LEFT_TICK

# Smoothing / control
DEAD_ZONE_NORM = 0.08  # do nothing while target is near center
SEND_INTERVAL_SEC = 0.08  # rate limit serial commands
MIN_SEND_DELTA = 2  # only send if changed by at least this many ticks
LOST_TARGET_TIMEOUT_SEC = 1.0  # after this, slowly return to center
RETURN_TO_CENTER_STEP = 3  # slow return step when no target
SMOOTH_ALPHA = 0.20  # low-pass filter for target x

# Step sizes by error magnitude
SMALL_ERR = 0.20
MEDIUM_ERR = 0.40
STEP_SMALL = 3
STEP_MEDIUM = 6
STEP_LARGE = 10

# Detection/tracking
PERSON_CONF_THRESHOLD = 0.6
YOLO_MODEL_PATH = "models/yolov8n.pt"
REID_MODEL_PATH = "models/ckpt.t7"

WINDOW_NAME = "Head Tracking"


# =========================
# Device / models
# =========================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, cuda_available={torch.cuda.is_available()}")

model_yolo = YOLO(YOLO_MODEL_PATH)
model_yolo.to(device)

extractor = Extractor(model_path=REID_MODEL_PATH, use_cuda=torch.cuda.is_available())

metric = knnd("cosine", matching_threshold=0.5)
tracker = DeepSortTracker(metric, max_age=30000, n_init=2)


# =========================
# Serial head controller
# =========================


class HeadSerialController:
    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.5)  # give Arduino/USB serial time to settle after open
        self.current_tick = HEAD_CENTER_TICK
        self.last_sent_tick = None
        self.last_send_time = 0.0

        self.send_head_tick(self.current_tick, force=True)
        print(f"Serial connected on {port}, initial head tick = {self.current_tick}")

    def clamp_tick(self, tick: int) -> int:
        return max(HEAD_MIN_TICK, min(HEAD_MAX_TICK, int(tick)))

    def send_head_tick(self, tick: int, force: bool = False):
        tick = self.clamp_tick(tick)
        now = time.time()

        if not force:
            if (
                self.last_sent_tick is not None
                and abs(tick - self.last_sent_tick) < MIN_SEND_DELTA
            ):
                return
            if now - self.last_send_time < SEND_INTERVAL_SEC:
                return

        cmd = f"set {HEAD_CHANNEL} {tick}\n"
        self.ser.write(cmd.encode("utf-8"))
        self.ser.flush()

        self.current_tick = tick
        self.last_sent_tick = tick
        self.last_send_time = now

    def step_toward(self, target_tick: int, max_step: int):
        target_tick = self.clamp_tick(target_tick)

        if target_tick > self.current_tick:
            new_tick = min(self.current_tick + max_step, target_tick)
        elif target_tick < self.current_tick:
            new_tick = max(self.current_tick - max_step, target_tick)
        else:
            new_tick = self.current_tick

        self.send_head_tick(new_tick)

    def return_to_center(self):
        self.step_toward(HEAD_CENTER_TICK, RETURN_TO_CENTER_STEP)

    def close(self):
        try:
            self.send_head_tick(HEAD_CENTER_TICK, force=True)
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass


# =========================
# Tracking helpers
# =========================


def get_features_for_tracker(frame):
    # same general pattern as your file: YOLO person detect -> crop -> re-id feature -> Detection list
    result_yolo = model_yolo.predict(
        frame, device=0 if torch.cuda.is_available() else "cpu", verbose=False
    )
    detections = []

    for box in result_yolo[0].boxes:
        x1_yolo, y1_yolo, x2_yolo, y2_yolo, conf_yolo, cls_yolo = box.data[0]

        if int(cls_yolo) != 0 or float(conf_yolo) < PERSON_CONF_THRESHOLD:
            continue

        x1_yolo = int(x1_yolo)
        y1_yolo = int(y1_yolo)
        w1_yolo = int(x2_yolo - x1_yolo)
        h1_yolo = int(y2_yolo - y1_yolo)
        cls_yolo = int(cls_yolo)

        crop_yolo = frame[y1_yolo : y1_yolo + h1_yolo, x1_yolo : x1_yolo + w1_yolo]
        if crop_yolo.size == 0:
            continue

        crop_rgb_yolo = cv2.cvtColor(crop_yolo, cv2.COLOR_BGR2RGB)
        feature_yolo = extractor([crop_rgb_yolo])[0]

        detection = Detection(
            np.array([x1_yolo, y1_yolo, w1_yolo, h1_yolo]),
            float(conf_yolo),
            cls_yolo,
            feature_yolo,
        )
        detections.append(detection)

    return detections


def draw_track_bbox(track, frame, color=(100, 255, 150)):
    x1, y1, w1, h1 = track.to_tlwh()
    x2 = int(x1 + w1)
    y2 = int(y1 + h1)
    track_id = str(track.track_id)

    cv2.rectangle(frame, (int(x1), int(y1)), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"ID {track_id}",
        (int(x1), int(y1) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )


def choose_target_track(tracks, frame_width):
    """
    Pick one confirmed target.
    First version: largest confirmed, freshly updated track.
    """
    best = None
    best_score = -1

    for track in tracks:
        if not track.is_confirmed():
            continue
        if track.time_since_update != 0:
            continue

        x1, y1, w1, h1 = track.to_tlwh()
        area = w1 * h1
        if area > best_score:
            best_score = area
            best = track

    return best


def choose_step(abs_error_norm: float) -> int:
    if abs_error_norm < DEAD_ZONE_NORM:
        return 0
    if abs_error_norm < SMALL_ERR:
        return STEP_SMALL
    if abs_error_norm < MEDIUM_ERR:
        return STEP_MEDIUM
    return STEP_LARGE


# =========================
# Main
# =========================


def main():
    head = None
    cap = None

    try:
        head = HeadSerialController(SERIAL_PORT, SERIAL_BAUD)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera opened: width={frame_w}, height={frame_h}, fps={fps}")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1200, 800)

        t_start = time.perf_counter()
        frames = 0

        smoothed_target_x = None
        last_target_time = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frames += 1
            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width / 2.0

            detections = get_features_for_tracker(frame)
            tracker.predict()
            tracker.update(detections)

            # draw image center line
            cv2.line(
                frame,
                (int(frame_center_x), 0),
                (int(frame_center_x), frame_height),
                (255, 255, 0),
                2,
            )

            target_track = choose_target_track(tracker.tracks, frame_width)

            if target_track is not None:
                x1p, y1p, w1p, h1p = target_track.to_tlwh()
                x2p = x1p + w1p
                y2p = y1p + h1p
                area_person = w1p * h1p

                target_x = x1p + w1p / 2.0
                target_y = y1p + h1p / 2.0

                if smoothed_target_x is None:
                    smoothed_target_x = target_x
                else:
                    smoothed_target_x = (
                        1.0 - SMOOTH_ALPHA
                    ) * smoothed_target_x + SMOOTH_ALPHA * target_x

                error_x = smoothed_target_x - frame_center_x
                error_norm = error_x / frame_center_x
                abs_error_norm = abs(error_norm)

                step = choose_step(abs_error_norm)

                if step > 0:
                    # target on LEFT side of image -> turn head LEFT -> tick should increase
                    if error_norm < 0:
                        new_tick = head.current_tick + step
                    else:
                        new_tick = head.current_tick - step

                    new_tick = max(HEAD_MIN_TICK, min(HEAD_MAX_TICK, new_tick))
                    head.send_head_tick(new_tick)

                last_target_time = time.time()

                # drawing
                draw_track_bbox(target_track, frame, color=(100, 255, 150))

                cv2.circle(frame, (int(target_x), int(target_y)), 6, (0, 255, 0), -1)
                cv2.circle(
                    frame, (int(smoothed_target_x), int(target_y)), 6, (0, 165, 255), -1
                )

                cv2.putText(
                    frame,
                    f"target ID={target_track.track_id} area={int(area_person)}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"target_x={target_x:.1f} smooth_x={smoothed_target_x:.1f}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"error_norm={error_norm:.3f} step={step}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                print(
                    f"frame={frames} "
                    f"fw={frame_width} fh={frame_height} "
                    f"track_id={target_track.track_id} "
                    f"x_left={int(x1p)} x_right={int(x2p)} "
                    f"y_top={int(y1p)} y_bottom={int(y2p)} "
                    f"area={int(area_person)} "
                    f"target_x={target_x:.1f} smooth_x={smoothed_target_x:.1f} "
                    f"error_norm={error_norm:.3f} head_tick={head.current_tick}"
                )

            else:
                # no target: after a short timeout, return slowly toward center
                if time.time() - last_target_time > LOST_TARGET_TIMEOUT_SEC:
                    head.return_to_center()

                cv2.putText(
                    frame,
                    "No target",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # show current head tick
            cv2.putText(
                frame,
                f"head_tick={head.current_tick}",
                (20, frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        t_end = time.perf_counter()
        elapsed = t_end - t_start
        fps_proc = frames / elapsed if elapsed > 0 else 0.0

        print(f"Processed {frames} frames in {elapsed:.3f} seconds")
        print(f"Processing speed: {fps_proc:.2f} FPS")

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        if head is not None:
            head.close()


if __name__ == "__main__":
    main()

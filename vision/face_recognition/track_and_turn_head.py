from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import numpy as np

from .common import (
    PEOPLE_LABELS_PATH,
    FaceRecognitionError,
    annotate_detections,
    best_detection,
    create_detection_interpreter,
    create_embedding_interpreter,
    detect_faces,
    input_size,
    load_labels,
    load_people_embeddings,
    recommended_live_edge_tpu_modes,
    recognize_detection,
    require_cv2,
    require_pil_image,
    resolve_data_dir,
)

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30

DEFAULT_SERIAL_PORT = "/dev/ttyCH341USB0"
DEFAULT_SERIAL_BAUD = 115200
SERIAL_STARTUP_WAIT_SEC = 2.5

HEAD_CHANNEL = 2
HEAD_RIGHT_TICK = 150
HEAD_CENTER_TICK = 350
HEAD_LEFT_TICK = 550
HEAD_MIN_TICK = min(HEAD_RIGHT_TICK, HEAD_LEFT_TICK)
HEAD_MAX_TICK = max(HEAD_RIGHT_TICK, HEAD_LEFT_TICK)
HEAD_CENTER_TRIM = 0
HEAD_OPTICAL_CENTER_TICK = HEAD_CENTER_TICK + HEAD_CENTER_TRIM

FACE_SCORE_THRESHOLD = 0.80
MEDIAN_WINDOW = 3
SMOOTH_ALPHA = 0.45
DEADBAND_PX = 30
REVERSE_HYSTERESIS_PX = 15
SEND_INTERVAL_SEC = 0.03
MIN_SEND_DELTA = 1
MAX_STEP_BASE = 1
MAX_STEP_EXTRA = 6
MAX_STEP_CAP = 6
LOST_TARGET_TIMEOUT_SEC = 1.8
RETURN_TO_CENTER_STEP = 1
EDGE_MARGIN_RATIO = 0.18


class HeadController:
    def __init__(self, port: str, baud: int):
        self.enabled = False
        self.ser = None

        self.current_tick = HEAD_OPTICAL_CENTER_TICK
        self.last_sent_tick = None
        self.last_send_time = 0.0

        self.recent_face_x = deque(maxlen=MEDIAN_WINDOW)
        self.smoothed_x = None
        self.last_target_time = 0.0
        self.last_command_dir = 0

        try:
            import serial

            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(SERIAL_STARTUP_WAIT_SEC)
            self.enabled = True
            print(f"Serial connected: {port} @ {baud}")
            self.send_tick(self.current_tick, force=True)
        except ImportError as exc:
            print(f"WARNING: pyserial is not installed: {exc}")
            print("Head control disabled; vision still runs.")
        except Exception as exc:
            print(f"WARNING: serial not available: {exc}")
            print("Head control disabled; vision still runs.")

    def clamp_tick(self, tick):
        return max(HEAD_MIN_TICK, min(HEAD_MAX_TICK, int(round(tick))))

    def send_tick(self, tick, force=False):
        tick = self.clamp_tick(tick)
        now = time.monotonic()

        if not force:
            if (
                self.last_sent_tick is not None
                and abs(tick - self.last_sent_tick) < MIN_SEND_DELTA
            ):
                return False
            if now - self.last_send_time < SEND_INTERVAL_SEC:
                return False

        old_tick = self.current_tick
        self.current_tick = tick

        if self.enabled and self.ser is not None:
            cmd = f"set {HEAD_CHANNEL} {tick}\n"
            self.ser.write(cmd.encode("utf-8"))
            self.ser.flush()

        if tick > old_tick:
            self.last_command_dir = +1
        elif tick < old_tick:
            self.last_command_dir = -1

        self.last_sent_tick = tick
        self.last_send_time = now
        return True

    def reset_tracking_state(self):
        self.recent_face_x.clear()
        self.smoothed_x = None
        self.last_command_dir = 0

    def update_from_face_x(self, face_center_x, frame_width):
        now = time.monotonic()
        frame_center_x = frame_width / 2.0

        self.recent_face_x.append(face_center_x)
        filtered_x = float(np.median(self.recent_face_x))

        if self.smoothed_x is None:
            self.smoothed_x = filtered_x
        else:
            self.smoothed_x = (
                1.0 - SMOOTH_ALPHA
            ) * self.smoothed_x + SMOOTH_ALPHA * filtered_x

        raw_error_px = self.smoothed_x - frame_center_x
        effective_error_px = raw_error_px

        desired_dir = 0
        if raw_error_px < -DEADBAND_PX:
            desired_dir = +1
        elif raw_error_px > DEADBAND_PX:
            desired_dir = -1

        if (
            self.last_command_dir != 0
            and desired_dir != 0
            and desired_dir != self.last_command_dir
            and abs(raw_error_px) < (DEADBAND_PX + REVERSE_HYSTERESIS_PX)
        ):
            desired_dir = 0
            effective_error_px = 0.0

        if abs(raw_error_px) < DEADBAND_PX:
            desired_dir = 0
            effective_error_px = 0.0

        self.last_target_time = now

        edge_margin_px = frame_width * EDGE_MARGIN_RATIO
        left_x = edge_margin_px
        right_x = frame_width - edge_margin_px
        mapped_x = min(max(self.smoothed_x, left_x), right_x)

        target_tick = np.interp(
            mapped_x, [left_x, right_x], [HEAD_LEFT_TICK, HEAD_RIGHT_TICK]
        )

        if desired_dir == 0:
            target_tick = self.current_tick

        target_tick = self.clamp_tick(target_tick)
        error_ratio = abs(effective_error_px) / max(1.0, frame_center_x)
        dynamic_max_step = min(
            MAX_STEP_BASE + int(error_ratio * MAX_STEP_EXTRA), MAX_STEP_CAP
        )

        if desired_dir == 0:
            dynamic_max_step = 0

        tick_error = target_tick - self.current_tick
        if tick_error > dynamic_max_step:
            tick_error = dynamic_max_step
        elif tick_error < -dynamic_max_step:
            tick_error = -dynamic_max_step
        else:
            tick_error = int(round(tick_error))

        self.send_tick(self.current_tick + tick_error)

        return {
            "filtered_x": filtered_x,
            "smoothed_x": self.smoothed_x,
            "raw_error_px": raw_error_px,
            "effective_error_px": effective_error_px,
            "target_tick": int(round(target_tick)),
            "new_tick": self.current_tick,
            "dynamic_max_step": dynamic_max_step,
            "desired_dir": desired_dir,
        }

    def on_target_lost(self):
        now = time.monotonic()

        if now - self.last_target_time < LOST_TARGET_TIMEOUT_SEC:
            return {"returning": False, "new_tick": self.current_tick}

        self.reset_tracking_state()

        if self.current_tick < HEAD_OPTICAL_CENTER_TICK:
            new_tick = min(
                self.current_tick + RETURN_TO_CENTER_STEP, HEAD_OPTICAL_CENTER_TICK
            )
        elif self.current_tick > HEAD_OPTICAL_CENTER_TICK:
            new_tick = max(
                self.current_tick - RETURN_TO_CENTER_STEP, HEAD_OPTICAL_CENTER_TICK
            )
        else:
            new_tick = self.current_tick

        self.send_tick(new_tick)
        return {"returning": True, "new_tick": self.current_tick}

    def close(self):
        try:
            self.send_tick(HEAD_OPTICAL_CENTER_TICK, force=True)
        except Exception:
            pass
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track the strongest face and turn the robot head toward it."
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="OpenCV camera index."
    )
    parser.add_argument(
        "--edge-tpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Edge TPU acceleration for live tracking. On Linux ARM, defaults to TPU detector + CPU embedder.",
    )
    parser.add_argument(
        "--detector-edge-tpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Edge TPU usage for the face detector.",
    )
    parser.add_argument(
        "--embedder-edge-tpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Edge TPU usage for the face embedding model.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for scanned_people data. Defaults to vision/face_recognition/scanned_people.",
    )
    parser.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--serial-baud", type=int, default=DEFAULT_SERIAL_BAUD)
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument(
        "--recognition-threshold", type=float, default=FACE_SCORE_THRESHOLD
    )
    parser.add_argument("--match-threshold", type=float, default=0.5)
    return parser


def open_camera(cv2, camera_index: int):
    if hasattr(cv2, "CAP_V4L2"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_index)


def run(args: argparse.Namespace) -> None:
    cv2 = require_cv2()
    Image = require_pil_image()

    data_dir = resolve_data_dir(args.data_dir)
    labels = load_labels()
    people_labels = load_labels(PEOPLE_LABELS_PATH)
    people_embeddings = load_people_embeddings(data_dir, required=False)
    if not people_embeddings:
        print(
            f"WARNING: no embeddings found under {data_dir}; tracking will run without person names."
        )

    default_detector_edge_tpu, default_embedder_edge_tpu = (
        recommended_live_edge_tpu_modes(args.edge_tpu)
    )
    detector_edge_tpu = (
        default_detector_edge_tpu
        if args.detector_edge_tpu is None
        else args.detector_edge_tpu
    )
    embedder_edge_tpu = (
        default_embedder_edge_tpu
        if args.embedder_edge_tpu is None
        else args.embedder_edge_tpu
    )

    detection_interpreter = create_detection_interpreter(detector_edge_tpu)
    embedding_interpreter = create_embedding_interpreter(embedder_edge_tpu)
    input_width, input_height = input_size(detection_interpreter)

    cap = open_camera(cv2, args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise FaceRecognitionError(f"Could not open camera index {args.camera_index}.")

    head = HeadController(args.serial_port, args.serial_baud)
    print(
        "Runtime: "
        f"detector={'Edge TPU' if detector_edge_tpu else 'CPU'}, "
        f"embedder={'Edge TPU' if embedder_edge_tpu else 'CPU'}"
    )
    print("Press q to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise FaceRecognitionError("Failed to grab frame from camera.")

            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width / 2.0
            image_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_image = image_rgb.convert("RGB").resize(
                (input_width, input_height), Image.LANCZOS
            )

            start_time = time.monotonic()
            detections = detect_faces(
                detection_interpreter,
                detection_image,
                args.detection_threshold,
                frame_width,
                frame_height,
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000

            annotate_detections(frame, detections, labels, cv2)
            cv2.putText(
                frame,
                f"{elapsed_ms:.1f}ms",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.line(
                frame,
                (int(frame_center_x), 0),
                (int(frame_center_x), frame_height),
                (255, 255, 0),
                2,
            )

            detection = best_detection(detections)
            if detection is not None and detection.score > args.recognition_threshold:
                face_center_x = (detection.xmin + detection.xmax) / 2.0
                face_center_y = (detection.ymin + detection.ymax) / 2.0
                debug = head.update_from_face_x(face_center_x, frame_width)

                cv2.circle(
                    frame, (int(face_center_x), int(face_center_y)), 5, (0, 255, 0), -1
                )
                cv2.circle(
                    frame,
                    (int(debug["filtered_x"]), int(face_center_y)),
                    5,
                    (255, 0, 255),
                    -1,
                )
                cv2.circle(
                    frame,
                    (int(debug["smoothed_x"]), int(face_center_y)),
                    5,
                    (0, 165, 255),
                    -1,
                )
                cv2.putText(
                    frame,
                    f"raw_err={debug['raw_error_px']:.1f} eff_err={debug['effective_error_px']:.1f}",
                    (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"target={debug['target_tick']} head={debug['new_tick']} dir={debug['desired_dir']}",
                    (5, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                if people_embeddings:
                    match = recognize_detection(
                        embedding_interpreter,
                        np.asarray(image_rgb),
                        detection,
                        people_labels,
                        people_embeddings,
                        args.match_threshold,
                    )
                    print(
                        f"person on pic: {match.name or 'Unknown'} ({match.confidence:.2f})"
                    )
            else:
                debug = head.on_target_lost()
                if debug["returning"]:
                    cv2.putText(
                        frame,
                        f"returning to center, head={debug['new_tick']}",
                        (5, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2,
                    )

            cv2.putText(
                frame,
                f"head_tick={head.current_tick}",
                (5, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Face Recognition Head Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        head.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except FaceRecognitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

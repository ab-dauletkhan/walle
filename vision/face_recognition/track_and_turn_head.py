from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vision.camera_source import default_headless_mode, open_camera_source
    from vision.face_recognition.coral_runtime import (
        reexec_module_with_coral_python,
        should_delegate_edge_tpu,
    )
    from vision.face_recognition.errors import FaceRecognitionError
else:
    from vision.camera_source import default_headless_mode, open_camera_source
    from .coral_runtime import reexec_module_with_coral_python, should_delegate_edge_tpu
    from .errors import FaceRecognitionError

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30

DEFAULT_SERIAL_PORT = "/dev/ttyCH341USB0"
DEFAULT_SERIAL_BAUD = 115200
SERIAL_STARTUP_WAIT_SEC = 2.5

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

BODY_ASSIST_HEAD_RATIO = 0.60
BODY_ASSIST_STRONG_HEAD_RATIO = 0.78
BODY_ASSIST_EDGE_DWELL_SEC = 0.22
BODY_ASSIST_STRONG_EDGE_DWELL_SEC = 0.12
BODY_ASSIST_COOLDOWN_SEC = 0.80
BODY_ASSIST_SETTLE_SEC = 0.35
BODY_ASSIST_CENTER_BAND_TICKS = 24
BODY_ASSIST_RETURN_STEP = 8
BODY_TURN_PWM = 150
BODY_TURN_45_SEC = 0.28
BODY_TURN_90_SEC = 0.56


class HeadController:
    def __init__(self, port: str, baud: int, np_module):
        self.enabled = False
        self.ser = None
        self.np = np_module

        self.current_tick = HEAD_OPTICAL_CENTER_TICK
        self.last_sent_tick = None
        self.last_send_time = 0.0

        self.recent_face_x = deque(maxlen=MEDIAN_WINDOW)
        self.smoothed_x = None
        self.last_target_time = 0.0
        self.last_command_dir = 0
        self.body_turn_active = False
        self.body_turn_dir = 0
        self.body_turn_started_at = 0.0
        self.body_turn_end_at = 0.0
        self.body_turn_cooldown_until = 0.0
        self.body_settle_until = 0.0
        self.edge_hold_started_at = 0.0
        self.edge_hold_dir = 0

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
            cmd = f"head tick {tick}\n"
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
        self.edge_hold_started_at = 0.0
        self.edge_hold_dir = 0

    def send_raw_command(self, command: str):
        if self.enabled and self.ser is not None:
            self.ser.write(f"{command}\n".encode("utf-8"))
            self.ser.flush()

    def send_spin_command(self, speed: int):
        self.send_raw_command(f"spin {int(speed)}")

    def stop_body_turn(self):
        if self.body_turn_active:
            self.send_raw_command("stopm")
        self.body_turn_active = False
        self.body_turn_dir = 0
        self.body_turn_started_at = 0.0
        self.body_turn_end_at = 0.0
        self.body_turn_cooldown_until = time.monotonic() + BODY_ASSIST_COOLDOWN_SEC
        self.body_settle_until = time.monotonic() + BODY_ASSIST_SETTLE_SEC

    def start_body_turn(self, turn_dir: int, duration_sec: float):
        if turn_dir == 0:
            return False

        if turn_dir > 0:
            spin_speed = BODY_TURN_PWM
        else:
            spin_speed = -BODY_TURN_PWM

        self.send_spin_command(spin_speed)
        now = time.monotonic()
        self.body_turn_active = True
        self.body_turn_dir = turn_dir
        self.body_turn_started_at = now
        self.body_turn_end_at = now + duration_sec
        self.edge_hold_started_at = 0.0
        self.edge_hold_dir = 0
        return True

    def update_body_turn_state(self):
        if not self.body_turn_active:
            return {"turning": False, "just_finished": False, "dir": 0}

        now = time.monotonic()
        if now < self.body_turn_end_at:
            return {"turning": True, "just_finished": False, "dir": self.body_turn_dir}

        turn_dir = self.body_turn_dir
        self.stop_body_turn()
        return {"turning": False, "just_finished": True, "dir": turn_dir}

    def body_assist_debug(self, phase: str, face_center_x=None, frame_width=None):
        head_half_range = max(1.0, (HEAD_MAX_TICK - HEAD_MIN_TICK) / 2.0)
        head_error_ticks = self.current_tick - HEAD_OPTICAL_CENTER_TICK
        head_ratio = abs(head_error_ticks) / head_half_range
        raw_error_px = None
        effective_error_px = None
        desired_dir = 0

        if face_center_x is not None and frame_width is not None:
            frame_center_x = frame_width / 2.0
            raw_error_px = face_center_x - frame_center_x
            effective_error_px = raw_error_px
            if raw_error_px < -DEADBAND_PX:
                desired_dir = +1
            elif raw_error_px > DEADBAND_PX:
                desired_dir = -1
            else:
                effective_error_px = 0.0

        return {
            "phase": phase,
            "filtered_x": face_center_x if face_center_x is not None else 0.0,
            "smoothed_x": face_center_x if face_center_x is not None else 0.0,
            "raw_error_px": raw_error_px,
            "effective_error_px": effective_error_px,
            "target_tick": self.current_tick,
            "new_tick": self.current_tick,
            "dynamic_max_step": 0,
            "desired_dir": desired_dir,
            "head_ratio": head_ratio,
            "body_turn_active": self.body_turn_active,
            "body_turn_dir": self.body_turn_dir,
        }

    def update_from_face_x(self, face_center_x, frame_width):
        now = time.monotonic()
        frame_center_x = frame_width / 2.0

        turn_state = self.update_body_turn_state()
        if turn_state["turning"]:
            return self.body_assist_debug(
                phase="body_turn", face_center_x=face_center_x, frame_width=frame_width
            )

        if turn_state["just_finished"]:
            self.reset_tracking_state()

        if now < self.body_settle_until:
            target_tick = HEAD_OPTICAL_CENTER_TICK
            tick_error = target_tick - self.current_tick
            if tick_error > BODY_ASSIST_RETURN_STEP:
                tick_error = BODY_ASSIST_RETURN_STEP
            elif tick_error < -BODY_ASSIST_RETURN_STEP:
                tick_error = -BODY_ASSIST_RETURN_STEP
            self.send_tick(self.current_tick + tick_error)
            return {
                "filtered_x": face_center_x,
                "smoothed_x": face_center_x,
                "raw_error_px": face_center_x - frame_center_x,
                "effective_error_px": 0.0,
                "target_tick": target_tick,
                "new_tick": self.current_tick,
                "dynamic_max_step": BODY_ASSIST_RETURN_STEP,
                "desired_dir": 0,
                "head_ratio": abs(self.current_tick - HEAD_OPTICAL_CENTER_TICK)
                / max(1.0, (HEAD_MAX_TICK - HEAD_MIN_TICK) / 2.0),
                "body_turn_active": False,
                "body_turn_dir": 0,
                "phase": "settle",
            }

        self.recent_face_x.append(face_center_x)
        filtered_x = float(self.np.median(self.recent_face_x))

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

        target_tick = self.np.interp(
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

        head_half_range = max(1.0, (HEAD_MAX_TICK - HEAD_MIN_TICK) / 2.0)
        head_error_ticks = self.current_tick - HEAD_OPTICAL_CENTER_TICK
        head_ratio = abs(head_error_ticks) / head_half_range

        head_dir = 0
        if head_error_ticks > BODY_ASSIST_CENTER_BAND_TICKS:
            head_dir = +1
        elif head_error_ticks < -BODY_ASSIST_CENTER_BAND_TICKS:
            head_dir = -1

        body_turn_dir = 0
        if head_dir == desired_dir and head_ratio >= BODY_ASSIST_HEAD_RATIO:
            if self.edge_hold_dir != desired_dir:
                self.edge_hold_dir = desired_dir
                self.edge_hold_started_at = now
            dwell_sec = now - self.edge_hold_started_at
            strong_hold = (
                head_ratio >= BODY_ASSIST_STRONG_HEAD_RATIO
                and dwell_sec >= BODY_ASSIST_STRONG_EDGE_DWELL_SEC
            )
            normal_hold = dwell_sec >= BODY_ASSIST_EDGE_DWELL_SEC
            if now >= self.body_turn_cooldown_until and (strong_hold or normal_hold):
                body_turn_dir = desired_dir
                turn_duration = BODY_TURN_90_SEC if strong_hold else BODY_TURN_45_SEC
                self.start_body_turn(body_turn_dir, turn_duration)
        else:
            self.edge_hold_dir = 0
            self.edge_hold_started_at = 0.0

        return {
            "filtered_x": filtered_x,
            "smoothed_x": self.smoothed_x,
            "raw_error_px": raw_error_px,
            "effective_error_px": effective_error_px,
            "target_tick": int(round(target_tick)),
            "new_tick": self.current_tick,
            "dynamic_max_step": dynamic_max_step,
            "desired_dir": desired_dir,
            "head_ratio": head_ratio,
            "body_turn_active": self.body_turn_active,
            "body_turn_dir": body_turn_dir if body_turn_dir else self.body_turn_dir,
            "phase": "body_turn_start" if body_turn_dir else "head_track",
        }

    def on_target_lost(self):
        now = time.monotonic()

        turn_state = self.update_body_turn_state()
        if turn_state["turning"]:
            return {
                "returning": False,
                "new_tick": self.current_tick,
                "body_turn_active": True,
                "phase": "body_turn",
            }
        if turn_state["just_finished"]:
            self.reset_tracking_state()

        if now - self.last_target_time < LOST_TARGET_TIMEOUT_SEC:
            return {
                "returning": False,
                "new_tick": self.current_tick,
                "body_turn_active": False,
                "phase": "hold",
            }

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
        return {
            "returning": True,
            "new_tick": self.current_tick,
            "body_turn_active": False,
            "phase": "return_center",
        }

    def close(self):
        try:
            self.send_raw_command("stopm")
        except Exception:
            pass
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
    parser.add_argument(
        "--headless",
        "--no-window",
        dest="headless",
        action="store_true",
        help="Run without the OpenCV preview window.",
    )
    parser.add_argument(
        "--window",
        dest="headless",
        action="store_false",
        help="Force the OpenCV preview window on.",
    )
    parser.set_defaults(headless=None)
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument(
        "--recognition-threshold", type=float, default=FACE_SCORE_THRESHOLD
    )
    parser.add_argument("--match-threshold", type=float, default=0.5)
    return parser


def _load_runtime_dependencies():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise FaceRecognitionError(
                "NumPy is not installed. Run `uv sync --extra vision-coral` from "
                "the repo root for the main runtime, and `scripts/setup_jetson_coral39.sh` "
                "for the Coral Python 3.9 runtime."
            ) from exc
        raise

    if __package__ in {None, ""}:
        from vision.face_recognition.common import (
            PEOPLE_LABELS_PATH,
            annotate_detections,
            best_detection,
            create_detection_interpreter,
            create_embedding_interpreter,
            detect_faces,
            input_size,
            load_labels,
            load_people_embeddings,
            recognize_detection,
            recommended_live_edge_tpu_modes,
            require_cv2,
            require_pil_image,
            resolve_data_dir,
        )
    else:
        from .common import (
            PEOPLE_LABELS_PATH,
            annotate_detections,
            best_detection,
            create_detection_interpreter,
            create_embedding_interpreter,
            detect_faces,
            input_size,
            load_labels,
            load_people_embeddings,
            recognize_detection,
            recommended_live_edge_tpu_modes,
            require_cv2,
            require_pil_image,
            resolve_data_dir,
        )

    return {
        "np": np,
        "PEOPLE_LABELS_PATH": PEOPLE_LABELS_PATH,
        "annotate_detections": annotate_detections,
        "best_detection": best_detection,
        "create_detection_interpreter": create_detection_interpreter,
        "create_embedding_interpreter": create_embedding_interpreter,
        "detect_faces": detect_faces,
        "input_size": input_size,
        "load_labels": load_labels,
        "load_people_embeddings": load_people_embeddings,
        "recognize_detection": recognize_detection,
        "recommended_live_edge_tpu_modes": recommended_live_edge_tpu_modes,
        "require_cv2": require_cv2,
        "require_pil_image": require_pil_image,
        "resolve_data_dir": resolve_data_dir,
    }


def run(args: argparse.Namespace) -> None:
    deps = _load_runtime_dependencies()
    np = deps["np"]
    cv2 = deps["require_cv2"]()
    Image = deps["require_pil_image"]()
    headless = args.headless if args.headless is not None else default_headless_mode()

    data_dir = deps["resolve_data_dir"](args.data_dir)
    labels = deps["load_labels"]()
    people_labels = deps["load_labels"](deps["PEOPLE_LABELS_PATH"])
    people_embeddings = deps["load_people_embeddings"](data_dir, required=False)
    if not people_embeddings:
        print(
            f"WARNING: no embeddings found under {data_dir}; tracking will run without person names."
        )

    default_detector_edge_tpu, default_embedder_edge_tpu = deps[
        "recommended_live_edge_tpu_modes"
    ](args.edge_tpu)
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

    detection_interpreter = deps["create_detection_interpreter"](detector_edge_tpu)
    embedding_interpreter = deps["create_embedding_interpreter"](embedder_edge_tpu)
    input_width, input_height = deps["input_size"](detection_interpreter)

    opened = open_camera_source(
        args.camera_index,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.fps,
    )
    if opened.source is None or opened.first_frame is None:
        detail = opened.error or f"could not open camera index {args.camera_index}"
        raise FaceRecognitionError(detail)
    camera = opened.source
    pending_frame = opened.first_frame

    head = HeadController(args.serial_port, args.serial_baud, np)
    print(
        "Runtime: "
        f"detector={'Edge TPU' if detector_edge_tpu else 'CPU'}, "
        f"embedder={'Edge TPU' if embedder_edge_tpu else 'CPU'}"
    )
    print(f"Camera source: {opened.backend_name}")
    if headless:
        print("Headless mode active. Press Ctrl+C to stop.")
    else:
        print("Press q to stop.")

    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                ret, frame = camera.read()
                if not ret or frame is None:
                    raise FaceRecognitionError("Failed to grab frame from camera.")

            if frame is None:
                raise FaceRecognitionError("Failed to grab frame from camera.")

            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width / 2.0
            image_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_image = image_rgb.convert("RGB").resize(
                (input_width, input_height), Image.LANCZOS
            )

            start_time = time.monotonic()
            detections = deps["detect_faces"](
                detection_interpreter,
                detection_image,
                args.detection_threshold,
                frame_width,
                frame_height,
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000

            deps["annotate_detections"](frame, detections, labels, cv2)
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

            detection = deps["best_detection"](detections)
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
                    "target="
                    f"{debug['target_tick']} head={debug['new_tick']} "
                    f"dir={debug['desired_dir']} phase={debug.get('phase', 'head_track')}",
                    (5, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"head_ratio={debug.get('head_ratio', 0.0):.2f} body_turn={int(debug.get('body_turn_active', False))}",
                    (5, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                if people_embeddings:
                    match = deps["recognize_detection"](
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
                elif debug.get("body_turn_active"):
                    cv2.putText(
                        frame,
                        "body turning to recenter head",
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
            if not headless:
                cv2.imshow("Face Recognition Head Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        if not headless:
            cv2.destroyAllWindows()
        head.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if should_delegate_edge_tpu(args.edge_tpu):
        return reexec_module_with_coral_python(
            "vision.face_recognition.track_and_turn_head", argv or sys.argv[1:]
        )
    try:
        run(args)
    except KeyboardInterrupt:
        print("stopped.", file=sys.stderr)
        return 130
    except FaceRecognitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import os
import re
import time
from collections import deque

import cv2
import numpy as np
import serial
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate

# Lower resolution usually reduces end-to-end latency.
# If detection quality becomes too poor, try 960x720 or 1280x960 again.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# =========================
# Serial / head config
# =========================
SERIAL_PORT = "/dev/ttyCH341USB0"  # change if needed
SERIAL_BAUD = 115200
SERIAL_STARTUP_WAIT_SEC = 2.5

HEAD_CHANNEL = 2

# Your calibrated values
HEAD_RIGHT_TICK = 150
HEAD_CENTER_TICK = 350
HEAD_LEFT_TICK = 550

HEAD_MIN_TICK = min(HEAD_RIGHT_TICK, HEAD_LEFT_TICK)
HEAD_MAX_TICK = max(HEAD_RIGHT_TICK, HEAD_LEFT_TICK)

# Optical center trim if needed
HEAD_CENTER_TRIM = 0
HEAD_OPTICAL_CENTER_TICK = HEAD_CENTER_TICK + HEAD_CENTER_TRIM

# If head turns the wrong way, flip to -1
HEAD_DIRECTION = 1

# =========================
# Detection threshold
# =========================
FACE_SCORE_THRESHOLD = 0.80

# =========================
# Stabilization / control
# =========================
# Use a tiny recent window, not a big average
MEDIAN_WINDOW = 3
SMOOTH_ALPHA = 0.45

# Center stability
DEADBAND_PX = 30
REVERSE_HYSTERESIS_PX = 15

# PD controller
KP_TICKS = 90.0
KD_TICKS_PER_SEC = 14.0

# Serial pacing
SEND_INTERVAL_SEC = 0.03
MIN_SEND_DELTA = 1

# Small dynamic steps near center
MAX_STEP_BASE = 1
MAX_STEP_EXTRA = 6
MAX_STEP_CAP = 6

# Lost target behavior
LOST_TARGET_TIMEOUT_SEC = 1.8
RETURN_TO_CENTER_STEP = 1


EDGE_MARGIN_RATIO = 0.18


class HeadController:
    def __init__(self, port, baud):
        self.enabled = False
        self.ser = None

        self.current_tick = HEAD_OPTICAL_CENTER_TICK
        self.last_sent_tick = None
        self.last_send_time = 0.0

        self.recent_face_x = deque(maxlen=MEDIAN_WINDOW)
        self.smoothed_x = None

        self.prev_error_norm = 0.0
        self.prev_control_time = None
        self.last_target_time = 0.0

        # +1 means last command moved left (tick increased)
        # -1 means last command moved right (tick decreased)
        self.last_command_dir = 0

        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(SERIAL_STARTUP_WAIT_SEC)
            self.enabled = True
            print(f"Serial connected: {port} @ {baud}")
            self.send_tick(self.current_tick, force=True)
        except Exception as e:
            print(f"WARNING: serial not available: {e}")
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
        self.prev_error_norm = 0.0
        self.prev_control_time = None
        self.last_command_dir = 0

    def update_from_face_x(self, face_center_x, frame_width):
        now = time.monotonic()
        frame_center_x = frame_width / 2.0

        # Median-of-3 to reject jitter spikes
        self.recent_face_x.append(face_center_x)
        filtered_x = float(np.median(self.recent_face_x))

        # Small EMA on top
        if self.smoothed_x is None:
            self.smoothed_x = filtered_x
        else:
            self.smoothed_x = (
                1.0 - SMOOTH_ALPHA
            ) * self.smoothed_x + SMOOTH_ALPHA * filtered_x

        raw_error_px = self.smoothed_x - frame_center_x
        effective_error_px = raw_error_px

        # Desired direction:
        # face left  -> tick should increase -> +1
        # face right -> tick should decrease -> -1
        desired_dir = 0
        if raw_error_px < -DEADBAND_PX:
            desired_dir = +1
        elif raw_error_px > DEADBAND_PX:
            desired_dir = -1

        # Reverse hysteresis: do not instantly flip around the center
        if (
            self.last_command_dir != 0
            and desired_dir != 0
            and desired_dir != self.last_command_dir
            and abs(raw_error_px) < (DEADBAND_PX + REVERSE_HYSTERESIS_PX)
        ):
            desired_dir = 0
            effective_error_px = 0.0

        # Standard deadband
        if abs(raw_error_px) < DEADBAND_PX:
            desired_dir = 0
            effective_error_px = 0.0

        self.last_target_time = now

        # Map face position to the FULL servo range
        # We do not require the face center to touch the exact image edge.
        edge_margin_px = frame_width * EDGE_MARGIN_RATIO
        left_x = edge_margin_px
        right_x = frame_width - edge_margin_px

        mapped_x = min(max(self.smoothed_x, left_x), right_x)

        # left image side  -> left servo tick
        # right image side -> right servo tick
        target_tick = np.interp(
            mapped_x,
            [left_x, right_x],
            [HEAD_LEFT_TICK, HEAD_RIGHT_TICK],
        )

        # If inside deadband / hysteresis hold current position
        if desired_dir == 0:
            target_tick = self.current_tick

        target_tick = self.clamp_tick(target_tick)

        # Small dynamic step: tiny near center, larger farther away
        error_ratio = abs(effective_error_px) / max(1.0, frame_center_x)
        dynamic_max_step = MAX_STEP_BASE + int(error_ratio * MAX_STEP_EXTRA)
        dynamic_max_step = min(dynamic_max_step, MAX_STEP_CAP)

        # If holding position, don't move
        if desired_dir == 0:
            dynamic_max_step = 0

        tick_error = target_tick - self.current_tick

        if tick_error > dynamic_max_step:
            tick_error = dynamic_max_step
        elif tick_error < -dynamic_max_step:
            tick_error = -dynamic_max_step
        else:
            tick_error = int(round(tick_error))

        new_tick = self.current_tick + tick_error
        sent = self.send_tick(new_tick)

        return {
            "filtered_x": filtered_x,
            "smoothed_x": self.smoothed_x,
            "raw_error_px": raw_error_px,
            "effective_error_px": effective_error_px,
            "target_tick": int(round(target_tick)),
            "new_tick": self.current_tick,
            "dynamic_max_step": dynamic_max_step,
            "desired_dir": desired_dir,
            "sent": sent,
        }

    def on_target_lost(self):
        now = time.monotonic()

        if now - self.last_target_time < LOST_TARGET_TIMEOUT_SEC:
            return {
                "returning": False,
                "new_tick": self.current_tick,
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

        sent = self.send_tick(new_tick)

        return {
            "returning": True,
            "new_tick": self.current_tick,
            "sent": sent,
        }

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


def main():
    ifEdgeTPU_1_else_0 = 1

    labels = load_labels("coco_labels.txt")
    people_lables = load_labels("people_labels.txt")
    preloaded_embeddings = preload_embeddings("scanned_people/")

    if ifEdgeTPU_1_else_0 == 1:
        interpreter = Interpreter(
            model_path="models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
            experimental_delegates=[load_delegate("libedgetpu.so.1.0")],
        )
    else:
        interpreter = Interpreter(
            model_path="models/ssd_mobilenet_v2_face_quant_postprocess.tflite"
        )

    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

    if ifEdgeTPU_1_else_0 == 1:
        interpreter_emb = Interpreter(
            model_path="models/Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite",
            experimental_delegates=[load_delegate("libedgetpu.so.1.0")],
        )
    else:
        interpreter_emb = Interpreter(
            model_path="models/Mobilenet1_triplet1589223569_triplet_quant.tflite"
        )

    interpreter_emb.allocate_tensors()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    head = HeadController(SERIAL_PORT, SERIAL_BAUD)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width / 2.0

            image_large = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = image_large.convert("RGB").resize(
                (input_width, input_height), Image.LANCZOS
            )

            start_time = time.monotonic()
            results = detect_objects(interpreter, image, 0.5)
            elapsed_ms = (time.monotonic() - start_time) * 1000

            annotate_objects(frame, results, labels, frame_width, frame_height)

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

            ymin, xmin, ymax, xmax, score = get_best_box_param(
                results, frame_width, frame_height
            )

            print(f"frame_width: {frame_width}, frame_height: {frame_height}")

            if score > FACE_SCORE_THRESHOLD:
                print(
                    f"x_left_face: {xmin}, x_right_face: {xmax}, "
                    f"y_top_face: {ymin}, y_bottom_face: {ymax}"
                )

                face_center_x = (xmin + xmax) / 2.0
                face_center_y = (ymin + ymax) / 2.0

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
                cv2.putText(
                    frame,
                    f"step_cap={debug['dynamic_max_step']}",
                    (5, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )

            else:
                print("face: NOT DETECTED")
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

            if score > FACE_SCORE_THRESHOLD:
                img = np.array(image_large)
                img_cut = img[ymin:ymax, xmin:xmax, :]

                if (
                    img_cut is not None
                    and img_cut.size != 0
                    and img_cut.shape[0] > 0
                    and img_cut.shape[1] > 0
                ):
                    img_cut = cv2.resize(
                        img_cut, dsize=(96, 96), interpolation=cv2.INTER_CUBIC
                    ).astype("uint8")
                    img_cut = img_cut.reshape(1, 96, 96, 3) / 255.0
                    emb = img_to_emb(interpreter_emb, img_cut)
                    get_person_from_embedding(people_lables, emb, preloaded_embeddings)

            cv2.putText(
                frame,
                f"head_tick={head.current_tick}",
                (5, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        head.close()


def preload_embeddings(path, num_emb_check=20):
    folders = sorted(os.listdir(path))
    embeddings = {}
    for folder in folders:
        emb_path = os.path.join(path, folder, "embeddings")
        files = sorted(os.listdir(emb_path))[:num_emb_check]
        embeddings[folder] = [np.load(os.path.join(emb_path, f)) for f in files]
        print(f"Loaded {len(embeddings[folder])} embeddings for {folder}")
    return embeddings


def get_person_from_embedding(people_lables, emb, preloaded_embeddings):
    folders = sorted(preloaded_embeddings.keys())
    averages = np.zeros(len(folders))

    for i, folder in enumerate(folders):
        embs = preloaded_embeddings[folder]
        diffs = np.array([np.sum((emb - e) ** 2) for e in embs])
        averages[i] = diffs.mean()

    who_is_on_pic = 0
    lowest_norm_found = 999
    for run, average in enumerate(averages):
        if average < 0.9 and average < lowest_norm_found:
            lowest_norm_found = average
            who_is_on_pic = run + 1
        print(average)

    print("person on pic: ", people_lables[who_is_on_pic])


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r"[:\s]+", content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details["index"]))
    return tensor


def set_input_tensor_emb(interpreter, input):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details["index"]
    scale, zero_point = input_details["quantization"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.uint8(input / scale + zero_point)


def img_to_emb(interpreter, input):
    set_input_tensor_emb(interpreter, input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    emb = interpreter.get_tensor(output_details["index"])
    scale, zero_point = output_details["quantization"]
    emb = scale * (emb - zero_point)
    return emb


def detect_objects(interpreter, image, threshold):
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                "bounding_box": boxes[i],
                "class_id": classes[i],
                "score": scores[i],
            }
            results.append(result)
    return results


def get_best_box_param(results, frame_width, frame_height):
    best_boxvalue = 0
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1

    for obj in results:
        if obj["score"] > best_boxvalue:
            best_boxvalue = obj["score"]
            ymin, xmin, ymax, xmax = obj["bounding_box"]

            xmin = max(0.0, xmin)
            xmax = min(1.0, xmax)
            ymin = max(0.0, ymin)
            ymax = min(1.0, ymax)

            xmin = int(xmin * frame_width)
            xmax = int(xmax * frame_width)
            ymin = int(ymin * frame_height)
            ymax = int(ymax * frame_height)

    return ymin, xmin, ymax, xmax, best_boxvalue


def annotate_objects(frame, results, labels, frame_width, frame_height):
    for obj in results:
        ymin, xmin, ymax, xmax = obj["bounding_box"]
        xmin = int(xmin * frame_width)
        xmax = int(xmax * frame_width)
        ymin = int(ymin * frame_height)
        ymax = int(ymax * frame_height)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "%s %.2f" % (labels[obj["class_id"]], obj["score"]),
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


if __name__ == "__main__":
    main()

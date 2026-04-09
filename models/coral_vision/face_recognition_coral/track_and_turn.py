"""
Head-tracking with face recognition.

Vision pipeline (from recognize_face.py):
  - SSD MobileNetV2 face detector (TFLite / Edge TPU)
  - MobileNet triplet-embedding model for identity

Head control (from track_and_turn_head.py):
  - Serial commands to a servo controller
  - Smoothed proportional step control
"""

import os
import re
import time

import cv2
import numpy as np
import serial
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# Config — Vision
# =========================

IF_EDGE_TPU = 1          # 1 = use Edge TPU delegates, 0 = CPU only

CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 960

FACE_DETECT_THRESHOLD = 0.50   # minimum score to treat as a face
FACE_TRACK_THRESHOLD  = 0.80   # minimum score to run recognition + drive head

FACE_DETECT_MODEL_TPU = "models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
FACE_DETECT_MODEL_CPU = "models/ssd_mobilenet_v2_face_quant_postprocess.tflite"

FACE_EMB_MODEL_TPU = "models/Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite"
FACE_EMB_MODEL_CPU = "models/Mobilenet1_triplet1589223569_triplet_quant.tflite"

SCANNED_PEOPLE_PATH = "scanned_people/"
PEOPLE_LABELS_PATH  = "people_labels.txt"
NUM_EMB_CHECK       = 20        # how many stored embeddings to compare per person
RECOGNITION_THRESH  = 0.9       # L2-distance threshold for identity match

# =========================
# Config — Head servo
# =========================

SERIAL_PORT  = "/dev/ttyACM0"
SERIAL_BAUD  = 115200

HEAD_CHANNEL    = 2
HEAD_RIGHT_TICK = 150
HEAD_CENTER_TICK= 350
HEAD_LEFT_TICK  = 550
HEAD_MIN_TICK   = HEAD_RIGHT_TICK
HEAD_MAX_TICK   = HEAD_LEFT_TICK

DEAD_ZONE_NORM        = 0.08
SEND_INTERVAL_SEC     = 0.08
MIN_SEND_DELTA        = 2
LOST_TARGET_TIMEOUT   = 1.0
RETURN_TO_CENTER_STEP = 3
SMOOTH_ALPHA          = 0.20

SMALL_ERR   = 0.20
MEDIUM_ERR  = 0.40
STEP_SMALL  = 3
STEP_MEDIUM = 6
STEP_LARGE  = 10

WINDOW_NAME = "Head Tracking – Face Recognition"


# =========================
# Serial head controller  (unchanged from original)
# =========================

class HeadSerialController:
    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.5)
        self.current_tick   = HEAD_CENTER_TICK
        self.last_sent_tick = None
        self.last_send_time = 0.0
        self.send_head_tick(self.current_tick, force=True)
        print(f"Serial connected on {port}, initial head tick = {self.current_tick}")

    def clamp_tick(self, tick: int) -> int:
        return max(HEAD_MIN_TICK, min(HEAD_MAX_TICK, int(tick)))

    def send_head_tick(self, tick: int, force: bool = False):
        tick = self.clamp_tick(tick)
        now  = time.time()
        if not force:
            if (self.last_sent_tick is not None
                    and abs(tick - self.last_sent_tick) < MIN_SEND_DELTA):
                return
            if now - self.last_send_time < SEND_INTERVAL_SEC:
                return
        cmd = f"set {HEAD_CHANNEL} {tick}\n"
        self.ser.write(cmd.encode("utf-8"))
        self.ser.flush()
        self.current_tick   = tick
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
# TFLite helpers  (from recognize_face.py)
# =========================

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


def preload_embeddings(path, num_emb_check=NUM_EMB_CHECK):
    folders    = sorted(os.listdir(path))
    embeddings = {}
    for folder in folders:
        emb_path = os.path.join(path, folder, "embeddings")
        files    = sorted(os.listdir(emb_path))[:num_emb_check]
        embeddings[folder] = [np.load(os.path.join(emb_path, f)) for f in files]
        print(f"Loaded {len(embeddings[folder])} embeddings for {folder}")
    return embeddings


def set_input_tensor(interpreter, image):
    tensor_index  = interpreter.get_input_details()[0]["index"]
    input_tensor  = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    return np.squeeze(interpreter.get_tensor(output_details["index"]))


def set_input_tensor_emb(interpreter, inp):
    input_details = interpreter.get_input_details()[0]
    tensor_index  = input_details["index"]
    scale, zero_point = input_details["quantization"]
    input_tensor  = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.uint8(inp / scale + zero_point)


def img_to_emb(interpreter, inp):
    set_input_tensor_emb(interpreter, inp)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    emb = interpreter.get_tensor(output_details["index"])
    scale, zero_point = output_details["quantization"]
    return scale * (emb - zero_point)


def detect_objects(interpreter, image, threshold):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    boxes   = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores  = get_output_tensor(interpreter, 2)
    count   = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            results.append({
                "bounding_box": boxes[i],
                "class_id":     classes[i],
                "score":        scores[i],
            })
    return results


def get_best_detection(results, cam_w, cam_h):
    """Return pixel coords + score of the highest-confidence detection."""
    best_score = 0
    xmin = xmax = ymin = ymax = 0
    for obj in results:
        if obj["score"] > best_score:
            best_score      = obj["score"]
            ymin, xmin, ymax, xmax = obj["bounding_box"]
            xmin = int(max(0.0, xmin) * cam_w)
            xmax = int(min(1.0, xmax) * cam_w)
            ymin = int(max(0.0, ymin) * cam_h)
            ymax = int(min(1.0, ymax) * cam_h)
    return ymin, xmin, ymax, xmax, best_score


def annotate_detections(frame, results, cam_w, cam_h, person_name=None):
    for obj in results:
        ymin, xmin, ymax, xmax = obj["bounding_box"]
        xmin = int(xmin * cam_w)
        xmax = int(xmax * cam_w)
        ymin = int(ymin * cam_h)
        ymax = int(ymax * cam_h)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = person_name if person_name else f"{obj['score']:.2f}"
        cv2.putText(frame, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def identify_face(interpreter_emb, image_large, ymin, xmin, ymax, xmax,
                  people_labels, preloaded_embeddings):
    """Crop best face, embed it, and return identified name (or 'Unknown')."""
    img      = np.array(image_large)
    img_cut  = img[ymin:ymax, xmin:xmax, :]
    if img_cut is None or img_cut.size == 0:
        return "Unknown"

    img_cut = cv2.resize(img_cut, (96, 96),
                         interpolation=cv2.INTER_CUBIC).astype("uint8")
    img_cut = img_cut.reshape(1, 96, 96, 3) / 255.0
    emb     = img_to_emb(interpreter_emb, img_cut)

    folders  = sorted(preloaded_embeddings.keys())
    averages = np.array([
        np.mean([np.sum((emb - e) ** 2) for e in preloaded_embeddings[folder]])
        for folder in folders
    ])

    best_idx   = 0
    best_dist  = 999.0
    for i, avg in enumerate(averages):
        if avg < RECOGNITION_THRESH and avg < best_dist:
            best_dist = avg
            best_idx  = i + 1   # labels are 1-indexed (0 = unknown)

    return people_labels.get(best_idx, "Unknown")


# =========================
# Step-size helper  (from original)
# =========================

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
    # ---- load labels & embeddings ----
    people_labels       = load_labels(PEOPLE_LABELS_PATH)
    preloaded_embeddings = preload_embeddings(SCANNED_PEOPLE_PATH)

    # ---- face detection interpreter ----
    if IF_EDGE_TPU:
        interpreter = Interpreter(
            model_path=FACE_DETECT_MODEL_TPU,
            experimental_delegates=[load_delegate("libedgetpu.so.1.0")])
    else:
        interpreter = Interpreter(model_path=FACE_DETECT_MODEL_CPU)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

    # ---- face embedding interpreter ----
    if IF_EDGE_TPU:
        interpreter_emb = Interpreter(
            model_path=FACE_EMB_MODEL_TPU,
            experimental_delegates=[load_delegate("libedgetpu.so.1.0")])
    else:
        interpreter_emb = Interpreter(model_path=FACE_EMB_MODEL_CPU)
    interpreter_emb.allocate_tensors()

    # ---- camera ----
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ---- head controller ----
    head = HeadSerialController(SERIAL_PORT, SERIAL_BAUD)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1200, 800)

    smoothed_target_x = None
    last_target_time  = 0.0
    frames            = 0
    t_start           = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frames += 1
            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width / 2.0

            # -- resize for detector --
            image_large = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_resized = image_large.convert("RGB").resize(
                (input_width, input_height), Image.LANCZOS)

            # -- detect faces --
            t0      = time.monotonic()
            results = detect_objects(interpreter, image_resized, FACE_DETECT_THRESHOLD)
            elapsed_ms = (time.monotonic() - t0) * 1000

            ymin, xmin, ymax, xmax, best_score = get_best_detection(
                results, CAMERA_WIDTH, CAMERA_HEIGHT)

            # -- identify & display name for strong detections --
            person_name = None
            if best_score > FACE_TRACK_THRESHOLD:
                person_name = identify_face(
                    interpreter_emb, image_large,
                    ymin, xmin, ymax, xmax,
                    people_labels, preloaded_embeddings)
                print(f"frame={frames} face detected → {person_name} "
                      f"score={best_score:.2f} "
                      f"x=[{xmin},{xmax}] y=[{ymin},{ymax}]")
            else:
                print(f"frame={frames} face: NOT DETECTED (best_score={best_score:.2f})")

            # -- annotate frame --
            annotate_detections(frame, results, CAMERA_WIDTH, CAMERA_HEIGHT, person_name)
            cv2.putText(frame, f"{elapsed_ms:.1f}ms", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # center line
            cv2.line(frame,
                     (int(frame_center_x), 0),
                     (int(frame_center_x), frame_height),
                     (255, 255, 0), 2)

            # -- head control --
            if best_score > FACE_TRACK_THRESHOLD:
                target_x = (xmin + xmax) / 2.0

                if smoothed_target_x is None:
                    smoothed_target_x = target_x
                else:
                    smoothed_target_x = (
                        (1.0 - SMOOTH_ALPHA) * smoothed_target_x
                        + SMOOTH_ALPHA * target_x)

                error_x        = smoothed_target_x - frame_center_x
                error_norm     = error_x / frame_center_x
                abs_error_norm = abs(error_norm)
                step           = choose_step(abs_error_norm)

                if step > 0:
                    # face on LEFT  → error_norm < 0 → increase tick (turn left)
                    # face on RIGHT → error_norm > 0 → decrease tick (turn right)
                    new_tick = (head.current_tick + step
                                if error_norm < 0
                                else head.current_tick - step)
                    head.send_head_tick(new_tick)

                last_target_time = time.time()

                # HUD
                cx, cy = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.circle(frame, (int(smoothed_target_x), cy), 6, (0, 165, 255), -1)

                cv2.putText(frame,
                            f"face: {person_name}  score={best_score:.2f}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame,
                            f"target_x={target_x:.1f}  smooth_x={smoothed_target_x:.1f}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame,
                            f"error_norm={error_norm:.3f}  step={step}",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                print(f"  → smooth_x={smoothed_target_x:.1f} "
                      f"error_norm={error_norm:.3f} step={step} "
                      f"head_tick={head.current_tick}")
            else:
                # no face: drift back to center after timeout
                if time.time() - last_target_time > LOST_TARGET_TIMEOUT:
                    head.return_to_center()

                cv2.putText(frame, "No face detected",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            cv2.putText(frame, f"head_tick={head.current_tick}",
                        (20, frame_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        t_end    = time.perf_counter()
        elapsed  = t_end - t_start
        fps_proc = frames / elapsed if elapsed > 0 else 0.0
        print(f"\nProcessed {frames} frames in {elapsed:.3f}s  →  {fps_proc:.2f} FPS")

        cap.release()
        cv2.destroyAllWindows()
        head.close()


if __name__ == "__main__":
    main()

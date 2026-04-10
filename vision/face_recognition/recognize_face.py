from __future__ import annotations

import argparse
import sys

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
    recognize_detection,
    require_cv2,
    require_pil_image,
    resolve_data_dir,
)


DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 960
DEFAULT_CAMERA_FPS = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize enrolled people from a live camera stream."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--edge-tpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the Edge TPU models and libedgetpu delegate.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for scanned_people data. Defaults to vision/face_recognition/scanned_people.",
    )
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--recognition-threshold", type=float, default=0.8)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    return parser


def run(args: argparse.Namespace) -> None:
    cv2 = require_cv2()
    Image = require_pil_image()

    data_dir = resolve_data_dir(args.data_dir)
    labels = load_labels()
    people_labels = load_labels(PEOPLE_LABELS_PATH)
    people_embeddings = load_people_embeddings(data_dir, required=True)

    detection_interpreter = create_detection_interpreter(args.edge_tpu)
    embedding_interpreter = create_embedding_interpreter(args.edge_tpu)
    input_width, input_height = input_size(detection_interpreter)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise FaceRecognitionError(f"Could not open camera index {args.camera_index}.")

    print(f"Loaded embeddings for {len(people_embeddings)} people from {data_dir}")
    print("Press q to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise FaceRecognitionError("Failed to grab frame from camera.")

            frame_height, frame_width = frame.shape[:2]
            image_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_image = image_rgb.convert("RGB").resize((input_width, input_height), Image.LANCZOS)

            detections = detect_faces(
                detection_interpreter,
                detection_image,
                args.detection_threshold,
                frame_width,
                frame_height,
            )
            annotate_detections(frame, detections, labels, cv2)

            detection = best_detection(detections)
            if detection is None:
                print("face: NOT DETECTED")
            else:
                print(
                    "face: "
                    f"x_left={detection.xmin}, x_right={detection.xmax}, "
                    f"y_top={detection.ymin}, y_bottom={detection.ymax}, "
                    f"score={detection.score:.2f}"
                )

            if detection is not None and detection.score >= args.recognition_threshold:
                match = recognize_detection(
                    embedding_interpreter,
                    np.asarray(image_rgb),
                    detection,
                    people_labels,
                    people_embeddings,
                    args.match_threshold,
                )
                name = match.name or "Unknown"
                print(f"person on pic: {name} ({match.confidence:.2f})")

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


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

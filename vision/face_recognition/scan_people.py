from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np

from .common import (
    FaceRecognitionError,
    annotate_detections,
    best_detection,
    create_detection_interpreter,
    crop_face_rgb,
    detect_faces,
    input_size,
    load_labels,
    require_cv2,
    require_pil_image,
    resolve_data_dir,
)


DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 960
DEFAULT_CAMERA_FPS = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enroll one person by saving detected 96x96 face crops."
    )
    parser.add_argument("--person", type=int, required=True, help="Person folder number to create.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--edge-tpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the Edge TPU model and libedgetpu delegate.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for scanned_people data. Defaults to vision/face_recognition/scanned_people.",
    )
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--threshold", type=float, default=0.9, help="Minimum face detection score to save a crop.")
    parser.add_argument("--max-images", type=int, default=0, help="Stop after this many saved crops. 0 means unlimited.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the selected person's existing scan folder.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    cv2 = require_cv2()
    Image = require_pil_image()

    data_dir = resolve_data_dir(args.data_dir)
    person_dir = data_dir / str(args.person)
    png_dir = person_dir / "png"
    npy_dir = person_dir / "npy"

    if person_dir.exists():
        if not args.overwrite:
            raise FaceRecognitionError(
                f"{person_dir} already exists. Pass --overwrite to replace that person's scan data."
            )
        shutil.rmtree(person_dir)

    png_dir.mkdir(parents=True)
    npy_dir.mkdir(parents=True)

    labels = load_labels()
    interpreter = create_detection_interpreter(args.edge_tpu)
    input_width, input_height = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise FaceRecognitionError(f"Could not open camera index {args.camera_index}.")

    saved = 0
    print(f"Saving scans to {person_dir}")
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
                interpreter,
                detection_image,
                args.threshold,
                frame_width,
                frame_height,
            )
            annotate_detections(frame, detections, labels, cv2)

            detection = best_detection(detections)
            if detection is not None:
                crop = crop_face_rgb(np.asarray(image_rgb), detection)
                if crop is not None:
                    Image.fromarray(crop).save(png_dir / f"img_{saved}.png")
                    np.save(npy_dir / f"img_{saved}.npy", crop)
                    saved += 1
                    print(f"saved crop {saved}: score={detection.score:.2f}")

            cv2.imshow("Face Enrollment", frame)

            if args.max_images and saved >= args.max_images:
                break
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

from __future__ import annotations

import argparse
import base64
import importlib
import json
import sys
from typing import Any

import numpy as np

from .common import (
    DEFAULT_DATA_DIR,
    PEOPLE_LABELS_PATH,
    create_detection_interpreter,
    create_embedding_interpreter,
    detect_faces,
    input_size,
    load_labels,
    load_people_embeddings,
    recognize_detection,
)


class CoralRecognizer:
    def __init__(self):
        self._cv2: Any = importlib.import_module("cv2")
        self._Image: Any = importlib.import_module("PIL.Image")
        self._detector = create_detection_interpreter(edge_tpu=True)
        self._embedder = create_embedding_interpreter(edge_tpu=True)
        self._input_width, self._input_height = input_size(self._detector)
        self._people_labels = load_labels(PEOPLE_LABELS_PATH)
        self._people_embeddings = load_people_embeddings(
            DEFAULT_DATA_DIR, required=False
        )

    def detect_only(
        self, rgb_resized: np.ndarray, frame_w: int, frame_h: int
    ) -> list[dict]:
        """Detection-only path for the 30 FPS head tracker.

        Caller pre-resizes to (input_w, input_h) RGB on the main side
        so this skips PIL/cv2 resize + colour convert inside the worker.
        """
        detections = detect_faces(
            self._detector,
            rgb_resized,
            threshold=0.5,
            frame_width=int(frame_w),
            frame_height=int(frame_h),
        )
        return [
            {
                "score": round(float(d.score), 4),
                "xmin": int(d.xmin),
                "ymin": int(d.ymin),
                "xmax": int(d.xmax),
                "ymax": int(d.ymax),
            }
            for d in detections
        ]

    def detect_and_recognize(self, frame_bgr: np.ndarray) -> list[dict]:
        image_rgb = self._Image.fromarray(
            self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        )
        frame_height, frame_width = frame_bgr.shape[:2]
        detection_image = image_rgb.convert("RGB").resize(
            (self._input_width, self._input_height), self._Image.LANCZOS
        )

        detections = detect_faces(
            self._detector,
            detection_image,
            threshold=0.5,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        faces = []
        for detection in detections:
            name = None
            confidence = detection.score
            if detection.score >= 0.8 and self._people_embeddings:
                match = recognize_detection(
                    self._embedder,
                    np.asarray(image_rgb),
                    detection,
                    self._people_labels,
                    self._people_embeddings,
                    match_threshold=0.5,
                )
                name = match.name
                confidence = match.confidence if match.name else detection.score

            faces.append(
                {
                    "name": name or "Unknown",
                    "confidence": round(float(confidence), 2),
                    "location": f"({detection.xmin},{detection.ymin})-({detection.xmax},{detection.ymax})",
                }
            )
        return faces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Python 3.9 Coral worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("serve", help="Run JSON-line Coral worker server.")
    return parser


def decode_frame(payload: dict) -> np.ndarray:
    cv2 = importlib.import_module("cv2")
    encoded = payload["frame_jpeg_base64"]
    data = base64.b64decode(encoded)
    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode input frame.")
    return frame


def _decode_resized_rgb(payload: dict, input_w: int, input_h: int) -> np.ndarray:
    b64 = payload["frame_rgb_b64"]
    raw = base64.b64decode(b64)
    expected = input_w * input_h * 3
    if len(raw) != expected:
        raise RuntimeError(
            f"detect_only expected {expected} bytes ({input_w}x{input_h}x3), "
            f"got {len(raw)}"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(input_h, input_w, 3)


def serve() -> int:
    recognizer = CoralRecognizer()

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            op = payload.get("op")
            if op == "detect_recognize":
                frame = decode_frame(payload)
                faces = recognizer.detect_and_recognize(frame)
                print(json.dumps({"ok": True, "faces": faces}), flush=True)
            elif op == "detect_only":
                rgb = _decode_resized_rgb(
                    payload, recognizer._input_width, recognizer._input_height
                )
                detections = recognizer.detect_only(
                    rgb, int(payload["frame_w"]), int(payload["frame_h"])
                )
                print(
                    json.dumps({"ok": True, "detections": detections}), flush=True
                )
            elif op == "info":
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "input_w": int(recognizer._input_width),
                            "input_h": int(recognizer._input_height),
                        }
                    ),
                    flush=True,
                )
            else:
                raise RuntimeError(f"Unsupported op: {op}")
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

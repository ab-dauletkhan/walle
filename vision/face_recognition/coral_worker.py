from __future__ import annotations

import argparse
import base64
import importlib
import json
import struct
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
    subparsers.add_parser(
        "serve-detect-only",
        help="Run a detection-only binary-framed worker (30 FPS head tracking).",
    )
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


def serve() -> int:
    recognizer = CoralRecognizer()

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            op = payload.get("op")
            if op != "detect_recognize":
                raise RuntimeError(f"Unsupported op: {op}")
            frame = decode_frame(payload)
            faces = recognizer.detect_and_recognize(frame)
            print(json.dumps({"ok": True, "faces": faces}), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
    return 0


class DetectOnlyWorker:
    """Detection-only worker.

    Binary-framed stdin protocol to eliminate the ~20-30 ms/frame cost
    of JPEG encode + base64 + JSON wrap that the main detect_recognize
    worker pays. At 30 FPS this matters — every millisecond of per-frame
    overhead shaves FPS off the ceiling.

    Request frame on stdin:
        <4 bytes LE uint32 payload_len>
        <4 bytes LE uint32 frame_w>
        <4 bytes LE uint32 frame_h>
        <input_w * input_h * 3 bytes raw uint8 RGB>

    The input size is advertised in the handshake line written to stdout
    at startup. The main process pre-resizes every frame to that size
    so the worker skips the PIL/cv2 resize step entirely.

    Response on stdout: one JSON line per request:
        {"ok": true, "detections": [{"score":f, "xmin":i, "ymin":i, "xmax":i, "ymax":i}, ...]}
        {"ok": false, "error": "..."}
    """

    _PAYLOAD_HEADER_STRUCT = struct.Struct("<II")  # frame_w, frame_h

    def __init__(self) -> None:
        self._detector = create_detection_interpreter(edge_tpu=True)
        self._input_w, self._input_h = input_size(self._detector)
        self._bytes_per_frame = self._input_w * self._input_h * 3

    def handshake(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "protocol": "binary-detect-v1",
                "input_w": int(self._input_w),
                "input_h": int(self._input_h),
            }
        )

    def _read_exact(self, n: int) -> bytes:
        buf = sys.stdin.buffer.read(n)
        if len(buf) != n:
            raise EOFError(f"expected {n} bytes, got {len(buf)}")
        return buf

    def serve_forever(self) -> int:
        # Emit handshake so the parent knows the input size and that
        # the detector initialized OK.
        sys.stdout.write(self.handshake() + "\n")
        sys.stdout.flush()

        while True:
            try:
                # Read the 4-byte length prefix for the whole payload.
                len_bytes = sys.stdin.buffer.read(4)
                if not len_bytes:
                    return 0  # parent closed stdin
                if len(len_bytes) != 4:
                    raise EOFError(f"short length prefix: {len(len_bytes)}")
                (payload_len,) = struct.unpack("<I", len_bytes)
                payload = self._read_exact(payload_len)
            except EOFError:
                return 0
            except Exception as exc:
                sys.stdout.write(
                    json.dumps({"ok": False, "error": f"read: {exc}"}) + "\n"
                )
                sys.stdout.flush()
                continue

            try:
                frame_w, frame_h = self._PAYLOAD_HEADER_STRUCT.unpack_from(
                    payload, 0
                )
                raw = payload[self._PAYLOAD_HEADER_STRUCT.size :]
                if len(raw) != self._bytes_per_frame:
                    raise RuntimeError(
                        f"expected {self._bytes_per_frame} image bytes "
                        f"({self._input_w}x{self._input_h}x3), got {len(raw)}"
                    )
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape(
                    self._input_h, self._input_w, 3
                )
                detections = detect_faces(
                    self._detector,
                    rgb,
                    threshold=0.5,
                    frame_width=int(frame_w),
                    frame_height=int(frame_h),
                )
                out = [
                    {
                        "score": round(float(d.score), 4),
                        "xmin": int(d.xmin),
                        "ymin": int(d.ymin),
                        "xmax": int(d.xmax),
                        "ymax": int(d.ymax),
                    }
                    for d in detections
                ]
                sys.stdout.write(
                    json.dumps({"ok": True, "detections": out}) + "\n"
                )
                sys.stdout.flush()
            except Exception as exc:
                sys.stdout.write(
                    json.dumps({"ok": False, "error": str(exc)}) + "\n"
                )
                sys.stdout.flush()


def serve_detect_only() -> int:
    worker = DetectOnlyWorker()
    return worker.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve()
    if args.command == "serve-detect-only":
        return serve_detect_only()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

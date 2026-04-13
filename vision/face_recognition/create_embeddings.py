from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np

from .common import (
    FaceRecognitionError,
    create_embedding_interpreter,
    image_to_embedding,
    resolve_data_dir,
)
from .coral_runtime import reexec_module_with_coral_python, should_delegate_edge_tpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create TFLite face embeddings from enrolled face crops."
    )
    parser.add_argument(
        "--person", type=int, required=True, help="Person folder number to process."
    )
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
    return parser


def run(args: argparse.Namespace) -> None:
    data_dir = resolve_data_dir(args.data_dir)
    person_dir = data_dir / str(args.person)
    npy_dir = person_dir / "npy"
    embeddings_dir = person_dir / "embeddings"

    if not npy_dir.is_dir():
        raise FaceRecognitionError(
            f"No scanned crops found at {npy_dir}. Run `uv run walle-face-scan --person {args.person}` first."
        )

    files = sorted(npy_dir.glob("*.npy"))
    if not files:
        raise FaceRecognitionError(f"No .npy face crops found in {npy_dir}.")

    if embeddings_dir.exists():
        shutil.rmtree(embeddings_dir)
    embeddings_dir.mkdir(parents=True)

    interpreter = create_embedding_interpreter(args.edge_tpu)

    for path in files:
        face_crop = np.load(path).reshape(1, 96, 96, 3) / 255.0
        embedding = image_to_embedding(interpreter, face_crop)
        np.save(embeddings_dir / path.name, embedding)
        print(f"created embedding: {embeddings_dir / path.name}")

    print(f"Created {len(files)} embeddings for person {args.person}.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if should_delegate_edge_tpu(args.edge_tpu):
        return reexec_module_with_coral_python(
            "vision.face_recognition.create_embeddings", argv or sys.argv[1:]
        )
    try:
        run(args)
    except FaceRecognitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

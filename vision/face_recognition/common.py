from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

FACE_DIR = Path(__file__).resolve().parent
MODELS_DIR = FACE_DIR / "models"
DEFAULT_DATA_DIR = FACE_DIR / "scanned_people"
COCO_LABELS_PATH = FACE_DIR / "coco_labels.txt"
PEOPLE_LABELS_PATH = FACE_DIR / "people_labels.txt"

DETECTION_MODEL = "ssd_mobilenet_v2_face_quant_postprocess.tflite"
DETECTION_EDGE_TPU_MODEL = "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
EMBEDDING_MODEL = "Mobilenet1_triplet1589223569_triplet_quant.tflite"
EMBEDDING_EDGE_TPU_MODEL = "Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite"


class FaceRecognitionError(RuntimeError):
    """Raised when the face recognition utilities cannot run safely."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    ymin: int
    xmin: int
    ymax: int
    xmax: int


@dataclass(frozen=True)
class FaceMatch:
    name: Optional[str]
    confidence: float


def resolve_data_dir(data_dir: Optional[str]) -> Path:
    if data_dir:
        return Path(data_dir).expanduser().resolve()
    return DEFAULT_DATA_DIR


def detection_model_path(edge_tpu: bool) -> Path:
    return MODELS_DIR / (DETECTION_EDGE_TPU_MODEL if edge_tpu else DETECTION_MODEL)


def embedding_model_path(edge_tpu: bool) -> Path:
    return MODELS_DIR / (EMBEDDING_EDGE_TPU_MODEL if edge_tpu else EMBEDDING_MODEL)


def ensure_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FaceRecognitionError(f"{description} not found: {path}")
    return path


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise FaceRecognitionError(
            "OpenCV is not installed. Run `uv sync --extra vision-coral` "
            "or `uv sync --extra vision-cpu` from the repo root."
        ) from exc
    return cv2


def require_pil_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise FaceRecognitionError(
            "Pillow is not installed. Run `uv sync --extra vision-coral` "
            "or `uv sync --extra vision-cpu` from the repo root."
        ) from exc
    return Image


def require_tflite_runtime():
    try:
        from tflite_runtime.interpreter import (  # ty: ignore[unresolved-import]
            Interpreter,
            load_delegate,
        )
    except ImportError as exc:
        raise FaceRecognitionError(
            "tflite-runtime is not installed. Run `uv sync --extra vision-coral` "
            "on a supported Linux/Jetson target, or install a compatible "
            "TensorFlow Lite runtime for this Python."
        ) from exc
    return Interpreter, load_delegate


def create_interpreter(model_path: Path, edge_tpu: bool):
    Interpreter, load_delegate = require_tflite_runtime()
    ensure_file(model_path, "TFLite model")

    delegates = None
    if edge_tpu:
        try:
            delegates = [load_delegate("libedgetpu.so.1.0")]
        except (OSError, ValueError) as exc:
            raise FaceRecognitionError(
                "Coral Edge TPU runtime library `libedgetpu.so.1.0` is missing "
                "or failed to load. Install the Coral Edge TPU runtime on the "
                "Jetson/Linux host, then rerun the command."
            ) from exc

    if delegates is None:
        return Interpreter(model_path=str(model_path))
    return Interpreter(model_path=str(model_path), experimental_delegates=delegates)


def create_detection_interpreter(edge_tpu: bool):
    interpreter = create_interpreter(detection_model_path(edge_tpu), edge_tpu)
    interpreter.allocate_tensors()
    return interpreter


def create_embedding_interpreter(edge_tpu: bool):
    interpreter = create_interpreter(embedding_model_path(edge_tpu), edge_tpu)
    interpreter.allocate_tensors()
    return interpreter


def running_on_linux_arm() -> bool:
    machine = platform.machine().lower()
    return machine in {"aarch64", "arm64"} and platform.system() == "Linux"


def recommended_live_edge_tpu_modes(edge_tpu: bool) -> tuple[bool, bool]:
    if not edge_tpu:
        return False, False
    if running_on_linux_arm():
        return True, False
    return True, True


def input_size(interpreter) -> tuple[int, int]:
    _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]
    return int(input_width), int(input_height)


def load_labels(path: Path = COCO_LABELS_PATH) -> dict[int, str]:
    labels: dict[int, str] = {}
    if not path.exists():
        return labels

    with path.open("r", encoding="utf-8") as handle:
        for row_number, content in enumerate(handle):
            content = content.strip()
            if not content:
                continue
            pair = re.split(r"[:\s]+", content, maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def label_for(labels: dict[int, str], class_id: int) -> str:
    return labels.get(int(class_id), str(class_id))


def set_input_tensor(interpreter, image) -> None:
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.asarray(image, dtype=input_tensor.dtype)


def output_tensor(interpreter, index: int):
    output_details = interpreter.get_output_details()[index]
    return np.squeeze(interpreter.get_tensor(output_details["index"]))


def detect_faces(
    interpreter,
    image,
    threshold: float,
    frame_width: int,
    frame_height: int,
) -> list[Detection]:
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    boxes = np.atleast_2d(output_tensor(interpreter, 0))
    classes = np.atleast_1d(output_tensor(interpreter, 1))
    scores = np.atleast_1d(output_tensor(interpreter, 2))
    count = int(np.squeeze(output_tensor(interpreter, 3)))

    detections: list[Detection] = []
    for i in range(min(count, len(scores), len(boxes))):
        score = float(scores[i])
        if score < threshold:
            continue

        ymin, xmin, ymax, xmax = [float(value) for value in boxes[i]]
        xmin = min(1.0, max(0.0, xmin))
        xmax = min(1.0, max(0.0, xmax))
        ymin = min(1.0, max(0.0, ymin))
        ymax = min(1.0, max(0.0, ymax))

        detections.append(
            Detection(
                class_id=int(classes[i]),
                score=score,
                ymin=int(ymin * frame_height),
                xmin=int(xmin * frame_width),
                ymax=int(ymax * frame_height),
                xmax=int(xmax * frame_width),
            )
        )
    return detections


def best_detection(detections: list[Detection]) -> Optional[Detection]:
    if not detections:
        return None
    return max(detections, key=lambda detection: detection.score)


def annotate_detections(
    frame, detections: list[Detection], labels: dict[int, str], cv2_module=None
) -> None:
    cv2 = cv2_module or require_cv2()
    for detection in detections:
        cv2.rectangle(
            frame,
            (detection.xmin, detection.ymin),
            (detection.xmax, detection.ymax),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{label_for(labels, detection.class_id)} {detection.score:.2f}",
            (detection.xmin, max(0, detection.ymin - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


def crop_face_rgb(image_rgb_array, detection: Detection, size: int = 96):
    cv2 = require_cv2()
    crop = image_rgb_array[
        detection.ymin : detection.ymax, detection.xmin : detection.xmax, :
    ]
    if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return None
    return cv2.resize(crop, dsize=(size, size), interpolation=cv2.INTER_CUBIC).astype(
        "uint8"
    )


def set_embedding_input_tensor(interpreter, input_array) -> None:
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details["index"]
    scale, zero_point = input_details["quantization"]
    if not scale:
        raise FaceRecognitionError(
            "Embedding model has unsupported input quantization."
        )

    input_tensor = interpreter.tensor(tensor_index)()[0]
    quantized = np.clip(input_array / scale + zero_point, 0, 255).astype(
        input_tensor.dtype
    )
    input_tensor[:, :] = quantized


def image_to_embedding(interpreter, input_array):
    set_embedding_input_tensor(interpreter, input_array)
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    embedding = interpreter.get_tensor(output_details["index"])
    scale, zero_point = output_details["quantization"]
    if scale:
        embedding = scale * (embedding - zero_point)
    return embedding


def crop_to_embedding(interpreter, face_crop_rgb):
    input_array = face_crop_rgb.reshape(1, 96, 96, 3) / 255.0
    return image_to_embedding(interpreter, input_array)


def load_people_embeddings(
    data_dir: Path,
    max_embeddings: int = 20,
    required: bool = False,
) -> dict[str, list[np.ndarray]]:
    if not data_dir.is_dir():
        if required:
            raise FaceRecognitionError(
                f"No scanned people directory found at {data_dir}. Run "
                "`uv run walle-face-scan --person N` and then "
                "`uv run walle-face-embeddings --person N` first."
            )
        return {}

    people: dict[str, list[np.ndarray]] = {}
    for person_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        embeddings_dir = person_dir / "embeddings"
        if not embeddings_dir.is_dir():
            continue
        embeddings = [
            np.load(path)
            for path in sorted(embeddings_dir.glob("*.npy"))[:max_embeddings]
        ]
        if embeddings:
            people[person_dir.name] = embeddings

    if required and not people:
        raise FaceRecognitionError(
            f"No face embeddings found under {data_dir}. Run "
            "`uv run walle-face-scan --person N` and then "
            "`uv run walle-face-embeddings --person N` first."
        )
    return people


def person_name(people_labels: dict[int, str], folder_name: str) -> str:
    try:
        return people_labels.get(int(folder_name), folder_name)
    except ValueError:
        return folder_name


def match_embedding(
    people_labels: dict[int, str],
    embedding,
    people_embeddings: dict[str, list[np.ndarray]],
    threshold: float,
    max_embeddings: int = 20,
) -> FaceMatch:
    embedding_flat = np.ravel(embedding).astype(float)
    embedding_norm = np.linalg.norm(embedding_flat)
    if embedding_norm < 1e-12:
        return FaceMatch(None, 0.0)

    embedding_unit = embedding_flat / embedding_norm
    best_name = None
    best_similarity = -1.0

    for folder_name, references in people_embeddings.items():
        similarities = []
        for reference in references[:max_embeddings]:
            reference_flat = np.ravel(reference).astype(float)
            reference_norm = np.linalg.norm(reference_flat)
            if reference_norm < 1e-12:
                continue
            similarities.append(
                float(np.dot(embedding_unit, reference_flat / reference_norm))
            )

        if not similarities:
            continue

        average_similarity = float(np.mean(similarities))
        if average_similarity > best_similarity:
            best_similarity = average_similarity
            best_name = person_name(people_labels, folder_name)

    if best_name is None or best_similarity < threshold:
        return FaceMatch(None, max(0.0, best_similarity))
    return FaceMatch(best_name, best_similarity)


def recognize_detection(
    embedding_interpreter,
    image_rgb_array,
    detection: Detection,
    people_labels: dict[int, str],
    people_embeddings: dict[str, list[np.ndarray]],
    match_threshold: float,
) -> FaceMatch:
    crop = crop_face_rgb(image_rgb_array, detection)
    if crop is None:
        return FaceMatch(None, 0.0)
    embedding = crop_to_embedding(embedding_interpreter, crop)
    return match_embedding(people_labels, embedding, people_embeddings, match_threshold)

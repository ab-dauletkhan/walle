# Face Recognition with Edge TPU

This folder contains WALL-E's Coral/TFLite face enrollment and recognition
utilities. Run them from the repository root with uv; do not create a separate
virtual environment inside this folder.

## Setup

```bash
uv sync --extra vision-coral
```

The `vision-coral` extra installs Python dependencies where wheels are
available. Real Edge TPU runs also need the native Coral runtime that provides
`libedgetpu.so.1.0` on the Jetson/Linux host.

Use `--no-edge-tpu` to run the non-TPU TFLite models.

## Enroll a Person

```bash
uv run walle-face-scan --person 1 --edge-tpu
uv run walle-face-embeddings --person 1 --edge-tpu
```

Then edit `people_labels.txt` so the folder number maps to the person's name:

```text
1  Dauletkhan
```

Generated enrollment data is stored under:

```text
vision/face_recognition/scanned_people/{person_number}/
    png/
    npy/
    embeddings/
```

That directory is ignored by git.

## Recognize and Track

```bash
uv run walle-face-recognize --edge-tpu
uv run walle-face-track-head --edge-tpu --serial-port /dev/ttyCH341USB0
```

On Linux ARM boards such as Jetson, the live commands above default to `detector=Edge TPU`
and `embedder=CPU` to avoid instability when loading both live models through the
Edge TPU delegate in one process. You can override either side explicitly with
`--detector-edge-tpu` / `--no-detector-edge-tpu` and
`--embedder-edge-tpu` / `--no-embedder-edge-tpu`.

All commands support `--help` without requiring a camera, TFLite runtime, or
Coral hardware to be present.

## Models

The tracked TFLite models live in `vision/face_recognition/models`:

- `ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite`
- `ssd_mobilenet_v2_face_quant_postprocess.tflite`
- `Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite`
- `Mobilenet1_triplet1589223569_triplet_quant.tflite`

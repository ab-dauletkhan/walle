# Jetson Coral Runbook

This runbook is for the Jetson setup where:

- the main project runs from the normal `uv` environment
- Coral Edge TPU code runs from the dedicated Python 3.9 environment at `.venv-coral39`
- Edge TPU CLI commands automatically re-exec into that Python 3.9 runtime

## Current known-good state

- Coral detection works with Python `3.9` and `tflite-runtime==2.5.0.post1`
- `.venv-coral39` is populated from:
  - `scripts/setup_jetson_coral39.sh`
  - the Coral Debian package `python3-tflite-runtime_2.5.0.post1_arm64.deb`
- `uv run walle-face-scan --edge-tpu` works again on Jetson

## Fresh start tomorrow

Run from the repo root:

```bash
cd ~/Desktop/walle/walle
git status
```

If `.venv-coral39` already exists, the Edge TPU commands should find it automatically.
Optional explicit export:

```bash
export WALLE_CORAL_PYTHON39="$PWD/.venv-coral39/bin/python"
```

If you need to rebuild the Coral runtime:

```bash
export WALLE_CORAL_TFLITE_DEB=/tmp/coral-deb/python3-tflite-runtime_2.5.0.post1_arm64.deb
bash scripts/setup_jetson_coral39.sh
export WALLE_CORAL_PYTHON39="$PWD/.venv-coral39/bin/python"
```

Quick health checks:

```bash
"$WALLE_CORAL_PYTHON39" -c "import importlib.metadata as m; print(m.version('tflite-runtime'))"
"$WALLE_CORAL_PYTHON39" -c "from vision.face_recognition.common import create_detection_interpreter; create_detection_interpreter(True); print('det ok')"
uv run python -c "from vision.face_recognition.coral_runtime import resolve_coral_python; print(resolve_coral_python())"
```

Expected output:

- `2.5.0.post1`
- `det ok`
- the path to `.venv-coral39/bin/python`

## Enrollment flow

If you want to recreate person `1`:

```bash
uv run walle-face-scan --person 1 --edge-tpu --overwrite
uv run walle-face-embeddings --person 1 --edge-tpu
```

Then confirm embeddings exist:

```bash
find vision/face_recognition/scanned_people/1/embeddings -type f | wc -l
```

## Face recognition test

```bash
uv run walle-face-recognize --edge-tpu
```

Expected startup behavior:

- the command should run without segfaulting
- it should use the Python 3.9 Coral runtime automatically
- it should print face detections and recognized names after embeddings exist

## Head tracking test

Primary command:

```bash
uv run walle-face-track-head --edge-tpu --serial-port /dev/ttyCH341USB0 --headless
```

Expected behavior:

- opens camera
- connects to serial
- prints `Serial connected: /dev/ttyCH341USB0 @ 115200`
- runs without a tracking window in headless mode
- moves the head in small steps to center the strongest detected face
- when the target is lost long enough, it slowly returns toward center
- on exit, it sends the head back to center

## Code review notes for `track_and_turn_head.py`

The current code is reasonable to run as-is. Important behavior details:

- On Jetson with `--edge-tpu`, live mode defaults to:
  - detector = Edge TPU
  - embedder = CPU
- That default comes from `recommended_live_edge_tpu_modes()` in `vision/face_recognition/common.py`
- Head movement depends only on face detection, so tracking should still work even if embeddings or name matching are missing
- If serial connection fails, the script does not crash; it logs a warning and keeps vision running
- The script uses `/dev/ttyCH341USB0` by default, so override `--serial-port` if your Arduino enumerates differently
- The script uses the shared camera-source helper:
  - OpenCV V4L2 first
  - OpenCV default second
  - Python GStreamer fallback last
- If `DISPLAY` is unset, headless mode is the default even without `--headless`

## Recommended pre-run checks for head tracking

```bash
ls /dev/ttyCH341USB0
uv run walle-face-embeddings --person 1 --edge-tpu
```

Optional serial sanity check:

```bash
python3 - <<'PY'
import serial
ser = serial.Serial('/dev/ttyCH341USB0', 115200, timeout=1)
print('serial ok')
ser.close()
PY
```

## If `walle-face-track-head --edge-tpu` does not behave correctly

Check runtime resolution:

```bash
uv run python -c "from vision.face_recognition.coral_runtime import resolve_coral_python; print(resolve_coral_python())"
```

Check detector directly:

```bash
"$WALLE_CORAL_PYTHON39" -c "from vision.face_recognition.common import create_detection_interpreter; create_detection_interpreter(True); print('det ok')"
```

If camera opens but the head does not move:

- confirm the script printed `Serial connected`
- confirm the correct serial port is used
- confirm the face score is high enough to exceed the movement threshold

If the command says no embeddings exist:

```bash
uv run walle-face-embeddings --person 1 --edge-tpu
```

## Main app startup

If `.venv-coral39/bin/python` exists, `scripts/start.sh` exports it automatically:

```bash
bash scripts/start.sh
```

That allows the main app vision service to use the Coral Python 3.9 worker while the rest of the app remains in the standard environment.

For full voice mode, make sure both Ollama and Mimic3 are running before or during startup:

```bash
curl -sf http://localhost:11434/api/tags >/dev/null
curl -sf http://localhost:59125 >/dev/null
```

The unified `walle` entrypoint now accepts the same Jetson-friendly audio flags as the standalone assistant:

```bash
uv run walle --jetson \
  --mic-device "USB Composite Device" \
  --speaker-device "UACDemoV1.0" \
  --speaker-rate 48000 \
  --speaker-channels 2
```

Useful notes:

- Jetson voice mode defaults to `tiny-streaming` unless `--stt-model` is explicitly set
- Vision is only reported healthy after the first frame is captured
- `capture_image` now returns a camera-not-ready error instead of pretending vision is available when no live frame exists

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="${WALLE_CORAL_VENV:-$PROJECT_ROOT/.venv-coral39}"
PYTHON_BIN="${WALLE_CORAL_PYTHON39:-python3.9}"
REQ_FILE="$PROJECT_ROOT/vision/face_recognition/requirements-coral39.txt"
TFLITE_DEB="${WALLE_CORAL_TFLITE_DEB:-}"

echo "=== Coral Python 3.9 setup ==="
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python 3.9 interpreter not found: $PYTHON_BIN"
    exit 1
fi

echo "Using Python: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" -m venv "$VENV_PATH"

source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQ_FILE"

if [[ -n "${WALLE_CORAL_TFLITE_WHEEL:-}" ]]; then
    python -m pip install "$WALLE_CORAL_TFLITE_WHEEL"
fi

if [[ -n "$TFLITE_DEB" ]]; then
    if [[ ! -f "$TFLITE_DEB" ]]; then
        echo "ERROR: WALLE_CORAL_TFLITE_DEB does not exist: $TFLITE_DEB"
        exit 1
    fi

    TMP_DIR="$(mktemp -d)"
    dpkg-deb -x "$TFLITE_DEB" "$TMP_DIR"
    SITE_PACKAGES="$VENV_PATH/lib/python3.9/site-packages"
    mkdir -p "$SITE_PACKAGES"
    cp -R "$TMP_DIR/usr/lib/python3/dist-packages/tflite_runtime" "$SITE_PACKAGES/"
    cp -R "$TMP_DIR/usr/lib/python3/dist-packages"/tflite_runtime-*.egg-info "$SITE_PACKAGES/"
    rm -rf "$TMP_DIR"
fi

python - <<'PY'
import sys

print("Python:", sys.version)
try:
    import tflite_runtime
    from tflite_runtime.interpreter import Interpreter, load_delegate  # noqa: F401
    print("tflite_runtime:", getattr(tflite_runtime, "__file__", None))
    print("tflite_runtime.interpreter: OK")
except Exception as exc:  # pragma: no cover - setup script only
    raise SystemExit(
        "ERROR: the Coral Python 3.9 environment does not have a working "
        "tflite_runtime.interpreter module. Install a known-good Jetson wheel, "
        "or set WALLE_CORAL_TFLITE_WHEEL / WALLE_CORAL_TFLITE_DEB.\n"
        f"Original error: {exc}"
    )
PY

echo ""
echo "Coral runtime ready."
echo "Set:"
echo "  export WALLE_CORAL_PYTHON39=\"$VENV_PATH/bin/python\""

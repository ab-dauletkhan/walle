#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="${WALLE_CORAL_VENV:-$PROJECT_ROOT/.venv-coral39}"
CONFIGURED_PYTHON_BIN="${WALLE_CORAL_PYTHON39:-}"
REQ_FILE="$PROJECT_ROOT/vision/face_recognition/requirements-coral39.txt"
TFLITE_DEB="${WALLE_CORAL_TFLITE_DEB:-}"
AUTO_DOWNLOAD_DEB="${WALLE_CORAL_AUTO_TFLITE_DEB:-1}"

cleanup_paths=()

cleanup() {
    local path
    for path in "${cleanup_paths[@]:-}"; do
        [[ -n "$path" && -e "$path" ]] && rm -rf "$path"
    done
}

trap cleanup EXIT

resolve_python39() {
    local candidate=""
    local resolved=""

    for candidate in \
        "$CONFIGURED_PYTHON_BIN" \
        python3.9 \
        /usr/local/bin/python3.9 \
        /usr/bin/python3.9
    do
        [[ -z "$candidate" ]] && continue

        if command -v "$candidate" >/dev/null 2>&1; then
            resolved="$(command -v "$candidate")"
        elif [[ -x "$candidate" ]]; then
            resolved="$candidate"
        else
            continue
        fi

        if "$resolved" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 9) else 1)
PY
        then
            printf '%s\n' "$resolved"
            return 0
        fi
    done

    return 1
}

venv_python() {
    printf '%s\n' "$VENV_PATH/bin/python"
}

site_packages_dir() {
    "$(venv_python)" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
}

verify_tflite_runtime() {
    "$(venv_python)" - <<'PY'
import importlib
import sys

print("Python:", sys.version)
tflite_runtime = importlib.import_module("tflite_runtime")
importlib.import_module("tflite_runtime.interpreter")
print("tflite_runtime:", getattr(tflite_runtime, "__file__", None))
print("tflite_runtime.interpreter: OK")
PY
}

install_tflite_runtime_from_deb() {
    local deb_path="$1"
    local tmp_dir=""
    local site_packages=""
    local dist_packages=""
    local metadata_path=""

    if [[ ! -f "$deb_path" ]]; then
        echo "ERROR: WALLE_CORAL_TFLITE_DEB does not exist: $deb_path"
        exit 1
    fi

    tmp_dir="$(mktemp -d)"
    cleanup_paths+=("$tmp_dir")
    dpkg-deb -x "$deb_path" "$tmp_dir"

    site_packages="$(site_packages_dir)"
    dist_packages="$(find "$tmp_dir/usr/lib" -type d -path '*/dist-packages' | head -n 1)"

    if [[ -z "$dist_packages" ]]; then
        echo "ERROR: Could not find dist-packages in $deb_path"
        exit 1
    fi

    mkdir -p "$site_packages"
    rm -rf "$site_packages/tflite_runtime"
    rm -rf "$site_packages"/tflite_runtime-*.egg-info
    rm -rf "$site_packages"/tflite_runtime-*.dist-info

    cp -R "$dist_packages/tflite_runtime" "$site_packages/"

    metadata_path="$(find "$dist_packages" -maxdepth 1 \
        \( -name 'tflite_runtime-*.egg-info' -o -name 'tflite_runtime-*.dist-info' \) \
        | head -n 1)"
    if [[ -n "$metadata_path" ]]; then
        cp -R "$metadata_path" "$site_packages/"
    fi
}

download_tflite_runtime_deb() {
    local tmp_dir=""
    local downloaded=""
    local candidate=""

    [[ "$AUTO_DOWNLOAD_DEB" == "0" ]] && return 1
    command -v apt >/dev/null 2>&1 || return 1
    command -v apt-cache >/dev/null 2>&1 || return 1

    candidate="$(apt-cache policy python3-tflite-runtime 2>/dev/null | awk '/Candidate:/ {print $2; exit}')"
    if [[ -z "$candidate" || "$candidate" == "(none)" ]]; then
        return 1
    fi

    tmp_dir="$(mktemp -d)"
    cleanup_paths+=("$tmp_dir")

    if ! (
        cd "$tmp_dir"
        apt download python3-tflite-runtime >/dev/null 2>&1
    ); then
        return 1
    fi

    downloaded="$(find "$tmp_dir" -maxdepth 1 -name 'python3-tflite-runtime*.deb' | head -n 1)"
    [[ -n "$downloaded" ]] || return 1

    printf '%s\n' "$downloaded"
}

echo "=== Coral Python 3.9 setup ==="
PYTHON_BIN="$(resolve_python39 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: Python 3.9 interpreter not found. Set WALLE_CORAL_PYTHON39 to a working Python 3.9 binary."
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -m venv "$VENV_PATH"

source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQ_FILE"

if [[ -n "${WALLE_CORAL_TFLITE_WHEEL:-}" ]]; then
    python -m pip install "$WALLE_CORAL_TFLITE_WHEEL"
fi

if [[ -n "$TFLITE_DEB" ]]; then
    install_tflite_runtime_from_deb "$TFLITE_DEB"
fi

if ! verify_tflite_runtime; then
    if [[ -z "$TFLITE_DEB" ]]; then
        auto_downloaded_deb="$(download_tflite_runtime_deb || true)"
        if [[ -n "${auto_downloaded_deb:-}" ]]; then
            echo "Auto-downloaded Coral tflite-runtime package: $auto_downloaded_deb"
            install_tflite_runtime_from_deb "$auto_downloaded_deb"
        fi
    fi
fi

if ! verify_tflite_runtime; then
    cat <<'EOF'
ERROR: the Coral Python 3.9 environment does not have a working
tflite_runtime.interpreter module.

If the Jetson has the Coral APT repo configured, rerun this script and it will
auto-download python3-tflite-runtime. Otherwise provide one of:
  WALLE_CORAL_TFLITE_WHEEL=/path/to/wheel.whl
  WALLE_CORAL_TFLITE_DEB=/path/to/python3-tflite-runtime_*.deb
EOF
    exit 1
fi

echo ""
echo "Coral runtime ready."
echo "Set:"
echo "  export WALLE_CORAL_PYTHON39=\"$VENV_PATH/bin/python\""

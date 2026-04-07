#!/usr/bin/env bash
# WALL-E Startup Script
# Ensures all services are running before launching the orchestrator.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- Configuration ----------
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
MIMIC3_URL="${MIMIC3_URL:-http://localhost:59125}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv}"
WALLE_ARGS="${WALLE_ARGS:-}"

# Collect PIDs of services we start so we can clean up
CHILD_PIDS=()

cleanup() {
    echo ""
    echo "Shutting down WALL-E services..."
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# ---------- 1. Ollama ----------
echo "=== Checking Ollama ==="
if curl -sf "$OLLAMA_URL" >/dev/null 2>&1; then
    echo "  Ollama is running."
else
    echo "  Starting Ollama..."
    ollama serve &
    CHILD_PIDS+=($!)

    # Wait for Ollama to be ready (max 30s)
    for i in $(seq 1 30); do
        if curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
            echo "  Ollama ready."
            break
        fi
        sleep 1
    done

    if ! curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
        echo "  ERROR: Ollama failed to start within 30s."
        exit 1
    fi
fi

# Check if model is pulled
if ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
    echo "  Model '$OLLAMA_MODEL' available."
else
    echo "  Model '$OLLAMA_MODEL' not found. Pulling..."
    ollama pull "$OLLAMA_MODEL"
fi

# ---------- 2. Mimic3 TTS ----------
echo "=== Checking Mimic3 TTS ==="
if curl -sf "$MIMIC3_URL" >/dev/null 2>&1; then
    echo "  Mimic3 TTS is running."
else
    echo "  Mimic3 TTS not found at $MIMIC3_URL."
    echo "  Start it manually: mimic3-server --port 59125"
    echo "  Continuing without TTS (will use console fallback)..."
    WALLE_ARGS="$WALLE_ARGS --no-tts"
fi

# ---------- 3. Python virtual environment ----------
echo "=== Setting up Python ==="
if [ -d "$VENV_DIR" ]; then
    echo "  Activating venv: $VENV_DIR"
    source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null || true
else
    echo "  No venv found at $VENV_DIR, using system Python."
fi

echo "  Python: $(python --version 2>&1)"

# ---------- 4. Launch WALL-E ----------
echo ""
echo "=== Launching WALL-E ==="
echo "  Args: $WALLE_ARGS $*"
echo ""

python "$SCRIPT_DIR/walle_main.py" $WALLE_ARGS "$@"

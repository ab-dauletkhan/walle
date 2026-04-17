#!/usr/bin/env bash
# WALL-E Startup Script
# Ensures all services are running before launching the orchestrator.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------- Configuration ----------
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
MIMIC3_URL="${MIMIC3_URL:-http://localhost:59125}"
UV_EXTRAS="${UV_EXTRAS:---extra jetson}"
WALLE_ARGS="${WALLE_ARGS:-}"
CORAL39_PY="${WALLE_CORAL_PYTHON39:-$PROJECT_ROOT/.venv-coral39/bin/python}"
# Quick mode: ./scripts/start.sh --llm-only  →  text REPL, no vision/tts
if [[ " $* " == *" --llm-only "* ]]; then
    WALLE_ARGS="$WALLE_ARGS --text-mode --no-vision --no-tts"
    set -- $(echo "$@" | sed 's/--llm-only//')
fi


# Jetson UMA: keep model pinned on GPU and shrink KV cache so cudaMalloc
# doesn't fragment once walle's Python stack loads alongside it.
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-2048}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"

# Free reclaimable page cache before Ollama tries to allocate on UMA.
# Non-fatal if sudo is not available (running outside Jetson).
sudo -n sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true

# Kill any stale ollama runner processes from a crashed previous session.
pkill -f "ollama runner" 2>/dev/null || true

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
# Always kill any running ollama so it is restarted with our Jetson env vars
# (OLLAMA_KEEP_ALIVE, OLLAMA_CONTEXT_LENGTH). An ollama started from a plain
# shell won't have them, which causes UMA OOM when walle connects later.
if curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    echo "  Existing Ollama detected — restarting with Jetson env vars..."
    pkill -f "ollama serve" 2>/dev/null || true
    pkill -f "ollama runner" 2>/dev/null || true
    for i in $(seq 1 10); do
        curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1 || break
        sleep 0.5
    done
fi

echo "  Starting Ollama (keep_alive=$OLLAMA_KEEP_ALIVE, ctx=$OLLAMA_CONTEXT_LENGTH)..."
ollama serve >/tmp/ollama.log 2>&1 &
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
    echo "  ERROR: Ollama failed to start within 30s. See /tmp/ollama.log"
    exit 1
fi

# Check if model is pulled
if ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
    echo "  Model '$OLLAMA_MODEL' available."
else
    echo "  Model '$OLLAMA_MODEL' not found. Pulling..."
    ollama pull "$OLLAMA_MODEL"
fi

# Preload model onto GPU BEFORE walle's Python stack grabs RAM.
# On Jetson UMA, delaying this until the first chat request fragments
# the allocator and fails with NvMapMemAllocInternal error 12.
echo "  Preloading '$OLLAMA_MODEL' onto GPU (num_ctx=$OLLAMA_CONTEXT_LENGTH, keep_alive=$OLLAMA_KEEP_ALIVE)..."
curl -sf "$OLLAMA_URL/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$OLLAMA_MODEL\",\"prompt\":\"\",\"keep_alive\":\"$OLLAMA_KEEP_ALIVE\",\"options\":{\"num_ctx\":$OLLAMA_CONTEXT_LENGTH}}" \
    >/dev/null && echo "  Model resident on GPU." \
    || echo "  WARNING: preload call failed; chat may OOM on first request."

# Verify model is actually loaded (api/ps should list it).
if curl -sf "$OLLAMA_URL/api/ps" 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
    echo "  Verified: $OLLAMA_MODEL is resident."
else
    echo "  WARNING: $OLLAMA_MODEL not found in /api/ps after preload."
fi

# ---------- 2. TTS ----------
# Detect requested engine — default piper (local, natural). Users can opt
# back into Mimic3 with --tts-engine mimic3 or --tts-engine=mimic3.
TTS_ENGINE="piper"
if [[ " $* " == *" --tts-engine mimic3 "* ]] || [[ " $* " == *" --tts-engine=mimic3 "* ]]; then
    TTS_ENGINE="mimic3"
fi

if [[ "$TTS_ENGINE" == "piper" ]]; then
    echo "=== Checking Piper TTS voice ==="
    # Match the default --piper-model path in walle/startup.py. Override
    # by exporting PIPER_MODEL=/path/to/voice.onnx before running.
    PIPER_MODEL="${PIPER_MODEL:-$PROJECT_ROOT/voices/en_US-lessac-medium.onnx}"
    PIPER_ONNX_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    PIPER_JSON_URL="${PIPER_ONNX_URL}.json"
    if [[ -f "$PIPER_MODEL" && -f "$PIPER_MODEL.json" ]]; then
        echo "  Piper voice ready: $PIPER_MODEL"
    else
        mkdir -p "$(dirname "$PIPER_MODEL")"
        echo "  Piper voice not found — downloading en_US-lessac-medium (~63 MB)..."
        if curl -fL --retry 3 -o "$PIPER_MODEL" "$PIPER_ONNX_URL" \
        && curl -fL --retry 3 -o "$PIPER_MODEL.json" "$PIPER_JSON_URL"; then
            echo "  Piper voice downloaded."
        else
            echo "  WARNING: Piper voice download failed. Console TTS fallback."
            WALLE_ARGS="$WALLE_ARGS --no-tts"
        fi
    fi
else
    echo "=== Checking Mimic3 TTS ==="
    if curl -sf "$MIMIC3_URL" >/dev/null 2>&1; then
        echo "  Existing Mimic3 detected — reusing."
    else
        if command -v mimic3-server >/dev/null 2>&1; then
            echo "  Starting mimic3-server on $MIMIC3_URL..."
            mimic3-server --port 59125 >/tmp/mimic3.log 2>&1 &
            CHILD_PIDS+=($!)
            for i in $(seq 1 30); do
                if curl -sf "$MIMIC3_URL" >/dev/null 2>&1; then
                    echo "  Mimic3 ready."
                    break
                fi
                sleep 1
            done
            if ! curl -sf "$MIMIC3_URL" >/dev/null 2>&1; then
                echo "  WARNING: Mimic3 failed to start within 30s. See /tmp/mimic3.log"
                echo "  Continuing with console TTS fallback..."
                WALLE_ARGS="$WALLE_ARGS --no-tts"
            fi
        else
            echo "  mimic3-server not installed on PATH. Install it or pass --no-tts."
            echo "  Continuing with console TTS fallback..."
            WALLE_ARGS="$WALLE_ARGS --no-tts"
        fi
    fi
fi

# ---------- 3. Python project environment ----------
echo "=== Setting up Python with uv ==="
if ! command -v uv >/dev/null 2>&1; then
    echo "  ERROR: uv is required. Install uv, then run this script again."
    exit 1
fi

echo "  Sync args: $UV_EXTRAS"
uv sync $UV_EXTRAS
echo "  Python: $(uv run $UV_EXTRAS python --version 2>&1)"
if [[ -x "$CORAL39_PY" ]]; then
    export WALLE_CORAL_PYTHON39="$CORAL39_PY"
    echo "  Coral worker: $WALLE_CORAL_PYTHON39"
fi

# ---------- 4. Launch WALL-E ----------
echo ""
echo "=== Launching WALL-E ==="
echo "  Args: $WALLE_ARGS $*"
echo ""

uv run $UV_EXTRAS walle $WALLE_ARGS "$@"

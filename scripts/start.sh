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
# Vision-describer model used by the `capture_image` LLM tool. Small and
# fast on Jetson; pulled alongside the chat model so the capture_image
# path doesn't silently 404 when the LLM decides to use it.
OLLAMA_VISION_MODEL="${OLLAMA_VISION_MODEL:-moondream}"
MIMIC3_URL="${MIMIC3_URL:-http://localhost:59125}"
UV_EXTRAS="${UV_EXTRAS:---extra jetson}"
# Use an array so device names containing spaces (e.g. "ReSpeaker 4 Mic
# Array") survive expansion into the final `uv run walle ...` command.
# Back-compat: if a caller passed WALLE_ARGS as a string, re-split it
# with the shell's normal word rules.
if [[ -n "${WALLE_ARGS:-}" ]]; then
    # shellcheck disable=SC2206  # we want word-splitting here
    WALLE_ARGS_LIST=( $WALLE_ARGS )
else
    WALLE_ARGS_LIST=()
fi
CORAL39_PY="${WALLE_CORAL_PYTHON39:-$PROJECT_ROOT/.venv-coral39/bin/python}"

# Hardware defaults for this deployment — the physical WALL-E Jetson setup.
# Each is overridable by exporting the env var or passing the flag
# explicitly; we only inject the default when the user didn't. This is
# what turns `bash scripts/start.sh` into the working voice pipeline
# without forcing operators to remember the long incantation every time.
DEFAULT_SERIAL_PORT="${WALLE_SERIAL_PORT:-/dev/ttyCH341USB0}"
DEFAULT_MIC_DEVICE="${WALLE_MIC_DEVICE:-ReSpeaker 4 Mic Array}"
DEFAULT_SPEAKER_DEVICE="${WALLE_SPEAKER_DEVICE:-UACDemoV1.0}"
DEFAULT_SPEAKER_RATE="${WALLE_SPEAKER_RATE:-48000}"
DEFAULT_SPEAKER_CHANNELS="${WALLE_SPEAKER_CHANNELS:-2}"

# Quick mode: ./scripts/start.sh --llm-only  →  text REPL, no vision/tts
if [[ " $* " == *" --llm-only "* ]]; then
    WALLE_ARGS_LIST+=(--text-mode --no-vision --no-tts)
    # Strip --llm-only from positional args without breaking quoting.
    new_args=()
    for a in "$@"; do
        [[ "$a" == "--llm-only" ]] || new_args+=("$a")
    done
    set -- "${new_args[@]}"
fi

# The `--extra jetson` uv install implies this is a Jetson deployment;
# pass --jetson to walle so the runtime overrides (model pin, FPS cap,
# STT model, embedding device) actually apply. Skip if the operator is
# already passing --jetson or opted into --no-jetson.
_needs_jetson=1
_needs_serial=1
_needs_mic=1
_needs_speaker=1
_needs_speaker_rate=1
_needs_speaker_channels=1
_text_mode=0
for a in "$@"; do
    case "$a" in
        --jetson|--jetson=*) _needs_jetson=0 ;;
        --no-jetson) _needs_jetson=0 ;;
        --serial-port|--serial-port=*) _needs_serial=0 ;;
        --mic-device|--mic-device=*) _needs_mic=0 ;;
        --speaker-device|--speaker-device=*) _needs_speaker=0 ;;
        --speaker-rate|--speaker-rate=*) _needs_speaker_rate=0 ;;
        --speaker-channels|--speaker-channels=*) _needs_speaker_channels=0 ;;
        --text-mode) _text_mode=1 ;;
    esac
done
# WALLE_ARGS_LIST entries count for flag-detection too.
for a in "${WALLE_ARGS_LIST[@]}"; do
    case "$a" in
        --jetson|--jetson=*|--no-jetson) _needs_jetson=0 ;;
        --serial-port|--serial-port=*) _needs_serial=0 ;;
        --mic-device|--mic-device=*) _needs_mic=0 ;;
        --speaker-device|--speaker-device=*) _needs_speaker=0 ;;
        --speaker-rate|--speaker-rate=*) _needs_speaker_rate=0 ;;
        --speaker-channels|--speaker-channels=*) _needs_speaker_channels=0 ;;
        --text-mode) _text_mode=1 ;;
    esac
done

if [[ "$_needs_jetson" == "1" ]]; then
    WALLE_ARGS_LIST+=(--jetson)
fi
if [[ "$_text_mode" == "0" ]]; then
    # Audio + serial defaults only apply to voice mode. Text mode has no
    # mic/speaker and may be running on a workstation with no Arduino.
    if [[ "$_needs_serial" == "1" && -e "$DEFAULT_SERIAL_PORT" ]]; then
        WALLE_ARGS_LIST+=(--serial-port "$DEFAULT_SERIAL_PORT")
    fi
    if [[ "$_needs_mic" == "1" ]]; then
        WALLE_ARGS_LIST+=(--mic-device "$DEFAULT_MIC_DEVICE")
    fi
    if [[ "$_needs_speaker" == "1" ]]; then
        WALLE_ARGS_LIST+=(--speaker-device "$DEFAULT_SPEAKER_DEVICE")
    fi
    if [[ "$_needs_speaker_rate" == "1" ]]; then
        WALLE_ARGS_LIST+=(--speaker-rate "$DEFAULT_SPEAKER_RATE")
    fi
    if [[ "$_needs_speaker_channels" == "1" ]]; then
        WALLE_ARGS_LIST+=(--speaker-channels "$DEFAULT_SPEAKER_CHANNELS")
    fi
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

# Check if chat model is pulled
if ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
    echo "  Model '$OLLAMA_MODEL' available."
else
    echo "  Model '$OLLAMA_MODEL' not found. Pulling..."
    ollama pull "$OLLAMA_MODEL"
fi

# Vision model for the capture_image tool. Pre-pull so the first time the
# LLM decides to use capture_image it doesn't 404 against an unpulled
# model and come back with "there was an issue capturing the image".
if [[ -n "$OLLAMA_VISION_MODEL" ]]; then
    if ollama list 2>/dev/null | grep -q "$OLLAMA_VISION_MODEL"; then
        echo "  Vision model '$OLLAMA_VISION_MODEL' available."
    else
        echo "  Vision model '$OLLAMA_VISION_MODEL' not found. Pulling..."
        ollama pull "$OLLAMA_VISION_MODEL" \
            || echo "  WARNING: vision model pull failed — capture_image will degrade to face-recognition summary."
    fi
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
echo "  Args:" "${WALLE_ARGS_LIST[@]}" "$@"
echo ""

# shellcheck disable=SC2086  # UV_EXTRAS is intentionally word-split
uv run $UV_EXTRAS walle "${WALLE_ARGS_LIST[@]}" "$@"

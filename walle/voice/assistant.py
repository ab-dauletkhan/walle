"""
Voice Assistant — Moonshine STT + Intent Recognition + Streaming LLM

Architecture:
    Mic → MicTranscriber (Moonshine STT)
              ↓
         IntentRecognizer ──→ robot command matched? → RobotController
              ↓ (no match)
         SpeechRouter ──→ send to LLM (streaming) → TTSEngine

Each layer is an abstract interface so implementations can be swapped
without touching the orchestration logic.

To start:
python3 -m stt_tts.main --wake-word "hey, rocket" --listen-timeout 3.0
"""

import argparse
import io
import os
import queue
import random
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import requests

# SpeechRouter subclasses TranscriptEventListener for the type; if
# moonshine_voice isn't installed (e.g. text-mode-only hosts), fall
# back to a plain object so module import still succeeds.
try:
    from moonshine_voice import TranscriptEventListener  # type: ignore
except ImportError:  # pragma: no cover - exercised on voice-less hosts

    class TranscriptEventListener:  # type: ignore[no-redef]
        """Stub base used when moonshine_voice is unavailable."""


from walle.voice.llm_client import LLMClient, MockLLMClient


# sounddevice and moonshine_voice are voice-only deps. Importing them
# eagerly makes `walle --text-mode` crash on hosts where they aren't
# installed (e.g. Jetson with `uv sync --extra dev`). Resolve lazily.
def _sd():
    import sounddevice as sd  # noqa: PLC0415

    return sd


def _moonshine():
    import moonshine_voice  # noqa: PLC0415

    return moonshine_voice


def _wavfile():
    from scipy.io import wavfile  # noqa: PLC0415

    return wavfile


# STT model choice map — resolved lazily because it references
# moonshine_voice.ModelArch which may not be installed in text mode.
def _stt_model_choices() -> dict:
    mv = _moonshine()
    return {
        "tiny-streaming": mv.ModelArch.TINY_STREAMING,
        "small-streaming": mv.ModelArch.SMALL_STREAMING,
        "medium-streaming": mv.ModelArch.MEDIUM_STREAMING,
        "tiny": mv.ModelArch.TINY,
        "base": mv.ModelArch.BASE,
    }


# String-keyed list for argparse `choices=` (no import of moonshine_voice).
STT_MODEL_NAMES = [
    "tiny-streaming",
    "small-streaming",
    "medium-streaming",
    "tiny",
    "base",
]

# Audio playback tuning for embedded Linux devices where default
# low-latency settings can underrun under concurrent STT/LLM load.
AUDIO_BLOCKSIZE = 4096
AUDIO_LATENCY = "high"

# Module-level overrides for PortAudio output. Set once at startup from
# --speaker-device / --speaker-rate so every _play_audio_stable() call
# (TTS, wake sound) targets the same physical speaker at a rate the DAC
# actually accepts. Needed on Jetson because:
#   1. The ALSA/pulse default can silently route to HDMI.
#   2. Cheap USB DACs (e.g. UACDemoV1.0) only accept a single fixed rate
#      like 48000 Hz stereo, and PortAudio's hw: path does not resample.
OUTPUT_DEVICE: Optional[int] = None
OUTPUT_RATE: Optional[int] = None
OUTPUT_CHANNELS: Optional[int] = None


def set_output_device(
    device: Optional[int],
    rate: Optional[int] = None,
    channels: Optional[int] = None,
) -> None:
    global OUTPUT_DEVICE, OUTPUT_RATE, OUTPUT_CHANNELS
    OUTPUT_DEVICE = device
    OUTPUT_RATE = rate
    OUTPUT_CHANNELS = channels


def _to_float32_audio(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return np.ascontiguousarray(audio)


def _resample_mono(
    audio: np.ndarray, src_rate: int, dst_rate: int
) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    from scipy.signal import resample_poly  # noqa: PLC0415
    from math import gcd  # noqa: PLC0415

    g = gcd(int(dst_rate), int(src_rate))
    up = int(dst_rate) // g
    down = int(src_rate) // g
    return resample_poly(audio, up, down).astype(np.float32)


def _play_audio_stable(
    audio: np.ndarray, sample_rate: int, blocking: bool = True
) -> None:
    audio = _to_float32_audio(audio)

    # Resample + channel-expand to whatever the selected output device
    # actually accepts. PortAudio's ALSA hw: path does not resample, so
    # a 22050 Hz TTS stream into a 48000-only USB DAC would raise
    # paInvalidSampleRate. Doing it here keeps every caller unaware.
    target_rate = OUTPUT_RATE if OUTPUT_RATE is not None else sample_rate
    if target_rate != sample_rate:
        audio = _resample_mono(audio, sample_rate, target_rate)
        sample_rate = target_rate

    if OUTPUT_CHANNELS is not None and audio.ndim == 1 and OUTPUT_CHANNELS >= 2:
        audio = np.repeat(audio[:, None], OUTPUT_CHANNELS, axis=1)

    audio = np.ascontiguousarray(audio)

    _sd().stop()
    kwargs = dict(
        samplerate=sample_rate,
        blocking=blocking,
        latency=AUDIO_LATENCY,
        blocksize=AUDIO_BLOCKSIZE,
    )
    if OUTPUT_DEVICE is not None:
        kwargs["device"] = OUTPUT_DEVICE
    try:
        _sd().play(audio, **kwargs)
    except Exception as primary_err:
        # The default ALSA output on Jetson frequently rejects Mimic3's
        # native rate (22050 Hz) with paErrorCode -9999. Fall back once to
        # 48000 Hz stereo — the near-universal rate for USB DACs and HDMI.
        # If the user passed --speaker-rate/--speaker-device we respect
        # their choice and don't retry.
        if OUTPUT_RATE is not None:
            raise
        print(
            f"  TTS playback retry: default output rejected {sample_rate} Hz "
            f"({primary_err}); resampling to 48000 Hz stereo.",
            file=sys.stderr,
        )
        mono_1d = audio if audio.ndim == 1 else audio[:, 0]
        retry_audio = _resample_mono(mono_1d, sample_rate, 48000)
        retry_audio = np.ascontiguousarray(
            np.repeat(retry_audio[:, None], 2, axis=1)
        )
        retry_kwargs = dict(
            samplerate=48000,
            blocking=blocking,
            latency=AUDIO_LATENCY,
            blocksize=AUDIO_BLOCKSIZE,
        )
        if OUTPUT_DEVICE is not None:
            retry_kwargs["device"] = OUTPUT_DEVICE
        _sd().play(retry_audio, **retry_kwargs)


# ─────────────────────────────────────────────────────────────
# Layer 1: Robot Controller
# ─────────────────────────────────────────────────────────────


class BaseRobotController(ABC):
    """Interface for executing physical robot actions."""

    @abstractmethod
    def execute(self, action: str, utterance: str, confidence: float) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class StubRobotController(BaseRobotController):
    """Prints actions to console. Replace with serial/ROS controller later."""

    def execute(self, action: str, utterance: str, confidence: float) -> None:
        print(
            f"  [Executing ROBOT action] '{action}' (heard: '{utterance}', confidence: {confidence:.0%})"
        )

    def close(self) -> None:
        pass


class SerialRobotController(BaseRobotController):
    """Maps wake-word ROBOT_INTENTS to RobotControlExecutor tool calls.

    Mirrors walle.startup._RobotBridge but stays in voice.assistant so
    `python -m walle.voice.assistant --serial-port ...` can drive real
    hardware without importing the full startup composition root
    (which pulls in memory, personality, vision, OpenAI, etc.).
    """

    _ACTION_MAP = {
        "forward": ("drive", {"direction": "forward", "speed": 85, "duration_ms": 2000}),
        "backward": ("drive", {"direction": "backward", "speed": 85, "duration_ms": 2000}),
        "left": ("drive", {"direction": "left", "speed": 75, "duration_ms": 900}),
        "right": ("drive", {"direction": "right", "speed": 75, "duration_ms": 900}),
        "stop": ("stop_movement", {}),
    }

    def __init__(self, serial_manager, robot_exec):
        self._serial = serial_manager
        self._robot = robot_exec

    def execute(self, action: str, utterance: str, confidence: float) -> None:
        mapping = self._ACTION_MAP.get(action)
        if mapping is None:
            print(f"  [ROBOT] unknown action '{action}'", file=sys.stderr)
            return
        tool_name, tool_args = mapping
        result = self._robot.execute(tool_name, tool_args)
        print(
            f"  [ROBOT] {action} → {tool_name}{tool_args} "
            f"(heard: '{utterance}', conf {confidence:.0%}) → {result}"
        )

    def close(self) -> None:
        try:
            self._serial.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Layer 2: TTS Engine
# ─────────────────────────────────────────────────────────────


class BaseTTSEngine(ABC):
    """Interface for text-to-speech output.

    Subclasses implement `_synthesize_and_play(text)` as a blocking call.
    The base class wraps it in a daemon worker thread so callers can
    `enqueue(text)` from the LLM streaming loop and let synthesis +
    playback overlap with further token generation. This is the core
    trick that makes sentence-level pipelining possible: the first
    sentence starts speaking while the LLM is still producing the rest.
    """

    def __init__(self) -> None:
        self._tts_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=50)
        self._tts_busy = threading.Event()  # set while synthesising or playing
        # Signalled by drain() so an in-flight _synthesize_and_play() can
        # short-circuit between the HTTP round-trip and playback. Cleared
        # on the next successful enqueue so the next turn starts fresh.
        self._stop_event = threading.Event()
        self._tts_worker = threading.Thread(
            target=self._tts_worker_loop, daemon=True
        )
        self._tts_worker.start()

    @abstractmethod
    def _synthesize_and_play(self, text: str) -> None:
        """Blocking synth + playback for one sentence / utterance."""

    def _tts_worker_loop(self) -> None:
        while True:
            item = self._tts_queue.get()
            if item is None:
                self._tts_queue.task_done()
                break
            try:
                self._tts_busy.set()
                self._synthesize_and_play(item)
            except Exception as e:  # pragma: no cover - engine-specific
                print(f"  TTS worker error: {e}", file=sys.stderr)
            finally:
                self._tts_busy.clear()
                self._tts_queue.task_done()

    def enqueue(self, text: str) -> None:
        """Async: push a sentence to the worker and return immediately."""
        text = text.strip()
        if not text:
            return
        # Fresh sentence means the stop-event from a previous drain is stale.
        self._stop_event.clear()
        self._tts_queue.put(text)

    def wait_until_idle(
        self,
        poll: float = 0.05,
        max_wait: float = 20.0,
        stop_flag: Optional[threading.Event] = None,
    ) -> None:
        """Block until the queue is drained AND the last sentence finished playing.

        Polls instead of `queue.join()` so a hung TTS worker (e.g. Mimic3 HTTP
        stuck) cannot deadlock the LLM thread — the old `join()` waited for
        `task_done()`, which never fires if the worker is blocked inside
        `requests.get()`. After `max_wait` we forcibly drain and return.

        `stop_flag` lets the caller bail out early (used by SpeechRouter to
        honour a mid-TTS "stop" intent).
        """
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            if stop_flag is not None and stop_flag.is_set():
                self.drain()
                return
            if self._tts_queue.empty() and not self._tts_busy.is_set():
                return
            time.sleep(poll)
        print(
            "  ... TTS wait_until_idle timeout — draining to unblock pipeline",
            file=sys.stderr,
        )
        self.drain()

    def drain(self) -> None:
        """Drop any pending sentences and interrupt in-flight playback."""
        # Signal the worker to skip playback if it's between HTTP and aplay.
        self._stop_event.set()
        try:
            while True:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
        except queue.Empty:
            pass
        try:
            _sd().stop()
        except Exception:
            pass

    def speak(self, text: str) -> None:
        """Synchronous convenience wrapper — enqueue + wait."""
        self.enqueue(text)
        self.wait_until_idle()


class ConsoleTTSEngine(BaseTTSEngine):
    """Prints to console instead of speaking."""

    def _synthesize_and_play(self, text: str) -> None:
        print(f"  [TTS] {text}")


class Mimic3TTSEngine(BaseTTSEngine):
    """Speaks aloud via a Mimic3 HTTP server, matching rocket_assistant.py."""

    def __init__(
        self,
        url: str = "http://localhost:59125",
        voice: str = "en_US/vctk_low#p236",
    ):
        super().__init__()
        self._url = url
        self._voice = voice

    def _synthesize_and_play(self, text: str) -> None:
        # Early-out if a drain() has already been requested — avoids a slow
        # HTTP round-trip whose audio we would throw away anyway.
        if self._stop_event.is_set():
            return
        try:
            resp = requests.get(
                f"{self._url}/api/tts",
                params={"text": text, "voice": self._voice},
                timeout=8,  # lower than 30s so a hung Mimic3 can't stall the pipeline
            )
            if resp.status_code != 200:
                print(f"  TTS error: HTTP {resp.status_code}", file=sys.stderr)
                return
            content_type = resp.headers.get("Content-Type", "")
            if (
                "wav" not in content_type
                and "audio" not in content_type
                and "octet" not in content_type
            ):
                print(
                    f"  TTS error: unexpected Content-Type '{content_type}'",
                    file=sys.stderr,
                )
                return
            try:
                sr, audio = _wavfile().read(io.BytesIO(resp.content))
            except Exception as e:
                print(f"  TTS error: corrupt audio data: {e}", file=sys.stderr)
                return
            # Second checkpoint: drain() may have fired while HTTP was in flight.
            if self._stop_event.is_set():
                return
            _play_audio_stable(audio, sr)
        except requests.exceptions.ConnectionError:
            print(
                f"  TTS error: cannot connect to Mimic3 at {self._url} — falling back to console",
                file=sys.stderr,
            )
            print(f"  [TTS-fallback] {text}")
        except requests.exceptions.Timeout:
            print(
                f"  TTS error: Mimic3 HTTP timeout at {self._url} — skipping sentence",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"  TTS error: {e} — falling back to console", file=sys.stderr)
            print(f"  [TTS-fallback] {text}")


# ─────────────────────────────────────────────────────────────
# Layer 3: Speech Router (Intent + LLM fallback)
# ─────────────────────────────────────────────────────────────


class SpeechRouter(TranscriptEventListener):
    """Routes transcription events through a wake-word-gated state machine.

    States:
        IDLE      – Ignoring speech (only robot intents work, handled separately).
        LISTENING – Wake word was heard; collecting speech for LLM query.

    Flow:
        1. User says "hey robot" (or variant) → state becomes LISTENING.
        2. Any text after the wake word in the same line, plus subsequent
           completed lines, are accumulated as the command.
        3. After `listen_timeout` seconds of silence the accumulated text
           is sent to the streaming LLM.
        4. Robot intents (IntentRecognizer) always work regardless of state.
    """

    IDLE = "idle"
    LISTENING = "listening"

    def __init__(
        self,
        llm: LLMClient,
        tts: BaseTTSEngine,
        system_prompt: str,
        wake_word: str = "hey",
        listen_timeout: float = 0.5,
        listen_timeout_long: float = 1.0,
        max_utterance: float = 15.0,
        wake_sounds_dir: Optional[str] = None,
        mic_pause: Optional[Callable[[], None]] = None,
        mic_resume: Optional[Callable[[], None]] = None,
    ):
        self._llm = llm
        self._tts = tts
        self._mic_pause_cb = mic_pause
        self._mic_resume_cb = mic_resume
        self._system_prompt = system_prompt
        self._wake_word = wake_word.strip()
        self._wake_tokens = set(re.sub(r"[^\w\s]", "", wake_word).lower().split())
        self._listen_timeout = listen_timeout
        self._listen_timeout_long = listen_timeout_long
        self._max_utterance = max_utterance
        self._wake_sounds_dir = wake_sounds_dir

        self._last_stable_text: str = ""
        self._last_partial_raw: str = ""
        self._listen_started_monotonic: float = 0.0

        self._handled_utterances: set[str] = set()
        self._conversation: list[dict] = []
        self._max_conversation_length = (
            20  # Rolling window to prevent unbounded memory growth
        )
        self._processing = False
        # `_stop_requested` is a threading.Event rather than a plain bool so
        # it can be passed straight into `BaseTTSEngine.wait_until_idle` to
        # short-circuit the polling loop the instant a "stop" intent fires.
        self._stop_requested = threading.Event()
        self._speaking = False
        self._last_text_length = 0

        # Unified mic-gate, ported from box/pipeline_mem/g4-e2b-boxy_streaming.py.
        # `_audio_active` is set for the full LLM+TTS span so callbacks can
        # drop transcripts in one consistent check. `_mic_gate_until` adds a
        # post-turn time cooldown that absorbs speaker echo still in the
        # PortAudio buffer after TTS stops.
        self._audio_active = threading.Event()
        self._mic_gate_until: float = 0.0

        self._state = self.IDLE
        self._command_text = ""
        self._timeout_timer: Optional[threading.Timer] = None

        # Literal stop-intent whitelist. Matched by normalized text only —
        # never routed through the LLM, so it passes even while processing.
        self._stop_phrases: frozenset[str] = frozenset(
            {
                "stop",
                "cancel",
                "quiet",
                "shut up",
                "wall-e stop",
                "wall e stop",
                "stop wall-e",
                "stop wall e",
            }
        )

        # Echo filter: store words from last TTS output to reject mic echo
        self._echo_words: set[str] = set()
        self._echo_suppress_until: float = 0
        self._wake_play_lock = threading.Lock()
        self._last_wake_play_time = 0.0

    # -- Called by IntentRecognizer callback to mark a line as handled --

    def mark_handled(self, utterance: str) -> None:
        self._handled_utterances.add(utterance.strip())

    # -- TTS with echo suppression --
    #    Two layers:
    #    1) _speaking flag stays True during playback + cooldown (time-based)
    #    2) _is_echo() checks word overlap against last spoken text (content-based)
    #    Together they reliably filter self-heard audio on slow STT pipelines.

    _ECHO_COOLDOWN = 0.25  # hard mic-mute after TTS; covers PortAudio buffer flush on Jetson
    _ECHO_WINDOW = (
        1.5  # seconds after TTS to keep content-based echo filtering
    )
    _POST_TURN_GATE = 0.5  # time-based mic gate after a full LLM+TTS turn ends
    _WAKE_DEBOUNCE_SECONDS = 1.5

    _INITIAL_LISTEN_WINDOW = 10.0  # seconds to start speaking after wake / after TTS

    @property
    def _post_speak_timeout(self) -> float:
        """Initial window for the user to BEGIN speaking after wake word or
        after WALL-E finishes. Once speech starts, the per-sentence silence
        timer ( _current_silence_timeout ) takes over, so making this roomy
        is free — it just means we wait longer for the first word.
        Must also exceed the echo window so self-echo can't trigger commit.
        """
        return max(self._INITIAL_LISTEN_WINDOW, self._ECHO_WINDOW + 1)

    def _is_gated(self) -> bool:
        """Single source of truth for 'should we ignore this mic transcript?'.

        Replaces the old (_processing or _speaking or echo-window) triad so
        there is no race window where flags are half-set. Mirrors box's
        `audio_tracker.active.is_set() or time.time() < mic_gate_until`.
        """
        return self._audio_active.is_set() or time.time() < self._mic_gate_until

    def _pause_mic(self) -> None:
        """Legacy no-op: kept for API compatibility.

        Previously tore down the PortAudio input stream via `self._mic.stop()`.
        That was heavy, slow, and anything buffered before the stop leaked into
        the next callback as "user speech". We now gate transcripts via
        `_is_gated()` instead and leave the stream running.
        """
        if self._mic_pause_cb is None:
            return
        try:
            self._mic_pause_cb()
        except Exception as e:
            print(f"  ... mic pause error: {e}", file=sys.stderr)

    def _resume_mic(self) -> None:
        if self._mic_resume_cb is None:
            return
        try:
            self._mic_resume_cb()
        except Exception as e:
            print(f"  ... mic resume error: {e}", file=sys.stderr)

    _SENTENCE_ENDS = ".?!\n"

    def _begin_speaking(self) -> None:
        """Enter speaking state: gate mic so TTS can't self-echo."""
        self._speaking = True
        self._pause_mic()

    def _end_speaking(self, full_text: str) -> None:
        """Leave speaking state after the TTS queue has drained."""
        try:
            self._echo_words = set(
                self._strip_punctuation(full_text).lower().split()
            )
            time.sleep(self._ECHO_COOLDOWN)
            self._echo_suppress_until = time.time() + self._ECHO_WINDOW
        finally:
            self._resume_mic()
            self._speaking = False

    def _speak(self, text: str) -> None:
        """Synchronous speak (used by callers outside the LLM stream loop)."""
        self._begin_speaking()
        try:
            self._tts.enqueue(text)
            self._tts.wait_until_idle()
        finally:
            self._end_speaking(text)

    _WAKE_ECHO_WORDS = {
        "yes",
        "how",
        "can",
        "i",
        "help",
        "you",
        "what",
        "do",
        "need",
        "hey",
        "hi",
        "hello",
        "listening",
        "ready",
        "here",
    }

    # Single-token transcripts Moonshine hallucinates on silence/noise.
    # Used to drop ghost 1-word commands without killing legitimate
    # short follow-ups like "what?", "okay", "stop", "yes", "no".
    _FILLERS = {
        "uh", "um", "hmm", "hm", "mm", "mhm", "huh",
        "ah", "er", "eh", "yeah", "yep", "the", "a",
    }

    def _play_wake_sound(self, initial_command: str = "") -> None:
        """Play a random WAV from wake_sounds_dir (non-blocking).

        After playback + cooldown, clears any echo that leaked into
        _command_text, then starts the listen timeout. This guarantees
        the user's actual speech is what gets collected.
        """
        initial_command = initial_command.strip()
        now = time.time()
        if now - self._last_wake_play_time < self._WAKE_DEBOUNCE_SECONDS:
            return

        if not self._wake_sounds_dir or not os.path.isdir(self._wake_sounds_dir):
            self._start_timeout(
                self._listen_timeout if initial_command else self._post_speak_timeout
            )
            return
        wavs = [
            os.path.join(self._wake_sounds_dir, f)
            for f in os.listdir(self._wake_sounds_dir)
            if f.lower().endswith(".wav")
        ]
        if not wavs:
            self._start_timeout(
                self._listen_timeout if initial_command else self._listen_timeout * 3
            )
            return
        self._speaking = True
        self._echo_words = self._WAKE_ECHO_WORDS.copy()
        self._echo_suppress_until = time.time() + self._ECHO_WINDOW
        self._last_wake_play_time = now
        self._pause_mic()

        def _play():
            try:
                with self._wake_play_lock:
                    sr, audio = _wavfile().read(random.choice(wavs))
                    _play_audio_stable(audio, sr)
                    time.sleep(self._ECHO_COOLDOWN)
            except Exception as e:
                print(f"  Wake sound error: {e}", file=sys.stderr)
            finally:
                self._resume_mic()
                self._speaking = False
                self._command_text = initial_command
                self._start_timeout(
                    self._listen_timeout
                    if initial_command
                    else self._post_speak_timeout
                )

        threading.Thread(target=_play, daemon=True).start()

    def _is_echo(self, text: str) -> bool:
        """Content-based check: is this text just the mic hearing our own TTS?"""
        if time.time() > self._echo_suppress_until:
            return False
        if not self._echo_words:
            return False
        words = set(self._strip_punctuation(text).lower().split())
        if not words:
            return False
        overlap = len(words & self._echo_words) / len(words)
        return (
            overlap > 0.6
        )  # Raised from 0.4 to reduce false positives on short phrases

    # -- Wake word detection --

    @staticmethod
    def _strip_punctuation(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)

    def _detect_wake(self, text: str) -> Optional[str]:
        """Check for wake word and return the text AFTER it, or None.

        Strips punctuation and lowercases both sides so that
        "Hey, rocket!" matches wake word "hey rocket" or "hey, rocket".

        Uses word-boundary-aware matching to avoid false positives
        (e.g. "rocket launcher" won't trigger on wake word "rocket").
        """
        clean = self._strip_punctuation(text).lower()
        wake_clean = self._strip_punctuation(self._wake_word).lower()

        # Word-boundary-aware token match: wake tokens must appear as
        # whole words in sequence within the text tokens
        wake_token_list = wake_clean.split()
        text_tokens = clean.split()
        wi = 0
        last_match_idx = -1
        for ti, token in enumerate(text_tokens):
            if wi < len(wake_token_list) and token == wake_token_list[wi]:
                wi += 1
                last_match_idx = ti
        if wi == len(wake_token_list):
            # Return text after the last matched wake token
            remaining_tokens = text_tokens[last_match_idx + 1 :]
            return " ".join(remaining_tokens)
        return None

    # -- Silence timeout management --

    def _current_silence_timeout(self) -> float:
        """Adaptive silence window: longer once the user is mid-sentence.

        Short commands commit fast; conversational utterances get a longer
        pause tolerance so mid-sentence thinking doesn't trigger early commit.
        """
        word_count = len(self._command_text.split())
        if word_count > 3:
            return self._listen_timeout_long
        return self._listen_timeout

    def _start_timeout(self, duration: float | None = None) -> None:
        self._cancel_timeout()
        # Hard cap: if we've been listening longer than max_utterance, commit now.
        if (
            self._state == self.LISTENING
            and self._listen_started_monotonic > 0
            and time.monotonic() - self._listen_started_monotonic
            >= self._max_utterance
        ):
            self._on_timeout()
            return
        if duration is None:
            timeout = self._current_silence_timeout()
        else:
            timeout = duration
        self._timeout_timer = threading.Timer(timeout, self._on_timeout)
        self._timeout_timer.daemon = True
        self._timeout_timer.start()

    def _cancel_timeout(self) -> None:
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None

    def _on_timeout(self) -> None:
        """Silence timer fired — send whatever we collected to the LLM."""
        if self._state != self.LISTENING:
            return
        # Prefer finalized text from on_line_completed; fall back to the
        # last partial transcript seen on-screen if Moonshine hasn't
        # finalized the line yet when silence fires.
        command = self._command_text.strip()
        if not command and self._last_partial_raw:
            command = self._last_partial_raw
        elapsed = (
            time.monotonic() - self._listen_started_monotonic
            if self._listen_started_monotonic > 0
            else 0.0
        )
        self._state = self.IDLE
        self._command_text = ""
        self._last_stable_text = ""
        self._last_partial_raw = ""
        self._listen_started_monotonic = 0.0
        if self._processing:
            # Previous turn is still running; drop whatever accumulated.
            return
        # Filter out Moonshine's filler-word hallucinations on silence/noise
        # without rejecting legitimate short follow-ups like "what?", "why?",
        # "okay", "stop", "yes", "no".
        if command:
            normalized = self._strip_punctuation(command).lower().strip()
            tokens = normalized.split()
            if len(tokens) == 1 and (
                tokens[0] in self._FILLERS or len(tokens[0]) < 2
            ):
                print(f"  ... ignoring filler '{command}'")
                return
        if command:
            # Gate the mic BEFORE spawning the thread — otherwise Moonshine's
            # partial/final callbacks can fire in the scheduling gap between
            # `Thread.start()` and the thread body setting `_processing=True`,
            # producing a false second LLM turn on top of the first.
            self._processing = True
            self._stop_requested.clear()
            self._audio_active.set()
            threading.Thread(
                target=self._handle_llm_query,
                args=(command,),
                daemon=True,
            ).start()
        else:
            print("  ... no command heard, going back to sleep.")

    def _is_stop_intent(self, text: str) -> bool:
        normalized = self._strip_punctuation(text).lower().strip()
        return normalized in self._stop_phrases

    def _request_stop(self) -> None:
        """Signal the active LLM/TTS turn to bail out as soon as possible."""
        print("  ✋ stop heard — interrupting.")
        self._stop_requested.set()
        try:
            self._tts.drain()
        except Exception:
            pass

    # -- TranscriptEventListener callbacks --

    def on_line_started(self, event):
        self._last_text_length = 0

    def on_line_text_changed(self, event):
        if self._is_gated() or self._is_echo(event.line.text):
            return
        text = event.line.text
        if self._state == self.LISTENING:
            # Only reset the silence timer when the transcript actually GREW
            # with new content. Moonshine re-emits partial refinements even
            # during silence/noise — resetting on every event keeps the mic
            # open forever. Stability endpointing fixes that.
            normalized = self._strip_punctuation(text).lower().strip()
            if normalized and len(normalized) > len(self._last_stable_text):
                self._last_stable_text = normalized
                self._last_partial_raw = text.strip()
                self._start_timeout()
            display = f"  👂 {text}"
        else:
            display = f"  🎙  {text}"
        print(f"\r{display}", end="", flush=True)
        if len(display) < self._last_text_length:
            print(" " * (self._last_text_length - len(display)), end="", flush=True)
        self._last_text_length = len(display)

    def on_line_completed(self, event):
        text = event.line.text.strip()
        if not text:
            return

        # Suppress self-heard TTS and any stray transcripts while a turn
        # is in flight (LLM streaming + TTS + cooldown). Intents still fire
        # via IntentRecognizer, which is registered separately.
        if self._is_gated() or self._is_echo(text):
            # Allow a literal stop-intent to still interrupt — this is the
            # only intentional barge-in channel.
            if self._audio_active.is_set() and self._is_stop_intent(text):
                self._request_stop()
            return

        print(f"\r  🎙  {text}" + " " * max(0, self._last_text_length - len(text) - 6))
        self._last_text_length = 0

        # Skip lines already handled by IntentRecognizer
        if text in self._handled_utterances:
            self._handled_utterances.discard(text)
            return

        # --- State machine ---

        if self._state == self.IDLE:
            tail = self._detect_wake(text)
            if tail is None:
                return  # Not addressed to us — ignore background speech
            initial_command = tail.strip()
            print("  👂 Yes, how can I help you?")
            self._state = self.LISTENING
            self._command_text = initial_command
            self._last_stable_text = self._strip_punctuation(initial_command).lower().strip()
            self._last_partial_raw = initial_command
            self._listen_started_monotonic = time.monotonic()
            self._play_wake_sound(initial_command)

        elif self._state == self.LISTENING:
            if self._command_text:
                self._command_text += " " + text
            else:
                self._command_text = text
            normalized = self._strip_punctuation(self._command_text).lower().strip()
            if len(normalized) > len(self._last_stable_text):
                self._last_stable_text = normalized
            self._start_timeout()

    # -- LLM query on background thread --

    def _handle_llm_query(self, text: str) -> None:
        # `_processing` and `_audio_active` were set by `_on_timeout` before
        # this thread was spawned, so callbacks are already gated. We just
        # need to keep them gated through the whole LLM+TTS span and clear
        # on exit.
        speaking_started = False
        full_response = ""
        try:
            self._conversation.append({"role": "user", "content": text})

            print(f"  🗣  You: {text}")
            print("  🤖 ", end="", flush=True)

            buf = ""
            full_chunks: list[str] = []
            for chunk in self._llm.stream_chat(self._conversation):
                if self._stop_requested.is_set():
                    break
                print(chunk, end="", flush=True)
                full_chunks.append(chunk)
                buf += chunk
                # Flush every completed sentence to the TTS worker so
                # playback of sentence N overlaps with generation of N+1.
                while True:
                    idx = -1
                    for ender in self._SENTENCE_ENDS:
                        pos = buf.find(ender)
                        if pos != -1 and (idx == -1 or pos < idx):
                            idx = pos
                    if idx < 0:
                        break
                    sentence, buf = buf[: idx + 1].strip(), buf[idx + 1 :]
                    if sentence:
                        if not speaking_started:
                            self._begin_speaking()
                            speaking_started = True
                        self._tts.enqueue(sentence)
            print()
            # Flush the tail (any trailing text without a final punctuation).
            tail = buf.strip()
            if tail and not self._stop_requested.is_set():
                if not speaking_started:
                    self._begin_speaking()
                    speaking_started = True
                self._tts.enqueue(tail)

            full_response = "".join(full_chunks)
            self._conversation.append({"role": "assistant", "content": full_response})

            # Trim conversation to rolling window to prevent unbounded memory growth
            if len(self._conversation) > self._max_conversation_length:
                self._conversation = self._conversation[
                    -self._max_conversation_length :
                ]

            if self._stop_requested.is_set():
                self._tts.drain()
            elif speaking_started:
                # Pass our stop flag so a mid-sentence "stop" utterance that
                # fires in the callback while we're waiting is honoured
                # promptly instead of after the current sentence finishes.
                self._tts.wait_until_idle(stop_flag=self._stop_requested)
        except Exception as e:
            print(f"\n  ... LLM error: {e}", file=sys.stderr)
        finally:
            if speaking_started:
                self._end_speaking(full_response)
            # Post-turn time cooldown — mirrors box's 500 ms mic_gate_until
            # after each LLM response. Absorbs any speaker echo still in
            # PortAudio's buffer after TTS playback stops.
            self._mic_gate_until = time.time() + self._POST_TURN_GATE
            self._audio_active.clear()
            self._processing = False
            self._stop_requested.clear()
            self._state = self.LISTENING
            self._command_text = ""
            self._last_stable_text = ""
            self._last_partial_raw = ""
            self._listen_started_monotonic = time.monotonic()
            self._start_timeout(self._post_speak_timeout)
            print("  👂 Listening for follow-up... (or stay silent to end)")


# ─────────────────────────────────────────────────────────────
# Layer 4: Voice Assistant (Orchestrator)
# ─────────────────────────────────────────────────────────────

ROBOT_INTENTS = {
    "move forward": "forward",
    "move backward": "backward",
    "turn left": "left",
    "turn right": "right",
    "stop moving": "stop",
}


class VoiceAssistant:
    """Wires together STT, intent recognition, LLM, robot control, and TTS."""

    def __init__(
        self,
        llm: LLMClient,
        robot: BaseRobotController,
        tts: BaseTTSEngine,
        language: str = "en",
        stt_model_arch=None,
        embedding_model: str = "embeddinggemma-300m",
        embedding_quantization: str = "q4",
        intent_threshold: float = 0.65,
        intents: Optional[dict[str, str]] = None,
        wake_word: str = "hey",
        listen_timeout: float = 0.5,
        listen_timeout_long: float = 1.0,
        max_utterance: float = 15.0,
        wake_sounds_dir: Optional[str] = None,
        mic_device: Optional[int] = None,
        mic_channels: int = 1,
        mic_blocksize: int = 2048,
        system_prompt: str = (
            "You are a helpful voice assistant running on a Jetson-powered robot. "
            "Keep responses to 1-3 sentences — they will be spoken aloud via TTS."
        ),
    ):
        self._robot = robot
        self._tts = tts

        # Report which PortAudio device we're about to hand to Moonshine.
        # On Jetson the ALSA `default` is routed through pulse, which often
        # picks an empty source and makes walle feel "deaf". Accepting an
        # explicit device index (see `python -m sounddevice` for the list)
        # bypasses pulse entirely.
        if mic_device is not None:
            try:
                sd_mod = _sd()
                info = sd_mod.query_devices(mic_device)
                print(
                    f"Mic: device {mic_device} "
                    f"({info['name']}, {info['max_input_channels']} in)",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"  ... mic_device {mic_device} lookup failed: {e}", file=sys.stderr)

        mv = _moonshine()
        if stt_model_arch is None:
            # Tiny is the realtime-safe default on Jetson CPU; medium causes
            # input-overflow because ONNX inference lags the audio stream.
            stt_model_arch = mv.ModelArch.TINY_STREAMING

        # -- STT model --
        print("Loading STT model...", file=sys.stderr)
        model_path, model_arch = mv.get_model_for_language(language, stt_model_arch)
        # Larger blocksize reduces callback pressure → fewer input-overflows
        # when LLM + TTS + STT all hit CPU at once. Explicit channels=1 keeps
        # PortAudio from delivering 6 raw channels from the ReSpeaker array.
        mic_kwargs: dict = {
            "model_path": model_path,
            "model_arch": model_arch,
            "channels": mic_channels,
            "blocksize": mic_blocksize,
        }
        if mic_device is not None:
            mic_kwargs["device"] = mic_device
        self._mic = mv.MicTranscriber(**mic_kwargs)

        # -- Intent recognizer --
        print("Loading embedding model for intent recognition...", file=sys.stderr)
        emb_path, emb_arch = mv.get_embedding_model(
            embedding_model, embedding_quantization
        )
        self._intent_recognizer = mv.IntentRecognizer(
            model_path=emb_path,
            model_arch=emb_arch,
            model_variant=embedding_quantization,
            threshold=intent_threshold,
        )

        # -- Speech router (wake-word-gated LLM fallback) --
        self._router = SpeechRouter(
            llm=llm,
            tts=tts,
            system_prompt=system_prompt,
            wake_word=wake_word,
            listen_timeout=listen_timeout,
            listen_timeout_long=listen_timeout_long,
            max_utterance=max_utterance,
            wake_sounds_dir=wake_sounds_dir,
            # Intentionally NOT passing mic_pause/mic_resume. Previously we
            # called self._mic.stop()/start() each turn to gate the mic, which
            # tore down the PortAudio input stream and leaked buffered audio
            # into the next callback. We now gate transcripts via
            # SpeechRouter._is_gated() and leave the stream running — same
            # pattern box uses for its `audio_tracker.active` gate.
        )

        # -- Register robot intents --
        self._intents = intents or ROBOT_INTENTS
        for phrase, action in self._intents.items():
            self._intent_recognizer.register_intent(
                phrase,
                self._make_intent_handler(action),
            )

        # Add listeners in order: intent recognizer first, then router
        self._mic.add_listener(self._intent_recognizer)
        self._mic.add_listener(self._router)

    def _make_intent_handler(self, action: str):
        """Create a closure that handles a matched intent.

        If the utterance is actually the wake word, let the SpeechRouter
        handle it instead of executing a robot command.
        """

        def handler(trigger: str, utterance: str, similarity: float):
            if self._router._detect_wake(utterance) is not None:
                return
            self._router.mark_handled(utterance)
            self._robot.execute(action, utterance, similarity)

        return handler

    def run(self) -> None:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print("  Voice Assistant", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
        print(f"  Robot commands ({len(self._intents)}):", file=sys.stderr)
        for phrase, action in self._intents.items():
            print(f'    • "{phrase}" → {action}', file=sys.stderr)
        print(
            f'  Wake word: "{self._router._wake_word}" → then speak your question for LLM',
            file=sys.stderr,
        )
        print(f"  Silence timeout: {self._router._listen_timeout}s", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
        print("  Press Ctrl+C to stop.\n", file=sys.stderr)

        self._mic.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n  Shutting down...")
        finally:
            self._mic.stop()
            self._mic.close()
            self._intent_recognizer.close()
            self._robot.close()


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────


def _parse_device(value: str):
    """--mic-device / --speaker-device accept either a numeric PortAudio
    index or a substring of a device name. Name matching is resolved at
    startup against the live device list, so it survives USB re-enumeration
    that would shift numeric indices between runs."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value  # resolved against sd.query_devices() later


def _resolve_device(value, kind: str) -> Optional[int]:
    """Turn a --*-device value (int or name substring) into an index."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    needle = value.lower()
    try:
        devs = _sd().query_devices()
    except Exception as e:
        print(f"  ... device query failed: {e}", file=sys.stderr)
        return None
    want_in = kind == "input"
    for idx, d in enumerate(devs):
        chans = d["max_input_channels"] if want_in else d["max_output_channels"]
        if chans <= 0:
            continue
        if needle in d["name"].lower():
            return idx
    print(
        f"  ... no {kind} device matching '{value}' — falling back to default",
        file=sys.stderr,
    )
    return None


def parse_args():
    p = argparse.ArgumentParser(
        description="Voice Assistant: Moonshine STT + Intent + LLM"
    )

    p.add_argument("--language", default="en", help="STT language (default: en)")
    p.add_argument(
        "--stt-model",
        default="tiny-streaming",
        choices=STT_MODEL_NAMES,
        help="Moonshine STT model (default: tiny-streaming — medium-streaming "
             "causes input-overflow on Jetson CPU).",
    )
    p.add_argument(
        "--wake-word",
        default="hey",
        help="Wake phrase to activate LLM queries (default: 'hey')",
    )
    p.add_argument(
        "--listen-timeout",
        type=float,
        default=0.5,
        help="Short-command silence window in seconds (default: 0.5)",
    )
    p.add_argument(
        "--listen-timeout-long",
        type=float,
        default=1.0,
        help="Conversational silence window once the user is mid-sentence (default: 1.0)",
    )
    p.add_argument(
        "--max-utterance",
        type=float,
        default=15.0,
        help="Hard upper bound on a single listen window in seconds (default: 15.0)",
    )
    p.add_argument(
        "--wake-sounds-dir",
        default=None,
        help="Directory with .wav files to play on wake (default: wake_up_sounds/ next to script)",
    )
    p.add_argument(
        "--mic-device",
        type=_parse_device,
        default=None,
        help="PortAudio input device — integer index OR substring of a device "
             "name (see `python -m sounddevice`). Prefer the name: numeric "
             "indices shift between runs when USB devices come and go.",
    )
    p.add_argument(
        "--mic-channels",
        type=int,
        default=1,
        help="Mic input channels (default: 1). The ReSpeaker array is 6-ch "
             "but Moonshine only needs mono — leaving this at 1 avoids pulling "
             "5 channels of audio we'd throw away.",
    )
    p.add_argument(
        "--speaker-device",
        type=_parse_device,
        default=None,
        help="PortAudio output device index for TTS / wake sounds (see "
             "`python -m sounddevice`). Set this when the ALSA default routes "
             "to HDMI instead of the USB speaker.",
    )
    p.add_argument(
        "--speaker-rate",
        type=int,
        default=None,
        help="Resample all playback audio to this Hz before sending to "
             "--speaker-device. Needed for USB DACs that only accept a fixed "
             "rate (e.g. UACDemoV1.0 = 48000).",
    )
    p.add_argument(
        "--speaker-channels",
        type=int,
        default=None,
        help="Expand mono audio to this many channels before playback. "
             "Set to 2 for USB DACs that only accept stereo.",
    )
    p.add_argument(
        "--mic-blocksize",
        type=int,
        default=2048,
        help="PortAudio blocksize in samples (default: 2048 = 128 ms @ 16 kHz). "
             "Raise to 4096 if you still see 'MicTranscriber: input overflow'.",
    )
    p.add_argument(
        "--intent-threshold",
        type=float,
        default=0.65,
        help="Intent match confidence threshold 0-1 (default: 0.65)",
    )
    p.add_argument("--embedding-model", default="embeddinggemma-300m")
    p.add_argument("--embedding-quantization", default="q4", help="q4, q8, fp16, fp32")

    tts_group = p.add_argument_group("TTS")
    tts_group.add_argument(
        "--no-tts", action="store_true", help="Disable audio TTS (console-only)"
    )
    tts_group.add_argument(
        "--tts-url",
        default="http://localhost:59125",
        help="Mimic3 TTS server URL (default: http://localhost:59125)",
    )
    tts_group.add_argument(
        "--tts-voice",
        default="en_US/vctk_low#p236",
        help="TTS voice (default: en_US/vctk_low#p236)",
    )

    robot_group = p.add_argument_group("Robot")
    robot_group.add_argument(
        "--serial-port",
        default=None,
        help="Arduino serial port (e.g. /dev/ttyACM0). "
             "Omit to run robot actions in simulation (stub) mode.",
    )
    robot_group.add_argument(
        "--baud-rate",
        type=int,
        default=115200,
        help="Arduino serial baud rate (default: 115200, must match firmware).",
    )

    llm_group = p.add_argument_group("LLM")
    llm_group.add_argument(
        "--use-ollama", action="store_true", help="Use real Ollama instead of mock LLM"
    )
    llm_group.add_argument("--ollama-url", default="http://localhost:11434")
    llm_group.add_argument("--ollama-model", default="qwen2.5:3b")

    return p.parse_args()


def main():
    args = parse_args()

    args.mic_device = _resolve_device(args.mic_device, "input")
    args.speaker_device = _resolve_device(args.speaker_device, "output")

    set_output_device(
        args.speaker_device,
        rate=args.speaker_rate,
        channels=args.speaker_channels,
    )
    if args.speaker_device is not None:
        extras = []
        if args.speaker_rate is not None:
            extras.append(f"{args.speaker_rate} Hz")
        if args.speaker_channels is not None:
            extras.append(f"{args.speaker_channels}ch")
        suffix = f" ({', '.join(extras)})" if extras else ""
        print(
            f"Speaker: PortAudio device {args.speaker_device}{suffix}",
            file=sys.stderr,
        )

    # -- Select LLM backend --
    if args.use_ollama:
        from walle.voice.llm_client import OllamaLLMClient

        llm = OllamaLLMClient(base_url=args.ollama_url, model=args.ollama_model)
        print(f"LLM: Ollama ({args.ollama_model})", file=sys.stderr)
    else:
        llm = MockLLMClient(token_delay=0.05)
        print("LLM: Mock (use --use-ollama for real)", file=sys.stderr)

    if args.serial_port:
        from walle.serial_manager import SerialManager
        from walle.tools.robot.executor import RobotControlExecutor

        serial_mgr = SerialManager(port=args.serial_port, baud_rate=args.baud_rate)
        robot_exec = RobotControlExecutor(serial_manager=serial_mgr)
        robot = SerialRobotController(serial_mgr, robot_exec)
        mode = "simulation" if serial_mgr.simulation else "connected"
        print(
            f"Robot: Arduino {args.serial_port} @ {args.baud_rate} ({mode})",
            file=sys.stderr,
        )
    else:
        robot = StubRobotController()
        print("Robot: stub (no --serial-port)", file=sys.stderr)

    if args.no_tts:
        tts = ConsoleTTSEngine()
        print("TTS: Console only (use without --no-tts for audio)", file=sys.stderr)
    else:
        tts = Mimic3TTSEngine(url=args.tts_url, voice=args.tts_voice)
        print(
            f"TTS: Mimic3 at {args.tts_url} (voice: {args.tts_voice})", file=sys.stderr
        )

    stt_arch = _stt_model_choices()[args.stt_model]

    wake_sounds = args.wake_sounds_dir
    if wake_sounds is None:
        default_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "wake_up_sounds"
        )
        if os.path.isdir(default_dir):
            wake_sounds = default_dir
    if wake_sounds:
        n = len([f for f in os.listdir(wake_sounds) if f.endswith(".wav")])
        print(f"Wake sounds: {wake_sounds} ({n} files)", file=sys.stderr)

    assistant = VoiceAssistant(
        llm=llm,
        robot=robot,
        tts=tts,
        language=args.language,
        stt_model_arch=stt_arch,
        embedding_model=args.embedding_model,
        embedding_quantization=args.embedding_quantization,
        intent_threshold=args.intent_threshold,
        wake_word=args.wake_word,
        listen_timeout=args.listen_timeout,
        listen_timeout_long=args.listen_timeout_long,
        max_utterance=args.max_utterance,
        wake_sounds_dir=wake_sounds,
        mic_device=args.mic_device,
        mic_channels=args.mic_channels,
        mic_blocksize=args.mic_blocksize,
    )
    assistant.run()


if __name__ == "__main__":
    main()

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


def _to_float32_audio(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return np.ascontiguousarray(audio)


def _play_audio_stable(
    audio: np.ndarray, sample_rate: int, blocking: bool = True
) -> None:
    audio = _to_float32_audio(audio)
    _sd().stop()
    _sd().play(
        audio,
        sample_rate,
        blocking=blocking,
        latency=AUDIO_LATENCY,
        blocksize=AUDIO_BLOCKSIZE,
    )


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


# ─────────────────────────────────────────────────────────────
# Layer 2: TTS Engine
# ─────────────────────────────────────────────────────────────


class BaseTTSEngine(ABC):
    """Interface for text-to-speech output."""

    @abstractmethod
    def speak(self, text: str) -> None: ...


class ConsoleTTSEngine(BaseTTSEngine):
    """Prints to console instead of speaking."""

    def speak(self, text: str) -> None:
        print(f"  [TTS] {text}")


class Mimic3TTSEngine(BaseTTSEngine):
    """Speaks aloud via a Mimic3 HTTP server, matching rocket_assistant.py."""

    def __init__(
        self,
        url: str = "http://localhost:59125",
        voice: str = "en_UK/apope_low",
    ):
        self._url = url
        self._voice = voice

    def speak(self, text: str) -> None:
        try:
            resp = requests.get(
                f"{self._url}/api/tts",
                params={"text": text, "voice": self._voice},
                timeout=30,
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
            _play_audio_stable(audio, sr)
        except requests.exceptions.ConnectionError:
            print(
                f"  TTS error: cannot connect to Mimic3 at {self._url} — falling back to console",
                file=sys.stderr,
            )
            print(f"  [TTS-fallback] {text}")
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
        wake_word: str = "hey robot",
        listen_timeout: float = 1.0,
        listen_timeout_long: float = 2.0,
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
        self._stop_requested = False
        self._speaking = False
        self._last_text_length = 0

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

    _ECHO_COOLDOWN = 2.0
    _ECHO_WINDOW = (
        4.0  # seconds after TTS to keep content-based filtering (reduced from 8s)
    )
    _WAKE_DEBOUNCE_SECONDS = 1.5

    @property
    def _post_speak_timeout(self) -> float:
        """Timeout after speaking — must exceed echo window to avoid suppressing user's first words."""
        return max(self._listen_timeout * 3, self._ECHO_WINDOW + 1)

    def _pause_mic(self) -> None:
        """Hard-gate the mic: stop audio capture so TTS can't self-echo."""
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

    def _speak(self, text: str) -> None:
        """Speak and block. Hard-gates mic + sets echo suppression as defence-in-depth."""
        self._speaking = True
        self._echo_words = set(self._strip_punctuation(text).lower().split())
        self._pause_mic()
        try:
            self._tts.speak(text)
            time.sleep(self._ECHO_COOLDOWN)
            self._echo_suppress_until = time.time() + self._ECHO_WINDOW
        finally:
            self._resume_mic()
            self._speaking = False

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
        # Min-growth guard: discard tiny fragments from early finalisation.
        if command and len(command.split()) < 2 and elapsed < 0.8:
            print("  ... heard too little, ignoring.")
            return
        if command:
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
        self._stop_requested = True
        try:
            _sd().stop()
        except Exception:
            pass

    # -- TranscriptEventListener callbacks --

    def on_line_started(self, event):
        self._last_text_length = 0

    def on_line_text_changed(self, event):
        if self._processing or self._speaking or self._is_echo(event.line.text):
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
        if self._processing or self._speaking or self._is_echo(text):
            # Allow a literal stop-intent to still interrupt
            if self._processing and self._is_stop_intent(text):
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
        self._processing = True
        self._stop_requested = False
        try:
            self._conversation.append({"role": "user", "content": text})

            print(f"  🗣  You: {text}")
            print("  🤖 ", end="", flush=True)
            chunks: list[str] = []
            for chunk in self._llm.stream_chat(self._conversation):
                if self._stop_requested:
                    break
                print(chunk, end="", flush=True)
                chunks.append(chunk)
            print()
            full_response = "".join(chunks)

            self._conversation.append({"role": "assistant", "content": full_response})

            # Trim conversation to rolling window to prevent unbounded memory growth
            if len(self._conversation) > self._max_conversation_length:
                self._conversation = self._conversation[
                    -self._max_conversation_length :
                ]

            if not self._stop_requested and full_response:
                self._speak(full_response)
        except Exception as e:
            print(f"\n  ... LLM error: {e}", file=sys.stderr)
        finally:
            self._processing = False
            self._stop_requested = False
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
    "wave hello": "wave",
    "dance": "dance",
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
        wake_word: str = "hey robot",
        listen_timeout: float = 1.0,
        listen_timeout_long: float = 2.0,
        max_utterance: float = 15.0,
        wake_sounds_dir: Optional[str] = None,
        system_prompt: str = (
            "You are a helpful voice assistant running on a Jetson-powered robot. "
            "Keep responses to 1-3 sentences — they will be spoken aloud via TTS."
        ),
    ):
        self._robot = robot
        self._tts = tts

        mv = _moonshine()
        if stt_model_arch is None:
            stt_model_arch = mv.ModelArch.MEDIUM_STREAMING

        # -- STT model --
        print("Loading STT model...", file=sys.stderr)
        model_path, model_arch = mv.get_model_for_language(language, stt_model_arch)
        self._mic = mv.MicTranscriber(model_path=model_path, model_arch=model_arch)

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
            mic_pause=self._mic.stop,
            mic_resume=self._mic.start,
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Voice Assistant: Moonshine STT + Intent + LLM"
    )

    p.add_argument("--language", default="en", help="STT language (default: en)")
    p.add_argument(
        "--stt-model",
        default="medium-streaming",
        choices=STT_MODEL_NAMES,
        help="Moonshine STT model (default: medium-streaming)",
    )
    p.add_argument(
        "--wake-word",
        default="hey robot",
        help="Wake phrase to activate LLM queries (default: 'hey robot')",
    )
    p.add_argument(
        "--listen-timeout",
        type=float,
        default=1.0,
        help="Short-command silence window in seconds (default: 1.0)",
    )
    p.add_argument(
        "--listen-timeout-long",
        type=float,
        default=2.0,
        help="Conversational silence window once the user is mid-sentence (default: 2.0)",
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
        default="en_UK/apope_low",
        help="TTS voice (default: en_UK/apope_low)",
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

    # -- Select LLM backend --
    if args.use_ollama:
        from walle.voice.llm_client import OllamaLLMClient

        llm = OllamaLLMClient(base_url=args.ollama_url, model=args.ollama_model)
        print(f"LLM: Ollama ({args.ollama_model})", file=sys.stderr)
    else:
        llm = MockLLMClient(token_delay=0.05)
        print("LLM: Mock (use --use-ollama for real)", file=sys.stderr)

    robot = StubRobotController()

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
    )
    assistant.run()


if __name__ == "__main__":
    main()

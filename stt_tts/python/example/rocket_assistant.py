#!/usr/bin/env python3

"""
Rocket Voice Assistant - Vosk STT + Ollama LLM Integration
Usage: python rocket_assistant.py
Say "hey rocket" followed by your question/command
"""

import argparse
import queue
import sys
import json
import time
import os
import random
import threading
import requests
import sounddevice as sd
import numpy as np
import io
from scipy.io import wavfile
from enum import Enum
from vosk import Model, KaldiRecognizer, SetLogLevel
import subprocess

# Disable Vosk debug logs for cleaner output
SetLogLevel(-1)

class State(Enum):
    LISTENING_FOR_WAKE_WORD = 1
    COLLECTING_COMMAND = 2

class RocketAssistant:
    def __init__(self, wake_word="hey rocket", silence_timeout=3.0, 
                 ollama_model="qwen3:1.7b", ollama_url="http://localhost:11434",
                 ollama_timeout=60, ollama_keep_alive="10m",
                 ollama_num_thread=None, ollama_num_batch=512,
                 tts_url="http://localhost:59125", tts_voice="en_UK/apope_low",
                 enable_tts=True, system_prompt=None):
        self.start_time = time.time()
        self.wake_word = wake_word.lower()
        self.silence_timeout = silence_timeout
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.ollama_timeout = ollama_timeout
        self.ollama_keep_alive = ollama_keep_alive
        self.ollama_num_thread = ollama_num_thread or os.cpu_count()
        self.ollama_num_batch = ollama_num_batch
        self.tts_url = tts_url
        self.tts_voice = tts_voice
        self.enable_tts = enable_tts
        
        # Default system prompt for concise responses
        if system_prompt is None:
            self.system_prompt = (
                "You are Rocket, a helpful voice assistant. "
                "Keep your responses SHORT and CONCISE - use only 3-4 sentences maximum. "
                "Be direct and to the point. Avoid lengthy explanations unless specifically asked."
            )
        else:
            self.system_prompt = system_prompt
        
        self.state = State.LISTENING_FOR_WAKE_WORD
        self.command_text = ""
        self._last_chunk = ""
        self.last_speech_time = time.time()
        self.audio_queue = queue.Queue()
        self.awaiting_llm = False
        self.stop_filler_event = threading.Event()
        self.filler_thread = None
        self._last_filler_phrase = None
        self.play_lock = threading.Lock()
        self.rec_lock = threading.Lock()
        self.is_playing_audio = False
        self._playback_generation = 0
        self._drain_thread = None
        self.last_mic_frame_time = None
        self.thinking_phrases_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thinking_phrases")
        self._last_thinking_file = None
        self.wake_sounds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wake_up_sounds")
        self.thinking_phrases = [
            "Yep, let me think about that.",
            "One moment, I remember I read about that yesterday.",
            "I thought you would never ask.",
            "Let me check a couple of things.",
            "Good question. Give me a second.",
            "Working on it.",
            "Alright, thinking it through.",
            "Hold on, connecting the dots.",
            "Let me recall the details.",
        ]
        
        # Initialize Vosk model
        print("🚀 Loading Vosk model...")
        self.model = Model(lang="en-us")
        print("✅ Model loaded!")
        
        # Trigger background warm-up of the LLM to reduce first-token latency
        threading.Thread(target=self._warm_llm_model, daemon=True).start()
        
    def _clean_chunk(self, text: str) -> str:
        t = text.replace(self.wake_word, " ")
        t = " ".join(t.split())
        return t

    def _sanitize_command(self, text: str) -> str:
        # remove wake word anywhere and collapse whitespace
        t = text.replace(self.wake_word, " ")
        t = " ".join(t.split())
        return t

    def audio_callback(self, indata, frames, time_info, status):
        """Called for each audio block from microphone"""
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(bytes(indata))
        self.last_mic_frame_time = time.time()
    
    def query_ollama(self, prompt):
        """Send prompt to Ollama and get response along with timing metadata."""
        req_start = time.time()
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "system": self.system_prompt,
                    "stream": False,
                    "keep_alive": self.ollama_keep_alive,
                    "options": {
                        "num_thread": int(self.ollama_num_thread or 0),
                        "num_batch": int(self.ollama_num_batch),
                        "temperature": 0.2
                    }
                },
                timeout=(5, self.ollama_timeout)
            )
            done_ts = time.time()
            try:
                first_token_ts = req_start + float(response.elapsed.total_seconds())
            except Exception:
                first_token_ts = done_ts
            if response.status_code == 200:
                text = response.json().get("response", "").strip()
            else:
                text = f"Error: Ollama returned status {response.status_code}"
            return text, {"request_start": req_start, "first_token": first_token_ts, "done": done_ts}
        except requests.exceptions.ConnectionError:
            done_ts = time.time()
            return "Error: Cannot connect to Ollama. Is it running? (ollama serve)", {"request_start": req_start, "first_token": done_ts, "done": done_ts}
        except Exception as e:
            done_ts = time.time()
            return f"Error: {str(e)}", {"request_start": req_start, "first_token": done_ts, "done": done_ts}

    def _warm_llm_model(self):
        """Perform a tiny generate to force-load the model into memory."""
        try:
            self.log_event("llm_warm_start")
            _ = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": "ok",
                    "system": self.system_prompt,
                    "stream": False,
                    "keep_alive": self.ollama_keep_alive,
                    "options": {
                        "num_thread": int(self.ollama_num_thread or 0),
                        "num_batch": int(self.ollama_num_batch),
                        "num_predict": 1,
                        "temperature": 0.0
                    }
                },
                timeout=(3, min(15, self.ollama_timeout))
            )
            self.log_event("llm_warm_done")
        except Exception:
            # Warm-up is best-effort
            pass

    def log_event(self, event, ts=None, **fields):
        """Structured event logger with relative timestamp."""
        t = ts if ts is not None else time.time()
        delta = t - self.start_time
        parts = [f"[{delta:8.3f}s] {event}"]
        if fields:
            kv = " ".join(f"{k}={fields[k]}" for k in sorted(fields.keys()))
            parts.append(kv)
        print(" ".join(parts))
    
    def _play_audio(self, audio_array, sample_rate, wait=True):
        """Thread-safe audio playback helper using sounddevice."""
        try:
            # Ensure float32 numpy array in C-order
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.asarray(audio_array)
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32, copy=False)
            audio_array = np.ascontiguousarray(audio_array)

            with self.play_lock:
                # Stop any ongoing playback to avoid overlap
                try:
                    sd.stop()
                except Exception:
                    pass
                # Mark playback started and drain input
                gen = self._on_playback_start()
                sd.play(audio_array, sample_rate)
            if wait:
                sd.wait()
                self._on_playback_end(gen)
            else:
                # Clear flag after estimated duration
                duration = float(len(audio_array)) / float(max(1, sample_rate))
                threading.Timer(duration + 0.05, self._on_playback_end, args=(gen,)).start()
        except Exception as e:
            print(f"⚠️  Audio playback error: {e}", file=sys.stderr)

    def _on_playback_start(self):
        """Set playback flag, reset recognizer, and start draining input queue."""
        self._playback_generation += 1
        gen = self._playback_generation
        self.is_playing_audio = True
        # Reset recognizer partial state if available
        try:
            if hasattr(self, "rec") and self.rec is not None:
                with self.rec_lock:
                    self.rec.Reset()
        except Exception:
            pass
        # Start queue drain thread if not running
        if not self._drain_thread or not self._drain_thread.is_alive():
            self._drain_thread = threading.Thread(target=self._drain_audio_loop, daemon=True)
            self._drain_thread.start()
        return gen

    def _on_playback_end(self, gen):
        """Clear playback flag if generation matches latest."""
        if gen == self._playback_generation:
            self.is_playing_audio = False

    def _drain_audio_loop(self):
        """Continuously drain microphone queue while audio is playing to avoid self-capture."""
        while True:
            if not self.is_playing_audio:
                # quick final drain
                try:
                    while True:
                        self.audio_queue.get_nowait()
                except Exception:
                    pass
                return
            try:
                self.audio_queue.get(timeout=0.05)
            except Exception:
                pass

    def speak_text(self, text):
        """Convert text to speech and play it using Mimic3"""
        if not self.enable_tts:
            return
        
        try:
            self.log_event("tts_request_start", text_len=len(text))
            response = requests.get(
                f"{self.tts_url}/api/tts",
                params={
                    "text": text,
                    "voice": self.tts_voice
                },
                timeout=10
            )
            if response.status_code == 200:
                audio_data = io.BytesIO(response.content)
                sample_rate, audio_array = wavfile.read(audio_data)
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                elif audio_array.dtype == np.int32:
                    audio_array = audio_array.astype(np.float32) / 2147483648.0
                duration = float(len(audio_array)) / float(max(1, sample_rate))
                self.log_event("tts_audio_ready", sample_rate=sample_rate, duration_s=f"{duration:.3f}")
                self.log_event("tts_playback_start", duration_s=f"{duration:.3f}")
                self._play_audio(audio_array, sample_rate, wait=True)
                self.log_event("tts_playback_end")
            else:
                print(f"⚠️  TTS Error: Server returned status {response.status_code}", file=sys.stderr)
        except requests.exceptions.ConnectionError:
            print(f"⚠️  TTS Error: Cannot connect to Mimic3 at {self.tts_url}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  TTS Error: {str(e)}", file=sys.stderr)
    
    def process_final_result(self, text):
        """Process final recognition result"""
        text = text.strip().lower()
        if getattr(self, "awaiting_llm", False) or getattr(self, "is_playing_audio", False):
            return
        
        if self.state == State.LISTENING_FOR_WAKE_WORD:
            # Check for exact wake word match
            if text == self.wake_word or self.wake_word in text:
                print(f"\n🎤 Wake word detected! Listening for your command...")
                self.state = State.COLLECTING_COMMAND
                # If wake word included more words, keep only the tail as initial command
                if self.wake_word in text:
                    tail = text.split(self.wake_word, maxsplit=1)[1].strip()
                    self.command_text = self._clean_chunk(tail)
                else:
                    self.command_text = ""
                self._last_speech_time = time.time()
                self.last_speech_time = time.time()
                self.play_wake_sound()
                
        elif self.state == State.COLLECTING_COMMAND:
            # Collecting the command after wake word
            if text:
                chunk = self._clean_chunk(text)
                if chunk:
                    # Skip duplicates: identical to last chunk or already present at the end
                    if chunk != self._last_chunk and not self.command_text.endswith(chunk):
                        if self.command_text:
                            self.command_text += " " + chunk
                        else:
                            self.command_text = chunk
                        self._last_chunk = chunk
                        self.last_speech_time = time.time()
    
    def check_command_timeout(self):
        """Check if we should send the command to LLM due to silence"""
        if getattr(self, "awaiting_llm", False) or getattr(self, "is_playing_audio", False):
            return
        if self.state == State.COLLECTING_COMMAND:
            silence_duration = time.time() - self.last_speech_time
            
            # Only timeout if we have collected some command text
            # This prevents premature timeout when user is just starting to speak
            if self.command_text.strip() and silence_duration >= self.silence_timeout:
                # Final sanitize before sending to LLM
                self.command_text = self._sanitize_command(self.command_text)
                print(f"\n{'='*60}")
                print(f"🗣️  You: {self.command_text}")
                print(f"{'='*60}")
                print("🤔 Thinking...")
                if self.last_mic_frame_time is not None:
                    self.log_event("stt_last_frame", ts=self.last_mic_frame_time)
                # Start background LLM processing with filler phrases
                self.awaiting_llm = True
                self.start_thinking_filler()
                threading.Thread(target=self._process_command_async, args=(self.command_text,), daemon=True).start()
    
    def run(self, samplerate=16000, device=None):
        """Main loop"""
        print(f"\n{'='*60}")
        print(f"🚀 Rocket Assistant Started!")
        print(f"{'='*60}")
        print(f"Wake word: '{self.wake_word}'")
        print(f"Silence timeout: {self.silence_timeout}s")
        print(f"Ollama model: {self.ollama_model}")
        print(f"TTS: {'Enabled' if self.enable_tts else 'Disabled'} ({self.tts_voice if self.enable_tts else 'N/A'})")
        print(f"{'='*60}\n")
        print("👂 Listening for 'hey rocket'... (Press Ctrl+C to stop)\n")
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code != 200:
                print("⚠️  Warning: Ollama might not be running properly")
        except:
            print("⚠️  Warning: Cannot connect to Ollama. Make sure it's running: 'ollama serve'")
            print()
        
        try:
            with sd.RawInputStream(
                samplerate=samplerate, 
                blocksize=8000,
                device=device,
                dtype="int16", 
                channels=1,
                callback=self.audio_callback
            ):
                self.rec = KaldiRecognizer(self.model, samplerate)
                
                while True:
                    # Get audio data
                    data = self.audio_queue.get()
                    
                    # Process with Vosk
                    if not self.is_playing_audio:
                        try:
                            with self.rec_lock:
                                accepted = self.rec.AcceptWaveform(data)
                                if accepted:
                                    # Final result (end of speech segment)
                                    result = json.loads(self.rec.Result())
                        except Exception as e:
                            print(f"⚠️  STT error: {e}", file=sys.stderr)
                            # Attempt soft recovery by recreating recognizer
                            try:
                                with self.rec_lock:
                                    self.rec = KaldiRecognizer(self.model, samplerate)
                            except Exception:
                                pass
                            continue

                        if 'result' in locals():
                            text = result.get("text", "")
                            if text:
                                self.process_final_result(text)
                    
                    # Check for command timeout
                    self.check_command_timeout()
                    
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)

    def play_wake_sound(self):
        """Play a random wake-up sound from the wake_up_sounds folder (non-blocking)."""
        try:
            if not os.path.isdir(self.wake_sounds_dir):
                print(f"⚠️  Wake sound folder not found: {self.wake_sounds_dir}")
                return
            candidates = [
                os.path.join(self.wake_sounds_dir, f)
                for f in os.listdir(self.wake_sounds_dir)
                if f.lower().endswith((".wav", ".mp3"))
            ]
            if not candidates:
                print("⚠️  No .wav or .mp3 files found in wake_up_sounds folder")
                return
            path = random.choice(candidates)
            sample_rate, audio_array = self._load_audio_any(path)
            # Non-blocking playback
            threading.Thread(target=self._play_audio, args=(audio_array, sample_rate, False), daemon=True).start()
        except Exception as e:
            print(f"⚠️  Wake sound error: {e}", file=sys.stderr)

    def _load_audio_any(self, path):
        """Load WAV or MP3 file and return (sample_rate, float32 mono numpy array)."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".wav":
            sr, arr = wavfile.read(path)
        elif ext == ".mp3":
            # Decode via ffmpeg into WAV on stdout, then read with wavfile
            try:
                proc = subprocess.run(
                    [
                        "ffmpeg", "-v", "error", "-i", path,
                        "-f", "wav", "-ac", "1", "-ar", "22050", "-"
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
            except FileNotFoundError:
                raise RuntimeError("ffmpeg not found. Install ffmpeg to play mp3 wake sounds.")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"ffmpeg failed to decode mp3: {e.stderr.decode(errors='ignore')}")
            bio = io.BytesIO(proc.stdout)
            sr, arr = wavfile.read(bio)
        else:
            raise RuntimeError(f"Unsupported audio format: {ext}")

        # Convert to float32 mono
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        elif arr.dtype == np.int32:
            arr = arr.astype(np.float32) / 2147483648.0
        elif arr.dtype == np.float32:
            arr = arr.astype(np.float32)
        else:
            # Fallback: normalize generically
            arr = arr.astype(np.float32)
            maxv = np.max(np.abs(arr)) or 1.0
            arr = arr / maxv
        return sr, arr

    def start_thinking_filler(self):
        """Start a background thread that speaks random filler phrases until stopped."""
        if not self.enable_tts:
            return
        if self.filler_thread and self.filler_thread.is_alive():
            return
        self.stop_filler_event.clear()
        self.filler_thread = threading.Thread(target=self._thinking_filler_loop, daemon=True)
        self.filler_thread.start()

    def stop_thinking_filler(self):
        """Signal the filler thread to stop after current phrase."""
        self.stop_filler_event.set()

    def _thinking_filler_loop(self):
        """Loop to speak filler phrases until stop event is set."""
        try:
            while not self.stop_filler_event.is_set():
                if not os.path.isdir(self.thinking_phrases_dir):
                    print(f"⚠️  Thinking phrases folder not found: {self.thinking_phrases_dir}")
                    return
                candidates = [
                    os.path.join(self.thinking_phrases_dir, f)
                    for f in os.listdir(self.thinking_phrases_dir)
                    if f.lower().endswith((".wav", ".mp3"))
                ]
                if not candidates:
                    print("⚠️  No .wav or .mp3 files found in thinking_phrases folder")
                    return
                file_path = random.choice(candidates)
                if self._last_thinking_file and len(candidates) > 1:
                    tries = 0
                    while file_path == self._last_thinking_file and tries < 3:
                        file_path = random.choice(candidates)
                        tries += 1
                self._last_thinking_file = file_path
                sr, arr = self._load_audio_any(file_path)
                self._play_audio(arr, sr, wait=True)
                if self.stop_filler_event.is_set():
                    break
                time.sleep(random.uniform(2.0, 3.5))
        except Exception:
            pass

    def _process_command_async(self, text):
        """Background task: query LLM, stop filler, speak result, and reset state."""
        try:
            response, meta = self.query_ollama(text)
            # Log LLM timings
            if isinstance(meta, dict):
                if "request_start" in meta:
                    self.log_event("llm_request_start", ts=meta["request_start"])  # precise request start
                if "first_token" in meta:
                    self.log_event("llm_first_token", ts=meta["first_token"]) 
                if "done" in meta:
                    self.log_event("llm_done", ts=meta["done"]) 
        except Exception as e:
            response = f"Error: {str(e)}"
        finally:
            self.stop_thinking_filler()
        print(f"\n🤖 Rocket: {response}")
        print(f"{'='*60}\n")
        if self.enable_tts:
            print("🔊 Speaking...")
            self.speak_text(response)
        self.state = State.LISTENING_FOR_WAKE_WORD
        self.command_text = ""
        self.awaiting_llm = False
        print("👂 Listening for 'hey rocket'...")

def main():
    parser = argparse.ArgumentParser(
        description="Rocket Voice Assistant - Voice activated AI assistant"
    )
    parser.add_argument(
        "-w", "--wake-word", 
        type=str, 
        default="hey rocket",
        help="Wake word to activate assistant (default: 'hey rocket')"
    )
    parser.add_argument(
        "-t", "--timeout", 
        type=float, 
        default=3.0,
        help="Silence timeout in seconds (default: 3.0)"
    )
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        default="qwen3:1.7b",
        help="Ollama model name (default: qwen3:1.7b)"
    )
    parser.add_argument(
        "-u", "--url", 
        type=str, 
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=60,
        help="Ollama read timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--llm-keep-alive",
        type=str,
        default="10m",
        help="How long Ollama should keep the model loaded (e.g. '10m', '-1')"
    )
    parser.add_argument(
        "--llm-num-thread",
        type=int,
        help="Threads to use for Ollama inference (default: CPU count)"
    )
    parser.add_argument(
        "--llm-num-batch",
        type=int,
        default=512,
        help="Batch size for prompt processing (higher = faster prompt ingest)"
    )
    parser.add_argument(
        "-d", "--device", 
        type=int,
        help="Audio input device ID (use --list-devices to see options)"
    )
    parser.add_argument(
        "-l", "--list-devices", 
        action="store_true",
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "--tts-url", 
        type=str, 
        default="http://localhost:59125",
        help="Mimic3 TTS server URL (default: http://localhost:59125)"
    )
    parser.add_argument(
        "--tts-voice", 
        type=str, 
        default="en_UK/apope_low",
        help="TTS voice (default: en_UK/apope_low)"
    )
    parser.add_argument(
        "--no-tts", 
        action="store_true",
        help="Disable text-to-speech output"
    )
    parser.add_argument(
        "--system-prompt", 
        type=str,
        help="Custom system prompt for the LLM (default: short 3-4 sentence responses)"
    )
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        sys.exit(0)
    
    # Create and run assistant
    assistant = RocketAssistant(
        wake_word=args.wake_word,
        silence_timeout=args.timeout,
        ollama_model=args.model,
        ollama_url=args.url,
        ollama_timeout=args.llm_timeout,
        ollama_keep_alive=args.llm_keep_alive,
        ollama_num_thread=args.llm_num_thread,
        ollama_num_batch=args.llm_num_batch,
        tts_url=args.tts_url,
        tts_voice=args.tts_voice,
        enable_tts=not args.no_tts,
        system_prompt=args.system_prompt
    )
    
    assistant.run(device=args.device)

if __name__ == "__main__":
    main()


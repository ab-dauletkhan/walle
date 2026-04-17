"""Standalone STT pipeline diagnostic.

Runs the *exact same* Moonshine MicTranscriber the voice assistant uses,
with a verbose listener that prints every start / partial / completed
event on arrival. No vision, no TTS, no LLM, no Arduino — so if this
script stays silent, the mic pipeline is broken independent of anything
else in WALL-E.

Usage (on the Jetson):

    # 1. list audio devices so you can pick the right --device
    python scripts/debug_stt.py --list-devices

    # 2. run against the default input for 20 s
    python scripts/debug_stt.py

    # 3. run against an explicit device (skips PulseAudio's "default"
    #    source, which on Jetson is often silent)
    python scripts/debug_stt.py --device 7 --channels 1

Each line of output is tagged with an elapsed-seconds timestamp so you
can correlate "I said hey at t=3s" with "first partial at t=3.4s".
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Optional

from _diag_common import (
    describe_audio_device,
    parse_device,
    print_audio_devices,
    resolve_audio_device,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Moonshine STT pipeline diagnostic")
    p.add_argument("--device", type=parse_device, default=None,
                   help="Input device index or name substring (default: PortAudio default)")
    p.add_argument("--channels", type=int, default=1, help="Mic channels (default 1)")
    p.add_argument("--blocksize", type=int, default=16384,
                   help="PortAudio blocksize (matches assistant.py default)")
    p.add_argument("--seconds", type=float, default=20.0,
                   help="How long to listen before exiting")
    p.add_argument("--language", default="en")
    p.add_argument("--model", default="small-streaming",
                   choices=("tiny-streaming", "small-streaming", "medium-streaming",
                            "tiny", "base"))
    p.add_argument("--list-devices", action="store_true",
                   help="Print PortAudio device table and exit")
    p.add_argument("--probe-level", action="store_true",
                   help="Also open a raw sounddevice stream on the same device "
                        "and print peak RMS every 0.5s so you can confirm audio "
                        "frames are actually flowing")
    return p


def _start_level_probe(device: Optional[int], channels: int, rate: int,
                       stop_event: threading.Event, start_ts: float) -> threading.Thread:
    """Parallel RMS meter. Tells us if PortAudio is delivering audio at all,
    independent of whether Moonshine parses it into text."""
    import numpy as np  # noqa: PLC0415
    import sounddevice as sd  # noqa: PLC0415

    stats = {"peak": 0.0, "rms": 0.0, "blocks": 0}

    def cb(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            _log(start_ts, f"[level-probe] status: {status}")
        mono = np.mean(indata.astype(np.float32), axis=1) if indata.ndim > 1 else indata.astype(np.float32)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        stats["peak"] = max(stats["peak"], peak)
        stats["rms"] = max(stats["rms"], rms)
        stats["blocks"] += 1

    def run():
        try:
            with sd.InputStream(samplerate=rate, channels=channels, device=device,
                                blocksize=2048, dtype="float32", callback=cb):
                while not stop_event.is_set():
                    stop_event.wait(0.5)
                    _log(start_ts,
                         f"[level-probe] blocks={stats['blocks']} "
                         f"peak={stats['peak']:.3f} rms={stats['rms']:.3f}")
                    stats["peak"] = 0.0
                    stats["rms"] = 0.0
        except Exception as e:
            _log(start_ts, f"[level-probe] FAILED: {e!r}")

    t = threading.Thread(target=run, name="level-probe", daemon=True)
    t.start()
    return t


def _log(start_ts: float, msg: str) -> None:
    print(f"[{time.monotonic() - start_ts:6.2f}s] {msg}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    if args.list_devices:
        print_audio_devices()
        return 0

    mic_device = resolve_audio_device(args.device, "input")
    print(f"Selected mic: {describe_audio_device(mic_device, 'input')}", flush=True)
    print(f"Channels={args.channels} blocksize={args.blocksize} duration={args.seconds}s",
          flush=True)

    try:
        import moonshine_voice as mv
    except ImportError as e:
        print(f"moonshine_voice not importable: {e}", file=sys.stderr)
        return 2

    model_arch = {
        "tiny-streaming": mv.ModelArch.TINY_STREAMING,
        "small-streaming": mv.ModelArch.SMALL_STREAMING,
        "medium-streaming": mv.ModelArch.MEDIUM_STREAMING,
        "tiny": mv.ModelArch.TINY,
        "base": mv.ModelArch.BASE,
    }[args.model]

    print(f"Loading model: {args.model}", flush=True)
    model_path, resolved_arch = mv.get_model_for_language(args.language, model_arch)

    mic_kwargs = {
        "model_path": model_path,
        "model_arch": resolved_arch,
        "channels": args.channels,
        "blocksize": args.blocksize,
    }
    if mic_device is not None:
        mic_kwargs["device"] = mic_device

    mic = mv.MicTranscriber(**mic_kwargs)
    print(f"MicTranscriber created. Public attrs: "
          f"{sorted(a for a in dir(mic) if not a.startswith('_'))}", flush=True)

    start_ts = time.monotonic()
    counters = {"started": 0, "changed": 0, "completed": 0}

    class EchoListener(mv.TranscriptEventListener):
        def on_line_started(self, event):
            counters["started"] += 1
            _log(start_ts, f"[EVENT start #{counters['started']}]")

        def on_line_text_changed(self, event):
            counters["changed"] += 1
            text = getattr(getattr(event, "line", None), "text", "")
            _log(start_ts, f"[EVENT partial #{counters['changed']}] {text!r}")

        def on_line_completed(self, event):
            counters["completed"] += 1
            text = getattr(getattr(event, "line", None), "text", "")
            _log(start_ts, f"[EVENT done    #{counters['completed']}] {text!r}")

    mic.add_listener(EchoListener())
    print("Listener attached.", flush=True)

    stop_event = threading.Event()
    level_thread = None
    if args.probe_level:
        level_thread = _start_level_probe(
            mic_device, args.channels, 16000, stop_event, start_ts
        )

    print(f"\n{'=' * 60}", flush=True)
    print(f"Speak now. Listening for {args.seconds:.0f} s.", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    _log(start_ts, "calling mic.start() …")
    try:
        mic.start()
    except Exception as e:
        _log(start_ts, f"mic.start() raised: {e!r}")
        return 3
    _log(start_ts, "mic.start() returned")

    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        time.sleep(1.0)
        _log(start_ts,
             f"[tick] started={counters['started']} partials={counters['changed']} "
             f"completed={counters['completed']}")

    _log(start_ts, "stopping …")
    stop_event.set()
    try:
        mic.stop()
    except Exception as e:
        _log(start_ts, f"mic.stop() raised: {e!r}")
    if level_thread is not None:
        level_thread.join(timeout=2.0)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Summary: started={counters['started']} partials={counters['changed']} "
          f"completed={counters['completed']}", flush=True)
    if counters["started"] == 0 and counters["changed"] == 0 and counters["completed"] == 0:
        print("DIAGNOSIS: Moonshine received no audio frames for the whole window.",
              flush=True)
        print("Next step: re-run with --probe-level. If the level-probe shows non-zero",
              flush=True)
        print("RMS while Moonshine stays silent, the mic is fine and Moonshine itself",
              flush=True)
        print("isn't being driven. If the level-probe ALSO shows 0 RMS, PortAudio's",
              flush=True)
        print("chosen source is a silent/empty one — pass --device <N> with the right",
              flush=True)
        print("index from --list-devices.", flush=True)
    else:
        print("DIAGNOSIS: STT pipeline is alive on this device.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

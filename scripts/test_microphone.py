from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

from _diag_common import (
    describe_audio_device,
    parse_device,
    print_audio_devices,
    resolve_audio_device,
)


def _prepare_output_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_rate: Optional[int],
    target_channels: int,
) -> tuple[np.ndarray, int]:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if target_rate is not None and sample_rate != target_rate:
        from math import gcd

        from scipy.signal import resample_poly  # noqa: PLC0415

        factor = gcd(int(target_rate), int(sample_rate))
        up = int(target_rate) // factor
        down = int(sample_rate) // factor
        audio = resample_poly(audio, up, down).astype(np.float32)
        sample_rate = target_rate

    if audio.ndim == 1 and target_channels >= 2:
        audio = np.repeat(audio[:, None], target_channels, axis=1)

    return np.ascontiguousarray(audio), sample_rate


def _meter_bar(value: float, width: int = 40) -> str:
    filled = max(0, min(width, int(round(value * width * 4))))
    return "#" * filled + "." * (width - filled)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone microphone diagnostic")
    parser.add_argument(
        "--device",
        type=parse_device,
        default=None,
        help="Input device index or name substring",
    )
    parser.add_argument("--rate", type=int, default=16000, help="Capture sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Capture channels")
    parser.add_argument("--seconds", type=float, default=5.0, help="Record duration")
    parser.add_argument("--blocksize", type=int, default=2048, help="PortAudio blocksize")
    parser.add_argument(
        "--save-path",
        default="/tmp/walle-mic-test.wav",
        help="Where to save the recorded WAV",
    )
    parser.add_argument(
        "--playback",
        action="store_true",
        help="Play the recording back after capture",
    )
    parser.add_argument(
        "--speaker-device",
        type=parse_device,
        default=None,
        help="Playback device index or name substring",
    )
    parser.add_argument(
        "--speaker-rate",
        type=int,
        default=48000,
        help="Playback sample rate when --playback is enabled",
    )
    parser.add_argument(
        "--speaker-channels",
        type=int,
        default=2,
        help="Playback channels when --playback is enabled",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List PortAudio devices and exit",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list_devices:
        print_audio_devices()
        return 0

    import sounddevice as sd  # noqa: PLC0415
    from scipy.io import wavfile  # noqa: PLC0415

    mic_device = resolve_audio_device(args.device, "input")
    print(f"Selected microphone: {describe_audio_device(mic_device, 'input')}")

    chunks: list[np.ndarray] = []
    stats = {"peak": 0.0, "rms": 0.0, "blocks": 0}

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"\nPortAudio status: {status}")
        mono = np.mean(indata.astype(np.float32), axis=1)
        chunks.append(mono.copy())
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        stats["peak"] = max(stats["peak"], peak)
        stats["rms"] = max(stats["rms"], rms)
        stats["blocks"] += 1

    deadline = time.monotonic() + args.seconds
    print(f"Recording for {args.seconds:.1f}s. Speak normally into the microphone.")
    with sd.InputStream(
        samplerate=args.rate,
        channels=args.channels,
        blocksize=args.blocksize,
        device=mic_device,
        callback=callback,
        dtype="float32",
    ):
        while time.monotonic() < deadline:
            bar = _meter_bar(stats["peak"])
            print(
                f"\rPeak {stats['peak']:.3f}  RMS {stats['rms']:.3f}  [{bar}]",
                end="",
                flush=True,
            )
            time.sleep(0.15)
    print()

    if not chunks:
        print("No audio blocks were captured.")
        return 1

    audio = np.concatenate(chunks)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    out_path = Path(args.save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out_path, args.rate, np.int16(np.clip(audio, -1.0, 1.0) * 32767))

    print(f"Saved recording to {out_path}")
    print(f"Captured {audio.shape[0]} samples, peak={peak:.3f}, rms={rms:.3f}")
    if peak < 0.01:
        print("Very little audio energy was detected. The mic may be silent or muted.")
    else:
        print("The mic captured non-trivial audio energy.")

    if args.playback:
        speaker_device = resolve_audio_device(args.speaker_device, "output")
        print(f"Selected playback device: {describe_audio_device(speaker_device, 'output')}")
        playback_audio, playback_rate = _prepare_output_audio(
            audio,
            args.rate,
            target_rate=args.speaker_rate,
            target_channels=args.speaker_channels,
        )
        sd.play(
            playback_audio,
            samplerate=playback_rate,
            blocking=True,
            device=speaker_device,
        )
        sd.stop()
        print("Playback finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

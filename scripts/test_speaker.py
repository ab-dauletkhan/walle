from __future__ import annotations

import argparse
import io
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
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
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
    elif audio.ndim == 2 and audio.shape[1] != target_channels:
        if audio.shape[1] == 1 and target_channels >= 2:
            audio = np.repeat(audio, target_channels, axis=1)
        else:
            audio = audio[:, :target_channels]

    return np.ascontiguousarray(audio), sample_rate


def _play(audio: np.ndarray, sample_rate: int, *, device, channels: int) -> None:
    import sounddevice as sd  # noqa: PLC0415

    sd.stop()
    sd.play(audio, samplerate=sample_rate, blocking=True, device=device)
    sd.stop()
    print(
        f"Played {audio.shape[0]} samples at {sample_rate} Hz "
        f"through {describe_audio_device(device, 'output')} ({channels}ch)"
    )


def _build_tone(seconds: float, sample_rate: int, frequency: float, volume: float):
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    tone = np.sin(2.0 * np.pi * frequency * t).astype(np.float32) * volume
    fade_samples = max(1, int(sample_rate * 0.02))
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    tone[:fade_samples] *= ramp
    tone[-fade_samples:] *= ramp[::-1]
    return tone


def _fetch_tts_audio(url: str, text: str, voice: str) -> tuple[np.ndarray, int]:
    import requests  # noqa: PLC0415
    from scipy.io import wavfile  # noqa: PLC0415

    response = requests.get(
        f"{url}/api/tts",
        params={"text": text, "voice": voice},
        timeout=30,
    )
    response.raise_for_status()
    sample_rate, audio = wavfile.read(io.BytesIO(response.content))
    return audio, int(sample_rate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone speaker diagnostic")
    parser.add_argument(
        "--device",
        type=parse_device,
        default=None,
        help="Output device index or name substring",
    )
    parser.add_argument("--rate", type=int, default=48000, help="Playback rate")
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Playback channels (USB DACs usually need 2)",
    )
    parser.add_argument("--seconds", type=float, default=1.0, help="Tone duration")
    parser.add_argument("--frequency", type=float, default=523.25, help="Tone frequency")
    parser.add_argument("--volume", type=float, default=0.25, help="Tone volume 0-1")
    parser.add_argument(
        "--tts-url",
        default=None,
        help="Optional Mimic3 URL. If set, a TTS phrase is also played.",
    )
    parser.add_argument(
        "--tts-text",
        default="Hello. This is the WALL-E speaker test.",
        help="Text to synthesize when --tts-url is set.",
    )
    parser.add_argument(
        "--tts-voice",
        default="en_US/vctk_low#p236",
        help="Mimic3 voice",
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

    device = resolve_audio_device(args.device, "output")
    print(f"Selected speaker: {describe_audio_device(device, 'output')}")

    tone = _build_tone(args.seconds, args.rate, args.frequency, args.volume)
    tone, rate = _prepare_output_audio(
        tone,
        args.rate,
        target_rate=args.rate,
        target_channels=args.channels,
    )
    print("Playing tone test...")
    _play(tone, rate, device=device, channels=args.channels)

    if args.tts_url:
        print(f"Fetching TTS from {args.tts_url} ...")
        audio, sample_rate = _fetch_tts_audio(args.tts_url, args.tts_text, args.tts_voice)
        audio, sample_rate = _prepare_output_audio(
            audio,
            sample_rate,
            target_rate=args.rate,
            target_channels=args.channels,
        )
        print("Playing TTS test...")
        _play(audio, sample_rate, device=device, channels=args.channels)

    print("If you heard the tone (and optional TTS), the speaker path works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

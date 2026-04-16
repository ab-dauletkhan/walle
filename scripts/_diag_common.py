from __future__ import annotations

from typing import Optional


def parse_device(value: Optional[str]):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def query_audio_devices():
    import sounddevice as sd  # noqa: PLC0415

    return sd.query_devices()


def resolve_audio_device(value, kind: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    needle = value.lower()
    want_input = kind == "input"
    for index, device in enumerate(query_audio_devices()):
        channels = (
            device["max_input_channels"]
            if want_input
            else device["max_output_channels"]
        )
        if channels <= 0:
            continue
        if needle in device["name"].lower():
            return index
    raise SystemExit(f"No {kind} audio device matching {value!r}")


def describe_audio_device(index: Optional[int], kind: str) -> str:
    if index is None:
        return f"default {kind}"
    device = query_audio_devices()[index]
    channels = (
        device["max_input_channels"]
        if kind == "input"
        else device["max_output_channels"]
    )
    return f"#{index} {device['name']} ({channels} {kind} channels)"


def print_audio_devices() -> None:
    for index, device in enumerate(query_audio_devices()):
        in_ch = int(device["max_input_channels"])
        out_ch = int(device["max_output_channels"])
        print(f"{index:>2}: in={in_ch:<2} out={out_ch:<2} {device['name']}")


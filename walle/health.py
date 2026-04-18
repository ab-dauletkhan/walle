"""One-shot startup health probes.

Runs once during :func:`walle.startup.main` and prints a compact
``[OK] …`` / ``[--] …`` banner so operators see at a glance whether
the stack came up healthy. Deliberately *not* a running daemon —
that would duplicate the periodic heartbeat logs the voice pipeline
already emits. The user asked for a single startup check: this is it.

Each probe returns ``(ok: bool, detail: str)``. ``detail`` is a short
human-readable string (a path, a device name, an error message).
Probes must not raise; any failure is returned as ``(False, <reason>)``.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
from typing import Callable, List, Optional, Tuple

_log = logging.getLogger("walle.health")


ProbeResult = Tuple[bool, str]
ProbeFn = Callable[[], ProbeResult]


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------


def probe_hwc_socket(path: str) -> ProbeResult:
    """Try a proper IDENT handshake against the HWC UDS.

    Connects as ``IDENT cli`` (the debug/probe identity), waits for
    ``READY``, then cleanly disconnects. Doing a proper handshake
    avoids the ``ERR bad IDENT`` / ``BrokenPipeError`` that a
    raw-connect-then-close triggers on the server side.
    """
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(path)
        s.sendall(b"IDENT cli\n")
        f = s.makefile("rb", buffering=0)
        ready = f.readline().decode("utf-8", errors="ignore").strip()
        s.close()
    except Exception as exc:
        return False, f"{path}: {exc}"
    if ready != "READY":
        return False, f"{path}: expected READY, got {ready!r}"
    return True, path


def probe_ollama(url: str) -> ProbeResult:
    try:
        import requests  # local import — dev hosts might not have it
    except ImportError:
        return False, "requests not installed"
    try:
        r = requests.get(f"{url}/api/tags", timeout=2.0)
    except Exception as exc:
        return False, f"{url}: {exc}"
    if r.status_code != 200:
        return False, f"{url}: HTTP {r.status_code}"
    return True, url


def probe_piper_model(model_path: str) -> ProbeResult:
    if not model_path:
        return False, "no --piper-model given"
    if not os.path.exists(model_path):
        return False, f"missing: {model_path}"
    if not os.path.exists(model_path + ".json"):
        return False, f"missing config: {model_path}.json"
    return True, os.path.basename(model_path)


def probe_amixer() -> ProbeResult:
    """Probe whatever playback control actually exists on this host.

    Calls the same auto-detect helper used by VolumeToolExecutor so
    the banner reports [OK] for hardware that doesn't have ``Master``
    (e.g. the UACDemoV1.0 USB DAC uses ``PCM``).
    """
    try:
        from walle.tools.audio import _detect_mixer_control
    except Exception as exc:
        return False, f"import failed: {exc}"
    control = _detect_mixer_control(None)
    if control is None:
        return False, "no playback control found"
    try:
        r = subprocess.run(
            ["amixer", "-M", "sget", control],
            capture_output=True, text=True, timeout=2.0, check=False,
        )
    except FileNotFoundError:
        return False, "amixer not installed"
    except Exception as exc:
        return False, str(exc)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout).strip() or f"rc={r.returncode}"
    return True, f"amixer {control} OK"


def probe_mic_device(device) -> ProbeResult:
    try:
        import sounddevice as sd
    except Exception as exc:
        return False, f"sounddevice import failed: {exc}"
    try:
        info = sd.query_devices(device if device is not None else sd.default.device[0])
    except Exception as exc:
        return False, str(exc)
    chans = info.get("max_input_channels", 0)
    if chans <= 0:
        return False, f"{info.get('name', '?')} has no input channels"
    return True, f"{info.get('name', '?')} ({chans}ch)"


def probe_speaker_device(device) -> ProbeResult:
    try:
        import sounddevice as sd
    except Exception as exc:
        return False, f"sounddevice import failed: {exc}"
    try:
        info = sd.query_devices(device if device is not None else sd.default.device[1])
    except Exception as exc:
        return False, str(exc)
    chans = info.get("max_output_channels", 0)
    if chans <= 0:
        return False, f"{info.get('name', '?')} has no output channels"
    return True, f"{info.get('name', '?')} ({chans}ch)"


# ---------------------------------------------------------------------------
# Runner + banner formatting
# ---------------------------------------------------------------------------


def run_probes(probes: List[Tuple[str, ProbeFn]]) -> List[Tuple[str, bool, str]]:
    """Execute a list of named probes and return their results.

    Each probe function is wrapped in a try/except so one bad probe can't
    take the banner down. We don't parallelise — the total budget is a
    few seconds of I/O waits, below the human threshold for "slow".
    """
    out: List[Tuple[str, bool, str]] = []
    for name, fn in probes:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"probe raised: {exc}"
        out.append((name, ok, detail))
    return out


def format_banner(
    results: List[Tuple[str, bool, str]], title: str = "WALL-E System Status"
) -> str:
    width = 48
    lines = [f"\n{'=' * width}", f"  {title}", "=" * width]
    for name, ok, detail in results:
        tag = "[OK]" if ok else "[--]"
        lines.append(f"  {name:<14} {detail[:27]:<28} {tag}")
    lines.append("=" * width)
    return "\n".join(lines)

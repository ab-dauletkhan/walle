"""Audio control tools for the voice assistant.

Exposes a single `adjust_volume` LLM tool that shells out to ALSA
(`amixer`) on the Jetson. Persistence works in two layers:

1. ALSA itself remembers volume across the current session. On most
   Linux distros the `alsa-restore` systemd service stores state on
   shutdown and reapplies on boot, so the user's preferred volume
   carries over reboots without anything extra on our side.

2. We also write the last-requested volume to a state file in the
   project root (`.walle_volume.json`). On startup the voice assistant
   reads this and reapplies the volume before the "At your command."
   announcement — belt-and-suspenders for setups where ALSA state
   isn't persisted (no systemd service, or a fresh `sudo` user).

Only one mixer control is touched (`Master` by default), and failures
surface as a returned error string instead of raising — a voice
assistant that refuses every subsequent command after a failed volume
change is a worse user experience than a polite apology.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Optional

_log = logging.getLogger("walle.tools.audio")

_STATE_FILE = os.environ.get(
    "WALLE_VOLUME_STATE", os.path.expanduser("~/.walle_volume.json")
)
# Controls that are commonly present on consumer / USB audio hardware.
# We probe them in order and pick the first that exists. "Master" is
# canonical on most internal sound cards; USB DACs like UACDemoV1.0 only
# expose "PCM" (or sometimes "Speaker"). Override with WALLE_MIXER_CONTROL
# if your device uses something exotic.
_DEFAULT_MIXER_CONTROL = os.environ.get("WALLE_MIXER_CONTROL")
_MIXER_CONTROL_CANDIDATES = [
    "PCM", "Master", "Speaker", "Headphone", "Digital", "Playback",
]
_MIXER_CARD = os.environ.get("WALLE_MIXER_CARD")  # e.g. "1" for hw:1
_VOLUME_STEP_PERCENT = 10
_VOLUME_MIN = 0
_VOLUME_MAX = 100


def _detect_mixer_control(card: Optional[str]) -> Optional[str]:
    """Find a usable amixer simple-control name.

    Returns the first control from ``_MIXER_CONTROL_CANDIDATES`` that
    ``amixer scontrols`` actually reports. Returns ``None`` if amixer
    is unavailable or reports no controls.
    """
    args = ["amixer"]
    if card:
        args += ["-c", card]
    args.append("scontrols")
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=2.0, check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    # Each line looks like: `Simple mixer control 'PCM',0`
    available: set[str] = set()
    for line in proc.stdout.splitlines():
        start = line.find("'")
        end = line.find("'", start + 1)
        if start != -1 and end != -1:
            available.add(line[start + 1:end])
    for cand in _MIXER_CONTROL_CANDIDATES:
        if cand in available:
            return cand
    # Last resort — take whatever is first in the list so we still do
    # *something* when the card uses a non-standard name.
    return next(iter(available), None)


def get_volume_tools() -> list:
    """Tool schema for the voice-level control tool.

    Single tool instead of three so the 3B-class model doesn't have
    to disambiguate between volume_up / volume_down / set_volume —
    it just picks the one tool and supplies `action`.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "adjust_volume",
                "description": (
                    "Change the speaker volume. Call this whenever the "
                    "user says anything like 'louder', 'quieter', 'turn "
                    "it up', 'turn it down', 'volume up', 'volume down', "
                    "'set volume to 60 percent', or 'mute'. The change "
                    "persists across sessions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["up", "down", "set", "mute", "unmute"],
                            "description": (
                                "up / down = ±10 percent of current. "
                                "set = absolute, use `level`. "
                                "mute / unmute = toggle audio."
                            ),
                        },
                        "level": {
                            "type": "integer",
                            "minimum": _VOLUME_MIN,
                            "maximum": _VOLUME_MAX,
                            "description": (
                                "Target percent for action=set. Ignored "
                                "for up/down/mute/unmute."
                            ),
                        },
                    },
                    "required": ["action"],
                },
            },
        }
    ]


class VolumeToolExecutor:
    """Handles the adjust_volume tool call."""

    def __init__(self, mixer_control: Optional[str] = None,
                 mixer_card: Optional[str] = _MIXER_CARD):
        # Resolution order:
        #   1. explicit arg (constructor or WALLE_MIXER_CONTROL env var)
        #   2. auto-probe via amixer scontrols — picks the first common
        #      name that exists on this card ("PCM" on USB DACs like
        #      UACDemoV1.0, "Master" on most onboard codecs).
        #   3. "Master" as a last-ditch default so errors are recognisable.
        control = mixer_control or _DEFAULT_MIXER_CONTROL
        if control is None:
            control = _detect_mixer_control(mixer_card)
            if control is not None:
                _log.info(
                    "auto-detected amixer control %r on card=%s",
                    control, mixer_card or "default",
                )
            else:
                _log.warning(
                    "could not auto-detect amixer control on card=%s; "
                    "falling back to 'Master'. Set WALLE_MIXER_CONTROL "
                    "to override.",
                    mixer_card or "default",
                )
                control = "Master"
        self._control = control
        self._card = mixer_card

    def execute(self, fn_name: str, args: dict) -> str:
        if fn_name != "adjust_volume":
            return f"Unknown tool: {fn_name}"
        action = str(args.get("action", "")).lower().strip()
        level_arg = args.get("level")
        if action == "up":
            return self._apply_delta(+_VOLUME_STEP_PERCENT)
        if action == "down":
            return self._apply_delta(-_VOLUME_STEP_PERCENT)
        if action == "set":
            if level_arg is None:
                return "Set volume needs a level between 0 and 100."
            try:
                level = int(level_arg)
            except (TypeError, ValueError):
                return f"Invalid level: {level_arg!r}"
            level = max(_VOLUME_MIN, min(_VOLUME_MAX, level))
            return self._apply_absolute(level)
        if action == "mute":
            return self._apply_mute(True)
        if action == "unmute":
            return self._apply_mute(False)
        return f"Unknown action: {action!r}"

    # ----- internals -------------------------------------------------

    def _amixer_args(self) -> list[str]:
        # -q keeps stdout quiet so we can parse our own result cleanly.
        # -M maps to logarithmic volume, matching desktop mixers (a 50%
        # setting sounds halfway between mute and max rather than being
        # dominated by the top of the dB curve).
        base = ["amixer", "-q", "-M"]
        if self._card:
            base += ["-c", self._card]
        return base

    def _apply_delta(self, delta_percent: int) -> str:
        sign = "+" if delta_percent >= 0 else "-"
        cmd = self._amixer_args() + [
            "sset", self._control, f"{abs(delta_percent)}%{sign}", "unmute",
        ]
        ok, msg = _run(cmd)
        if not ok:
            return f"Could not change volume: {msg}"
        current = _current_volume_percent(self._control, self._card)
        if current is not None:
            _save_state({"level": current, "muted": False})
            return f"Volume {'up' if delta_percent > 0 else 'down'} — now {current}%."
        return f"Volume {'up' if delta_percent > 0 else 'down'} (level unknown)."

    def _apply_absolute(self, level_percent: int) -> str:
        cmd = self._amixer_args() + [
            "sset", self._control, f"{level_percent}%", "unmute",
        ]
        ok, msg = _run(cmd)
        if not ok:
            return f"Could not set volume: {msg}"
        _save_state({"level": level_percent, "muted": False})
        return f"Volume set to {level_percent}%."

    def _apply_mute(self, mute: bool) -> str:
        cmd = self._amixer_args() + [
            "sset", self._control, "mute" if mute else "unmute",
        ]
        ok, msg = _run(cmd)
        if not ok:
            return f"Could not {'mute' if mute else 'unmute'}: {msg}"
        state = _load_state() or {}
        state["muted"] = mute
        _save_state(state)
        return "Muted." if mute else "Unmuted."


# ---------------------------------------------------------------------------
# Startup integration
# ---------------------------------------------------------------------------


def restore_persisted_volume(
    mixer_control: Optional[str] = None,
    mixer_card: Optional[str] = _MIXER_CARD,
) -> Optional[str]:
    mixer_control = (
        mixer_control
        or _DEFAULT_MIXER_CONTROL
        or _detect_mixer_control(mixer_card)
        or "Master"
    )
    """Reapply the last user-commanded volume on startup.

    Returns a short status string for logging, or None if nothing was
    stored. Silent-best-effort: a missing state file, missing amixer,
    or a mixer that doesn't exist are all non-fatal.
    """
    state = _load_state()
    if not state:
        return None
    level = state.get("level")
    muted = bool(state.get("muted"))
    base = ["amixer", "-q", "-M"]
    if mixer_card:
        base += ["-c", mixer_card]
    if isinstance(level, int):
        ok, _ = _run(base + ["sset", mixer_control, f"{level}%"])
        if not ok:
            return None
    ok, _ = _run(base + ["sset", mixer_control, "mute" if muted else "unmute"])
    if not ok:
        return None
    return f"volume restored to {level}%{' (muted)' if muted else ''}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3, check=False,
        )
    except FileNotFoundError:
        return False, f"{cmd[0]} not installed"
    except subprocess.TimeoutExpired:
        return False, f"{cmd[0]} timed out"
    except Exception as exc:
        return False, str(exc)
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        return False, stderr or f"rc={proc.returncode}"
    return True, proc.stdout.strip()


def _current_volume_percent(
    control: str, card: Optional[str]
) -> Optional[int]:
    base = ["amixer", "-M"]
    if card:
        base += ["-c", card]
    ok, out = _run(base + ["sget", control])
    if not ok or not out:
        return None
    # amixer sget output has lines like: "Front Left: Playback 64 [50%] [on]"
    for line in out.splitlines():
        idx = line.find("[")
        if idx == -1:
            continue
        end = line.find("%]", idx)
        if end == -1:
            continue
        try:
            return int(line[idx + 1:end])
        except ValueError:
            continue
    return None


def _load_state() -> Optional[dict[str, Any]]:
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        _log.warning("volume state load failed: %s", exc)
        return None


def _save_state(state: dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(_STATE_FILE) or ".", exist_ok=True)
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception as exc:
        _log.warning("volume state save failed: %s", exc)

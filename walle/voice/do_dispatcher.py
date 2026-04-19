"""Routes ``DO:`` lines from the LLM output parser to the right executor.

Two kinds of commands land here today:

1. Arduino CLI lines (``drive …``, ``head pos …``, ``scan``, ``stop``…)
   go to the Hardware Controller over the Unix socket.
2. Host-side commands (``volume up``, ``volume down``, ``volume set N``,
   ``mute``, ``unmute``) go to :class:`VolumeToolExecutor`, which shells
   out to ``amixer`` and saves the level to disk.

Keeping the routing in one place means the system prompt can list both
kinds of commands side by side without special-casing them upstream.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from walle.hw_client import HardwareClient
from walle.tools.audio import VolumeToolExecutor

_log = logging.getLogger("walle.voice.do_dispatcher")


_ARDUINO_VERBS = frozenset(
    {
        "drive", "spin", "stop", "stopm", "stopall",
        "head", "home", "off",
        "trim", "left", "right", "ma", "mb", "both",
        "tank", "mix", "status", "help",
    }
)

# "scan" isn't an Arduino CLI verb — the original walle implementation
# walked through multiple head positions on the Python side. We keep
# that pattern here: DoDispatcher expands "scan" into a sequence of
# head pos commands with inter-step sleeps, each still arbitrated by
# HWC. The arbiter's 3 s override window is long enough that the
# tracker doesn't fight us between steps (each head command pushes the
# override forward).
_SCAN_POSITIONS = (20, 80, 50)  # left-ish → right-ish → center
_SCAN_STEP_DELAY_S = 0.9


class DoDispatcher:
    """Dispatch a single ``DO:`` line to the correct executor."""

    def __init__(
        self,
        hw_client: HardwareClient,
        volume_exec: Optional[VolumeToolExecutor] = None,
    ) -> None:
        self._hw = hw_client
        self._volume = volume_exec

    def dispatch(self, cli: str) -> str:
        """Execute one ``DO:`` line. Returns a human-readable status string."""
        line = cli.strip()
        if not line:
            return "empty"
        first = line.split(" ", 1)[0].lower()

        # Host-side commands.
        if first == "volume":
            return self._dispatch_volume(line)
        if first in {"mute", "unmute"}:
            return self._dispatch_volume(line)

        # Multi-step commands that need decomposition on the client side.
        if first == "scan":
            return self._dispatch_scan()

        # Arduino CLI — shipped verbatim to the hardware controller.
        if first in _ARDUINO_VERBS:
            reply = self._hw.send(line)
            if reply.startswith("ERR"):
                _log.warning("DO %r → %s", line, reply)
            else:
                _log.info("DO %r → %s", line, reply)
            return reply

        _log.warning("DO %r: unknown verb %r — skipping", line, first)
        return f"ERR unknown command: {first}"

    def _dispatch_scan(self) -> str:
        """Walk the head through a preset sweep on a background thread.

        Fire-and-forget so the caller (typically a dispatcher consumer
        thread) isn't blocked for 3 s. Each sub-command pushes the
        HWC arbiter's override forward, so the face tracker stays out
        of our way for the duration of the sweep.
        """
        def _walk() -> None:
            for i, pct in enumerate(_SCAN_POSITIONS):
                reply = self._hw.send(f"head pos {pct}")
                if reply.startswith("ERR"):
                    _log.warning(
                        "scan step %d (head pos %d) → %s", i + 1, pct, reply
                    )
                    # Keep going — mid-scan rejections usually mean the
                    # tracker moved the physical position between our
                    # sub-steps; the next step still lands correctly.
                if i < len(_SCAN_POSITIONS) - 1:
                    time.sleep(_SCAN_STEP_DELAY_S)

        threading.Thread(
            target=_walk, daemon=True, name="do-dispatcher-scan"
        ).start()
        return "OK (scan started)"

    # -- internals --

    def _dispatch_volume(self, line: str) -> str:
        if self._volume is None:
            _log.warning("DO %r: no volume executor registered", line)
            return "ERR volume executor missing"
        # Parse "volume up", "volume down", "volume set 60", "mute",
        # "unmute" into the VolumeToolExecutor's args format.
        tokens = line.split()
        action: Optional[str] = None
        level: Optional[int] = None
        if tokens[0].lower() == "mute":
            action = "mute"
        elif tokens[0].lower() == "unmute":
            action = "unmute"
        elif tokens[0].lower() == "volume":
            if len(tokens) < 2:
                return "ERR volume needs an argument"
            sub = tokens[1].lower()
            if sub in {"up", "down", "mute", "unmute"}:
                action = sub
            elif sub in {"set", "to", "="}:
                if len(tokens) < 3:
                    return "ERR volume set needs a level"
                try:
                    level = int(tokens[2].rstrip("%"))
                except ValueError:
                    return f"ERR invalid level: {tokens[2]!r}"
                action = "set"
            else:
                # Bare ``volume 60`` = set to 60.
                try:
                    level = int(sub.rstrip("%"))
                    action = "set"
                except ValueError:
                    return f"ERR unknown volume subcommand: {sub!r}"

        args: dict = {"action": action}
        if level is not None:
            args["level"] = level
        result = self._volume.execute("adjust_volume", args)
        _log.info("DO %r → %s", line, result)
        return result

"""Streaming ``SAY:`` / ``DO:`` line extractor for the LLM output.

The LLM is prompted to reply in two tagged line types:

    SAY: <short sentence for TTS>
    DO:  <one Arduino CLI or host command>

This parser accepts token chunks from the streaming chat API and fires
``on_say(text)`` and ``on_do(cli)`` callbacks the moment a newline
arrives for each line. That's what lets TTS start speaking sentence N
while the model is still generating sentence N+1, and lets motor
commands fire as soon as they're decoded instead of after the full turn.

Graceful fallback: lines that don't start with ``SAY:`` or ``DO:`` are
treated as speech. A 3B model occasionally drops the tag for a line;
we'd rather hear the sentence than drop it on the floor. Unknown
commands in ``DO:`` lines still surface to the dispatcher, which logs
and skips them.
"""

from __future__ import annotations

import logging
from typing import Callable

_log = logging.getLogger("walle.voice.llm_parser")


class TaggedStreamParser:
    """Feed streaming chunks; emit ``on_say`` / ``on_do`` per completed line.

    Not thread-safe — feed from a single producer (the LLM streaming loop).
    """

    def __init__(
        self,
        on_say: Callable[[str], None],
        on_do: Callable[[str], None],
    ) -> None:
        self._on_say = on_say
        self._on_do = on_do
        self._buf = ""

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._dispatch(line.strip())

    def flush(self) -> None:
        """Flush any trailing content that wasn't newline-terminated."""
        tail = self._buf.strip()
        self._buf = ""
        if tail:
            self._dispatch(tail)

    # -- internals --

    def _dispatch(self, line: str) -> None:
        if not line:
            return
        upper = line.upper()
        if upper.startswith("SAY:"):
            text = line[4:].strip()
            if text:
                try:
                    self._on_say(text)
                except Exception:
                    _log.exception("on_say callback raised on %r", text)
            return
        if upper.startswith("DO:"):
            cli = line[3:].strip()
            if cli:
                try:
                    self._on_do(cli)
                except Exception:
                    _log.exception("on_do callback raised on %r", cli)
            return
        # Unformatted line — treat as speech. Matches what the user
        # would prefer hearing over silence when the model forgets
        # the prefix on one sentence.
        try:
            self._on_say(line)
        except Exception:
            _log.exception("on_say fallback raised on %r", line)

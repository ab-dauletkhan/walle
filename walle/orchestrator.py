"""
WALL-E Orchestrator — thin Facade over ChatLoop.

All object construction happens in startup.py (Composition Root).
This module provides WallEOrchestrator which implements the LLMClient
interface for the voice pipeline and exposes chat() for API/REPL.
"""

import logging
import re
import time
from typing import Iterator, Optional

_log = logging.getLogger("walle.orchestrator")

# Small local models (qwen2.5:3b / qwen3:4b) frequently emit the send_message
# tool call as literal plain text instead of a structured call, producing
# prefixes like "Message sent to user: ...", "Send_message: ...", or
# "Message sent_message: ...". Strip them before handing the reply to TTS.
_TOOL_PREFIX_RE = re.compile(
    r"^\s*(?:message\s+sent(?:_message|\s+to\s+user)?|send_?message)\s*[:\-]\s*",
    re.IGNORECASE,
)


def _strip_tool_prefix(text: str) -> str:
    if not text:
        return text
    cleaned = text
    # Loop — handles cases like "Message sent_message: Send_message: hi"
    for _ in range(3):
        new = _TOOL_PREFIX_RE.sub("", cleaned, count=1)
        if new == cleaned:
            break
        cleaned = new
    return cleaned.strip() or text


class WallEOrchestrator:
    """Thin facade over ChatLoop with a streaming API for non-voice callers.

    The voice path invokes ChatLoop.run directly with SAY/DO callbacks
    and doesn't go through this class. The API server and text REPL use
    ``chat`` / ``stream_chat`` — those rebuild the accumulated SAY text
    back into word-by-word yields for display, keeping the old contract
    so nothing external has to change.
    """

    def __init__(
        self,
        chat_loop,
        recall_mem,
        context_manager,
        serial_manager=None,
        vision_service=None,
    ):
        self._chat_loop = chat_loop
        self.recall_mem = recall_mem
        self.context_manager = context_manager
        self.vision_service = vision_service
        # Kept on the attribute surface so callers that want to reach the
        # hardware (e.g. API server) can still do so; None is fine in the
        # post-HWC world because walle itself no longer opens the port.
        self.serial_manager = serial_manager

    def set_vision_service(self, vision_service):
        self.vision_service = vision_service

    def chat(self, user_input: str) -> Optional[str]:
        reply = self._chat_loop.run(user_input)
        return _strip_tool_prefix(reply) if reply else reply

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break

        if not user_text:
            yield "I didn't catch that."
            return

        response = self.chat(user_text)
        if not response:
            response = "I'm not sure what to say."

        words = response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.02)


def main():
    from walle.startup import main as startup_main

    startup_main()


if __name__ == "__main__":
    main()

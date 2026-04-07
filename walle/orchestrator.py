"""
WALL-E Orchestrator — thin Facade over ChatLoop.

All object construction happens in startup.py (Composition Root).
This module provides WallEOrchestrator which implements the LLMClient
interface for the voice pipeline and exposes chat() for API/REPL.
"""

import logging
import time
from typing import Iterator, Optional

_log = logging.getLogger("walle.orchestrator")


class WallEOrchestrator:
    """Thin facade: accepts injected dependencies, delegates to ChatLoop.

    Implements stream_chat() (duck-typed LLMClient) for the voice pipeline
    and chat() for the API server and text REPL.
    """

    def __init__(self, chat_loop, comm_exec, recall_mem, context_manager,
                 vision_service, serial_manager, tool_suite):
        self._chat_loop = chat_loop
        self._comm_exec = comm_exec
        self.recall_mem = recall_mem
        self.context_manager = context_manager
        self.vision_service = vision_service
        self.serial_manager = serial_manager
        self.tool_suite = tool_suite

    def set_vision_service(self, vision_service):
        self.vision_service = vision_service

    def chat(self, user_input: str) -> Optional[str]:
        return self._chat_loop.run(user_input)

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
    from startup import main as startup_main
    startup_main()


if __name__ == "__main__":
    main()

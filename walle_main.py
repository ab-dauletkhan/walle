"""
WALL-E Orchestrator — thin Facade over ChatLoop.

All object construction happens in startup.py (Composition Root).
This module provides the WallEOrchestrator (LLMClient interface)
and RobotBridge (voice intent adapter).
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from memory.robot_tools import RobotControlExecutor
from stt_tts.mock_llm import LLMClient

try:
    from stt_tts.main import BaseRobotController
except ImportError:
    class BaseRobotController(ABC):
        @abstractmethod
        def execute(self, action: str, utterance: str, confidence: float) -> None: ...
        @abstractmethod
        def close(self) -> None: ...

_log = logging.getLogger("walle.orchestrator")


# ---------------------------------------------------------------------------
# WallEOrchestrator — Facade delegating to ChatLoop
# ---------------------------------------------------------------------------
class WallEOrchestrator(LLMClient):
    """Thin facade: accepts injected dependencies, delegates to ChatLoop."""

    def __init__(self, chat_loop, comm_exec, recall_mem, context_manager,
                 vision_service, serial_manager, tool_suite):
        self._chat_loop = chat_loop
        self._comm_exec = comm_exec
        self.recall_mem = recall_mem
        self.context_manager = context_manager
        self.vision_service = vision_service
        self.serial_manager = serial_manager
        self.tool_suite = tool_suite

    def set_vision_service(self, vision_service, capture_image_exec):
        """Attach vision service and capture executor."""
        self.vision_service = vision_service
        self.tool_suite.set_capture_image_executor(capture_image_exec)

    def chat(self, user_input: str) -> Optional[str]:
        """Public chat entry point for API, REPL, and voice mode."""
        return self._chat_loop.run(user_input)

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """LLMClient interface for the SpeechRouter voice pipeline."""
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


# ---------------------------------------------------------------------------
# RobotBridge — adapts RobotControlExecutor to BaseRobotController interface
# ---------------------------------------------------------------------------
class RobotBridge(BaseRobotController):
    """Maps voice intent actions to RobotControlExecutor tool calls."""

    ACTION_MAP = {
        "forward":  ("drive_forward",   {"speed": 50, "duration_ms": 1000}),
        "backward": ("drive_backward",  {"speed": 50, "duration_ms": 1000}),
        "left":     ("turn_left",       {"speed": 50, "duration_ms": 500}),
        "right":    ("turn_right",      {"speed": 50, "duration_ms": 500}),
        "stop":     ("stop_movement",   {}),
        "wave":     ("wave_hello",      {}),
        "dance":    ("express_emotion", {"emotion": "happy"}),
    }

    def __init__(self, robot_exec: RobotControlExecutor):
        self._robot = robot_exec

    def execute(self, action: str, utterance: str, confidence: float) -> None:
        mapping = self.ACTION_MAP.get(action)
        if mapping:
            tool_name, args = mapping
            _log.debug("RobotBridge: %s -> %s (confidence=%.0f%%)", action, tool_name, confidence * 100)
            self._robot.execute(tool_name, args)
        else:
            _log.warning("RobotBridge: unknown action '%s'", action)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Entry point (delegates to startup.py)
# ---------------------------------------------------------------------------
def main():
    from startup import main as startup_main
    startup_main()


if __name__ == "__main__":
    main()

"""
Communication tools for WALL-E.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SendMessageArgs:
    message: str
    request_heartbeat: bool = False

    @classmethod
    def model_validate(cls, payload: Dict[str, Any]) -> "SendMessageArgs":
        if not isinstance(payload, dict):
            raise TypeError("send_message args must be an object")

        raw_message = payload.get("message", "")
        if isinstance(raw_message, dict):
            raw_message = raw_message.get("content", str(raw_message))

        message = str(raw_message).strip() if raw_message else ""
        request_heartbeat = bool(payload.get("request_heartbeat", False))
        return cls(message=message, request_heartbeat=request_heartbeat)

    @classmethod
    def model_json_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send to the user",
                },
                "request_heartbeat": {
                    "type": "boolean",
                    "description": "Continue processing after sending",
                    "default": False,
                },
            },
            "required": ["message"],
        }


def get_communication_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a message to the user. ONLY way to communicate. Keep messages to 1-3 sentences (TTS output). Your raw text is invisible internal thought.",
                "parameters": SendMessageArgs.model_json_schema(),
            },
        }
    ]


class CommunicationExecutor:
    """Thread-safe message handoff from tool execution to the outer interface."""

    def __init__(self):
        self._lock = Lock()
        self.last_message: Optional[str] = None

    def execute(self, fn_name: str, args: Dict[str, Any]) -> str:
        if fn_name == "send_message":
            return self._send_message(args)
        return f"Unknown communication tool: {fn_name}"

    def _send_message(self, args: Dict[str, Any]) -> str:
        try:
            validated = SendMessageArgs.model_validate(args)
        except Exception as exc:
            return f"Validation error: {exc}"

        if not validated.message:
            return "Error: message cannot be empty"

        with self._lock:
            self.last_message = validated.message
        return "Message sent to user"

    def get_last_message(self) -> Optional[str]:
        with self._lock:
            msg = self.last_message
            self.last_message = None
        return msg

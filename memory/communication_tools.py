"""
Communication Tools for WALL-E
Implements MemGPT-style inner monologue pattern.

The LLM's raw text output is internal thought (not shown to user).
To communicate with user, LLM must explicitly call send_message.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class SendMessageArgs(BaseModel):
    """Pydantic model for send_message tool arguments."""
    message: str = Field(..., description="The message to send to the user")
    request_heartbeat: bool = Field(default=False, description="Continue processing after sending")

    @field_validator('message', mode='before')
    @classmethod
    def extract_message(cls, v):
        """Handle cases where LLM returns dict instead of string."""
        if isinstance(v, dict):
            return v.get('content', str(v))
        return str(v) if v else ""


def get_communication_tools():
    """Return communication tools for LLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a message to the user. You MUST use this tool to communicate. Your text output is internal thought - user can't see it.",
                "parameters": SendMessageArgs.model_json_schema()
            }
        }
    ]


class CommunicationExecutor:
    """Executor for communication tools with Pydantic validation."""

    def __init__(self):
        self.last_message: Optional[str] = None

    def execute(self, fn_name: str, args: Dict[str, Any]) -> str:
        """Execute a communication tool."""
        if fn_name == "send_message":
            return self._send_message(args)
        return f"Unknown communication tool: {fn_name}"

    def _send_message(self, args: Dict[str, Any]) -> str:
        """Handle send_message with Pydantic validation."""
        try:
            validated = SendMessageArgs.model_validate(args)
            if not validated.message:
                return "Error: message cannot be empty"
            self.last_message = validated.message
            return "Message sent to user"
        except Exception as e:
            return f"Validation error: {e}"

    def get_last_message(self) -> Optional[str]:
        """Get the last message sent."""
        msg = self.last_message
        self.last_message = None
        return msg

"""
Integration support objects for the WALL-E orchestrator.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

from walle.tools.communication import SendMessageArgs, get_communication_tools
from walle.memory.heartbeat import add_heartbeat_to_tools

_log = logging.getLogger("walle.orchestrator.support")


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_result: str
    user_message: Optional[str] = None
    heartbeat_requested: bool = False


class ToolProvider(ABC):
    """Extensible interface for a group of tool schemas plus execution logic."""

    @abstractmethod
    def get_schemas(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def handles(self, name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        name: str,
        args: dict,
        *,
        user_message_already_sent: bool,
    ) -> ToolExecutionResult:
        raise NotImplementedError


class CommunicationToolProvider(ToolProvider):
    def __init__(self, communication_executor):
        self._communication_executor = communication_executor

    def get_schemas(self) -> list[dict]:
        return get_communication_tools()

    def handles(self, name: str) -> bool:
        return name == "send_message"

    def execute(
        self,
        name: str,
        args: dict,
        *,
        user_message_already_sent: bool,
    ) -> ToolExecutionResult:
        heartbeat_requested = bool(args.pop("request_heartbeat", False))

        try:
            validated = SendMessageArgs.model_validate(args)
        except Exception as exc:
            return ToolExecutionResult(tool_result=f"Validation error: {exc}")

        if not validated.message:
            return ToolExecutionResult(tool_result="Error: message cannot be empty")
        if user_message_already_sent:
            return ToolExecutionResult(tool_result="Message already sent (duplicate ignored)")

        result = self._communication_executor.execute(name, args)
        return ToolExecutionResult(
            tool_result=f"{result}: {validated.message}",
            user_message=validated.message,
            heartbeat_requested=heartbeat_requested or validated.request_heartbeat,
        )


class SchemaExecutorToolProvider(ToolProvider):
    def __init__(
        self,
        schema_factory: Callable[[], list[dict]],
        executor,
        *,
        request_heartbeat_after: bool = False,
    ):
        self._schema_factory = schema_factory
        self._executor = executor
        self._request_heartbeat_after = request_heartbeat_after

    def get_schemas(self) -> list[dict]:
        return self._schema_factory()

    def handles(self, name: str) -> bool:
        return any(schema["function"]["name"] == name for schema in self.get_schemas())

    def execute(
        self,
        name: str,
        args: dict,
        *,
        user_message_already_sent: bool,
    ) -> ToolExecutionResult:
        del user_message_already_sent
        heartbeat_requested = bool(args.pop("request_heartbeat", False))

        try:
            result = self._executor.execute(name, args)
        except Exception as exc:
            _log.error("Tool %s failed: %s", name, exc)
            result = f"Error executing {name}: {exc}"

        return ToolExecutionResult(
            tool_result=str(result),
            heartbeat_requested=heartbeat_requested or self._request_heartbeat_after,
        )


class ToolRegistry:
    """Registry of tool providers used by the orchestrator."""

    def __init__(self):
        self._providers: list[ToolProvider] = []

    def register(self, provider: ToolProvider) -> None:
        self._providers.append(provider)

    def get_schemas(self) -> list[dict]:
        schemas = []
        for provider in self._providers:
            schemas.extend(provider.get_schemas())
        return add_heartbeat_to_tools(schemas)

    def execute(self, name: str, args: dict, *, user_message_already_sent: bool) -> ToolExecutionResult:
        for provider in self._providers:
            if provider.handles(name):
                return provider.execute(
                    name,
                    args,
                    user_message_already_sent=user_message_already_sent,
                )
        return ToolExecutionResult(tool_result=f"Unknown tool: {name}")


class SystemPromptBuilder:
    """Builder for the per-turn system prompt."""

    def __init__(self, context_manager, personality_engine, core_memory):
        self._context_manager = context_manager
        self._personality_engine = personality_engine
        self._core_memory = core_memory

    def build(self, relevant_memories: str = "") -> str:
        context_str = self._context_manager.get_context_string()
        relevant_memories_section = f"{relevant_memories}\n" if relevant_memories else ""
        return (
            # Identity
            "You are WALL-E, a physical robot companion. "
            "You have a body with motors, arms, a head, and a neck. "
            "You exist in the real world and can move, look around, and interact.\n\n"

            # Communication rules
            "COMMUNICATION:\n"
            "- The user CANNOT see your text output. Use send_message tool for ALL replies.\n"
            "- Keep messages to 1-3 sentences. A TTS module reads them aloud.\n"
            "- Your raw text is internal thought — use it to reason before acting.\n\n"

            # Movement rules
            "MOVEMENT:\n"
            "- Each movement call is limited to 5 seconds. For longer moves, call multiple times.\n"
            "- After movement, call get_robot_status to verify you completed the action.\n"
            "- If status shows motors still running or position is wrong, adjust and retry.\n"
            "- You can chain multiple tools in one turn: move, check, move again, then respond.\n\n"

            # Behavior
            "BEHAVIOR:\n"
            f"{self._personality_engine.get_system_prompt_addition()}\n"
            "- Act, don't narrate. Call tools directly instead of describing what you plan to do.\n"
            "- Be honest about your limitations. If you can't see (no vision), say so.\n"
            "- Express yourself physically — wave when greeting, show emotion through posture.\n\n"

            # Dynamic context
            f"{context_str}\n"
            f"{relevant_memories_section}"
            f"{self._core_memory.compile()}\n"
        )


class RelevantMemoryProvider:
    """Facade for assembling relevant recall and archival context."""

    def __init__(self, recall_memory, archival_memory, recall_limit: int = 3, archival_limit: int = 2):
        self._recall_memory = recall_memory
        self._archival_memory = archival_memory
        self._recall_limit = recall_limit
        self._archival_limit = archival_limit

    def build_context(self, query: str) -> str:
        with ThreadPoolExecutor(max_workers=2) as executor:
            recall_future = executor.submit(self._recall_memory.search, query, self._recall_limit)
            archival_future = executor.submit(self._archival_memory.search, query, self._archival_limit)
            recall_hits = recall_future.result()
            archival_hits = archival_future.result()

        if not recall_hits and not archival_hits:
            return ""

        lines = ["", "[RELEVANT MEMORIES]"]
        for hit in recall_hits:
            lines.append(f"- {hit['role']}: {hit['content']}")
        for hit in archival_hits:
            lines.append(f"- Fact: {hit['content']}")
        return "\n".join(lines) + "\n\n"


class ToolSuiteFacade:
    """Facade that delegates to a ToolRegistry.

    The registry is built in the Composition Root (startup.py).
    Adding a new tool = one new ToolProvider + one registry.register() line.
    No changes to this class needed.
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def register(self, provider: ToolProvider) -> None:
        """Register an additional provider (e.g. capture_image after vision init)."""
        self._registry.register(provider)

    def build_schemas(self) -> list[dict]:
        return self._registry.get_schemas()

    def execute_tool(self, name: str, args: dict, user_message_already_sent: bool) -> ToolExecutionResult:
        return self._registry.execute(
            name,
            args,
            user_message_already_sent=user_message_already_sent,
        )

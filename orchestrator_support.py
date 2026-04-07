"""
Integration support objects for the WALL-E orchestrator.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

from memory.communication_tools import SendMessageArgs, get_communication_tools
from memory.heartbeat import add_heartbeat_to_tools
from memory.knowledge_tools import get_knowledge_tools
from memory.memory_tools import get_memory_tools
from memory.personality_system import get_personality_tools
from memory.robot_tools import get_robot_control_tools, get_robot_tool_names
from vision_service import get_capture_image_tools

_log = logging.getLogger("walle.orchestrator.support")


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_result: str
    user_message: Optional[str] = None
    heartbeat_requested: bool = False


@dataclass(frozen=True)
class ToolBinding:
    execute: Callable[[str, dict], str]
    request_heartbeat_after: bool = False


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
            "You are WALL-E, a robot companion running on Jetson. "
            "Reply via send_message tool ONLY.\n"
            "Your raw text is internal thought (invisible to user). "
            "Keep replies to 1-3 sentences max - a TTS module reads your output aloud.\n"
            f"{self._personality_engine.get_system_prompt_addition()}\n"
            "Use tools directly. Don't narrate what you plan to do.\n"
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
    """Facade that unifies tool schemas and execution across subsystems."""

    def __init__(
        self,
        communication_executor,
        robot_executor,
        memory_executor,
        personality_executor,
        knowledge_executor,
        capture_image_executor=None,
    ):
        self._communication_executor = communication_executor
        self._robot_executor = robot_executor
        self._memory_executor = memory_executor
        self._personality_executor = personality_executor
        self._knowledge_executor = knowledge_executor
        self._capture_image_executor = capture_image_executor
        self._bindings = self._build_bindings()

    def set_capture_image_executor(self, capture_image_executor) -> None:
        self._capture_image_executor = capture_image_executor
        self._bindings = self._build_bindings()

    def build_schemas(self) -> list[dict]:
        tools = (
            get_communication_tools()
            + get_robot_control_tools()
            + get_memory_tools()
            + get_personality_tools()
            + get_knowledge_tools()
        )
        if self._capture_image_executor is not None:
            tools += get_capture_image_tools()
        return add_heartbeat_to_tools(tools)

    def execute_tool(self, name: str, args: dict, user_message_already_sent: bool) -> ToolExecutionResult:
        heartbeat_requested = bool(args.pop("request_heartbeat", False))

        if name == "send_message":
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

        binding = self._bindings.get(name)
        if binding is None:
            return ToolExecutionResult(tool_result=f"Unknown tool: {name}", heartbeat_requested=heartbeat_requested)

        try:
            result = binding.execute(name, args)
        except Exception as exc:
            _log.error("Tool %s failed: %s", name, exc)
            result = f"Error executing {name}: {exc}"

        return ToolExecutionResult(
            tool_result=str(result),
            heartbeat_requested=heartbeat_requested or binding.request_heartbeat_after,
        )

    def _build_bindings(self) -> dict[str, ToolBinding]:
        bindings: dict[str, ToolBinding] = {}

        for tool_name in get_robot_tool_names():
            bindings[tool_name] = ToolBinding(self._robot_executor.execute)

        for schema in get_memory_tools():
            tool_name = schema["function"]["name"]
            bindings[tool_name] = ToolBinding(self._memory_executor.execute)

        for schema in get_personality_tools():
            tool_name = schema["function"]["name"]
            bindings[tool_name] = ToolBinding(self._personality_executor.execute)

        for schema in get_knowledge_tools():
            tool_name = schema["function"]["name"]
            bindings[tool_name] = ToolBinding(
                self._knowledge_executor.execute,
                request_heartbeat_after=True,
            )

        if self._capture_image_executor is not None:
            for schema in get_capture_image_tools():
                tool_name = schema["function"]["name"]
                bindings[tool_name] = ToolBinding(
                    self._capture_image_executor.execute,
                    request_heartbeat_after=True,
                )

        return bindings

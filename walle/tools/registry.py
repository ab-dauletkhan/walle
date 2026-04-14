"""
Integration support objects for the WALL-E orchestrator.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

from walle.memory.heartbeat import add_heartbeat_to_tools
from walle.tools.communication import SendMessageArgs, get_communication_tools

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
            return ToolExecutionResult(
                tool_result="Message already sent (duplicate ignored)"
            )

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

    def execute(
        self, name: str, args: dict, *, user_message_already_sent: bool
    ) -> ToolExecutionResult:
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
        relevant_memories_section = (
            f"{relevant_memories}\n" if relevant_memories else ""
        )

        # Static prefix — byte-stable across turns so Ollama/llama.cpp can
        # prefix-cache it. Per-turn values must live in the dynamic tail
        # below, not here.
        static_prefix = (
            "/no_think\n\n"
            "You are WALL-E, a physical robot companion with motors, arms, a head, and a neck.\n\n"
            "RULES:\n"
            "- Reply with send_message for ALL user-visible messages (1-3 sentences, TTS reads them aloud).\n"
            "- For greetings, small talk, and questions, call send_message ONLY. Do NOT call motion tools.\n"
            "- Only call motion tools (drive_*, turn_*, stop_movement, wave_hello, scan_surroundings, express_emotion) when the user explicitly asks for physical action.\n"
            "- Movement calls cap at 5000ms; chain calls for longer moves.\n"
            "- Act, don't narrate. Be honest about your limitations.\n"
            f"{self._personality_engine.get_system_prompt_addition()}\n\n"
            f"{self._core_memory.compile()}\n"
        )

        # Dynamic tail — per-turn values go here so only the tail re-prefills.
        dynamic_tail = ""
        if relevant_memories_section:
            dynamic_tail += f"\n{relevant_memories_section}"
        if context_str:
            dynamic_tail += f"\n{context_str}\n"

        return static_prefix + dynamic_tail


class RelevantMemoryProvider:
    """Facade for assembling relevant recall and archival context.

    Uses score threshold instead of fixed count — only memories
    semantically close to the query are included, like human
    associative recall.
    """

    _SCORE_THRESHOLD = 0.35  # cosine distance — lower = more relevant
    _FETCH_LIMIT = 20  # max candidates to fetch from FAISS
    _MAX_RECALL = 3  # hard cap on recall hits injected into the prompt
    _MAX_ARCHIVAL = 2  # hard cap on archival facts injected into the prompt
    _MAX_CHARS_PER_HIT = 240  # truncate long memories to keep prompt tight

    def __init__(self, recall_memory, archival_memory):
        self._recall_memory = recall_memory
        self._archival_memory = archival_memory

    def build_context(self, query: str) -> str:
        with ThreadPoolExecutor(max_workers=2) as executor:
            recall_future = executor.submit(
                self._recall_memory.search, query, self._FETCH_LIMIT
            )
            archival_future = executor.submit(
                self._archival_memory.search, query, self._FETCH_LIMIT
            )
            recall_hits = recall_future.result()
            archival_hits = archival_future.result()

        # Filter by score, then hard-cap regardless of how many pass.
        # Without the cap, a query like "hi" matches every past greeting
        # and dumps hundreds of tokens of memories into every prompt.
        recall_relevant = [
            h for h in recall_hits if h.get("score", 0) <= self._SCORE_THRESHOLD
        ][: self._MAX_RECALL]
        archival_relevant = [
            h for h in archival_hits if h.get("score", 0) <= self._SCORE_THRESHOLD
        ][: self._MAX_ARCHIVAL]

        # Fallback for non-scored backends (FTS5, recent).
        if not recall_relevant and recall_hits:
            recall_relevant = recall_hits[: self._MAX_RECALL]
        if not archival_relevant and archival_hits:
            archival_relevant = archival_hits[: self._MAX_ARCHIVAL]

        if not recall_relevant and not archival_relevant:
            return ""

        def _trim(text: str) -> str:
            text = text or ""
            return text if len(text) <= self._MAX_CHARS_PER_HIT else text[: self._MAX_CHARS_PER_HIT - 1] + "…"

        lines = ["", "[RELEVANT MEMORIES]"]
        for hit in recall_relevant:
            lines.append(f"- {hit['role']}: {_trim(hit['content'])}")
        for hit in archival_relevant:
            lines.append(f"- Fact: {_trim(hit['content'])}")
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

    def execute_tool(
        self, name: str, args: dict, user_message_already_sent: bool
    ) -> ToolExecutionResult:
        return self._registry.execute(
            name,
            args,
            user_message_already_sent=user_message_already_sent,
        )

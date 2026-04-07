"""
WALL-E Chat Loop — Mediator between LLM, tools, memory, and communication.

Extracted from WallELLMClient to isolate the MemGPT-style tool-calling loop
from object construction and lifecycle management.
"""

import json
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from walle.memory.config import conf
from walle.memory.context_manager import InteractionContext, SensorSimulator

_log = logging.getLogger("walle.chat")


# ---------------------------------------------------------------------------
# Data types for streaming tool calls
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolFunctionCall:
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCallEnvelope:
    id: str
    function: ToolFunctionCall
    type: str = "function"


# ---------------------------------------------------------------------------
# ChatSession — lightweight conversation buffer
# ---------------------------------------------------------------------------
class ChatSession:
    def __init__(self):
        self.history = deque(maxlen=conf.MAX_CONTEXT_MESSAGES)

    def add(self, role, content, tool_calls=None):
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.history.append(msg)

    def add_tool_result(self, tool_id, name, content):
        self.history.append({"role": "tool", "tool_call_id": tool_id, "name": name, "content": content})

    def get_messages(self, system_prompt: str):
        return [{"role": "system", "content": system_prompt}] + list(self.history)


# ---------------------------------------------------------------------------
# LLMStreamer — streams responses from Ollama-compatible API
# ---------------------------------------------------------------------------
class LLMStreamer:
    """Wraps the OpenAI-compatible streaming API."""

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def stream(self, messages, tools, max_retries=2):
        """Stream response, return (content, tool_calls_list)."""
        for attempt in range(max_retries):
            try:
                stream = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                )

                full_content = ""
                tool_calls_map = {}
                inner_thought_started = False

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_content += delta.content
                        if _log.isEnabledFor(logging.DEBUG):
                            if not inner_thought_started:
                                print("   \U0001f4ad ", end="", flush=True)
                                inner_thought_started = True
                            print(delta.content, end="", flush=True)

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_map:
                                tool_calls_map[idx] = {"id": "", "func": {"name": "", "args": ""}}
                            if tc.id:
                                tool_calls_map[idx]["id"] += tc.id
                            if tc.function.name:
                                tool_calls_map[idx]["func"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_map[idx]["func"]["args"] += tc.function.arguments

                if inner_thought_started:
                    print()

                tool_calls_list = []
                for idx in sorted(tool_calls_map.keys()):
                    t = tool_calls_map[idx]
                    tool_calls_list.append(ToolCallEnvelope(
                        id=t["id"] or f"call_{idx}",
                        function=ToolFunctionCall(
                            name=t["func"]["name"],
                            arguments=t["func"]["args"],
                        ),
                        type="function",
                    ))

                return full_content, tool_calls_list

            except Exception as e:
                _log.warning("Stream error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                time.sleep(1)

        _log.error("Failed to generate LLM response after %d retries", max_retries)
        return "I'm having trouble reaching my brain right now. Please check that Ollama is running.", []

    def summarize(self, text: str) -> str:
        """Generate a concise summary for memory compression."""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": f"Summarize this concisely:\n{text}"}],
                max_tokens=200,
            )
            return resp.choices[0].message.content
        except Exception:
            return text[:500] + "..."


# ---------------------------------------------------------------------------
# ChatLoop — the MemGPT-style tool-calling loop (Mediator)
# ---------------------------------------------------------------------------
class ChatLoop:
    """Mediates between LLM streaming, tool execution, memory, and communication.

    All dependencies are injected — ChatLoop creates nothing.
    """

    def __init__(
        self,
        llm_streamer: LLMStreamer,
        tool_suite,
        session: ChatSession,
        prompt_builder,
        memory_provider,
        comm_exec,
        recall_mem,
        archival_mem,
        heartbeat,
        context_manager,
    ):
        self._llm = llm_streamer
        self._tool_suite = tool_suite
        self._session = session
        self._prompt_builder = prompt_builder
        self._memory_provider = memory_provider
        self._comm_exec = comm_exec
        self._recall_mem = recall_mem
        self._archival_mem = archival_mem
        self._heartbeat = heartbeat
        self._context_manager = context_manager
        self._compression_lock = threading.Lock()

    @property
    def session(self) -> ChatSession:
        return self._session

    def run(self, user_input: str) -> Optional[str]:
        """Run one full MemGPT-style chat turn. Returns the message for the user."""
        # Drop any stale message from a prior interrupted turn
        self._comm_exec.get_last_message()

        # 1. Context update
        interaction_count = self._recall_mem.get_count()
        self._context_manager.update_interaction(InteractionContext(
            last_interaction=datetime.now(),
            interaction_count=interaction_count,
        ))
        if interaction_count % 5 == 0:
            self._context_manager.update_environment(
                SensorSimulator.simulate_environment_context(
                    battery=max(0, 80 - interaction_count)
                )
            )

        # 2. Retrieve relevant memories
        memory_context = self._memory_provider.build_context(user_input)

        # 3. Insert user message (deferred embedding)
        self._recall_mem.insert("user", user_input, defer_embedding=True)
        self._session.add("user", user_input)
        self._heartbeat.reset()

        # 4. Compress old memories if needed (non-blocking)
        self._start_compression_if_needed()

        # 5. Tool execution loop
        return self._tool_loop(memory_context)

    def _tool_loop(self, memory_context: str) -> Optional[str]:
        """Inner loop: stream LLM → execute tools → repeat until send_message."""
        iteration = 0
        user_received_message = False

        while iteration < 10:
            iteration += 1
            tools = self._tool_suite.build_schemas()
            system_prompt = self._prompt_builder.build(memory_context)

            content, tool_calls = self._llm.stream(
                self._session.get_messages(system_prompt), tools
            )

            if not tool_calls:
                if content:
                    self._session.add("assistant", content)
                    if not user_received_message:
                        _log.debug("No send_message called, prompting...")
                        continue
                break

            self._session.add("assistant", content or "", tool_calls)
            hb_req = False

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    _log.warning("JSON error in tool %s: %s", name, e)
                    self._session.add_tool_result(tc.id, name, f"Error: Invalid JSON - {e}")
                    continue

                execution = self._tool_suite.execute_tool(
                    name=name,
                    args=args,
                    user_message_already_sent=user_received_message,
                )

                if execution.user_message and not user_received_message:
                    _log.debug("WALL-E: %s", execution.user_message)
                    self._recall_mem.insert("assistant", execution.user_message)
                    user_received_message = True

                hb_req = hb_req or execution.heartbeat_requested
                self._session.add_tool_result(tc.id, name, execution.tool_result)

            if user_received_message:
                break

            if hb_req and self._heartbeat.can_heartbeat():
                self._heartbeat.request_heartbeat()
                _log.debug("Heartbeat — continuing...")
                continue

            _log.debug("Continuing for response...")

        return self._comm_exec.get_last_message()

    # -- Memory compression (background) --

    def _start_compression_if_needed(self) -> None:
        if self._recall_mem.get_count() <= conf.RECALL_MEMORY_LIMIT:
            return
        if not self._compression_lock.acquire(blocking=False):
            return

        def _compress_then_unlock():
            try:
                self._recall_mem.compress_old_memories(
                    self._llm.summarize,
                    self._archival_mem,
                )
            finally:
                self._compression_lock.release()

        t = threading.Thread(
            target=_compress_then_unlock,
            daemon=False,
            name="memory-compress",
        )
        t.start()
        self._compress_thread = t

    def wait_for_compression(self, timeout: float = 15.0) -> None:
        """Wait for in-flight memory compression (called during shutdown)."""
        compress_thread = getattr(self, "_compress_thread", None)
        if compress_thread is not None and compress_thread.is_alive():
            _log.info("Waiting for memory compression to finish...")
            compress_thread.join(timeout=timeout)
            if compress_thread.is_alive():
                _log.warning("Compression thread did not finish in time")

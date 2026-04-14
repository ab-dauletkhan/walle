"""
WALL-E Chat Loop — Mediator between LLM, tools, memory, and communication.

Extracted from WallELLMClient to isolate the MemGPT-style tool-calling loop
from object construction and lifecycle management.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

_DUMP_PROMPT = os.environ.get("WALLE_DUMP_PROMPT", "").lower() in ("1", "true", "yes")

from walle.memory.config import conf
from walle.memory.context_manager import InteractionContext, SensorSimulator

_log = logging.getLogger("walle.chat")

def _dump_messages_debug(messages, tools) -> None:
    """Print the exact compiled prompt with per-section char/rough-token counts.

    Enabled by WALLE_DUMP_PROMPT=1. Rough-token = chars // 4 (indicative only;
    real counts come from Ollama's usage.prompt_tokens in the response).
    """
    print("=" * 72, flush=True)
    print("WALLE_DUMP_PROMPT: compiled request", flush=True)
    print("=" * 72, flush=True)
    total_chars = 0
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        ch = len(content)
        total_chars += ch
        header = f"[{i}] role={role} chars={ch} (~{ch // 4} tok)"
        if msg.get("tool_calls"):
            header += f" tool_calls={len(msg['tool_calls'])}"
        if msg.get("tool_call_id"):
            header += f" tool_call_id={msg['tool_call_id']}"
        print(header, flush=True)
        if content:
            preview = content if len(content) < 2000 else content[:2000] + "\n...[truncated]"
            print(preview, flush=True)
        print("-" * 72, flush=True)

    tool_chars = len(json.dumps(tools)) if tools else 0
    total_chars += tool_chars
    print(
        f"[tools] count={len(tools or [])} json_chars={tool_chars} (~{tool_chars // 4} tok)",
        flush=True,
    )
    print(f"TOTAL: ~{total_chars} chars (~{total_chars // 4} tok rough)", flush=True)
    print("=" * 72, flush=True)


_MOTOR_TOOLS = frozenset(
    {
        "drive",
        "stop_movement",
        "get_robot_status",
    }
)


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
            msg["tool_calls"] = [asdict(tc) for tc in tool_calls]
        self.history.append(msg)

    def add_tool_result(self, tool_id, name, content):
        self.history.append(
            {"role": "tool", "tool_call_id": tool_id, "name": name, "content": content}
        )

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
        """Call LLM and return (content, tool_calls_list).

        Streams the response in real time, prints content as it arrives,
        and logs TTFT + tokens/sec in debug mode.
        """
        debug = _log.isEnabledFor(logging.DEBUG)

        if _DUMP_PROMPT:
            _dump_messages_debug(messages, tools)

        for attempt in range(max_retries):
            try:
                t_start = time.perf_counter()
                t_first: Optional[float] = None
                usage = None

                stream = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                    stream_options={"include_usage": True},
                )

                acc_content: list[str] = []
                acc_tools: dict[int, dict] = {}

                for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if delta.content:
                        if t_first is None:
                            t_first = time.perf_counter()
                        acc_content.append(delta.content)
                        if debug:
                            print(delta.content, end="", flush=True)

                    if delta.tool_calls:
                        if t_first is None:
                            t_first = time.perf_counter()
                        for tc_delta in delta.tool_calls:
                            slot = acc_tools.setdefault(
                                tc_delta.index,
                                {"id": "", "name": "", "arguments": ""},
                            )
                            if tc_delta.id:
                                slot["id"] = tc_delta.id
                            fn = tc_delta.function
                            if fn is not None:
                                if fn.name:
                                    slot["name"] += fn.name
                                if fn.arguments:
                                    slot["arguments"] += fn.arguments

                t_end = time.perf_counter()
                content = "".join(acc_content)

                tool_calls_list = [
                    ToolCallEnvelope(
                        id=slot["id"],
                        function=ToolFunctionCall(
                            name=slot["name"],
                            arguments=slot["arguments"] or "{}",
                        ),
                    )
                    for _, slot in sorted(acc_tools.items())
                ]

                if debug:
                    if acc_content:
                        print()
                    total = t_end - t_start
                    # Prefer real token counts from Ollama's usage payload.
                    if usage is not None:
                        prompt_tokens = getattr(usage, "prompt_tokens", None)
                        completion_tokens = getattr(usage, "completion_tokens", None)
                    else:
                        prompt_tokens = None
                        completion_tokens = None
                    n_out = completion_tokens if completion_tokens is not None else (
                        len(content.split())
                        + sum(len(s["arguments"].split()) for s in acc_tools.values())
                    )
                    if t_first is not None:
                        ttft = t_first - t_start
                        gen = max(t_end - t_first, 1e-6)
                        rate = n_out / gen
                        if prompt_tokens is not None:
                            _log.debug(
                                "[prompt %d tok | TTFT %.2fs => prefill %.1f tok/s] [%d out tok @ %.1f tok/s] [total %.2fs]",
                                prompt_tokens,
                                ttft,
                                prompt_tokens / max(ttft, 1e-6),
                                n_out,
                                rate,
                                total,
                            )
                        else:
                            _log.debug(
                                "[TTFT %.2fs] [%d tok @ %.1f tok/s] [total %.2fs]",
                                ttft,
                                n_out,
                                rate,
                                total,
                            )
                    else:
                        _log.debug("[no tokens] [total %.2fs]", total)

                return content, tool_calls_list

            except Exception as e:
                _log.warning(
                    "LLM error (attempt %d/%d): %s", attempt + 1, max_retries, e
                )
                time.sleep(1)

        _log.error("Failed to generate LLM response after %d retries", max_retries)
        return (
            "I'm having trouble reaching my brain right now. Please check that Ollama is running.",
            [],
        )

    def summarize(self, text: str) -> str:
        """Generate a concise summary for memory compression."""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": f"/no_think\n\nSummarize this concisely:\n{text}",
                    }
                ],
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
        on_turn_start=None,
        on_turn_end=None,
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
        self._on_turn_start = on_turn_start
        self._on_turn_end = on_turn_end
        self._compression_lock = threading.Lock()

    @property
    def session(self) -> ChatSession:
        return self._session

    def run(self, user_input: str) -> Optional[str]:
        """Run one full MemGPT-style chat turn. Returns the message for the user."""
        if self._on_turn_start:
            self._on_turn_start()

        # Drop any stale message from a prior interrupted turn
        self._comm_exec.get_last_message()

        # 1. Context update
        interaction_count = self._recall_mem.get_count()
        self._context_manager.update_interaction(
            InteractionContext(
                last_interaction=datetime.now(),
                interaction_count=interaction_count,
            )
        )
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
        result = self._tool_loop(memory_context)

        if self._on_turn_end:
            self._on_turn_end()

        return result

    _MAX_ITERATIONS = 20
    _MAX_STALE = 3

    def _tool_loop(self, memory_context: str) -> Optional[str]:
        """Inner loop: call LLM → execute tools → repeat until send_message."""
        iteration = 0
        stale_count = 0
        user_received_message = False

        while iteration < self._MAX_ITERATIONS:
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
                        stale_count += 1
                        if stale_count >= self._MAX_STALE:
                            _log.debug(
                                "Stale after %d attempts without send_message",
                                stale_count,
                            )
                            break
                        _log.debug(
                            "No send_message called, prompting... (%d/%d)",
                            stale_count,
                            self._MAX_STALE,
                        )
                        continue
                break

            # Validate all tool call arguments before committing to session
            parsed_calls = []
            valid = True
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    parsed_calls.append((tc, args))
                except json.JSONDecodeError as e:
                    _log.warning("Malformed tool call '%s': %s", tc.function.name, e)
                    valid = False
                    break

            if not valid:
                # Don't poison session history — discard and retry
                continue

            self._session.add("assistant", content or "", tool_calls)
            stale_count = 0
            hb_req = False

            for tc, args in parsed_calls:
                execution = self._tool_suite.execute_tool(
                    name=tc.function.name,
                    args=args,
                    user_message_already_sent=user_received_message,
                )

                if execution.user_message and not user_received_message:
                    _log.debug("send_message: %s", execution.user_message[:80])
                    self._recall_mem.insert("assistant", execution.user_message)
                    user_received_message = True

                # Update robot context for movement tools
                if tc.function.name in _MOTOR_TOOLS:
                    self._context_manager.update_robot(
                        command=tc.function.name,
                        result=execution.tool_result,
                        motors_active="continuous" in execution.tool_result,
                    )

                hb_req = hb_req or execution.heartbeat_requested
                self._session.add_tool_result(
                    tc.id, tc.function.name, execution.tool_result
                )

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

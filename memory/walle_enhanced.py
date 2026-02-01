"""
WALL-E v3.1 - Memory-Augmented Robot Brain
Features: Ollama/Qwen3 Backend, FAISS Search, Importance Decay, Streaming
Optimized for NVIDIA Jetson Orin Nano (8GB VRAM)
"""
import json
import re
import time
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from config import conf

# Import subsystems
from memory_system import Memory, RecallMemory, ArchivalMemory
from memory_tools import get_memory_tools, MemoryToolExecutor
from personality_system import PersonalityEngine, get_personality_tools, PersonalityToolExecutor
from robot_tools import get_robot_control_tools, RobotControlExecutor, get_robot_tool_names
from heartbeat import HeartbeatManager, add_heartbeat_to_tools
from knowledge_tools import get_knowledge_tools, KnowledgeToolExecutor
from context_manager import ContextManager, EnvironmentContext, InteractionContext, SensorSimulator
from communication_tools import get_communication_tools, CommunicationExecutor, SendMessageArgs

# Initialize Client - Ollama is the recommended backend
from openai import OpenAI
print(f"🚀 Initializing Ollama backend with {conf.OLLAMA_MODEL}...")
client = OpenAI(base_url=f"{conf.OLLAMA_BASE_URL}/v1", api_key="ollama")
MODEL_NAME = conf.OLLAMA_MODEL

# Initialize Systems
core_mem = Memory()
recall_mem = RecallMemory(use_semantic=conf.USE_SEMANTIC_SEARCH)
archival_mem = ArchivalMemory(use_semantic=conf.USE_SEMANTIC_SEARCH)
mem_exec = MemoryToolExecutor(core_mem, recall_mem, archival_mem)
personality = PersonalityEngine.load()
personality_exec = PersonalityToolExecutor(personality)
robot = RobotControlExecutor(serial_port=conf.SERIAL_PORT, baud_rate=conf.BAUD_RATE)
heartbeat = HeartbeatManager()
knowledge_exec = KnowledgeToolExecutor()
context_manager = ContextManager()
comm_exec = CommunicationExecutor()

# --- TIMER CLASS ---
class PerformanceTimer:
    def __init__(self):
        self.start_time = 0
        self.first_token_time = 0
        self.end_time = 0
        self.token_count = 0
    
    def start(self):
        self.start_time = time.perf_counter()
        print(f"\n⏱️  [Timer Started]", end="", flush=True)
        
    def mark_first_token(self):
        if self.first_token_time == 0:
            self.first_token_time = time.perf_counter()
            
    def stop(self):
        self.end_time = time.perf_counter()
    
    def report(self):
        if self.start_time == 0 or self.end_time == 0: return
        
        total_duration = self.end_time - self.start_time
        ttft = self.first_token_time - self.start_time if self.first_token_time > 0 else total_duration
        tps = self.token_count / total_duration if total_duration > 0 else 0
        
        print(f"\n   [⏱️ Metrics: TTFT: {ttft*1000:.0f}ms | Total: {total_duration:.2f}s | Speed: {tps:.1f} tok/s]")

def summarize_text(text: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Summarize this concisely:\n{text}"}],
            max_tokens=200
        )
        return resp.choices[0].message.content
    except Exception:
        return text[:500] + "..."

def retrieve_relevant_context(query: str) -> str:
    """Retrieve relevant context from memory (parallel search for ~50ms speedup)."""
    # Parallel search: recall and archival run concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        recall_future = executor.submit(recall_mem.search, query, 3)
        archival_future = executor.submit(archival_mem.search, query, 2)
        hits = recall_future.result()
        facts = archival_future.result()

    if not hits and not facts:
        return ""

    ctx = "\n[RELEVANT MEMORIES]\n"
    for h in hits:
        ctx += f"- {h['role']}: {h['content']}\n"
    for f in facts:
        ctx += f"- Fact: {f['content']}\n"
    return ctx + "\n"

def get_system_prompt():
    context_str = context_manager.get_context_string()
    return f"""You are WALL-E, a helpful robot companion.
{personality.get_system_prompt_addition()}

RULES:
1. Your text = internal thought (user can't see)
2. To reply to user: call send_message tool
3. To remember info: call core_memory_append THEN send_message
4. Keep thoughts SHORT. Act, don't over-analyze.

IMPORTANT: Actually CALL the tools, don't just think about them!

{context_str}
{core_mem.compile()}
"""

class ChatSession:
    def __init__(self):
        self.history = deque(maxlen=conf.MAX_CONTEXT_MESSAGES)
    
    def add(self, role, content, tool_calls=None):
        msg = {"role": role, "content": content}
        if tool_calls: msg["tool_calls"] = tool_calls
        self.history.append(msg)
    
    def add_tool_result(self, tool_id, name, content):
        self.history.append({"role": "tool", "tool_call_id": tool_id, "name": name, "content": content})

    def get_messages(self):
        return [{"role": "system", "content": get_system_prompt()}] + list(self.history)

session = ChatSession()

def stream_chat_response(messages, tools, max_retries=2, show_inner_thoughts=True):
    """
    Stream response with inner monologue support.

    In MemGPT style:
    - Raw content is inner thoughts (shown with 💭 prefix if show_inner_thoughts=True)
    - User only sees send_message tool outputs
    """
    timer = PerformanceTimer()

    for attempt in range(max_retries):
        try:
            timer.start()

            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=True
            )

            full_content = ""
            tool_calls_map = {}
            has_tool_calls = False
            inner_thought_started = False

            for chunk in stream:
                timer.mark_first_token()

                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    timer.token_count += 1
                    # Show inner thoughts with prefix (debug mode)
                    if show_inner_thoughts:
                        if not inner_thought_started:
                            print("   💭 ", end="", flush=True)
                            inner_thought_started = True
                        print(delta.content, end="", flush=True)

                if delta.tool_calls:
                    has_tool_calls = True
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {"id": "", "func": {"name": "", "args": ""}}
                        if tc.id: tool_calls_map[idx]["id"] += tc.id
                        if tc.function.name: tool_calls_map[idx]["func"]["name"] += tc.function.name
                        if tc.function.arguments: tool_calls_map[idx]["func"]["args"] += tc.function.arguments

            if inner_thought_started:
                print()  # Newline after inner thoughts

            timer.stop()
            timer.report()

            # Reconstruct Tools
            tool_calls_list = []
            for idx in sorted(tool_calls_map.keys()):
                t = tool_calls_map[idx]
                from openai.types.chat import ChatCompletionMessageToolCall
                from openai.types.chat.chat_completion_message_tool_call import Function
                tool_calls_list.append(ChatCompletionMessageToolCall(
                    id=t["id"] or f"call_{idx}",
                    function=Function(name=t["func"]["name"], arguments=t["func"]["args"]),
                    type="function"
                ))

            return full_content, tool_calls_list

        except Exception as e:
            print(f"\n⚠️ Stream interrupted (Attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(1)

    print("❌ Failed to generate response.")
    return "", []

def chat(user_input: str):
    # 1. Context & Memory Setup
    context_manager.update_interaction(InteractionContext(
        last_interaction=datetime.now(),
        interaction_count=recall_mem.get_count()
    ))
    
    # Simulate sensor update occasionally
    if recall_mem.get_count() % 5 == 0:
        context_manager.update_environment(SensorSimulator.simulate_environment_context(battery=80 - recall_mem.get_count()))

    memory_context = retrieve_relevant_context(user_input)
    full_input = memory_context + user_input if memory_context else user_input
    
    # Insert with deferred embedding (~100ms TTFT improvement)
    # Embedding is generated in background thread after LLM starts
    recall_mem.insert("user", user_input, defer_embedding=True)
    session.add("user", full_input)
    heartbeat.reset()
    
    # 2. Maintenance
    if recall_mem.get_count() > conf.RECALL_MEMORY_LIMIT:
        recall_mem.compress_old_memories(summarize_text, archival_mem)

    # 3. Execution Loop (MemGPT-style inner monologue)
    iteration = 0
    user_received_message = False

    while iteration < 10:
        iteration += 1
        # Include send_message as FIRST tool - it's the primary way to respond
        tools = get_communication_tools() + get_robot_control_tools() + get_memory_tools() + get_personality_tools() + get_knowledge_tools()
        tools = add_heartbeat_to_tools(tools)

        content, tool_calls = stream_chat_response(session.get_messages(), tools)

        # No tool calls = just inner thought, need to prompt for send_message
        if not tool_calls:
            if content:
                # Inner thought without action - save but don't show
                session.add("assistant", content)
                # If no message was sent to user, we need another turn
                if not user_received_message:
                    print("   ⚠️ No send_message called, prompting...")
                    continue
            break

        session.add("assistant", content or "", tool_calls)
        hb_req = False

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON Error in args for {name}: {e}")
                session.add_tool_result(tc.id, name, f"Error: Invalid JSON arguments - {e}")
                continue

            if args.pop("request_heartbeat", False):
                hb_req = True

            # Handle send_message - THIS is what user sees
            if name == "send_message":
                try:
                    validated = SendMessageArgs.model_validate(args)
                    message = validated.message
                except Exception as e:
                    print(f"   ⚠️ Validation error: {e}")
                    session.add_tool_result(tc.id, name, f"Validation error: {e}")
                    continue

                if message and not user_received_message:  # Only first message
                    print(f"🤖 WALL-E: {message}")
                    recall_mem.insert("assistant", message)
                    user_received_message = True
                    result = "Message delivered to user"
                    # Note: ignore request_heartbeat for send_message - response is done
                else:
                    result = "Message already sent (duplicate ignored)"
                session.add_tool_result(tc.id, name, result)
                continue

            print(f"   ⚙️ Tool: {name}")

            try:
                if name == "consult_internet_for_facts":
                    result = knowledge_exec.execute(name, args)
                    hb_req = True
                elif name in get_robot_tool_names():
                    result = robot.execute(name, args)
                elif name == "set_personality":
                    result = personality_exec.execute(name, args)
                else:
                    result = mem_exec.execute(name, args)
            except Exception as e:
                result = f"Error: {e}"

            session.add_tool_result(tc.id, name, str(result)[:1000])
            if name == "consult_internet_for_facts":
                print(f"   🌍 Search feedback loop active...")

        # If user got a message, we're done - don't continue even with heartbeat
        if user_received_message:
            break

        # Only continue for heartbeat if we haven't responded yet
        if hb_req and heartbeat.can_heartbeat():
            heartbeat.request_heartbeat()
            print("   💓 Thinking chain active...")
            continue

        # Otherwise continue to get send_message
        print("   (Continuing for response...)")

def main():
    print(f"🤖 WALL-E Online v3.1 - Ollama ({conf.OLLAMA_MODEL})")
    print(f"   Memory: {'Semantic' if conf.USE_SEMANTIC_SEARCH else 'Text'} Search | Embedding: {conf.EMBEDDING_MODEL}")

    if not conf.validate():
        print(f"⚠️ System checks failed. Please check Ollama configuration.")
        print(f"   Make sure Ollama is running: ollama serve")
        print(f"   And model is pulled: ollama pull {conf.OLLAMA_MODEL}")
        x = input("Continue anyway? (y/n): ")
        if x.lower() != 'y': return

    context_manager.update_environment(SensorSimulator.simulate_environment_context())
    
    try:
        while True:
            try:
                u = input("\nYou: ").strip()
                if not u: continue
                if u.lower() in ['exit', 'quit']:
                    print("\n👋 Shutting down WALL-E...")
                    break
                chat(u)
            except KeyboardInterrupt:
                print("\n\n👋 Keyboard interrupt received. Shutting down...")
                break
    finally:
        # Cleanup resources
        robot.close()
        print("✅ WALL-E shutdown complete")

if __name__ == "__main__":
    main()
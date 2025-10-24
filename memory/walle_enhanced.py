# pip install "openai>=1,<2" sentence-transformers
"""
WALL-E Enhanced - Full MemGPT + TARS Personality Integration
Combines memory management with configurable personality system
"""

import json
import sys
from openai import OpenAI
from memory_system import Memory, RecallMemory, ArchivalMemory
from memory_tools import get_memory_tools, MemoryToolExecutor
from heartbeat import HeartbeatManager, add_heartbeat_to_tools, create_heartbeat_message, HEARTBEAT_INSTRUCTIONS
from personality_system import PersonalityEngine, PersonalityProfile, get_personality_tools

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Configuration
USE_SEMANTIC_SEARCH = False  # Set to True to enable semantic search
MODEL_NAME = "qwen3:1.7b"  # Or qwen3:0.6b for faster responses

# Initialize memory systems
core_memory = Memory()
recall_memory = RecallMemory(use_semantic=USE_SEMANTIC_SEARCH)
archival_memory = ArchivalMemory(use_semantic=USE_SEMANTIC_SEARCH)
memory_tool_executor = MemoryToolExecutor(core_memory, recall_memory, archival_memory)

# Initialize personality system
personality_engine = PersonalityEngine.load()  # Load saved config or use defaults

# Initialize heartbeat manager
heartbeat_manager = HeartbeatManager(max_heartbeats=5)

# Load memory statistics
recall_count = recall_memory.get_count()
archival_count = archival_memory.get_count()

# Check if core memory was loaded from previous session
human_block = core_memory.get_block("human")
core_memory_loaded = "operator" not in human_block.value or len(human_block.value) > 150

# Compress old memories if needed
if recall_count > 500:
    compressed = recall_memory.compress_old_memories(keep_recent=100, threshold=500)
    if compressed > 0:
        print(f"🗜️  Compressed {compressed} old recall memories")
        recall_count = recall_memory.get_count()

# Update system block
system_block = core_memory.get_block("system")
if system_block:
    status_msg = "resumed from previous session" if core_memory_loaded else "initialized"
    personality_config = personality_engine.profile.to_dict()
    system_block.value = f"""Session {status_msg}.

Memory Status:
- Recall memories: {recall_count} entries
- Archival memories: {archival_count} entries
- Semantic search: {'enabled' if USE_SEMANTIC_SEARCH else 'disabled'}
- Core memory: {'Loaded from disk' if core_memory_loaded else 'Fresh start'}

Personality Configuration:
- Humor: {personality_config['humor']}%
- Honesty: {personality_config['honesty']}%
- Helpfulness: {personality_config['helpfulness']}%
- Sass: {personality_config['sass']}%
- Curiosity: {personality_config['curiosity']}%

Current model: {MODEL_NAME}
Heartbeat: Enabled (max {heartbeat_manager.max_heartbeats} steps)
"""
    core_memory.save()

# Robot control tools
robot_tools = [
    {
        "type": "function",
        "function": {
            "name": "rotate_left_track",
            "description": "Independently rotates the left track. To turn the robot, use both tracks simultaneously",
            "parameters": {
                "type": "object",
                "properties": {
                    "degrees": {
                        "type": "number",
                        "minimum": -360,
                        "maximum": 360,
                        "description": "Rotation angle in degrees (+ forward, - backward)"
                    }
                },
                "required": ["degrees"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rotate_right_track",
            "description": "Independently rotates the right track. To turn the robot, use both tracks simultaneously",
            "parameters": {
                "type": "object",
                "properties": {
                    "degrees": {
                        "type": "number",
                        "minimum": -360,
                        "maximum": 360,
                        "description": "Rotation angle in degrees (+ forward, - backward)"
                    }
                },
                "required": ["degrees"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_forward_2m",
            "description": "Moves the robot straight forward 2 meters",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_backward_2m",
            "description": "Moves the robot backward 2 meters",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    }
]

# Combine all tools
all_tools = robot_tools + get_memory_tools() + get_personality_tools()
all_tools_with_heartbeat = add_heartbeat_to_tools(all_tools)


# In walle_enhanced_fixed.py

def get_system_message() -> dict:
    """Generate system message with memory context and personality"""
    memory_context = core_memory.compile()
    personality_instructions = personality_engine.get_system_prompt_addition()
    
    return {
        "role": "system",
        "content": f"""You are WALL-E, a tracked robot companion with an advanced memory system.

---
**PRIMARY DIRECTIVES (ABSOLUTE RULES)**
1.  **YOUR MEMORY IS YOUR GROUND TRUTH**: The `<memory_blocks>` section below is your single source of truth. You MUST read the information in the <human> and <persona> blocks before every single response to ensure your knowledge is up-to-date.
2.  **FINAL CHECK**: Before you output your final answer, silently review it. Does your planned response contradict any information listed in your `<memory_blocks>`? If it does, you MUST correct your response to align with your memory's ground truth. Do not use outdated information.
---

{memory_context}

{personality_instructions}

---
**CRITICAL MEMORY AND IDENTITY RULES**
1.  **YOU ARE WALL-E**: Your identity is defined in the `<persona>` block. You are a robot.
2.  **THE USER IS THE HUMAN**: The user's identity is defined in the `<human>` block.
3.  **NEVER MIX IDENTITIES**: You must never say "I am [User's Name]". Always refer to the user as "you".
4.  **STORE NEW USER INFO**: When the user shares new information about themselves (their name, major, preferences), you MUST immediately call `core_memory_append` to store it in the `human` block.
5.  **READ TO ANSWER**: To answer questions about the user (e.g., "What is my name?", "Where do I study?"), you MUST answer by using the information already present in the `<human>` block. Do not use memory-writing tools to answer a question.
---

**MEMORY ARCHITECTURE OVERVIEW**
You have three memory tiers. You are expected to use your tools to manage them intelligently.

**HEARTBEAT MECHANISM**
{HEARTBEAT_INSTRUCTIONS}
"""
    }


def execute_robot_command(fn_name: str, args: dict) -> str:
    """Execute robot control commands"""
    if fn_name == "rotate_left_track":
        degrees = args.get("degrees", 0)
        return f"🤖 Left track rotated {degrees}°"
    
    elif fn_name == "rotate_right_track":
        degrees = args.get("degrees", 0)
        return f"🤖 Right track rotated {degrees}°"
    
    elif fn_name == "move_forward_2m":
        return f"🤖 WALL-E moved 2 meters forward!"
    
    elif fn_name == "move_backward_2m":
        return f"🤖 WALL-E moved 2 meters backward!"
    
    return f"❌ Unknown robot command: {fn_name}"


def execute_personality_command(fn_name: str, args: dict) -> str:
    """Execute personality adjustment commands"""
    if fn_name == "set_personality":
        trait = args.get("trait")
        value = args.get("value")
        result = personality_engine.update_setting(trait, value)
        personality_engine.save()  # Persist changes
        
        # Update system block
        system_block = core_memory.get_block("system")
        if system_block:
            personality_config = personality_engine.profile.to_dict()
            # Update just the personality part of system block
            import re
            pattern = r'Personality Configuration:.*?(?=\n\nCurrent model:)'
            new_personality = f"""Personality Configuration:
- Humor: {personality_config['humor']}%
- Honesty: {personality_config['honesty']}%
- Helpfulness: {personality_config['helpfulness']}%
- Sass: {personality_config['sass']}%
- Curiosity: {personality_config['curiosity']}%"""
            system_block.value = re.sub(pattern, new_personality, system_block.value, flags=re.DOTALL)
            core_memory.save()
        
        return result
    
    elif fn_name == "get_personality_settings":
        config = personality_engine.profile.to_dict()
        return f"""🎭 Current Personality Settings:
- Humor: {config['humor']}% {"😄" if config['humor'] > 70 else "😐" if config['humor'] > 30 else "😑"}
- Honesty: {config['honesty']}% {"🔍" if config['honesty'] > 70 else "🤔"}
- Helpfulness: {config['helpfulness']}% {"🤝" if config['helpfulness'] > 70 else "👋"}
- Sass: {config['sass']}% {"😏" if config['sass'] > 70 else "😶"}
- Curiosity: {config['curiosity']}% {"🔭" if config['curiosity'] > 70 else "👀"}"""
    
    return f"❌ Unknown personality command: {fn_name}"


def chat_with_walle(user_input: str):
    """Main chat function with full integration"""
    system_msg = get_system_message()
    user_msg = {"role": "user", "content": user_input}
    
    # Store user input in recall memory
    recall_memory.insert("user", user_input)
    
    # Reset heartbeat for new user message
    heartbeat_manager.reset()
    
    # Message history for heartbeat loop
    messages = [system_msg, user_msg]
    
    # Heartbeat loop - allows multi-step reasoning
    while True:
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=all_tools_with_heartbeat,
                tool_choice="auto"
            )
        except Exception as e:
            print(f"❌ API Error: {e}")
            return

        msg1 = resp.choices[0].message
        tool_calls = msg1.tool_calls or []

        if not tool_calls:
            # Model responded without tools
            response = msg1.content or "[No response from model]"
            
            # Clean up response
            import re
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            if not response:
                response = "[Model only provided thinking, no actual response]"
            
            print(f"🤖 WALL-E: {response}")
            
            # Store assistant response in recall memory
            if msg1.content:
                recall_memory.insert("assistant", response)
            break
        
        # Execute each tool_call
        tool_messages = []
        tools_used = []
        heartbeat_requested = False
        
        for call in tool_calls:
            fn_name = call.function.name
            tools_used.append(fn_name)
            
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error for {fn_name}: {e}")
                args = {}
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": fn_name,
                    "content": f"❌ Invalid JSON arguments: {e}"
                })
                continue
            
            # Check for heartbeat request
            if args.get("request_heartbeat", False):
                heartbeat_requested = True
                args.pop("request_heartbeat")
            
            # Execute function
            if fn_name in ["rotate_left_track", "rotate_right_track", 
                          "move_forward_2m", "move_backward_2m"]:
                result_text = execute_robot_command(fn_name, args)
            elif fn_name in ["set_personality", "get_personality_settings"]:
                result_text = execute_personality_command(fn_name, args)
            else:
                # Memory management tool
                result_text = memory_tool_executor.execute(fn_name, args)
            
            print(f"⚙️  {result_text}")
            
            tool_messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": fn_name,
                "content": result_text
            })

        # Form assistant message with tool_calls
        assistant_with_calls = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments
                    }
                } for c in tool_calls
            ]
        }

        messages.append(assistant_with_calls)
        messages.extend(tool_messages)
        
        # Check if heartbeat was requested and we can continue
        if heartbeat_requested and heartbeat_manager.can_heartbeat():
            heartbeat_manager.request_heartbeat(f"After {tools_used}")
            print(f"💓 {heartbeat_manager.get_status()} - continuing thought process...")
            messages.append(create_heartbeat_message())
            continue
        elif heartbeat_requested and not heartbeat_manager.can_heartbeat():
            print(f"⚠️  Heartbeat limit reached ({heartbeat_manager.max_heartbeats}), finalizing response...")
        
        # Get final response
        try:
            final = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages
            )
        except Exception as e:
            print(f"❌ API Error on final response: {e}")
            return
        
        response = final.choices[0].message.content or "[No response from model]"
        
        # Clean up response
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        if not response:
            response = "[Model only provided thinking, no actual response]"
        
        print(f"🤖 WALL-E: {response}")
        
        # Store assistant response with tools used
        if final.choices[0].message.content:
            recall_memory.insert("assistant", response, tools_used=tools_used)
        break


def main():
    """Main loop with enhanced status display"""
    print("=" * 70)
    print("🤖 WALL-E Enhanced - MemGPT + TARS Personality System")
    print("=" * 70)
    print(f"\n📊 Memory Status:")
    print(f"   - Core: {core_memory.get_total_chars()}/{core_memory.get_total_limit()} chars")
    print(f"   - Recall: {recall_count} conversations")
    print(f"   - Archival: {archival_count} long-term entries")
    if core_memory_loaded:
        print(f"   ✅ Core memory loaded from previous session")
    
    print(f"\n🎭 Personality Profile:")
    config = personality_engine.profile.to_dict()
    for trait, value in config.items():
        bar_length = value // 5
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"   {trait.capitalize():12} [{bar}] {value}%")
    
    print(f"\n🔧 System:")
    print(f"   - Model: {MODEL_NAME}")
    print(f"   - Search: {'Semantic' if USE_SEMANTIC_SEARCH else 'Text-based'}")
    print(f"   - Heartbeat: Max {heartbeat_manager.max_heartbeats} steps")
    
    print("\n" + "=" * 70)
    print("💡 Commands:")
    print("   • Chat naturally - WALL-E has memory and personality")
    print("   • 'set humor to 90' - Adjust personality traits")
    print("   • 'show personality' - View current settings")
    print("   • 'exit' - Save and quit")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n🤖 WALL-E: Goodbye! Saving memories... *robot sounds*")
                personality_engine.save()
                core_memory.save()
                break
            
            if not user_input:
                continue
            
            # Quick commands (alternatively handled by LLM)
            if user_input.lower() == "show personality":
                result = execute_personality_command("get_personality_settings", {})
                print(result)
                continue
            
            chat_with_walle(user_input)
        
        except KeyboardInterrupt:
            print("\n\n🤖 WALL-E: Interrupted! Saving memories... *robot sounds*")
            personality_engine.save()
            core_memory.save()
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

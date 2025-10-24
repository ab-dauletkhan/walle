"""
Memory management tools that the LLM can call to manage its own memory
Based on MemGPT function calling architecture
"""

from typing import Dict, List, Any
from memory_system import Memory, RecallMemory, ArchivalMemory


# In memory_tools.py

def get_memory_tools() -> List[Dict]:
    """
    Returns OpenAI-compatible tool definitions for memory management
    These tools allow the LLM to edit its own memory
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "core_memory_append",
                "description": """Append content to one of your core memory blocks.
                
CRITICAL RULE: Use 'human' block for information ABOUT THE USER. Use 'persona' block for information ABOUT YOURSELF (WALL-E). Do NOT mix these up.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["persona", "human"],
                            "description": "The memory block to append to. Use 'human' for user info, 'persona' for your own."
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to append to the memory block"
                        }
                    },
                    "required": ["label", "content"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "core_memory_replace",
                "description": """Replace or edit specific content within a core memory block.
                
CRITICAL RULE: Use 'human' block to edit info ABOUT THE USER. Use 'persona' block for info ABOUT YOURSELF (WALL-E).""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["persona", "human"],
                            "description": "The memory block to edit. Use 'human' for user info, 'persona' for your own."
                        },
                        "old_content": {
                            "type": "string",
                            "description": "The exact text to find and replace"
                        },
                        "new_content": {
                            "type": "string",
                            "description": "The new text to replace it with"
                        }
                    },
                    "required": ["label", "old_content", "new_content"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "archival_memory_insert",
                "description": "Store important, long-term facts, preferences, or detailed information in archival memory for future recall.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Category to organize the information (e.g., 'user_preferences', 'user_projects', 'learned_facts')"
                        },
                        "content": {
                            "type": "string",
                            "description": "The information to store"
                        },
                        "importance": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Importance level (1=trivial, 10=critical)."
                        }
                    },
                    "required": ["category", "content"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "archival_memory_search",
                "description": "Search long-term archival memory for stored FACTS and INFORMATION. DO NOT use this to remember the last few turns of a conversation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query - what you're looking for"
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional: filter results to a specific category"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "description": "Maximum results to return (default 5)"
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "recall_memory_search",
                "description": "Search RECENT CONVERSATION history. Use this to answer questions like 'What did we just talk about?' or 'What did I ask you a moment ago?'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional: search query. Omit to get the most recent history."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Maximum results to return (default 10)"
                        }
                    },
                    "required": [],
                    "additionalProperties": False
                }
            }
        }
    ]


class MemoryToolExecutor:
    """Executes memory management tool calls"""
    
    def __init__(self, memory: Memory, recall_memory: RecallMemory, archival_memory: ArchivalMemory):
        self.memory = memory
        self.recall_memory = recall_memory
        self.archival_memory = archival_memory
    
    def execute(self, function_name: str, args: Dict[str, Any]) -> str:
        """Execute a memory management function"""
        
        if function_name == "core_memory_append":
            return self._core_memory_append(args)
        elif function_name == "core_memory_replace":
            return self._core_memory_replace(args)
        elif function_name == "archival_memory_insert":
            return self._archival_memory_insert(args)
        elif function_name == "archival_memory_search":
            return self._archival_memory_search(args)
        elif function_name == "recall_memory_search":
            return self._recall_memory_search(args)
        else:
            return f"❌ Unknown memory function: {function_name}"
    
    def _core_memory_append(self, args: Dict) -> str:
        """Append content to a core memory block"""
        label = args.get("label")
        content = args.get("content")
        
        block = self.memory.get_block(label)
        if not block:
            return f"❌ Memory block '{label}' not found"
        
        success, message = block.append("\n" + content)
        if success:
            self.memory.save()  # Persist to disk
            return f"✅ Appended to {label} memory ({block.chars_current}/{block.limit} chars)"
        else:
            return f"❌ Failed to append: {message}"
    
    def _core_memory_replace(self, args: Dict) -> str:
        """Replace content in a core memory block"""
        label = args.get("label")
        old_content = args.get("old_content")
        new_content = args.get("new_content")
        
        block = self.memory.get_block(label)
        if not block:
            return f"❌ Memory block '{label}' not found"
        
        success, message = block.replace(old_content, new_content)
        if success:
            self.memory.save()  # Persist to disk
            return f"✅ Replaced content in {label} memory ({block.chars_current}/{block.limit} chars)"
        else:
            return f"❌ Failed to replace: {message}"
    
    def _archival_memory_insert(self, args: Dict) -> str:
        """Insert information into archival memory"""
        category = args.get("category")
        content = args.get("content")
        importance = args.get("importance", 5)
        
        self.archival_memory.insert(category, content, importance)
        count = self.archival_memory.get_count()
        
        return f"✅ Stored in archival memory [{category}] (importance: {importance}/10, total: {count} entries)"
    
    def _archival_memory_search(self, args: Dict) -> str:
        """Search archival memory"""
        query = args.get("query")
        category = args.get("category")
        limit = args.get("limit", 5)
        
        results = self.archival_memory.search(query, category, limit)
        
        if not results:
            return f"📭 No archival memories found for query: '{query}'"
        
        output = f"📚 Found {len(results)} archival memories:\n"
        for i, result in enumerate(results, 1):
            output += f"\n{i}. [{result['category']}] (importance: {result['importance']}/10)\n"
            output += f"   {result['content'][:200]}...\n" if len(result['content']) > 200 else f"   {result['content']}\n"
        
        return output
    
    def _recall_memory_search(self, args: Dict) -> str:
        """Search recall memory (recent conversations)"""
        query = args.get("query")
        limit = args.get("limit", 10)
        
        results = self.recall_memory.search(query, limit)
        
        if not results:
            if query:
                return f"📭 No recall memories found for query: '{query}'"
            else:
                return f"📭 No recent conversations found"
        
        output = f"💭 Found {len(results)} recall memories:\n"
        for i, result in enumerate(results, 1):
            output += f"\n{i}. [{result['role']}] {result['timestamp']}\n"
            output += f"   {result['content'][:150]}...\n" if len(result['content']) > 150 else f"   {result['content']}\n"
        
        return output

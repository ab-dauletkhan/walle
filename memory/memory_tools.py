"""
Memory management tools for LLM interaction
"""
from typing import Dict, List, Any
from memory_system import Memory, RecallMemory, ArchivalMemory

def get_memory_tools() -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "core_memory_append",
                "description": "Append to 'human' (user info) or 'persona' (self info).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": ["persona", "human"]},
                        "content": {"type": "string"}
                    },
                    "required": ["label", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "core_memory_replace",
                "description": "Edit specific text in core memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": ["persona", "human"]},
                        "old_content": {"type": "string"},
                        "new_content": {"type": "string"}
                    },
                    "required": ["label", "old_content", "new_content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "archival_memory_insert",
                "description": "Save a fact or preference to long-term storage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "content": {"type": "string"},
                        "importance": {"type": "integer", "minimum": 1, "maximum": 10}
                    },
                    "required": ["category", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "recall_memory_search",
                "description": "Search recent conversation history.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"}
                    }
                }
            }
        }
    ]

class MemoryToolExecutor:
    def __init__(self, memory: Memory, recall: RecallMemory, archival: ArchivalMemory):
        self.memory = memory
        self.recall = recall
        self.archival = archival

    def execute(self, name: str, args: dict) -> str:
        method = getattr(self, f"_{name}", None)
        if method: return method(args)
        return f"Unknown tool: {name}"

    def _core_memory_append(self, args):
        block = self.memory.get_block(args['label'])
        if not block: return "Block not found"
        success, msg = block.append("\n" + args['content'])
        if success: self.memory.save()
        return f"Append result: {msg}"

    def _core_memory_replace(self, args):
        block = self.memory.get_block(args['label'])
        if not block: return "Block not found"
        success, msg = block.replace(args['old_content'], args['new_content'])
        if success: self.memory.save()
        return f"Replace result: {msg}"

    def _archival_memory_insert(self, args):
        self.archival.insert(args['category'], args['content'], args.get('importance', 5))
        return "Saved to archival memory."

    def _recall_memory_search(self, args):
        res = self.recall.search(args.get('query'), args.get('limit', 5))
        return f"Results: {res}"
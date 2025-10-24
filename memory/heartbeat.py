"""
Heartbeat mechanism for Chain of Thought
Allows LLM to chain multiple function calls without user interruption
"""

from typing import List, Dict, Any, Optional
import json


class HeartbeatManager:
    """
    Manages the heartbeat mechanism for multi-step reasoning
    LLM can request a heartbeat to continue thinking without user input
    """
    
    def __init__(self, max_heartbeats: int = 5):
        """
        Args:
            max_heartbeats: Maximum number of consecutive heartbeats to prevent infinite loops
        """
        self.max_heartbeats = max_heartbeats
        self.heartbeat_count = 0
        self.heartbeat_history = []
    
    def reset(self):
        """Reset heartbeat counter for a new user message"""
        self.heartbeat_count = 0
        self.heartbeat_history = []
    
    def check_heartbeat_request(self, tool_calls: List[Any]) -> bool:
        """
        Check if any tool call requests a heartbeat
        
        In MemGPT, tools can have a special parameter:
        {"request_heartbeat": True}
        
        This tells the system to immediately call the LLM again
        without waiting for user input
        """
        if not tool_calls:
            return False
            
        for call in tool_calls:
            try:
                # Handle different types of tool call objects
                if hasattr(call, 'function'):
                    args_str = call.function.arguments if hasattr(call.function, 'arguments') else "{}"
                else:
                    continue
                    
                args = json.loads(args_str)
                if args.get("request_heartbeat", False):
                    return True
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
        
        return False
    
    def can_heartbeat(self) -> bool:
        """Check if we can still heartbeat (haven't hit the limit)"""
        return self.heartbeat_count < self.max_heartbeats
    
    def request_heartbeat(self, reason: str = "continuing thought process"):
        """Request a heartbeat"""
        if self.can_heartbeat():
            self.heartbeat_count += 1
            self.heartbeat_history.append({
                "count": self.heartbeat_count,
                "reason": reason
            })
            return True
        return False
    
    def get_status(self) -> str:
        """Get current heartbeat status"""
        return f"💓 Heartbeat {self.heartbeat_count}/{self.max_heartbeats}"


def add_heartbeat_to_tools(tools: List[Dict]) -> List[Dict]:
    """
    Add request_heartbeat parameter to all memory tools
    This allows the LLM to chain function calls
    """
    enhanced_tools = []
    
    for tool in tools:
        tool_copy = tool.copy()
        
        # Add request_heartbeat parameter to function parameters
        if "function" in tool_copy:
            params = tool_copy["function"]["parameters"]
            if "properties" not in params:
                params["properties"] = {}
            
            # Add heartbeat parameter
            params["properties"]["request_heartbeat"] = {
                "type": "boolean",
                "description": "Request continuation without user input to chain this operation with additional actions"
            }
        
        enhanced_tools.append(tool_copy)
    
    return enhanced_tools


def create_heartbeat_message() -> Dict:
    """
    Create a system message for heartbeat continuation
    This is sent to the LLM to continue thinking
    """
    return {
        "role": "system",
        "content": "[SYSTEM: Heartbeat requested. Continue your thought process or respond to the user if done.]"
    }


# Example usage in system prompt
HEARTBEAT_INSTRUCTIONS = """
MULTI-STEP REASONING:
You can chain multiple tool calls by setting "request_heartbeat": true in any tool's parameters.
This lets you continue thinking and acting without waiting for user input.

Use for complex tasks requiring multiple steps (search then respond, update then search, etc.).
For simple single operations, no heartbeat needed - just execute and respond.
"""

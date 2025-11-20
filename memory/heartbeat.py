"""
Heartbeat mechanism for Multi-step reasoning
"""
import json
from typing import List, Dict

class HeartbeatManager:
    def __init__(self, max_beats=5):
        self.max = max_beats
        self.count = 0

    def reset(self):
        self.count = 0

    def can_heartbeat(self):
        return self.count < self.max

    def request_heartbeat(self):
        self.count += 1

def add_heartbeat_to_tools(tools: List[Dict]) -> List[Dict]:
    new_tools = []
    for t in tools:
        tc = t.copy()
        props = tc['function']['parameters']['properties']
        props['request_heartbeat'] = {
            "type": "boolean",
            "description": "Set true to continue thinking after this tool (chaining actions)."
        }
        new_tools.append(tc)
    return new_tools

def create_heartbeat_message():
    return {"role": "system", "content": "Heartbeat requested. Continue..."}
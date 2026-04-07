"""
Personality Engine
"""
import json
import logging
import os
import threading
from dataclasses import dataclass

from walle.tools.base_executor import BaseToolExecutor
from walle.memory.config import MEMORY_DIR

_log = logging.getLogger("walle.personality")

_personality_save_lock = threading.Lock()
_PERSONALITY_PATH = os.path.join(MEMORY_DIR, "personality.json")

@dataclass
class PersonalityProfile:
    humor: int = 50
    honesty: int = 90
    sass: int = 20
    
    def get_system_prompt_addition(self):
        return f"Personality: humor={self.humor}%, honesty={self.honesty}%, sass={self.sass}%. Reflect these in tone."

class PersonalityEngine:
    def __init__(self, profile=None):
        self.profile = profile or PersonalityProfile()

    def get_system_prompt_addition(self):
        """Wrapper to get the prompt from the profile"""
        return self.profile.get_system_prompt_addition()

    @classmethod
    def load(cls):
        try:
            with open(_PERSONALITY_PATH, "r") as f:
                data = json.load(f)
            # Handle case where JSON might be nested or flat
            if 'profile' in data:
                return cls(PersonalityProfile(**data['profile']))
            return cls(PersonalityProfile(**data))
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()

    def save(self):
        """Atomic, thread-safe save - write to temp file then rename."""
        import tempfile

        with _personality_save_lock:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.json', dir=MEMORY_DIR)
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(self.profile.__dict__, f)
                os.replace(temp_path, _PERSONALITY_PATH)  # Atomic on POSIX
            except Exception:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise

def get_personality_tools():
    return [{
        "type": "function",
        "function": {
            "name": "set_personality",
            "description": "Adjust the robot's personality traits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trait": {"type": "string", "enum": ["humor", "honesty", "sass"]},
                    "value": {"type": "integer", "minimum": 0, "maximum": 100}
                },
                "required": ["trait", "value"]
            }
        }
    }]


class PersonalityToolExecutor(BaseToolExecutor):
    """Executor for personality-related tools."""

    def __init__(self, engine: PersonalityEngine):
        self.engine = engine

    def _set_personality(self, args: dict) -> str:
        """Set a personality trait value."""
        trait = args.get('trait')
        value = args.get('value')

        if not hasattr(self.engine.profile, trait):
            return f"Unknown trait: {trait}"

        setattr(self.engine.profile, trait, value)
        self.engine.save()
        return f"Personality trait '{trait}' set to {value}."

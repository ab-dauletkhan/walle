"""
Personality Engine
"""
import json
from dataclasses import dataclass
from base_executor import BaseToolExecutor

@dataclass
class PersonalityProfile:
    humor: int = 50
    honesty: int = 90
    sass: int = 20
    
    def get_system_prompt_addition(self):
        return f"""PERSONALITY SETTINGS:
        - Humor: {self.humor}%
        - Honesty: {self.honesty}%
        - Sass: {self.sass}%
        Adjust your tone and responsiveness to reflect these traits naturally."""

class PersonalityEngine:
    def __init__(self, profile=None):
        self.profile = profile or PersonalityProfile()

    def get_system_prompt_addition(self):
        """Wrapper to get the prompt from the profile"""
        return self.profile.get_system_prompt_addition()

    @classmethod
    def load(cls):
        try:
            with open("personality.json", "r") as f:
                data = json.load(f)
            # Handle case where JSON might be nested or flat
            if 'profile' in data:
                return cls(PersonalityProfile(**data['profile']))
            return cls(PersonalityProfile(**data))
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()

    def save(self):
        with open("personality.json", "w") as f:
            json.dump(self.profile.__dict__, f)

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
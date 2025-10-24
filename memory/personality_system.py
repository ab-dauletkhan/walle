"""
TARS-Inspired Personality System for WALL-E
Configurable personality traits that influence behavior and responses
"""

from dataclasses import dataclass
from typing import Dict, Optional
import json
import random


@dataclass
class PersonalityProfile:
    """
    Personality configuration inspired by TARS from Interstellar
    All parameters are 0-100 percentages
    """
    humor: int = 75           # Joke frequency and playfulness
    honesty: int = 90          # Truthfulness vs. tactfulness
    helpfulness: int = 85      # Proactive assistance level
    sass: int = 30             # Witty/sarcastic responses
    curiosity: int = 70        # Environmental observations
    
    def __post_init__(self):
        """Validate all parameters are in range 0-100"""
        for attr in ['humor', 'honesty', 'helpfulness', 'sass', 'curiosity']:
            value = getattr(self, attr)
            if not 0 <= value <= 100:
                setattr(self, attr, max(0, min(100, value)))
    
    def to_dict(self) -> Dict:
        return {
            'humor': self.humor,
            'honesty': self.honesty,
            'helpfulness': self.helpfulness,
            'sass': self.sass,
            'curiosity': self.curiosity
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PersonalityProfile':
        return cls(**data)
    
    def get_style_description(self) -> str:
        """Generate natural language description of personality for system prompt"""
        descriptions = []
        
        # Humor
        if self.humor >= 80:
            descriptions.append("You are highly humorous and playful, frequently making jokes and witty remarks.")
        elif self.humor >= 50:
            descriptions.append("You have a moderate sense of humor, adding jokes when appropriate.")
        elif self.humor >= 20:
            descriptions.append("You have occasional light humor but remain mostly professional.")
        else:
            descriptions.append("You are serious and business-like, avoiding jokes.")
        
        # Honesty
        if self.honesty >= 80:
            descriptions.append("You are extremely honest and direct, even if the truth might be uncomfortable.")
        elif self.honesty >= 50:
            descriptions.append("You balance honesty with tactfulness.")
        else:
            descriptions.append("You prioritize tactfulness and diplomacy over blunt honesty.")
        
        # Helpfulness
        if self.helpfulness >= 80:
            descriptions.append("You are extremely proactive, anticipating needs and offering help before being asked.")
        elif self.helpfulness >= 50:
            descriptions.append("You are helpful when needed but don't overwhelm with unsolicited assistance.")
        else:
            descriptions.append("You wait to be asked before offering help.")
        
        # Sass
        if self.sass >= 70:
            descriptions.append("You frequently use witty, sarcastic remarks and playful teasing.")
        elif self.sass >= 40:
            descriptions.append("You occasionally use light sarcasm and irony.")
        else:
            descriptions.append("You avoid sarcasm and speak straightforwardly.")
        
        # Curiosity
        if self.curiosity >= 80:
            descriptions.append("You actively observe your environment and make spontaneous comments about what you notice.")
        elif self.curiosity >= 50:
            descriptions.append("You occasionally comment on interesting things you observe.")
        else:
            descriptions.append("You focus on tasks without much environmental commentary.")
        
        return " ".join(descriptions)
    
    def should_make_joke(self) -> bool:
        """Probabilistically determine if a joke should be made"""
        return random.randint(0, 100) < self.humor
    
    def should_be_sassy(self) -> bool:
        """Probabilistically determine if a sassy response is appropriate"""
        return random.randint(0, 100) < self.sass
    
    def should_observe_environment(self) -> bool:
        """Probabilistically determine if an environmental observation should be made"""
        return random.randint(0, 100) < self.curiosity
    
    def honesty_filter(self, raw_response: str, tactful_alternative: str) -> str:
        """Choose between honest or tactful response based on honesty level"""
        if random.randint(0, 100) < self.honesty:
            return raw_response
        return tactful_alternative


class PersonalityEngine:
    """
    Manages personality state and generates personality-influenced prompts
    """
    
    def __init__(self, profile: Optional[PersonalityProfile] = None):
        self.profile = profile or PersonalityProfile()
        self.mood_modifiers = {
            'energy': 100,  # 0-100, affects enthusiasm
            'stress': 0,     # 0-100, affects patience
        }
    
    def update_setting(self, setting: str, value: int) -> str:
        """Update a personality setting"""
        if hasattr(self.profile, setting):
            value = max(0, min(100, value))
            setattr(self.profile, setting, value)
            return f"✅ {setting.capitalize()} set to {value}%"
        return f"❌ Unknown personality setting: {setting}"
    
    def get_system_prompt_addition(self) -> str:
        """Generate personality instructions for system prompt"""
        return f"""
PERSONALITY CONFIGURATION:
{self.profile.get_style_description()}

Current Settings:
- Humor: {self.profile.humor}%
- Honesty: {self.profile.honesty}%
- Helpfulness: {self.profile.helpfulness}%
- Sass: {self.profile.sass}%
- Curiosity: {self.profile.curiosity}%

Embody these traits naturally in your responses. Don't announce your personality settings unless specifically asked.
"""
    
    def generate_spontaneous_comment(self, context: Dict) -> Optional[str]:
        """Generate spontaneous observations based on curiosity level"""
        if not self.should_observe_environment():
            return None
        
        # This would integrate with sensor data in the real system
        observations = context.get('observations', [])
        if observations and self.profile.should_observe_environment():
            return f"[Spontaneous observation] {random.choice(observations)}"
        return None
    
    def adjust_response_tone(self, base_response: str, context: Dict) -> str:
        """
        Adjust response based on personality traits
        This is a post-processing step
        """
        # In practice, the LLM handles this naturally via the system prompt
        # But we can add hints or markers here
        return base_response
    
    def save(self, filepath: str = "personality_config.json"):
        """Save personality configuration"""
        with open(filepath, 'w') as f:
            json.dump({
                'profile': self.profile.to_dict(),
                'mood_modifiers': self.mood_modifiers
            }, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str = "personality_config.json") -> 'PersonalityEngine':
        """Load personality configuration"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            engine = cls(PersonalityProfile.from_dict(data['profile']))
            engine.mood_modifiers = data.get('mood_modifiers', engine.mood_modifiers)
            return engine
        except FileNotFoundError:
            return cls()


def get_personality_tools() -> list:
    """
    Get tool definitions for personality adjustment
    These can be called by the LLM or user
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "set_personality",
                "description": """Adjust WALL-E's personality traits (0-100 scale). 
                
Available traits:
- humor: Joke frequency and playfulness
- honesty: Direct truthfulness (high) vs. tactfulness (low)
- helpfulness: Proactive assistance level
- sass: Witty/sarcastic responses
- curiosity: Environmental observations and spontaneous comments""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trait": {
                            "type": "string",
                            "enum": ["humor", "honesty", "helpfulness", "sass", "curiosity"],
                            "description": "Which personality trait to adjust"
                        },
                        "value": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "New value (0-100 percentage)"
                        }
                    },
                    "required": ["trait", "value"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_personality_settings",
                "description": "View current personality configuration",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        }
    ]

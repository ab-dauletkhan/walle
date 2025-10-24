"""
Context Manager for WALL-E
Handles multi-modal inputs: vision, sensors, environment
Integrates with memory system to provide rich context to LLM
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class VisualContext:
    """Context from vision systems"""
    faces_detected: List[Dict] = field(default_factory=list)  # [{name, confidence, location}]
    objects_detected: List[Dict] = field(default_factory=list)  # [{object, confidence, location}]
    scene_description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_natural_language(self) -> str:
        """Convert visual context to natural language for LLM"""
        if not self.faces_detected and not self.objects_detected:
            return ""
        
        parts = []
        
        if self.faces_detected:
            if len(self.faces_detected) == 1:
                face = self.faces_detected[0]
                if face.get('name'):
                    parts.append(f"I can see {face['name']}")
                else:
                    parts.append("I can see a person")
            else:
                known_faces = [f['name'] for f in self.faces_detected if f.get('name')]
                if known_faces:
                    parts.append(f"I can see {', '.join(known_faces)}")
                parts.append(f"and {len(self.faces_detected) - len(known_faces)} other people")
        
        if self.objects_detected:
            interesting_objects = [obj['object'] for obj in self.objects_detected[:5]]  # Top 5
            if interesting_objects:
                parts.append(f"I notice {', '.join(interesting_objects)}")
        
        if self.scene_description:
            parts.append(f"Scene: {self.scene_description}")
        
        return ". ".join(parts) + "." if parts else ""


@dataclass
class EnvironmentContext:
    """Context about the environment"""
    location: Optional[str] = None
    lighting: Optional[str] = None  # "bright", "dim", "dark"
    temperature: Optional[float] = None  # celsius
    battery_level: Optional[int] = None  # 0-100
    obstacles_nearby: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_natural_language(self) -> str:
        """Convert environment context to natural language"""
        parts = []
        
        if self.location:
            parts.append(f"I'm in {self.location}")
        
        if self.battery_level is not None:
            if self.battery_level < 20:
                parts.append(f"⚠️ My battery is low ({self.battery_level}%)")
            elif self.battery_level < 50:
                parts.append(f"My battery is at {self.battery_level}%")
        
        if self.lighting:
            if self.lighting == "dark":
                parts.append("It's quite dark here")
            elif self.lighting == "dim":
                parts.append("The lighting is dim")
        
        if self.obstacles_nearby:
            parts.append("⚠️ I detect obstacles nearby")
        
        return ". ".join(parts) + "." if parts else ""


@dataclass
class InteractionContext:
    """Context about current interaction"""
    user_name: Optional[str] = None
    user_mood: Optional[str] = None  # Detected from tone/expression
    conversation_start: datetime = field(default_factory=datetime.now)
    last_interaction: Optional[datetime] = None
    interaction_count: int = 0
    
    def to_natural_language(self) -> str:
        """Convert interaction context to natural language"""
        parts = []
        
        if self.user_name:
            parts.append(f"Speaking with {self.user_name}")
        
        if self.last_interaction:
            time_since = datetime.now() - self.last_interaction
            if time_since.total_seconds() < 3600:  # Less than an hour
                minutes = int(time_since.total_seconds() / 60)
                parts.append(f"We spoke {minutes} minutes ago")
            elif time_since.days == 0:
                hours = int(time_since.total_seconds() / 3600)
                parts.append(f"We last spoke {hours} hours ago")
            elif time_since.days == 1:
                parts.append("We last spoke yesterday")
            else:
                parts.append(f"We last spoke {time_since.days} days ago")
        
        if self.user_mood:
            parts.append(f"User seems {self.user_mood}")
        
        return ". ".join(parts) + "." if parts else ""


class ContextManager:
    """
    Manages all contextual information for WALL-E
    Combines sensor data, environment, and interaction history
    """
    
    def __init__(self):
        self.visual: Optional[VisualContext] = None
        self.environment: Optional[EnvironmentContext] = None
        self.interaction: Optional[InteractionContext] = None
        self.spontaneous_observations: List[str] = []
    
    def update_visual(self, visual_context: VisualContext):
        """Update visual context from camera/vision system"""
        self.visual = visual_context
        self._generate_observations()
    
    def update_environment(self, env_context: EnvironmentContext):
        """Update environment context from sensors"""
        self.environment = env_context
        self._generate_observations()
    
    def update_interaction(self, interaction_context: InteractionContext):
        """Update interaction context"""
        self.interaction = interaction_context
    
    def _generate_observations(self):
        """Generate spontaneous observations based on context changes"""
        # This is where curiosity-driven observations would be generated
        # Based on changes in visual or environmental context
        pass
    
    def get_context_string(self) -> str:
        """
        Get complete context as natural language string
        This is added to the system prompt or user message
        """
        parts = ["[CURRENT CONTEXT]"]
        
        if self.visual:
            visual_str = self.visual.to_natural_language()
            if visual_str:
                parts.append(f"Vision: {visual_str}")
        
        if self.environment:
            env_str = self.environment.to_natural_language()
            if env_str:
                parts.append(f"Environment: {env_str}")
        
        if self.interaction:
            interaction_str = self.interaction.to_natural_language()
            if interaction_str:
                parts.append(f"Interaction: {interaction_str}")
        
        if len(parts) == 1:  # Only header
            return ""
        
        return "\n".join(parts) + "\n"
    
    def get_observations(self) -> List[str]:
        """Get list of observations for personality system"""
        observations = []
        
        if self.visual and self.visual.objects_detected:
            for obj in self.visual.objects_detected[:3]:
                observations.append(f"I see a {obj['object']}")
        
        if self.environment:
            if self.environment.battery_level and self.environment.battery_level < 30:
                observations.append("My battery is getting low")
            if self.environment.obstacles_nearby:
                observations.append("There are obstacles around me")
        
        return observations
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for storage"""
        return {
            'visual': self.visual.__dict__ if self.visual else None,
            'environment': self.environment.__dict__ if self.environment else None,
            'interaction': self.interaction.__dict__ if self.interaction else None,
            'observations': self.spontaneous_observations
        }


# Sensor simulation helpers (for testing without actual hardware)
class SensorSimulator:
    """Simulates sensor inputs for testing"""
    
    @staticmethod
    def simulate_visual_context() -> VisualContext:
        """Simulate visual input"""
        return VisualContext(
            faces_detected=[],
            objects_detected=[
                {'object': 'chair', 'confidence': 0.95, 'location': 'center'},
                {'object': 'table', 'confidence': 0.87, 'location': 'left'}
            ],
            scene_description="indoor room with furniture"
        )
    
    @staticmethod
    def simulate_environment_context(battery: int = 75) -> EnvironmentContext:
        """Simulate environment sensors"""
        return EnvironmentContext(
            location="living room",
            lighting="bright",
            temperature=22.5,
            battery_level=battery,
            obstacles_nearby=False
        )
    
    @staticmethod
    def simulate_interaction_context(user_name: str = None) -> InteractionContext:
        """Simulate interaction context"""
        return InteractionContext(
            user_name=user_name,
            user_mood=None,
            interaction_count=0
        )


# Example usage in main chat loop:
"""
# In walle_enhanced.py, add:

context_manager = ContextManager()

# Before each chat interaction:
def chat_with_walle(user_input: str, context_manager: ContextManager = None):
    system_msg = get_system_message()
    
    # Add context if available
    context_string = ""
    if context_manager:
        context_string = context_manager.get_context_string()
    
    user_msg = {
        "role": "user", 
        "content": context_string + user_input if context_string else user_input
    }
    
"""


def demo_context_system():
    """Demonstrate the context system"""
    print("🔍 Context System Demo\n")
    
    # Create context manager
    cm = ContextManager()
    
    # Simulate visual input
    visual = VisualContext(
        faces_detected=[
            {'name': 'Sarah', 'confidence': 0.95, 'location': 'center'}
        ],
        objects_detected=[
            {'object': 'coffee mug', 'confidence': 0.92, 'location': 'table'},
            {'object': 'laptop', 'confidence': 0.88, 'location': 'desk'},
            {'object': 'book', 'confidence': 0.85, 'location': 'shelf'}
        ],
        scene_description="cozy home office"
    )
    cm.update_visual(visual)
    
    # Simulate environment
    env = EnvironmentContext(
        location="home office",
        lighting="bright",
        battery_level=45,
        temperature=21.5,
        obstacles_nearby=False
    )
    cm.update_environment(env)
    
    # Simulate interaction
    interaction = InteractionContext(
        user_name="Sarah",
        user_mood="focused",
        last_interaction=datetime(2025, 10, 22, 14, 30)
    )
    cm.update_interaction(interaction)
    
    # Display context
    print(cm.get_context_string())
    print("\n📝 Observations:")
    for obs in cm.get_observations():
        print(f"  • {obs}")


if __name__ == "__main__":
    demo_context_system()
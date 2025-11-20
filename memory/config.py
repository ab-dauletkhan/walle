import os
import requests
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # --- AI Settings ---
    # Use the custom model we created. Fallback to qwen2.5:3b if needed.
    MODEL_NAME: str = "walle-brain" 
    EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # --- Memory Settings ---
    MAX_CONTEXT_MESSAGES: int = 15
    USE_SEMANTIC_SEARCH: bool = False
    RECALL_MEMORY_LIMIT: int = 60
    
    # --- Search Settings ---
    MAX_SEARCH_RESULTS: int = 5
    # Try these regions in order. 'wt-wt' is Global, 'us-en' is USA.
    SEARCH_REGIONS: List[str] = field(default_factory=lambda: ["wt-wt", "us-en"])
    
    # --- Robot Settings ---
    SERIAL_PORT: str = None 
    BAUD_RATE: int = 9600

    def validate(self) -> bool:
        """Checks if Ollama is running and model exists."""
        try:
            # Check connection
            res = requests.get(self.OLLAMA_BASE_URL)
            if res.status_code != 200: return False
            
            # Check model
            res = requests.get(f"{self.OLLAMA_BASE_URL}/api/tags")
            models = [m['name'] for m in res.json()['models']]
            if self.MODEL_NAME not in models and f"{self.MODEL_NAME}:latest" not in models:
                print(f"⚠️ Warning: Model '{self.MODEL_NAME}' not found. Available: {models}")
                return False
            return True
        except Exception as e:
            print(f"❌ Config Validation Failed: {e}")
            return False

    @classmethod
    def load(cls):
        return cls()

conf = Config.load()
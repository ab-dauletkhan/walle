"""
Streaming LLM client interface and mock implementation.

The interface mirrors how Ollama/OpenAI streaming works:
    for chunk in client.stream_chat(messages):
        print(chunk, end="", flush=True)

To switch to real Ollama, replace MockLLMClient with OllamaLLMClient.
"""

import random
import time
from abc import ABC, abstractmethod
from typing import Iterator


class LLMClient(ABC):
    """Abstract streaming LLM client.

    All implementations yield text chunks from a generator,
    matching the pattern used by OpenAI-compatible streaming APIs.
    """

    @abstractmethod
    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """Send conversation messages and yield response text chunks."""
        ...


class MockLLMClient(LLMClient):
    """Simulates streaming LLM responses with realistic token-by-token delays.

    Picks a contextual response based on keywords in the last user message,
    then yields it word-by-word to mimic real token generation.
    """

    RESPONSES = {
        "weather": "It's currently 22 degrees and sunny outside. Perfect weather for a walk!",
        "time": "I don't have a real clock, but I'd guess it's time for something fun.",
        "name": "I'm your voice assistant, running right here on the Jetson. Nice to meet you!",
        "help": "I can answer questions, control the robot, or just chat. What would you like?",
        "joke": "Why do programmers prefer dark mode? Because light attracts bugs.",
        "hello": "Hey there! What can I help you with?",
        "how are": "I'm running great, all systems nominal! How about you?",
    }

    DEFAULT_RESPONSES = [
        "That's an interesting question. Let me think about it for a moment. "
        "I'd say the answer depends on several factors.",
        "Sure, I can help with that. Based on what I know, "
        "here's a brief summary of the key points.",
        "Good question! The short answer is that it's a bit complicated, "
        "but I'll do my best to explain.",
        "Hmm, let me consider that. I think the most important thing to note "
        "is that there are multiple perspectives on this.",
    ]

    def __init__(self, token_delay: float = 0.05):
        self._token_delay = token_delay

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content", "").lower()
                break

        response = self._pick_response(user_text)
        words = response.split()
        for i, word in enumerate(words):
            time.sleep(random.uniform(self._token_delay * 0.5, self._token_delay * 1.5))
            yield word + (" " if i < len(words) - 1 else "")

    def _pick_response(self, user_text: str) -> str:
        for keyword, response in self.RESPONSES.items():
            if keyword in user_text:
                return response
        return random.choice(self.DEFAULT_RESPONSES)


class OllamaLLMClient(LLMClient):
    """Real Ollama streaming client using OpenAI-compatible API.

    This matches the streaming pattern in walle_enhanced.py:
        stream = client.chat.completions.create(..., stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        system_prompt: str = (
            "You are WALL-E, a voice assistant on a physical robot. Your replies "
            "are spoken aloud, so they MUST be extremely short. Hard rules: "
            "respond in ONE sentence by default, two at most. Never use lists, "
            "bullets, markdown, emojis, or multiple paragraphs. No preamble, "
            "no 'sure!', no repeating the question. Under 30 words."
        ),
        max_tokens: int = 80,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            stream=True,
            max_tokens=self.max_tokens,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

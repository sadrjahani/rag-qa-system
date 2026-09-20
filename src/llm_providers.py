"""LLM provider implementations for RAGGenerator.

Add new providers here without touching generation.py or RAGGenerator.
Each provider must implement generate(prompt: str) -> str.
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str: ...


class OllamaProvider:
    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        from ollama_client import OllamaClient
        self.client = OllamaClient(model=model, base_url=base_url)
        self.model = model

    def generate(self, prompt: str) -> str:
        logger.info("Generating with Ollama model: %s", self.model)
        return self.client.generate(prompt)


class GroqProvider:
    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        from groq import Groq
        self.client = Groq()  # GROQ_API_KEY رو از .env می‌خونه
        self.model = model

    def generate(self, prompt: str) -> str:
        logger.info("Generating with Groq model: %s", self.model)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


def get_provider(provider: str, model: str) -> LLMProvider:
    """Factory: provider name + model name → LLMProvider instance."""
    if provider == "groq":
        return GroqProvider(model=model)
    if provider == "ollama":
        return OllamaProvider(model=model)
    raise ValueError(f"Unknown provider: {provider!r}. Choose 'ollama' or 'groq'.")
"""Local Ollama adapter used by RAG answer generation."""

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


def get_provider(model: str) -> LLMProvider:
    """Return the local Ollama provider used by this project."""
    return OllamaProvider(model=model)

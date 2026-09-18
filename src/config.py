"""Shared configuration for the rag-qa-system pipeline, read from .env.

Any script in this project that needs a configurable value (embedding
model, log level, future retrieval/eval settings) should read it from here
instead of defining its own Config class or hardcoding it locally.
"""

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class Config:
    embedding_model_name: str
    embedding_dimension: int | None
    log_level: str
    llm_model: str
     
    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(find_dotenv())
        raw_dimension = os.getenv("EMBEDDING_DIMENSION")
        return cls(
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
            embedding_dimension=int(raw_dimension) if raw_dimension else None,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            llm_model=os.getenv("LLM_MODEL", "qwen3:14b"),
        )

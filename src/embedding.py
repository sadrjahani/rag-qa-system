"""Build a local FAISS vector database from ``chunking.py`` JSONL output.

Each input line must be a JSON object with at least a ``text`` field. The
original chunk record is saved alongside the FAISS index, so a retriever can
map search results back to source text and section later.

Example:
    python src/embedding.py data/chapter_chunks.jsonl -o data/chapter_vectors
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import find_dotenv, load_dotenv
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    embedding_model_name: str
    embedding_dimension: int | None
    log_level: str

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(find_dotenv())
        raw_dimension = os.getenv("EMBEDDING_DIMENSION")
        return cls(
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
            embedding_dimension=int(raw_dimension) if raw_dimension else None,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


@dataclass
class EmbeddingModel:
    model_name: str
    _model: SentenceTransformer | None = field(default=None, init=False, repr=False)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")
        logger.info("Encoding %d chunk(s).", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) >= 100,
        )
        return np.ascontiguousarray(embeddings, dtype=np.float32)


class VectorStoreError(ValueError):
    """Raised when documents and embeddings cannot form an index."""


@dataclass
class FAISSVectorStore:
    dimension: int
    _documents: list[dict[str, Any]] = field(default_factory=list, init=False)
    _index: faiss.Index = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise VectorStoreError("Embedding dimension must be positive.")
        self._index = faiss.IndexFlatIP(self.dimension)

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return self._index.ntotal

    def add_documents(self, documents: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        if len(documents) != len(embeddings):
            raise VectorStoreError(
                "Document count and embedding count differ: "
                f"{len(documents)} != {len(embeddings)}."
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Expected embeddings with shape (n, {self.dimension}), received {embeddings.shape}."
            )

        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        # Normalized inner product is cosine similarity.
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._documents.extend(documents)
        logger.info("Added %d chunk(s); index now contains %d.", len(documents), self.size)

    def save(self, directory: Path, model_name: str) -> None:
        """Persist the index, aligned chunk metadata, and build information."""
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        with (directory / "documents.jsonl").open("w", encoding="utf-8") as handle:
            for document in self._documents:
                handle.write(json.dumps(document, ensure_ascii=False) + "\n")
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "dimension": self.dimension,
                    "metric": "cosine_similarity",
                    "document_count": self.size,
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved vector database to %s", directory)


def read_chunks_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read and validate the line-delimited records written by chunking.py."""
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VectorStoreError(
                    f"Invalid JSON on line {line_number} of {path}: {exc.msg}"
                ) from exc
            if not isinstance(document, dict) or not isinstance(document.get("text"), str):
                raise VectorStoreError(
                    f"Line {line_number} of {path} must be an object with a string 'text' field."
                )
            if not document["text"].strip():
                raise VectorStoreError(f"Line {line_number} of {path} has empty text.")
            documents.append(document)

    if not documents:
        raise VectorStoreError(f"No chunk records found in {path}.")
    return documents


def main() -> None:
    config = Config.from_env()
    parser = argparse.ArgumentParser(description="Embed chunk JSONL and build a local FAISS vector database.")
    parser.add_argument("input", type=Path, help="Path to the .jsonl file produced by chunking.py.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory (default: <input>_vectors next to the input).")
    parser.add_argument("--model-name", default=config.embedding_model_name, help="SentenceTransformer model name.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (default: 32).")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    documents = read_chunks_from_jsonl(args.input)
    embedder = EmbeddingModel(args.model_name)
    embeddings = embedder.encode_batch([document["text"] for document in documents], args.batch_size)

    actual_dimension = int(embeddings.shape[1])
    if config.embedding_dimension is not None and config.embedding_dimension != actual_dimension:
        raise VectorStoreError(
            "EMBEDDING_DIMENSION does not match the selected model: "
            f"{config.embedding_dimension} != {actual_dimension}. Remove or correct it in .env."
        )

    store = FAISSVectorStore(dimension=actual_dimension)
    store.add_documents(documents, embeddings)
    output_path = args.output or args.input.with_name(args.input.stem + "_vectors")
    store.save(output_path, args.model_name)
    print(f"Built a vector database with {store.size} chunks: {output_path}")


if __name__ == "__main__":
    main()

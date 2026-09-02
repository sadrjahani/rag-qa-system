"""FAISS-backed vector store and shared JSONL chunk-reading, used by both
embedding.py (building the database) and Retrieval.py (loading it back).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


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

        # copy=True guarantees this is a fresh array: np.ascontiguousarray
        # would return the *same* object (no copy) when embeddings is
        # already contiguous float32, and normalize_L2 mutates in place —
        # without the copy, callers who keep a reference to the original
        # `embeddings` array after calling add_documents() would see it
        # silently mutated too.
        vectors = np.array(embeddings, dtype=np.float32, copy=True)
        # Defensive: normalizes again even though EmbeddingModel.encode_batch
        # already returns L2-normalized vectors. Renormalizing an already-
        # unit-length vector is a no-op, so this only matters if
        # add_documents is ever called with embeddings from somewhere else
        # that didn't normalize. Normalized inner product == cosine similarity.
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
    """Read and validate the line-delimited records written by chunking.py
    (or the documents.jsonl saved alongside a FAISS index by FAISSVectorStore.save).
    """
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

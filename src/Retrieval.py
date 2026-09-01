"""Retrieve relevant chunks from a local FAISS vector database.

The database must be created first with ``embedding.py`` and contain:

    <database_dir>/index.faiss
    <database_dir>/documents.jsonl
    <database_dir>/metadata.json

Example:
    python src/Retrieval.py data/chapter_6_vectors "What is RAG?" --top-k 3
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss

from config import Config
from embedding import EmbeddingModel, VectorStoreError, read_chunks_from_jsonl


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by vector similarity search."""

    score: float
    text: str
    section: str
    chunk_id: int
    page: int | None = None
    source: str | None = None


@dataclass
class FAISSRetriever:
    """A loaded FAISS index and its metadata records in matching order."""

    index: faiss.Index
    documents: list[dict[str, Any]]
    model_name: str

    @classmethod
    def load(cls, database_dir: Path) -> "FAISSRetriever":
        """Load and validate a vector database created by embedding.py."""
        if not database_dir.is_dir():
            raise VectorStoreError(f"Database directory does not exist: {database_dir}")

        index_path = database_dir / "index.faiss"
        documents_path = database_dir / "documents.jsonl"
        metadata_path = database_dir / "metadata.json"
        for path in (index_path, documents_path, metadata_path):
            if not path.is_file():
                raise VectorStoreError(f"Required database file is missing: {path}")

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VectorStoreError(f"Invalid JSON in {metadata_path}: {exc.msg}") from exc

        if not isinstance(metadata, dict):
            raise VectorStoreError(f"{metadata_path} must contain a JSON object.")
        model_name = metadata.get("model_name")
        expected_dimension = metadata.get("dimension")
        expected_count = metadata.get("document_count")
        if not isinstance(model_name, str) or not model_name:
            raise VectorStoreError(f"{metadata_path} has no valid 'model_name'.")
        if not isinstance(expected_dimension, int) or expected_dimension <= 0:
            raise VectorStoreError(f"{metadata_path} has no valid 'dimension'.")
        if not isinstance(expected_count, int) or expected_count < 0:
            raise VectorStoreError(f"{metadata_path} has no valid 'document_count'.")

        documents = read_chunks_from_jsonl(documents_path)
        try:
            index = faiss.read_index(str(index_path))
        except RuntimeError as exc:
            raise VectorStoreError(f"Could not read FAISS index {index_path}: {exc}") from exc

        if index.ntotal != len(documents):
            raise VectorStoreError(
                "FAISS index and documents.jsonl are misaligned: "
                f"{index.ntotal} vectors, {len(documents)} documents."
            )
        if index.ntotal != expected_count:
            raise VectorStoreError(
                "FAISS index and metadata.json are inconsistent: "
                f"{index.ntotal} vectors, metadata says {expected_count}."
            )
        if index.d != expected_dimension:
            raise VectorStoreError(
                "FAISS index and metadata.json have different dimensions: "
                f"{index.d} != {expected_dimension}."
            )
        if index.ntotal == 0:
            raise VectorStoreError("The FAISS index is empty.")

        logger.info("Loaded %d chunk(s) from %s", index.ntotal, database_dir)
        return cls(index=index, documents=documents, model_name=model_name)

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Embed a question and return its highest-scoring stored chunks."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        embedder = EmbeddingModel(self.model_name)
        query_vector = embedder.encode_batch([question])
        if query_vector.shape[1] != self.index.d:
            raise VectorStoreError(
                "Query embedding dimension does not match the FAISS index: "
                f"{query_vector.shape[1]} != {self.index.d}."
            )

        result_count = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, result_count)
        results: list[RetrievedChunk] = []

        for score, document_index in zip(scores[0], indices[0]):
            if document_index == -1:
                continue
            document = self.documents[int(document_index)]
            results.append(
                RetrievedChunk(
                    score=float(score),
                    text=document["text"],
                    section=str(document.get("section", "document")),
                    chunk_id=int(document.get("chunk_id", document_index)),
                    page=document.get("page"),
                    source=document.get("source"),
                )
            )
        return results


def main() -> None:
    config = Config.from_env()
    parser = argparse.ArgumentParser(
        description="Search a local FAISS vector database created by embedding.py."
    )
    parser.add_argument(
        "database",
        type=Path,
        help="Directory containing index.faiss, documents.jsonl, and metadata.json.",
    )
    parser.add_argument("question", help="Question to search for.")
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of chunks to return (default: 3)."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    try:
        retriever = FAISSRetriever.load(args.database)
        results = retriever.retrieve(args.question, args.top_k)
    except (ValueError, VectorStoreError) as exc:
        parser.error(str(exc))

    print(f"\nResults for: {args.question}\n")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. Score: {result.score:.4f}")
        if result.source:
            print(f"   Source: {result.source}")
        print(f"   Section: {result.section}")
        if result.page is not None:
            print(f"   Page: {result.page}")
        print(f"   Chunk ID: {result.chunk_id}")
        print(f"   Text: {result.text}\n")


if __name__ == "__main__":
    main()

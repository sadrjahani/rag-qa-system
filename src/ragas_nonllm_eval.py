"""
Semantic, non-LLM retrieval evaluation.

This module evaluates a FAISS-based RAG retriever without using:
    - an LLM judge
    - Ollama
    - Groq
    - an API key

Instead of string-level similarity such as Jaro-Winkler, it compares
retrieved contexts and reference contexts using embedding cosine similarity.

The important requirement is that the SAME embedding model used to build
the FAISS index is also used for evaluation.

Golden-set format:

{
    "id": "ret_001",
    "question": "...",
    "reference_contexts": [
        "verbatim passage from the book"
    ]
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from retrieval import FAISSRetriever
from vector_store import VectorStoreError

logger = logging.getLogger(__name__)

# Similarity above this value is considered a semantic match.
#
# Start with 0.75. After seeing the first evaluation results, this can be
# tuned using the actual embedding model used by the project.
SEMANTIC_SIMILARITY_THRESHOLD = 0.75

def load_golden_set_with_contexts(path: Path) -> list[dict[str, Any]]:
    """
    Load retrieval evaluation records.

    Every record must contain:
        - question
        - reference_contexts

    reference_contexts must contain the actual source passages, not answers.
    """

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VectorStoreError(
                    f"Invalid JSON on line {line_number} of {path}: {exc.msg}"
                ) from exc

            question = record.get("question")
            reference_contexts = record.get("reference_contexts")

            if not question or not reference_contexts:
                logger.warning(
                    "Skipping line %d of %s: missing 'question' "
                    "or 'reference_contexts'.",
                    line_number,
                    path,
                )
                continue

            if isinstance(reference_contexts, str):
                reference_contexts = [reference_contexts]

            if not isinstance(reference_contexts, list):
                logger.warning(
                    "Skipping line %d of %s: "
                    "'reference_contexts' must be a list.",
                    line_number,
                    path,
                )
                continue

            reference_contexts = [
                str(context).strip()
                for context in reference_contexts
                if str(context).strip()
            ]

            if not reference_contexts:
                logger.warning(
                    "Skipping line %d of %s: empty reference_contexts.",
                    line_number,
                    path,
                )
                continue

            records.append(
                {
                    "question": str(question),
                    "reference_contexts": reference_contexts,
                }
            )

    if not records:
        raise VectorStoreError(
            f"No usable records found in {path}. "
            "Each record needs 'question' and 'reference_contexts'."
        )

    return records



def _get_embedding_model(retriever: FAISSRetriever) -> Any:
    """
    Obtain the embedding model from FAISSRetriever.

    The project retriever is expected to expose its embedding model.

    We intentionally do not create a second, unrelated embedding model here.
    Evaluation must use the same embedding space as the FAISS index.
    """

    possible_names = (
        "embedding_model",
        "embedder",
        "model",
    )

    for name in possible_names:
        model = getattr(retriever, name, None)

        if model is not None:
            return model

    raise AttributeError(
        "Could not find the embedding model on FAISSRetriever. "
        "Expected one of: embedding_model, embedder, model. "
        "The evaluator must use the same embedding model that created "
        "the FAISS index."
    )


def _embed_texts(model: Any, texts: list[str]) -> np.ndarray:
    """
    Convert texts to normalized embedding vectors.

    This adapter supports the most common embedding-model APIs used in
    local RAG projects.

    Supported methods:
        - encode(texts)
        - embed_documents(texts)
        - embed(texts)

    A single-text fallback is also supported.
    """

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = None

    # SentenceTransformer-style API
    if hasattr(model, "encode"):
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # LangChain-style API
    elif hasattr(model, "embed_documents"):
        embeddings = model.embed_documents(texts)

    # Generic embedding API
    elif hasattr(model, "embed"):
        embeddings = model.embed(texts)

    else:
        raise AttributeError(
            "Embedding model does not expose a supported embedding method. "
            "Expected encode(), embed_documents(), or embed()."
        )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Some embedding implementations return shape (dimension,)
    # for a single text.
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got shape {embeddings.shape}."
        )

    # Normalize here as a safety measure.
    #
    # cosine_similarity(a, b) = dot(a, b) after L2 normalization.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)

    embeddings = embeddings / norms

    return embeddings


def _cosine_similarity_matrix(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """
    Return pairwise cosine similarity between two normalized matrices.

    Shape:
        left  = [N, D]
        right = [M, D]

        result = [N, M]
    """

    if left.size == 0 or right.size == 0:
        return np.empty((len(left), len(right)), dtype=np.float32)

    if left.shape[1] != right.shape[1]:
        raise ValueError(
            "Embedding dimensions do not match: "
            f"{left.shape[1]} != {right.shape[1]}"
        )

    return np.matmul(left, right.T)



def _score_retrieval(
    retrieved_texts: list[str],
    reference_contexts: list[str],
    embedding_model: Any,
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> tuple[float, float, float]:
    """
    Calculate semantic context precision and recall.

    Precision:
        Of the retrieved chunks, how many are semantically relevant
        to at least one reference context?

    Recall:
        Of the reference contexts, how many have at least one
        semantically matching retrieved chunk?

    Returns:
        (precision, recall, max_similarity)
    """

    if not retrieved_texts:
        return 0.0, 0.0, 0.0

    if not reference_contexts:
        return 0.0, 0.0, 0.0

    retrieved_embeddings = _embed_texts(
        embedding_model,
        retrieved_texts,
    )

    reference_embeddings = _embed_texts(
        embedding_model,
        reference_contexts,
    )

    similarities = _cosine_similarity_matrix(
        retrieved_embeddings,
        reference_embeddings,
    )

    if similarities.size == 0:
        return 0.0, 0.0, 0.0

    # For every retrieved chunk, find the closest reference context.
    retrieved_best = similarities.max(axis=1)

    # For every reference context, find the closest retrieved chunk.
    reference_best = similarities.max(axis=0)

    # Semantic precision.
    retrieved_matches = retrieved_best >= threshold
    precision = float(np.mean(retrieved_matches))

    # Semantic recall.
    reference_matches = reference_best >= threshold
    recall = float(np.mean(reference_matches))

    max_similarity = float(similarities.max())

    return precision, recall, max_similarity



def evaluate_with_ragas_nonllm(
    records: list[dict[str, Any]],
    retriever: FAISSRetriever,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Evaluate FAISS retrieval using semantic embedding similarity.

    The function keeps the original public API so existing evaluation
    code can continue calling:

        evaluate_with_ragas_nonllm(records, retriever, top_k)

    No LLM is used.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    embedding_model = _get_embedding_model(retriever)

    rows: list[dict[str, Any]] = []

    for i, record in enumerate(records, start=1):
        question = record["question"]
        reference_contexts = record["reference_contexts"]

        logger.info(
            "Scoring question %d/%d",
            i,
            len(records),
        )

        results = retriever.retrieve(
            question,
            top_k=top_k,
        )

        retrieved_texts = [
            result.text
            for result in results
            if getattr(result, "text", None)
        ]

        precision, recall, max_similarity = _score_retrieval(
            retrieved_texts=retrieved_texts,
            reference_contexts=reference_contexts,
            embedding_model=embedding_model,
        )

        rows.append(
            {
                "question": question,
                "context_precision": precision,
                "context_recall": recall,
                "max_semantic_similarity": max_similarity,
            }
        )

    if not rows:
        raise VectorStoreError(
            "No questions were evaluated."
        )

    aggregate = {
        "context_precision": float(
            np.mean(
                [row["context_precision"] for row in rows]
            )
        ),
        "context_recall": float(
            np.mean(
                [row["context_recall"] for row in rows]
            )
        ),
        "question_count": len(rows),
        "top_k": top_k,
        "judge": "semantic_embedding",
        "similarity_metric": "cosine",
        "similarity_threshold": SEMANTIC_SIMILARITY_THRESHOLD,
    }

    return aggregate, rows
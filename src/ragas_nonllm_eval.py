"""Retrieval evaluation for the FAISS-based RAG system.

This evaluator is designed around the actual FAISSRetriever used by the
project.

It reports:
- RAGAS non-LLM context precision
- RAGAS non-LLM context recall
- retrieval scores
- retrieved chunk IDs
- retrieved texts

The golden set must contain:
    - "question"
    - "reference_contexts": list[str]

Important:
Reference contexts should be passages from the source document that contain
the information needed to answer the question. They do not need to be the
same size as the retrieved chunks.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
)
from ragas.metrics._string import (
    DistanceMeasure,
    NonLLMStringSimilarity,
)

from retrieval import FAISSRetriever
from vector_store import VectorStoreError


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAGAS metrics
# ---------------------------------------------------------------------------

_precision_metric = NonLLMContextPrecisionWithReference(
    distance_measure=NonLLMStringSimilarity(
        distance_measure=DistanceMeasure.JARO_WINKLER
    )
)

_recall_metric = NonLLMContextRecall()


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------

def load_golden_set_with_contexts(
    path: Path,
) -> list[dict[str, Any]]:
    """Load retrieval evaluation records from JSONL."""

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
                    "Skipping line %d: missing question or reference_contexts.",
                    line_number,
                )
                continue

            if isinstance(reference_contexts, str):
                reference_contexts = [reference_contexts]

            if not isinstance(reference_contexts, list):
                logger.warning(
                    "Skipping line %d: reference_contexts must be a list.",
                    line_number,
                )
                continue

            reference_contexts = [
                str(context)
                for context in reference_contexts
                if str(context).strip()
            ]

            if not reference_contexts:
                logger.warning(
                    "Skipping line %d: reference_contexts is empty.",
                    line_number,
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


# ---------------------------------------------------------------------------
# RAGAS scoring
# ---------------------------------------------------------------------------

async def _score_with_ragas(
    retrieved_texts: list[str],
    reference_contexts: list[str],
) -> dict[str, float]:
    """Calculate RAGAS non-LLM context metrics."""

    sample = SingleTurnSample(
        retrieved_contexts=retrieved_texts,
        reference_contexts=reference_contexts,
    )

    precision = await _precision_metric.single_turn_ascore(sample)
    recall = await _recall_metric.single_turn_ascore(sample)

    return {
        "context_precision": float(precision),
        "context_recall": float(recall),
    }


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

async def _evaluate_one(
    question: str,
    reference_contexts: list[str],
    retriever: FAISSRetriever,
    top_k: int,
) -> dict[str, Any]:
    """Retrieve and evaluate one question."""

    results = retriever.retrieve(
        question,
        top_k=top_k,
    )

    retrieved_texts = [result.text for result in results]

    ragas_scores = await _score_with_ragas(
        retrieved_texts=retrieved_texts,
        reference_contexts=reference_contexts,
    )

    return {
        "question": question,

        "retrieved": [
            {
                "rank": rank,
                "chunk_id": result.chunk_id,
                "score": result.score,
                "section": result.section,
                "page": result.page,
                "source": result.source,
                "text": result.text,
            }
            for rank, result in enumerate(results, start=1)
        ],

        "reference_contexts": reference_contexts,

        "context_precision": ragas_scores["context_precision"],
        "context_recall": ragas_scores["context_recall"],
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_with_ragas_nonllm(
    records: list[dict[str, Any]],
    retriever: FAISSRetriever,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate retrieval using FAISSRetriever and RAGAS non-LLM metrics."""

    if not records:
        raise ValueError("records must not be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    async def _run_all() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for index, record in enumerate(records, start=1):
            logger.info(
                "Evaluating retrieval %d/%d",
                index,
                len(records),
            )

            row = await _evaluate_one(
                question=record["question"],
                reference_contexts=record["reference_contexts"],
                retriever=retriever,
                top_k=top_k,
            )

            rows.append(row)

        return rows

    per_question = asyncio.run(_run_all())

    aggregate = {
        "context_precision": (
            sum(row["context_precision"] for row in per_question)
            / len(per_question)
        ),
        "context_recall": (
            sum(row["context_recall"] for row in per_question)
            / len(per_question)
        ),
        "question_count": len(per_question),
        "top_k": top_k,
        "judge": "ragas_nonllm",
    }

    return aggregate, per_question

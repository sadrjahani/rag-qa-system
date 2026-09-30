"""Free, no-LLM RAGAS retrieval evaluation.

Uses ragas.metrics.NonLLMContextPrecisionWithReference and
NonLLMContextRecall — both compare retrieved text against reference_contexts
using a non-LLM string-distance measure (Jaro-Winkler). No API key,
no local model, no Ollama required.

Each golden-set record must have:
    - "question"
    - "reference_contexts": a list of one or more verbatim reference passages
      (NOT the answer — the actual source text the answer should come from)
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
from ragas.metrics._string import NonLLMStringSimilarity, DistanceMeasure


from retrieval import FAISSRetriever
from vector_store import VectorStoreError

logger = logging.getLogger(__name__)

_precision_metric = NonLLMContextPrecisionWithReference(
    distance_measure=NonLLMStringSimilarity(
        distance_measure=DistanceMeasure.JARO_WINKLER 
    )
)
_recall_metric = NonLLMContextRecall()


def load_golden_set_with_contexts(path: Path) -> list[dict[str, Any]]:
    """Read golden-set records that include reference_contexts."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VectorStoreError(f"Invalid JSON on line {line_number} of {path}: {exc.msg}") from exc

            question = record.get("question")
            reference_contexts = record.get("reference_contexts")
            if not question or not reference_contexts:
                logger.warning(
                    "Skipping line %d of %s: missing 'question' or 'reference_contexts'.",
                    line_number, path,
                )
                continue
            if isinstance(reference_contexts, str):
                reference_contexts = [reference_contexts]

            records.append({"question": question, "reference_contexts": reference_contexts})

    if not records:
        raise VectorStoreError(
            f"No usable records found in {path}. Each record needs 'question' and 'reference_contexts'."
        )
    return records


async def _score_one(question: str, retrieved: list[str], reference_contexts: list[str]) -> dict[str, float]:
    sample = SingleTurnSample(retrieved_contexts=retrieved, reference_contexts=reference_contexts)
    precision = await _precision_metric.single_turn_ascore(sample)
    recall = await _recall_metric.single_turn_ascore(sample)
    return {"question": question, "context_precision": precision, "context_recall": recall}


def evaluate_with_ragas_nonllm(
    records: list[dict[str, Any]], retriever: FAISSRetriever, top_k: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score retrieval with RAGAS's non-LLM metrics — free, no model required."""

    async def _run_all() -> list[dict[str, Any]]:
        rows = []
        for i, record in enumerate(records, start=1):
            logger.info("Scoring question %d/%d", i, len(records))
            results = retriever.retrieve(record["question"], top_k=top_k)
            retrieved_texts = [r.text for r in results]
            row = await _score_one(record["question"], retrieved_texts, record["reference_contexts"])
            rows.append(row)
        return rows

    per_question = asyncio.run(_run_all())

    aggregate = {
        "context_precision": sum(row["context_precision"] for row in per_question) / len(per_question),
        "context_recall": sum(row["context_recall"] for row in per_question) / len(per_question),
        "question_count": len(records),
        "top_k": top_k,
        "judge": "ragas_nonllm",
    }
    return aggregate, per_question

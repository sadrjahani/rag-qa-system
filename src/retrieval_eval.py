from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from retrieval import FAISSRetriever

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    question: str
    relevant_chunk_ids: list[int]
    retrieved_chunk_ids: list[int]
    hit_at_k: float
    recall_at_k: float
    reciprocal_rank: float


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    """Load retrieval evaluation questions from JSONL."""
    if not path.is_file():
        raise FileNotFoundError(f"Golden set does not exist: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            question = record.get("question")
            relevant_chunk_ids = record.get("relevant_chunk_ids")

            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Line {line_number}: 'question' must be a non-empty string."
                )

            if not isinstance(relevant_chunk_ids, list):
                raise ValueError(
                    f"Line {line_number}: "
                    "'relevant_chunk_ids' must be a list."
                )

            try:
                relevant_chunk_ids = [
                    int(chunk_id) for chunk_id in relevant_chunk_ids
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Line {line_number}: "
                    "'relevant_chunk_ids' must contain integers."
                ) from exc

            if not relevant_chunk_ids:
                raise ValueError(
                    f"Line {line_number}: "
                    "'relevant_chunk_ids' must not be empty."
                )

            records.append(
                {
                    "question": question,
                    "relevant_chunk_ids": relevant_chunk_ids,
                }
            )

    if not records:
        raise ValueError(f"Golden set is empty: {path}")

    return records


def evaluate_question(
    retriever: FAISSRetriever,
    question: str,
    relevant_chunk_ids: list[int],
    top_k: int,
) -> QuestionResult:
    """Evaluate one question against the retriever."""

    retrieved = retriever.retrieve(question, top_k=top_k)

    retrieved_ids = [chunk.chunk_id for chunk in retrieved]

    relevant_set = set(relevant_chunk_ids)

    hits = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant_set
    ]

    # Hit@K:
    # 1 if at least one relevant chunk was retrieved, otherwise 0.
    hit_at_k = 1.0 if hits else 0.0

    # Recall@K:
    # fraction of all relevant chunks that were retrieved.
    retrieved_relevant = len(set(hits))
    recall_at_k = retrieved_relevant / len(relevant_set)

    # Reciprocal Rank:
    # 1/rank of the first relevant result.
    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return QuestionResult(
        question=question,
        relevant_chunk_ids=relevant_chunk_ids,
        retrieved_chunk_ids=retrieved_ids,
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
    )


def evaluate(
    golden_set_path: Path,
    database_dir: Path,
    top_k: int = 3,
) -> dict[str, Any]:
    """Run retrieval evaluation."""

    golden_set = load_golden_set(golden_set_path)

    retriever = FAISSRetriever.load(database_dir)

    results: list[QuestionResult] = []

    for record in golden_set:
        result = evaluate_question(
            retriever=retriever,
            question=record["question"],
            relevant_chunk_ids=record["relevant_chunk_ids"],
            top_k=top_k,
        )

        results.append(result)

    question_count = len(results)

    hit_at_k = sum(
        result.hit_at_k for result in results
    ) / question_count

    recall_at_k = sum(
        result.recall_at_k for result in results
    ) / question_count

    mrr_at_k = sum(
        result.reciprocal_rank for result in results
    ) / question_count

    return {
        "aggregate_scores": {
            f"hit_at_{top_k}": hit_at_k,
            f"recall_at_{top_k}": recall_at_k,
            f"mrr_at_{top_k}": mrr_at_k,
            "question_count": question_count,
            "top_k": top_k,
        },
        "per_question": [
            asdict(result)
            for result in results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FAISS retrieval using Hit@K, Recall@K and MRR@K."
    )

    parser.add_argument(
        "golden_set",
        type=Path,
        help="Path to retrieval golden set JSONL.",
    )

    parser.add_argument(
        "database",
        type=Path,
        help="Path to FAISS vector database.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to evaluate.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    if args.top_k <= 0:
        parser.error("--top-k must be positive.")

    evaluation = evaluate(
        golden_set_path=args.golden_set,
        database_dir=args.database,
        top_k=args.top_k,
    )

    print(json.dumps(evaluation, indent=2, ensure_ascii=False))

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                evaluation,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
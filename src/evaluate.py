"""Evaluate retrieval quality against a golden set, using RAGAS.

Each line of the golden set (.jsonl) must have at least a "question" field
and a ground-truth answer field (either "reference_answer" or "ground_truth").
For every question, this script retrieves the top-k chunks from a FAISS
vector database (built by embedding.py) and scores the retrieved contexts
against the ground truth with RAGAS's context_precision and context_recall
metrics. Only retrieval is evaluated here — faithfulness and answer_relevancy
need a generated answer, which this project doesn't produce yet.

Example:
    python src/evaluate.py data/eval/golden_set.jsonl data/vector_db/AI_Engineering \
        -o data/eval/baseline_results.json --top-k 3
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

from config import Config
from Retrieval import FAISSRetriever
from vector_store import VectorStoreError

logger = logging.getLogger(__name__)


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    """Read golden-set records, normalizing the ground-truth field name."""
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
            ground_truth = record.get("reference_answer") or record.get("ground_truth")
            if not question or not ground_truth:
                logger.warning("Skipping line %d of %s: missing question or ground truth.", line_number, path)
                continue

            records.append({"question": question, "ground_truth": ground_truth})

    if not records:
        raise VectorStoreError(f"No usable golden-set records found in {path}.")
    return records


def build_ragas_dataset(records: list[dict[str, Any]], retriever: FAISSRetriever, top_k: int) -> Dataset:
    """Run retrieval for every question and assemble the RAGAS input dataset."""
    questions, contexts, ground_truths = [], [], []

    for i, record in enumerate(records, start=1):
        logger.info("Retrieving for question %d/%d", i, len(records))
        results = retriever.retrieve(record["question"], top_k=top_k)
        questions.append(record["question"])
        contexts.append([r.text for r in results])
        ground_truths.append(record["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def save_results(scores: dict[str, Any], per_question: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"aggregate_scores": scores, "per_question": per_question}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved evaluation results to %s", path)


def main() -> None:
    config = Config.from_env()
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against a golden set using RAGAS.")
    parser.add_argument("golden_set", type=Path, help="Path to the golden set .jsonl file.")
    parser.add_argument("database", type=Path, help="Path to the vector database directory (from embedding.py).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Where to save results (default: <golden_set_stem>_results.json next to the golden set).")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve per question (default: 3).")
    args = parser.parse_args()

    if not args.golden_set.is_file():
        parser.error(f"Golden set file does not exist: {args.golden_set}")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    records = load_golden_set(args.golden_set)
    print(f"Loaded {len(records)} golden-set question(s).")

    retriever = FAISSRetriever.load(args.database)
    dataset = build_ragas_dataset(records, retriever, args.top_k)

    print("Scoring retrieval with RAGAS (context_precision, context_recall)...")
    result = evaluate(dataset, metrics=[context_precision, context_recall])

    scores_df = result.to_pandas()
    aggregate_scores = {
        "context_precision": float(scores_df["context_precision"].mean()),
        "context_recall": float(scores_df["context_recall"].mean()),
        "question_count": len(records),
        "top_k": args.top_k,
    }
    per_question = [
        {
            "question": row["question"],
            "context_precision": float(row["context_precision"]),
            "context_recall": float(row["context_recall"]),
        }
        for _, row in scores_df.iterrows()
    ]

    output_path = args.output or args.golden_set.with_name(args.golden_set.stem + "_results.json")
    save_results(aggregate_scores, per_question, output_path)

    print("\n=== Aggregate scores ===")
    print(f"context_precision: {aggregate_scores['context_precision']:.4f}")
    print(f"context_recall:    {aggregate_scores['context_recall']:.4f}")
    print(f"\nSaved full results to {output_path}")


if __name__ == "__main__":
    main()

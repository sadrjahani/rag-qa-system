"""Evaluate retrieval and generated answers against a golden set.

Each line of the golden set (.jsonl) must have at least a "question" field.
By default (--retrieval-judge ragas_nonllm) this script scores retrieval with
RAGAS's free, no-LLM metrics (needs a "reference_contexts" field per record).
"--retrieval-judge ragas" uses full LLM-based RAGAS and requires a paid
OPENAI_API_KEY — only use this if you're willing to pay; it is NOT the
default because this project must stay free/local.

With ``--llm-model`` this instead uses a local Ollama model to generate a
grounded answer and a second judge call to score correctness, faithfulness,
and relevance. Ollama is free to run locally and does not require an API key.
This overrides --retrieval-judge, since it evaluates generated answers, not
retrieval directly. Individual question failures are skipped (logged as
warnings) rather than aborting the whole run.

Example:
    python src/evaluate.py data/eval/golden_set_ragas_nonllm.jsonl data/vector_db/AI_Engineering \
        -o data/eval/retrieval_results.json --top-k 3

    python src/evaluate.py data/eval/golden_set.jsonl data/vector_db/AI_Engineering \
        --llm-model llama3.2:3b -o data/eval/llm_results.json --top-k 3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from config import Config
from ollama_client import OllamaClient, OllamaError
from Retrieval import FAISSRetriever
from vector_store import VectorStoreError
from ragas_nonllm_eval import evaluate_with_ragas_nonllm, load_golden_set_with_contexts

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
    from datasets import Dataset

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


def token_set(text: str) -> set[str]:
    """Return lowercase word tokens for the local retrieval judge."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def evaluate_with_token_overlap(records: list[dict[str, Any]], retriever: FAISSRetriever,
                                top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate context coverage without a hosted LLM judge.

    Precision is the mean fraction of each retrieved context's tokens that
    overlap the reference answer. Recall is the fraction of reference tokens
    covered by the union of all retrieved contexts.
    """
    per_question = []
    for i, record in enumerate(records, start=1):
        logger.info("Retrieving for question %d/%d", i, len(records))
        results = retriever.retrieve(record["question"], top_k=top_k)
        reference_tokens = token_set(record["ground_truth"])
        context_tokens = [token_set(result.text) for result in results]
        precision = sum(
            len(tokens & reference_tokens) / len(tokens) if tokens else 0.0
            for tokens in context_tokens
        ) / len(context_tokens) if context_tokens else 0.0
        covered_tokens = set().union(*context_tokens) if context_tokens else set()
        recall = len(covered_tokens & reference_tokens) / len(reference_tokens) if reference_tokens else 0.0
        per_question.append({
            "question": record["question"],
            "context_precision": precision,
            "context_recall": recall,
        })

    aggregate = {
        "context_precision": sum(row["context_precision"] for row in per_question) / len(per_question),
        "context_recall": sum(row["context_recall"] for row in per_question) / len(per_question),
        "question_count": len(records),
        "top_k": top_k,
        "judge": "token_overlap",
    }
    return aggregate, per_question


def save_results(scores: dict[str, Any], per_question: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"aggregate_scores": scores, "per_question": per_question}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved evaluation results to %s", path)


def parse_judge_response(response: str) -> dict[str, float]:
    """Parse a judge JSON object, tolerating a markdown code fence."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        scores = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Judge did not return valid JSON: {response}") from exc
    if not isinstance(scores, dict):
        raise OllamaError("Judge response must be a JSON object.")
    required = ("correctness", "faithfulness", "relevance")
    if any(key not in scores for key in required):
        raise OllamaError(f"Judge response is missing one of: {', '.join(required)}")
    try:
        return {key: max(0.0, min(1.0, float(scores[key]))) for key in required}
    except (TypeError, ValueError) as exc:
        raise OllamaError("Judge scores must be numeric values from 0 to 1.") from exc


def evaluate_with_ollama(records: list[dict[str, Any]], retriever: FAISSRetriever, top_k: int,
                         client: OllamaClient) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate and judge grounded answers with a local Ollama model."""
    per_question = []
    for i, record in enumerate(records, start=1):
        logger.info("Generating and judging question %d/%d", i, len(records))
        try:
            results = retriever.retrieve(record["question"], top_k=top_k)
            context = "\n\n".join(f"[Context {rank}] {item.text}" for rank, item in enumerate(results, 1))
            answer = client.generate(
                "Answer the question using only the context. If the context is insufficient, say so. "
                "Be concise and include [Context N] citations.\n\n"
                f"Question: {record['question']}\n\nContext:\n{context}"
            )
            judge_prompt = (
                "You are a strict evaluator of a closed-world RAG system. Return JSON only with three "
                "numeric scores from 0 to 1: correctness (matches the reference), faithfulness "
                "(every material claim is supported by context), and relevance (answers the question). "
                "Do not add markdown or explanation.\n\n"
                f"Question: {record['question']}\n"
                f"Reference answer: {record['ground_truth']}\n"
                f"Context:\n{context}\n"
                f"Generated answer: {answer}"
            )
            scores = parse_judge_response(client.generate(judge_prompt))
            per_question.append({"question": record["question"], "answer": answer, **scores})
        except OllamaError as exc:
            logger.warning("Skipping question %d/%d due to error: %s", i, len(records), exc)
            continue

    if not per_question:
        raise VectorStoreError("No questions were successfully evaluated with Ollama.")

    aggregate = {
        metric: sum(row[metric] for row in per_question) / len(per_question)
        for metric in ("correctness", "faithfulness", "relevance")
    }
    aggregate.update({
        "question_count": len(records),
        "evaluated_count": len(per_question),
        "top_k": top_k,
    })
    return aggregate, per_question


def main() -> None:
    config = Config.from_env()
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval, or generated answers with local Ollama.")
    parser.add_argument("golden_set", type=Path, help="Path to the golden set .jsonl file.")
    parser.add_argument("database", type=Path, help="Path to the vector database directory (from embedding.py).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Where to save results (default: <golden_set_stem>_results.json next to the golden set).")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve per question (default: 3).")
    parser.add_argument("--llm-model", default=None, help="Use Ollama for answer generation and judging (for example, llama3.2:3b).")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL (default: http://localhost:11434).")
    parser.add_argument("--retrieval-judge", choices=("ragas", "ragas_nonllm", "token_overlap"), default="ragas_nonllm",
                        help="Retrieval judge: free RAGAS non-LLM metrics (default), full LLM-based RAGAS (needs OPENAI_API_KEY, costs money), or free local token overlap (not comparable to real RAGAS numbers).")
    
    args = parser.parse_args()

    if not args.golden_set.is_file():
        parser.error(f"Golden set file does not exist: {args.golden_set}")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    retriever = FAISSRetriever.load(args.database)
    if args.llm_model:
        records = load_golden_set(args.golden_set)
        print(f"Loaded {len(records)} golden-set question(s).")
        print(f"Generating and judging answers with local Ollama model: {args.llm_model}")
        aggregate_scores, per_question = evaluate_with_ollama(
            records, retriever, args.top_k, OllamaClient(args.llm_model, args.ollama_url)
        )
    elif args.retrieval_judge == "token_overlap":
        records = load_golden_set(args.golden_set)
        print(f"Loaded {len(records)} golden-set question(s).")
        print("Scoring retrieval with local token overlap (context_precision, context_recall)...")
        aggregate_scores, per_question = evaluate_with_token_overlap(records, retriever, args.top_k)
    elif args.retrieval_judge == "ragas_nonllm":
        records = load_golden_set_with_contexts(args.golden_set)
        print(f"Loaded {len(records)} golden-set question(s) with reference contexts.")
        print("Scoring retrieval with RAGAS non-LLM metrics (free, no model needed)...")
        aggregate_scores, per_question = evaluate_with_ragas_nonllm(records, retriever, args.top_k)
    else:
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall

        records = load_golden_set(args.golden_set)
        print(f"Loaded {len(records)} golden-set question(s).")
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
            {"question": row["question"], "context_precision": float(row["context_precision"]),
             "context_recall": float(row["context_recall"])}
            for _, row in scores_df.iterrows()
        ]

    output_path = args.output or args.golden_set.with_name(args.golden_set.stem + "_results.json")
    save_results(aggregate_scores, per_question, output_path)

    print("\n=== Aggregate scores ===")
    for metric, score in aggregate_scores.items():
        if metric not in {"question_count", "top_k", "judge", "evaluated_count"}:
            print(f"{metric}: {score:.4f}")
    print(f"\nSaved full results to {output_path}")


if __name__ == "__main__":
    main()

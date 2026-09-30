"""Evaluate retrieval and generated answers against a golden set.

The golden set is a JSONL file. Each record must contain:

    - "question"
    - "reference_contexts": one or more reference passages

Optional:
    - "reference_answer" or "ground_truth"

Retrieval evaluation:
    Uses RAGAS non-LLM metrics by default. This is free and does not
    require an API key or an LLM judge.

Generation evaluation:
    Uses the same RAGGenerator and LLMProvider architecture as generation.py.
    The provider can be Ollama or Groq.

Examples:

    Retrieval evaluation:
        python src/evaluate.py \
            data/evaluation/golden_set.jsonl \
            data/vector_db/AI_Engineering \
            --mode retrieval\
            --top-k 3 \
            -o results/retrieval_results.json

    Generation evaluation with Ollama:
        python src/evaluate.py \
            data/evaluation/golden_set.jsonl \
            data/vector_db/AI_Engineering \
            --provider ollama \
            --model llama3.2 \
            --top-k 3 \
            -o results/llama3.2_eval.json

    Generation evaluation with Groq:
        python src/evaluate.py \
            data/evaluation/golden_set.jsonl \
            data/vector_db/AI_Engineering \
            --provider groq \
            --model llama-3.3-70b-versatile \
            --top-k 3 \
            -o results/groq_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from config import Config
from generation import RAGGenerator
from llm_providers import get_provider
from ragas_nonllm_eval import evaluate_with_ragas_nonllm, load_golden_set_with_contexts
from retrieval import FAISSRetriever
from vector_store import VectorStoreError


logger = logging.getLogger(__name__)


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    """Load generation-evaluation records from a JSONL golden set."""

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
                    "Skipping line %d of %s: missing question or reference_contexts.",
                    line_number,
                    path,
                )
                continue

            if isinstance(reference_contexts, str):
                reference_contexts = [reference_contexts]

            reference_answer = (
                record.get("reference_answer")
                or record.get("ground_truth")
            )

            records.append(
                {
                    "id": record.get("id"),
                    "question": question,
                    "reference_contexts": reference_contexts,
                    "reference_answer": reference_answer,
                }
            )

    if not records:
        raise VectorStoreError(
            f"No usable golden-set records found in {path}."
        )

    return records


def parse_judge_response(response: str) -> dict[str, float]:
    """Parse a judge JSON response containing three scores from 0 to 1."""

    cleaned = response.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        scores = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Judge did not return valid JSON: {response}"
        ) from exc

    if not isinstance(scores, dict):
        raise ValueError("Judge response must be a JSON object.")

    required = ("correctness", "faithfulness", "relevance")

    if any(key not in scores for key in required):
        raise ValueError(
            "Judge response is missing one of: "
            + ", ".join(required)
        )

    try:
        return {
            key: max(0.0, min(1.0, float(scores[key])))
            for key in required
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Judge scores must be numeric values from 0 to 1."
        ) from exc


def evaluate_with_generation(
    records: list[dict[str, Any]],
    generator: RAGGenerator,
    judge_provider: Any,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate and judge answers using the project's shared RAG architecture."""

    per_question: list[dict[str, Any]] = []

    for i, record in enumerate(records, start=1):
        logger.info(
            "Generating and judging question %d/%d",
            i,
            len(records),
        )

        try:
            result = generator.generate_answer(
                record["question"],
                top_k=top_k,
            )

            reference_contexts = "\n\n".join(
                f"[Reference Context {rank}] {text}"
                for rank, text in enumerate(
                    record["reference_contexts"],
                    start=1,
                )
            )

            retrieved_contexts = "\n\n".join(
                f"[Retrieved Context {rank}] {source.text}"
                for rank, source in enumerate(
                    result.sources,
                    start=1,
                )
            )

            reference_answer = record.get("reference_answer")

            if reference_answer:
                reference_section = (
                    f"Reference answer:\n{reference_answer}\n\n"
                )
            else:
                reference_section = ""

            judge_prompt = (
                "You are a strict evaluator of a closed-world RAG system.\n\n"
                "Evaluate the generated answer using the question, "
                "reference evidence, retrieved context, and generated answer.\n\n"
                "Return JSON only with exactly three numeric scores from 0 to 1:\n"
                "- correctness: Does the answer correctly address the question "
                "and agree with the reference evidence?\n"
                "- faithfulness: Are the material claims in the answer supported "
                "by the retrieved context?\n"
                "- relevance: Does the answer directly and concisely answer "
                "the question?\n\n"
                "Do not add markdown or explanation.\n\n"
                f"Question:\n{record['question']}\n\n"
                f"{reference_section}"
                f"Reference evidence:\n{reference_contexts}\n\n"
                f"Retrieved context:\n{retrieved_contexts}\n\n"
                f"Generated answer:\n{result.answer}"
            )

            scores = parse_judge_response(
                judge_provider.generate(judge_prompt)
            )

            per_question.append(
                {
                    "id": record.get("id"),
                    "question": record["question"],
                    "answer": result.answer,
                    "sources": [
                        {
                            "score": source.score,
                            "page": source.page,
                            "section": source.section,
                            "chunk_id": source.chunk_id,
                            "text": source.text,
                        }
                        for source in result.sources
                    ],
                    **scores,
                }
            )

        except Exception as exc:
            logger.warning(
                "Skipping question %d/%d due to error: %s",
                i,
                len(records),
                exc,
            )
            continue

    if not per_question:
        raise VectorStoreError(
            "No questions were successfully evaluated."
        )

    metrics = ("correctness", "faithfulness", "relevance")

    aggregate = {
        metric: sum(row[metric] for row in per_question)
        / len(per_question)
        for metric in metrics
    }

    aggregate.update(
        {
            "question_count": len(records),
            "evaluated_count": len(per_question),
            "top_k": top_k,
            "judge": "llm",
        }
    )

    return aggregate, per_question


def save_results(
    scores: dict[str, Any],
    per_question: list[dict[str, Any]],
    path: Path,
) -> None:
    """Save aggregate and per-question evaluation results."""

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "aggregate_scores": scores,
        "per_question": per_question,
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Saved evaluation results to %s", path)


def main() -> None:
    config = Config.from_env()

    logging.basicConfig(
        level=getattr(
            logging,
            config.log_level.upper(),
            logging.INFO,
        ),
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval or generated answers."
    )

    parser.add_argument(
        "golden_set",
        type=Path,
        help="Path to the golden-set JSONL file.",
    )

    parser.add_argument(
        "database",
        type=Path,
        help="Path to the FAISS vector database directory.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to save results. "
            "Default: <golden_set_stem>_results.json"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per question (default: 3).",
    )

    parser.add_argument(
        "--provider",
        choices=("ollama", "groq"),
        default=config.generation_provider,
        help="LLM provider for generation evaluation.",
    )

    parser.add_argument(
        "--model",
        default=config.generation_model,
        help="LLM model used for generation and judging.",
    )

    parser.add_argument(
        "--mode",
        choices=("retrieval", "generation"),
        default="retrieval",
        help="Evaluation mode: retrieval or generation.",
    )

    args = parser.parse_args()

    if not args.database.is_dir():
        parser.error(
            f"Database directory does not exist: {args.database}"
        )

    if not args.golden_set.is_file():
        parser.error(
            f"Golden set file does not exist: {args.golden_set}"
        )

    if args.top_k <= 0:
        parser.error("--top-k must be positive.")

    retriever = FAISSRetriever.load(args.database)

    if args.mode == "retrieval":
        records = load_golden_set_with_contexts(args.golden_set)

        print(f"Loaded {len(records)} golden-set question(s) with reference contexts.")
        print("Scoring retrieval with RAGAS non-LLM metrics...")

        aggregate_scores, per_question = evaluate_with_ragas_nonllm(
            records,
            retriever,
            args.top_k,
        )

    else:
        records = load_golden_set(args.golden_set)

        provider = get_provider(
            provider=args.provider,
            model=args.model,
        )

        generator = RAGGenerator(
            retriever=retriever,
            llm=provider,
        )

        print(f"Loaded {len(records)} golden-set question(s).")
        print(f"Generating answers with {args.provider}/{args.model}...")

        aggregate_scores, per_question = evaluate_with_generation(
            records,
            generator,
            provider,
            args.top_k,
        )
    output_path = (
        args.output
        or args.golden_set.with_name(
            args.golden_set.stem + "_results.json"
        )
    )

    save_results(
        aggregate_scores,
        per_question,
        output_path,
    )

    print("\n=== Aggregate scores ===")

    for metric, score in aggregate_scores.items():
        if metric not in {
            "question_count",
            "evaluated_count",
            "top_k",
            "judge",
        }:
            print(f"{metric}: {score:.4f}")

    print(f"\nSaved full results to {output_path}")


if __name__ == "__main__":
    main()

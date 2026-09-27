
"""
RAG answer generation using retrieved context and a local LLM.
 
 python src/generation.py --top-k 3 --model qwen3:14b "data/vector_db/AI Engineering" "What is RAG?" 
            

"""


import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path


from config import Config
from ollama_client import OllamaClient, OllamaError
from retrieval import FAISSRetriever, RetrievedChunk
from llm_providers import get_provider, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk]

class RAGGenerator:

    # logger.info("Retrieving top-%d chunks for question: %s", top_k, question[:60])

    def __init__(
        self,
        retriever: FAISSRetriever,
        llm: LLMProvider,
    ) -> None:
        self.retriever = retriever
        self.llm = llm

    def build_prompt(
        self,
        question: str,
        contexts: list[RetrievedChunk],
    ) -> str:
        context_text = "\n\n".join(
            f"[Context {i}]\n{chunk.text}"
            for i, chunk in enumerate(contexts, start=1)
        )
        
        complete_prompt = f"""You are a helpful assistant answering questions
        using retrieved context.

        Use ONLY the information provided in the context.
        If the context does not contain enough information to answer
        the question, say that you do not have enough information.

        Cite the relevant context numbers in your answer, for example:
        [Context 1]

        Context:
        {context_text}

        Question:
        {question}

        Answer:"""

        return complete_prompt

    def generate_answer(
        self,
        question: str,
        top_k: int = 3,
    ) -> RAGAnswer:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        retrieved_chunks = self.retriever.retrieve(
            question,
            top_k=top_k,
        )
        
        logger.info("Retrieved %d chunk(s) for question: %s", len(retrieved_chunks), question[:60])

        if not retrieved_chunks:
            raise ValueError("No relevant context was retrieved.")
        prompt = self.build_prompt(
            question,
            retrieved_chunks,
        )

        try:
            answer = self.llm.generate(prompt)
        except OllamaError as exc:
            logger.error("LLM generation failed: %s", exc)
            raise


        logger.info("Generated answer with %d source(s).", len(retrieved_chunks))

        return RAGAnswer(
            question=question,
            answer=answer,
            sources=retrieved_chunks,
        )

def main() -> None:
    
    app_config: Config = Config.from_env()

    logging.basicConfig(
        level=getattr(logging, app_config.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate an answer using retrieved context and a local LLM."
    )

    parser.add_argument(
        "database",
        type=Path,
        help="Path to the FAISS vector database.",
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question to answer.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to use (default: 3).",
    )

    parser.add_argument(
        "--provider",
        choices=("ollama", "groq"),
        default=app_config.generation_provider,
        help="LLM provider (default: from .env)",
    )

    parser.add_argument(
        "--model",
        default=app_config.generation_model,
        help="Model name (default: from .env)",
    )

    parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Optional path to save the generated result as JSON.",
    )

    args = parser.parse_args()

    retriever = FAISSRetriever.load(args.database)

    llm = get_provider(args.provider, args.model)

    generator = RAGGenerator(
        retriever=retriever,
        llm=llm,
    )

    result = generator.generate_answer(
        question=args.question,
        top_k=args.top_k,
    )

    if args.output:
        output_data = {
            "provider": args.provider,
            "model": args.model,
            "question": result.question,
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
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)

        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info("Result saved to: %s", args.output)

    print("\n=== Answer ===")
    print(result.answer)

    print("\n=== Sources ===")
    for i, source in enumerate(result.sources, start=1):
        print(
            f"\n[Context {i}] "
            f"score={source.score:.4f}, "
            f"page={source.page}, "
            f"section={source.section}"
        )
        print(source.text[:200] + "..." if len(source.text) > 200 else source.text)


if __name__ == "__main__":
    main()

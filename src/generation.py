
"""
RAG answer generation using retrieved context and a local LLM.
 
 python src/generation.py data/vector_db/AI Engineer "What is RAG?"

"""



import logging
import argparse
from pathlib import Path

from dataclasses import dataclass


from ollama_client import OllamaClient
from retrieval import FAISSRetriever, RetrievedChunk

logger = logging.getLogger(__name__)

@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk]

class RAGGenerator:
    def __init__(
        self,
        retriever: FAISSRetriever,
        llm: OllamaClient,
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

        if not retrieved_chunks:
            raise ValueError("No relevant context was retrieved.")
        prompt = self.build_prompt(
            question,
            retrieved_chunks,
        )

        answer = self.llm.generate(prompt)

        return RAGAnswer(
            question=question,
            answer=answer,
            sources=retrieved_chunks,
        )

def main() -> None:
    

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
        "--model",
        type=str,
        default="qwen3:14b",
        help="Ollama model name (default: qwen3:14b).",
    )

    args = parser.parse_args()

    retriever = FAISSRetriever.load(args.database)

    llm = OllamaClient(
        model=args.model,
    )

    generator = RAGGenerator(
        retriever=retriever,
        llm=llm,
    )

    result = generator.generate_answer(
        question=args.question,
        top_k=args.top_k,
    )

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
        print(source.text)


if __name__ == "__main__":
    main()

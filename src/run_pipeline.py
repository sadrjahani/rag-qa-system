"""
Run the full rag-qa-system pipeline (PDF -> clean text -> chunks -> vector
database -> optional retrieval) in one command, by importing and calling the
actual stage functions directly — no subprocess calls and no repeated flags.

Must be run from the same folder as the other three scripts (src/), since
it imports them as local modules.

Usage:
    python run_pipeline.py "../data/chapter 6 Rag & Agents.pdf"
    python run_pipeline.py "../data/chapter.pdf" --chunk-size 400 --overlap 60
    python run_pipeline.py "../data/chapter.pdf" --model-name all-MiniLM-L6-v2
    python run_pipeline.py "../data/chapter.pdf" --query "What is RAG?" --top-k 3
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import convert_pdf_to_text as pdf_stage
import chunking as chunking_stage
import embedding as embedding_stage
from Retrieval import FAISSRetriever
from config import Config


def run_pipeline(
    pdf_path: Path,
    chunk_size: int = 400,
    overlap: int = 60,
    heading_pattern: str | None = None,
    model_name: str | None = None,
    batch_size: int = 32,
    output_dir: Path | None = None,
    query: str | None = None,
    top_k: int = 3,
) -> Path:
    """Run PDF extraction, chunking, and embedding. If ``query`` is given,
    search the newly saved vector database as a final stage. Returns the path
    to the saved vector database folder.
    """
    output_dir = output_dir or pdf_path.with_name(pdf_path.stem + "_vectors")
    text_path = pdf_path.with_suffix(".txt")
    meta_path = pdf_path.with_suffix(".meta.json")
    chunks_path = pdf_path.with_name(pdf_path.stem + "_chunks.jsonl")

    # --- Stage 1: PDF -> clean text + heading/page metadata ---
    print("=== Stage 1/3: extracting text from PDF ===")
    text, metadata = pdf_stage.extract_text_with_filters(pdf_path)
    pdf_stage.save_text_to_file(text, text_path)
    pdf_stage.save_metadata_to_file(metadata, meta_path)
    print(f"Saved text to {text_path}")
    print(f"Saved metadata ({len(metadata['headings'])} heading candidates) to {meta_path}\n")

    # --- Stage 2: clean text -> chunks ---
    print("=== Stage 2/3: chunking text ===")
    chunks = chunking_stage.chunk_document(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
        heading_pattern=heading_pattern,
        headings=metadata["headings"],
    )
    chunking_stage.save_chunks_to_jsonl(chunks, chunks_path)
    print(f"Created {len(chunks)} chunks, saved to {chunks_path}\n")

    # --- Stage 3: chunks -> embeddings -> FAISS vector database ---
    print("=== Stage 3/3: embedding chunks and building the vector database ===")
    config = Config.from_env()
    resolved_model_name = model_name or config.embedding_model_name

    documents = [asdict(chunk) for chunk in chunks]
    embedder = embedding_stage.EmbeddingModel(resolved_model_name)
    embeddings = embedder.encode_batch([doc["text"] for doc in documents], batch_size)

    actual_dimension = int(embeddings.shape[1])
    store = embedding_stage.FAISSVectorStore(dimension=actual_dimension)
    store.add_documents(documents, embeddings)
    store.save(output_dir, resolved_model_name)
    print(f"Built a vector database with {store.size} chunks: {output_dir}\n")

    if query is not None:
        print("=== Stage 4/4: retrieving relevant chunks ===")
        retriever = FAISSRetriever.load(output_dir)
        results = retriever.retrieve(query, top_k)
        print(f"\nResults for: {query}\n")
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. Score: {result.score:.4f}")
            if result.source:
                print(f"   Source: {result.source}")
            print(f"   Section: {result.section}")
            if result.page is not None:
                print(f"   Page: {result.page}")
            print(f"   Chunk ID: {result.chunk_id}")
            print(f"   Text: {result.text}\n")

    print("=== Pipeline complete ===")
    print(f"  Text:     {text_path}")
    print(f"  Metadata: {meta_path}")
    print(f"  Chunks:   {chunks_path}")
    print(f"  Vectors:  {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run the full PDF -> text -> chunks -> vectors pipeline in one command.")
    parser.add_argument("pdf", type=Path, help="Path to the input PDF file.")
    parser.add_argument("--chunk-size", type=int, default=400, help="Target chunk size in tokens (default: 400).")
    parser.add_argument("--overlap", type=int, default=60, help="Overlap between chunks in tokens (default: 60).")
    parser.add_argument(
        "--heading-pattern", type=str, default=None,
        help="Optional regex for section headings. If omitted, headings auto-detected from the PDF are used instead."
    )
    parser.add_argument("--model-name", type=str, default=None, help="SentenceTransformer model name (default: from .env, else all-MiniLM-L6-v2).")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (default: 32).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory for the vector database (default: <pdf_stem>_vectors next to the PDF).")
    parser.add_argument("--query", type=str, default=None, help="Optional question to search after building the vector database.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to return for --query (default: 3).")
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"Input file does not exist: {args.pdf}")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.overlap < 0 or args.overlap >= args.chunk_size:
        parser.error("--overlap must be non-negative and smaller than --chunk-size")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.query is not None and not args.query.strip():
        parser.error("--query must not be empty")

    run_pipeline(
        args.pdf,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        heading_pattern=args.heading_pattern,
        model_name=args.model_name,
        batch_size=args.batch_size,
        output_dir=args.output,
        query=args.query,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

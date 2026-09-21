"""
Run the full rag-qa-system pipeline (PDF -> clean text -> chunks -> vector
database -> optional retrieval) in one command, by importing and calling the
actual stage functions directly — no subprocess calls and no repeated flags.

By default, a PDF in ``data/source/`` produces artifacts in:

    data/cleaned/<pdf_stem>.txt and .meta.json
    data/chunks/<pdf_stem>_chunks.jsonl
    data/vector_db/<pdf_stem>/

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
from config import Config
from embedding_model import EmbeddingModel
from vector_store import FAISSVectorStore, VectorStoreError
from retrieval import FAISSRetriever, print_results, save_results_to_json


def default_artifact_directories(pdf_path: Path) -> tuple[Path, Path, Path]:
    """Return clean, predictable output directories for a source PDF."""
    source_dir = pdf_path.parent
    data_dir = source_dir.parent if source_dir.name.lower() == "source" else source_dir
    return (
        data_dir / "cleaned",
        data_dir / "chunks",
        data_dir / "vector_db" / pdf_path.stem,
    )


def run_pipeline(
    pdf_path: Path,
    chunk_size: int = 400,
    overlap: int = 60,
    heading_pattern: str | None = None,
    model_name: str | None = None,
    batch_size: int = 32,
    cleaned_dir: Path | None = None,
    chunks_dir: Path | None = None,
    output_dir: Path | None = None,
    query: str | None = None,
    top_k: int = 3,
) -> Path:
    """Run PDF extraction, chunking, and embedding. If ``query`` is given,
    search the newly saved vector database as a final stage. Generated text,
    chunks, and vectors are written to separate directories. Returns the path
    to the saved vector database folder.
    """
    total_stages = 4 if query is not None else 3
    default_cleaned_dir, default_chunks_dir, default_output_dir = default_artifact_directories(pdf_path)
    cleaned_dir = cleaned_dir or default_cleaned_dir
    chunks_dir = chunks_dir or default_chunks_dir
    output_dir = output_dir or default_output_dir
    text_path = cleaned_dir / f"{pdf_path.stem}.txt"
    meta_path = cleaned_dir / f"{pdf_path.stem}.meta.json"
    chunks_path = chunks_dir / f"{pdf_path.stem}_chunks.jsonl"

    # --- Stage 1: PDF -> clean text + heading/page metadata ---
    print(f"=== Stage 1/{total_stages}: extracting text from PDF ===")
    text, metadata = pdf_stage.extract_text_with_filters(pdf_path)
    pdf_stage.save_text_to_file(text, text_path)
    pdf_stage.save_metadata_to_file(metadata, meta_path)
    print(f"Saved text to {text_path}")
    print(f"Saved metadata ({len(metadata['headings'])} heading candidates) to {meta_path}\n")

    # --- Stage 2: clean text -> chunks ---
    print(f"=== Stage 2/{total_stages}: chunking text ===")
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
    print(f"=== Stage 3/{total_stages}: embedding chunks and building the vector database ===")
    config = Config.from_env()
    resolved_model_name = model_name or config.embedding_model_name

    documents = [asdict(chunk) for chunk in chunks]
    embedder = EmbeddingModel(resolved_model_name)
    embeddings = embedder.encode_batch(
        [doc["text"] for doc in documents], 
         batch_size=batch_size,
        )

    actual_dimension = int(embeddings.shape[1])
    if (
        config.embedding_dimension is not None
        and config.embedding_dimension != actual_dimension
    ):
        raise VectorStoreError(
            "EMBEDDING_DIMENSION does not match the selected model: "
            f"{config.embedding_dimension} != {actual_dimension}. "
            "Correct the value in .env before building the database."
        )
    store = FAISSVectorStore(dimension=actual_dimension)
    store.add_documents(documents, embeddings)
    store.save(output_dir, resolved_model_name)
    print(f"Built a vector database with {store.size} chunks: {output_dir}\n")

    # --- Stage 4 (optional): retrieve and save results ---
    if query is not None:
        print(f"=== Stage 4/{total_stages}: retrieving relevant chunks ===")
        retriever = FAISSRetriever.load(output_dir)
        results = retriever.retrieve(query, top_k)
        print_results(query, results)

        results_path = output_dir / "query_results.json"
        save_results_to_json(query, results, results_path)
        print(f"Saved results to {results_path}\n")

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
    parser.add_argument("--cleaned-dir", type=Path, default=None, help="Directory for cleaned text and metadata (default: data/cleaned for PDFs in data/source).")
    parser.add_argument("--chunks-dir", type=Path, default=None, help="Directory for chunk JSONL files (default: data/chunks for PDFs in data/source).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Vector database directory (default: data/vector_db/<pdf_stem> for PDFs in data/source).")
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
        cleaned_dir=args.cleaned_dir,
        chunks_dir=args.chunks_dir,
        output_dir=args.output,
        query=args.query,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

"""Build a local FAISS vector database from ``chunking.py`` JSONL output.

Each input line must be a JSON object with at least a ``text`` field. The
original chunk record is saved alongside the FAISS index, so a retriever can
map search results back to source text and section later.

Example:
    python src/embedding.py data/chapter_chunks.jsonl -o data/chapter_vectors
"""

import argparse
import logging
from pathlib import Path

from config import Config
from embedding_model import EmbeddingModel
from vector_store import FAISSVectorStore, VectorStoreError, read_chunks_from_jsonl

logger = logging.getLogger(__name__)


def main() -> None:
    config = Config.from_env()
    parser = argparse.ArgumentParser(description="Embed chunk JSONL and build a local FAISS vector database.")
    parser.add_argument("input", type=Path, help="Path to the .jsonl file produced by chunking.py.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory (default: <input>_vectors next to the input).")
    parser.add_argument("--model-name", default=config.embedding_model_name, help="SentenceTransformer model name.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (default: 32).")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    documents = read_chunks_from_jsonl(args.input)
    embedder = EmbeddingModel(args.model_name)
    embeddings = embedder.encode_batch([document["text"] for document in documents], args.batch_size)

    actual_dimension = int(embeddings.shape[1])
    if config.embedding_dimension is not None and config.embedding_dimension != actual_dimension:
        logger.error(
            "EMBEDDING_DIMENSION mismatch: .env says %d but model '%s' produces %d. "
            "Update or remove EMBEDDING_DIMENSION in .env before rebuilding.",
            config.embedding_dimension, args.model_name, actual_dimension,
        )
        raise VectorStoreError(
            "EMBEDDING_DIMENSION does not match the selected model: "
            f"{config.embedding_dimension} != {actual_dimension}. Remove or correct it in .env."
        )

    store = FAISSVectorStore(dimension=actual_dimension)
    store.add_documents(documents, embeddings)
    output_path = args.output or args.input.with_name(args.input.stem + "_vectors")
    store.save(output_path, args.model_name)
    print(f"Built a vector database with {store.size} chunks: {output_path}")


if __name__ == "__main__":
    main()

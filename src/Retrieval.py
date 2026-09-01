
import argparse
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import Config

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):


@dataclass
class EmbeddingModel:
    text: str
    model_name : str
    _model: SentenceTransformer | None = None


    @property
    def model() -> SentenceTransformer:
        if _model is None:
            logger.info("Loading Embedding model: %s",model_name)
            _model = SentenceTransformer(model_name)
        return _model

    def encode(text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("The input has not to be empty!")
        return model.encode(text)

        
    
@dataclass
class FAISSVectorStore:
    dimension: int
    model_name : str
    _documents: list[str] = []
    _index = faiss.IndexFlatIP(dimension)


    def search(query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:

        if size == 0:
            raise VectorStoreError("The index is empty; first call add_documen")

        query = query_embedding.reshape(1, -1).astype("float32").copy()
        faiss.normalize_L2(query)

        top_k = min(top_k, size)
        scores, indices = _index.search(query, top_k)

        return [
            (_documents[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
        ]

    @classmethod
    def load(cls, path: str | Path, dimension: int) -> "FAISSVectorStore":
        """Retrieve saved index from disc."""
        path = Path(path)
        store = cls(dimension)
        store._index = faiss.read_index(str(path / "index.faiss"))
        store._documents = (
            (path / "VDB.txt").read_text(encoding="utf-8").splitlines()
        )
        logger.info("retrieved index from : %s (%d document)", path, store.size)
        yield store

def read_chunks_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read and validate the line-delimited records written by chunking.py."""
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VectorStoreError(
                    f"Invalid JSON on line {line_number} of {path}: {exc.msg}"
                ) from exc
            if not isinstance(document, dict) or not isinstance(document.get("text"), str):
                raise VectorStoreError(
                    f"Line {line_number} of {path} must be an object with a string 'text' field."
                )
            if not document["text"].strip():
                raise VectorStoreError(f"Line {line_number} of {path} has empty text.")
            documents.append(document)

    if not documents:
        raise VectorStoreError(f"No chunk records found in {path}.")
    return documents


def main() -> None:
    config = Config.from_env()
    parser = argparse.ArgumentParser(description="Embed Query and find answers from local FAISS vector database.")
    parser.add_argument("input", type=Path, help="""Path to the local FAISS vector database folder <input>_vectors containing
         metadata.json file ,documents.jsonl and index.faiss file produced by embedding.py.
         Example: /path/to/your_vdb_folder,
         Example: /path/to/your_vdb_folder/metadata.json,
         Example: /path/to/your_vdb_folder/documents.jsonl,
         Example: /path/to/your_vdb_folder/index.faiss""")
    parser.add_argument("query", type=str, help="Query to search for in the vector database.")
    
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory (default:  <input>_top_answers next to the input).")
    parser.add_argument("--model-name", default=config.embedding_model_name, help="SentenceTransformer model name.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (default: 32).")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    # ---------------change loader def for all files
    documents = read_chunks_from_jsonl(args.input)
    embedder = EmbeddingModel(args.model_name)

    logger.info("search for: %s", args.query)
    query_embedding = embedder.encode(args.query)
    

    actual_dimension = int(embeddings.shape[1])
    if config.embedding_dimension is not None and config.embedding_dimension != actual_dimension:
        raise VectorStoreError(
            "EMBEDDING_DIMENSION does not match the selected model: "
            f"{config.embedding_dimension} != {actual_dimension}. Remove or correct it in .env."
        )


    store = FAISSVectorStore(dimension=config.embedding_dimension)

    
  
    results = store.search(query_embedding, top_k=config.top_k)

    print(f"\nنتایج جست‌وجو برای «{QUERY}»:\n")
    for rank, (document, score) in enumerate(results, start=1):
        print(f"  {rank}. [{score:.4f}] {document}")


if __name__ == "__main__":
    main()
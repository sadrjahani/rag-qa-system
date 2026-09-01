
import argparse
import json

from dataclasses import dataclass
from pathlib import Path

import logging
import numpy as np
import faiss


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



def main() -> None:
    store = FAISSVectorStore(dimension=config.embedding_dimension)

    logger.info("جست‌وجو برای: %s", QUERY)
    query_embedding = embedder.encode(QUERY)
    results = store.search(query_embedding, top_k=config.top_k)

    print(f"\nنتایج جست‌وجو برای «{QUERY}»:\n")
    for rank, (document, score) in enumerate(results, start=1):
        print(f"  {rank}. [{score:.4f}] {document}")


if __name__ == "__main__":
    main()
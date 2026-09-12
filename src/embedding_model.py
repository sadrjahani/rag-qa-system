"""The embedding model wrapper, shared by embedding.py (building the vector
database) and retrieval.py (embedding queries at search time).
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModel:
    model_name: str
    _model: SentenceTransformer | None = field(default=None, init=False, repr=False)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")
        logger.info("Encoding %d chunk(s).", len(texts))
        # normalize_embeddings=True already L2-normalizes here, so vectors
        # arriving in FAISSVectorStore.add_documents are normalized once,
        # by this method. The second normalize_L2 call there is a defensive
        # no-op for these vectors, kept in case add_documents is ever called
        # with embeddings from a different source that skipped this step.
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar = show_progress,
        )
        return np.array(embeddings, dtype=np.float32, copy=True)

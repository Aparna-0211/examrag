"""Multilingual embeddings via sentence-transformers (runs fully offline once cached)."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class Embedder:
    """Wraps a sentence-transformers model. Vectors are L2-normalised, so
    cosine similarity equals the dot product."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = _load_model(self.model_name)
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=16,
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class SentenceTransformerBackend:
    """Embeds text with a small pretrained sentence-transformer model."""

    name = "sentence-transformer"

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(vectors.astype("float32"))


class TfidfBackend:
    """Fallback embedding backend used when sentence-transformers/torch can't
    be loaded (e.g. no internet access to download model weights). It keeps
    the retrieval agent's embedding-based design working fully offline.
    """

    name = "tfidf"

    def __init__(self, vectorizer: TfidfVectorizer | None = None):
        self._vectorizer = vectorizer or TfidfVectorizer(ngram_range=(1, 2), max_features=4096)

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._vectorizer.transform(texts).toarray().astype("float32")
        return _normalize(vectors)

    @property
    def vectorizer(self) -> TfidfVectorizer:
        return self._vectorizer


def build_embedding_backend(corpus_texts: list[str]):
    """Try the real embedding model first, fall back to TF-IDF on failure."""
    try:
        return SentenceTransformerBackend()
    except Exception:
        backend = TfidfBackend()
        backend.fit(corpus_texts)
        return backend

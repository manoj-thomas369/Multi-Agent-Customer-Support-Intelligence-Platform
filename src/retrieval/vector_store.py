from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import faiss
import joblib

from src import config
from src.retrieval.embeddings import SentenceTransformerBackend, TfidfBackend, build_embedding_backend


def _tfidf_vectorizer_path():
    return config.VECTOR_INDEX_DIR / "tfidf_vectorizer.joblib"


@dataclass
class CorpusItem:
    text: str
    source: str
    category: str = ""


def build_index(corpus: list[CorpusItem], embed_texts: list[str] | None = None) -> None:
    """Builds the FAISS index. `embed_texts`, when given, is what gets
    embedded for similarity search (e.g. the FAQ question) while
    `item.text` is what's actually shown to the customer (e.g. the answer).
    """
    texts = embed_texts or [item.text for item in corpus]
    backend = build_embedding_backend(texts)
    vectors = backend.encode(texts)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))

    if isinstance(backend, TfidfBackend):
        joblib.dump(backend.vectorizer, _tfidf_vectorizer_path())

    metadata = {
        "backend": backend.name,
        "dim": int(vectors.shape[1]),
        "items": [item.__dict__ for item in corpus],
    }
    config.FAISS_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class VectorStore:
    def __init__(self) -> None:
        self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        metadata = json.loads(config.FAISS_METADATA_PATH.read_text(encoding="utf-8"))
        self.items: list[CorpusItem] = [CorpusItem(**item) for item in metadata["items"]]
        self.backend = self._load_backend(metadata["backend"])

    @staticmethod
    def _load_backend(backend_name: str):
        if backend_name == "sentence-transformer":
            return SentenceTransformerBackend()
        vectorizer = joblib.load(_tfidf_vectorizer_path())
        return TfidfBackend(vectorizer=vectorizer)

    def search(self, query: str, top_k: int | None = None, category: str | None = None):
        top_k = top_k or config.RETRIEVAL_TOP_K
        fetch_k = top_k * 4 if category else top_k
        fetch_k = min(fetch_k, len(self.items)) or 1
        q_vec = self.backend.encode([query])
        scores, indices = self.index.search(q_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = self.items[idx]
            if category and item.category and item.category != category:
                continue
            results.append((item, float(score)))
            if len(results) >= top_k:
                break

        if not results:
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                results.append((self.items[idx], float(score)))
                if len(results) >= top_k:
                    break
        return results

    def add_document(self, text: str, source: str, category: str = "", embed_text: str | None = None) -> None:
        """Grows long-term memory by folding a newly resolved ticket back
        into the retrieval corpus, so future similar tickets benefit from it.
        `text` should be a clean, reusable resolution (like a FAQ answer) -
        not a full formatted reply - since it may itself be shown verbatim
        in a future response. `embed_text` (e.g. complaint + resolution) is
        only used for similarity search when given.
        """
        vector = self.backend.encode([embed_text or text])
        self.index.add(vector)
        self.items.append(CorpusItem(text=text, source=source, category=category))
        self._persist()

    def _persist(self) -> None:
        faiss.write_index(self.index, str(config.FAISS_INDEX_PATH))
        metadata = {
            "backend": self.backend.name,
            "dim": self.index.d,
            "items": [item.__dict__ for item in self.items],
        }
        config.FAISS_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()

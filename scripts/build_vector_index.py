"""Builds the FAISS knowledge base index from the FAQ dataset. The Learning
Agent grows this same index at runtime with confidently resolved tickets, so
this only needs to be re-run if the FAQ dataset itself changes.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.retrieval.vector_store import CorpusItem, build_index


def main() -> None:
    faq_df = pd.read_csv(config.FAQ_DATASET_PATH)

    corpus = [
        CorpusItem(text=row.resolution, source=f"faq:{row.category}:{i}", category=row.category)
        for i, row in enumerate(faq_df.itertuples(index=False))
    ]
    embed_texts = [f"{row.question} {row.resolution}" for row in faq_df.itertuples(index=False)]

    build_index(corpus, embed_texts=embed_texts)
    print(f"Built vector index with {len(corpus)} FAQ entries at {config.FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()

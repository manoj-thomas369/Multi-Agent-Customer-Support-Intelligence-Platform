import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.agents import classification_agent as classification_agent_module
from src.db import database as database_module
from src.ml.classifier import save_pipeline, train_and_evaluate
from src.retrieval import vector_store as vector_store_module
from src.retrieval.embeddings import TfidfBackend

TINY_TICKETS = [
    ("My order #ORD12345 for the wireless earbuds hasn't arrived in 10 days.", "Delivery", "High"),
    ("Order #ORD22222 is delayed, I need the smartwatch urgently.", "Delivery", "Medium"),
    ("Can you track my shipment for order #ORD22223? The phone case is missing.", "Delivery", "Medium"),
    ("Order #ORD22224 shows delivered but I never received my backpack.", "Delivery", "Low"),
    ("I returned my laptop stand from order #ORD33333 but no refund yet.", "Refund", "High"),
    ("Refund for order #ORD44444 is still pending after 3 days.", "Refund", "Medium"),
    ("Requesting a refund for order #ORD44445 since the yoga mat is defective.", "Refund", "Medium"),
    ("Refund status for order #ORD44446 still shows pending.", "Refund", "Low"),
    ("I was charged twice for order #ORD55555, please refund the extra.", "Payment", "High"),
    ("Payment failed for order #ORD66666 but money was deducted.", "Payment", "High"),
    ("I'm unable to complete payment for the air fryer, checkout keeps erroring.", "Payment", "Medium"),
    ("There's an incorrect charge on my account for order #ORD66668.", "Payment", "Medium"),
    ("The backpack from order #ORD77777 arrived damaged.", "Product Issue", "Medium"),
    ("My gaming mouse from order #ORD88888 stopped working.", "Product Issue", "Low"),
    ("I ordered a coffee maker but received the wrong item in order #ORD88889.", "Product Issue", "Medium"),
    ("The power bank from order #ORD88890 is missing accessories.", "Product Issue", "Low"),
    ("I can't log into my account, password reset isn't working.", "Account", "Low"),
    ("My account got locked after failed login attempts.", "Account", "Medium"),
    ("I'm unable to update my shipping address on my account.", "Account", "Low"),
    ("My account shows someone else's order history.", "Account", "Medium"),
]

TINY_FAQ = [
    ("Delivery", "My order hasn't arrived, what should I do?", "We'll send a replacement or refund if it doesn't arrive within 2 extra days."),
    ("Refund", "How long does a refund take?", "Refunds are processed within 5-7 business days of us receiving the return."),
    ("Payment", "I was charged twice, what now?", "Duplicate authorization holds release automatically; we refund any real duplicate within 3 days."),
    ("Product Issue", "My product arrived damaged.", "Share a photo and we'll send a free replacement, no return needed."),
    ("Account", "I can't log in.", "Use the password reset link; we can also manually verify and unlock your account."),
]


@pytest.fixture()
def project_dirs(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    vector_dir = tmp_path / "vector_index"
    for d in (data_dir, models_dir, vector_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "VECTOR_INDEX_DIR", vector_dir)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "support.db")
    monkeypatch.setattr(config, "TICKETS_DATASET_PATH", data_dir / "tickets_dataset.csv")
    monkeypatch.setattr(config, "FAQ_DATASET_PATH", data_dir / "faq_knowledge_base.csv")
    monkeypatch.setattr(config, "CATEGORY_MODEL_PATH", models_dir / "category_classifier.joblib")
    monkeypatch.setattr(config, "PRIORITY_MODEL_PATH", models_dir / "priority_classifier.joblib")
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", vector_dir / "knowledge_base.index")
    monkeypatch.setattr(config, "FAISS_METADATA_PATH", vector_dir / "knowledge_base_meta.json")

    # Force the offline TF-IDF embedding backend so tests never need network
    # access to download a sentence-transformer model.
    def _tfidf_only(texts):
        backend = TfidfBackend()
        backend.fit(texts)
        return backend

    monkeypatch.setattr(vector_store_module, "build_embedding_backend", _tfidf_only)
    vector_store_module.get_vector_store.cache_clear()
    classification_agent_module._category_pipeline.cache_clear()
    classification_agent_module._priority_pipeline.cache_clear()
    database_module.reset_engine_cache()
    database_module.init_db()

    yield tmp_path

    vector_store_module.get_vector_store.cache_clear()
    classification_agent_module._category_pipeline.cache_clear()
    classification_agent_module._priority_pipeline.cache_clear()
    database_module.reset_engine_cache()


@pytest.fixture()
def trained_models(project_dirs):
    df = pd.DataFrame(TINY_TICKETS, columns=["text", "category", "priority"])
    category_pipeline, _ = train_and_evaluate(df["text"], df["category"], test_size=0.3, seed=1)
    priority_pipeline, _ = train_and_evaluate(df["text"], df["priority"], test_size=0.3, seed=1)
    save_pipeline(category_pipeline, config.CATEGORY_MODEL_PATH)
    save_pipeline(priority_pipeline, config.PRIORITY_MODEL_PATH)
    return df


@pytest.fixture()
def vector_index(project_dirs):
    from src.retrieval.vector_store import CorpusItem, build_index

    corpus = [
        CorpusItem(text=resolution, source=f"faq:{category}", category=category)
        for category, _question, resolution in TINY_FAQ
    ]
    embed_texts = [f"{question} {resolution}" for _category, question, resolution in TINY_FAQ]
    build_index(corpus, embed_texts=embed_texts)
    return corpus

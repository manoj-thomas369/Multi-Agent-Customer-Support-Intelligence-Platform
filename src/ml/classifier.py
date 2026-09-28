from __future__ import annotations

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_and_evaluate(texts: pd.Series, labels: pd.Series, test_size: float = 0.2, seed: int = 42):
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    return pipeline, metrics


def save_pipeline(pipeline: Pipeline, path) -> None:
    joblib.dump(pipeline, path)


def load_pipeline(path) -> Pipeline:
    return joblib.load(path)


def predict_with_confidence(pipeline: Pipeline, text: str) -> tuple[str, float]:
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    best_index = proba.argmax()
    return classes[best_index], float(proba[best_index])

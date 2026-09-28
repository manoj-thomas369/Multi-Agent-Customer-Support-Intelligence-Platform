"""Trains the category and priority classifiers on the ticket dataset and
saves both the model artifacts and a metrics report."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.ml.classifier import save_pipeline, train_and_evaluate


def main() -> None:
    df = pd.read_csv(config.TICKETS_DATASET_PATH)

    category_pipeline, category_metrics = train_and_evaluate(df["text"], df["category"])
    save_pipeline(category_pipeline, config.CATEGORY_MODEL_PATH)

    priority_pipeline, priority_metrics = train_and_evaluate(df["text"], df["priority"])
    save_pipeline(priority_pipeline, config.PRIORITY_MODEL_PATH)

    report = {"category": category_metrics, "priority": priority_metrics}
    config.TRAINING_REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Saved models to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()

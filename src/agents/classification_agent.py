from functools import lru_cache

from src import config
from src.agents.base import Agent
from src.ml.classifier import load_pipeline, predict_with_confidence
from src.state import TicketContext


@lru_cache(maxsize=1)
def _category_pipeline():
    return load_pipeline(config.CATEGORY_MODEL_PATH)


@lru_cache(maxsize=1)
def _priority_pipeline():
    return load_pipeline(config.PRIORITY_MODEL_PATH)


class ClassificationAgent(Agent):
    """Predicts the ticket category and priority using trained ML models."""

    name = "classification_agent"

    def run(self, context: TicketContext) -> TicketContext:
        category, confidence = predict_with_confidence(_category_pipeline(), context.cleaned_text)
        priority, _ = predict_with_confidence(_priority_pipeline(), context.cleaned_text)

        context.category = category
        context.category_confidence = confidence
        context.priority = priority

        context.log(
            self.name,
            "classified_ticket",
            category=category,
            category_confidence=round(confidence, 3),
            priority=priority,
        )
        return context

import re

from src.agents.base import Agent
from src.nlp.entities import extract_entities, extract_intent
from src.nlp.sentiment import analyze_sentiment
from src.state import TicketContext

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


class IntakeAgent(Agent):
    """Cleans the raw ticket and extracts intent, sentiment and entities."""

    name = "intake_agent"

    def run(self, context: TicketContext) -> TicketContext:
        context.cleaned_text = clean_text(context.raw_text)
        context.entities = extract_entities(context.cleaned_text)
        context.intent = extract_intent(context.cleaned_text)

        label, score = analyze_sentiment(context.cleaned_text)
        context.sentiment = label
        context.sentiment_score = score

        context.log(
            self.name,
            "parsed_ticket",
            intent=context.intent,
            entities=context.entities,
            sentiment=label,
            sentiment_score=round(score, 3),
        )
        return context

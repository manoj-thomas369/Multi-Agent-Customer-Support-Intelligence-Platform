from src import config
from src.agents.base import Agent
from src.state import TicketContext

_ESCALATION_NOTE = (
    "\n\nI've also flagged this ticket for a specialist on our team to personally "
    "review and make sure it's fully resolved for you."
)


class EscalationAgent(Agent):
    """Decides whether a ticket needs human escalation."""

    name = "escalation_agent"

    def run(self, context: TicketContext) -> TicketContext:
        reason = self._decide(context)
        context.escalate = reason is not None
        context.escalation_reason = reason

        if context.escalate:
            context.response += _ESCALATION_NOTE

        context.log(
            self.name,
            "escalation_decision",
            escalate=context.escalate,
            reason=reason,
        )
        return context

    @staticmethod
    def _decide(context: TicketContext) -> str | None:
        if context.category_confidence < config.ESCALATION_CONFIDENCE_THRESHOLD:
            return "low_classification_confidence"
        if context.sentiment_score <= config.NEGATIVE_SENTIMENT_THRESHOLD:
            return "strongly_negative_sentiment"
        best_score = context.retrieved_docs[0].score if context.retrieved_docs else 0.0
        if best_score < config.RETRIEVAL_RELEVANCE_THRESHOLD:
            return "no_confident_knowledge_match"
        return None

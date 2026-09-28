from src.agents import (
    ClassificationAgent,
    EscalationAgent,
    IntakeAgent,
    LearningAgent,
    ResponseAgent,
    RetrievalAgent,
)
from src.state import TicketContext


class TicketOrchestrator:
    """Runs a ticket through the full agentic pipeline, in order:

    Intake -> Classification -> Retrieval (RAG) -> Response -> Escalation -> Learning
    """

    def __init__(self) -> None:
        self._pipeline = [
            IntakeAgent(),
            ClassificationAgent(),
            RetrievalAgent(),
            ResponseAgent(),
            EscalationAgent(),
            LearningAgent(),
        ]

    def process(self, ticket_id: str, raw_text: str) -> TicketContext:
        context = TicketContext(ticket_id=ticket_id, raw_text=raw_text)
        for agent in self._pipeline:
            context = agent.run(context)
        return context

from datetime import datetime, timezone

from src import config
from src.agents.base import Agent
from src.db.database import get_session
from src.db.models import AgentLog, Ticket
from src.retrieval.vector_store import get_vector_store
from src.state import TicketContext


class LearningAgent(Agent):
    """Persists the ticket outcome and grows long-term retrieval memory."""

    name = "learning_agent"

    def run(self, context: TicketContext) -> TicketContext:
        # Log before persisting so this agent's own decision is included in
        # the log rows written to the database.
        context.log(self.name, "logged_outcome", escalate=context.escalate)
        self._persist(context)
        self._maybe_learn(context)
        return context

    @staticmethod
    def _persist(context: TicketContext) -> None:
        session = get_session()
        try:
            ticket = session.get(Ticket, context.ticket_id)
            if ticket is None:
                ticket = Ticket(id=context.ticket_id, raw_text=context.raw_text)
                session.add(ticket)

            ticket.cleaned_text = context.cleaned_text
            ticket.intent = context.intent
            ticket.entities = context.entities
            ticket.sentiment = context.sentiment
            ticket.sentiment_score = context.sentiment_score
            ticket.category = context.category
            ticket.category_confidence = context.category_confidence
            ticket.priority = context.priority
            ticket.response = context.response
            ticket.escalate = context.escalate
            ticket.escalation_reason = context.escalation_reason
            ticket.status = "processed"
            ticket.processed_at = datetime.now(timezone.utc)

            for entry in context.agent_logs:
                session.add(
                    AgentLog(
                        ticket_id=context.ticket_id,
                        agent=entry.agent,
                        decision=entry.decision,
                        detail=entry.detail,
                    )
                )
            session.commit()
        finally:
            session.close()

    @staticmethod
    def _maybe_learn(context: TicketContext) -> None:
        if context.escalate or context.category_confidence < config.LEARNING_CONFIDENCE_THRESHOLD:
            return
        if not context.resolution_text or context.used_fallback_resolution:
            return
        try:
            store = get_vector_store()
            store.add_document(
                # Store just the clean resolution (like a FAQ answer) so it
                # can be safely reused verbatim in a future reply, without
                # baking in this ticket's own greeting/closing boilerplate.
                text=context.resolution_text,
                source=f"resolved_ticket:{context.ticket_id}",
                category=context.category,
                embed_text=f"{context.cleaned_text}\n{context.resolution_text}",
            )
        except Exception:
            # Learning is a best-effort enhancement; never fail the pipeline over it.
            pass

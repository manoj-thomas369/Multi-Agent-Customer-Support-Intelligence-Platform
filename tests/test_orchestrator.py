from src.db.database import get_session
from src.db.models import Ticket
from src.orchestrator import TicketOrchestrator


def test_pipeline_processes_ticket_end_to_end(trained_models, vector_index):
    orchestrator = TicketOrchestrator()
    context = orchestrator.process(
        "ticket-1",
        "My order #ORD11111 for the wireless earbuds hasn't arrived in 10 days. This is unacceptable!",
    )

    assert context.category
    assert context.priority
    assert context.sentiment
    assert context.response
    assert isinstance(context.escalate, bool)

    agent_names = [entry.agent for entry in context.agent_logs]
    assert agent_names == [
        "intake_agent",
        "classification_agent",
        "retrieval_agent",
        "response_agent",
        "escalation_agent",
        "learning_agent",
    ]


def test_pipeline_persists_ticket_to_db(trained_models, vector_index):
    orchestrator = TicketOrchestrator()
    orchestrator.process("ticket-2", "I can't log into my account, password reset isn't working.")

    session = get_session()
    try:
        ticket = session.get(Ticket, "ticket-2")
        assert ticket is not None
        assert ticket.status == "processed"
        assert len(ticket.logs) == 6
    finally:
        session.close()

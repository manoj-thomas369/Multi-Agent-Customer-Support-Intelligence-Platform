from src.agents.intake_agent import IntakeAgent, clean_text
from src.state import TicketContext


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello   world  \n") == "hello world"


def test_extracts_order_id_and_product():
    context = TicketContext(
        ticket_id="t1",
        raw_text="My order #ORD98765 for the wireless earbuds hasn't arrived yet.",
    )
    result = IntakeAgent().run(context)

    assert result.entities["order_id"] == "ORD98765"
    assert result.entities["product"] == "wireless earbuds"
    assert result.intent == "delivery_delay"


def test_sentiment_reflects_tone():
    angry = IntakeAgent().run(
        TicketContext(ticket_id="t2", raw_text="This is completely unacceptable, I am furious!")
    )
    polite = IntakeAgent().run(
        TicketContext(ticket_id="t3", raw_text="Thanks so much, I really appreciate your help!")
    )

    assert angry.sentiment == "negative"
    assert polite.sentiment == "positive"
    assert angry.sentiment_score < polite.sentiment_score


def test_logs_a_decision_entry():
    context = IntakeAgent().run(TicketContext(ticket_id="t4", raw_text="Hello there"))
    assert len(context.agent_logs) == 1
    assert context.agent_logs[0].agent == "intake_agent"

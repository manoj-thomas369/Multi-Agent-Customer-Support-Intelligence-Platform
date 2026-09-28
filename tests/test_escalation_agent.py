from src.agents.escalation_agent import EscalationAgent
from src.state import RetrievedDoc, TicketContext


def _context(**overrides) -> TicketContext:
    defaults = dict(
        ticket_id="t1",
        raw_text="raw",
        category="Delivery",
        category_confidence=0.9,
        sentiment_score=0.0,
        retrieved_docs=[RetrievedDoc(text="resolution", source="faq:Delivery:0", score=0.8)],
        response="Draft reply.",
    )
    defaults.update(overrides)
    return TicketContext(**defaults)


def test_escalates_on_low_classification_confidence():
    context = EscalationAgent().run(_context(category_confidence=0.3))
    assert context.escalate is True
    assert context.escalation_reason == "low_classification_confidence"


def test_escalates_on_strongly_negative_sentiment():
    context = EscalationAgent().run(_context(sentiment_score=-0.8))
    assert context.escalate is True
    assert context.escalation_reason == "strongly_negative_sentiment"


def test_escalates_on_weak_knowledge_match():
    context = EscalationAgent().run(
        _context(retrieved_docs=[RetrievedDoc(text="x", source="faq:Delivery:0", score=0.1)])
    )
    assert context.escalate is True
    assert context.escalation_reason == "no_confident_knowledge_match"


def test_does_not_escalate_confident_ticket():
    context = EscalationAgent().run(_context())
    assert context.escalate is False
    assert context.escalation_reason is None


def test_escalation_note_appended_to_response():
    context = EscalationAgent().run(_context(category_confidence=0.2))
    assert "specialist" in context.response.lower()

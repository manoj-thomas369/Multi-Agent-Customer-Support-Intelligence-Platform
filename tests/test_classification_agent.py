from src.agents.classification_agent import ClassificationAgent
from src.state import TicketContext


def test_predicts_category_and_priority(trained_models):
    context = TicketContext(
        ticket_id="t1",
        raw_text="ignored",
        cleaned_text="My order #ORD11111 for the smartwatch hasn't arrived in 12 days.",
    )
    result = ClassificationAgent().run(context)

    assert result.category in {"Delivery", "Refund", "Payment", "Product Issue", "Account"}
    assert result.priority in {"Low", "Medium", "High"}
    assert 0.0 <= result.category_confidence <= 1.0
    assert result.agent_logs[-1].decision == "classified_ticket"

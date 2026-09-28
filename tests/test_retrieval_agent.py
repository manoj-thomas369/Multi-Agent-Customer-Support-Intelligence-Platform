from src.agents.retrieval_agent import RetrievalAgent
from src.state import TicketContext


def test_retrieves_relevant_faq(vector_index):
    context = TicketContext(
        ticket_id="t1",
        raw_text="ignored",
        cleaned_text="My order hasn't arrived, what should I do?",
        category="Delivery",
        category_confidence=0.9,
    )
    result = RetrievalAgent().run(context)

    assert len(result.retrieved_docs) > 0
    assert result.retrieved_docs[0].score > 0
    assert "replacement" in result.retrieved_docs[0].text.lower() or "refund" in result.retrieved_docs[0].text.lower()

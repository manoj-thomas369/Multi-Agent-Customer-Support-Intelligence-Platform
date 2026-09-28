from src import config
from src.agents.base import Agent
from src.retrieval.vector_store import get_vector_store
from src.state import RetrievedDoc, TicketContext


class RetrievalAgent(Agent):
    """Finds the most relevant FAQ / past-resolution knowledge for a ticket."""

    name = "retrieval_agent"

    def run(self, context: TicketContext) -> TicketContext:
        store = get_vector_store()
        category = context.category if context.category_confidence >= config.ESCALATION_CONFIDENCE_THRESHOLD else None
        matches = store.search(context.cleaned_text, top_k=config.RETRIEVAL_TOP_K, category=category)

        context.retrieved_docs = [
            RetrievedDoc(text=item.text, source=item.source, score=score) for item, score in matches
        ]

        context.log(
            self.name,
            "retrieved_knowledge",
            top_k=config.RETRIEVAL_TOP_K,
            category_filter=category,
            best_score=round(context.retrieved_docs[0].score, 3) if context.retrieved_docs else 0.0,
            sources=[doc.source for doc in context.retrieved_docs],
        )
        return context

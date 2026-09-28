import os

from src import config
from src.agents.base import Agent
from src.state import TicketContext

_CATEGORY_INTRO = {
    "Delivery": "I understand you're having a delivery issue",
    "Refund": "I understand you're following up on a refund",
    "Payment": "I understand you've run into a payment issue",
    "Product Issue": "I'm sorry to hear about a problem with your product",
    "Account": "I understand you're having trouble with your account",
}

_FALLBACK_BODY = (
    "I've noted the details of your issue and I'm looping in a specialist from our "
    "support team who will personally review your case and follow up shortly."
)


def _format_reference(context: TicketContext) -> str:
    parts = []
    if "product" in context.entities:
        parts.append(context.entities["product"])
    if "order_id" in context.entities:
        parts.append(f"order #{context.entities['order_id']}")
    if not parts:
        return "this"
    return " for ".join(parts)


def _resolve_body(context: TicketContext) -> tuple[str, bool]:
    """Returns (resolution_text, used_fallback)."""
    best_doc = context.retrieved_docs[0] if context.retrieved_docs else None
    if best_doc and best_doc.score >= config.RETRIEVAL_RELEVANCE_THRESHOLD:
        return best_doc.text, False
    return _FALLBACK_BODY, True


def _build_draft(context: TicketContext, resolution_text: str) -> str:
    intro = _CATEGORY_INTRO.get(context.category, "Thanks for reaching out")
    reference = _format_reference(context)

    lines = ["Hi,", "", f"{intro} regarding {reference}."]

    if context.sentiment == "negative":
        lines.append("I'm sorry for the frustration this has caused you.")

    lines.append(resolution_text)

    lines.append("")
    lines.append("Let us know if there's anything else we can help with.")
    return "\n".join(lines)


def _maybe_refine_with_llm(draft: str, context: TicketContext) -> str:
    """Optionally polishes the templated draft with an LLM if the operator
    configured an API key. The pipeline is fully functional without this, so
    a missing key or network error just keeps the templated draft.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return draft
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite the customer support reply to be warm, concise and professional. "
                    "Keep all facts and any order/product references unchanged.",
                },
                {"role": "user", "content": draft},
            ],
            temperature=0.4,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return draft


class ResponseAgent(Agent):
    """Generates a grounded customer-facing reply from retrieved knowledge."""

    name = "response_agent"

    def run(self, context: TicketContext) -> TicketContext:
        resolution_text, used_fallback = _resolve_body(context)
        context.resolution_text = resolution_text
        context.used_fallback_resolution = used_fallback

        draft = _build_draft(context, resolution_text)
        context.response = _maybe_refine_with_llm(draft, context)

        context.log(
            self.name,
            "generated_response",
            used_retrieval=bool(context.retrieved_docs),
            used_fallback=used_fallback,
            llm_refined=bool(os.getenv("OPENAI_API_KEY")),
        )
        return context

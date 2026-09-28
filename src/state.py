from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RetrievedDoc(BaseModel):
    text: str
    source: str
    score: float


class AgentLogEntry(BaseModel):
    agent: str
    decision: str
    detail: dict[str, Any] = Field(default_factory=dict)


class TicketContext(BaseModel):
    """Shared state object threaded through the agent pipeline."""

    ticket_id: str
    raw_text: str
    cleaned_text: str = ""
    entities: dict[str, str] = Field(default_factory=dict)
    intent: str = ""

    sentiment: str = ""
    sentiment_score: float = 0.0

    category: str = ""
    category_confidence: float = 0.0
    priority: str = ""

    retrieved_docs: list[RetrievedDoc] = Field(default_factory=list)

    response: str = ""
    # Just the resolution content used in `response`, without the greeting /
    # closing boilerplate. Kept separate so the Learning Agent can fold a
    # clean, reusable answer back into the knowledge base instead of an
    # entire formatted reply (which would duplicate greetings/closings if
    # ever retrieved again).
    resolution_text: str = ""
    used_fallback_resolution: bool = False

    escalate: bool = False
    escalation_reason: Optional[str] = None

    agent_logs: list[AgentLogEntry] = Field(default_factory=list)

    def log(self, agent: str, decision: str, **detail: Any) -> None:
        self.agent_logs.append(AgentLogEntry(agent=agent, decision=decision, detail=detail))

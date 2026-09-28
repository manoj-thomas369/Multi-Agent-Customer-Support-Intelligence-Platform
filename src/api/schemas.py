from datetime import datetime

from pydantic import BaseModel


class TicketCreateRequest(BaseModel):
    text: str


class TicketCreateResponse(BaseModel):
    ticket_id: str
    status: str


class ProcessRequest(BaseModel):
    ticket_id: str


class AgentLogItem(BaseModel):
    agent: str
    decision: str
    detail: dict
    created_at: datetime | None = None


class TicketResponse(BaseModel):
    ticket_id: str
    raw_text: str
    cleaned_text: str
    intent: str
    entities: dict[str, str]
    sentiment: str
    sentiment_score: float
    category: str
    category_confidence: float
    priority: str
    response: str
    escalate: bool
    escalation_reason: str | None
    status: str
    logs: list[AgentLogItem] = []

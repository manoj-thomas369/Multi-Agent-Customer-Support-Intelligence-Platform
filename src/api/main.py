import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    AgentLogItem,
    ProcessRequest,
    TicketCreateRequest,
    TicketCreateResponse,
    TicketResponse,
)
from src.db.database import get_session, init_db
from src.db.models import Ticket
from src.orchestrator import TicketOrchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Multi-Agent Customer Support Intelligence Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator = TicketOrchestrator()


def _to_response(ticket: Ticket) -> TicketResponse:
    return TicketResponse(
        ticket_id=ticket.id,
        raw_text=ticket.raw_text,
        cleaned_text=ticket.cleaned_text,
        intent=ticket.intent,
        entities=ticket.entities or {},
        sentiment=ticket.sentiment,
        sentiment_score=ticket.sentiment_score,
        category=ticket.category,
        category_confidence=ticket.category_confidence,
        priority=ticket.priority,
        response=ticket.response,
        escalate=ticket.escalate,
        escalation_reason=ticket.escalation_reason,
        status=ticket.status,
        logs=[
            AgentLogItem(agent=log.agent, decision=log.decision, detail=log.detail, created_at=log.created_at)
            for log in ticket.logs
        ],
    )


@app.post("/ticket", response_model=TicketCreateResponse)
def submit_ticket(payload: TicketCreateRequest) -> TicketCreateResponse:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Ticket text must not be empty")

    ticket_id = str(uuid.uuid4())
    session = get_session()
    try:
        session.add(Ticket(id=ticket_id, raw_text=payload.text, status="received"))
        session.commit()
    finally:
        session.close()

    return TicketCreateResponse(ticket_id=ticket_id, status="received")


@app.post("/process", response_model=TicketResponse)
def process_ticket(payload: ProcessRequest) -> TicketResponse:
    session = get_session()
    try:
        ticket = session.get(Ticket, payload.ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        raw_text = ticket.raw_text
    finally:
        session.close()

    _orchestrator.process(payload.ticket_id, raw_text)

    session = get_session()
    try:
        ticket = session.get(Ticket, payload.ticket_id)
        return _to_response(ticket)
    finally:
        session.close()


@app.get("/ticket/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: str) -> TicketResponse:
    session = get_session()
    try:
        ticket = session.get(Ticket, ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return _to_response(ticket)
    finally:
        session.close()


@app.get("/logs/{ticket_id}", response_model=list[AgentLogItem])
def get_logs(ticket_id: str) -> list[AgentLogItem]:
    session = get_session()
    try:
        ticket = session.get(Ticket, ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return [
            AgentLogItem(agent=log.agent, decision=log.decision, detail=log.detail, created_at=log.created_at)
            for log in ticket.logs
        ]
    finally:
        session.close()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

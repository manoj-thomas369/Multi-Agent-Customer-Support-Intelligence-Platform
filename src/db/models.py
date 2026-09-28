from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    raw_text: Mapped[str] = mapped_column(Text)
    cleaned_text: Mapped[str] = mapped_column(Text, default="")
    intent: Mapped[str] = mapped_column(String(64), default="")
    entities: Mapped[dict] = mapped_column(JSON, default=dict)

    sentiment: Mapped[str] = mapped_column(String(16), default="")
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)

    category: Mapped[str] = mapped_column(String(64), default="")
    category_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    priority: Mapped[str] = mapped_column(String(16), default="")

    response: Mapped[str] = mapped_column(Text, default="")
    escalate: Mapped[bool] = mapped_column(Boolean, default=False)
    escalation_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)

    status: Mapped[str] = mapped_column(String(16), default="received")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    logs: Mapped[list["AgentLog"]] = relationship(back_populates="ticket", cascade="all, delete-orphan")


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(ForeignKey("tickets.id"))
    agent: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(64))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    ticket: Mapped["Ticket"] = relationship(back_populates="logs")

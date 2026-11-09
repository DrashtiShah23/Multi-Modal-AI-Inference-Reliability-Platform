"""Persistence for baselines, evaluated windows, incidents and canary verdicts.

PostgreSQL in production (Cloud SQL on GKE), SQLite for local runs and tests.
Repeated alarms of the same kind fold into one open incident whose
``occurrences`` and ``last_seen_at`` advance, so a sustained problem pages once.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from bellwether.analyzer.baseline import Baseline
from bellwether.analyzer.monitor import WindowResult


def _ts(value: float | None = None) -> datetime:
    return datetime.fromtimestamp(value if value else time.time(), tz=UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON}


class BaselineRow(Base):
    __tablename__ = "baselines"
    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    summary: Mapped[dict[str, Any]]
    blob: Mapped[bytes] = mapped_column(LargeBinary)


class WindowRow(Base):
    __tablename__ = "windows"
    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16))
    n: Mapped[int] = mapped_column(Integer)
    fired: Mapped[bool] = mapped_column(Boolean)
    score: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    summary: Mapped[str] = mapped_column(String(512))
    details: Mapped[dict[str, Any]]
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class IncidentRow(Base):
    __tablename__ = "incidents"
    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(32))
    severity: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    summary: Mapped[str] = mapped_column(String(512))
    details: Mapped[dict[str, Any]]
    occurrences: Mapped[int] = mapped_column(Integer, default=1)
    first_window_id: Mapped[int | None] = mapped_column(ForeignKey("windows.id"), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class VerdictRow(Base):
    __tablename__ = "canary_verdicts"
    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column(String(128), index=True)
    candidate_version: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(16))
    details: Mapped[dict[str, Any]]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def _incident_dict(row: IncidentRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "model": row.model,
        "version": row.version,
        "kind": row.kind,
        "severity": row.severity,
        "status": row.status,
        "summary": row.summary,
        "details": row.details,
        "occurrences": row.occurrences,
        "opened_at": row.opened_at.isoformat(),
        "last_seen_at": row.last_seen_at.isoformat(),
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
    }


class Store:
    def __init__(self, url: str = "sqlite:///bellwether.db"):
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(url, **kwargs)
        Base.metadata.create_all(self.engine)
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> Session:
        return self._session()

    # baselines
    def save_baseline(self, baseline: Baseline) -> int:
        with self.session() as s, s.begin():
            row = BaselineRow(
                model=baseline.model,
                version=baseline.version,
                created_at=_ts(baseline.created_at),
                summary=baseline.summary(),
                blob=baseline.to_bytes(),
            )
            s.add(row)
            s.flush()
            return row.id

    def latest_baseline(self, model: str) -> Baseline | None:
        with self.session() as s:
            row = s.scalars(
                select(BaselineRow).where(BaselineRow.model == model).order_by(BaselineRow.id.desc())
            ).first()
            return Baseline.from_bytes(row.blob) if row else None

    # windows and incidents
    def record_window(self, w: WindowResult) -> tuple[int, int | None]:
        """Persist a window; open or extend an incident if it fired. Returns (window_id, incident_id)."""
        with self.session() as s, s.begin():
            row = WindowRow(
                model=w.model,
                version=w.version,
                kind=w.kind,
                n=w.n,
                fired=w.fired,
                score=w.score,
                threshold=w.threshold,
                summary=w.summary[:512],
                details=w.details,
                started_at=_ts(w.started_at),
                ended_at=_ts(w.ended_at),
            )
            s.add(row)
            s.flush()
            if not w.fired:
                return row.id, None
            incident = s.scalars(
                select(IncidentRow).where(
                    IncidentRow.model == w.model,
                    IncidentRow.version == w.version,
                    IncidentRow.kind == w.incident_kind,
                    IncidentRow.status == "open",
                )
            ).first()
            now = _ts(w.ended_at)
            if incident is None:
                incident = IncidentRow(
                    model=w.model,
                    version=w.version,
                    kind=w.incident_kind or w.kind,
                    severity=w.severity or "warning",
                    status="open",
                    summary=w.summary[:512],
                    details=w.details,
                    occurrences=1,
                    first_window_id=row.id,
                    opened_at=now,
                    last_seen_at=now,
                )
                s.add(incident)
            else:
                incident.occurrences += 1
                incident.last_seen_at = now
                incident.summary = w.summary[:512]
                incident.details = w.details
                if w.severity == "critical":
                    incident.severity = "critical"
            s.flush()
            return row.id, incident.id

    def list_incidents(
        self, model: str | None = None, status: str | None = None, limit: int = 50
    ) -> list[dict]:
        with self.session() as s:
            q = select(IncidentRow).order_by(IncidentRow.id.desc()).limit(limit)
            if model:
                q = q.where(IncidentRow.model == model)
            if status:
                q = q.where(IncidentRow.status == status)
            return [_incident_dict(r) for r in s.scalars(q)]

    def resolve_incident(self, incident_id: int) -> bool:
        with self.session() as s, s.begin():
            row = s.get(IncidentRow, incident_id)
            if row is None or row.status != "open":
                return False
            row.status = "resolved"
            row.resolved_at = _ts()
            return True

    def count_open(self, model: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for inc in self.list_incidents(model=model, status="open", limit=1000):
            out[inc["kind"]] = out.get(inc["kind"], 0) + 1
        return out

    def list_windows(self, model: str, limit: int = 50, fired_only: bool = False) -> list[dict]:
        with self.session() as s:
            q = select(WindowRow).where(WindowRow.model == model).order_by(WindowRow.id.desc()).limit(limit)
            if fired_only:
                q = q.where(WindowRow.fired.is_(True))
            return [
                {
                    "id": r.id,
                    "version": r.version,
                    "kind": r.kind,
                    "n": r.n,
                    "fired": r.fired,
                    "score": r.score,
                    "threshold": r.threshold,
                    "summary": r.summary,
                    "ended_at": r.ended_at.isoformat(),
                }
                for r in s.scalars(q)
            ]

    def save_verdict(self, verdict: dict) -> None:
        with self.session() as s, s.begin():
            s.add(
                VerdictRow(
                    model=verdict["model"],
                    candidate_version=verdict["candidate_version"],
                    decision=verdict["decision"],
                    details=verdict,
                    created_at=_ts(),
                )
            )

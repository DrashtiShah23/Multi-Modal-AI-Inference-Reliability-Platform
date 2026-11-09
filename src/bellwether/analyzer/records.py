"""Gateway samples and the per-item records derived from them.

The gateway forwards one sample per sampled request: the raw KServe request,
the primary exchange and, for mirrored requests, the candidate exchange. The
analyzer explodes each batched request into one ``ItemRecord`` per item and
route so contracts, drift and canary checks all work at item granularity.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from bellwether.analyzer.contracts import Contract


class SampleError(ValueError):
    pass


@dataclass
class Exchange:
    version: str
    status: int
    latency_ms: float
    response: Any = None
    error: str | None = None

    @classmethod
    def parse(cls, raw: Any) -> Exchange:
        if not isinstance(raw, Mapping):
            raise SampleError("exchange must be an object")
        try:
            return cls(
                version=str(raw.get("version") or "unknown"),
                status=int(raw.get("status", 0)),
                latency_ms=float(raw.get("latency_ms", 0.0)),
                response=raw.get("response"),
                error=raw.get("error") or None,
            )
        except (TypeError, ValueError) as exc:
            raise SampleError(f"bad exchange: {exc}") from exc


@dataclass
class GatewaySample:
    model: str
    primary: Exchange
    candidate: Exchange | None = None
    request: Any = None
    request_error: str | None = None
    trace_id: str = ""
    traceparent: str = ""
    received_at: float = field(default_factory=time.time)


def _parse_time(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time()


def parse_sample(raw: Any) -> GatewaySample:
    if not isinstance(raw, Mapping):
        raise SampleError("sample must be an object")
    if "primary" not in raw:
        raise SampleError("sample has no primary exchange")
    candidate = raw.get("candidate")
    return GatewaySample(
        model=str(raw.get("model") or "unknown"),
        primary=Exchange.parse(raw["primary"]),
        candidate=Exchange.parse(candidate) if candidate else None,
        request=raw.get("request"),
        request_error=raw.get("request_error") or None,
        trace_id=str(raw.get("trace_id") or ""),
        traceparent=str(raw.get("traceparent") or ""),
        received_at=_parse_time(raw.get("received_at")),
    )


@dataclass
class ItemRecord:
    model: str
    version: str
    route: str
    status: int
    latency_ms: float
    image: np.ndarray | None
    text: str | None
    parameters: dict[str, Any]
    image_embedding: np.ndarray | None
    text_embedding: np.ndarray | None
    similarity: float | None
    violations: tuple[str, ...]
    trace_id: str = ""
    received_at: float = 0.0

    @property
    def valid(self) -> bool:
        return (
            not self.violations
            and self.status == 200
            and self.image is not None
            and self.text is not None
            and self.image_embedding is not None
            and self.text_embedding is not None
            and self.similarity is not None
        )


def _route_items(
    sample: GatewaySample, req, exchange: Exchange, route: str, contract: Contract
) -> list[ItemRecord]:
    resp = contract.check_response(exchange.response, exchange.status, req.n)
    out = resp.outputs or {}
    items = []
    for i in range(req.n):
        items.append(
            ItemRecord(
                model=sample.model,
                version=exchange.version,
                route=route,
                status=exchange.status,
                latency_ms=exchange.latency_ms,
                image=req.images[i] if req.images is not None else None,
                text=req.texts[i],
                parameters=req.parameters,
                image_embedding=out["image_embedding"][i] if "image_embedding" in out else None,
                text_embedding=out["text_embedding"][i] if "text_embedding" in out else None,
                similarity=float(out["similarity"][i]) if "similarity" in out else None,
                violations=tuple(dict.fromkeys(req.violations[i] + resp.violations[i])),
                trace_id=sample.trace_id,
                received_at=sample.received_at,
            )
        )
    return items


def explode(sample: GatewaySample, contract: Contract) -> tuple[list[ItemRecord], list[ItemRecord]]:
    """Return (primary items, candidate items) for one gateway sample."""
    req = contract.check_request(None if sample.request_error else sample.request)
    primary = _route_items(sample, req, sample.primary, "primary", contract)
    candidate = _route_items(sample, req, sample.candidate, "candidate", contract) if sample.candidate else []
    return primary, candidate

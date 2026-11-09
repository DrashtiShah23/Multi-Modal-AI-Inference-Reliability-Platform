"""gRPC front end for the analyzer (bellwether.v1.AnalyzerService)."""

from __future__ import annotations

import json
from concurrent import futures
from datetime import datetime
from typing import Any

import grpc
from google.protobuf import timestamp_pb2

from bellwether.analyzer.service import AnalyzerService
from bellwether.v1 import analyzer_pb2 as pb
from bellwether.v1 import analyzer_pb2_grpc as pb_grpc

_DECISIONS = {
    "HOLD": pb.CanaryVerdict.DECISION_HOLD,
    "PROMOTE": pb.CanaryVerdict.DECISION_PROMOTE,
    "ROLLBACK": pb.CanaryVerdict.DECISION_ROLLBACK,
}


def _json_or_none(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _exchange(ex: pb.Exchange) -> dict[str, Any]:
    return {
        "version": ex.version,
        "status": ex.status,
        "latency_ms": ex.latency_ms,
        "response": _json_or_none(ex.response_json),
        "error": ex.error or None,
    }


def sample_to_dict(s: pb.Sample) -> dict[str, Any]:
    request = _json_or_none(s.request_json)
    out: dict[str, Any] = {
        "trace_id": s.trace_id,
        "traceparent": s.traceparent,
        "model": s.model,
        "received_at": s.received_at.ToDatetime().timestamp() if s.HasField("received_at") else None,
        "request": request,
        "request_error": s.request_error
        or (None if request is not None or not s.request_json else "invalid_json"),
        "primary": _exchange(s.primary),
    }
    if s.HasField("candidate"):
        out["candidate"] = _exchange(s.candidate)
    return out


def _ts(iso: str) -> timestamp_pb2.Timestamp:
    t = timestamp_pb2.Timestamp()
    t.FromDatetime(datetime.fromisoformat(iso))
    return t


class AnalyzerServicer(pb_grpc.AnalyzerServiceServicer):
    def __init__(self, service: AnalyzerService):
        self.service = service

    def IngestSamples(self, request_iterator, context):  # noqa: N802 - gRPC naming
        totals = {"accepted": 0, "rejected": 0, "items": 0, "windows": 0}
        for req in request_iterator:
            result = self.service.ingest([sample_to_dict(s) for s in req.samples])
            totals["accepted"] += result["accepted"]
            totals["rejected"] += result["rejected"]
            totals["items"] += result["items"]
            totals["windows"] += len(result["windows"])
        return pb.IngestSamplesResponse(
            accepted=totals["accepted"],
            rejected=totals["rejected"],
            items=totals["items"],
            windows_closed=totals["windows"],
        )

    def GetCanaryVerdict(self, request, context):  # noqa: N802
        if not request.model or not request.candidate_version:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model and candidate_version are required")
        v = self.service.verdict(request.model, request.candidate_version)
        return pb.CanaryVerdict(
            model=v.model,
            candidate_version=v.candidate_version,
            decision=_DECISIONS[v.decision],
            checks=[
                pb.CanaryCheck(
                    name=c.name,
                    value=c.value,
                    threshold=c.threshold,
                    passed=c.passed,
                    sufficient_data=c.sufficient_data,
                    samples=c.samples,
                    detail=c.detail,
                )
                for c in v.checks
            ],
        )

    def ListIncidents(self, request, context):  # noqa: N802
        rows = self.service.store.list_incidents(
            model=request.model or None, status=request.status or None, limit=request.limit or 50
        )
        return pb.ListIncidentsResponse(
            incidents=[
                pb.Incident(
                    id=r["id"],
                    model=r["model"],
                    version=r["version"],
                    kind=r["kind"],
                    severity=r["severity"],
                    status=r["status"],
                    summary=r["summary"],
                    occurrences=r["occurrences"],
                    opened_at=_ts(r["opened_at"]),
                    last_seen_at=_ts(r["last_seen_at"]),
                    details_json=json.dumps(r["details"]),
                )
                for r in rows
            ]
        )


def serve(service: AnalyzerService, address: str = "[::]:50051", workers: int = 8) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)],
    )
    pb_grpc.add_AnalyzerServiceServicer_to_server(AnalyzerServicer(service), server)
    server.add_insecure_port(address)
    server.start()
    return server

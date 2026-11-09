"""REST API. Argo Rollouts' web metric provider polls the canary verdict endpoint."""

from __future__ import annotations

import gzip
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from bellwether.analyzer.service import AnalyzerService

MAX_BODY_BYTES = 64 * 1024 * 1024


def create_app(service: AnalyzerService) -> FastAPI:
    app = FastAPI(title="bellwether-analyzer", version="0.1.0")

    @app.post("/v1/samples:batch", status_code=202)
    async def ingest(request: Request) -> Any:
        raw = await request.body()
        if request.headers.get("content-encoding", "").lower() == "gzip":
            try:
                raw = gzip.decompress(raw)
            except OSError as exc:
                raise HTTPException(400, f"bad gzip body: {exc}") from exc
        if len(raw) > MAX_BODY_BYTES:
            raise HTTPException(413, "batch too large")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"invalid JSON: {exc.msg}") from exc
        samples = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(samples, list):
            raise HTTPException(400, 'expected a list of samples or {"samples": [...]}')
        result = await run_in_threadpool(service.ingest, samples)
        return JSONResponse(result, status_code=202)

    @app.get("/v1/models/{model}/status")
    def status(model: str) -> dict:
        return service.status(model)

    @app.get("/v1/models/{model}/incidents")
    def incidents(model: str, status: str | None = "open", limit: int = 50) -> dict:
        return {"incidents": service.store.list_incidents(model=model, status=status or None, limit=limit)}

    @app.post("/v1/incidents/{incident_id}:resolve")
    def resolve(incident_id: int) -> dict:
        if not service.store.resolve_incident(incident_id):
            raise HTTPException(404, "no open incident with that id")
        return {"resolved": incident_id}

    @app.get("/v1/models/{model}/windows")
    def windows(model: str, limit: int = 50, fired_only: bool = False) -> dict:
        return {"windows": service.store.list_windows(model, limit=limit, fired_only=fired_only)}

    @app.get("/v1/models/{model}/canary/{candidate}/verdict")
    def verdict(model: str, candidate: str) -> dict:
        return service.verdict(model, candidate).as_dict()

    @app.post("/v1/models/{model}/canary/{candidate}:reset")
    def reset_canary(model: str, candidate: str) -> dict:
        service.reset_canary(model, candidate)
        return {"reset": candidate}

    @app.post("/v1/models/{model}/baseline:relearn")
    def relearn(model: str) -> dict:
        service.relearn_baseline(model)
        return {"learning": True, "items_needed": service.settings.baseline_items}

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/readyz")
    def readyz() -> Response:
        try:
            service.store.list_incidents(limit=1)
        except Exception as exc:  # database unreachable
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)
        return JSONResponse({"ok": True})

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(service.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app

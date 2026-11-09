"""Local stand-in for Triton that speaks the same KServe v2 HTTP protocol.

It serves the ensemble from the same model repository Triton loads, so the
gateway and analyzer can be developed and tested without GPUs. Two knobs,
``inject_latency_ms`` and ``inject_error_rate``, emulate a slow or failing
release for canary drills.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

from bellwether import EMBED_DIM, IMAGE_SIZE, MODEL_NAME
from bellwether.kserve import TensorError, encode_tensor
from bellwether.models.runtime import MultimodalRuntime, parse_infer_request


def create_app(
    repository: Path | str,
    version: str,
    *,
    model_name: str = MODEL_NAME,
    inject_latency_ms: float = 0.0,
    inject_error_rate: float = 0.0,
    seed: int | None = None,
) -> FastAPI:
    runtime = MultimodalRuntime(repository)
    rng = random.Random(seed)
    registry = CollectorRegistry()
    requests = Counter(
        "modelserver_requests_total", "Inference requests", ["model", "version", "code"], registry=registry
    )
    latency = Histogram(
        "modelserver_infer_seconds",
        "Model execution time",
        ["model", "version"],
        buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        registry=registry,
    )

    app = FastAPI(title="bellwether-modelserver", version=version, docs_url=None, redoc_url=None)
    metadata = {
        "name": model_name,
        "versions": [version],
        "platform": "ensemble",
        "inputs": [
            {"name": "image", "datatype": "UINT8", "shape": [-1, IMAGE_SIZE, IMAGE_SIZE, 3]},
            {"name": "text", "datatype": "BYTES", "shape": [-1]},
        ],
        "outputs": [
            {"name": "image_embedding", "datatype": "FP32", "shape": [-1, EMBED_DIM]},
            {"name": "text_embedding", "datatype": "FP32", "shape": [-1, EMBED_DIM]},
            {"name": "similarity", "datatype": "FP32", "shape": [-1]},
        ],
    }

    def error(status: int, message: str) -> JSONResponse:
        requests.labels(model_name, version, str(status)).inc()
        return JSONResponse({"error": message}, status_code=status)

    @app.get("/v2")
    def server_metadata() -> dict[str, Any]:
        return {"name": "bellwether-modelserver", "version": version, "extensions": []}

    @app.get("/v2/health/live")
    def live() -> Response:
        return Response(status_code=200)

    @app.get("/v2/health/ready")
    def ready() -> Response:
        return Response(status_code=200)

    @app.get("/v2/models/{name}")
    def model_metadata(name: str) -> Any:
        if name != model_name:
            return JSONResponse({"error": f"unknown model {name}"}, status_code=404)
        return metadata

    @app.get("/v2/models/{name}/ready")
    def model_ready(name: str) -> Response:
        return Response(status_code=200 if name == model_name else 404)

    @app.post("/v2/models/{name}/infer")
    async def infer(name: str, request: Request) -> Any:
        if name != model_name:
            return error(404, f"unknown model {name}")
        raw = await request.body()
        try:
            body = json.loads(raw)
            parsed = parse_infer_request(body)
        except json.JSONDecodeError as exc:
            return error(400, f"invalid JSON: {exc.msg}")
        except TensorError as exc:
            return error(400, str(exc))
        if inject_latency_ms > 0:
            await asyncio.sleep(inject_latency_ms / 1000.0)
        if inject_error_rate > 0 and rng.random() < inject_error_rate:
            return error(500, "injected failure")
        start = time.perf_counter()
        outputs = await run_in_threadpool(runtime.infer, parsed.images, parsed.texts)
        latency.labels(model_name, version).observe(time.perf_counter() - start)
        requests.labels(model_name, version, "200").inc()
        payload: dict[str, Any] = {
            "model_name": model_name,
            "model_version": version,
            "outputs": [encode_tensor(k, v) for k, v in outputs.items()],
        }
        if parsed.request_id:
            payload["id"] = parsed.request_id
        return payload

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app

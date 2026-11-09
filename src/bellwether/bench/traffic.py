"""Build gateway-format samples from in-process inference.

The benchmark and the baseline CLI run the model in-process for speed but feed
the analyzer exactly what the gateway would send, so they exercise the same
contract and drift code paths. Tensors are passed as flat numpy arrays instead
of JSON lists, which ``bellwether.kserve.decode_tensor`` accepts.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bellwether import MODEL_NAME
from bellwether.analyzer.contracts import Contract
from bellwether.analyzer.records import ItemRecord, explode, parse_sample
from bellwether.synth import Batch


def tensor(name: str, value: np.ndarray | Sequence[str], datatype: str | None = None) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        dt = datatype or {"uint8": "UINT8", "float32": "FP32", "int64": "INT64"}[value.dtype.name]
        return {"name": name, "shape": list(value.shape), "datatype": dt, "data": value.reshape(-1)}
    return {"name": name, "shape": [len(value)], "datatype": datatype or "BYTES", "data": list(value)}


def request_body(
    images: np.ndarray, texts: Sequence[str], parameters: Mapping[str, Any] | None = None
) -> dict:
    body: dict[str, Any] = {"inputs": [tensor("image", images), tensor("text", texts)]}
    if parameters is not None:
        body["parameters"] = dict(parameters)
    return body


def response_body(outputs: Mapping[str, np.ndarray], version: str, model: str = MODEL_NAME) -> dict:
    return {
        "model_name": model,
        "model_version": version,
        "outputs": [tensor(k, np.ascontiguousarray(v, dtype=np.float32)) for k, v in outputs.items()],
    }


def exchange(
    version: str, status: int = 200, latency_ms: float = 5.0, response: Any = None, error: str | None = None
) -> dict[str, Any]:
    return {
        "version": version,
        "status": status,
        "latency_ms": latency_ms,
        "response": response,
        "error": error,
    }


def sample(
    request: Any,
    primary: dict[str, Any],
    candidate: dict[str, Any] | None = None,
    *,
    model: str = MODEL_NAME,
    received_at: float | None = None,
    request_error: str | None = None,
) -> dict:
    s: dict[str, Any] = {
        "trace_id": uuid.uuid4().hex,
        "model": model,
        "request": request,
        "request_error": request_error,
        "primary": primary,
    }
    if received_at is not None:
        s["received_at"] = received_at
    if candidate is not None:
        s["candidate"] = candidate
    return s


def slice_outputs(outputs: Mapping[str, np.ndarray], i: int, j: int) -> dict[str, np.ndarray]:
    return {k: v[i:j] for k, v in outputs.items()}


def samples_from_batch(
    batch: Batch,
    outputs: Mapping[str, np.ndarray],
    *,
    version: str,
    model: str = MODEL_NAME,
    latency_ms: float = 5.0,
) -> list[dict]:
    """One single-item request per element of ``batch``."""
    out = []
    for i in range(len(batch)):
        req = request_body(batch.images[i : i + 1], batch.captions[i : i + 1], batch.metadata[i])
        resp = response_body(slice_outputs(outputs, i, i + 1), version, model)
        out.append(sample(req, exchange(version, 200, latency_ms, resp), model=model))
    return out


def items_from_batch(
    batch: Batch, outputs: Mapping[str, np.ndarray], *, model: str, version: str, contract: Contract
) -> list[ItemRecord]:
    items: list[ItemRecord] = []
    for s in samples_from_batch(batch, outputs, version=version, model=model):
        primary, _ = explode(parse_sample(s), contract)
        items.extend(primary)
    return items

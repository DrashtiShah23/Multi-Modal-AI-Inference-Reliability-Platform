"""Helpers for the KServe v2 / Open Inference Protocol JSON encoding.

Triton, KServe and the bundled model server all speak this protocol, so the
gateway, analyzer and benchmarks share these helpers instead of each parsing
tensors their own way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DATATYPES: dict[str, np.dtype | None] = {
    "BOOL": np.dtype(np.bool_),
    "UINT8": np.dtype(np.uint8),
    "INT32": np.dtype(np.int32),
    "INT64": np.dtype(np.int64),
    "FP16": np.dtype(np.float16),
    "FP32": np.dtype(np.float32),
    "FP64": np.dtype(np.float64),
    "BYTES": None,
}

_NP_TO_KSERVE = {v: k for k, v in DATATYPES.items() if v is not None}


class TensorError(ValueError):
    """Raised when a tensor in a request or response is malformed."""


def kserve_datatype(arr: np.ndarray) -> str:
    try:
        return _NP_TO_KSERVE[arr.dtype]
    except KeyError as exc:  # pragma: no cover - defensive
        raise TensorError(f"unsupported numpy dtype {arr.dtype}") from exc


def encode_tensor(name: str, value: np.ndarray | Sequence[str]) -> dict[str, Any]:
    """Encode a numpy array (or list of strings for BYTES) as a KServe v2 tensor."""
    if isinstance(value, np.ndarray) and value.dtype.kind not in ("U", "S", "O"):
        return {
            "name": name,
            "shape": list(value.shape),
            "datatype": kserve_datatype(value),
            "data": value.reshape(-1).tolist(),
        }
    items = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in value]
    return {"name": name, "shape": [len(items)], "datatype": "BYTES", "data": items}


def decode_tensor(tensor: Mapping[str, Any]) -> np.ndarray | list[str]:
    """Decode one KServe v2 tensor, validating that shape and data agree."""
    try:
        name = tensor["name"]
        shape = [int(d) for d in tensor["shape"]]
        datatype = tensor["datatype"]
        data = tensor["data"]
    except (KeyError, TypeError, ValueError) as exc:
        raise TensorError(f"tensor is missing name/shape/datatype/data: {exc}") from exc
    if datatype not in DATATYPES:
        raise TensorError(f"{name}: unsupported datatype {datatype!r}")
    if any(d < 0 for d in shape):
        raise TensorError(f"{name}: negative dimension in shape {shape}")
    expected = int(np.prod(shape)) if shape else 1
    # JSON payloads carry flat lists; in-process callers (benchmarks) may pass flat arrays.
    if not (isinstance(data, list) or (isinstance(data, np.ndarray) and data.ndim == 1)):
        raise TensorError(f"{name}: data must be a flat list")
    if len(data) != expected:
        raise TensorError(f"{name}: shape {shape} needs {expected} values, got {len(data)}")
    if datatype == "BYTES":
        if isinstance(data, np.ndarray) or not all(isinstance(v, str) for v in data):
            raise TensorError(f"{name}: BYTES data must be strings")
        return list(data)
    dtype = DATATYPES[datatype]
    try:
        arr = np.asarray(data, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TensorError(f"{name}: data is not valid {datatype}: {exc}") from exc
    return arr.reshape(shape)


def build_infer_request(
    images: np.ndarray,
    texts: Sequence[str],
    *,
    request_id: str | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an infer request for the bundled multimodal model."""
    body: dict[str, Any] = {
        "inputs": [encode_tensor("image", images), encode_tensor("text", list(texts))],
    }
    if request_id:
        body["id"] = request_id
    if parameters:
        body["parameters"] = dict(parameters)
    return body


def tensors_by_name(tensors: Sequence[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    if not isinstance(tensors, list):
        raise TensorError("expected a list of tensors")
    out: dict[str, Mapping[str, Any]] = {}
    for t in tensors:
        if not isinstance(t, Mapping) or "name" not in t:
            raise TensorError("tensor without a name")
        out[str(t["name"])] = t
    return out


def decode_outputs(response: Mapping[str, Any]) -> dict[str, np.ndarray | list[str]]:
    return {name: decode_tensor(t) for name, t in tensors_by_name(response.get("outputs")).items()}

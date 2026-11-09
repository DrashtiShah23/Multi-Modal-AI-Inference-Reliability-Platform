"""In-process execution of the multimodal ensemble with ONNX Runtime.

Mirrors the Triton ensemble (tokenizer -> text_encoder, image_encoder,
similarity) so the local server, the benchmark and Triton produce the same
outputs from the same model repository.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from bellwether import IMAGE_SIZE
from bellwether.kserve import TensorError, decode_tensor, tensors_by_name
from bellwether.models import tokenizer


@dataclass
class ParsedRequest:
    images: np.ndarray
    texts: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None


def parse_infer_request(body: Mapping[str, Any]) -> ParsedRequest:
    """Validate a KServe v2 request for the multimodal model; raise TensorError if malformed."""
    if not isinstance(body, Mapping):
        raise TensorError("request body must be a JSON object")
    tensors = tensors_by_name(body.get("inputs"))
    missing = {"image", "text"} - tensors.keys()
    if missing:
        raise TensorError(f"missing input tensor(s): {sorted(missing)}")
    if tensors["image"].get("datatype") != "UINT8":
        raise TensorError(f"image: expected datatype UINT8, got {tensors['image'].get('datatype')!r}")
    images = decode_tensor(tensors["image"])
    texts = decode_tensor(tensors["text"])
    if (
        not isinstance(images, np.ndarray)
        or images.ndim != 4
        or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3)
    ):
        shape = getattr(images, "shape", None)
        raise TensorError(f"image: expected shape [N,{IMAGE_SIZE},{IMAGE_SIZE},3], got {list(shape or [])}")
    if not isinstance(texts, list):
        raise TensorError("text: expected datatype BYTES")
    if len(texts) != images.shape[0]:
        raise TensorError(f"batch mismatch: {images.shape[0]} images, {len(texts)} texts")
    params = body.get("parameters") or {}
    if not isinstance(params, Mapping):
        raise TensorError("parameters must be an object")
    return ParsedRequest(images=images, texts=texts, parameters=dict(params), request_id=body.get("id"))


class MultimodalRuntime:
    def __init__(self, repository: Path | str, threads: int | None = None):
        repo = Path(repository)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or int(os.environ.get("BELLWETHER_ORT_THREADS", "1"))
        opts.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]

        def load(name: str) -> ort.InferenceSession:
            # newest numeric version directory, as Triton's default version policy would pick
            versions = sorted(
                (int(p.name) for p in (repo / name).iterdir() if p.name.isdigit()), reverse=True
            )
            if not versions:
                raise FileNotFoundError(f"no versions of {name} in {repo}")
            path = repo / name / str(versions[0]) / "model.onnx"
            return ort.InferenceSession(str(path), sess_options=opts, providers=providers)

        self.image_encoder = load("image_encoder")
        self.text_encoder = load("text_encoder")
        self.similarity = load("similarity")

    def infer(self, images: np.ndarray, texts: Sequence[str]) -> dict[str, np.ndarray]:
        if len(texts) == 0:
            raise TensorError("empty batch")
        ie = self.image_encoder.run(
            ["image_embedding"], {"image": np.ascontiguousarray(images, dtype=np.uint8)}
        )[0]
        te = self.text_encoder.run(["text_embedding"], {"token_ids": tokenizer.encode(texts)})[0]
        sim = self.similarity.run(["similarity"], {"image_embedding": ie, "text_embedding": te})[0]
        return {"image_embedding": ie, "text_embedding": te, "similarity": sim.reshape(-1)}

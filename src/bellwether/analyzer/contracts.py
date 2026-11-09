"""Per-item data contracts for requests and responses.

Checks never raise on bad data: they return violation codes such as
``inputs.image.shape`` or ``outputs.text_embedding.norm`` so a malformed
request becomes a measurable signal instead of a crash in the analyzer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bellwether.kserve import TensorError, decode_outputs, decode_tensor, tensors_by_name


@dataclass(frozen=True)
class ParamSpec:
    required: bool = False
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputSpec:
    dim: int | None = None
    l2_norm: tuple[float, float] | None = None
    value_range: tuple[float, float] | None = None


@dataclass
class RequestCheck:
    n: int
    images: np.ndarray | None
    texts: list[str | None]
    parameters: dict[str, Any]
    violations: list[list[str]]


@dataclass
class ResponseCheck:
    outputs: dict[str, np.ndarray] | None
    violations: list[list[str]]


def status_code_violation(status: int) -> str:
    if status <= 0:
        return "response.unavailable"
    if status >= 500:
        return "response.status_5xx"
    if status >= 400:
        return "response.status_4xx"
    return "response.status_other"


@dataclass
class Contract:
    model: str
    max_violation_rate: float = 0.02
    image_datatype: str = "UINT8"
    image_shape: tuple[int, ...] = (64, 64, 3)
    text_datatype: str = "BYTES"
    min_chars: int = 1
    max_chars: int = 200
    parameters: dict[str, ParamSpec] = field(default_factory=dict)
    outputs: dict[str, OutputSpec] = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Contract:
        inputs = raw.get("inputs", {})
        image, text = inputs.get("image", {}), inputs.get("text", {})
        params = {
            name: ParamSpec(
                required=bool(spec.get("required", False)),
                values=tuple(str(v) for v in spec.get("values", [])),
            )
            for name, spec in (raw.get("parameters") or {}).items()
        }
        outputs = {}
        for name, spec in (raw.get("outputs") or {}).items():
            outputs[name] = OutputSpec(
                dim=spec.get("dim"),
                l2_norm=tuple(spec["l2_norm"]) if "l2_norm" in spec else None,
                value_range=tuple(spec["range"]) if "range" in spec else None,
            )
        return cls(
            model=str(raw["model"]),
            max_violation_rate=float(raw.get("max_violation_rate", 0.02)),
            image_datatype=image.get("datatype", "UINT8"),
            image_shape=tuple(image.get("shape", (64, 64, 3))),
            text_datatype=text.get("datatype", "BYTES"),
            min_chars=int(text.get("min_chars", 1)),
            max_chars=int(text.get("max_chars", 200)),
            parameters=params,
            outputs=outputs,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> Contract:
        if path is None:
            text = resources.files("bellwether").joinpath("contracts/multimodal.yaml").read_text()
        else:
            text = Path(path).read_text()
        return cls.from_dict(yaml.safe_load(text))

    # ------------------------------------------------------------------ checks
    def check_request(self, body: Any) -> RequestCheck:
        if not isinstance(body, Mapping):
            return RequestCheck(1, None, [None], {}, [["request.invalid_json"]])

        shared: list[str] = []
        params = body.get("parameters") or {}
        if not isinstance(params, Mapping):
            shared.append("parameters.invalid")
            params = {}
        for name, spec in self.parameters.items():
            value = params.get(name)
            if value is None:
                if spec.required:
                    shared.append(f"parameters.{name}.missing")
            elif spec.values and str(value) not in spec.values:
                shared.append(f"parameters.{name}.unknown_value")

        try:
            tensors = tensors_by_name(body.get("inputs"))
        except TensorError:
            return RequestCheck(1, None, [None], dict(params), [[*shared, "inputs.invalid"]])

        images, n_img, image_codes = self._check_image(tensors.get("image"))
        texts, text_codes = self._check_text(tensors.get("text"))
        n_txt = len(texts) if texts is not None else None
        n = n_img or n_txt or 1
        if n_img and n_txt and n_img != n_txt:
            shared.append("inputs.batch_mismatch")

        violations: list[list[str]] = []
        for i in range(n):
            codes = [*shared, *image_codes, *text_codes]
            if texts is not None and i < len(texts):
                s = texts[i]
                if len(s) < self.min_chars:
                    codes.append("inputs.text.too_short")
                elif len(s) > self.max_chars:
                    codes.append("inputs.text.too_long")
                if "�" in s:
                    codes.append("inputs.text.encoding")
            violations.append(codes)
        item_texts = [texts[i] if texts is not None and i < len(texts) else None for i in range(n)]
        item_images = images if images is not None and images.shape[0] == n else None
        return RequestCheck(n, item_images, item_texts, dict(params), violations)

    def _check_image(
        self, tensor: Mapping[str, Any] | None
    ) -> tuple[np.ndarray | None, int | None, list[str]]:
        if tensor is None:
            return None, None, ["inputs.image.missing"]
        codes: list[str] = []
        if tensor.get("datatype") != self.image_datatype:
            codes.append("inputs.image.datatype")
        try:
            arr = decode_tensor(tensor)
        except TensorError:
            return None, None, [*codes, "inputs.image.malformed"]
        if not isinstance(arr, np.ndarray):
            return None, len(arr), [*codes, "inputs.image.datatype"]
        n = int(arr.shape[0]) if arr.ndim >= 1 else None
        if arr.ndim != 1 + len(self.image_shape) or tuple(arr.shape[1:]) != self.image_shape:
            codes.append("inputs.image.shape")
        if codes:
            return None, n, codes
        return arr.astype(np.uint8, copy=False), n, codes

    def _check_text(self, tensor: Mapping[str, Any] | None) -> tuple[list[str] | None, list[str]]:
        if tensor is None:
            return None, ["inputs.text.missing"]
        if tensor.get("datatype") != self.text_datatype:
            return None, ["inputs.text.datatype"]
        try:
            values = decode_tensor(tensor)
        except TensorError:
            return None, ["inputs.text.malformed"]
        return (values, []) if isinstance(values, list) else (None, ["inputs.text.datatype"])

    def check_response(self, body: Any, status: int, n: int) -> ResponseCheck:
        if status != 200:
            return ResponseCheck(None, [[status_code_violation(status)] for _ in range(n)])
        if not isinstance(body, Mapping):
            return ResponseCheck(None, [["response.invalid"] for _ in range(n)])
        try:
            raw = decode_outputs(body)
        except TensorError:
            return ResponseCheck(None, [["response.invalid"] for _ in range(n)])

        violations: list[list[str]] = [[] for _ in range(n)]
        outputs: dict[str, np.ndarray] = {}
        complete = True
        for name, spec in self.outputs.items():
            value = raw.get(name)
            if value is None or not isinstance(value, np.ndarray):
                for v in violations:
                    v.append(f"outputs.{name}.missing")
                complete = False
                continue
            arr = value.astype(np.float32, copy=False)
            arr = arr.reshape(n, -1) if arr.size % max(n, 1) == 0 and arr.size else arr
            if (
                arr.ndim != 2
                or arr.shape[0] != n
                or (spec.dim is not None and arr.shape[1] != spec.dim)
                or (spec.dim is None and arr.shape[1] != 1)
            ):
                for v in violations:
                    v.append(f"outputs.{name}.shape")
                complete = False
                continue
            finite = np.isfinite(arr).all(axis=1)
            for i in np.flatnonzero(~finite):
                violations[i].append(f"outputs.{name}.non_finite")
            if spec.l2_norm is not None:
                norms = np.linalg.norm(np.nan_to_num(arr), axis=1)
                bad = finite & ((norms < spec.l2_norm[0]) | (norms > spec.l2_norm[1]))
                for i in np.flatnonzero(bad):
                    violations[i].append(f"outputs.{name}.norm")
            if spec.value_range is not None:
                flat = arr[:, 0]
                bad = finite & ((flat < spec.value_range[0]) | (flat > spec.value_range[1]))
                for i in np.flatnonzero(bad):
                    violations[i].append(f"outputs.{name}.range")
            outputs[name] = arr if spec.dim is not None else arr[:, 0]
        return ResponseCheck(outputs if complete else None, violations)

"""Fault injectors for the reliability benchmark.

Each fault models a failure seen in real multi-modal serving: a client release
that flips color channels, a resize bug, a camera stuck on black frames, a
locale launch, a malformed SDK, a post-processing bug or one replica running a
bad build. Magnitudes and the affected share of traffic are drawn from fixed
ranges per trial, so some trials are deliberately subtle.

A fault returns gateway-format samples for one chunk of traffic; the
benchmark feeds them through the same contract and drift code the service uses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bellwether.bench import traffic
from bellwether.kserve import TensorError
from bellwether.models.runtime import MultimodalRuntime, parse_infer_request
from bellwether.synth import Batch, generate

SPANISH = {
    "a": "un",
    "on": "sobre",
    "background": "fondo",
    "red": "rojo",
    "green": "verde",
    "blue": "azul",
    "yellow": "amarillo",
    "purple": "morado",
    "orange": "naranja",
    "white": "blanco",
    "black": "negro",
    "circle": "círculo",
    "square": "cuadrado",
    "triangle": "triángulo",
    "ring": "anillo",
    "stripes": "rayas",
    "cross": "cruz",
    "photo": "foto",
    "of": "de",
    "small": "pequeño",
    "medium": "mediano",
    "large": "grande",
    "there": "hay",
    "is": "",
    "in": "en",
    "scene": "escena",
    "with": "con",
    "near": "cerca",
    "the": "el",
    "light": "claro",
    "dark": "oscuro",
    "sky": "cielo",
    "grass": "césped",
    "center": "centro",
    "left": "izquierda",
    "right": "derecha",
    "top": "arriba",
    "bottom": "abajo",
}


@dataclass
class Runtimes:
    primary: MultimodalRuntime
    regressed: MultimodalRuntime


@dataclass
class Fault:
    name: str
    description: str
    expected: str  # contract | input_drift | output_drift
    apply: Callable[[Batch, np.random.Generator, Runtimes], list[dict]]


def _infer_samples(batch: Batch, rt: MultimodalRuntime, version: str = "v1") -> list[dict]:
    out = rt.infer(batch.images, batch.captions)
    return traffic.samples_from_batch(batch, out, version=version)


def _pick(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    frac = rng.uniform(lo, hi)
    return rng.random(n) < frac


def exposure_shift(batch, rng, rts):
    mask = _pick(rng, len(batch), 0.25, 1.0)
    factor = 1.0 - rng.uniform(0.10, 0.40)
    imgs = batch.images.astype(np.float32)
    imgs[mask] *= factor
    batch.images = np.clip(imgs, 0, 255).astype(np.uint8)
    return _infer_samples(batch, rts.primary)


def sensor_noise(batch, rng, rts):
    mask = _pick(rng, len(batch), 0.25, 1.0)
    sigma = rng.uniform(4, 20)
    imgs = batch.images.astype(np.float32)
    imgs[mask] += rng.normal(0, sigma, size=imgs[mask].shape)
    batch.images = np.clip(imgs, 0, 255).astype(np.uint8)
    return _infer_samples(batch, rts.primary)


def resize_blur(batch, rng, rts):
    mask = _pick(rng, len(batch), 0.25, 1.0)
    k = int(rng.choice([2, 4]))
    small = batch.images[mask].reshape(-1, 64 // k, k, 64 // k, k, 3).mean(axis=(2, 4))
    batch.images[mask] = np.repeat(np.repeat(small, k, axis=1), k, axis=2).astype(np.uint8)
    return _infer_samples(batch, rts.primary)


def channel_swap(batch, rng, rts):
    mask = _pick(rng, len(batch), 0.10, 0.60)
    batch.images[mask] = batch.images[mask][..., ::-1]
    return _infer_samples(batch, rts.primary)


def frozen_frames(batch, rng, rts):
    mask = _pick(rng, len(batch), 0.05, 0.30)
    batch.images[mask] = rng.choice([0, 128, 255])
    return _infer_samples(batch, rts.primary)


def locale_launch(batch, rng, rts):
    mask = _pick(rng, len(batch), 0.10, 0.50)
    for i in np.flatnonzero(mask):
        words = [SPANISH.get(w, w) for w in batch.captions[i].split()]
        batch.captions[i] = " ".join(w for w in words if w)
        batch.metadata[i]["locale"] = "es-ES"
    return _infer_samples(batch, rts.primary)


def caption_template(batch, rng, rts):
    mask = _pick(rng, len(batch), 0.20, 1.0)
    for i in np.flatnonzero(mask):
        batch.captions[i] = (
            f"user uploaded image 2026 11 05 caption: {batch.captions[i]} (auto generated, v2 template)"
        )
    return _infer_samples(batch, rts.primary)


def client_mix(batch, rng, rts):
    shift = rng.uniform(0.15, 0.40)  # share of traffic moving to android from a new app release
    for m in batch.metadata:
        if m["client"] != "android" and rng.random() < shift:
            m["client"] = "android"
    return _infer_samples(batch, rts.primary)


def schema_break(batch, rng, rts):
    """A broken SDK sends malformed requests; the model server rejects them with 400."""
    out = rts.primary.infer(batch.images, batch.captions)
    samples = traffic.samples_from_batch(batch, out, version="v1")
    mask = _pick(rng, len(batch), 0.03, 0.20)
    kind = rng.choice(["resolution", "dtype", "missing_param"])
    for i in np.flatnonzero(mask):
        req = samples[i]["request"]
        if kind == "resolution":
            req["inputs"][0] = traffic.tensor("image", np.zeros((1, 48, 48, 3), np.uint8))
        elif kind == "dtype":
            req["inputs"][0] = traffic.tensor("image", batch.images[i : i + 1].astype(np.float32))
        else:
            req["parameters"].pop("client", None)
        try:  # the server validates exactly as the bundled model server does
            parse_infer_request(req)
            if kind == "missing_param":
                continue  # server accepts it; only the contract objects
        except TensorError as exc:
            samples[i]["primary"] = traffic.exchange("v1", 400, 1.0, {"error": str(exc)})
    return samples


def output_corruption(batch, rng, rts):
    out = rts.primary.infer(batch.images, batch.captions)
    mask = _pick(rng, len(batch), 0.03, 0.20)
    if rng.random() < 0.5:
        out["text_embedding"][mask] *= rng.uniform(1.5, 4.0)  # normalization dropped
    else:
        out["image_embedding"][mask] = np.nan
    return traffic.samples_from_batch(batch, out, version="v1")


def bad_replica(batch, rng, rts):
    """One or more replicas run the regressed build; inputs are unchanged."""
    mask = _pick(rng, len(batch), 0.25, 1.0)
    good = rts.primary.infer(batch.images, batch.captions)
    bad = rts.regressed.infer(batch.images, batch.captions)
    mixed = {k: np.where(mask[:, None] if v.ndim == 2 else mask, bad[k], v) for k, v in good.items()}
    return traffic.samples_from_batch(batch, mixed, version="v1")


FAULTS: list[Fault] = [
    Fault(
        "exposure_shift",
        "camera exposure or white balance regression darkens 10 to 40%",
        "input_drift",
        exposure_shift,
    ),
    Fault("sensor_noise", "codec or sensor degradation adds noise", "input_drift", sensor_noise),
    Fault("resize_blur", "resize bug downsamples by 2x or 4x and upsamples", "input_drift", resize_blur),
    Fault("channel_swap", "client release sends BGR instead of RGB", "input_drift", channel_swap),
    Fault("frozen_frames", "stuck cameras send black, gray or white frames", "input_drift", frozen_frames),
    Fault("locale_launch", "new locale sends Spanish captions", "input_drift", locale_launch),
    Fault(
        "caption_template",
        "upstream template change wraps captions in boilerplate",
        "input_drift",
        caption_template,
    ),
    Fault("client_mix", "app release shifts 15 to 40% of traffic to android", "input_drift", client_mix),
    Fault(
        "schema_break", "SDK sends 48x48 images, FP32 tensors or drops parameters", "contract", schema_break
    ),
    Fault(
        "output_corruption", "post-processing drops normalization or emits NaN", "contract", output_corruption
    ),
    Fault("bad_replica", "some replicas serve the BGR-regressed build", "output_drift", bad_replica),
]


def clean(batch: Batch, rng: np.random.Generator, rts: Runtimes) -> list[dict]:
    return _infer_samples(batch, rts.primary)


def chunk(seed: int, n: int) -> Batch:
    return generate(n, seed=seed)

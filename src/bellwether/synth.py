"""Procedural image and caption traffic.

The generator produces 64x64 RGB scenes (a colored shape on a textured
background) with captions and request metadata. It is used to fit the bundled
encoders, to build baselines, and to drive the fault-injection benchmark, so a
fresh clone can exercise the whole system without downloading a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bellwether import IMAGE_SIZE

COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 180, 60),
    "blue": (40, 70, 220),
    "yellow": (235, 210, 40),
    "purple": (140, 50, 170),
    "orange": (240, 130, 30),
    "white": (242, 242, 242),
    "black": (22, 22, 22),
}
BACKGROUNDS: dict[str, tuple[int, int, int]] = {
    "light": (214, 214, 206),
    "dark": (42, 42, 50),
    "sky": (118, 168, 230),
    "grass": (88, 150, 72),
}
SHAPES = ("circle", "square", "triangle", "ring", "stripes", "cross")
SIZES = {"small": 9, "medium": 14, "large": 20}
TEMPLATES = (
    "a {size} {color} {shape} on a {bg} background",
    "{color} {shape}",
    "photo of a {color} {shape}",
    "there is a {size} {shape} in {color}",
    "{bg} scene with a {color} {shape}",
    "a {color} {shape} near the {pos}",
)
POSITIONS = ("center", "left", "right", "top", "bottom")
CLIENTS = (("ios", 0.50), ("android", 0.35), ("web", 0.15))
LOCALES = (("en-US", 0.60), ("en-GB", 0.15), ("es-ES", 0.10), ("fr-FR", 0.08), ("de-DE", 0.07))


@dataclass
class Batch:
    images: np.ndarray  # uint8 [N, H, W, 3]
    captions: list[str]
    metadata: list[dict[str, str]]
    labels: list[dict[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.captions)

    def subset(self, idx: np.ndarray | list[int]) -> Batch:
        idx = list(idx)
        return Batch(
            images=self.images[idx].copy(),
            captions=[self.captions[i] for i in idx],
            metadata=[dict(self.metadata[i]) for i in idx],
            labels=[dict(self.labels[i]) for i in idx] if self.labels else [],
        )


_GRID_Y, _GRID_X = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE].astype(np.float32)


def _shape_mask(shape: str, cx: float, cy: float, r: float, rng: np.random.Generator) -> np.ndarray:
    dx, dy = _GRID_X - cx, _GRID_Y - cy
    if shape == "circle":
        return dx**2 + dy**2 <= r**2
    if shape == "square":
        return (np.abs(dx) <= r * 0.85) & (np.abs(dy) <= r * 0.85)
    if shape == "triangle":
        return (dy <= r * 0.8) & (dy >= -r) & (np.abs(dx) <= (dy + r) * 0.6)
    if shape == "ring":
        d2 = dx**2 + dy**2
        return (d2 <= r**2) & (d2 >= (r * 0.55) ** 2)
    if shape == "stripes":
        period = rng.integers(5, 9)
        return (
            ((_GRID_X + _GRID_Y * rng.choice([0, 1])) // period % 2 == 0)
            & (np.abs(dx) <= r)
            & (np.abs(dy) <= r)
        )
    if shape == "cross":
        w = r * 0.3
        return ((np.abs(dx) <= w) & (np.abs(dy) <= r)) | ((np.abs(dy) <= w) & (np.abs(dx) <= r))
    raise ValueError(shape)


def _position_name(cx: float, cy: float) -> str:
    c = IMAGE_SIZE / 2
    if abs(cx - c) < 8 and abs(cy - c) < 8:
        return "center"
    if abs(cx - c) >= abs(cy - c):
        return "left" if cx < c else "right"
    return "top" if cy < c else "bottom"


def _pick(rng: np.random.Generator, options: tuple[tuple[str, float], ...]) -> str:
    names = [o[0] for o in options]
    probs = np.array([o[1] for o in options], dtype=np.float64)
    return names[rng.choice(len(names), p=probs / probs.sum())]


def generate(
    n: int,
    seed: int,
    *,
    clients: tuple[tuple[str, float], ...] = CLIENTS,
    locales: tuple[tuple[str, float], ...] = LOCALES,
) -> Batch:
    """Generate ``n`` image, caption and metadata triples deterministically from ``seed``."""
    rng = np.random.default_rng(seed)
    images = np.empty((n, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    captions: list[str] = []
    metadata: list[dict[str, str]] = []
    labels: list[dict[str, str]] = []
    color_names = list(COLORS)
    bg_names = list(BACKGROUNDS)
    ramp = np.linspace(-1.0, 1.0, IMAGE_SIZE, dtype=np.float32)[:, None, None]
    for i in range(n):
        bg = bg_names[rng.integers(len(bg_names))]
        color = color_names[rng.integers(len(color_names))]
        while np.abs(np.subtract(COLORS[color], BACKGROUNDS[bg])).sum() < 120:
            color = color_names[rng.integers(len(color_names))]
        shape = SHAPES[rng.integers(len(SHAPES))]
        size = list(SIZES)[rng.integers(len(SIZES))]
        r = SIZES[size] * rng.uniform(0.85, 1.15)
        cx = rng.uniform(r, IMAGE_SIZE - r)
        cy = rng.uniform(r, IMAGE_SIZE - r)

        canvas = np.empty((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        canvas[:] = np.asarray(BACKGROUNDS[bg], dtype=np.float32)
        canvas += ramp * rng.uniform(5, 25)
        mask = _shape_mask(shape, cx, cy, r, rng)
        fg = np.asarray(COLORS[color], dtype=np.float32) + rng.normal(0, 10, size=3)
        canvas[mask] = fg
        canvas += rng.normal(0, 6.0, size=canvas.shape)
        images[i] = np.clip(canvas, 0, 255).astype(np.uint8)

        template = TEMPLATES[rng.integers(len(TEMPLATES))]
        captions.append(
            template.format(size=size, color=color, shape=shape, bg=bg, pos=_position_name(cx, cy))
        )
        metadata.append({"client": _pick(rng, clients), "locale": _pick(rng, locales)})
        labels.append({"shape": shape, "color": color, "bg": bg, "size": size})
    return Batch(images=images, captions=captions, metadata=metadata, labels=labels)

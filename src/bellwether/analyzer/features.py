"""Feature extraction for drift detection.

Each valid item contributes image statistics (exposure, color, sharpness), a
model-independent thumbnail of the pixels, text statistics, request metadata
and the model's outputs. Features are
grouped as ``input`` or ``output`` so an incident can say whether traffic
changed or the model's behavior changed on similar-looking traffic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bellwether.analyzer.records import ItemRecord
from bellwether.models import tokenizer

SCALAR_GROUPS: dict[str, str] = {
    "image.brightness": "input",
    "image.contrast": "input",
    "image.red_mean": "input",
    "image.green_mean": "input",
    "image.blue_mean": "input",
    "image.clipped_fraction": "input",
    "image.sharpness": "input",
    "image.colorfulness": "input",
    "text.chars": "input",
    "text.tokens": "input",
    "text.non_ascii_fraction": "input",
    "text.digit_fraction": "input",
    "output.similarity": "output",
}
CATEGORICAL_GROUPS: dict[str, str] = {"param.client": "input", "param.locale": "input"}
EMBEDDING_GROUPS: dict[str, str] = {
    # model-independent view of the pixels, so an input change is attributed to inputs
    # even when no hand-picked statistic moves (and a model change never touches it)
    "image.thumbnail": "input",
    "output.image_embedding": "output",
    "output.text_embedding": "output",
}
FEATURE_GROUPS = {**SCALAR_GROUPS, **CATEGORICAL_GROUPS, **EMBEDDING_GROUPS}


def image_stats(images: np.ndarray, chunk: int = 1024) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {k: [] for k in SCALAR_GROUPS if k.startswith("image.")}
    for start in range(0, len(images), chunk):
        x = images[start : start + chunk].astype(np.float32)
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        gray = x.mean(axis=-1)
        lap = (
            4 * gray[:, 1:-1, 1:-1]
            - gray[:, :-2, 1:-1]
            - gray[:, 2:, 1:-1]
            - gray[:, 1:-1, :-2]
            - gray[:, 1:-1, 2:]
        )
        rg, yb = r - g, 0.5 * (r + g) - b
        axes = (1, 2)
        parts["image.brightness"].append(gray.mean(axis=axes))
        parts["image.contrast"].append(gray.std(axis=axes))
        parts["image.red_mean"].append(r.mean(axis=axes))
        parts["image.green_mean"].append(g.mean(axis=axes))
        parts["image.blue_mean"].append(b.mean(axis=axes))
        parts["image.clipped_fraction"].append(((x <= 2) | (x >= 253)).mean(axis=(1, 2, 3)))
        parts["image.sharpness"].append(np.log1p(lap.var(axis=axes)))
        parts["image.colorfulness"].append(
            np.sqrt(rg.std(axis=axes) ** 2 + yb.std(axis=axes) ** 2)
            + 0.3 * np.sqrt(rg.mean(axis=axes) ** 2 + yb.mean(axis=axes) ** 2)
        )
    return {k: np.concatenate(v) if v else np.empty(0) for k, v in parts.items()}


def thumbnails(images: np.ndarray, size: int = 8) -> np.ndarray:
    """Block-averaged size x size RGB thumbnails, flattened and scaled to [0, 1]."""
    n, h, w, c = images.shape
    k = h // size
    return images.reshape(n, size, k, size, k, c).mean(axis=(2, 4)).reshape(n, -1) / 255.0


def text_stats(texts: Sequence[str]) -> dict[str, np.ndarray]:
    chars = np.array([len(t) for t in texts], dtype=np.float64)
    safe = np.maximum(chars, 1)
    return {
        "text.chars": chars,
        "text.tokens": np.array([len(tokenizer.tokenize(t)) for t in texts], dtype=np.float64),
        "text.non_ascii_fraction": np.array([sum(ord(c) > 127 for c in t) for t in texts]) / safe,
        "text.digit_fraction": np.array([sum(c.isdigit() for c in t) for t in texts]) / safe,
    }


@dataclass
class FeatureFrame:
    scalars: dict[str, np.ndarray]
    categoricals: dict[str, np.ndarray]
    embeddings: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return len(next(iter(self.scalars.values()))) if self.scalars else 0

    @classmethod
    def from_items(cls, items: Sequence[ItemRecord]) -> FeatureFrame:
        if not items:
            raise ValueError("no items")
        images = np.stack([it.image for it in items])
        texts = [it.text or "" for it in items]
        scalars = {**image_stats(images), **text_stats(texts)}
        scalars["output.similarity"] = np.array([it.similarity for it in items], dtype=np.float64)
        categoricals = {
            "param.client": np.array(
                [str(it.parameters.get("client", "<missing>")) for it in items], dtype=object
            ),
            "param.locale": np.array(
                [str(it.parameters.get("locale", "<missing>")) for it in items], dtype=object
            ),
        }
        embeddings = {
            "image.thumbnail": thumbnails(images),
            "output.image_embedding": np.stack([it.image_embedding for it in items]).astype(np.float64),
            "output.text_embedding": np.stack([it.text_embedding for it in items]).astype(np.float64),
        }
        return cls(scalars, categoricals, embeddings)

    def take(self, idx: np.ndarray) -> FeatureFrame:
        return FeatureFrame(
            {k: v[idx] for k, v in self.scalars.items()},
            {k: v[idx] for k, v in self.categoricals.items()},
            {k: v[idx] for k, v in self.embeddings.items()},
        )

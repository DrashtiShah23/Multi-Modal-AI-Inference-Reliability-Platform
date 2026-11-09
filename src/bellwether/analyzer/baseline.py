"""Reference profiles and window-level drift scoring.

A baseline is built from known-good traffic and split in two:

* a comparison set that every new window is tested against, and
* a disjoint holdout used only to simulate clean windows.

Running all 18 tests on simulated clean windows gives the empirical null
distribution of the *smallest* p-value in a window. The alarm threshold is that
distribution's ``alpha`` quantile, so the window-level false alarm rate is
calibrated directly, accounting for correlation between tests, instead of
relying on a Bonferroni bound.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field

import numpy as np

from bellwether.analyzer import detectors
from bellwether.analyzer.features import CATEGORICAL_GROUPS, EMBEDDING_GROUPS, SCALAR_GROUPS, FeatureFrame

# Once a window fires, attribute it to inputs if any input test shows evidence at
# this looser level. Detection stays governed by the calibrated threshold; the
# looser bar only keeps an input change from being reported as a model change
# when the pixels or text moved but no single input test crossed the alarm line.
ATTRIBUTION_P = 1e-3


@dataclass
class DriftEvaluation:
    min_p: float
    threshold: float
    fired: bool
    tests: list[detectors.TestResult]

    @property
    def fired_tests(self) -> list[detectors.TestResult]:
        return sorted((t for t in self.tests if t.p_value < self.threshold), key=lambda t: t.p_value)

    @property
    def kind(self) -> str | None:
        if not self.fired:
            return None
        bar = max(self.threshold, ATTRIBUTION_P)
        return (
            "input_drift"
            if any(t.group == "input" and t.p_value < bar for t in self.tests)
            else "output_drift"
        )


@dataclass
class Baseline:
    model: str
    version: str
    window_size: int
    alpha: float
    threshold: float = 0.0
    created_at: float = field(default_factory=time.time)
    n_reference: int = 0
    scalar_ref: dict[str, np.ndarray] = field(default_factory=dict)
    psi_edges: dict[str, np.ndarray] = field(default_factory=dict)
    psi_ref: dict[str, np.ndarray] = field(default_factory=dict)
    cat_ref: dict[str, dict[str, int]] = field(default_factory=dict)
    emb_ref: dict[str, np.ndarray] = field(default_factory=dict)
    emb_gamma: dict[str, float] = field(default_factory=dict)
    emb_null: dict[str, detectors.GammaNull] = field(default_factory=dict)
    calibration: dict[str, float] = field(default_factory=dict)
    _mmd: dict[str, detectors.MMD] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- scoring
    def _detector(self, name: str) -> detectors.MMD:
        if name not in self._mmd:
            self._mmd[name] = detectors.MMD(self.emb_ref[name], gamma=self.emb_gamma.get(name))
        return self._mmd[name]

    def p_values(
        self, frame: FeatureFrame, *, embeddings_null: dict[str, detectors.GammaNull] | None = None
    ) -> list[detectors.TestResult]:
        nulls = embeddings_null if embeddings_null is not None else self.emb_null
        results: list[detectors.TestResult] = []
        for name, ref in self.scalar_ref.items():
            cur = frame.scalars[name]
            stat, p = detectors.ks_test(ref, cur)
            effect = detectors.psi(self.psi_ref[name], detectors.binned_fractions(cur, self.psi_edges[name]))
            results.append(detectors.TestResult(name, "ks", SCALAR_GROUPS[name], stat, p, effect))
        for name, counts in self.cat_ref.items():
            stat, p, effect = detectors.categorical_test(counts, frame.categoricals[name])
            results.append(detectors.TestResult(name, "chi2", CATEGORICAL_GROUPS[name], stat, p, effect))
        for name in self.emb_ref:
            value = self._detector(name).statistic(frame.embeddings[name])
            p = nulls[name].p_value(value) if name in nulls else 1.0
            results.append(detectors.TestResult(name, "mmd", EMBEDDING_GROUPS[name], value, p, value))
        return results

    def evaluate(self, frame: FeatureFrame) -> DriftEvaluation:
        tests = self.p_values(frame)
        min_p = min(t.p_value for t in tests)
        return DriftEvaluation(
            min_p=min_p, threshold=self.threshold, fired=min_p < self.threshold, tests=tests
        )

    # ---------------------------------------------------------------- building
    @classmethod
    def build(
        cls,
        frame: FeatureFrame,
        *,
        model: str,
        version: str,
        window_size: int = 400,
        alpha: float = 0.01,
        n_compare: int = 2000,
        n_embedding_ref: int = 1000,
        n_gamma_windows: int = 300,
        n_calibration_windows: int = 1000,
        seed: int = 0,
    ) -> Baseline:
        rng = np.random.default_rng(seed)
        n = frame.n
        if n < n_compare + 2 * window_size:
            raise ValueError(f"need at least {n_compare + 2 * window_size} reference items, got {n}")
        perm = rng.permutation(n)
        ref, hold = frame.take(perm[:n_compare]), frame.take(perm[n_compare:])

        b = cls(model=model, version=version, window_size=window_size, alpha=alpha, n_reference=n)
        for name, values in ref.scalars.items():
            b.scalar_ref[name] = np.sort(values)
            b.psi_edges[name] = detectors.quantile_edges(values)
            b.psi_ref[name] = detectors.binned_fractions(values, b.psi_edges[name])
        for name, values in ref.categoricals.items():
            cats, counts = np.unique(values.astype(str), return_counts=True)
            b.cat_ref[name] = dict(zip(cats.tolist(), counts.tolist(), strict=True))
        for name, values in ref.embeddings.items():
            b.emb_ref[name] = values[: min(n_embedding_ref, len(values))]
            b.emb_gamma[name] = b._detector(name).gamma

        def null_window() -> FeatureFrame:
            return hold.take(rng.choice(hold.n, size=window_size, replace=False))

        # 1) gamma approximation of each embedding MMD null
        null_mmd: dict[str, list[float]] = {name: [] for name in b.emb_ref}
        for _ in range(n_gamma_windows):
            w = null_window()
            for name in b.emb_ref:
                null_mmd[name].append(b._detector(name).statistic(w.embeddings[name]))
        b.emb_null = {name: detectors.GammaNull.fit(np.array(v)) for name, v in null_mmd.items()}

        # 2) empirical null of the window's smallest p-value, on fresh windows
        min_ps = np.array(
            [min(t.p_value for t in b.p_values(null_window())) for _ in range(n_calibration_windows)]
        )
        b.threshold = float(np.quantile(min_ps, alpha))
        b.calibration = {
            "null_windows": float(n_calibration_windows),
            "null_min_p_median": float(np.median(min_ps)),
            "threshold": b.threshold,
            "bonferroni_equivalent": float(alpha / (len(b.scalar_ref) + len(b.cat_ref) + len(b.emb_ref))),
        }
        return b

    # ---------------------------------------------------------------- persistence
    def to_bytes(self) -> bytes:
        arrays: dict[str, np.ndarray] = {}
        for name, v in self.scalar_ref.items():
            arrays[f"scalar|{name}"] = v
            arrays[f"edges|{name}"] = self.psi_edges[name]
            arrays[f"psi|{name}"] = self.psi_ref[name]
        for name, v in self.emb_ref.items():
            arrays[f"emb|{name}"] = v
        meta = {
            "model": self.model,
            "version": self.version,
            "window_size": self.window_size,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "created_at": self.created_at,
            "n_reference": self.n_reference,
            "cat_ref": self.cat_ref,
            "emb_gamma": self.emb_gamma,
            "emb_null": {k: [v.shape, v.scale] for k, v in self.emb_null.items()},
            "calibration": self.calibration,
        }
        arrays["meta"] = np.array(json.dumps(meta))
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> Baseline:
        npz = np.load(io.BytesIO(data), allow_pickle=False)
        meta = json.loads(str(npz["meta"]))
        b = cls(
            model=meta["model"],
            version=meta["version"],
            window_size=int(meta["window_size"]),
            alpha=float(meta["alpha"]),
            threshold=float(meta["threshold"]),
            created_at=float(meta["created_at"]),
            n_reference=int(meta["n_reference"]),
            cat_ref={k: {c: int(n) for c, n in v.items()} for k, v in meta["cat_ref"].items()},
            emb_gamma={k: float(v) for k, v in meta["emb_gamma"].items()},
            emb_null={k: detectors.GammaNull(float(v[0]), float(v[1])) for k, v in meta["emb_null"].items()},
            calibration=meta.get("calibration", {}),
        )
        for key in npz.files:
            kind, _, name = key.partition("|")
            if kind == "scalar":
                b.scalar_ref[name] = npz[key]
            elif kind == "edges":
                b.psi_edges[name] = npz[key]
            elif kind == "psi":
                b.psi_ref[name] = npz[key]
            elif kind == "emb":
                b.emb_ref[name] = npz[key]
        return b

    def summary(self) -> dict:
        return {
            "model": self.model,
            "version": self.version,
            "window_size": self.window_size,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "n_reference": self.n_reference,
            "created_at": self.created_at,
            "tests": len(self.scalar_ref) + len(self.cat_ref) + len(self.emb_ref),
            "calibration": self.calibration,
        }

"""Two-sample tests used for drift detection.

* Scalars: Kolmogorov-Smirnov p-value, with population stability index (PSI) as effect size.
* Categoricals: chi-square test of homogeneity, with PSI over categories.
* Embeddings: kernel maximum mean discrepancy (RBF kernel, median-heuristic
  bandwidth). Its null distribution is approximated by a gamma fitted to MMD
  values from reference windows (Gretton et al., 2009), which avoids running a
  permutation test on every window.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import stats

_EPS = 1e-4


@dataclass(frozen=True)
class TestResult:
    feature: str
    test: str
    group: str
    statistic: float
    p_value: float
    effect: float

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "test": self.test,
            "group": self.group,
            "statistic": round(float(self.statistic), 6),
            "p_value": float(self.p_value),
            "effect": round(float(self.effect), 6),
        }


def quantile_edges(ref: np.ndarray, bins: int = 10) -> np.ndarray:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    return edges


def binned_fractions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / max(len(values), 1)


def psi(ref_frac: np.ndarray, cur_frac: np.ndarray) -> float:
    r = np.clip(ref_frac, _EPS, None)
    c = np.clip(cur_frac, _EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_test(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    res = stats.ks_2samp(ref, cur, method="asymp")
    return float(res.statistic), float(res.pvalue)


def categorical_test(ref_counts: Mapping[str, int], cur: np.ndarray) -> tuple[float, float, float]:
    values, counts = np.unique(cur.astype(str), return_counts=True)
    cur_counts = dict(zip(values.tolist(), counts.tolist(), strict=True))
    cats = sorted(set(ref_counts) | set(cur_counts))
    table = np.array(
        [[ref_counts.get(c, 0) for c in cats], [cur_counts.get(c, 0) for c in cats]], dtype=np.float64
    )
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 0.0, 1.0, 0.0
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    effect = psi(table[0] / table[0].sum(), table[1] / table[1].sum())
    return float(chi2), float(p), effect


class MMD:
    """Biased (V-statistic) squared MMD against a fixed reference sample."""

    def __init__(self, reference: np.ndarray, gamma: float | None = None):
        self.reference = np.asarray(reference, dtype=np.float64)
        self._ref_sq = (self.reference**2).sum(axis=1)
        if gamma is None:
            sample = self.reference[: min(len(self.reference), 1000)]
            d2 = self._sqdist(sample, (sample**2).sum(1), sample, (sample**2).sum(1))
            med = np.median(d2[np.triu_indices(len(sample), k=1)])
            gamma = 1.0 / max(med, 1e-12)
        self.gamma = float(gamma)
        self.kyy = float(self._kernel(self.reference, self._ref_sq, self.reference, self._ref_sq).mean())

    @staticmethod
    def _sqdist(a: np.ndarray, a_sq: np.ndarray, b: np.ndarray, b_sq: np.ndarray) -> np.ndarray:
        return np.maximum(a_sq[:, None] + b_sq[None, :] - 2.0 * (a @ b.T), 0.0)

    def _kernel(self, a, a_sq, b, b_sq) -> np.ndarray:
        return np.exp(-self.gamma * self._sqdist(a, a_sq, b, b_sq))

    def statistic(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        x_sq = (x**2).sum(axis=1)
        kxx = self._kernel(x, x_sq, x, x_sq).mean()
        kxy = self._kernel(x, x_sq, self.reference, self._ref_sq).mean()
        return float(kxx + self.kyy - 2.0 * kxy)


@dataclass(frozen=True)
class GammaNull:
    shape: float
    scale: float

    @classmethod
    def fit(cls, null_values: np.ndarray) -> GammaNull:
        v = np.asarray(null_values, dtype=np.float64)
        mean, var = v.mean(), v.var(ddof=1)
        return cls(shape=float(mean**2 / var), scale=float(var / mean))

    def p_value(self, value: float) -> float:
        return float(stats.gamma.sf(value, self.shape, scale=self.scale))

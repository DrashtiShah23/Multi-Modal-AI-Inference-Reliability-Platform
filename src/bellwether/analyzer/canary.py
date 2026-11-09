"""Shadow-canary analysis.

The gateway mirrors a fraction of live requests to the candidate release, so
every mirrored request yields a *paired* comparison on identical input. That
lets the verdict check behavior (embedding agreement, similarity error) and
not just health, which matters for embedding models: a release can return
200s at the same latency while silently moving every vector in the index.

Verdict rules
  ROLLBACK  any check with enough data fails
  PROMOTE   every check has enough data and passes
  HOLD      otherwise (still collecting)
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from bellwether.analyzer.records import Exchange, ItemRecord
from bellwether.analyzer.slo import SLO


@dataclass(frozen=True)
class CanaryConfig:
    min_requests: int = 100
    min_pairs: int = 200
    max_observations: int = 5000
    slo_target: float = 0.999
    max_error_burn: float = 14.4
    max_latency_ratio: float = 1.25
    min_latency_delta_ms: float = 5.0
    min_image_cosine_p10: float = 0.85
    min_text_cosine_p10: float = 0.85
    max_similarity_mae: float = 0.08
    max_violation_excess: float = 0.005


@dataclass
class Check:
    name: str
    value: float
    threshold: float
    passed: bool
    sufficient_data: bool
    samples: int
    detail: str = ""


@dataclass
class Verdict:
    model: str
    candidate_version: str
    decision: str  # HOLD | PROMOTE | ROLLBACK
    checks: list[Check]

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "candidate_version": self.candidate_version,
            "decision": self.decision,
            "checks": [asdict(c) for c in self.checks],
        }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class CanaryTracker:
    def __init__(self, model: str, candidate_version: str, config: CanaryConfig | None = None):
        self.model = model
        self.candidate_version = candidate_version
        self.config = config or CanaryConfig()
        n = self.config.max_observations
        self._requests: deque[tuple[int, int, float, float]] = deque(maxlen=n)
        self._pairs: deque[tuple[float, float, float]] = deque(maxlen=n)
        self._violations: deque[tuple[bool, bool]] = deque(maxlen=n)

    def add(
        self,
        primary: Exchange,
        candidate: Exchange,
        primary_items: list[ItemRecord],
        candidate_items: list[ItemRecord],
    ) -> None:
        self._requests.append((primary.status, candidate.status, primary.latency_ms, candidate.latency_ms))
        for p, c in zip(primary_items, candidate_items, strict=False):
            self._violations.append((bool(p.violations), bool(c.violations)))
            if p.valid and c.valid:
                self._pairs.append(
                    (
                        _cos(p.image_embedding, c.image_embedding),
                        _cos(p.text_embedding, c.text_embedding),
                        abs(float(p.similarity) - float(c.similarity)),
                    )
                )

    def verdict(self) -> Verdict:
        cfg = self.config
        checks: list[Check] = []
        reqs = np.array(self._requests, dtype=np.float64).reshape(-1, 4)
        n_req = len(reqs)

        # 1. error budget burn of the candidate, relative to an SLO
        slo = SLO(cfg.slo_target)
        cand_err = float((reqs[:, 1] != 200).mean()) if n_req else 0.0
        prim_err = float((reqs[:, 0] != 200).mean()) if n_req else 0.0
        burn = slo.burn_rate(cand_err)
        checks.append(
            Check(
                "error_budget_burn",
                burn,
                cfg.max_error_burn,
                passed=not (burn > cfg.max_error_burn and cand_err > prim_err),
                sufficient_data=n_req >= cfg.min_requests,
                samples=n_req,
                detail=f"candidate error ratio {cand_err:.4f} vs primary {prim_err:.4f}",
            )
        )

        # 2. latency regression on paired requests that both succeeded
        both_ok = reqs[(reqs[:, 0] == 200) & (reqs[:, 1] == 200)] if n_req else reqs
        if len(both_ok):
            p95_p, p95_c = np.percentile(both_ok[:, 2], 95), np.percentile(both_ok[:, 3], 95)
            ratio = float(p95_c / max(p95_p, 1e-9))
            regressed = ratio > cfg.max_latency_ratio and (p95_c - p95_p) > cfg.min_latency_delta_ms
            detail = f"p95 {p95_c:.1f} ms vs {p95_p:.1f} ms"
        else:
            ratio, regressed, detail = 1.0, False, "no paired successes"
        checks.append(
            Check(
                "latency_p95_ratio",
                ratio,
                cfg.max_latency_ratio,
                not regressed,
                len(both_ok) >= cfg.min_requests,
                len(both_ok),
                detail,
            )
        )

        # 3. behavioral agreement on identical inputs
        pairs = np.array(self._pairs, dtype=np.float64).reshape(-1, 3)
        enough = len(pairs) >= cfg.min_pairs
        img_p10 = float(np.percentile(pairs[:, 0], 10)) if len(pairs) else 1.0
        txt_p10 = float(np.percentile(pairs[:, 1], 10)) if len(pairs) else 1.0
        mae = float(pairs[:, 2].mean()) if len(pairs) else 0.0
        checks.append(
            Check(
                "image_embedding_cosine_p10",
                img_p10,
                cfg.min_image_cosine_p10,
                img_p10 >= cfg.min_image_cosine_p10,
                enough,
                len(pairs),
            )
        )
        checks.append(
            Check(
                "text_embedding_cosine_p10",
                txt_p10,
                cfg.min_text_cosine_p10,
                txt_p10 >= cfg.min_text_cosine_p10,
                enough,
                len(pairs),
            )
        )
        checks.append(
            Check(
                "similarity_mae",
                mae,
                cfg.max_similarity_mae,
                mae <= cfg.max_similarity_mae,
                enough,
                len(pairs),
            )
        )

        # 4. contract violations introduced by the candidate
        viol = np.array(self._violations, dtype=bool).reshape(-1, 2)
        excess = float(viol[:, 1].mean() - viol[:, 0].mean()) if len(viol) else 0.0
        checks.append(
            Check(
                "contract_violation_excess",
                excess,
                cfg.max_violation_excess,
                excess <= cfg.max_violation_excess,
                len(viol) >= cfg.min_pairs,
                len(viol),
            )
        )

        if any(c.sufficient_data and not c.passed for c in checks):
            decision = "ROLLBACK"
        elif all(c.sufficient_data and c.passed for c in checks):
            decision = "PROMOTE"
        else:
            decision = "HOLD"
        return Verdict(self.model, self.candidate_version, decision, checks)

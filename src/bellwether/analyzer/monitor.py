"""Streaming window evaluation for one (model, version) on the primary route.

Two windows run side by side:

* contract windows close every ``window_size`` items and alarm when the share of
  items with any violation exceeds the contract's ``max_violation_rate``;
* drift windows close every ``window_size`` *valid* items and alarm when the
  smallest p-value across all tests falls below the baseline's calibrated
  threshold. Using only valid items keeps the drift test's null distribution
  the same size the baseline was calibrated for.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from bellwether.analyzer.baseline import Baseline
from bellwether.analyzer.contracts import Contract
from bellwether.analyzer.features import FeatureFrame
from bellwether.analyzer.records import ItemRecord


@dataclass
class WindowResult:
    kind: str  # "contract" | "drift"
    model: str
    version: str
    n: int
    fired: bool
    score: float
    threshold: float
    incident_kind: str | None
    severity: str | None
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    ended_at: float = 0.0


class StreamMonitor:
    def __init__(
        self,
        model: str,
        version: str,
        contract: Contract,
        window_size: int = 400,
        baseline: Baseline | None = None,
    ):
        self.model = model
        self.version = version
        self.contract = contract
        self.window_size = window_size
        self.baseline = baseline
        self._contract_n = 0
        self._contract_bad = 0
        self._codes: Counter[str] = Counter()
        self._contract_started = 0.0
        self._drift: list[ItemRecord] = []

    def add(self, items: list[ItemRecord]) -> list[WindowResult]:
        results: list[WindowResult] = []
        for it in items:
            if self._contract_n == 0:
                self._contract_started = it.received_at or time.time()
            self._contract_n += 1
            if it.violations:
                self._contract_bad += 1
                self._codes.update(it.violations)
            if it.valid:
                self._drift.append(it)
            if self._contract_n >= self.window_size:
                results.append(self._close_contract(it.received_at or time.time()))
            if self.baseline is not None and len(self._drift) >= self.baseline.window_size:
                results.append(self._close_drift())
        return results

    def _close_contract(self, ended_at: float) -> WindowResult:
        n, bad = self._contract_n, self._contract_bad
        rate = bad / n
        limit = self.contract.max_violation_rate
        fired = rate > limit
        top = self._codes.most_common(5)
        self._contract_n = self._contract_bad = 0
        self._codes = Counter()
        summary = (
            f"{rate:.1%} of {n} items broke the data contract (limit {limit:.1%}); top: "
            + ", ".join(f"{c} x{k}" for c, k in top)
            if fired
            else f"{rate:.1%} contract violations in {n} items"
        )
        return WindowResult(
            kind="contract",
            model=self.model,
            version=self.version,
            n=n,
            fired=fired,
            score=rate,
            threshold=limit,
            incident_kind="contract" if fired else None,
            severity=("critical" if rate > 0.2 else "warning") if fired else None,
            summary=summary,
            details={"violation_rate": rate, "top_violations": [{"code": c, "count": k} for c, k in top]},
            started_at=self._contract_started,
            ended_at=ended_at,
        )

    def _close_drift(self) -> WindowResult:
        assert self.baseline is not None
        size = self.baseline.window_size
        window, self._drift = self._drift[:size], self._drift[size:]
        evaluation = self.baseline.evaluate(FeatureFrame.from_items(window))
        fired = evaluation.fired_tests
        if evaluation.fired:
            names = ", ".join(f"{t.feature} (p={t.p_value:.1e})" for t in fired[:4])
            summary = f"{evaluation.kind.replace('_', ' ')} in {size} items: {names}"
            severity = "critical" if evaluation.min_p < evaluation.threshold / 100 else "warning"
        else:
            summary = f"no drift in {size} items (min p={evaluation.min_p:.2e})"
            severity = None
        return WindowResult(
            kind="drift",
            model=self.model,
            version=self.version,
            n=size,
            fired=evaluation.fired,
            score=evaluation.min_p,
            threshold=evaluation.threshold,
            incident_kind=evaluation.kind,
            severity=severity,
            summary=summary,
            details={
                "fired_tests": [t.as_dict() for t in fired[:8]],
                "tests": [t.as_dict() for t in evaluation.tests],
            },
            started_at=window[0].received_at,
            ended_at=window[-1].received_at,
        )

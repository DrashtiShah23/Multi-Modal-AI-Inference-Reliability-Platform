"""Availability SLOs and multi-window, multi-burn-rate alerting.

Follows the Google SRE Workbook ("Alerting on SLOs"): an alert fires only when
both a long and a short window burn the error budget faster than a policy's
rate, which keeps detection fast while letting alerts reset quickly once a
problem is fixed. The same policies are expressed as Prometheus rules in
deploy/k8s/base/prometheus-rules.yaml for the gateway's metrics.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class SLO:
    target: float = 0.999
    period_days: float = 30.0

    @property
    def error_budget(self) -> float:
        return 1.0 - self.target

    def burn_rate(self, error_ratio: float) -> float:
        return error_ratio / self.error_budget

    def budget_consumed(self, burn_rate: float, window_hours: float) -> float:
        """Fraction of the period's error budget spent by burning at ``burn_rate`` for ``window_hours``."""
        return burn_rate * window_hours / (self.period_days * 24.0)


@dataclass(frozen=True)
class BurnPolicy:
    name: str
    long_seconds: int
    short_seconds: int
    burn_rate: float
    severity: str


DEFAULT_POLICIES = (
    BurnPolicy("fast", long_seconds=3600, short_seconds=300, burn_rate=14.4, severity="page"),
    BurnPolicy("medium", long_seconds=6 * 3600, short_seconds=1800, burn_rate=6.0, severity="page"),
    BurnPolicy("slow", long_seconds=3 * 86400, short_seconds=6 * 3600, burn_rate=1.0, severity="ticket"),
)


class RollingCounter:
    """Request and error counts in fixed-width time buckets over a bounded horizon."""

    def __init__(self, bucket_seconds: int = 10, horizon_seconds: int = 3 * 86400):
        self.bucket_seconds = bucket_seconds
        self.horizon = horizon_seconds
        self._buckets: deque[list[float]] = deque()  # [bucket_start, total, errors]

    def add(self, ts: float, ok: bool, count: int = 1) -> None:
        start = math.floor(ts / self.bucket_seconds) * self.bucket_seconds
        if self._buckets and self._buckets[-1][0] == start:
            bucket = self._buckets[-1]
        elif self._buckets and start < self._buckets[-1][0]:
            bucket = next((b for b in reversed(self._buckets) if b[0] == start), None)
            if bucket is None:
                return  # too old or out of order beyond the newest bucket; ignore
        else:
            bucket = [start, 0.0, 0.0]
            self._buckets.append(bucket)
        bucket[1] += count
        if not ok:
            bucket[2] += count
        while self._buckets and self._buckets[0][0] < start - self.horizon:
            self._buckets.popleft()

    def window(self, now: float, seconds: int) -> tuple[float, float]:
        total = errors = 0.0
        for start, t, e in reversed(self._buckets):
            if start < now - seconds:
                break
            total += t
            errors += e
        return total, errors

    def error_ratio(self, now: float, seconds: int) -> float:
        total, errors = self.window(now, seconds)
        return errors / total if total else 0.0


@dataclass(frozen=True)
class BurnAlert:
    policy: str
    severity: str
    long_burn: float
    short_burn: float
    threshold: float


def evaluate_burn(
    counter: RollingCounter, slo: SLO, now: float, policies: tuple[BurnPolicy, ...] = DEFAULT_POLICIES
) -> list[BurnAlert]:
    alerts = []
    for p in policies:
        long_burn = slo.burn_rate(counter.error_ratio(now, p.long_seconds))
        short_burn = slo.burn_rate(counter.error_ratio(now, p.short_seconds))
        if long_burn > p.burn_rate and short_burn > p.burn_rate:
            alerts.append(BurnAlert(p.name, p.severity, long_burn, short_burn, p.burn_rate))
    return alerts

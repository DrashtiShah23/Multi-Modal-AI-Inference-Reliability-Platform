"""Analyzer core shared by the REST and gRPC front ends."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field, fields
from typing import Any

from bellwether.analyzer.baseline import Baseline
from bellwether.analyzer.canary import CanaryConfig, CanaryTracker, Verdict
from bellwether.analyzer.contracts import Contract
from bellwether.analyzer.features import FeatureFrame
from bellwether.analyzer.monitor import StreamMonitor, WindowResult
from bellwether.analyzer.records import ItemRecord, SampleError, explode, parse_sample
from bellwether.analyzer.slo import SLO, RollingCounter
from bellwether.analyzer.store import Store
from bellwether.analyzer.telemetry import Metrics, link_from_traceparent, setup_tracing

log = logging.getLogger(__name__)
DECISION_VALUE = {"PROMOTE": 1, "HOLD": 0, "ROLLBACK": -1}


@dataclass
class Settings:
    db_url: str = "sqlite:///bellwether.db"
    contract_path: str | None = None
    window_size: int = 400
    alpha: float = 0.01
    learn_baseline: bool = True
    baseline_items: int = 6000
    slo_target: float = 0.999
    canary: CanaryConfig = field(default_factory=CanaryConfig)

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        canary_overrides = {}
        for f in fields(CanaryConfig):
            raw = env(f"BELLWETHER_CANARY_{f.name.upper()}")
            if raw is not None:
                canary_overrides[f.name] = type(getattr(CanaryConfig(), f.name))(raw)
        return cls(
            db_url=env("BELLWETHER_DB_URL", cls.db_url),
            contract_path=env("BELLWETHER_CONTRACT"),
            window_size=int(env("BELLWETHER_WINDOW_SIZE", cls.window_size)),
            alpha=float(env("BELLWETHER_ALPHA", cls.alpha)),
            learn_baseline=env("BELLWETHER_LEARN_BASELINE", "true").lower() == "true",
            baseline_items=int(env("BELLWETHER_BASELINE_ITEMS", cls.baseline_items)),
            slo_target=float(env("BELLWETHER_SLO_TARGET", cls.slo_target)),
            canary=CanaryConfig(**canary_overrides),
        )


class AnalyzerService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: Store | None = None,
        contract: Contract | None = None,
        metrics: Metrics | None = None,
    ):
        self.settings = settings or Settings()
        self.store = store or Store(self.settings.db_url)
        self.contract = contract or Contract.load(self.settings.contract_path)
        self.metrics = metrics or Metrics()
        self.tracer = setup_tracing()
        self.slo = SLO(self.settings.slo_target)
        self._lock = threading.RLock()
        self._monitors: dict[tuple[str, str], StreamMonitor] = {}
        self._canaries: dict[tuple[str, str], CanaryTracker] = {}
        self._baselines: dict[str, Baseline | None] = {}
        self._learning: dict[str, list[ItemRecord]] = {}
        self._building: set[str] = set()
        self._counters: dict[tuple[str, str], RollingCounter] = {}
        self._last_decision: dict[tuple[str, str], str] = {}

    # ---------------------------------------------------------------- baselines
    def baseline(self, model: str) -> Baseline | None:
        with self._lock:
            if model not in self._baselines:
                self._baselines[model] = self.store.latest_baseline(model)
                self.metrics.baseline_ready.labels(model).set(1 if self._baselines[model] else 0)
            return self._baselines[model]

    def install_baseline(self, baseline: Baseline, persist: bool = True) -> None:
        if persist:
            self.store.save_baseline(baseline)
        with self._lock:
            self._baselines[baseline.model] = baseline
            self._learning.pop(baseline.model, None)
            for (model, _), monitor in self._monitors.items():
                if model == baseline.model:
                    monitor.baseline = baseline
        self.metrics.baseline_ready.labels(baseline.model).set(1)
        self.metrics.drift_threshold.labels(baseline.model, baseline.version).set(baseline.threshold)
        log.info("baseline installed for %s (threshold %.3g)", baseline.model, baseline.threshold)

    def build_baseline(self, model: str, version: str, items: list[ItemRecord], **kwargs: Any) -> Baseline:
        frame = FeatureFrame.from_items([it for it in items if it.valid])
        b = Baseline.build(
            frame,
            model=model,
            version=version,
            window_size=self.settings.window_size,
            alpha=self.settings.alpha,
            **kwargs,
        )
        self.install_baseline(b)
        return b

    def relearn_baseline(self, model: str) -> None:
        with self._lock:
            self._baselines[model] = None
            self._learning[model] = []
            for (m, _), monitor in self._monitors.items():
                if m == model:
                    monitor.baseline = None
        self.metrics.baseline_ready.labels(model).set(0)

    def _maybe_learn(self, model: str, version: str, items: list[ItemRecord]) -> None:
        if not self.settings.learn_baseline or self.baseline(model) is not None or model in self._building:
            return
        pool = self._learning.setdefault(model, [])
        pool.extend(it for it in items if it.valid)
        if len(pool) >= self.settings.baseline_items:
            self._building.add(model)
            snapshot = pool[: self.settings.baseline_items]
            self._learning[model] = []

            def run() -> None:
                try:
                    self.build_baseline(model, version, snapshot)
                except Exception:  # pragma: no cover - logged for operators
                    log.exception("baseline build failed for %s", model)
                finally:
                    self._building.discard(model)

            threading.Thread(target=run, name=f"baseline-{model}", daemon=True).start()

    # ---------------------------------------------------------------- ingest
    def _monitor(self, model: str, version: str) -> StreamMonitor:
        key = (model, version)
        if key not in self._monitors:
            self._monitors[key] = StreamMonitor(
                model, version, self.contract, self.settings.window_size, baseline=self.baseline(model)
            )
        return self._monitors[key]

    def _counter(self, model: str, version: str) -> RollingCounter:
        return self._counters.setdefault((model, version), RollingCounter())

    def ingest(self, raw_samples: list[Any]) -> dict[str, Any]:
        links = [
            link
            for s in raw_samples[:32]
            if isinstance(s, dict)
            for link in [link_from_traceparent(str(s.get("traceparent", "")))]
            if link
        ]
        with self.tracer.start_as_current_span("analyzer.ingest", links=links) as span:
            span.set_attribute("bellwether.samples", len(raw_samples))
            accepted = rejected = items = 0
            closed: list[WindowResult] = []
            touched: set[tuple[str, str]] = set()
            with self._lock:
                for raw in raw_samples:
                    try:
                        sample = parse_sample(raw)
                    except SampleError as exc:
                        rejected += 1
                        self.metrics.samples.labels("unknown", "rejected").inc()
                        log.debug("rejected sample: %s", exc)
                        continue
                    accepted += 1
                    self.metrics.samples.labels(sample.model, "accepted").inc()
                    primary, candidate = explode(sample, self.contract)
                    items += len(primary) + len(candidate)
                    self._account(primary)
                    self._account(candidate)
                    p = sample.primary
                    self._counter(sample.model, p.version).add(sample.received_at, p.status == 200)
                    touched.add((sample.model, p.version))
                    if sample.candidate is not None:
                        c = sample.candidate
                        self._counter(sample.model, c.version).add(sample.received_at, c.status == 200)
                        touched.add((sample.model, c.version))
                        tracker = self._canaries.setdefault(
                            (sample.model, c.version),
                            CanaryTracker(sample.model, c.version, self.settings.canary),
                        )
                        tracker.add(p, c, primary, candidate)
                    self._maybe_learn(sample.model, p.version, primary)
                    start = time.perf_counter()
                    windows = self._monitor(sample.model, p.version).add(primary)
                    if windows:
                        self.metrics.eval_seconds.labels(windows[-1].kind).observe(
                            time.perf_counter() - start
                        )
                    closed.extend(windows)
            for w in closed:
                self._record(w)
            self._update_burn(touched)
            span.set_attribute("bellwether.windows_closed", len(closed))
        return {
            "accepted": accepted,
            "rejected": rejected,
            "items": items,
            "windows": [{"kind": w.kind, "fired": w.fired, "summary": w.summary} for w in closed],
        }

    def _account(self, items: list[ItemRecord]) -> None:
        for it in items:
            self.metrics.items.labels(it.model, it.version, it.route, str(it.valid).lower()).inc()
            for code in it.violations:
                self.metrics.violations.labels(it.model, it.version, code).inc()

    def _record(self, w: WindowResult) -> None:
        self.metrics.windows.labels(w.model, w.version, w.kind, str(w.fired).lower()).inc()
        if w.kind == "drift":
            self.metrics.drift_min_p.labels(w.model, w.version).set(w.score)
            self.metrics.drift_threshold.labels(w.model, w.version).set(w.threshold)
        else:
            self.metrics.violation_rate.labels(w.model, w.version).set(w.score)
        self.store.record_window(w)
        if w.fired:
            log.warning("incident: %s", w.summary)
            for kind, n in self.store.count_open(w.model).items():
                self.metrics.open_incidents.labels(w.model, kind).set(n)

    def _update_burn(self, touched: set[tuple[str, str]]) -> None:
        now = time.time()
        for model, version in touched:
            counter = self._counter(model, version)
            for label, seconds in (("5m", 300), ("1h", 3600)):
                self.metrics.burn_rate.labels(model, version, label).set(
                    self.slo.burn_rate(counter.error_ratio(now, seconds))
                )

    # ---------------------------------------------------------------- queries
    def verdict(self, model: str, candidate_version: str) -> Verdict:
        with self._lock:
            tracker = self._canaries.get((model, candidate_version))
            verdict = (tracker or CanaryTracker(model, candidate_version, self.settings.canary)).verdict()
        self.metrics.canary_decision.labels(model, candidate_version).set(DECISION_VALUE[verdict.decision])
        key = (model, candidate_version)
        if self._last_decision.get(key) != verdict.decision:
            self._last_decision[key] = verdict.decision
            self.store.save_verdict(verdict.as_dict())
        return verdict

    def reset_canary(self, model: str, candidate_version: str) -> None:
        with self._lock:
            self._canaries.pop((model, candidate_version), None)
            self._last_decision.pop((model, candidate_version), None)

    def status(self, model: str) -> dict[str, Any]:
        b = self.baseline(model)
        return {
            "model": model,
            "baseline": b.summary() if b else None,
            "learning_items": len(self._learning.get(model, [])),
            "baseline_building": model in self._building,
            "open_incidents": self.store.count_open(model),
            "canaries": sorted(c for (m, c) in self._canaries if m == model),
        }

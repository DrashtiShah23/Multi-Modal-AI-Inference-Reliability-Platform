"""Prometheus metrics and OpenTelemetry tracing for the analyzer.

Traces are exported over OTLP only when OTEL_EXPORTER_OTLP_ENDPOINT is set.
Each ingest span links to the gateway's W3C ``traceparent`` so a slow or
alarming request can be followed from the gateway into the analyzer.
"""

from __future__ import annotations

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Link, SpanContext, TraceFlags
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

log = logging.getLogger(__name__)
_configured = False


def setup_tracing(service_name: str = "bellwether-analyzer") -> trace.Tracer:
    global _configured
    if not _configured and os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:  # pragma: no cover - optional extra
            log.warning("OTEL_EXPORTER_OTLP_ENDPOINT set but the otlp extra is not installed")
        else:
            provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            trace.set_tracer_provider(provider)
        _configured = True
    return trace.get_tracer("bellwether.analyzer")


def link_from_traceparent(traceparent: str) -> Link | None:
    """Build a span link from a W3C traceparent header value (``00-<trace>-<span>-<flags>``)."""
    parts = traceparent.split("-")
    if len(parts) != 4 or len(parts[1]) != 32 or len(parts[2]) != 16:
        return None
    try:
        ctx = SpanContext(
            int(parts[1], 16), int(parts[2], 16), is_remote=True, trace_flags=TraceFlags(int(parts[3], 16))
        )
    except ValueError:
        return None
    return Link(ctx) if ctx.is_valid else None


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.samples = Counter(
            "bellwether_analyzer_samples_total", "Gateway samples received", ["model", "outcome"], registry=r
        )
        self.items = Counter(
            "bellwether_analyzer_items_total",
            "Items analyzed",
            ["model", "version", "route", "valid"],
            registry=r,
        )
        self.violations = Counter(
            "bellwether_analyzer_contract_violations_total",
            "Contract violations by code",
            ["model", "version", "code"],
            registry=r,
        )
        self.windows = Counter(
            "bellwether_analyzer_windows_total",
            "Evaluated windows",
            ["model", "version", "kind", "fired"],
            registry=r,
        )
        self.drift_min_p = Gauge(
            "bellwether_analyzer_drift_min_p",
            "Smallest p-value in the last drift window",
            ["model", "version"],
            registry=r,
        )
        self.drift_threshold = Gauge(
            "bellwether_analyzer_drift_threshold",
            "Calibrated min-p alarm threshold",
            ["model", "version"],
            registry=r,
        )
        self.violation_rate = Gauge(
            "bellwether_analyzer_contract_violation_rate",
            "Violation rate in the last contract window",
            ["model", "version"],
            registry=r,
        )
        self.open_incidents = Gauge(
            "bellwether_analyzer_open_incidents", "Open incidents", ["model", "kind"], registry=r
        )
        self.canary_decision = Gauge(
            "bellwether_analyzer_canary_decision",
            "Latest canary decision (1 promote, 0 hold, -1 rollback)",
            ["model", "candidate"],
            registry=r,
        )
        self.burn_rate = Gauge(
            "bellwether_analyzer_error_budget_burn_rate",
            "Error budget burn rate from samples",
            ["model", "version", "window"],
            registry=r,
        )
        self.eval_seconds = Histogram(
            "bellwether_analyzer_window_eval_seconds",
            "Window evaluation time",
            ["kind"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
            registry=r,
        )
        self.baseline_ready = Gauge(
            "bellwether_analyzer_baseline_ready", "1 when a drift baseline is loaded", ["model"], registry=r
        )

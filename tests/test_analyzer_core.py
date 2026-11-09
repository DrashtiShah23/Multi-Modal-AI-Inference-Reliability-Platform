from __future__ import annotations

import numpy as np
import pytest

from bellwether import synth
from bellwether.analyzer import detectors
from bellwether.analyzer.baseline import Baseline
from bellwether.analyzer.canary import CanaryConfig, CanaryTracker
from bellwether.analyzer.contracts import Contract
from bellwether.analyzer.features import FeatureFrame
from bellwether.analyzer.monitor import StreamMonitor
from bellwether.analyzer.records import Exchange, explode, parse_sample
from bellwether.analyzer.slo import SLO, RollingCounter, evaluate_burn
from bellwether.bench import traffic

CONTRACT = Contract.load()


@pytest.fixture(scope="module")
def reference_items(runtime_v1):
    batch = synth.generate(2600, seed=11)
    out = runtime_v1.infer(batch.images, batch.captions)
    return traffic.items_from_batch(batch, out, model="multimodal", version="v1", contract=CONTRACT)


@pytest.fixture(scope="module")
def small_baseline(reference_items):
    frame = FeatureFrame.from_items(reference_items)
    return Baseline.build(
        frame,
        model="multimodal",
        version="v1",
        window_size=200,
        n_compare=1000,
        n_embedding_ref=500,
        n_gamma_windows=120,
        n_calibration_windows=300,
        seed=1,
    )


def _items(runtime, n, seed, mutate=None, version="v1"):
    batch = synth.generate(n, seed=seed)
    if mutate:
        mutate(batch)
    out = runtime.infer(batch.images, batch.captions)
    return traffic.items_from_batch(batch, out, model="multimodal", version=version, contract=CONTRACT)


# ------------------------------------------------------------------ contracts
def test_contract_accepts_clean_items(reference_items):
    assert all(it.valid for it in reference_items[:200])


def test_contract_flags_request_and_response_problems(runtime_v1):
    batch = synth.generate(2, seed=3)
    out = runtime_v1.infer(batch.images, batch.captions)
    req = traffic.request_body(batch.images, ["", "x" * 500], {"client": "smart_tv"})
    bad_out = dict(out)
    bad_out["text_embedding"] = out["text_embedding"] * 3.0
    s = traffic.sample(req, traffic.exchange("v1", 200, 4.0, traffic.response_body(bad_out, "v1")))
    items, _ = explode(parse_sample(s), CONTRACT)
    assert "inputs.text.too_short" in items[0].violations
    assert "inputs.text.too_long" in items[1].violations
    assert "parameters.client.unknown_value" in items[0].violations
    assert "outputs.text_embedding.norm" in items[0].violations
    assert not any(it.valid for it in items)


def test_contract_handles_malformed_and_failed_requests():
    for request, status, code in [
        (None, 400, "request.invalid_json"),
        (
            {
                "inputs": [
                    traffic.tensor("image", np.zeros((1, 48, 48, 3), np.uint8)),
                    traffic.tensor("text", ["a"]),
                ],
                "parameters": {"client": "ios"},
            },
            400,
            "inputs.image.shape",
        ),
        (
            {"inputs": [traffic.tensor("text", ["a"])], "parameters": {"client": "ios"}},
            400,
            "inputs.image.missing",
        ),
        (
            {
                "inputs": [
                    traffic.tensor("image", np.zeros((1, 64, 64, 3), np.float32)),
                    traffic.tensor("text", ["a"]),
                ],
                "parameters": {"client": "ios"},
            },
            400,
            "inputs.image.datatype",
        ),
    ]:
        items, _ = explode(parse_sample(traffic.sample(request, traffic.exchange("v1", status))), CONTRACT)
        assert code in items[0].violations, code
        assert "response.status_4xx" in items[0].violations


# ------------------------------------------------------------------ detectors
def test_detectors_separate_same_and_shifted_distributions():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=4000)
    assert detectors.ks_test(ref, rng.normal(size=400))[1] > 0.01
    assert detectors.ks_test(ref, rng.normal(0.4, 1, size=400))[1] < 1e-6
    edges = detectors.quantile_edges(ref)
    base = detectors.binned_fractions(ref, edges)
    assert detectors.psi(base, detectors.binned_fractions(rng.normal(1, 1, 400), edges)) > 0.25
    _, p_same, _ = detectors.categorical_test({"a": 500, "b": 500}, np.array(["a"] * 200 + ["b"] * 200))
    _, p_new, _ = detectors.categorical_test({"a": 500, "b": 500}, np.array(["a"] * 200 + ["c"] * 200))
    assert p_same > 0.05 and p_new < 1e-10


def test_mmd_gamma_null_is_calibrated():
    rng = np.random.default_rng(1)
    ref, hold = rng.normal(size=(800, 8)), rng.normal(size=(5000, 8))
    mmd = detectors.MMD(ref)
    null = [mmd.statistic(hold[rng.choice(5000, 100, replace=False)]) for _ in range(300)]
    g = detectors.GammaNull.fit(np.array(null))
    fresh = [g.p_value(mmd.statistic(rng.normal(size=(100, 8)))) for _ in range(300)]
    assert 0.0 <= np.mean(np.array(fresh) < 0.05) <= 0.12
    assert g.p_value(mmd.statistic(rng.normal(0.5, 1, size=(100, 8)))) < 1e-4


# ------------------------------------------------------------------ baseline and monitor
def test_baseline_roundtrip(small_baseline):
    copy = Baseline.from_bytes(small_baseline.to_bytes())
    assert copy.threshold == pytest.approx(small_baseline.threshold)
    assert copy.cat_ref == small_baseline.cat_ref
    np.testing.assert_array_equal(
        copy.emb_ref["output.image_embedding"], small_baseline.emb_ref["output.image_embedding"]
    )
    assert 0 < copy.threshold < small_baseline.alpha


def test_clean_window_passes_and_shift_fires(runtime_v1, small_baseline):
    clean = small_baseline.evaluate(FeatureFrame.from_items(_items(runtime_v1, 200, seed=500)))
    assert not clean.fired

    def darken(batch):
        batch.images[:] = (batch.images.astype(np.float32) * 0.6).astype(np.uint8)

    dark = small_baseline.evaluate(FeatureFrame.from_items(_items(runtime_v1, 200, seed=501, mutate=darken)))
    assert dark.fired and dark.kind == "input_drift"
    assert dark.fired_tests[0].feature.startswith("image.")


def test_model_change_on_same_inputs_is_output_drift(model_repos, small_baseline):
    from bellwether.models.runtime import MultimodalRuntime

    regressed = MultimodalRuntime(model_repos["v3"])
    result = small_baseline.evaluate(FeatureFrame.from_items(_items(regressed, 200, seed=502)))
    assert result.fired and result.kind == "output_drift"


def test_stream_monitor_windows(runtime_v1, small_baseline):
    monitor = StreamMonitor("multimodal", "v1", CONTRACT, window_size=200, baseline=small_baseline)
    results = monitor.add(_items(runtime_v1, 200, seed=600))
    assert [r.kind for r in results] == ["contract", "drift"]
    assert not any(r.fired for r in results)

    bad = _items(runtime_v1, 200, seed=601)
    for it in bad[:20]:
        it.violations = ("inputs.text.too_short",)
    results = monitor.add(bad)
    contract = next(r for r in results if r.kind == "contract")
    assert contract.fired and contract.incident_kind == "contract"
    assert contract.details["top_violations"][0]["code"] == "inputs.text.too_short"


# ------------------------------------------------------------------ canary
def _canary(model_repos, candidate: str, n: int = 260, cand_status: int = 200, cand_latency: float = 5.0):
    from bellwether.models.runtime import MultimodalRuntime

    prim_rt, cand_rt = MultimodalRuntime(model_repos["v1"]), MultimodalRuntime(model_repos[candidate])
    batch = synth.generate(n, seed=77)
    po, co = prim_rt.infer(batch.images, batch.captions), cand_rt.infer(batch.images, batch.captions)
    tracker = CanaryTracker("multimodal", candidate, CanaryConfig(min_requests=100, min_pairs=200))
    for i in range(n):
        req = traffic.request_body(batch.images[i : i + 1], batch.captions[i : i + 1], batch.metadata[i])
        p = traffic.exchange("v1", 200, 5.0, traffic.response_body(traffic.slice_outputs(po, i, i + 1), "v1"))
        c_resp = (
            traffic.response_body(traffic.slice_outputs(co, i, i + 1), candidate)
            if cand_status == 200
            else None
        )
        c = traffic.exchange(candidate, cand_status, cand_latency, c_resp)
        s = parse_sample(traffic.sample(req, p, c))
        prim, cand = explode(s, CONTRACT)
        tracker.add(s.primary, s.candidate, prim, cand)
    return tracker.verdict()


def test_canary_promotes_compatible_retrain(model_repos):
    v = _canary(model_repos, "v2")
    assert v.decision == "PROMOTE", v.as_dict()


def test_canary_rolls_back_behavioral_regression(model_repos):
    v = _canary(model_repos, "v3")
    assert v.decision == "ROLLBACK"
    failed = {c.name for c in v.checks if not c.passed}
    assert "image_embedding_cosine_p10" in failed and "similarity_mae" in failed


def test_canary_rolls_back_latency_and_errors(model_repos):
    slow = _canary(model_repos, "v2", cand_latency=40.0)
    assert slow.decision == "ROLLBACK"
    assert {c.name for c in slow.checks if not c.passed} == {"latency_p95_ratio"}
    failing = _canary(model_repos, "v2", cand_status=503)
    assert failing.decision == "ROLLBACK"
    assert "error_budget_burn" in {c.name for c in failing.checks if not c.passed}


def test_canary_holds_without_enough_data():
    tracker = CanaryTracker("multimodal", "v2")
    tracker.add(Exchange("v1", 200, 5.0), Exchange("v2", 200, 5.0), [], [])
    assert tracker.verdict().decision == "HOLD"


# ------------------------------------------------------------------ SLO math
def test_burn_rate_math_matches_sre_workbook():
    slo = SLO(target=0.999, period_days=30)
    assert slo.burn_rate(0.0144) == pytest.approx(14.4)
    # burning at 14.4x for one hour spends 2% of a 30 day budget
    assert slo.budget_consumed(14.4, 1) == pytest.approx(0.02)
    assert slo.budget_consumed(6, 6) == pytest.approx(0.05)


def test_multiwindow_alerting_requires_both_windows():
    slo = SLO(0.999)
    counter = RollingCounter(bucket_seconds=10)
    now = 1_000_000.0
    for t in range(3600):  # one hour at 1 rps, 2% errors in the last 10 minutes only
        ts = now - 3600 + t
        counter.add(ts, ok=not (t >= 3000 and t % 50 == 0))
    alerts = evaluate_burn(counter, slo, now)
    # 5m burn is 20x but the 1h burn is only 20 * 600/3600 = 3.3x: no page, a slow-burn ticket only
    assert [a.policy for a in alerts] == ["slow"]
    assert all(a.severity == "ticket" for a in alerts)
    for t in range(3600):  # the whole next hour at 2% errors
        counter.add(now + t, ok=t % 50 != 0)
    fast = [a for a in evaluate_burn(counter, slo, now + 3600) if a.policy == "fast"]
    assert fast and fast[0].severity == "page"

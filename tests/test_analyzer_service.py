from __future__ import annotations

import gzip
import json
import os
import time
import uuid
from concurrent import futures

import grpc
import numpy as np
import pytest
from fastapi.testclient import TestClient

from bellwether import synth
from bellwether.analyzer.api import create_app
from bellwether.analyzer.grpc_server import AnalyzerServicer
from bellwether.analyzer.service import AnalyzerService, Settings
from bellwether.analyzer.store import Store
from bellwether.bench import traffic
from bellwether.models.runtime import MultimodalRuntime
from bellwether.v1 import analyzer_pb2 as pb
from bellwether.v1 import analyzer_pb2_grpc as pb_grpc

SMALL_BUILD = {
    "n_compare": 1000,
    "n_embedding_ref": 500,
    "n_gamma_windows": 120,
    "n_calibration_windows": 300,
}


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    return obj


def make_samples(runtime, n, seed, version="v1", candidate_runtime=None, candidate_version="v2", mutate=None):
    batch = synth.generate(n, seed=seed)
    if mutate:
        mutate(batch)
    out = runtime.infer(batch.images, batch.captions)
    cand = candidate_runtime.infer(batch.images, batch.captions) if candidate_runtime else None
    samples = []
    for i in range(n):
        req = traffic.request_body(batch.images[i : i + 1], batch.captions[i : i + 1], batch.metadata[i])
        p = traffic.exchange(
            version, 200, 6.0, traffic.response_body(traffic.slice_outputs(out, i, i + 1), version)
        )
        c = None
        if cand is not None:
            c = traffic.exchange(
                candidate_version,
                200,
                6.5,
                traffic.response_body(traffic.slice_outputs(cand, i, i + 1), candidate_version),
            )
        samples.append(traffic.sample(req, p, c))
    return samples


@pytest.fixture()
def service(tmp_path, runtime_v1):
    svc = AnalyzerService(
        Settings(db_url=f"sqlite:///{tmp_path}/a.db", window_size=200, learn_baseline=False)
    )
    batch = synth.generate(2600, seed=21)
    out = runtime_v1.infer(batch.images, batch.captions)
    items = traffic.items_from_batch(batch, out, model="multimodal", version="v1", contract=svc.contract)
    svc.build_baseline("multimodal", "v1", items, seed=3, **SMALL_BUILD)
    return svc


def _swap_channels(batch):
    batch.images[:] = batch.images[..., ::-1]


def test_rest_ingest_opens_and_dedupes_drift_incidents(service, runtime_v1):
    client = TestClient(create_app(service))
    clean = jsonable(make_samples(runtime_v1, 200, seed=40))
    r = client.post("/v1/samples:batch", json={"samples": clean})
    assert r.status_code == 202 and r.json()["accepted"] == 200
    assert client.get("/v1/models/multimodal/incidents").json()["incidents"] == []

    faulty = jsonable(make_samples(runtime_v1, 400, seed=41, mutate=_swap_channels))
    body = gzip.compress(json.dumps(faulty).encode())
    r = client.post(
        "/v1/samples:batch",
        content=body,
        headers={"content-encoding": "gzip", "content-type": "application/json"},
    )
    assert r.status_code == 202
    incidents = client.get("/v1/models/multimodal/incidents").json()["incidents"]
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc["kind"] == "input_drift" and inc["occurrences"] == 2
    assert any(t["feature"] in ("image.red_mean", "image.blue_mean") for t in inc["details"]["fired_tests"])

    fired = client.get("/v1/models/multimodal/windows", params={"fired_only": True}).json()["windows"]
    assert len(fired) == 2
    metrics = client.get("/metrics").text
    assert 'bellwether_analyzer_open_incidents{kind="input_drift",model="multimodal"} 1.0' in metrics
    assert "bellwether_analyzer_drift_min_p" in metrics

    assert client.post(f"/v1/incidents/{inc['id']}:resolve").status_code == 200
    assert client.get("/v1/models/multimodal/incidents").json()["incidents"] == []


def test_rest_rejects_bad_payloads(service):
    client = TestClient(create_app(service))
    assert (
        client.post(
            "/v1/samples:batch", content=b"[", headers={"content-type": "application/json"}
        ).status_code
        == 400
    )
    assert client.post("/v1/samples:batch", json={"samples": "nope"}).status_code == 400
    r = client.post("/v1/samples:batch", json=[{"no": "primary"}, 7])
    assert r.json() == {"accepted": 0, "rejected": 2, "items": 0, "windows": []}


def test_canary_verdict_endpoint(service, runtime_v1, model_repos):
    client = TestClient(create_app(service))
    cand = MultimodalRuntime(model_repos["v2"])
    assert client.get("/v1/models/multimodal/canary/v2/verdict").json()["decision"] == "HOLD"
    client.post(
        "/v1/samples:batch", json=jsonable(make_samples(runtime_v1, 250, seed=50, candidate_runtime=cand))
    )
    verdict = client.get("/v1/models/multimodal/canary/v2/verdict").json()
    assert verdict["decision"] == "PROMOTE", verdict
    assert (
        'bellwether_analyzer_canary_decision{candidate="v2",model="multimodal"} 1.0'
        in client.get("/metrics").text
    )

    bad = MultimodalRuntime(model_repos["v3"])
    client.post(
        "/v1/samples:batch",
        json=jsonable(make_samples(runtime_v1, 250, seed=51, candidate_runtime=bad, candidate_version="v3")),
    )
    assert client.get("/v1/models/multimodal/canary/v3/verdict").json()["decision"] == "ROLLBACK"


def test_grpc_api(service, runtime_v1, model_repos):
    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_AnalyzerServiceServicer_to_server(AnalyzerServicer(service), srv)
    port = srv.add_insecure_port("127.0.0.1:0")
    srv.start()
    try:
        stub = pb_grpc.AnalyzerServiceStub(grpc.insecure_channel(f"127.0.0.1:{port}"))
        cand = MultimodalRuntime(model_repos["v2"])
        mirrored = jsonable(make_samples(runtime_v1, 220, seed=60, candidate_runtime=cand))
        swapped = jsonable(make_samples(runtime_v1, 200, seed=61, mutate=_swap_channels))

        def to_pb(s):
            msg = pb.Sample(
                trace_id=s["trace_id"],
                model=s["model"],
                request_json=json.dumps(s["request"]).encode(),
                primary=pb.Exchange(
                    version="v1",
                    status=200,
                    latency_ms=6.0,
                    response_json=json.dumps(s["primary"]["response"]).encode(),
                ),
            )
            msg.received_at.GetCurrentTime()
            if "candidate" in s:
                msg.candidate.CopyFrom(
                    pb.Exchange(
                        version="v2",
                        status=200,
                        latency_ms=6.5,
                        response_json=json.dumps(s["candidate"]["response"]).encode(),
                    )
                )
            return msg

        stream = mirrored + swapped
        reqs = (
            pb.IngestSamplesRequest(samples=[to_pb(s) for s in stream[i : i + 50]]) for i in range(0, 420, 50)
        )
        ack = stub.IngestSamples(reqs)
        assert ack.accepted == 420 and ack.items == 640 and ack.windows_closed == 4

        verdict = stub.GetCanaryVerdict(
            pb.GetCanaryVerdictRequest(model="multimodal", candidate_version="v2")
        )
        assert verdict.decision == pb.CanaryVerdict.DECISION_PROMOTE
        incidents = stub.ListIncidents(pb.ListIncidentsRequest(model="multimodal", status="open"))
        assert [i.kind for i in incidents.incidents] == ["input_drift"]

        with pytest.raises(grpc.RpcError) as err:
            stub.GetCanaryVerdict(pb.GetCanaryVerdictRequest(model="multimodal"))
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        srv.stop(grace=None)


def test_learning_mode_builds_baseline_from_traffic(tmp_path, runtime_v1):
    svc = AnalyzerService(
        Settings(db_url=f"sqlite:///{tmp_path}/learn.db", window_size=200, baseline_items=2600)
    )
    assert svc.status("multimodal")["baseline"] is None
    svc.ingest(make_samples(runtime_v1, 2600, seed=70))
    deadline = time.time() + 120
    while svc.baseline("multimodal") is None and time.time() < deadline:
        time.sleep(0.5)
    b = svc.baseline("multimodal")
    assert b is not None and b.window_size == 200 and b.n_reference == 2600
    # a fresh service on the same database loads the persisted baseline
    again = AnalyzerService(Settings(db_url=f"sqlite:///{tmp_path}/learn.db", window_size=200))
    assert again.baseline("multimodal").threshold == pytest.approx(b.threshold)


@pytest.mark.postgres
@pytest.mark.skipif(not os.environ.get("BELLWETHER_TEST_PG_URL"), reason="BELLWETHER_TEST_PG_URL not set")
def test_store_on_postgres(runtime_v1):
    model = f"pg-{uuid.uuid4().hex[:8]}"  # unique per run, so a reused database cannot skew counts
    store = Store(os.environ["BELLWETHER_TEST_PG_URL"])
    svc = AnalyzerService(Settings(window_size=200, learn_baseline=False), store=store)
    batch = synth.generate(2600, seed=21)
    out = runtime_v1.infer(batch.images, batch.captions)
    items = traffic.items_from_batch(batch, out, model=model, version="v1", contract=svc.contract)
    svc.build_baseline(model, "v1", items, seed=3, **SMALL_BUILD)
    samples = make_samples(runtime_v1, 400, seed=41, mutate=_swap_channels)
    for s in samples:
        s["model"] = model
    svc.ingest(samples)
    incidents = store.list_incidents(model=model, status="open")
    assert incidents and incidents[0]["kind"] == "input_drift" and incidents[0]["occurrences"] == 2
    assert Store(os.environ["BELLWETHER_TEST_PG_URL"]).latest_baseline(model) is not None

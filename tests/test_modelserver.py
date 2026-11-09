from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from bellwether import EMBED_DIM, synth
from bellwether.kserve import build_infer_request, decode_outputs
from bellwether.modelserver.app import create_app


def _request(n: int = 2, seed: int = 1) -> dict:
    batch = synth.generate(n, seed=seed)
    return build_infer_request(batch.images, batch.captions, request_id="req-1", parameters=batch.metadata[0])


def test_infer_roundtrip(model_repos):
    client = TestClient(create_app(model_repos["v1"], "v1"))
    resp = client.post("/v2/models/multimodal/infer", json=_request(3))
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "v1" and body["id"] == "req-1"
    out = decode_outputs(body)
    assert out["image_embedding"].shape == (3, EMBED_DIM)
    assert out["similarity"].shape == (3,)
    assert np.all(np.abs(out["similarity"]) <= 1.0 + 1e-5)


def test_metadata_and_health(model_repos):
    client = TestClient(create_app(model_repos["v1"], "v1"))
    assert client.get("/v2/health/ready").status_code == 200
    assert client.get("/v2/models/multimodal").json()["inputs"][0]["name"] == "image"
    assert client.get("/v2/models/other").status_code == 404


def test_rejects_bad_requests_with_kserve_errors(model_repos):
    client = TestClient(create_app(model_repos["v1"], "v1"))
    bad_json = client.post(
        "/v2/models/multimodal/infer", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert bad_json.status_code == 400 and "invalid JSON" in bad_json.json()["error"]
    body = _request(1)
    body["inputs"][0]["shape"] = [1, 48, 48, 3]
    body["inputs"][0]["data"] = [0] * (48 * 48 * 3)
    resp = client.post("/v2/models/multimodal/infer", json=body)
    assert resp.status_code == 400 and "expected shape" in resp.json()["error"]


def test_injected_failures_and_metrics(model_repos):
    client = TestClient(create_app(model_repos["v1"], "v2", inject_error_rate=1.0, seed=0))
    resp = client.post("/v2/models/multimodal/infer", json=_request(1))
    assert resp.status_code == 500
    metrics = client.get("/metrics").text
    assert 'modelserver_requests_total{code="500",model="multimodal",version="v2"} 1.0' in metrics

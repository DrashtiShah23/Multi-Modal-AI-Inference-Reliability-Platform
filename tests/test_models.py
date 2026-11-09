from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import onnx
import pytest

from bellwether import EMBED_DIM, MAX_TOKENS, synth
from bellwether.kserve import TensorError, build_infer_request, decode_outputs, decode_tensor, encode_tensor
from bellwether.models import tokenizer
from bellwether.models.runtime import MultimodalRuntime, parse_infer_request


def test_tokenizer_is_deterministic_and_padded():
    ids = tokenizer.encode(["A red Circle", "", "red"])
    assert ids.shape == (3, MAX_TOKENS) and ids.dtype == np.int64
    assert ids[0, 1] == ids[2, 0] == tokenizer.token_id("red")
    assert (ids[1] == 0).all()
    assert (ids[0, 3:] == 0).all()


def test_onnx_models_are_valid(model_repos):
    for name in ("image_encoder", "text_encoder", "similarity"):
        model = onnx.load(model_repos["v1"] / name / "1" / "model.onnx")
        onnx.checker.check_model(model)
        assert model.opset_import[0].version == 17


def test_runtime_outputs_are_normalized_and_aligned(runtime_v1):
    batch = synth.generate(300, seed=5)
    out = runtime_v1.infer(batch.images, batch.captions)
    assert out["image_embedding"].shape == (300, EMBED_DIM)
    assert out["text_embedding"].shape == (300, EMBED_DIM)
    np.testing.assert_allclose(np.linalg.norm(out["image_embedding"], axis=1), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(out["text_embedding"], axis=1), 1.0, atol=1e-4)
    sims = out["image_embedding"] @ out["text_embedding"].T
    np.testing.assert_allclose(np.diag(sims), out["similarity"], atol=1e-5)
    off_diag = (sims.sum() - np.trace(sims)) / (300 * 299)
    assert out["similarity"].mean() > off_diag + 0.3


def test_regressed_variant_reads_bgr(model_repos):
    batch = synth.generate(50, seed=8)
    v1 = MultimodalRuntime(model_repos["v1"]).infer(batch.images, batch.captions)
    v3 = MultimodalRuntime(model_repos["v3"]).infer(batch.images, batch.captions)
    swapped = MultimodalRuntime(model_repos["v1"]).infer(batch.images[..., ::-1].copy(), batch.captions)
    np.testing.assert_allclose(v3["image_embedding"], swapped["image_embedding"], atol=1e-5)
    np.testing.assert_allclose(v3["text_embedding"], v1["text_embedding"], atol=1e-6)


def test_triton_repository_layout(model_repos):
    repo = model_repos["v1"]
    for name in ("image_encoder", "text_encoder", "similarity", "tokenizer", "multimodal"):
        cfg = (repo / name / "config.pbtxt").read_text()
        assert f'name: "{name}"' in cfg
    assert 'platform: "ensemble"' in (repo / "multimodal" / "config.pbtxt").read_text()


def test_triton_tokenizer_matches_local_tokenizer(model_repos, monkeypatch):
    stub = types.ModuleType("triton_python_backend_utils")
    monkeypatch.setitem(sys.modules, "triton_python_backend_utils", stub)
    spec = importlib.util.spec_from_file_location(
        "tok_backend", model_repos["v1"] / "tokenizer" / "1" / "model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    texts = ["a large red circle on a dark background", "café ünïcode 東京 tokens", ""]
    np.testing.assert_array_equal(module._encode(texts), tokenizer.encode(texts))


def test_kserve_roundtrip_and_validation():
    images = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    body = build_infer_request(images, ["a", "b"], request_id="r1", parameters={"client": "ios"})
    parsed = parse_infer_request(body)
    assert parsed.images.shape == (2, 64, 64, 3) and parsed.texts == ["a", "b"]
    assert parsed.parameters == {"client": "ios"} and parsed.request_id == "r1"

    t = encode_tensor("x", np.arange(6, dtype=np.float32).reshape(2, 3))
    assert decode_tensor(t).shape == (2, 3)
    t["shape"] = [4, 3]
    with pytest.raises(TensorError, match="needs 12 values"):
        decode_tensor(t)
    assert decode_outputs({"outputs": [encode_tensor("y", np.ones((1, 2), np.float32))]})["y"].shape == (1, 2)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda b: b["inputs"].pop(1), "missing input"),
        (lambda b: b["inputs"][0].update(datatype="FP32"), "UINT8"),
        (lambda b: b["inputs"][0].update(shape=[1, 32, 128, 3]), "expected shape"),
        (lambda b: b["inputs"][1].update(shape=[2], data=["a", "b"]), "batch mismatch"),
    ],
)
def test_parse_rejects_malformed_requests(mutate, message):
    body = build_infer_request(np.zeros((1, 64, 64, 3), np.uint8), ["x"])
    mutate(body)
    with pytest.raises(TensorError, match=message):
        parse_infer_request(body)

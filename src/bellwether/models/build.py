"""Build the bundled image and text encoders as ONNX models in a Triton model repository.

Architecture
  image_encoder  UINT8 [N,64,64,3] -> two stride-2 conv blocks -> pooled features (83)
                 -> CCA projection (32) -> L2 normalized ``image_embedding``
  tokenizer      BYTES [N,1] -> hashed token ids INT64 [N,32]   (Triton Python backend)
  text_encoder   token ids -> masked mean of a 4096x64 hashed embedding table
                 -> CCA projection (32) -> L2 normalized ``text_embedding``
  similarity     cosine similarity of the two embeddings
  multimodal     Triton ensemble wiring the four models together

The conv filters and token table are random features; the two projection heads
are fit with canonical correlation analysis on synthetic image-caption pairs so
that matching pairs score high similarity. That keeps the models tiny and
reproducible while still giving the reliability layer a real cross-modal signal
to watch. Any ONNX models with the same tensor names can be swapped in.

Variants
  v1  baseline release
  v2  retrained release: projection heads refit on fresh data, conv weights lightly perturbed,
      then rotated onto v1's embedding space (orthogonal Procrustes) to stay index compatible
  v3  regressed release: identical weights to v1 but the export reads images as BGR
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from bellwether import EMBED_DIM, IMAGE_SIZE, MAX_TOKENS, VOCAB_SIZE, synth
from bellwether.models import tokenizer

OPSET = 17
IR_VERSION = 8
TOKEN_DIM = 64
IMG_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMG_STD = np.array([0.25, 0.25, 0.25], dtype=np.float32)
VARIANTS = ("v1", "v2", "v3")


@dataclass
class Weights:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    token_table: np.ndarray
    swap_rgb: bool = False
    img_mu: np.ndarray | None = None
    img_proj: np.ndarray | None = None
    txt_mu: np.ndarray | None = None
    txt_proj: np.ndarray | None = None


def init_weights(seed: int) -> Weights:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, np.sqrt(2 / (3 * 25)), size=(16, 3, 5, 5)).astype(np.float32)
    w2 = rng.normal(0, np.sqrt(2 / (16 * 9)), size=(32, 16, 3, 3)).astype(np.float32)
    table = (rng.normal(0, 1, size=(VOCAB_SIZE, TOKEN_DIM)) / np.sqrt(TOKEN_DIM)).astype(np.float32)
    table[0] = 0.0  # padding row
    return Weights(
        w1=w1,
        b1=rng.normal(0, 0.05, size=16).astype(np.float32),
        w2=w2,
        b2=rng.normal(0, 0.05, size=32).astype(np.float32),
        token_table=table,
    )


def _const(name: str, value: np.ndarray) -> TensorProto:
    return numpy_helper.from_array(np.asarray(value), name=name)


def _l2_normalize(src: str, dst: str, prefix: str) -> tuple[list[onnx.NodeProto], list[TensorProto]]:
    nodes = [
        helper.make_node("ReduceL2", [src], [f"{prefix}_norm"], axes=[1], keepdims=1),
        helper.make_node("Max", [f"{prefix}_norm", f"{prefix}_eps"], [f"{prefix}_norm_safe"]),
        helper.make_node("Div", [src, f"{prefix}_norm_safe"], [dst]),
    ]
    return nodes, [_const(f"{prefix}_eps", np.array(1e-6, dtype=np.float32))]


def _finish(graph: onnx.GraphProto, doc: str) -> onnx.ModelProto:
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="bellwether"
    )
    model.ir_version = IR_VERSION
    model.doc_string = doc
    onnx.checker.check_model(model)
    return model


def image_model(w: Weights, *, with_projection: bool = True) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = [helper.make_node("Cast", ["image"], ["x_f"], to=TensorProto.FLOAT)]
    inits = [
        _const("c255", np.array(255.0, dtype=np.float32)),
        _const("mean", IMG_MEAN),
        _const("std", IMG_STD),
        _const("W1", w.w1),
        _const("B1", w.b1),
        _const("W2", w.w2),
        _const("B2", w.b2),
    ]
    x = "x_f"
    if w.swap_rgb:
        inits.append(_const("bgr_idx", np.array([2, 1, 0], dtype=np.int64)))
        nodes.append(helper.make_node("Gather", ["x_f", "bgr_idx"], ["x_bgr"], axis=3))
        x = "x_bgr"
    nodes += [
        helper.make_node("Div", [x, "c255"], ["x01"]),
        helper.make_node("Sub", ["x01", "mean"], ["xc"]),
        helper.make_node("Div", ["xc", "std"], ["xn"]),
        helper.make_node("Transpose", ["xn"], ["xchw"], perm=[0, 3, 1, 2]),
        helper.make_node("Conv", ["xchw", "W1", "B1"], ["c1"], strides=[2, 2], pads=[2, 2, 2, 2]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("Conv", ["r1", "W2", "B2"], ["c2"], strides=[2, 2], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c2"], ["r2"]),
        helper.make_node("GlobalAveragePool", ["r2"], ["gap2"]),
        helper.make_node("GlobalMaxPool", ["r2"], ["gmp2"]),
        helper.make_node("GlobalAveragePool", ["r1"], ["gap1"]),
        helper.make_node("GlobalAveragePool", ["xchw"], ["gap0"]),
        helper.make_node("Concat", ["gap2", "gmp2", "gap1", "gap0"], ["cat"], axis=1),
        helper.make_node("Flatten", ["cat"], ["feat"], axis=1),
    ]
    inputs = [helper.make_tensor_value_info("image", TensorProto.UINT8, ["N", IMAGE_SIZE, IMAGE_SIZE, 3])]
    if not with_projection:
        out = [helper.make_tensor_value_info("feat", TensorProto.FLOAT, ["N", 83])]
        return _finish(helper.make_graph(nodes, "image_features", inputs, out, inits), "image features")
    assert w.img_mu is not None and w.img_proj is not None
    inits += [
        _const("img_mu", w.img_mu.astype(np.float32)),
        _const("img_proj", w.img_proj.astype(np.float32)),
    ]
    nodes += [
        helper.make_node("Sub", ["feat", "img_mu"], ["feat_c"]),
        helper.make_node("MatMul", ["feat_c", "img_proj"], ["z"]),
    ]
    norm_nodes, norm_inits = _l2_normalize("z", "image_embedding", "img")
    out = [helper.make_tensor_value_info("image_embedding", TensorProto.FLOAT, ["N", EMBED_DIM])]
    graph = helper.make_graph(nodes + norm_nodes, "image_encoder", inputs, out, inits + norm_inits)
    return _finish(graph, "bellwether image encoder")


def text_features(w: Weights, token_ids: np.ndarray) -> np.ndarray:
    emb = w.token_table[token_ids]
    mask = (token_ids > 0).astype(np.float32)[..., None]
    return (emb * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)


def text_model(w: Weights) -> onnx.ModelProto:
    assert w.txt_mu is not None and w.txt_proj is not None
    inits = [
        _const("table", w.token_table),
        _const("zero_i64", np.array(0, dtype=np.int64)),
        _const("axis1", np.array([1], dtype=np.int64)),
        _const("axis2", np.array([2], dtype=np.int64)),
        _const("one_f", np.array(1.0, dtype=np.float32)),
        _const("txt_mu", w.txt_mu.astype(np.float32)),
        _const("txt_proj", w.txt_proj.astype(np.float32)),
    ]
    nodes = [
        helper.make_node("Gather", ["table", "token_ids"], ["e"], axis=0),
        helper.make_node("Greater", ["token_ids", "zero_i64"], ["m_b"]),
        helper.make_node("Cast", ["m_b"], ["m"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["m", "axis2"], ["m3"]),
        helper.make_node("Mul", ["e", "m3"], ["em"]),
        helper.make_node("ReduceSum", ["em", "axis1"], ["s"], keepdims=0),
        helper.make_node("ReduceSum", ["m", "axis1"], ["cnt"], keepdims=1),
        helper.make_node("Max", ["cnt", "one_f"], ["cnt1"]),
        helper.make_node("Div", ["s", "cnt1"], ["tmean"]),
        helper.make_node("Sub", ["tmean", "txt_mu"], ["tc"]),
        helper.make_node("MatMul", ["tc", "txt_proj"], ["tz"]),
    ]
    norm_nodes, norm_inits = _l2_normalize("tz", "text_embedding", "txt")
    graph = helper.make_graph(
        nodes + norm_nodes,
        "text_encoder",
        [helper.make_tensor_value_info("token_ids", TensorProto.INT64, ["N", MAX_TOKENS])],
        [helper.make_tensor_value_info("text_embedding", TensorProto.FLOAT, ["N", EMBED_DIM])],
        inits + norm_inits,
    )
    return _finish(graph, "bellwether text encoder")


def similarity_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["image_embedding", "text_embedding"], ["prod"]),
            helper.make_node("ReduceSum", ["prod", "axis1"], ["similarity"], keepdims=1),
        ],
        "similarity",
        [
            helper.make_tensor_value_info("image_embedding", TensorProto.FLOAT, ["N", EMBED_DIM]),
            helper.make_tensor_value_info("text_embedding", TensorProto.FLOAT, ["N", EMBED_DIM]),
        ],
        [helper.make_tensor_value_info("similarity", TensorProto.FLOAT, ["N", 1])],
        [_const("axis1", np.array([1], dtype=np.int64))],
    )
    return _finish(graph, "cosine similarity of L2 normalized embeddings")


def fit_cca(x: np.ndarray, y: np.ndarray, k: int, reg: float = 1e-3):
    """Regularized CCA. Returns centering vectors and projections scaled by sqrt(canonical corr)."""
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    n = len(x)
    cxx, cyy, cxy = xc.T @ xc / n, yc.T @ yc / n, xc.T @ yc / n
    cxx += reg * np.trace(cxx) / cxx.shape[0] * np.eye(cxx.shape[0])
    cyy += reg * np.trace(cyy) / cyy.shape[0] * np.eye(cyy.shape[0])

    def inv_sqrt(c: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(c)
        return vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T

    wx, wy = inv_sqrt(cxx), inv_sqrt(cyy)
    u, s, vt = np.linalg.svd(wx @ cxy @ wy)
    scale = np.sqrt(s[:k])
    return mx, (wx @ u[:, :k]) * scale, my, (wy @ vt.T[:, :k]) * scale, s[:k]


def _image_features(w: Weights, images: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        image_model(w, with_projection=False).SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out = [sess.run(["feat"], {"image": images[i : i + 512]})[0] for i in range(0, len(images), 512)]
    return np.concatenate(out)


def fit_heads(w: Weights, data_seed: int, n_train: int) -> np.ndarray:
    batch = synth.generate(n_train, seed=data_seed)
    fi = _image_features(w, batch.images)
    ft = text_features(w, tokenizer.encode(batch.captions))
    w.img_mu, w.img_proj, w.txt_mu, w.txt_proj, corr = fit_cca(
        fi.astype(np.float64), ft.astype(np.float64), EMBED_DIM
    )
    return corr


def _projected(w: Weights, images: np.ndarray, token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zi = (_image_features(w, images) - w.img_mu) @ w.img_proj
    zt = (text_features(w, token_ids) - w.txt_mu) @ w.txt_proj
    return zi, zt


def align_to(w: Weights, reference: Weights, anchor_seed: int = 303, n_anchor: int = 4000) -> float:
    """Rotate ``w``'s embedding space onto ``reference`` with orthogonal Procrustes.

    A retrained encoder lands in an arbitrary rotation of the shared space, which
    would invalidate every vector already indexed with the old release. Solving
    min_R ||Z_new R - Z_old|| over orthogonal R on anchor pairs keeps the new
    release backward compatible without changing any cosine similarity within it.
    Returns the mean anchor cosine to the reference after alignment.
    """
    batch = synth.generate(n_anchor, seed=anchor_seed)
    ids = tokenizer.encode(batch.captions)
    new_i, new_t = _projected(w, batch.images, ids)
    old_i, old_t = _projected(reference, batch.images, ids)
    a, b = np.vstack([new_i, new_t]), np.vstack([old_i, old_t])
    u, _, vt = np.linalg.svd(a.T @ b)
    rot = u @ vt
    w.img_proj = w.img_proj @ rot
    w.txt_proj = w.txt_proj @ rot
    aligned = a @ rot
    cos = (aligned * b).sum(1) / (np.linalg.norm(aligned, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return float(cos.mean())


def make_variant(name: str, n_train: int, reference: Weights | None = None) -> Weights:
    """Build one release. ``reference`` (the v1 weights) is reused when given to avoid refitting it."""
    if name in ("v2", "v3") and reference is None:
        reference = make_variant("v1", n_train)
    if name == "v3":
        assert reference is not None
        return replace(reference, swap_rgb=True)
    base = init_weights(seed=7)
    if name == "v1":
        fit_heads(base, data_seed=101, n_train=n_train)
        return base
    if name == "v2":
        rng = np.random.default_rng(2)
        for attr in ("w1", "b1", "w2", "b2"):
            arr = getattr(base, attr)
            setattr(base, attr, (arr + rng.normal(0, 0.02 * arr.std(), size=arr.shape)).astype(np.float32))
        fit_heads(base, data_seed=202, n_train=n_train)
        assert reference is not None
        align_to(base, reference)
        return base
    raise ValueError(f"unknown variant {name!r}")


# ---------------------------------------------------------------- Triton repository


def _config(name: str, body: str) -> str:
    return f'name: "{name}"\n{body.strip()}\n'


TRITON_CONFIGS = {
    "image_encoder": f"""
platform: "onnxruntime_onnx"
max_batch_size: 64
input [ {{ name: "image" data_type: TYPE_UINT8 dims: [ {IMAGE_SIZE}, {IMAGE_SIZE}, 3 ] }} ]
output [ {{ name: "image_embedding" data_type: TYPE_FP32 dims: [ {EMBED_DIM} ] }} ]
dynamic_batching {{ preferred_batch_size: [ 8, 16, 32 ] max_queue_delay_microseconds: 2000 }}
instance_group [ {{ kind: KIND_GPU count: 1 }} ]
""",
    "text_encoder": f"""
platform: "onnxruntime_onnx"
max_batch_size: 64
input [ {{ name: "token_ids" data_type: TYPE_INT64 dims: [ {MAX_TOKENS} ] }} ]
output [ {{ name: "text_embedding" data_type: TYPE_FP32 dims: [ {EMBED_DIM} ] }} ]
dynamic_batching {{ max_queue_delay_microseconds: 2000 }}
instance_group [ {{ kind: KIND_GPU count: 1 }} ]
""",
    "similarity": f"""
platform: "onnxruntime_onnx"
max_batch_size: 64
input [
  {{ name: "image_embedding" data_type: TYPE_FP32 dims: [ {EMBED_DIM} ] }},
  {{ name: "text_embedding" data_type: TYPE_FP32 dims: [ {EMBED_DIM} ] }}
]
output [ {{ name: "similarity" data_type: TYPE_FP32 dims: [ 1 ] }} ]
instance_group [ {{ kind: KIND_CPU count: 1 }} ]
""",
    "tokenizer": f"""
backend: "python"
max_batch_size: 64
input [ {{ name: "text" data_type: TYPE_STRING dims: [ 1 ] }} ]
output [ {{ name: "token_ids" data_type: TYPE_INT64 dims: [ {MAX_TOKENS} ] }} ]
instance_group [ {{ kind: KIND_CPU count: 2 }} ]
""",
    "multimodal": f"""
platform: "ensemble"
max_batch_size: 64
input [
  {{ name: "image" data_type: TYPE_UINT8 dims: [ {IMAGE_SIZE}, {IMAGE_SIZE}, 3 ] }},
  {{ name: "text" data_type: TYPE_STRING dims: [ 1 ] }}
]
output [
  {{ name: "image_embedding" data_type: TYPE_FP32 dims: [ {EMBED_DIM} ] }},
  {{ name: "text_embedding" data_type: TYPE_FP32 dims: [ {EMBED_DIM} ] }},
  {{ name: "similarity" data_type: TYPE_FP32 dims: [ 1 ] }}
]
ensemble_scheduling {{
  step [
    {{ model_name: "tokenizer" model_version: -1
      input_map {{ key: "text" value: "text" }}
      output_map {{ key: "token_ids" value: "token_ids" }} }},
    {{ model_name: "text_encoder" model_version: -1
      input_map {{ key: "token_ids" value: "token_ids" }}
      output_map {{ key: "text_embedding" value: "text_embedding" }} }},
    {{ model_name: "image_encoder" model_version: -1
      input_map {{ key: "image" value: "image" }}
      output_map {{ key: "image_embedding" value: "image_embedding" }} }},
    {{ model_name: "similarity" model_version: -1
      input_map {{ key: "image_embedding" value: "image_embedding" }}
      input_map {{ key: "text_embedding" value: "text_embedding" }}
      output_map {{ key: "similarity" value: "similarity" }} }}
  ]
}}
""",
}

TOKENIZER_BACKEND = '''"""Triton Python backend tokenizer. Generated by bellwether.models.build; mirrors
bellwether/models/tokenizer.py so Triton and the local server emit identical ids."""

import re
import zlib

import numpy as np
import triton_python_backend_utils as pb_utils

VOCAB_SIZE = {vocab}
MAX_TOKENS = {max_tokens}
_TOKEN_RE = re.compile(r"\\w+", re.UNICODE)


def _encode(texts):
    ids = np.zeros((len(texts), MAX_TOKENS), dtype=np.int64)
    for i, text in enumerate(texts):
        toks = _TOKEN_RE.findall(text.lower())[:MAX_TOKENS]
        for j, tok in enumerate(toks):
            ids[i, j] = zlib.crc32(tok.encode("utf-8")) % (VOCAB_SIZE - 1) + 1
    return ids


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "text").as_numpy().reshape(-1)
            texts = [t.decode("utf-8", "replace") if isinstance(t, bytes) else str(t) for t in raw]
            out = pb_utils.Tensor("token_ids", _encode(texts))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
'''


def write_repository(w: Weights, out: Path, version: int = 1) -> None:
    """Write a Triton model repository. The local server reads the same layout.

    Triton reports the numeric version directory as ``model_version``; publishing
    release N under version N keeps the gateway's release labels distinct.
    """
    if out.exists():
        shutil.rmtree(out)
    models = {
        "image_encoder": image_model(w),
        "text_encoder": text_model(w),
        "similarity": similarity_model(),
    }
    for name, body in TRITON_CONFIGS.items():
        (out / name / str(version)).mkdir(parents=True, exist_ok=True)
        (out / name / "config.pbtxt").write_text(_config(name, body))
        if name in models:
            onnx.save(models[name], out / name / str(version) / "model.onnx")
    (out / "tokenizer" / str(version) / "model.py").write_text(
        TOKENIZER_BACKEND.format(vocab=VOCAB_SIZE, max_tokens=MAX_TOKENS)
    )
    (out / "multimodal" / str(version) / ".gitkeep").write_text("")


def evaluate_retrieval(repo: Path, n: int = 2000, seed: int = 999) -> dict[str, float]:
    """Held-out check that the projection heads align the two modalities."""
    from bellwether.models.runtime import MultimodalRuntime

    rt = MultimodalRuntime(repo)
    batch = synth.generate(n, seed=seed)
    out = rt.infer(batch.images, batch.captions)
    ie, te = out["image_embedding"], out["text_embedding"]
    sims = ie @ te.T
    matched = float(np.diag(sims).mean())
    off = float((sims.sum() - np.trace(sims)) / (n * n - n))
    # image -> caption retrieval among 100 candidates (chunks of 100)
    hits = 0
    for start in range(0, n, 100):
        block = sims[start : start + 100, start : start + 100]
        hits += int((block.argmax(axis=1) == np.arange(len(block))).sum())
    return {
        "matched_similarity_mean": round(matched, 4),
        "mismatched_similarity_mean": round(off, 4),
        "top1_retrieval_at_100": round(hits / n, 4),
        "heldout_pairs": n,
    }


def build(out_dir: Path, variants: tuple[str, ...] = VARIANTS, n_train: int = 12000) -> dict[str, dict]:
    cards: dict[str, dict] = {}
    reference = make_variant("v1", n_train=n_train)
    for name in variants:
        w = reference if name == "v1" else make_variant(name, n_train=n_train, reference=reference)
        repo = out_dir / name
        write_repository(w, repo, version=VARIANTS.index(name) + 1)
        card = {"variant": name, "train_pairs": n_train, "embedding_dim": EMBED_DIM, "swap_rgb": w.swap_rgb}
        card.update(evaluate_retrieval(repo))
        (repo / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
        cards[name] = card
    return cards


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the bundled ONNX encoders as Triton model repositories"
    )
    parser.add_argument("--out", type=Path, default=Path("build/models"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--train-pairs", type=int, default=12000)
    args = parser.parse_args(argv)
    cards = build(args.out, tuple(args.variants), n_train=args.train_pairs)
    print(json.dumps(cards, indent=2))


if __name__ == "__main__":
    main()

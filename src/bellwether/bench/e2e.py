"""End-to-end run of the whole stack as separate processes.

Starts five model servers (v1 primary; v2 compatible retrain; v3 regressed;
v2 with +40 ms latency; v2 with 5% errors), the analyzer and the Go gateway,
then for each candidate replays synthetic traffic through the gateway with 10%
mirroring and reads the analyzer's canary verdict. A final phase sends
traffic from a client that swapped RGB to BGR and checks that an input drift
incident opens. Results go to bench/results/e2e.json.

  make models gateway && python -m bellwether.bench.e2e
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import httpx

from bellwether.kserve import build_infer_request
from bellwether.synth import generate

PY = sys.executable
CANDIDATES = [
    # label, repository variant, injected latency ms, injected error rate, expected decision
    ("v2", "v2", 0, 0.0, "PROMOTE"),
    ("v3", "v3", 0, 0.0, "ROLLBACK"),
    ("v2-slow", "v2", 40, 0.0, "ROLLBACK"),
    ("v2-errors", "v2", 0, 0.05, "ROLLBACK"),
]


def wait_ready(url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"{url} not ready after {timeout}s")


def bodies(n: int, seed: int, swap: bool = False) -> list[bytes]:
    batch = generate(n, seed=seed)
    if swap:
        batch.images = batch.images[..., ::-1].copy()
    return [
        json.dumps(
            build_infer_request(
                batch.images[i : i + 1], batch.captions[i : i + 1], parameters=batch.metadata[i]
            )
        ).encode()
        for i in range(n)
    ]


def drive(url: str, payloads: list[bytes], concurrency: int = 16) -> dict[str, float]:
    client = httpx.Client(timeout=30, limits=httpx.Limits(max_connections=concurrency))
    lat: list[float] = []
    errors = 0

    def one(body: bytes) -> None:
        nonlocal errors
        t = time.perf_counter()
        r = client.post(url, content=body, headers={"content-type": "application/json"})
        lat.append((time.perf_counter() - t) * 1000)
        if r.status_code != 200:
            errors += 1

    start = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(one, payloads))
    elapsed = time.perf_counter() - start
    lat.sort()
    return {
        "requests": len(payloads),
        "errors": errors,
        "rps": round(len(payloads) / elapsed, 1),
        "p50_ms": round(lat[len(lat) // 2], 2),
        "p99_ms": round(lat[int(len(lat) * 0.99) - 1], 2),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", type=Path, default=Path("build/models"))
    parser.add_argument("--gateway-bin", type=Path, default=Path("build/bin/gateway"))
    parser.add_argument("--requests", type=int, default=2500, help="requests per canary phase")
    parser.add_argument("--out", type=Path, default=Path("bench/results"))
    args = parser.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="bellwether-e2e-"))
    db = f"sqlite:///{work}/analyzer.db"
    env = {
        **os.environ,
        "BELLWETHER_DB_URL": db,
        "BELLWETHER_LOG_LEVEL": "WARNING",
        "BELLWETHER_LEARN_BASELINE": "false",
    }
    procs: list[subprocess.Popen] = []
    logs = ExitStack()

    def spawn(cmd: list[str], log: str, extra_env: dict | None = None) -> subprocess.Popen:
        stderr = logs.enter_context(open(work / log, "w"))  # noqa: SIM115 - closed by the ExitStack
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr, env={**env, **(extra_env or {})})
        procs.append(p)
        return p

    results: dict = {"phases": []}
    try:
        print("building drift baseline ...", flush=True)
        subprocess.run(
            [
                PY,
                "-m",
                "bellwether.analyzer",
                "build-baseline",
                "--model-repository",
                str(args.models / "v1"),
            ],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        ports = {"v1": 8101}
        spawn(
            [
                PY,
                "-m",
                "bellwether.modelserver",
                "--repository",
                str(args.models / "v1"),
                "--version",
                "v1",
                "--port",
                "8101",
            ],
            "ms-v1.log",
        )
        for i, (label, variant, lat_ms, err, _) in enumerate(CANDIDATES):
            ports[label] = 8102 + i
            spawn(
                [
                    PY,
                    "-m",
                    "bellwether.modelserver",
                    "--repository",
                    str(args.models / variant),
                    "--version",
                    label,
                    "--port",
                    str(ports[label]),
                    "--inject-latency-ms",
                    str(lat_ms),
                    "--inject-error-rate",
                    str(err),
                ],
                f"ms-{label}.log",
            )
        spawn(
            [PY, "-m", "bellwether.analyzer", "serve", "--port", "8110", "--grpc-port", "50071"],
            "analyzer.log",
        )
        for port in ports.values():
            wait_ready(f"http://127.0.0.1:{port}/v2/health/ready")
        wait_ready("http://127.0.0.1:8110/healthz")

        def gateway(candidate: str | None, sample_fraction: float) -> subprocess.Popen:
            cmd = [
                str(args.gateway_bin),
                "-listen",
                "127.0.0.1:8180",
                "-admin-listen",
                "127.0.0.1:9191",
                "-primary-url",
                "http://127.0.0.1:8101",
                "-primary-version",
                "v1",
                "-analyzer-url",
                "http://127.0.0.1:8110/v1/samples:batch",
                "-mirror-fraction",
                "0.1",
                "-sample-fraction",
                str(sample_fraction),
                "-sample-flush",
                "200ms",
                "-log-level",
                "warn",
            ]
            if candidate:
                cmd += [
                    "-candidate-url",
                    f"http://127.0.0.1:{ports[candidate]}",
                    "-candidate-version",
                    candidate,
                ]
            p = spawn(cmd, f"gateway-{candidate or 'none'}.log")
            wait_ready("http://127.0.0.1:9191/readyz")
            return p

        infer_url = "http://127.0.0.1:8180/v2/models/multimodal/infer"
        for i, (label, _, lat_ms, err, expected) in enumerate(CANDIDATES):
            gw = gateway(label, sample_fraction=0.0)
            traffic_stats = drive(infer_url, bodies(args.requests, seed=20_000 + i))
            gw.terminate()
            gw.wait(timeout=30)  # graceful shutdown drains queued samples
            verdict = httpx.get(f"http://127.0.0.1:8110/v1/models/multimodal/canary/{label}/verdict").json()
            phase = {
                "candidate": label,
                "injected_latency_ms": lat_ms,
                "injected_error_rate": err,
                "expected": expected,
                "decision": verdict["decision"],
                "traffic": traffic_stats,
                "checks": verdict["checks"],
            }
            results["phases"].append(phase)
            print(
                f"{label:10s} expected {expected:8s} got {verdict['decision']:8s} "
                f"pairs={verdict['checks'][2]['samples']} client p50={traffic_stats['p50_ms']} ms",
                flush=True,
            )

        gw = gateway(None, sample_fraction=1.0)
        drive(infer_url, bodies(400, seed=30_000))
        drive(infer_url, bodies(800, seed=30_001, swap=True))
        gw.terminate()
        gw.wait(timeout=30)
        incidents = httpx.get("http://127.0.0.1:8110/v1/models/multimodal/incidents").json()["incidents"]
        results["drift_phase"] = {
            "clean_requests": 400,
            "bgr_requests": 800,
            "incidents": [
                {k: i[k] for k in ("kind", "severity", "summary", "occurrences")} for i in incidents
            ],
        }
        print("drift phase incidents:", [(i["kind"], i["occurrences"]) for i in incidents], flush=True)
        metrics = httpx.get("http://127.0.0.1:8110/metrics").text
        results["analyzer_metrics_sample"] = [
            ln for ln in metrics.splitlines() if ln.startswith("bellwether_analyzer_canary_decision")
        ]
        results["all_expected"] = all(p["decision"] == p["expected"] for p in results["phases"]) and any(
            i["kind"] == "input_drift" for i in incidents
        )
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        logs.close()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "e2e.json").write_text(json.dumps(results, indent=2) + "\n")
    print("all expected outcomes:", results.get("all_expected"))
    shutil.rmtree(work, ignore_errors=True)
    if not results.get("all_expected"):
        sys.exit(1)


if __name__ == "__main__":
    main()

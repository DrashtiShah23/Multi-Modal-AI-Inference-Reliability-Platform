"""Gateway overhead: open-loop load straight to a model endpoint vs through the gateway.

Both upstreams are Go stubs with a fixed 2 ms service time so the measurement
isolates the gateway (body read, proxying, trace propagation, metrics, 10%
mirroring to a candidate and 5% sampling to an analyzer sink). Payloads are
real 64x64 KServe requests (~40 KB of JSON).

  make gateway && python -m bellwether.bench.overhead --rates 500 1000 2000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from bellwether.bench.e2e import bodies, wait_ready


def loadgen(bin_dir: Path, url: str, body: Path, rate: int, seconds: int) -> dict:
    out = subprocess.run(
        [
            str(bin_dir / "loadgen"),
            "-url",
            url,
            "-body",
            str(body),
            "-rate",
            str(rate),
            "-duration",
            f"{seconds}s",
            "-warmup",
            "3s",
            "-workers",
            "512",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bin", type=Path, default=Path("build/bin"))
    parser.add_argument("--rates", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("bench/results"))
    args = parser.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="bellwether-overhead-"))
    body = work / "request.json"
    body.write_bytes(bodies(1, seed=7)[0])
    procs = [
        subprocess.Popen([str(args.bin / "stubmodel"), "-listen", "127.0.0.1:8201", "-delay", "2ms"]),
        subprocess.Popen(
            [str(args.bin / "stubmodel"), "-listen", "127.0.0.1:8202", "-delay", "2ms", "-version", "v2"]
        ),
        subprocess.Popen([str(args.bin / "stubmodel"), "-listen", "127.0.0.1:8203"]),
        subprocess.Popen(
            [
                str(args.bin / "gateway"),
                "-listen",
                "127.0.0.1:8280",
                "-admin-listen",
                "127.0.0.1:9281",
                "-primary-url",
                "http://127.0.0.1:8201",
                "-candidate-url",
                "http://127.0.0.1:8202",
                "-mirror-fraction",
                "0.1",
                "-sample-fraction",
                "0.05",
                "-analyzer-url",
                "http://127.0.0.1:8203/v1/samples:batch",
                "-log-level",
                "error",
                "-max-inflight",
                "4096",
            ]
        ),
        subprocess.Popen(
            [
                str(args.bin / "gateway"),
                "-listen",
                "127.0.0.1:8281",
                "-admin-listen",
                "127.0.0.1:9282",
                "-primary-url",
                "http://127.0.0.1:8201",
                "-mirror-fraction",
                "0",
                "-sample-fraction",
                "0",
                "-log-level",
                "error",
                "-max-inflight",
                "4096",
            ]
        ),
    ]
    rows = []
    try:
        wait_ready("http://127.0.0.1:8201/v2/health/ready")
        wait_ready("http://127.0.0.1:9281/readyz")
        wait_ready("http://127.0.0.1:9282/readyz")
        path = "/v2/models/multimodal/infer"
        for rate in args.rates:
            direct = loadgen(args.bin, f"http://127.0.0.1:8201{path}", body, rate, args.seconds)
            time.sleep(1)
            proxy_only = loadgen(args.bin, f"http://127.0.0.1:8281{path}", body, rate, args.seconds)
            time.sleep(1)
            via = loadgen(args.bin, f"http://127.0.0.1:8280{path}", body, rate, args.seconds)
            time.sleep(1)
            row = {
                "rate": rate,
                "direct": direct,
                "gateway_proxy_only": proxy_only,
                "gateway_full": via,
                "proxy_only_added_p50_ms": round(proxy_only["p50_ms"] - direct["p50_ms"], 3),
                "proxy_only_added_p99_ms": round(proxy_only["p99_ms"] - direct["p99_ms"], 3),
                "full_added_p50_ms": round(via["p50_ms"] - direct["p50_ms"], 3),
                "full_added_p99_ms": round(via["p99_ms"] - direct["p99_ms"], 3),
            }
            rows.append(row)
            print(
                f"{rate:5d} rps  direct p50/p99 {direct['p50_ms']:.2f}/{direct['p99_ms']:.2f} ms | "
                f"proxy only +{row['proxy_only_added_p50_ms']:.2f}/+{row['proxy_only_added_p99_ms']:.2f} | "
                f"mirror+sampling +{row['full_added_p50_ms']:.2f}/+{row['full_added_p99_ms']:.2f} | "
                f"errors {via['errors']} dropped {via['dropped']}",
                flush=True,
            )
        metrics = subprocess.run(
            ["curl", "-s", "http://127.0.0.1:9281/metrics"], capture_output=True, text=True
        ).stdout
    finally:
        for p in reversed(procs):  # gateway first, so it can drain samples to a live sink
            p.terminate()
            p.wait(timeout=30)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gateway_overhead.json").write_text(
        json.dumps(
            {
                "upstream_service_time_ms": 2,
                "mirror_fraction": 0.1,
                "sample_fraction": 0.05,
                "payload_bytes": body.stat().st_size,
                "rows": rows,
                "gateway_counters": [
                    ln
                    for ln in metrics.splitlines()
                    if ln.startswith(
                        (
                            "bellwether_gateway_mirror_total",
                            "bellwether_gateway_samples_total",
                            "bellwether_gateway_shed_total",
                        )
                    )
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

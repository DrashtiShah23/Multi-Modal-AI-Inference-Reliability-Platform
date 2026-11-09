"""Fault-injection benchmark: detection rate and false alarm rate.

Protocol
  1. Fit a baseline on 6,000 clean requests (seed 0). Its alarm threshold is
     calibrated on the reference data only, never on trial data.
  2. Run N clean trials and N trials per fault, each on fresh seeds. A trial
     streams 400-request chunks into a new StreamMonitor until it has closed one
     contract window and one drift window (at most 3 chunks).
  3. A fault trial is detected if any window fires; a clean trial that fires is
     a false alarm. Fault magnitude and affected share are random per trial.

  python -m bellwether.bench.run_faults --models build/models --trials 30 --clean 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bellwether import MODEL_NAME
from bellwether.analyzer.baseline import Baseline
from bellwether.analyzer.contracts import Contract
from bellwether.analyzer.features import FeatureFrame
from bellwether.analyzer.monitor import StreamMonitor
from bellwether.analyzer.records import explode, parse_sample
from bellwether.bench import faults, traffic
from bellwether.models.runtime import MultimodalRuntime
from bellwether.synth import generate

WINDOW = 400
_state: dict = {}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def build_baseline(models: Path, contract: Contract, n: int = 6000) -> Baseline:
    rt = MultimodalRuntime(models / "v1", threads=os.cpu_count())
    batch = generate(n, seed=0)
    items = traffic.items_from_batch(
        batch, rt.infer(batch.images, batch.captions), model=MODEL_NAME, version="v1", contract=contract
    )
    return Baseline.build(FeatureFrame.from_items(items), model=MODEL_NAME, version="v1", window_size=WINDOW)


def _init(models: str, baseline_bytes: bytes) -> None:
    _state["contract"] = Contract.load()
    _state["baseline"] = Baseline.from_bytes(baseline_bytes)
    _state["rts"] = faults.Runtimes(
        MultimodalRuntime(Path(models) / "v1"), MultimodalRuntime(Path(models) / "v3")
    )


def run_trial(task: tuple[str, int]) -> dict:
    fault_name, trial = task
    contract, baseline, rts = _state["contract"], _state["baseline"], _state["rts"]
    fault = next((f for f in faults.FAULTS if f.name == fault_name), None)
    apply = fault.apply if fault else faults.clean
    rng = np.random.default_rng([zlib.crc32(fault_name.encode()), trial])  # stable across processes
    monitor = StreamMonitor(MODEL_NAME, "v1", contract, window_size=WINDOW, baseline=baseline)
    kinds_seen: set[str] = set()
    fired: list[str] = []
    start = time.perf_counter()
    for c in range(3):
        seed = 1_000_000 + 10 * trial + c + (0 if fault else 500_000)
        samples = apply(generate(WINDOW, seed=seed), rng, rts)
        items = [it for s in samples for it in explode(parse_sample(s), contract)[0]]
        for w in monitor.add(items):
            kinds_seen.add(w.kind)
            if w.fired:
                fired.append(w.incident_kind or w.kind)
        if {"contract", "drift"} <= kinds_seen:
            break
    return {
        "fault": fault_name,
        "trial": trial,
        "detected": bool(fired),
        "signals": sorted(set(fired)),
        "windows": sorted(kinds_seen),
        "seconds": time.perf_counter() - start,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", type=Path, default=Path("build/models"))
    parser.add_argument("--trials", type=int, default=30, help="trials per fault type")
    parser.add_argument("--clean", type=int, default=300, help="clean control trials")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path, default=Path("bench/results"))
    args = parser.parse_args(argv)

    contract = Contract.load()
    t0 = time.perf_counter()
    baseline = build_baseline(args.models, contract)
    build_seconds = time.perf_counter() - t0

    tasks = [("clean", t) for t in range(args.clean)]
    tasks += [(f.name, t) for f in faults.FAULTS for t in range(args.trials)]
    with ProcessPoolExecutor(
        args.workers, initializer=_init, initargs=(str(args.models), baseline.to_bytes())
    ) as pool:
        results = list(pool.map(run_trial, tasks, chunksize=8))

    clean = [r for r in results if r["fault"] == "clean"]
    fa = sum(r["detected"] for r in clean)
    per_fault = []
    for f in faults.FAULTS:
        rs = [r for r in results if r["fault"] == f.name]
        k = sum(r["detected"] for r in rs)
        signals: dict[str, int] = {}
        for r in rs:
            for s in r["signals"]:
                signals[s] = signals.get(s, 0) + 1
        per_fault.append(
            {
                "fault": f.name,
                "description": f.description,
                "expected_signal": f.expected,
                "trials": len(rs),
                "detected": k,
                "detection_rate": k / len(rs),
                "signals": signals,
            }
        )
    total = sum(p["trials"] for p in per_fault)
    detected = sum(p["detected"] for p in per_fault)
    summary = {
        "window_size": WINDOW,
        "baseline": {**baseline.summary(), "build_seconds": round(build_seconds, 1)},
        "clean_trials": len(clean),
        "false_alarms": fa,
        "false_alarm_rate": fa / len(clean),
        "false_alarm_rate_95ci": wilson(fa, len(clean)),
        "fault_trials": total,
        "detected": detected,
        "detection_rate": detected / total,
        "detection_rate_95ci": wilson(detected, total),
        "per_fault": per_fault,
        "mean_trial_seconds": float(np.mean([r["seconds"] for r in results])),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fault_benchmark.json").write_text(
        json.dumps({"summary": summary, "trials": results}, indent=2) + "\n"
    )
    lines = [
        "# Fault-injection benchmark",
        "",
        f"Window {WINDOW} requests; baseline threshold calibrated on reference data at alpha "
        f"{baseline.alpha} (min-p threshold {baseline.threshold:.2e}).",
        "",
        f"* Detection: **{detected}/{total} = {detected / total:.1%}** "
        f"(95% CI {summary['detection_rate_95ci'][0]:.1%} to {summary['detection_rate_95ci'][1]:.1%})",
        f"* False alarms on clean traffic: **{fa}/{len(clean)} = {fa / len(clean):.1%}** "
        f"(95% CI {summary['false_alarm_rate_95ci'][0]:.1%} to {summary['false_alarm_rate_95ci'][1]:.1%})",
        "",
        "| Fault | What it simulates | Detected | Signals |",
        "|---|---|---|---|",
    ]
    for p in per_fault:
        sig = ", ".join(f"{k} {v}" for k, v in sorted(p["signals"].items(), key=lambda kv: -kv[1]))
        lines.append(
            f"| {p['fault']} | {p['description']} | {p['detected']}/{p['trials']} | {sig or 'none'} |"
        )
    (args.out / "fault_benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

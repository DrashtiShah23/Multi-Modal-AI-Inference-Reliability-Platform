"""Analyzer entry point.

bellwether-analyzer serve            REST on :8010 and gRPC on :50051
bellwether-analyzer build-baseline   fit a drift baseline from synthetic reference traffic
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import uvicorn

from bellwether import MODEL_NAME
from bellwether.analyzer.service import AnalyzerService, Settings


def _serve(args: argparse.Namespace) -> None:
    from bellwether.analyzer.api import create_app
    from bellwether.analyzer.grpc_server import serve as serve_grpc

    service = AnalyzerService(Settings.from_env())
    grpc_server = serve_grpc(service, f"[::]:{args.grpc_port}")
    try:
        uvicorn.run(
            create_app(service), host=args.host, port=args.port, log_level="warning", access_log=False
        )
    finally:
        grpc_server.stop(grace=5)


def _build_baseline(args: argparse.Namespace) -> None:
    from bellwether.bench.traffic import items_from_batch
    from bellwether.models.runtime import MultimodalRuntime
    from bellwether.synth import generate

    service = AnalyzerService(Settings.from_env())
    runtime = MultimodalRuntime(args.model_repository, threads=os.cpu_count())
    batch = generate(args.items, seed=args.seed)
    outputs = runtime.infer(batch.images, batch.captions)
    items = items_from_batch(
        batch, outputs, model=args.model, version=args.version, contract=service.contract
    )
    baseline = service.build_baseline(args.model, args.version, items, seed=args.seed)
    print(
        {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in baseline.summary().items()
            if k != "calibration"
        }
    )
    print("calibration:", {k: float(np.round(v, 8)) for k, v in baseline.calibration.items()})


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=os.environ.get("BELLWETHER_LOG_LEVEL", "INFO"),
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    )
    parser = argparse.ArgumentParser(prog="bellwether-analyzer")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the REST and gRPC APIs")
    serve.add_argument("--host", default=os.environ.get("BELLWETHER_HOST", "0.0.0.0"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("BELLWETHER_PORT", "8010")))
    serve.add_argument("--grpc-port", type=int, default=int(os.environ.get("BELLWETHER_GRPC_PORT", "50051")))
    serve.set_defaults(func=_serve)
    build = sub.add_parser("build-baseline", help="fit and store a baseline from synthetic reference traffic")
    build.add_argument(
        "--model-repository", default=os.environ.get("BELLWETHER_MODEL_REPOSITORY", "build/models/v1")
    )
    build.add_argument("--model", default=MODEL_NAME)
    build.add_argument("--version", default="v1")
    build.add_argument("--items", type=int, default=6000)
    build.add_argument("--seed", type=int, default=0)
    build.set_defaults(func=_build_baseline)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

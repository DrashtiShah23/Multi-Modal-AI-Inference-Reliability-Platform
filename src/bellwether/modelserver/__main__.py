from __future__ import annotations

import argparse
import os

import uvicorn

from bellwether.modelserver.app import create_app


def main(argv: list[str] | None = None) -> None:
    env = os.environ.get
    parser = argparse.ArgumentParser(description="Serve the multimodal ensemble over KServe v2 HTTP")
    parser.add_argument("--repository", default=env("BELLWETHER_MODEL_REPOSITORY", "build/models/v1"))
    parser.add_argument("--version", default=env("BELLWETHER_MODEL_VERSION", "v1"))
    parser.add_argument("--host", default=env("BELLWETHER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(env("BELLWETHER_PORT", "8000")))
    parser.add_argument(
        "--inject-latency-ms", type=float, default=float(env("BELLWETHER_INJECT_LATENCY_MS", "0"))
    )
    parser.add_argument(
        "--inject-error-rate", type=float, default=float(env("BELLWETHER_INJECT_ERROR_RATE", "0"))
    )
    args = parser.parse_args(argv)
    app = create_app(
        args.repository,
        args.version,
        inject_latency_ms=args.inject_latency_ms,
        inject_error_rate=args.inject_error_rate,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()

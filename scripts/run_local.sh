#!/usr/bin/env bash
# Run the full stack on this machine without Docker:
#   model servers v1 (:8001) and v2 (:8002), analyzer (:8010, gRPC :50051),
#   gateway (:8080, metrics :9091). Ctrl-C stops everything.
set -euo pipefail
cd "$(dirname "$0")/.."

export BELLWETHER_DB_URL=${BELLWETHER_DB_URL:-sqlite:///build/bellwether.db}
pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

python3 -m bellwether.modelserver --repository build/models/v1 --version v1 --port 8001 & pids+=($!)
python3 -m bellwether.modelserver --repository build/models/v2 --version v2 --port 8002 & pids+=($!)
python3 -m bellwether.analyzer serve --port 8010 --grpc-port 50051 & pids+=($!)
sleep 3
build/bin/gateway -primary-url http://127.0.0.1:8001 -candidate-url http://127.0.0.1:8002 \
  -analyzer-url http://127.0.0.1:8010/v1/samples:batch -mirror-fraction 0.1 -sample-fraction 1 & pids+=($!)

cat <<'EOF'
Stack is up.
  infer:    POST http://localhost:8080/v2/models/multimodal/infer
  verdict:  GET  http://localhost:8010/v1/models/multimodal/canary/v2/verdict
  status:   GET  http://localhost:8010/v1/models/multimodal/status
  metrics:  http://localhost:9091/metrics  http://localhost:8010/metrics
The analyzer learns its drift baseline from the first 6,000 requests.
EOF
wait

# Bellwether runbook

Every alert in `deploy/prometheus/rules.yml` links to a section here. Analyzer
endpoints are on `bellwether-analyzer:8010`; gateway metrics on `:9091/metrics`.

## Error budget burn

The primary release is failing more than the 99.9% SLO allows.

1. Check whether failures are upstream or in the gateway:
   `bellwether_gateway_requests_total{route="primary"}` by `code`. 502 means the
   model server was unreachable, 504 a timeout, other 5xx came from the server.
2. Check model server health and recent Rollout events:
   `kubectl argo rollouts get rollout modelserver -n bellwether`.
3. If a promotion just happened, roll back:
   `kubectl argo rollouts undo modelserver -n bellwether`.
4. Fast burn (14.4x) spends 2% of the monthly budget per hour; act within the hour.

## Contract incident

Requests or responses broke the data contract (`src/bellwether/contracts/multimodal.yaml`).

1. `GET /v1/models/multimodal/incidents` shows the top violation codes.
2. `inputs.*` or `parameters.*` codes point at a client or SDK release; group
   by `client` in the incident details and contact that client's owners.
3. `outputs.*` codes (norm, non_finite, shape) point at the model server's
   post-processing; treat as a serving bug and roll back the last release.
4. Resolve with `POST /v1/incidents/<id>:resolve` once the cause is fixed.

## Output drift

Model outputs moved while monitored input statistics did not. Usually a model or
serving change: a replica on a different build, a preprocessing change in an
export, or a driver or runtime upgrade.

1. Compare `bellwether_gateway_requests_total` by `version` for unexpected releases.
2. Check the incident's `fired_tests`: `output.image_embedding` alone points at the
   image path; `output.text_embedding` at the tokenizer or text encoder.
3. If inputs might have changed in a way the monitored statistics miss, sample
   recent requests from the analyzer database and inspect them.

## Input drift

Traffic no longer resembles the baseline. This is often legitimate (a launch,
a new locale) and needs a decision, not a rollback.

1. Look at `fired_tests` and their `effect` (PSI): which features moved and how much.
2. If the change is expected and permanent, re-learn the baseline:
   `POST /v1/models/multimodal/baseline:relearn`.
3. If it is a client bug (channel order, resize, stuck frames), route to the
   client owners and keep the incident open.

## Canary rollback

A shadow canary failed analysis and the Rollout aborted. Clients were never
served by the candidate.

1. `GET /v1/models/multimodal/canary/<version>/verdict` lists each check, its value
   and threshold.
2. Embedding agreement failures mean the release is not index compatible with
   the current one; check the export and the Procrustes alignment step.
3. Latency or error budget failures: inspect the candidate pods.
4. `BellwetherCandidateBreakerOpen` means the gateway stopped mirroring after
   repeated candidate failures; it retries automatically after the cooldown.

## Analyzer unavailable

Samples are being dropped or failing to ship, so drift detection and canary
analysis are partly blind. Client traffic is unaffected by design.

1. Check analyzer pod status and `/readyz` (it checks the database).
2. Check `bellwether_gateway_samples_total` by `outcome` on the gateways.
3. Hold any in-progress Rollout; its analysis will turn inconclusive rather than pass.

## Load shedding

The gateway is returning 503 because its in-flight limit was reached.

1. Check the HPA: `kubectl -n bellwether get hpa bellwether-gateway`.
2. Check model server latency: slow upstreams hold in-flight slots longer.
3. Raise `-max-inflight` only if the model servers have headroom; shedding
   protects them from queue collapse.

## No baseline

The analyzer has no drift baseline for a model, so only contract checks run.
It learns one from the first 6,000 valid requests; on a new or low traffic
model, seed one with `bellwether-analyzer build-baseline` or wait for traffic.

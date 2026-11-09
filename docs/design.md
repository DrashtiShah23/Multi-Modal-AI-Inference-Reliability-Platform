# Bellwether design

Status: implemented (this repository). Author: Drashti Shah.

## Context

Multi-modal models fail in ways that health checks do not see. A client
release can start sending BGR frames, a resize change can blur every image, a
new locale can shift caption language, or a retrained encoder can return 200s
at normal latency while moving every vector in a downstream index. Each of
these shows up first in the *data*: what goes into the model and what comes out.

Bellwether is a reliability layer in front of KServe v2 model servers (Triton
in production) that turns those failures into measured, attributable signals
and uses the same signals to gate releases.

## Goals

* Detect input, output and contract failures within one window of traffic
  (400 requests by default) with a calibrated false alarm rate.
* Say whether an alarm is about the traffic or about the model.
* Gate model releases on behavior, not only on health: compare the candidate
  with the current release on identical requests before any client sees it.
* Add no meaningful latency and no failure modes to the client path.

## Non-goals

* Model quality evaluation against labels (no ground truth at serving time).
* Request routing or traffic splitting; Argo Rollouts and Kubernetes own that.
* Storing raw requests long term; samples are processed in memory and only
  window statistics, incidents and baselines are persisted.

## Overview

```
client ──> gateway (Go) ──> primary model server (Triton, stable release)
             │   └─ mirror p=0.1 ─> candidate model server (canary pods)
             └─ samples (async, bounded queue, gzip batches)
                      v
                 analyzer (Python) ── PostgreSQL (baselines, windows, incidents, verdicts)
                      │  REST + gRPC
                      ├─ /metrics ─> Prometheus ─> burn-rate alerts
                      └─ /canary/<v>/verdict <── Argo Rollouts AnalysisRun
```

## Gateway

A dependency-free Go reverse proxy for the KServe v2 protocol.

* **Shedding.** A semaphore bounds in-flight client requests; above it the
  gateway returns 503 with `Retry-After` instead of queueing until upstreams
  collapse.
* **Shadow mirroring.** With probability `mirror-fraction` the same body is sent
  to the candidate *concurrently* with the primary call, on a background
  context. The client response never waits for it. Mirror concurrency is
  bounded, and a consecutive-failure circuit breaker stops mirroring to a
  broken candidate (closed → open → half-open probe).
* **Sampling.** Mirrored requests and a `sample-fraction` of the rest become
  samples (raw request, both responses, statuses, latencies, W3C trace
  context). `Offer` is non-blocking: when the queue is full the sample is
  dropped and counted, so analyzer trouble cannot add client latency.
* **Release labels** come from the response's `model_version`, so a Rollout can
  change the candidate without reconfiguring the gateway.
* **Observability.** Prometheus counters and histograms (hand-written exposition,
  no client library) and W3C `traceparent` propagation into upstream calls and
  samples; the analyzer links its OpenTelemetry spans to it.

The gateway has no third-party Go dependencies, which keeps its supply chain
and container image (distroless, static) minimal.

## Analyzer

### Data contracts

Every item of every sampled request and response is checked against a YAML
contract: tensor datatypes and shapes, caption length and encoding, allowed
parameter values, embedding dimension, L2 norm range, finiteness, similarity
range, and HTTP status. Checks return violation codes rather than raising, so
malformed traffic is a measurement, not a crash. A window whose share of
violating items exceeds `max_violation_rate` (2%) opens a `contract` incident.

### Drift statistics

For each window of 400 *valid* items the analyzer runs 18 two-sample tests
against a reference set from the baseline:

| Feature family | Test | Group |
|---|---|---|
| 8 image statistics (brightness, contrast, channel means, clipped share, sharpness, colorfulness) | Kolmogorov-Smirnov, PSI as effect size | input |
| 4 text statistics (characters, tokens, non-ASCII share, digit share) | KS, PSI | input |
| client and locale | chi-square homogeneity, PSI | input |
| 8x8 pixel thumbnails (model independent) | RBF-kernel MMD | input |
| similarity score | KS | output |
| image and text embeddings | RBF-kernel MMD | output |

MMD uses the median-heuristic bandwidth. Its null distribution is approximated
by a gamma fitted to MMD values of simulated clean windows, which avoids a
permutation test per window.

### Calibrating the alarm

Running 18 tests per window inflates false alarms if each uses alpha. Instead
of a Bonferroni bound, the baseline splits its reference traffic into a
comparison set (2,000 items) and a disjoint holdout. It draws 1,000 simulated
clean windows from the holdout, records the *smallest* p-value of each, and
sets the alarm threshold at the 1% quantile of that distribution. The
window-level false alarm rate is then calibrated directly and accounts for the
correlation between tests (for example brightness and channel means). Gamma
fits and threshold calibration use separate windows, so the threshold is not
fitted to its own p-values.

### Attribution

A fired window is labelled `input_drift` if any input test shows evidence at
p < 1e-3, otherwise `output_drift`: the model's outputs moved while its inputs,
including the model-independent thumbnails, did not. Detection stays governed
by the calibrated threshold; the looser bar only prevents an input change from
being reported as a model change.

### Shadow canary verdict

Each mirrored request gives a *paired* comparison on identical input:

| Check | Fails when |
|---|---|
| error budget burn | candidate 5xx ratio / 0.1% budget > 14.4 (the SRE Workbook fast-burn rate) and above the primary's |
| latency | candidate p95 > 1.25x primary p95 and at least 5 ms slower |
| image embedding agreement | 10th percentile cosine(primary, candidate) < 0.85 |
| text embedding agreement | 10th percentile cosine < 0.85 |
| similarity error | mean absolute difference > 0.08 |
| contract violations | candidate violation rate exceeds primary's by > 0.5 points |

ROLLBACK if any check with enough data fails, PROMOTE when all checks have
enough data (200 pairs, 100 requests) and pass, HOLD otherwise. Agreement
checks are what catch the release that is healthy by every operational metric
but incompatible with the existing index.

### SLOs

Availability SLO of 99.9% over 30 days, alerted with multi-window,
multi-burn-rate rules (14.4x over 1h and 5m, 6x over 6h and 30m, 1x over 3d
and 6h). The same math is implemented in `analyzer/slo.py` with unit tests that
reproduce the workbook's budget figures.

## Release flow

The model server is an Argo Rollout with a shadow canary strategy:
`setCanaryScale` starts candidate pods behind `modelserver-canary` without moving
client traffic; an AnalysisRun polls the analyzer verdict every minute for 15
minutes (any ROLLBACK fails it; it must reach PROMOTE) and checks the
candidate's burn rate in Prometheus; success promotes, failure aborts.

## Models

The bundled encoders keep the system runnable anywhere: random conv features
and a hashed token table with projection heads fit by CCA on 12,000 synthetic
image-caption pairs (matched similarity 0.63 vs 0.00 for mismatched pairs).
Three releases exercise the canary: v1; v2, a retrain rotated onto v1's space
with orthogonal Procrustes to stay index compatible; v3, identical to v1 but
reading BGR. Triton serves them as an ensemble (Python-backend tokenizer, ONNX
encoders, ONNX similarity). Any ONNX models with the same tensor names can be
dropped in.

## Evaluation

See the README for numbers and `bench/` for how to reproduce them. Briefly: on
held-out seeds, 322 of 330 injected faults across 11 failure types were
detected with 0 false alarms in 300 clean windows; the end-to-end drill through
the real gateway promoted the compatible retrain and rolled back the regressed,
slow and failing candidates.

## Alternatives considered

* **Bonferroni across tests.** Simpler but ignores correlation; the empirical
  min-p threshold is calibrated on the same kind of windows it judges.
* **Permutation-test MMD per window.** Exact but about 100x the CPU; the gamma
  approximation is the standard trade-off.
* **Traffic-splitting canary.** Exposes users to the candidate and confounds
  comparisons with different inputs; mirroring gives paired, user-invisible
  comparisons at the cost of extra load on the candidate.
* **Sidecar per model server.** Would put analysis on the serving path's
  machines; a separate gateway keeps model servers unmodified (Triton included).

## Limitations and future work

* The analyzer holds windows in memory per (model, version); one replica per
  model. Sharding by model, or moving window state to a stream processor, is
  the path to scale.
* Latency comparison on mirrored traffic is biased: the candidate sees about 10%
  of the primary's load. Comparing against an equally loaded baseline replica
  would remove the bias.
* A trailing window only sees drift after it fills; low-traffic models detect
  slowly. Sequential tests (for example CUSUM on the min-p) would help.
* Samples contain user inputs. Production use needs a retention policy and PII
  review; today nothing but aggregates and incident details is persisted.

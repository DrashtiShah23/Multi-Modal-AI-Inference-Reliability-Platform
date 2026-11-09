// Package proxy is the gateway's request path: a KServe v2 reverse proxy that
// sheds load, propagates W3C trace context, mirrors a fraction of traffic to a
// candidate release in the background, and samples exchanges for the analyzer.
//
// Invariant: nothing the candidate or the analyzer does can delay or fail a
// client request. Mirroring and sampling run after the primary call, on
// bounded resources, and are skipped (and counted) when those are exhausted.
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/DrashtiShah23/bellwether/gateway/internal/breaker"
	"github.com/DrashtiShah23/bellwether/gateway/internal/metrics"
	"github.com/DrashtiShah23/bellwether/gateway/internal/sampler"
	"github.com/DrashtiShah23/bellwether/gateway/internal/trace"
)

// Config for the gateway.
type Config struct {
	PrimaryURL        *url.URL
	PrimaryVersion    string
	CandidateURL      *url.URL // nil disables mirroring
	CandidateVersion  string
	MirrorFraction    float64
	SampleFraction    float64
	MaxInFlight       int
	MaxMirrorInFlight int
	RequestTimeout    time.Duration
	MirrorTimeout     time.Duration
	MaxBodyBytes      int64
	BreakerFailures   int
	BreakerCooldown   time.Duration
}

// Metrics exported by the gateway.
type Metrics struct {
	Requests     *metrics.CounterVec
	Upstream     *metrics.HistogramVec
	Duration     *metrics.HistogramVec
	InFlight     *metrics.GaugeVec
	Shed         *metrics.CounterVec
	Mirror       *metrics.CounterVec
	BreakerState *metrics.GaugeVec
	Samples      *metrics.CounterVec
	Batches      *metrics.CounterVec
}

// NewMetrics registers the gateway's metric families.
func NewMetrics(r *metrics.Registry) *Metrics {
	return &Metrics{
		Requests: r.NewCounterVec("bellwether_gateway_requests_total",
			"Upstream inference calls by route (primary or candidate) and status code.",
			"model", "route", "version", "code"),
		Upstream: r.NewHistogramVec("bellwether_gateway_upstream_latency_seconds",
			"Latency of upstream model server calls.", metrics.LatencyBuckets, "model", "route", "version"),
		Duration: r.NewHistogramVec("bellwether_gateway_request_duration_seconds",
			"End-to-end time spent in the gateway per client request.", metrics.LatencyBuckets, "model"),
		InFlight: r.NewGaugeVec("bellwether_gateway_inflight_requests", "Client requests in flight.", "model"),
		Shed: r.NewCounterVec("bellwether_gateway_shed_total",
			"Requests rejected with 503 because the in-flight limit was reached.", "model"),
		Mirror: r.NewCounterVec("bellwether_gateway_mirror_total",
			"Mirroring decisions: sent, skipped_breaker_open or skipped_saturated.", "model", "outcome"),
		BreakerState: r.NewGaugeVec("bellwether_gateway_breaker_state",
			"Candidate circuit breaker state (0 closed, 1 half-open, 2 open).", "upstream"),
		Samples: r.NewCounterVec("bellwether_gateway_samples_total",
			"Samples for the analyzer: enqueued, dropped, sent or failed.", "outcome"),
		Batches: r.NewCounterVec("bellwether_gateway_sample_batches_total",
			"Sample batches shipped to the analyzer.", "outcome"),
	}
}

// SamplerStats wires sampler hooks to the gateway metrics.
func (m *Metrics) SamplerStats() sampler.Stats {
	add := func(outcome string) func(int) {
		c := m.Samples.With(outcome)
		return func(n int) { c.Add(float64(n)) }
	}
	return sampler.Stats{
		Enqueued: add("enqueued"), Dropped: add("dropped"), Sent: add("sent"), Failed: add("failed"),
		Batches: func(ok bool) {
			if ok {
				m.Batches.With("ok").Inc()
			} else {
				m.Batches.With("failed").Inc()
			}
		},
	}
}

// Gateway is the HTTP handler.
type Gateway struct {
	cfg      Config
	client   *http.Client
	inflight chan struct{}
	mirrors  chan struct{}
	breaker  *breaker.Breaker
	sampler  *sampler.Sampler
	m        *Metrics
	log      *slog.Logger
	random   func() float64
	mux      *http.ServeMux
}

// New builds a gateway. sampler may be nil to disable sampling.
func New(cfg Config, client *http.Client, s *sampler.Sampler, m *Metrics, log *slog.Logger) *Gateway {
	if cfg.MaxInFlight <= 0 {
		cfg.MaxInFlight = 512
	}
	if cfg.MaxMirrorInFlight <= 0 {
		cfg.MaxMirrorInFlight = 64
	}
	if cfg.MaxBodyBytes <= 0 {
		cfg.MaxBodyBytes = 8 << 20
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = 10 * time.Second
	}
	if cfg.MirrorTimeout <= 0 {
		cfg.MirrorTimeout = 5 * time.Second
	}
	if log == nil {
		log = slog.Default()
	}
	g := &Gateway{
		cfg:      cfg,
		client:   client,
		inflight: make(chan struct{}, cfg.MaxInFlight),
		mirrors:  make(chan struct{}, cfg.MaxMirrorInFlight),
		breaker:  breaker.New(max(cfg.BreakerFailures, 5), max(cfg.BreakerCooldown, time.Second)),
		sampler:  s,
		m:        m,
		log:      log,
		random:   rand.Float64,
		mux:      http.NewServeMux(),
	}
	g.mux.HandleFunc("POST /v2/models/{model}/infer", g.infer)
	g.mux.HandleFunc("POST /v2/models/{model}/versions/{version}/infer", g.infer)
	g.mux.HandleFunc("GET /v2/", g.passthrough)
	g.mux.HandleFunc("GET /v2", g.passthrough)
	return g
}

// SetRandom replaces the random source (tests).
func (g *Gateway) SetRandom(f func() float64) { g.random = f }

// Breaker exposes the candidate breaker (tests and readiness).
func (g *Gateway) Breaker() *breaker.Breaker { return g.breaker }

func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) { g.mux.ServeHTTP(w, r) }

type upstreamResult struct {
	status  int
	body    []byte
	header  http.Header
	latency time.Duration
	err     error
}

func (g *Gateway) call(ctx context.Context, base *url.URL, method, path, rawQuery string, body []byte,
	tc trace.Context, reqID string) upstreamResult {
	target := base.JoinPath(path)
	target.RawQuery = rawQuery
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, target.String(), rdr)
	if err != nil {
		return upstreamResult{err: err}
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set(trace.Header, tc.String())
	if reqID != "" {
		req.Header.Set("X-Request-Id", reqID)
	}
	start := time.Now()
	resp, err := g.client.Do(req)
	if err != nil {
		return upstreamResult{err: err, latency: time.Since(start)}
	}
	defer resp.Body.Close()
	data, err := readAllSized(io.LimitReader(resp.Body, 64<<20), resp.ContentLength)
	return upstreamResult{status: resp.StatusCode, body: data, header: resp.Header, latency: time.Since(start), err: err}
}

// readBody reads the request body in one allocation when Content-Length is known,
// which matters at ~40 KB per image request: io.ReadAll would grow and copy its
// buffer several times and multiply garbage under load.
func readBody(w http.ResponseWriter, r *http.Request, limit int64) ([]byte, error) {
	return readAllSized(http.MaxBytesReader(w, r.Body, limit), min(r.ContentLength, limit))
}

func readAllSized(rd io.Reader, size int64) ([]byte, error) {
	if size <= 0 {
		return io.ReadAll(rd)
	}
	buf := make([]byte, size)
	n, err := io.ReadFull(rd, buf)
	if err != nil {
		if errors.Is(err, io.ErrUnexpectedEOF) {
			return buf[:n], nil
		}
		return nil, err
	}
	// Drain anything beyond the declared length so MaxBytesReader can still enforce the limit.
	rest, err := io.ReadAll(rd)
	if err != nil {
		return nil, err
	}
	return append(buf, rest...), nil
}

func statusFor(err error) int {
	if errors.Is(err, context.DeadlineExceeded) {
		return http.StatusGatewayTimeout
	}
	return http.StatusBadGateway
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func (g *Gateway) infer(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	model := r.PathValue("model")
	defer func() { g.m.Duration.With(model).Observe(time.Since(start).Seconds()) }()

	select {
	case g.inflight <- struct{}{}:
	default:
		g.m.Shed.With(model).Inc()
		w.Header().Set("Retry-After", "1")
		writeJSONError(w, http.StatusServiceUnavailable, "gateway overloaded")
		return
	}
	gauge := g.m.InFlight.With(model)
	gauge.Add(1)
	defer func() { <-g.inflight; gauge.Add(-1) }()

	body, err := readBody(w, r, g.cfg.MaxBodyBytes)
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeJSONError(w, http.StatusRequestEntityTooLarge, "request body too large")
		} else {
			writeJSONError(w, http.StatusBadRequest, "could not read request body")
		}
		return
	}

	parent := trace.FromHeader(r.Header.Get(trace.Header))
	reqID := r.Header.Get("X-Request-Id")
	w.Header().Set("X-Bellwether-Trace-Id", parent.TraceIDHex())

	// Decide on mirroring before the primary call so both start together.
	var candidate chan sampler.Exchange
	if g.cfg.CandidateURL != nil && g.random() < g.cfg.MirrorFraction {
		candidate = g.startMirror(model, r.URL.Path, r.URL.RawQuery, body, parent.Child(), reqID)
	}

	ctx, cancel := context.WithTimeout(r.Context(), g.cfg.RequestTimeout)
	defer cancel()
	res := g.call(ctx, g.cfg.PrimaryURL, http.MethodPost, r.URL.Path, r.URL.RawQuery, body, parent.Child(), reqID)
	status := res.status
	if res.err != nil {
		status = statusFor(res.err)
	}
	g.m.Requests.With(model, "primary", g.cfg.PrimaryVersion, strconv.Itoa(status)).Inc()
	g.m.Upstream.With(model, "primary", g.cfg.PrimaryVersion).Observe(res.latency.Seconds())

	if res.err != nil {
		g.log.Warn("primary call failed", "model", model, "trace_id", parent.TraceIDHex(), "err", res.err)
		writeJSONError(w, status, fmt.Sprintf("primary upstream error: %v", res.err))
	} else {
		if ct := res.header.Get("Content-Type"); ct != "" {
			w.Header().Set("Content-Type", ct)
		}
		w.WriteHeader(res.status)
		_, _ = w.Write(res.body)
	}

	if g.sampler == nil || (candidate == nil && g.random() >= g.cfg.SampleFraction) {
		return
	}
	s := &sampler.Sample{
		TraceID:     parent.TraceIDHex(),
		Traceparent: parent.String(),
		Model:       model,
		ReceivedAt:  start.UTC(),
		Primary:     exchangeFrom(g.cfg.PrimaryVersion, status, res),
	}
	var ok bool
	if s.Request, ok = sampler.JSONOrNull(body); !ok {
		s.RequestError = "invalid_json"
	}
	if candidate == nil {
		g.sampler.Offer(s)
		return
	}
	go func() { // the client already has its response; wait for the shadow call off the request path
		ex := <-candidate
		s.Candidate = &ex
		g.sampler.Offer(s)
	}()
}

// exchangeFrom summarizes an upstream call for the analyzer. The release label
// comes from the response's model_version when present, so a Rollout can change
// the candidate release without reconfiguring the gateway; the flag value is
// the fallback.
func exchangeFrom(version string, status int, res upstreamResult) sampler.Exchange {
	ex := sampler.Exchange{Version: version, Status: status, LatencyMs: float64(res.latency.Microseconds()) / 1000}
	if res.err != nil {
		ex.Status = 0
		ex.Error = res.err.Error()
		ex.Response = json.RawMessage("null")
		return ex
	}
	var ok bool
	if ex.Response, ok = sampler.JSONOrNull(res.body); !ok {
		ex.Error = "invalid_json_response"
		return ex
	}
	var meta struct {
		ModelVersion string `json:"model_version"`
	}
	if json.Unmarshal(res.body, &meta) == nil && meta.ModelVersion != "" {
		ex.Version = meta.ModelVersion
	}
	return ex
}

// startMirror sends the request to the candidate in the background if the
// breaker and the mirror concurrency limit allow it. It returns nil when skipped.
func (g *Gateway) startMirror(model, path, rawQuery string, body []byte, tc trace.Context, reqID string) chan sampler.Exchange {
	select {
	case g.mirrors <- struct{}{}:
	default:
		g.m.Mirror.With(model, "skipped_saturated").Inc()
		return nil
	}
	if !g.breaker.Allow() {
		<-g.mirrors
		g.m.Mirror.With(model, "skipped_breaker_open").Inc()
		g.m.BreakerState.With("candidate").Set(float64(g.breaker.State()))
		return nil
	}
	g.m.Mirror.With(model, "sent").Inc()
	out := make(chan sampler.Exchange, 1)
	go func() {
		defer func() { <-g.mirrors }()
		ctx, cancel := context.WithTimeout(context.Background(), g.cfg.MirrorTimeout)
		defer cancel()
		res := g.call(ctx, g.cfg.CandidateURL, http.MethodPost, path, rawQuery, body, tc, reqID)
		status := res.status
		if res.err != nil {
			status = statusFor(res.err)
		}
		g.breaker.Record(res.err == nil && status < 500)
		g.m.BreakerState.With("candidate").Set(float64(g.breaker.State()))
		ex := exchangeFrom(g.cfg.CandidateVersion, status, res)
		g.m.Requests.With(model, "candidate", ex.Version, strconv.Itoa(status)).Inc()
		g.m.Upstream.With(model, "candidate", ex.Version).Observe(res.latency.Seconds())
		out <- ex
	}()
	return out
}

// passthrough forwards metadata and health calls to the primary.
func (g *Gateway) passthrough(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), g.cfg.RequestTimeout)
	defer cancel()
	tc := trace.FromHeader(r.Header.Get(trace.Header))
	res := g.call(ctx, g.cfg.PrimaryURL, http.MethodGet, r.URL.Path, r.URL.RawQuery, nil, tc.Child(), r.Header.Get("X-Request-Id"))
	if res.err != nil {
		writeJSONError(w, statusFor(res.err), fmt.Sprintf("primary upstream error: %v", res.err))
		return
	}
	if ct := res.header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	w.WriteHeader(res.status)
	_, _ = w.Write(res.body)
}

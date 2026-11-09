package proxy

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/DrashtiShah23/bellwether/gateway/internal/breaker"
	"github.com/DrashtiShah23/bellwether/gateway/internal/metrics"
	"github.com/DrashtiShah23/bellwether/gateway/internal/sampler"
	"github.com/DrashtiShah23/bellwether/gateway/internal/trace"
)

const reqBody = `{"inputs":[{"name":"text","shape":[1],"datatype":"BYTES","data":["a red circle"]}]}`

type captureSink struct {
	mu      sync.Mutex
	samples []*sampler.Sample
}

func (c *captureSink) Send(_ context.Context, b []*sampler.Sample) error {
	c.mu.Lock()
	c.samples = append(c.samples, b...)
	c.mu.Unlock()
	return nil
}

func (c *captureSink) all() []*sampler.Sample {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]*sampler.Sample(nil), c.samples...)
}

type upstream struct {
	*httptest.Server
	calls     atomic.Int64
	lastBody  atomic.Value
	lastTrace atomic.Value
	delay     time.Duration
	status    int
	version   string
}

func newUpstream(t *testing.T, version string, delay time.Duration, status int) *upstream {
	u := &upstream{delay: delay, status: status, version: version}
	u.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u.calls.Add(1)
		b, _ := io.ReadAll(r.Body)
		u.lastBody.Store(string(b))
		u.lastTrace.Store(r.Header.Get(trace.Header))
		time.Sleep(u.delay)
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(u.status)
		_, _ = io.WriteString(w, `{"model_name":"multimodal","model_version":"`+u.version+`","outputs":[]}`)
	}))
	t.Cleanup(u.Close)
	return u
}

type harness struct {
	gw   *Gateway
	srv  *httptest.Server
	sink *captureSink
	smp  *sampler.Sampler
	reg  *metrics.Registry
}

func newHarness(t *testing.T, cfg Config) *harness {
	t.Helper()
	reg := metrics.NewRegistry()
	m := NewMetrics(reg)
	sink := &captureSink{}
	smp := sampler.New(sampler.Config{MaxBatch: 1, FlushInterval: 10 * time.Millisecond}, sink, m.SamplerStats(), nil)
	smp.Start()
	gw := New(cfg, &http.Client{}, smp, m, nil)
	srv := httptest.NewServer(gw)
	t.Cleanup(srv.Close)
	return &harness{gw: gw, srv: srv, sink: sink, smp: smp, reg: reg}
}

func (h *harness) drain(t *testing.T) []*sampler.Sample {
	t.Helper()
	time.Sleep(50 * time.Millisecond) // let background mirror goroutines enqueue
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := h.smp.Stop(ctx); err != nil {
		t.Fatal(err)
	}
	return h.sink.all()
}

func (h *harness) metricsText() string {
	var b strings.Builder
	w := bufio.NewWriter(&b)
	h.reg.WriteTo(w)
	_ = w.Flush()
	return b.String()
}

func mustParse(t *testing.T, raw string) *url.URL {
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	return u
}

func post(t *testing.T, base, body string, header map[string]string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, base+"/v2/models/multimodal/infer", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range header {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestProxiesAndPropagatesTrace(t *testing.T) {
	primary := newUpstream(t, "v1", 0, 200)
	h := newHarness(t, Config{PrimaryURL: mustParse(t, primary.URL), PrimaryVersion: "v1", SampleFraction: 1})
	const parent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
	resp := post(t, h.srv.URL, reqBody, map[string]string{trace.Header: parent})
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || !strings.Contains(string(body), `"model_version":"v1"`) {
		t.Fatalf("got %d %s", resp.StatusCode, body)
	}
	if resp.Header.Get("X-Bellwether-Trace-Id") != "4bf92f3577b34da6a3ce929d0e0e4736" {
		t.Fatal("trace id header not set")
	}
	up, ok := trace.Parse(primary.lastTrace.Load().(string))
	if !ok || up.TraceIDHex() != "4bf92f3577b34da6a3ce929d0e0e4736" || up.String() == parent {
		t.Fatal("upstream should receive a child span of the caller's trace")
	}
	if primary.lastBody.Load().(string) != reqBody {
		t.Fatal("body not forwarded verbatim")
	}
	samples := h.drain(t)
	if len(samples) != 1 || samples[0].Candidate != nil || string(samples[0].Request) != reqBody ||
		samples[0].Primary.Status != 200 || samples[0].Traceparent != parent {
		t.Fatalf("unexpected sample %+v", samples)
	}
}

func TestMirrorsToCandidateWithoutBlockingClient(t *testing.T) {
	primary := newUpstream(t, "v1", 0, 200)
	candidate := newUpstream(t, "v2", 300*time.Millisecond, 200)
	h := newHarness(t, Config{PrimaryURL: mustParse(t, primary.URL), PrimaryVersion: "v1",
		CandidateURL: mustParse(t, candidate.URL), CandidateVersion: "fallback", MirrorFraction: 1})
	start := time.Now()
	resp := post(t, h.srv.URL, reqBody, nil)
	resp.Body.Close()
	if elapsed := time.Since(start); elapsed > 150*time.Millisecond {
		t.Fatalf("client waited %v for a shadow call", elapsed)
	}
	time.Sleep(400 * time.Millisecond)
	samples := h.drain(t)
	if candidate.calls.Load() != 1 || candidate.lastBody.Load().(string) != reqBody {
		t.Fatal("candidate should receive the same body once")
	}
	// the release label comes from the candidate's model_version, not the flag
	if len(samples) != 1 || samples[0].Candidate == nil || samples[0].Candidate.Version != "v2" ||
		samples[0].Candidate.LatencyMs < 300 {
		t.Fatalf("sample should pair both exchanges: %+v", samples)
	}
	if !strings.Contains(h.metricsText(), `bellwether_gateway_mirror_total{model="multimodal",outcome="sent"} 1`) {
		t.Fatal("mirror metric missing")
	}
}

func TestMirrorFractionAndSampling(t *testing.T) {
	primary := newUpstream(t, "v1", 0, 200)
	candidate := newUpstream(t, "v2", 0, 200)
	h := newHarness(t, Config{PrimaryURL: mustParse(t, primary.URL), CandidateURL: mustParse(t, candidate.URL),
		MirrorFraction: 0.25, SampleFraction: 0})
	rng := rand.New(rand.NewPCG(1, 2))
	var mu sync.Mutex
	h.gw.SetRandom(func() float64 { mu.Lock(); defer mu.Unlock(); return rng.Float64() })
	for n := 0; n < 400; n++ {
		post(t, h.srv.URL, reqBody, nil).Body.Close()
	}
	samples := h.drain(t)
	if got := candidate.calls.Load(); got < 70 || got > 130 {
		t.Fatalf("mirrored %d of 400, want about 100", got)
	}
	if len(samples) != int(candidate.calls.Load()) {
		t.Fatalf("%d samples for %d mirrored requests", len(samples), candidate.calls.Load())
	}
	for _, s := range samples {
		if s.Candidate == nil {
			t.Fatal("with SampleFraction 0 only mirrored requests are sampled")
		}
	}
}

func TestCandidateFailuresOpenBreaker(t *testing.T) {
	primary := newUpstream(t, "v1", 0, 200)
	candidate := newUpstream(t, "v2", 0, 503)
	h := newHarness(t, Config{PrimaryURL: mustParse(t, primary.URL), CandidateURL: mustParse(t, candidate.URL),
		MirrorFraction: 1, BreakerFailures: 5, BreakerCooldown: time.Minute})
	for n := 0; n < 20; n++ {
		resp := post(t, h.srv.URL, reqBody, nil)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatal("candidate failures must not affect clients")
		}
		time.Sleep(5 * time.Millisecond)
	}
	if h.gw.Breaker().State() != breaker.Open {
		t.Fatal("breaker should be open")
	}
	if calls := candidate.calls.Load(); calls > 6 {
		t.Fatalf("candidate got %d calls after the breaker opened", calls)
	}
	if !strings.Contains(h.metricsText(), `outcome="skipped_breaker_open"`) {
		t.Fatal("skips should be counted")
	}
	h.drain(t)
}

func TestShedsLoadAboveInFlightLimit(t *testing.T) {
	primary := newUpstream(t, "v1", 200*time.Millisecond, 200)
	h := newHarness(t, Config{PrimaryURL: mustParse(t, primary.URL), MaxInFlight: 1})
	var wg sync.WaitGroup
	codes := make(chan int, 4)
	for n := 0; n < 4; n++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp := post(t, h.srv.URL, reqBody, nil)
			resp.Body.Close()
			codes <- resp.StatusCode
		}()
		time.Sleep(10 * time.Millisecond)
	}
	wg.Wait()
	close(codes)
	shed := 0
	for c := range codes {
		if c == http.StatusServiceUnavailable {
			shed++
		}
	}
	if shed != 3 {
		t.Fatalf("shed %d requests, want 3", shed)
	}
	h.drain(t)
}

func TestUpstreamErrorsAndBadInput(t *testing.T) {
	down := httptest.NewServer(http.NotFoundHandler())
	downURL := down.URL
	down.Close()
	h := newHarness(t, Config{PrimaryURL: mustParse(t, downURL), SampleFraction: 1, MaxBodyBytes: 1024})
	resp := post(t, h.srv.URL, "{not json", nil)
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("got %d, want 502", resp.StatusCode)
	}
	big := post(t, h.srv.URL, strings.Repeat("x", 2048), nil)
	big.Body.Close()
	if big.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("got %d, want 413", big.StatusCode)
	}
	samples := h.drain(t)
	if len(samples) != 1 || samples[0].RequestError != "invalid_json" || samples[0].Primary.Status != 0 ||
		string(samples[0].Request) != "null" {
		t.Fatalf("unexpected sample %+v", samples[0])
	}

	slow := newUpstream(t, "v1", 300*time.Millisecond, 200)
	h2 := newHarness(t, Config{PrimaryURL: mustParse(t, slow.URL), RequestTimeout: 50 * time.Millisecond})
	resp = post(t, h2.srv.URL, reqBody, nil)
	resp.Body.Close()
	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Fatalf("got %d, want 504", resp.StatusCode)
	}
	h2.drain(t)
}

func TestHealthPassthrough(t *testing.T) {
	primary := newUpstream(t, "v1", 0, 200)
	h := newHarness(t, Config{PrimaryURL: mustParse(t, primary.URL)})
	resp, err := http.Get(h.srv.URL + "/v2/health/ready")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("got %d", resp.StatusCode)
	}
	h.drain(t)
}

func BenchmarkGatewayOverhead(b *testing.B) {
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		_, _ = io.WriteString(w, `{"outputs":[]}`)
	}))
	defer primary.Close()
	u, _ := url.Parse(primary.URL)
	m := NewMetrics(metrics.NewRegistry())
	gw := New(Config{PrimaryURL: u, SampleFraction: 0}, &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: 64}}, nil, m, nil)
	body := []byte(reqBody)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			req := httptest.NewRequest(http.MethodPost, "/v2/models/multimodal/infer", bytes.NewReader(body))
			rec := httptest.NewRecorder()
			gw.ServeHTTP(rec, req)
			if rec.Code != 200 {
				b.Fatal(rec.Code)
			}
		}
	})
}

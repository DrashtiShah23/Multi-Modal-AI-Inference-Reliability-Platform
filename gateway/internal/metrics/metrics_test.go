package metrics

import (
	"bufio"
	"strings"
	"sync"
	"testing"
)

func render(r *Registry) string {
	var b strings.Builder
	w := bufio.NewWriter(&b)
	r.WriteTo(w)
	_ = w.Flush()
	return b.String()
}

func TestExpositionFormat(t *testing.T) {
	r := NewRegistry()
	c := r.NewCounterVec("reqs_total", "Requests.", "code", "route")
	c.With("200", "primary").Add(3)
	c.With("500", "candidate").Inc()
	g := r.NewGaugeVec("inflight", "In flight.", "model")
	g.With(`we"ird`).Set(2.5)
	h := r.NewHistogramVec("lat_seconds", "Latency.", []float64{0.1, 0.5}, "route")
	for _, v := range []float64{0.05, 0.1, 0.3, 2} {
		h.With("primary").Observe(v)
	}
	out := render(r)
	for _, want := range []string{
		"# TYPE reqs_total counter",
		`reqs_total{code="200",route="primary"} 3`,
		`reqs_total{code="500",route="candidate"} 1`,
		`inflight{model="we\"ird"} 2.5`,
		"# TYPE lat_seconds histogram",
		`lat_seconds_bucket{route="primary",le="0.1"} 2`,
		`lat_seconds_bucket{route="primary",le="0.5"} 3`,
		`lat_seconds_bucket{route="primary",le="+Inf"} 4`,
		`lat_seconds_sum{route="primary"} 2.45`,
		`lat_seconds_count{route="primary"} 4`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
	if strings.Index(out, `code="200"`) > strings.Index(out, `code="500"`) {
		t.Error("series should be sorted by label values")
	}
}

func TestConcurrentUpdates(t *testing.T) {
	r := NewRegistry()
	c := r.NewCounterVec("n_total", "N.", "k")
	h := r.NewHistogramVec("h", "H.", LatencyBuckets, "k")
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				c.With("a").Inc()
				h.With("a").Observe(0.003)
			}
		}()
	}
	wg.Wait()
	if got := c.With("a").Value(); got != 16000 {
		t.Fatalf("counter = %v, want 16000", got)
	}
	if _, n, _ := h.With("a").Snapshot(); n != 16000 {
		t.Fatalf("histogram count = %d", n)
	}
}

func TestDuplicateRegistrationPanics(t *testing.T) {
	r := NewRegistry()
	r.NewCounterVec("x_total", "X.")
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	r.NewGaugeVec("x_total", "X.")
}

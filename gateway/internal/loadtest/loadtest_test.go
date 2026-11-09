package loadtest

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestPercentileNearestRank(t *testing.T) {
	v := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	if Percentile(v, 50) != 5 || Percentile(v, 90) != 9 || Percentile(v, 99) != 10 || Percentile(v, 0) != 1 {
		t.Fatal("unexpected percentiles")
	}
}

func TestOpenLoopRun(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(2 * time.Millisecond)
	}))
	defer srv.Close()
	res := Run(context.Background(), Config{URL: srv.URL, Body: []byte(`{}`), Rate: 200, Duration: time.Second,
		Warmup: 200 * time.Millisecond, Workers: 32})
	if res.Requests < 180 || res.Requests > 220 || res.Errors != 0 {
		t.Fatalf("unexpected result %+v", res)
	}
	if res.P50 < 2 || res.P99 < res.P50 {
		t.Fatalf("latencies look wrong: %+v", res)
	}
}

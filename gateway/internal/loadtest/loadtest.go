// Package loadtest is an open-loop HTTP load generator. Requests are scheduled
// at a fixed rate and latency is measured from each request's *intended* send
// time, so a stalled server shows up in the percentiles instead of silently
// lowering the offered load (coordinated omission).
package loadtest

import (
	"bytes"
	"context"
	"io"
	"math"
	"net/http"
	"sort"
	"sync"
	"time"
)

// Config for one run.
type Config struct {
	URL      string
	Body     []byte
	Rate     float64 // requests per second
	Duration time.Duration
	Warmup   time.Duration // excluded from results
	Workers  int
	Client   *http.Client
}

// Result summarizes a run (latencies in milliseconds).
type Result struct {
	URL         string  `json:"url"`
	TargetRPS   float64 `json:"target_rps"`
	AchievedRPS float64 `json:"achieved_rps"`
	Requests    int     `json:"requests"`
	Errors      int     `json:"errors"`
	Dropped     int     `json:"dropped"`
	P50         float64 `json:"p50_ms"`
	P90         float64 `json:"p90_ms"`
	P99         float64 `json:"p99_ms"`
	P999        float64 `json:"p999_ms"`
	Max         float64 `json:"max_ms"`
}

// Percentile returns the q-th percentile (0..100) of sorted values, nearest rank.
func Percentile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return math.NaN()
	}
	rank := int(math.Ceil(q/100*float64(len(sorted)))) - 1
	return sorted[min(max(rank, 0), len(sorted)-1)]
}

// Summarize builds a Result from per-request latencies.
func Summarize(cfg Config, latencies []float64, errors, dropped int, elapsed time.Duration) Result {
	sort.Float64s(latencies)
	r := Result{URL: cfg.URL, TargetRPS: cfg.Rate, Requests: len(latencies), Errors: errors, Dropped: dropped}
	if elapsed > 0 {
		r.AchievedRPS = float64(len(latencies)) / elapsed.Seconds()
	}
	if len(latencies) > 0 {
		r.P50, r.P90, r.P99 = Percentile(latencies, 50), Percentile(latencies, 90), Percentile(latencies, 99)
		r.P999, r.Max = Percentile(latencies, 99.9), latencies[len(latencies)-1]
	}
	return r
}

// Run executes the load test.
func Run(ctx context.Context, cfg Config) Result {
	if cfg.Workers <= 0 {
		cfg.Workers = 256
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: cfg.Workers, MaxIdleConns: cfg.Workers}}
	}
	type job struct{ intended time.Time }
	jobs := make(chan job, cfg.Workers*4)
	var (
		mu        sync.Mutex
		latencies []float64
		errs      int
	)
	measureFrom := time.Now().Add(cfg.Warmup)
	var wg sync.WaitGroup
	for i := 0; i < cfg.Workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				req, _ := http.NewRequestWithContext(ctx, http.MethodPost, cfg.URL, bytes.NewReader(cfg.Body))
				req.Header.Set("Content-Type", "application/json")
				resp, err := cfg.Client.Do(req)
				ok := err == nil && resp.StatusCode == http.StatusOK
				if err == nil {
					_, _ = io.Copy(io.Discard, resp.Body)
					resp.Body.Close()
				}
				lat := float64(time.Since(j.intended).Microseconds()) / 1000
				if j.intended.Before(measureFrom) {
					continue
				}
				mu.Lock()
				if ok {
					latencies = append(latencies, lat)
				} else {
					errs++
				}
				mu.Unlock()
			}
		}()
	}

	interval := time.Duration(float64(time.Second) / cfg.Rate)
	start := time.Now()
	end := start.Add(cfg.Warmup + cfg.Duration)
	dropped := 0
	for i := 0; ; i++ {
		intended := start.Add(time.Duration(i) * interval)
		if intended.After(end) || ctx.Err() != nil {
			break
		}
		if d := time.Until(intended); d > 0 {
			time.Sleep(d)
		}
		select {
		case jobs <- job{intended}:
		default:
			if !intended.Before(measureFrom) {
				dropped++ // generator saturated; reported so results are not silently optimistic
			}
		}
	}
	close(jobs)
	wg.Wait()
	return Summarize(cfg, latencies, errs, dropped, cfg.Duration)
}

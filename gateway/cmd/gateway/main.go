// Command gateway runs the Bellwether reliability gateway in front of KServe v2
// model servers (Triton or the bundled Python server).
//
// Every flag can also be set with an environment variable named
// BELLWETHER_GATEWAY_<FLAG> (upper case, dashes as underscores).
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/DrashtiShah23/bellwether/gateway/internal/metrics"
	"github.com/DrashtiShah23/bellwether/gateway/internal/proxy"
	"github.com/DrashtiShah23/bellwether/gateway/internal/sampler"
)

type options struct {
	listen, adminListen            string
	primaryURL, primaryVersion     string
	candidateURL, candidateVersion string
	analyzerURL                    string
	mirrorFraction, sampleFraction float64
	maxInFlight, maxMirrorInFlight int
	requestTimeout, mirrorTimeout  time.Duration
	maxBodyBytes                   int64
	queueSize, batchSize           int
	flushInterval                  time.Duration
	breakerFailures                int
	breakerCooldown                time.Duration
	logLevel                       string
}

func parse(args []string) (*options, error) {
	fs := flag.NewFlagSet("gateway", flag.ContinueOnError)
	o := &options{}
	fs.StringVar(&o.listen, "listen", ":8080", "client listen address")
	fs.StringVar(&o.adminListen, "admin-listen", ":9091", "metrics and health listen address")
	fs.StringVar(&o.primaryURL, "primary-url", "http://localhost:8000", "primary model server base URL")
	fs.StringVar(&o.primaryVersion, "primary-version", "v1", "primary release label")
	fs.StringVar(&o.candidateURL, "candidate-url", "", "candidate model server base URL (empty disables mirroring)")
	fs.StringVar(&o.candidateVersion, "candidate-version", "v2", "candidate release label")
	fs.StringVar(&o.analyzerURL, "analyzer-url", "", "analyzer ingest URL, e.g. http://analyzer:8010/v1/samples:batch")
	fs.Float64Var(&o.mirrorFraction, "mirror-fraction", 0.1, "fraction of requests mirrored to the candidate")
	fs.Float64Var(&o.sampleFraction, "sample-fraction", 0.05, "fraction of non-mirrored requests sampled to the analyzer")
	fs.IntVar(&o.maxInFlight, "max-inflight", 512, "client requests in flight before shedding with 503")
	fs.IntVar(&o.maxMirrorInFlight, "max-mirror-inflight", 64, "concurrent shadow calls to the candidate")
	fs.DurationVar(&o.requestTimeout, "request-timeout", 10*time.Second, "primary call timeout")
	fs.DurationVar(&o.mirrorTimeout, "mirror-timeout", 5*time.Second, "candidate call timeout")
	fs.Int64Var(&o.maxBodyBytes, "max-body-bytes", 8<<20, "largest accepted request body")
	fs.IntVar(&o.queueSize, "sample-queue", 10000, "sample queue capacity")
	fs.IntVar(&o.batchSize, "sample-batch", 64, "samples per analyzer batch")
	fs.DurationVar(&o.flushInterval, "sample-flush", 500*time.Millisecond, "max wait before shipping a partial batch")
	fs.IntVar(&o.breakerFailures, "breaker-failures", 5, "consecutive candidate failures that open the breaker")
	fs.DurationVar(&o.breakerCooldown, "breaker-cooldown", 10*time.Second, "time the breaker stays open")
	fs.StringVar(&o.logLevel, "log-level", "info", "debug, info, warn or error")

	var envErr error
	fs.VisitAll(func(f *flag.Flag) {
		key := "BELLWETHER_GATEWAY_" + strings.ToUpper(strings.ReplaceAll(f.Name, "-", "_"))
		if v, ok := os.LookupEnv(key); ok {
			if err := f.Value.Set(v); err != nil && envErr == nil {
				envErr = fmt.Errorf("%s: %w", key, err)
			}
		}
	})
	if envErr != nil {
		return nil, envErr
	}
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	if o.mirrorFraction < 0 || o.mirrorFraction > 1 || o.sampleFraction < 0 || o.sampleFraction > 1 {
		return nil, errors.New("fractions must be within [0, 1]")
	}
	return o, nil
}

func mustURL(raw string) (*url.URL, error) {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("invalid URL %q", raw)
	}
	return u, nil
}

func run(ctx context.Context, o *options, log *slog.Logger) error {
	primary, err := mustURL(o.primaryURL)
	if err != nil {
		return err
	}
	var candidate *url.URL
	if o.candidateURL != "" {
		if candidate, err = mustURL(o.candidateURL); err != nil {
			return err
		}
	}

	transport := &http.Transport{
		DialContext:         (&net.Dialer{Timeout: 3 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		MaxIdleConns:        2048,
		MaxIdleConnsPerHost: 1024,
		IdleConnTimeout:     90 * time.Second,
	}
	client := &http.Client{Transport: transport}

	reg := metrics.NewRegistry()
	m := proxy.NewMetrics(reg)
	var smp *sampler.Sampler
	if o.analyzerURL != "" {
		smp = sampler.New(sampler.Config{QueueSize: o.queueSize, MaxBatch: o.batchSize, FlushInterval: o.flushInterval},
			&sampler.HTTPSink{URL: o.analyzerURL, Client: &http.Client{Transport: transport}}, m.SamplerStats(), log)
		smp.Start()
	}
	gw := proxy.New(proxy.Config{
		PrimaryURL: primary, PrimaryVersion: o.primaryVersion,
		CandidateURL: candidate, CandidateVersion: o.candidateVersion,
		MirrorFraction: o.mirrorFraction, SampleFraction: o.sampleFraction,
		MaxInFlight: o.maxInFlight, MaxMirrorInFlight: o.maxMirrorInFlight,
		RequestTimeout: o.requestTimeout, MirrorTimeout: o.mirrorTimeout, MaxBodyBytes: o.maxBodyBytes,
		BreakerFailures: o.breakerFailures, BreakerCooldown: o.breakerCooldown,
	}, client, smp, m, log)

	admin := http.NewServeMux()
	admin.Handle("GET /metrics", reg.Handler())
	admin.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok\n")) })
	admin.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, primary.JoinPath("/v2/health/ready").String(), nil)
		resp, err := client.Do(req)
		if err != nil || resp.StatusCode != http.StatusOK {
			http.Error(w, "primary not ready", http.StatusServiceUnavailable)
			return
		}
		resp.Body.Close()
		_, _ = w.Write([]byte("ok\n"))
	})

	servers := []*http.Server{
		{Addr: o.listen, Handler: gw, ReadHeaderTimeout: 5 * time.Second, IdleTimeout: 120 * time.Second},
		{Addr: o.adminListen, Handler: admin, ReadHeaderTimeout: 5 * time.Second},
	}
	errc := make(chan error, len(servers))
	for _, s := range servers {
		go func(s *http.Server) {
			if err := s.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				errc <- err
			}
		}(s)
	}
	log.Info("gateway started", "listen", o.listen, "admin", o.adminListen, "primary", o.primaryURL,
		"candidate", o.candidateURL, "mirror_fraction", o.mirrorFraction, "analyzer", o.analyzerURL)

	select {
	case <-ctx.Done():
	case err := <-errc:
		return err
	}
	log.Info("shutting down")
	shutdown, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	for _, s := range servers {
		_ = s.Shutdown(shutdown)
	}
	if smp != nil {
		if err := smp.Stop(shutdown); err != nil {
			log.Warn("sampler did not drain before shutdown", "err", err)
		}
	}
	return nil
}

func main() {
	o, err := parse(os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	var level slog.Level
	if err := level.UnmarshalText([]byte(o.logLevel)); err != nil {
		level = slog.LevelInfo
	}
	log := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, o, log); err != nil {
		log.Error("gateway failed", "err", err)
		os.Exit(1)
	}
}

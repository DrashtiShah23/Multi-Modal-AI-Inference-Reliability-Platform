// Package sampler ships request/response samples to the analyzer off the
// request path. Offer never blocks: when the bounded queue is full the sample
// is dropped and counted, so analyzer slowness can never add serving latency.
package sampler

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

// Exchange is one call to an upstream model server.
type Exchange struct {
	Version   string          `json:"version"`
	Status    int             `json:"status"`
	LatencyMs float64         `json:"latency_ms"`
	Response  json.RawMessage `json:"response"`
	Error     string          `json:"error,omitempty"`
}

// Sample is the unit the analyzer ingests (see proto/bellwether/v1/analyzer.proto).
type Sample struct {
	TraceID      string          `json:"trace_id"`
	Traceparent  string          `json:"traceparent"`
	Model        string          `json:"model"`
	ReceivedAt   time.Time       `json:"received_at"`
	Request      json.RawMessage `json:"request"`
	RequestError string          `json:"request_error,omitempty"`
	Primary      Exchange        `json:"primary"`
	Candidate    *Exchange       `json:"candidate,omitempty"`
}

// JSONOrNull returns body as raw JSON, or JSON null if it is not valid JSON.
func JSONOrNull(body []byte) (json.RawMessage, bool) {
	if len(body) > 0 && json.Valid(body) {
		return json.RawMessage(body), true
	}
	return json.RawMessage("null"), false
}

// Sink delivers a batch.
type Sink interface {
	Send(ctx context.Context, batch []*Sample) error
}

// HTTPSink posts gzip-compressed JSON batches to the analyzer's REST API.
type HTTPSink struct {
	URL    string
	Client *http.Client
}

// Send implements Sink.
func (s *HTTPSink) Send(ctx context.Context, batch []*Sample) error {
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	if err := json.NewEncoder(zw).Encode(map[string]any{"samples": batch}); err != nil {
		return err
	}
	if err := zw.Close(); err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.URL, &buf)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Content-Encoding", "gzip")
	resp, err := s.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode >= 300 {
		return fmt.Errorf("analyzer returned %d", resp.StatusCode)
	}
	return nil
}

// Stats are hooks for metrics; any may be nil.
type Stats struct {
	Enqueued, Dropped, Sent, Failed func(n int)
	Batches                         func(ok bool)
}

func (s Stats) call(f func(int), n int) {
	if f != nil {
		f(n)
	}
}

// Config tunes batching.
type Config struct {
	QueueSize     int
	MaxBatch      int
	FlushInterval time.Duration
	MaxAttempts   int
	Backoff       time.Duration
}

// Sampler batches samples to a Sink in the background.
type Sampler struct {
	cfg   Config
	sink  Sink
	stats Stats
	log   *slog.Logger
	queue chan *Sample
	stop  chan struct{}
	done  chan struct{}
	once  sync.Once
}

// New creates a sampler; call Start to begin shipping.
func New(cfg Config, sink Sink, stats Stats, log *slog.Logger) *Sampler {
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = 10000
	}
	if cfg.MaxBatch <= 0 {
		cfg.MaxBatch = 64
	}
	if cfg.FlushInterval <= 0 {
		cfg.FlushInterval = 500 * time.Millisecond
	}
	if cfg.MaxAttempts <= 0 {
		cfg.MaxAttempts = 3
	}
	if cfg.Backoff <= 0 {
		cfg.Backoff = 100 * time.Millisecond
	}
	if log == nil {
		log = slog.Default()
	}
	return &Sampler{cfg: cfg, sink: sink, stats: stats, log: log, queue: make(chan *Sample, cfg.QueueSize),
		stop: make(chan struct{}), done: make(chan struct{})}
}

// Offer enqueues without blocking and reports whether the sample was kept.
func (s *Sampler) Offer(sample *Sample) bool {
	select {
	case s.queue <- sample:
		s.stats.call(s.stats.Enqueued, 1)
		return true
	default:
		s.stats.call(s.stats.Dropped, 1)
		return false
	}
}

// Start launches the shipping loop.
func (s *Sampler) Start() { go s.run() }

// Stop flushes what is queued, waiting at most until ctx is done.
func (s *Sampler) Stop(ctx context.Context) error {
	s.once.Do(func() { close(s.stop) })
	select {
	case <-s.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *Sampler) run() {
	defer close(s.done)
	ticker := time.NewTicker(s.cfg.FlushInterval)
	defer ticker.Stop()
	batch := make([]*Sample, 0, s.cfg.MaxBatch)
	flush := func() {
		if len(batch) > 0 {
			s.ship(batch)
			batch = make([]*Sample, 0, s.cfg.MaxBatch)
		}
	}
	for {
		select {
		case sm := <-s.queue:
			batch = append(batch, sm)
			if len(batch) >= s.cfg.MaxBatch {
				flush()
			}
		case <-ticker.C:
			flush()
		case <-s.stop:
			for {
				select {
				case sm := <-s.queue:
					batch = append(batch, sm)
					if len(batch) >= s.cfg.MaxBatch {
						flush()
					}
				default:
					flush()
					return
				}
			}
		}
	}
}

func (s *Sampler) ship(batch []*Sample) {
	var err error
	for attempt := 0; attempt < s.cfg.MaxAttempts; attempt++ {
		if attempt > 0 {
			time.Sleep(s.cfg.Backoff << (attempt - 1))
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		err = s.sink.Send(ctx, batch)
		cancel()
		if err == nil {
			s.stats.call(s.stats.Sent, len(batch))
			if s.stats.Batches != nil {
				s.stats.Batches(true)
			}
			return
		}
	}
	s.stats.call(s.stats.Failed, len(batch))
	if s.stats.Batches != nil {
		s.stats.Batches(false)
	}
	if !errors.Is(err, context.Canceled) {
		s.log.Warn("dropping sample batch after retries", "samples", len(batch), "err", err)
	}
}

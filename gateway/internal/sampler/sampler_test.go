package sampler

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type memSink struct {
	mu      sync.Mutex
	batches [][]*Sample
	fail    atomic.Int32
}

func (m *memSink) Send(_ context.Context, b []*Sample) error {
	if m.fail.Load() > 0 {
		m.fail.Add(-1)
		return errors.New("boom")
	}
	m.mu.Lock()
	m.batches = append(m.batches, b)
	m.mu.Unlock()
	return nil
}

func (m *memSink) total() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	n := 0
	for _, b := range m.batches {
		n += len(b)
	}
	return n
}

func TestBatchesBySizeAndDrainsOnStop(t *testing.T) {
	sink := &memSink{}
	var sent atomic.Int64
	s := New(Config{QueueSize: 100, MaxBatch: 10, FlushInterval: time.Hour}, sink,
		Stats{Sent: func(n int) { sent.Add(int64(n)) }}, nil)
	s.Start()
	for i := 0; i < 25; i++ {
		if !s.Offer(&Sample{Model: "m"}) {
			t.Fatal("queue should have room")
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := s.Stop(ctx); err != nil {
		t.Fatal(err)
	}
	if sink.total() != 25 || sent.Load() != 25 {
		t.Fatalf("shipped %d samples, want 25", sink.total())
	}
	if len(sink.batches[0]) != 10 {
		t.Fatalf("first batch has %d samples, want 10", len(sink.batches[0]))
	}
}

func TestOfferDropsWhenFull(t *testing.T) {
	var dropped atomic.Int64
	s := New(Config{QueueSize: 2}, &memSink{}, Stats{Dropped: func(n int) { dropped.Add(int64(n)) }}, nil)
	// not started: nothing drains the queue
	s.Offer(&Sample{})
	s.Offer(&Sample{})
	if s.Offer(&Sample{}) || dropped.Load() != 1 {
		t.Fatal("third offer should be dropped without blocking")
	}
}

func TestRetriesThenSucceeds(t *testing.T) {
	sink := &memSink{}
	sink.fail.Store(2)
	s := New(Config{MaxBatch: 1, FlushInterval: time.Hour, Backoff: time.Millisecond}, sink, Stats{}, nil)
	s.Start()
	s.Offer(&Sample{})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = s.Stop(ctx)
	if sink.total() != 1 {
		t.Fatal("sample should be delivered on the third attempt")
	}
}

func TestHTTPSinkSendsGzipJSON(t *testing.T) {
	var got struct {
		Samples []Sample `json:"samples"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Content-Encoding") != "gzip" {
			t.Error("expected gzip body")
		}
		zr, err := gzip.NewReader(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		if err := json.NewDecoder(zr).Decode(&got); err != nil {
			t.Error(err)
		}
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()
	req, _ := JSONOrNull([]byte(`{"inputs":[]}`))
	bad, ok := JSONOrNull([]byte(`{nope`))
	if ok || string(bad) != "null" {
		t.Fatal("invalid JSON must become null")
	}
	sink := &HTTPSink{URL: srv.URL, Client: srv.Client()}
	err := sink.Send(context.Background(), []*Sample{{Model: "multimodal", Request: req,
		Primary: Exchange{Version: "v1", Status: 200, Response: bad}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Samples) != 1 || got.Samples[0].Primary.Version != "v1" || string(got.Samples[0].Request) != `{"inputs":[]}` {
		t.Fatalf("unexpected payload %+v", got)
	}
}

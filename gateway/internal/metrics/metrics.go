// Package metrics is a small, dependency-free implementation of Prometheus
// counters, gauges and histograms with the text exposition format (0.0.4).
//
// The gateway deliberately avoids third-party modules so it builds anywhere
// with only the Go toolchain; the subset here covers what it exports.
package metrics

import (
	"bufio"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// Registry holds metric families and renders them for scraping.
type Registry struct {
	mu       sync.Mutex
	families []family
	names    map[string]bool
}

type family interface {
	write(w *bufio.Writer)
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return &Registry{names: map[string]bool{}} }

func (r *Registry) register(name string, f family) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.names[name] {
		panic("metrics: duplicate metric " + name)
	}
	r.names[name] = true
	r.families = append(r.families, f)
}

// Handler serves the registry in the Prometheus text format.
func (r *Registry) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		bw := bufio.NewWriter(w)
		r.WriteTo(bw)
		_ = bw.Flush()
	})
}

// WriteTo renders every family.
func (r *Registry) WriteTo(w *bufio.Writer) {
	r.mu.Lock()
	fams := append([]family(nil), r.families...)
	r.mu.Unlock()
	for _, f := range fams {
		f.write(w)
	}
}

// ------------------------------------------------------------------ labels

type vec[T any] struct {
	name, help, kind string
	labels           []string
	mu               sync.RWMutex
	series           map[string]*T
	values           map[string][]string
	newT             func() *T
}

func newVec[T any](name, help, kind string, labels []string, newT func() *T) *vec[T] {
	return &vec[T]{name: name, help: help, kind: kind, labels: labels,
		series: map[string]*T{}, values: map[string][]string{}, newT: newT}
}

func (v *vec[T]) with(values ...string) *T {
	if len(values) != len(v.labels) {
		panic(fmt.Sprintf("metrics: %s wants %d label values, got %d", v.name, len(v.labels), len(values)))
	}
	key := strings.Join(values, "\xff")
	v.mu.RLock()
	s, ok := v.series[key]
	v.mu.RUnlock()
	if ok {
		return s
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if s, ok = v.series[key]; ok {
		return s
	}
	s = v.newT()
	v.series[key] = s
	v.values[key] = append([]string(nil), values...)
	return s
}

func (v *vec[T]) sorted() []string {
	v.mu.RLock()
	keys := make([]string, 0, len(v.series))
	for k := range v.series {
		keys = append(keys, k)
	}
	v.mu.RUnlock()
	sort.Strings(keys)
	return keys
}

func (v *vec[T]) header(w *bufio.Writer) {
	fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s %s\n", v.name, v.help, v.name, v.kind)
}

func labelString(names, values []string, extra ...string) string {
	if len(names) == 0 && len(extra) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteByte('{')
	n := 0
	add := func(k, val string) {
		if n > 0 {
			b.WriteByte(',')
		}
		b.WriteString(k)
		b.WriteString(`="`)
		b.WriteString(escape(val))
		b.WriteByte('"')
		n++
	}
	for i, k := range names {
		add(k, values[i])
	}
	for i := 0; i+1 < len(extra); i += 2 {
		add(extra[i], extra[i+1])
	}
	b.WriteByte('}')
	return b.String()
}

func escape(s string) string {
	if !strings.ContainsAny(s, "\\\"\n") {
		return s
	}
	return strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`).Replace(s)
}

func formatFloat(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "+Inf"
	case math.IsInf(f, -1):
		return "-Inf"
	case math.IsNaN(f):
		return "NaN"
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

// ------------------------------------------------------------------ counter

// Counter is a monotonically increasing value.
type Counter struct{ bits atomic.Uint64 }

// Inc adds one.
func (c *Counter) Inc() { c.Add(1) }

// Add adds a non-negative value.
func (c *Counter) Add(delta float64) {
	if delta < 0 {
		panic("metrics: counter cannot decrease")
	}
	addFloat(&c.bits, delta)
}

// Value returns the current value.
func (c *Counter) Value() float64 { return math.Float64frombits(c.bits.Load()) }

func addFloat(bits *atomic.Uint64, delta float64) {
	for {
		old := bits.Load()
		if bits.CompareAndSwap(old, math.Float64bits(math.Float64frombits(old)+delta)) {
			return
		}
	}
}

// CounterVec is a counter partitioned by labels.
type CounterVec struct{ v *vec[Counter] }

// NewCounterVec registers a counter family.
func (r *Registry) NewCounterVec(name, help string, labels ...string) *CounterVec {
	c := &CounterVec{newVec(name, help, "counter", labels, func() *Counter { return &Counter{} })}
	r.register(name, c)
	return c
}

// With returns the counter for the given label values.
func (c *CounterVec) With(values ...string) *Counter { return c.v.with(values...) }

func (c *CounterVec) write(w *bufio.Writer) {
	c.v.header(w)
	for _, k := range c.v.sorted() {
		fmt.Fprintf(w, "%s%s %s\n", c.v.name, labelString(c.v.labels, c.v.values[k]), formatFloat(c.v.series[k].Value()))
	}
}

// ------------------------------------------------------------------ gauge

// Gauge is a value that can go up and down.
type Gauge struct{ bits atomic.Uint64 }

// Set replaces the value.
func (g *Gauge) Set(v float64) { g.bits.Store(math.Float64bits(v)) }

// Add changes the value by delta.
func (g *Gauge) Add(delta float64) { addFloat(&g.bits, delta) }

// Value returns the current value.
func (g *Gauge) Value() float64 { return math.Float64frombits(g.bits.Load()) }

// GaugeVec is a gauge partitioned by labels.
type GaugeVec struct{ v *vec[Gauge] }

// NewGaugeVec registers a gauge family.
func (r *Registry) NewGaugeVec(name, help string, labels ...string) *GaugeVec {
	g := &GaugeVec{newVec(name, help, "gauge", labels, func() *Gauge { return &Gauge{} })}
	r.register(name, g)
	return g
}

// With returns the gauge for the given label values.
func (g *GaugeVec) With(values ...string) *Gauge { return g.v.with(values...) }

func (g *GaugeVec) write(w *bufio.Writer) {
	g.v.header(w)
	for _, k := range g.v.sorted() {
		fmt.Fprintf(w, "%s%s %s\n", g.v.name, labelString(g.v.labels, g.v.values[k]), formatFloat(g.v.series[k].Value()))
	}
}

// ------------------------------------------------------------------ histogram

// Histogram counts observations into cumulative buckets.
type Histogram struct {
	mu      sync.Mutex
	bounds  []float64
	buckets []uint64
	count   uint64
	sum     float64
}

// Observe records one value.
func (h *Histogram) Observe(v float64) {
	i := sort.SearchFloat64s(h.bounds, v) // first bound >= v
	h.mu.Lock()
	if i < len(h.buckets) {
		h.buckets[i]++
	}
	h.count++
	h.sum += v
	h.mu.Unlock()
}

// Snapshot returns cumulative bucket counts, the total count and the sum.
func (h *Histogram) Snapshot() (cumulative []uint64, count uint64, sum float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	cumulative = make([]uint64, len(h.buckets))
	var acc uint64
	for i, b := range h.buckets {
		acc += b
		cumulative[i] = acc
	}
	return cumulative, h.count, h.sum
}

// HistogramVec is a histogram partitioned by labels.
type HistogramVec struct {
	v      *vec[Histogram]
	bounds []float64
}

// NewHistogramVec registers a histogram family with the given upper bounds.
func (r *Registry) NewHistogramVec(name, help string, bounds []float64, labels ...string) *HistogramVec {
	b := append([]float64(nil), bounds...)
	sort.Float64s(b)
	h := &HistogramVec{bounds: b}
	h.v = newVec(name, help, "histogram", labels, func() *Histogram {
		return &Histogram{bounds: b, buckets: make([]uint64, len(b))}
	})
	r.register(name, h)
	return h
}

// With returns the histogram for the given label values.
func (h *HistogramVec) With(values ...string) *Histogram { return h.v.with(values...) }

func (h *HistogramVec) write(w *bufio.Writer) {
	h.v.header(w)
	for _, k := range h.v.sorted() {
		cum, count, sum := h.v.series[k].Snapshot()
		vals := h.v.values[k]
		for i, bound := range h.bounds {
			fmt.Fprintf(w, "%s_bucket%s %d\n", h.v.name, labelString(h.v.labels, vals, "le", formatFloat(bound)), cum[i])
		}
		fmt.Fprintf(w, "%s_bucket%s %d\n", h.v.name, labelString(h.v.labels, vals, "le", "+Inf"), count)
		fmt.Fprintf(w, "%s_sum%s %s\n", h.v.name, labelString(h.v.labels, vals), formatFloat(sum))
		fmt.Fprintf(w, "%s_count%s %d\n", h.v.name, labelString(h.v.labels, vals), count)
	}
}

// LatencyBuckets are upper bounds in seconds suited to model serving.
var LatencyBuckets = []float64{0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10}

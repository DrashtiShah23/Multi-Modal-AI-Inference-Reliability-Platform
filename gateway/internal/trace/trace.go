// Package trace implements W3C Trace Context (traceparent) propagation so the
// gateway's requests join the same distributed trace as the client, the model
// server and the analyzer (which links its spans via OpenTelemetry).
package trace

import (
	"encoding/hex"
	"math/rand/v2"
	"strings"
)

// Context identifies one span within a trace.
type Context struct {
	TraceID [16]byte
	SpanID  [8]byte
	Flags   byte
}

// Header is the HTTP header carrying the context.
const Header = "traceparent"

// Parse reads a version-00 traceparent value. It rejects all-zero IDs.
func Parse(v string) (Context, bool) {
	var c Context
	v = strings.TrimSpace(v)
	if len(v) != 55 || v[2] != '-' || v[35] != '-' || v[52] != '-' || v[:2] != "00" {
		return c, false
	}
	if _, err := hex.Decode(c.TraceID[:], []byte(v[3:35])); err != nil {
		return c, false
	}
	if _, err := hex.Decode(c.SpanID[:], []byte(v[36:52])); err != nil {
		return c, false
	}
	var flags [1]byte
	if _, err := hex.Decode(flags[:], []byte(v[53:55])); err != nil {
		return c, false
	}
	c.Flags = flags[0]
	if c.TraceID == ([16]byte{}) || c.SpanID == ([8]byte{}) {
		return c, false
	}
	return c, true
}

// New starts a sampled trace.
func New() Context {
	var c Context
	fill(c.TraceID[:])
	fill(c.SpanID[:])
	c.Flags = 0x01
	return c
}

// FromHeader continues the caller's trace, or starts one if the header is absent or invalid.
func FromHeader(v string) Context {
	if c, ok := Parse(v); ok {
		return c
	}
	return New()
}

// Child returns a new span in the same trace.
func (c Context) Child() Context {
	child := c
	fill(child.SpanID[:])
	return child
}

// String renders the traceparent value.
func (c Context) String() string {
	return "00-" + hex.EncodeToString(c.TraceID[:]) + "-" + hex.EncodeToString(c.SpanID[:]) + "-" +
		hex.EncodeToString([]byte{c.Flags})
}

// TraceIDHex returns the 32-character trace ID.
func (c Context) TraceIDHex() string { return hex.EncodeToString(c.TraceID[:]) }

func fill(b []byte) {
	for {
		for i := 0; i < len(b); i += 8 {
			v := rand.Uint64()
			for j := 0; j < 8 && i+j < len(b); j++ {
				b[i+j] = byte(v >> (8 * j))
			}
		}
		for _, x := range b {
			if x != 0 {
				return
			}
		}
	}
}

package trace

import "testing"

func TestParseRoundTrip(t *testing.T) {
	const h = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
	c, ok := Parse(h)
	if !ok || c.String() != h || c.TraceIDHex() != "4bf92f3577b34da6a3ce929d0e0e4736" {
		t.Fatalf("Parse(%q) = %v, %v", h, c, ok)
	}
	child := c.Child()
	if child.TraceID != c.TraceID || child.SpanID == c.SpanID || child.Flags != c.Flags {
		t.Fatal("child must keep trace id and flags and get a new span id")
	}
}

func TestRejectsInvalid(t *testing.T) {
	for _, h := range []string{
		"",
		"01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
		"00-00000000000000000000000000000000-00f067aa0ba902b7-01",
		"00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
		"00-4bf92f3577b34da6a3ce929d0e0e47zz-00f067aa0ba902b7-01",
	} {
		if _, ok := Parse(h); ok {
			t.Errorf("accepted %q", h)
		}
	}
	if c := FromHeader("garbage"); c.TraceID == ([16]byte{}) {
		t.Fatal("FromHeader should start a new trace")
	}
}

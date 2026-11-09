package breaker

import (
	"testing"
	"time"
)

func TestOpensHalfOpensAndCloses(t *testing.T) {
	now := time.Unix(0, 0)
	b := New(3, 10*time.Second)
	b.SetClock(func() time.Time { return now })

	for i := 0; i < 2; i++ {
		if !b.Allow() {
			t.Fatal("closed breaker should allow")
		}
		b.Record(false)
	}
	b.Allow()
	b.Record(true) // success resets the consecutive count
	for i := 0; i < 3; i++ {
		b.Allow()
		b.Record(false)
	}
	if b.State() != Open || b.Allow() {
		t.Fatal("breaker should be open and reject")
	}

	now = now.Add(11 * time.Second)
	if !b.Allow() || b.State() != HalfOpen {
		t.Fatal("after cooldown one probe is allowed")
	}
	if b.Allow() {
		t.Fatal("only one probe at a time")
	}
	b.Record(false)
	if b.State() != Open {
		t.Fatal("failed probe reopens")
	}

	now = now.Add(11 * time.Second)
	b.Allow()
	b.Record(true)
	if b.State() != Closed || !b.Allow() {
		t.Fatal("successful probe closes")
	}
}

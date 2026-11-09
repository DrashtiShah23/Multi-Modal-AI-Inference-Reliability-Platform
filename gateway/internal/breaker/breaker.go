// Package breaker is a consecutive-failure circuit breaker. The gateway puts
// one in front of the candidate model so a broken release stops receiving
// mirrored traffic instead of piling up goroutines and timeouts.
package breaker

import (
	"sync"
	"time"
)

// State of the breaker.
type State int

const (
	Closed State = iota
	HalfOpen
	Open
)

func (s State) String() string {
	return [...]string{"closed", "half_open", "open"}[s]
}

// Breaker opens after Threshold consecutive failures, stays open for Cooldown,
// then lets a single probe through (half-open); the probe's outcome closes or
// reopens it.
type Breaker struct {
	mu        sync.Mutex
	threshold int
	cooldown  time.Duration
	now       func() time.Time
	state     State
	failures  int
	openedAt  time.Time
	probing   bool
}

// New returns a closed breaker.
func New(threshold int, cooldown time.Duration) *Breaker {
	if threshold < 1 {
		threshold = 1
	}
	return &Breaker{threshold: threshold, cooldown: cooldown, now: time.Now}
}

// SetClock replaces the time source (tests).
func (b *Breaker) SetClock(now func() time.Time) { b.now = now }

// Allow reports whether a call may proceed. Every allowed call must be
// followed by exactly one Record.
func (b *Breaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch b.state {
	case Closed:
		return true
	case Open:
		if b.now().Sub(b.openedAt) < b.cooldown {
			return false
		}
		b.state = HalfOpen
		b.probing = true
		return true
	default: // HalfOpen
		if b.probing {
			return false
		}
		b.probing = true
		return true
	}
}

// Record reports the outcome of an allowed call.
func (b *Breaker) Record(success bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.state == HalfOpen {
		b.probing = false
		if success {
			b.state, b.failures = Closed, 0
		} else {
			b.state, b.openedAt = Open, b.now()
		}
		return
	}
	if success {
		b.failures = 0
		return
	}
	b.failures++
	if b.state == Closed && b.failures >= b.threshold {
		b.state, b.openedAt = Open, b.now()
	}
}

// State returns the current state.
func (b *Breaker) State() State {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.state
}

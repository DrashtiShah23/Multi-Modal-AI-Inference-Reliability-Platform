// Command loadgen drives open-loop load at a fixed rate and prints latency
// percentiles as JSON. Used by `make bench-overhead` to measure what the
// gateway adds on top of a direct call to the model server.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/DrashtiShah23/bellwether/gateway/internal/loadtest"
)

func main() {
	url := flag.String("url", "http://localhost:8080/v2/models/multimodal/infer", "target URL")
	bodyFile := flag.String("body", "", "file with the JSON request body")
	rate := flag.Float64("rate", 500, "requests per second")
	duration := flag.Duration("duration", 30*time.Second, "measured duration")
	warmup := flag.Duration("warmup", 5*time.Second, "unmeasured warmup")
	workers := flag.Int("workers", 256, "concurrent workers")
	flag.Parse()

	body := []byte(`{}`)
	if *bodyFile != "" {
		b, err := os.ReadFile(*bodyFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		body = b
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	res := loadtest.Run(ctx, loadtest.Config{URL: *url, Body: body, Rate: *rate, Duration: *duration,
		Warmup: *warmup, Workers: *workers})
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(res)
}

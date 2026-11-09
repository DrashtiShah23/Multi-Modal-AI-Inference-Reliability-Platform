// Command stubmodel is a KServe v2 endpoint that returns a fixed response
// after an optional delay. It isolates the gateway's own overhead in load
// tests, since a real model's latency variance would swamp it.
package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand/v2"
	"net/http"
	"strings"
	"time"
)

func main() {
	listen := flag.String("listen", ":8000", "listen address")
	version := flag.String("version", "v1", "model_version to report")
	delay := flag.Duration("delay", 0, "fixed service time per request")
	jitter := flag.Duration("jitter", 0, "uniform random extra service time")
	dim := flag.Int("dim", 32, "embedding dimension in the canned response")
	flag.Parse()

	vec := strings.TrimSuffix(strings.Repeat("0.17677669,", *dim), ",") // unit norm for dim 32
	body := fmt.Sprintf(`{"model_name":"multimodal","model_version":%q,"outputs":[`+
		`{"name":"image_embedding","shape":[1,%d],"datatype":"FP32","data":[%s]},`+
		`{"name":"text_embedding","shape":[1,%d],"datatype":"FP32","data":[%s]},`+
		`{"name":"similarity","shape":[1],"datatype":"FP32","data":[1.0]}]}`, *version, *dim, vec, *dim, vec)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v2/models/{model}/infer", func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		d := *delay
		if *jitter > 0 {
			d += time.Duration(rand.Int64N(int64(*jitter)))
		}
		if d > 0 {
			time.Sleep(d)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	})
	// Also accepts the gateway's sample batches, so it can stand in for the analyzer in overhead tests.
	mux.HandleFunc("POST /v1/samples:batch", func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusAccepted)
	})
	mux.HandleFunc("GET /v2/health/ready", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("GET /v2/health/live", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	log.Fatal((&http.Server{Addr: *listen, Handler: mux, ReadHeaderTimeout: 5 * time.Second}).ListenAndServe())
}

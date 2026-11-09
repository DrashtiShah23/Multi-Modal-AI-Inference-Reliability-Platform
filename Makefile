PY ?= python3
MODELS := build/models
BIN := build/bin

.PHONY: install models gateway proto rules test test-go lint bench e2e overhead run-local docker k8s-render publish-triton clean

install:            ## editable install with dev extras
	$(PY) -m pip install -e ".[dev,postgres,otlp]"

models:             ## build the v1/v2/v3 ONNX releases as Triton model repositories
	$(PY) -m bellwether.models.build --out $(MODELS)

gateway:            ## build the Go gateway, load generator and stub model
	cd gateway && go build -o ../$(BIN)/ ./cmd/...

proto:              ## regenerate Python gRPC stubs
	$(PY) -m grpc_tools.protoc -I proto --python_out=src --pyi_out=src --grpc_python_out=src proto/bellwether/v1/analyzer.proto

rules:              ## regenerate the PrometheusRule from deploy/prometheus/rules.yml
	$(PY) scripts/gen_rules.py

test:               ## Python tests (set BELLWETHER_TEST_PG_URL to include PostgreSQL)
	$(PY) -m pytest

test-go:
	cd gateway && gofmt -l . && go vet ./... && go test -race -count=1 ./...

lint:
	ruff check src tests scripts

bench: models       ## fault-injection benchmark (about 3 minutes on 2 cores)
	$(PY) -m bellwether.bench.run_faults --models $(MODELS)

e2e: models gateway ## canary drill with every component as a separate process
	$(PY) -m bellwether.bench.e2e --models $(MODELS) --gateway-bin $(BIN)/gateway

overhead: gateway   ## gateway latency overhead with Go stubs
	$(PY) -m bellwether.bench.overhead --bin $(BIN)

run-local: models gateway  ## run the stack without Docker (Ctrl-C to stop)
	scripts/run_local.sh

docker:
	docker build -t bellwether/gateway:dev gateway
	docker build --target analyzer -t bellwether/analyzer:dev .
	docker build --target modelserver -t bellwether/modelserver:dev .

k8s-render:         ## render manifests: make k8s-render PROJECT_ID=... REGION=... BUCKET=...
	kustomize build deploy/k8s/overlays/gke | sed -e "s/PROJECT_ID/$(PROJECT_ID)/g" -e "s/REGION/$(REGION)/g" -e "s/BUCKET/$(BUCKET)/g"

publish-triton: models  ## upload releases to gs://$(BUCKET)/releases/<n>/
	scripts/publish_triton.sh $(MODELS) $(BUCKET)

clean:
	rm -rf build bench/results *.db

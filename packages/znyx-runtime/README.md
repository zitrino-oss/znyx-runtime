# znyx-runtime

A lightweight, dependency-minimal FastAPI service that evaluates LLM traffic
against guardrail policies. Rules-only out of the box; it gains ML detection when
pointed at a Znyx inference sidecar over HTTP.

```bash
pip install znyx-runtime
znyx-runtime serve --port 8080
```

Or with Docker:

```bash
docker run -p 8080:8080 znyx/runtime
```

The runtime is deliberately thin: FastAPI, uvicorn, httpx, and
[`znyx-core`](https://github.com/zitrino-oss/znyx-runtime) (the detection engine).
No database, no heavy ML libraries. Point it at a sidecar endpoint to enable
model-backed detectors; without one, it runs the deterministic rules path.

A model-backed detector is held at advisory `WARN` until it has been measured
against a labelled dataset. The second subcommand is how you do that, with no
control plane involved:

```bash
znyx-runtime benchmark --bundle policies.yaml --detector toxicity \
    --dataset labelled.jsonl --stamp
```

See the [repository README](https://github.com/zitrino-oss/znyx-runtime) for
configuration, deployment manifests, the evaluate API, and the dataset format and
gate thresholds that command uses.

## License

Apache-2.0

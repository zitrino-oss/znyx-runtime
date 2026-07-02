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

See the [repository README](https://github.com/zitrino-oss/znyx-runtime) for
configuration, deployment manifests, and the evaluate API.

## License

Apache-2.0

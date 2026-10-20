# znyx-runtime

The FastAPI service inside [`znyx`](https://github.com/zitrino-oss/znyx-runtime):
it evaluates LLM traffic against guardrail policies. Rules-only out of the box; it
gains ML detection when pointed at a Znyx model server over HTTP.

**Most people should install `znyx` instead.** It is one package, one command and
one version, and it pins `znyx-runtime` for you:

```bash
pip install znyx
znyx serve
```

Or with Docker:

```bash
docker run -p 8080:8080 znyx/znyx
```

`znyx serve` works from an empty folder (built-in starter policy, API key created
on first run, one port for every API and the local dashboard at `/dashboard`).
`znyx init` writes a `znyx.yaml` and a `policies.yaml` to edit, `--workers N|auto`
runs several worker processes on the one port, and `--ml` also starts the ML model
server. See the [repository README](https://github.com/zitrino-oss/znyx-runtime)
for all of it.

## Using znyx-runtime on its own

The `znyx-runtime` pip package and the `znyx/runtime` image keep working for a
deprecation window, so existing deployments do not need to change on day one:

```bash
pip install znyx-runtime
znyx-runtime serve --port 8080

docker run -p 8080:8080 znyx/runtime
```

To move over, replace `znyx-runtime serve` with `znyx serve`,
`pip install znyx-runtime` with `pip install znyx`, and the `znyx/runtime` image
with `znyx/znyx` (or `znyx/znyx:ml` to include the ML runtime).

The runtime is deliberately thin: FastAPI, uvicorn, httpx, and
[`znyx-core`](https://github.com/zitrino-oss/znyx-runtime) (the detection engine).
No database, no heavy ML libraries. Point it at a model server to enable
model-backed detectors; without one, it runs the deterministic rules path.

A model-backed detector is held at advisory `WARN` until it has been measured
against a labelled dataset. The benchmark command is how you do that, with no
control plane involved (`znyx benchmark` and `znyx-runtime benchmark` are the same
command):

```bash
znyx benchmark --bundle policies.yaml --detector toxicity \
    --dataset labelled.jsonl --stamp
```

See the [repository README](https://github.com/zitrino-oss/znyx-runtime) for
configuration, deployment manifests, the evaluate API, and the dataset format and
gate thresholds that command uses.

## License

Apache-2.0

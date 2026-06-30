# Znyx Runtime

[![Security](https://github.com/zitrino-oss/znyx-runtime/actions/workflows/security.yml/badge.svg)](https://github.com/zitrino-oss/znyx-runtime/actions/workflows/security.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

A stateless, zero-dependency-on-a-database guardrails evaluation server for LLM
applications. Znyx Runtime scans model **inputs**, **outputs**, and **tool calls**
against a YAML policy of detectors — PII, secrets, jailbreak, toxicity, prompt
injection/exfiltration, hallucination, and more — and returns an allow / block /
redact decision. It runs entirely on your machine; your data never leaves it in
local mode.

## Quick start

### Docker (recommended)

```bash
docker build -t znyx-runtime:local .
docker run -p 8080:8080 znyx-runtime:local

curl localhost:8080/healthz   # -> {"status":"ok","version":"1.0.0"}
```

### Python

```bash
pip install -r requirements.txt
ZNYX_MODE=local ZNYX_POLICY_PATH=./config/policies.yaml \
  uvicorn app.runtime.main:app --host 0.0.0.0 --port 8080
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/evaluate/input` | Evaluate a user/model input |
| `POST` | `/v1/evaluate/output` | Evaluate a model output |
| `POST` | `/v1/evaluate/tool` | Evaluate a tool call |
| `POST` | `/v1/evaluate/stream` | Streaming evaluation |
| `GET` | `/v1/bundle/status` | Active policy bundle status |
| `GET` | `/healthz`, `/readyz` | Liveness / readiness |
| `GET` | `/metrics` | Prometheus metrics |

Evaluation endpoints require authentication when enabled (see below).

## Configuration

All configuration is via environment variables (`ZNYX_*`; legacy `GUARDRAILS_*`
names are accepted as fallbacks).

| Variable | Default | Description |
|---|---|---|
| `ZNYX_MODE` | `local` | `local` (YAML/bundle file) or `managed` (fetch from control plane) |
| `ZNYX_POLICY_PATH` | `./config/policies.yaml` | Policy file (local mode) |
| `ZNYX_FAIL_MODE` | `closed` | `closed` blocks when no policy resolves; `open` allows |
| `RUNTIME_REQUIRE_AUTH` | `true` | Require API-key auth (always enforced in production) |
| `RUNTIME_API_KEY` | — | Required when auth is enabled |
| `ZNYX_TELEMETRY` | `true` | Anonymous install heartbeat — see below |
| `PORT` | `8080` | Listen port |

### Authentication

Auth is **on by default**. Set `RUNTIME_API_KEY` and send it as `X-API-Key` or
`Authorization: Bearer <key>`. In production (`ZNYX_ENV=production`) auth is always
enforced and cannot be disabled.

### Telemetry

Znyx Runtime sends an **anonymous daily install heartbeat** to Zitrino, on by
default. It contains no request data or content. Opt out at any time:

```bash
export ZNYX_TELEMETRY=false
```

Per-evaluation telemetry is **off** in local mode and only enabled in managed mode.

## Security

Please report vulnerabilities privately — see [SECURITY.md](./SECURITY.md). Do not
open public issues for security reports. The samples under `config/benchmarks/` are
**synthetic** detector test data, not real secrets.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) and our [Code of Conduct](./CODE_OF_CONDUCT.md).

## License

Apache-2.0 — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

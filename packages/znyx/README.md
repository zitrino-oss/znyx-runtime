# znyx

One install for AI guardrails. `pip install znyx` gives you the `znyx` command:
one package, one command, one port, one config file and one version.

Znyx evaluates prompts, model output, tool calls and agent steps against a policy
and returns an allow, warn, redact or block decision, inside your own perimeter.

## Quickstart

```bash
pip install znyx
znyx serve
```

or

```bash
docker run -p 8080:8080 znyx/znyx
```

`znyx serve` works from an empty folder. It starts on `http://127.0.0.1:8080` with
a built-in starter policy, creates an API key on first run in `~/.znyx/api-key`
(show it with `znyx key`), and prints a dashboard URL and a curl example. Every
API (evaluate, streaming, metrics) and the local dashboard at `/dashboard` are on
that one port. The container image listens on `0.0.0.0`.

```bash
curl -s http://127.0.0.1:8080/v1/evaluate/input \
  -H "X-API-Key: $(znyx key 2>/dev/null)" \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"1","tenant_id":"local","app_id":"demo","text":"My card is 4111 1111 1111 1111"}'
```

To edit the policy and settings, run `znyx init`: it writes `znyx.yaml` and
`policies.yaml` for you to change.

## Commands

| Command | What it does |
|---|---|
| `znyx serve` (alias `znyx start`) | Run Znyx. Flags: `--config`, `--host`, `--port`, `--workers`, `--policy`, `--mode`, `--ml` / `--no-ml`, `--ml-port`, `--airgapped`, `--log-level`, `--reload` |
| `znyx init [-o DIR] [--force]` | Write `znyx.yaml` and `policies.yaml` to edit |
| `znyx config [--json]` | Show the effective settings and where each came from |
| `znyx status` | Show the running workers |
| `znyx key [--rotate]` | Show this install's API key, or make a new one |
| `znyx version [--json]` | Show the one platform version and its components |
| `znyx models list` / `fetch` | List vetted model weights, or fetch and sha256-pin one |
| `znyx benchmark` | Certify a model-backed detector so it can block |

`--workers N|auto` runs N worker processes on the one port (`auto` is one per CPU,
up to 4). Throughput grows roughly in line: 9.3, 16.9 and 28.9 requests per second
at 1, 2 and 4 workers, measured with 8 KB inputs. `--ml` also starts the ML model
server, on loopback only.

## One version

`znyx` pins `znyx-core`, `znyx-runtime` and `znyx-inference` at exactly its own
version, so you never mix versions. `znyx version` shows the one version and its
components.

## Extras

| Install | Adds |
|---|---|
| `pip install "znyx[ml]"` | The ONNX CPU model runtime for the built-in model server (`znyx serve --ml`) |
| `pip install "znyx[export]"` | Offline model export (pulls torch) |
| `pip install "znyx[torch]"` | The generative guard runner |
| `pip install "znyx[redis]"` | Limits and budgets shared across replicas |

The container images are `znyx/znyx` (rules only) and `znyx/znyx:ml` (with the ML
runtime).

## Configuration

One file, `znyx.yaml`, configures everything; every setting is optional. Precedence,
highest first: command-line flags, then environment variables, then the file, then
defaults. Strings in the file can reference the environment as `${NAME}`, so
secrets need not be written into it. `znyx config` shows where each effective
value came from.

Air-gapped installs set `airgapped: true` (or pass `--airgapped`): no outbound
calls, no heartbeat and no model downloads, so mount your weights.

## Upgrading from znyx-runtime and znyx-inference

The `znyx-runtime` and `znyx-inference` pip packages and the `znyx/runtime` and
`znyx/inference` images keep working for a deprecation window. To move over, use
`pip install znyx`, replace `znyx-runtime serve` with `znyx serve`, and use the
`znyx/znyx` image (`znyx/znyx:ml` in place of `znyx/inference`).

## Client SDKs

The language SDKs are separate and optional. They are for checks that are not an
LLM call (retrieval results, agent plans and steps, memory writes) and for calling
the check API directly. See [znyx-sdk](https://github.com/zitrino-oss/znyx-sdk).

## Links

- [Source and full documentation](https://github.com/zitrino-oss/znyx-runtime)
- [Which package do I install?](https://znyx.ai/which-package)
- [Documentation](https://znyx.ai/documentation)
- [Issues](https://github.com/zitrino-oss/znyx-runtime/issues)

Apache-2.0.

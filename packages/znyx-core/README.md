# znyx-core

The Znyx guardrails detection engine: detectors, policy resolution, scoring, and
orchestration for LLM safety. Importable in-process, with no server or HTTP hop.

```bash
pip install znyx-core
```

`znyx-core` is the engine that powers the [`znyx-runtime`](https://github.com/zitrino-oss/znyx-runtime)
service. Install it directly when you want to run detection inside your own Python
process rather than calling the runtime over HTTP.

It runs rules-only by default. Model-backed (ML) detection is an optional layer
served by a separate Znyx inference sidecar over HTTP; without it, every detector
has a deterministic rules path.

See the [repository README](https://github.com/zitrino-oss/znyx-runtime) for
configuration, the policy schema, and usage examples.

## License

Apache-2.0

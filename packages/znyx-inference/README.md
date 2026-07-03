# znyx-inference

The optional Znyx inference sidecar: an HTTP service that serves ML models for
model-backed guardrail detection. The [`znyx-runtime`](https://github.com/zitrino-oss/znyx-runtime)
calls it over HTTP to escalate from rules to ML. Without it, the runtime still
works fully on its deterministic rules path.

```bash
# boots dependency-free on a stub runner (no ML stack needed):
pip install znyx-inference

# to serve real model weights, add the heavy runtimes:
pip install "znyx-inference[models]"
```

## Weights are never bundled

This package ships **code only, no model weights.** You fetch the models you want
explicitly, sha256-pin them, and mount them. Nothing is downloaded implicitly on
the serving path. See [MODELS.md](MODELS.md) for the curated model list, licenses,
and the fetch-and-pin workflow, including which models carry special licenses.

## How it fits

```
your app -> znyx-runtime (rules) --HTTP--> znyx-inference (ML) -> local, pinned weights
```

The sidecar loads only local, sha256-verified weights, fails closed, and never
phones home. If a model is not present it stays on the stub runner and the runtime
falls back to rules per the policy's fallback mode.

See the [repository README](https://github.com/zitrino-oss/znyx-runtime) for the
full architecture.

## License

Apache-2.0. Model weights you fetch are governed by their own licenses (see MODELS.md).

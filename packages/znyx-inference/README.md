# znyx-inference

The Znyx ML model server: an HTTP service that serves ML models for model-backed
guardrail detection. The Znyx runtime calls it over HTTP to escalate from rules to
ML. Without it, the runtime still works fully on its deterministic rules path.

**Most people should not install this package directly.** It is part of
[`znyx`](https://github.com/zitrino-oss/znyx-runtime), which pins it at the same
version as everything else, and `znyx serve --ml` starts it for you, on loopback
only:

```bash
pip install "znyx[ml]"       # adds the lean CPU runtime (onnxruntime + tokenizers, no torch)
znyx serve --ml

# or in a container, with the ML runtime included
docker run -p 8080:8080 znyx/znyx:ml
```

Other extras of `znyx`: `znyx[export]` (offline model export, pulls torch) and
`znyx[torch]` (the generative guard runner). `znyx models list` and
`znyx models fetch --task <task>` list and fetch vetted weights.

## Using znyx-inference on its own

The `znyx-inference` pip package and the `znyx/inference` image keep working for a
deprecation window, for deployments that run the model server as a separate
service:

```bash
# boots dependency-free on a stub runner (no ML stack needed):
pip install znyx-inference

# to serve real model weights on CPU: lean, torch-free (onnxruntime + tokenizers):
pip install "znyx-inference[onnx]"

# offline only: export + quantize a checkpoint to a pinned ONNX artifact (heavy: torch + optimum):
pip install "znyx-inference[export]"
```

To move over, use `pip install "znyx[ml]"` with `znyx serve --ml`, or the
`znyx/znyx:ml` image in place of `znyx/inference`.

## Weights are never bundled

This package ships **code only, no model weights.** You fetch the models you want
explicitly, sha256-pin them, and mount them. Nothing is downloaded implicitly on
the serving path. In an air-gapped install (`airgapped: true` or `--airgapped`) no
download is attempted at all, so mount the weights. See [MODELS.md](MODELS.md) for
the curated model list, licenses, and the fetch-and-pin workflow, including which
models carry special licenses.

## How it fits

```
your app -> znyx (rules) --HTTP, loopback--> model server (ML) -> local, pinned weights
```

The model server loads only local, sha256-verified weights, fails closed, and never
phones home. If a model is not present it stays on the stub runner and the runtime
falls back to rules per the policy's fallback mode.

See the [repository README](https://github.com/zitrino-oss/znyx-runtime) for the
full architecture.

## License

Apache-2.0. Model weights you fetch are governed by their own licenses (see MODELS.md).

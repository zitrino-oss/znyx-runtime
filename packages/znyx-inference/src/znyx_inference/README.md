# ZNYX Inference Service (F3)

An **optional, separately-deployable** FastAPI sidecar that hosts local ML models
(transformer classifiers, embeddings, NLI cross-encoders, an optional guard-LLM) behind
the stable scoring contract the runtime's extended `RemoteDetector` (F0.5) speaks.

**The core OSS runtime and control plane gain zero heavy dependencies.** The serving image
installs the LEAN CPU stack only — the `[onnx]` extra: `onnxruntime` + `tokenizers` + `numpy`
(no torch, no transformers, no CUDA). That is why `Dockerfile.inference` is hundreds of MB,
not multi-GB. With no ML stack installed at all, the service still boots and serves on the
dependency-free **StubRunner** (deterministic heuristics — for dev/CI, not production quality).

**Three install extras, one clean boundary — export offline (heavy), serve ONNX (lean):**

| Extra | Contents | Where | Purpose |
|-------|----------|-------|---------|
| `[onnx]` | onnxruntime, tokenizers, numpy | the serving image | **default** — serve pinned CPU ONNX artifacts, torch-free |
| `[export]` | torch, optimum, transformers | offline / CI only | export + int8-quantize a checkpoint to a pinned ONNX artifact |
| `[torch]` | torch, transformers | opt-in | enable the generative `guard_llm` runner (Llama-Guard-style) |

The multi-GB torch/CUDA payload lives only in the offline `[export]` step and never ships in
the serving image. GPU, when you want it, is an onnxruntime Execution Provider swap
(`onnxruntime-gpu`), not a torch rebuild.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/infer/{task}` | Score `{text}` (or `{texts}`) → confidence contract |
| GET | `/healthz` | Liveness |
| GET | `/v1/models` | Loaded models + availability (feeds the control-plane model registry) |
| GET | `/v1/stats` | Cache + batcher metrics |

The contract: `{decision, risk_score, confidence, label_scores, calibrated_score,
threshold, model_version, latency_ms}`. `decision` is the canonical ZNYX set
(ALLOW/WARN/BLOCK/REDACT/TRANSFORM).

## Run

```bash
# Dev (StubRunner, no ML deps, port 8086):
uvicorn znyx_inference.main:app --port 8086

# Container (opt-in profile; mount pinned models at ./models):
docker compose -f deploy/docker-compose.inference.yml --profile inference up
```

CPU works out of the box (`[onnx]`, no GPU required). For **GPU**, swap the CPU wheel for
`onnxruntime-gpu` and expose the device — no torch/CUDA rebuild of the image is needed.

## Adding a pinned model (no implicit downloads)

The service **never pulls weights from the network at startup** — that would violate
`no_external_calls` and the "data never leaves the box" posture. You must pre-bake or
mount the model artifacts; they load with `local_files_only=True` and are **sha256-
verified on load (startup fails on mismatch)**.

1. Export + pin a vetted checkpoint to ONNX (offline, needs `[export]`):
   `python -m scripts.fetch_inference_model --task prompt_injection`. It writes a quantized
   `model.onnx` + `tokenizer.json` under `./models/<model_id>/` and prints the sha256 to pin.
2. Or compute the pinned digest yourself (a stable hash over the sorted files):
   ```python
   from znyx_inference.runners._artifacts import artifact_sha256
   print(artifact_sha256("./models/<model_id>"))
   ```
3. Wire the task via `ZNYX_INFERENCE_TASKS` (JSON map):
   ```json
   {"prompt_injection": {"runner": "classifier",
                          "model_id": "protectai/deberta-v3-base-prompt-injection-v2",
                          "revision": "main", "sha256": "<digest>", "threshold": 0.5}}
   ```

Runner kinds — the first five are served on CPU via onnxruntime (torch-free): `stub`
(default, no deps) · `classifier` (ONNX seq-classification) · `ner` (ONNX token
classification, unstructured PII) · `language` (ONNX language-ID + allow/block) · `nli`
(ONNX cross-encoder entailment) · `embedding` (ONNX feature-extraction + mean-pooled
centroid; needs `unsafe_examples`) · `guard_llm` (generative causal-LM safety classifier,
**needs the `[torch]` extra**). A runner whose deps or artifacts are missing is reported
`available: false` in `/v1/models` and returns **503** for its task — it never crashes the
service.

## Knobs (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `ZNYX_INFERENCE_TASKS` | stub set | JSON `{task: spec}` map |
| `ZNYX_INFERENCE_ARTIFACTS_DIR` | `~/.znyx/models` | local model root |
| `ZNYX_INFERENCE_REQUIRE_LOCAL_FILES` | `true` | never download weights |
| `ZNYX_INFERENCE_MAX_BATCH` | `16` | dynamic-batch size cap |
| `ZNYX_INFERENCE_MAX_WAIT_MS` | `10` | batch-coalesce window |
| `ZNYX_INFERENCE_MAX_QUEUE` | `256` | queue cap (over → 429) |
| `ZNYX_INFERENCE_BUDGET_MS` | `2000` | per-request latency budget (over → 429) |
| `ZNYX_INFERENCE_CACHE_SIZE` | `4096` | content-hash LRU size |

## Privacy posture

- **No telemetry, no egress.** The sidecar makes no outbound calls.
- Weights are operator-supplied and pinned; provenance (model_id/revision/sha256) flows
  into the control-plane `model_registry_entries` (feeds the model card).
- The caller's runtime decides whether a call to this service counts as egress
  (`inference.in_boundary`, F4): a co-located sidecar is in-boundary; a networked one is
  gated/audited like any remote endpoint.

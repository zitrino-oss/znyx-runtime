# Test suite

Behavioral tests for the three packages in this repo, run against them exactly
as installed. Self-contained: no network access, no ML dependencies, no
database - the inference tests run on the dependency-free stub runner.

## Running

```bash
pip install ./packages/znyx-core ./packages/znyx-runtime ./packages/znyx-inference
pip install pytest
pytest tests/ -q
```

## Layout

| File | Covers |
|---|---|
| `test_detectors.py` | Representative detectors: ALLOW / BLOCK / REDACT paths |
| `test_policy_bundle.py` | Policy YAML loading, `PolicyBundle` round-trip, legacy signature verification |
| `test_policy_bundle_v2.py` | v2 envelope signatures incl. the per-field mutation matrix (skips on builds without v2) |
| `test_streaming.py` | Streaming evaluator: release-after-verdict, block containment |
| `test_remote_detector.py` | Remote detector: fail-open/closed, malformed responses, retries, circuit breaker, SSRF guard |
| `test_rate_limit.py` | Rate-limit middleware: enforcement, headers, bypass, production gate |
| `test_runtime_api.py` | Runtime FastAPI app end to end: health, auth on/off, evaluate round-trips |
| `test_inference_service.py` | Inference sidecar: cache identity, batching bounds, stub service round-trip |

Notes:

- `conftest.py` marks the process as a non-production environment; with
  nothing set the packages default to production behavior (mandatory auth,
  Redis-only rate limits) which is not what tests exercise.
- Credential-shaped fixtures are constructed at runtime so no secret-shaped
  literal exists in the tree for scanners to flag.

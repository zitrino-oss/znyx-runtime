# Contributing to Znyx runtime

Thanks for your interest in improving Znyx. This repository is the canonical home
of the open-source runtime and engine; contributions are welcome here.

## Ground rules

- Open an issue before a large change so we can agree on the approach.
- Keep pull requests focused. One logical change per PR.
- All contributions are accepted under the Apache-2.0 license (see LICENSE). By
  submitting a PR you certify you have the right to contribute the code (DCO
  sign-off: add `Signed-off-by: Your Name <you@example.com>` to commits).

## Development setup

```bash
# editable installs so changes are picked up immediately
python -m venv .venv && source .venv/bin/activate
pip install -e packages/znyx-core -e packages/znyx-runtime

# run the service locally (non-prod, auth off for convenience)
ZNYX_ENV=development RUNTIME_REQUIRE_AUTH=false znyx-runtime serve
```

## Tests

Run the test suite before opening a PR. A change to product code must come with a
test that exercises it.

## Coding conventions

- Match the style of the surrounding code.
- The engine (`znyx-core`) must stay free of any dependency on a control plane,
  database, or the inference sidecar. It talks to the sidecar only over HTTP.
- The runtime (`znyx-runtime`) must stay dependency-light: no database, no heavy
  ML libraries.
- Prefer clear, deterministic detector logic with a rules-only fallback.

## Adding a detector

A detector lives in `packages/znyx-core/src/znyx_core/detectors/`, is wired into
the orchestrator, and must have a deterministic path that works without any ML
sidecar. Include tests covering both the benign and the triggering case.

# Contributing to Znyx Runtime

Thanks for your interest in contributing! This project is maintained by Zitrino
under the `zitrino-oss` organization.

## Ground rules

- By contributing, you agree your contributions are licensed under the project's
  [Apache-2.0 license](./LICENSE).
- Be respectful — see our [Code of Conduct](./CODE_OF_CONDUCT.md).
- For **security issues, do not open a public issue** — follow [SECURITY.md](./SECURITY.md).

## Prerequisites

- **Python** 3.11+
- Optional: **Docker** (for building/running the container image)

## Getting started

```bash
git clone https://github.com/zitrino-oss/znyx-runtime.git
cd znyx-runtime
pip install -r requirements.txt
# Run the server (hot reload)
ZNYX_MODE=local uvicorn app.runtime.main:app --reload --port 8080
```

## Development workflow

| Task | Command |
|------|---------|
| Run the app (hot reload) | `uvicorn app.runtime.main:app --reload --port 8080` |
| Byte-compile (build check) | `python -m compileall app` |
| SAST (high-severity gate) | `bandit -r app/ -lll -iii` |
| Dependency audit | `pip-audit -r requirements.txt` |
| Build the image | `docker build -t znyx-runtime:local .` |

## Submitting changes

1. Fork the repo and create a topic branch (`git checkout -b fix/short-description`).
2. Keep changes focused; one logical change per pull request.
3. Make sure CI, **Dependency Audit**, and **Security** workflows pass.
4. Write a clear PR description and link any related issue. Fill in the PR template.
5. Comments should describe what the code does — please avoid narrative or
   changelog-style comments in source.
6. A maintainer (`@zitrino-oss/maintainers`) will review.

Do not commit secrets — real credentials belong in environment variables, never
in the repo. Note that `config/benchmarks/` and `config/packs/` contain
intentional **synthetic** secret/PII samples used to test the detectors.

## Reporting bugs / requesting features

Use the issue templates. Include reproduction steps, expected vs. actual behavior,
and version/commit information.

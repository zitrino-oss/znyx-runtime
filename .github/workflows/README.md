# Workflows

| Workflow | File | Triggers | Purpose |
|---|---|---|---|
| CI | `ci.yml` | push to main, PRs | Installs znyx-core, znyx-runtime, and znyx-inference (lean, no ML extras), byte-compiles and import-checks them, runs the `tests/` suite on Python 3.11 and 3.12 (required), builds and install-smoke-tests each package's wheel, and runs a non-blocking `pip-audit` against the resolved environment. |
| Dependency Audit | `audit.yml` | push, PRs, weekly (Mon 08:00 UTC) | `pip-audit` against the per-package requirements files; fails on known CVEs. |
| Security | `security.yml` | push, PRs, weekly (Mon 06:00 UTC), manual | SAST (bandit, semgrep), secret scan (trivy fs), container image scan (trivy), and CodeQL. |

Notes:
- The secret scan skips `config/benchmarks/` and `config/packs/`, which contain
  synthetic detector test data (fake credentials/PII), not real secrets.
- The image scan ignores unfixable base-image CVEs but fails on anything fixable.

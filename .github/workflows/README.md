# Workflows

| Workflow | File | Triggers | Purpose |
|---|---|---|---|
| CI | `ci.yml` | push to main, PRs | Byte-compile the package — a fast build status check for branch protection. |
| Dependency Audit | `audit.yml` | push, PRs, weekly (Mon 08:00 UTC) | `pip-audit` against `requirements.txt`; fails on known CVEs. |
| Security | `security.yml` | push, PRs, weekly (Mon 06:00 UTC), manual | SAST (bandit, semgrep), secret scan (trivy fs), container image scan (trivy), and CodeQL. |

Notes:
- The secret scan skips `config/benchmarks/` and `config/packs/`, which contain
  synthetic detector test data (fake credentials/PII), not real secrets.
- The image scan ignores unfixable base-image CVEs but fails on anything fixable.

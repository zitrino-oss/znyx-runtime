# Security Policy

## Reporting a Vulnerability

We take the security of Znyx Runtime seriously. If you believe you have found a
security vulnerability, please report it to us privately.

**Please do not report security vulnerabilities through public GitHub issues.**

Use GitHub's **[Security → Report a vulnerability](https://github.com/zitrino-oss/znyx-runtime/security/advisories/new)**
to open a private security advisory for this repository. Please include:

- a description of the vulnerability and its impact,
- steps to reproduce (proof-of-concept, affected endpoints/inputs),
- the affected version or commit, and
- a suggested fix, if you have one.

You can expect an acknowledgement within **3 business days** and a more detailed
response within **10 business days** indicating the next steps. We will keep you
informed of the progress toward a fix and may ask for additional information.

Please give us a reasonable window to remediate before any public disclosure. We
are happy to credit reporters in the release notes unless you prefer to remain
anonymous.

## Supported Versions

Security fixes are applied to the latest released version on the `main` branch.

## Scope and Operational Notes

- **Authentication is on by default.** The runtime requires `RUNTIME_API_KEY`
  to be set whenever auth is enabled (always enforced in production). Do not
  disable `RUNTIME_REQUIRE_AUTH` outside local development.
- **No real secrets ship in this repo.** The samples under `config/benchmarks/`
  are synthetic fixtures used to exercise the secret/PII detectors. Automated
  secret scanners may flag them; they are intentional test data, not leaks.
- **Bind address.** The service binds `0.0.0.0` by design (it runs in a
  container). Restrict exposure with your network/ingress layer.
- Run the runtime as a non-root user (the provided Dockerfile already does).

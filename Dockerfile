FROM python:3.14-slim

WORKDIR /app

# Patch base-image build tooling: the slim base ships setuptools/wheel with
# known HIGH CVEs (e.g. CVE-2026-24049, CVE-2026-23949). Upgrade before
# installing dependencies. These are build-time only and not used at runtime.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install only runtime dependencies (no postgres)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY config/ config/

# Non-root user — create cache dir with correct ownership before dropping privileges.
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/.cache/guardrails && \
    chown -R appuser:appuser /app/.cache
USER appuser

# Environment defaults. The runtime config accepts both ZNYX_* (preferred)
# and legacy GUARDRAILS_* variable names; use the current brand here.
ENV ZNYX_MODE=local
ENV ZNYX_POLICY_PATH=./config/policies.yaml
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')"

CMD ["python", "-m", "uvicorn", "app.runtime.main:app", "--host", "0.0.0.0", "--port", "8080"]

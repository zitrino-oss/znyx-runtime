# syntax=docker/dockerfile:1
#
# One product, three image variants:
#
#   docker build --target znyx    -t znyx/znyx .       # rules-only, small
#   docker build --target znyx-ml -t znyx/znyx:ml .    # plus the ML model server runtime
#   docker build --target znyx-media -t znyx/znyx:media . # offline OCR/PDF/speech
#
#   docker run -p 8080:8080 znyx/znyx
#
# Both run `znyx serve`: one port (8080) for every API, worker processes sized to
# the container's CPU limit, the local dashboard at /dashboard. The ML image also
# starts the model server inside the container on loopback (never published), so
# no sidecar container is needed. Weights are never baked in: they are fetched
# explicitly and sha256-verified into /models, or mounted there for air-gapped use.
#
# Mount /data to keep the API key, the shared limits and the audit spools across
# container restarts. Set RUNTIME_API_KEY from a secret in any real deployment;
# otherwise a key is generated into /data on first start (`docker exec <c> znyx key`).
#
# Configure it like any install: mount a znyx.yaml and pass `--config`, or set
# environment variables. The image's own defaults (listen on 0.0.0.0, and in the ML
# image the model server) sit BELOW your file (ZNYX_DEFAULTS), so nothing the image
# sets can override what you configure.
#
# Dockerfile.runtime and Dockerfile.inference still build the old znyx/runtime and
# znyx/inference images for one deprecation window.

# -- wheels: built once, installed by both images (no source tree in the images) --
FROM python:3.14-slim AS wheels
WORKDIR /src
COPY packages/znyx-core packages/znyx-core
COPY packages/znyx-runtime packages/znyx-runtime
COPY packages/znyx-inference packages/znyx-inference
COPY packages/znyx packages/znyx
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels \
      packages/znyx-core packages/znyx-runtime packages/znyx-inference packages/znyx

# -- common base ------------------------------------------------------------------
FROM python:3.14-slim AS base
# Pull the latest Debian security patches so image scans do not flag stale base
# packages (as root, before the user drop below).
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 appuser && mkdir -p /app/config /data /models \
    && chown -R appuser:appuser /data /models
WORKDIR /app
COPY config/ /app/config/
COPY <<EOF /app/znyx-defaults.yaml
# Defaults of the znyx image, below any znyx.yaml you mount (see ZNYX_DEFAULTS).
version: 1
server:
  host: 0.0.0.0
  port: 8080
policy: /app/config/policies.yaml
EOF
# HOME=/data: every file the services write (key, limits, spools, install state) lands
# in one directory, so one mount keeps it and a read-only root filesystem works.
# Only settings that znyx.yaml does not cover are environment variables here.
ENV HOME=/data \
    ZNYX_DEFAULTS=/app/znyx-defaults.yaml \
    ZNYX_CACHE_DIR=/data/cache \
    PYTHONUNBUFFERED=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.getenv('PORT', '8080'))"

# -- znyx: rules only ---------------------------------------------------------------
FROM base AS znyx
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/*.whl
USER appuser
CMD ["znyx", "serve"]

# -- znyx-media: rules plus offline OCR/PDF/speech service ---------------------------
FROM znyx AS znyx-media
USER root
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    pip install --no-cache-dir "$(ls /wheels/znyx_inference-*.whl)[media]"
COPY <<EOF /app/znyx-defaults.yaml
version: 1
server:
  host: 0.0.0.0
  port: 8080
policy: /app/config/policies.yaml
ml:
  enabled: true
  artifacts_dir: /models
EOF
ENV ZNYX_SPEECH_MODEL_PATH=/models/speech
USER appuser
CMD ["znyx", "serve"]

# -- znyx-ml: with the model server's runtime ----------------------------------------
FROM base AS znyx-ml
# CPU-only torch first, from the PyTorch CPU index (about 200 MB instead of 2.5 GB with
# CUDA), for the export stack the console's model Install button uses. Serving itself
# is ONNX (no torch).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/znyx_core-*.whl /wheels/znyx_runtime-*.whl \
      "$(ls /wheels/znyx_inference-*.whl)[onnx,export]" "$(ls /wheels/znyx-*.whl)[ml]"
COPY <<EOF /app/znyx-defaults.yaml
# Defaults of the znyx:ml image, below any znyx.yaml you mount (see ZNYX_DEFAULTS).
version: 1
server:
  host: 0.0.0.0
  port: 8080
policy: /app/config/policies.yaml
ml:
  enabled: true
  artifacts_dir: /models
EOF
USER appuser
CMD ["znyx", "serve"]

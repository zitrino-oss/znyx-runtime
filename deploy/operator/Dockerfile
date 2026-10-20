# The Znyx operator image. Build from this directory:
#   docker build -t znyx/znyx-operator:0.1.0 deploy/operator
FROM python:3.12-slim
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 operator
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY znyx_operator/ znyx_operator/
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER 1000:1000
EXPOSE 8081
# Cluster-wide by default. To watch only some namespaces set
# ZNYX_OPERATOR_NAMESPACES=team-a,team-b (main.py turns it into kopf's namespaces).
# main.py runs `kopf run --standalone` semantics: no peering objects, one replica.
ENTRYPOINT ["python", "-m", "znyx_operator.main"]

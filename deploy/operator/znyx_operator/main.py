"""Entry point: ``python -m znyx_operator.main``.

Configuration (environment):
  ZNYX_OPERATOR_NAMESPACES  comma-separated namespaces to watch; empty watches all of them
  ZNYX_OPERATOR_LIVENESS    liveness endpoint (default http://0.0.0.0:8081/healthz)
  ZNYX_OPERATOR_LOG_LEVEL   INFO (default), DEBUG, ...

Runs in kopf's standalone mode (no peering objects), so run a single replica.
"""
from __future__ import annotations

import logging
import os
from typing import List


def namespaces_from_env(env=None) -> List[str]:
    env = os.environ if env is None else env
    return [n.strip() for n in env.get("ZNYX_OPERATOR_NAMESPACES", "").split(",") if n.strip()]


def main() -> None:
    import kopf

    from . import handlers  # noqa: F401  (registers the handlers)

    logging.basicConfig(
        level=os.environ.get("ZNYX_OPERATOR_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    namespaces = namespaces_from_env()
    kopf.run(
        standalone=True,
        clusterwide=not namespaces,
        namespaces=namespaces,
        liveness_endpoint=os.environ.get("ZNYX_OPERATOR_LIVENESS", "http://0.0.0.0:8081/healthz"),
    )


if __name__ == "__main__":
    main()

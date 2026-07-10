"""
Runtime configuration loaded from environment variables.

Env vars use the ZNYX_* prefix. Legacy GUARDRAILS_* names are accepted as
fallbacks so existing deployments continue to work during the transition.
"""
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _env(znyx: str, legacy: str, default: str = "") -> str:
    """Read ZNYX_* first, fall back to GUARDRAILS_* for backward compat."""
    return os.getenv(znyx, os.getenv(legacy, default))


@dataclass
class RuntimeConfig:
    """Configuration for the ZNYX Runtime."""

    # Mode: "local" (YAML/bundle file) or "managed" (fetch from control plane)
    mode: str = "local"

    # Path to policies.yaml or bundle.json (local mode)
    policy_path: str = "./config/policies.yaml"

    # Control plane URL (managed mode)
    control_plane_url: str = ""

    # Auth token for control plane API (managed mode)
    runtime_token: str = ""

    # Ed25519 public key PEM for bundle signature verification
    bundle_public_key: str = ""

    # Fail mode: "open" (allow all if no policy) or "closed" (block all)
    fail_mode: str = "closed"

    # Bundle poll interval in seconds (managed mode)
    bundle_poll_interval: int = 30

    # Whether to require signed bundles (True when a public key is configured)
    require_signed_bundles: bool = False

    # Per-evaluation telemetry (events sent to control plane)
    telemetry_enabled: bool = False

    # Anonymous install heartbeat (daily ping to Zitrino - opt-in, off by default)
    heartbeat_enabled: bool = False

    # Egress audit sink. Backend: "spool" (durable local JSON-lines, drained
    # by the control plane) or "noop" (opt-out). fail_mode "closed" makes a failed
    # audit write deny the egress (no silent un-audited egress); "open" logs and
    # proceeds. Used by the egress gate before any boundary-crossing call.
    audit_sink_mode: str = "spool"
    audit_fail_mode: str = "closed"
    audit_spool_path: str = ""  # empty → ~/.znyx/egress-audit.spool

    # Judge audit. The runtime spools each local-judge call to a durable
    # JSON-lines file the control plane drains into judge_audit_events; the same path
    # backs the cached deny-of-wallet spend tally. enabled=False = no judge audit/budget
    # on the runtime path (judges still run). empty path → ~/.znyx/judge-audit.spool.
    judge_audit_enabled: bool = True
    judge_audit_spool_path: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # CORS
    allowed_origins: str = ""

    # Logging
    log_redacted_text: bool = False

    # Bundle disk cache location. Override with ZNYX_CACHE_DIR; avoid /tmp in production.
    cache_dir: str = "/app/.cache/guardrails"

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Load configuration from environment variables."""
        mode = _env("ZNYX_MODE", "GUARDRAILS_MODE", "local")
        return cls(
            mode=mode,
            policy_path=_env(
                "ZNYX_POLICY_PATH",
                "GUARDRAILS_POLICY_PATH",
                os.getenv("POLICY_PATH", "./config/policies.yaml"),
            ),
            control_plane_url=_env("ZNYX_CONTROL_PLANE_URL", "GUARDRAILS_CONTROL_PLANE_URL", ""),
            runtime_token=_env("ZNYX_RUNTIME_TOKEN", "GUARDRAILS_RUNTIME_TOKEN", ""),
            bundle_public_key=_env("ZNYX_BUNDLE_PUBLIC_KEY", "GUARDRAILS_BUNDLE_PUBLIC_KEY", ""),
            fail_mode=_env("ZNYX_FAIL_MODE", "GUARDRAILS_FAIL_MODE", "closed"),
            bundle_poll_interval=int(
                _env("ZNYX_BUNDLE_POLL_INTERVAL", "GUARDRAILS_BUNDLE_POLL_INTERVAL", "30")
            ),
            require_signed_bundles=_env(
                "ZNYX_REQUIRE_SIGNED_BUNDLES",
                "GUARDRAILS_REQUIRE_SIGNED_BUNDLES",
                # Default: require signatures only when a public key is configured
                # to verify them against. In managed mode the bundle is fetched over
                # an authenticated channel (X-API-Key), so without a public key we
                # trust that channel rather than failing closed on every fetch.
                # Provide ZNYX_BUNDLE_PUBLIC_KEY to enforce signature verification.
                "true" if (os.getenv("ZNYX_BUNDLE_PUBLIC_KEY") or os.getenv("GUARDRAILS_BUNDLE_PUBLIC_KEY")) else "false",
            ).lower() == "true",
            telemetry_enabled=_env(
                "ZNYX_TELEMETRY_ENABLED",
                "GUARDRAILS_TELEMETRY_ENABLED",
                "true" if mode == "managed" else "false",
            ).lower() == "true",
            # Opt-in must be affirmative: only explicit truthy values enable the
            # heartbeat. Empty strings, typos, or "off" all stay disabled.
            heartbeat_enabled=_env("ZNYX_TELEMETRY", "GUARDRAILS_TELEMETRY", "false").lower() in ("true", "1", "yes", "on"),
            audit_sink_mode=_env("ZNYX_AUDIT_SINK_MODE", "GUARDRAILS_AUDIT_SINK_MODE", "spool"),
            audit_fail_mode=_env("ZNYX_AUDIT_FAIL_MODE", "GUARDRAILS_AUDIT_FAIL_MODE", "closed"),
            audit_spool_path=_env("ZNYX_AUDIT_SPOOL_PATH", "GUARDRAILS_AUDIT_SPOOL_PATH", ""),
            judge_audit_enabled=_env("ZNYX_JUDGE_AUDIT", "GUARDRAILS_JUDGE_AUDIT", "true").lower()
                not in ("false", "0", "no", "off"),
            judge_audit_spool_path=_env("ZNYX_JUDGE_AUDIT_SPOOL_PATH", "GUARDRAILS_JUDGE_AUDIT_SPOOL_PATH", ""),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8080")),
            allowed_origins=os.getenv("ALLOWED_ORIGINS", ""),
            log_redacted_text=os.getenv("LOG_REDACTED_TEXT", "false").lower() == "true",
            cache_dir=_env("ZNYX_CACHE_DIR", "GUARDRAILS_CACHE_DIR", "/app/.cache/guardrails"),
        )

    def __post_init__(self) -> None:
        """Emit security warnings (and, in production, hard errors) for dangerous
        configuration combinations."""
        from znyx_core.utils.env import is_production

        if self.bundle_public_key and not self.require_signed_bundles:
            logger.warning(
                "ZNYX_BUNDLE_PUBLIC_KEY is set but ZNYX_REQUIRE_SIGNED_BUNDLES is False. "
                "Bundle signatures will not be enforced — set ZNYX_REQUIRE_SIGNED_BUNDLES=true "
                "to enable verification."
            )

        # Managed mode: the runtime token + policy bundle travel over
        # control_plane_url. A plaintext http:// endpoint exposes the X-API-Key
        # and lets a MITM inject a policy that disables every detector. Require
        # https in production (http://localhost stays allowed for local dev).
        if self.mode == "managed" and self.control_plane_url:
            from urllib.parse import urlparse

            cp = urlparse(self.control_plane_url)
            is_local = (cp.hostname or "") in ("localhost", "127.0.0.1", "::1")
            if cp.scheme != "https" and not is_local:
                msg = (
                    f"ZNYX_CONTROL_PLANE_URL must use https:// (got '{self.control_plane_url}'). "
                    "The runtime token and policy bundle are exposed over plaintext otherwise."
                )
                if is_production():
                    raise RuntimeError(msg)
                logger.warning("%s Allowed only because this is not a production environment.", msg)

            # In production without a verifying public key, bundle integrity
            # rests entirely on transport trust — flag it loudly.
            if is_production() and not self.require_signed_bundles:
                logger.warning(
                    "Managed mode in production without signed bundles: bundle integrity "
                    "relies on TLS transport alone. Set ZNYX_BUNDLE_PUBLIC_KEY and "
                    "ZNYX_REQUIRE_SIGNED_BUNDLES=true for end-to-end policy signing."
                )

        if self.cache_dir.startswith("/tmp"):
            logger.warning(
                "Bundle cache is stored in /tmp (%s), which is world-readable. "
                "Set ZNYX_CACHE_DIR to a path with restricted permissions in production.",
                self.cache_dir,
            )

"""
Bundle Manager: fetches, caches, and serves policy bundles for the runtime.

In local mode, loads from a YAML or JSON file on disk.
In managed mode, polls the control plane for the latest bundle.
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from znyx_core.policy.bundle import (
    PolicyBundle, load_bundle_from_file, save_bundle_to_file,
    validate_bundle,
)
from znyx_core.policy.loader import PolicyLoader
from znyx_core.policy.resolver import PolicyResolver
from znyx_runtime.config import RuntimeConfig

logger = logging.getLogger(__name__)



class BundleManager:
    """Manages the active policy bundle for the runtime."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._bundle: Optional[PolicyBundle] = None
        self._policy_resolver: Optional[PolicyResolver] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._etag: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """True if a policy bundle is loaded and available."""
        return self._bundle is not None or self._policy_resolver is not None

    @property
    def effective_env(self) -> str | None:
        """The environment this runtime is serving (from the loaded bundle), or None in YAML mode."""
        if self._bundle:
            return self._bundle.environment or None
        return None

    @property
    def bundle_info(self) -> Dict[str, Any]:
        """Return metadata about the current bundle."""
        if self._bundle:
            return {
                "bundle_id": self._bundle.bundle_id,
                "policy_hash": self._bundle.policy_hash,
                "published_at": self._bundle.published_at,
                "org_id": self._bundle.org_id,
                "project_id": self._bundle.project_id,
                "environment": self._bundle.environment,
            }
        return {"mode": "yaml", "policy_path": self.config.policy_path}

    async def start(self) -> None:
        """Initialize the bundle manager and load the initial policy."""
        if self.config.mode == "local":
            await self._load_local()
        elif self.config.mode == "managed":
            await self._load_managed()
            # Start background polling
            self._poll_task = asyncio.create_task(self._poll_loop())
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

    async def stop(self) -> None:
        """Stop background polling."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    def get_policy(self, tenant_id: str = "default", app_id: str = "default",
                   agent_id: str = "default", env: str = "prod") -> Dict[str, Any]:
        """
        Get the resolved policy for the given scope.

        Returns the cached policy dict. In YAML mode, uses the PolicyResolver
        for hierarchical resolution. In managed/bundle mode, returns the
        bundle's pre-resolved policies.
        """
        # YAML mode: use resolver for hierarchical policy
        if self._policy_resolver:
            return self._policy_resolver.resolve(
                tenant_id=tenant_id,
                app_id=app_id,
                agent_id=agent_id,
                env=env,
            )

        # Bundle mode: use scope-aware lookup.
        # Use the bundle's own org_id/project_id/environment for the scope key
        # lookup — scope_policies keys were built from UUIDs at publish time,
        # while the request tenant_id/app_id/env may use slug strings or a
        # different env name (e.g. SDK default "prod" vs published "dev").
        if self._bundle and (self._bundle.policies or self._bundle.scope_policies):
            effective_tenant = self._bundle.org_id or tenant_id
            effective_app    = self._bundle.project_id or app_id
            effective_env    = self._bundle.environment or env
            policy = self._bundle.resolve_scope(
                tenant_id=effective_tenant,
                app_id=effective_app,
                agent_id=agent_id,
                env=effective_env,
            )
            return {**policy, "policy_version": self._bundle.bundle_id or self._bundle.policy_hash}

        # No policy available - apply fail mode
        if self.config.fail_mode == "open":
            logger.warning("No policy loaded - fail-open: allowing all")
            return self._empty_policy()
        else:
            raise RuntimeError("No policy bundle loaded and fail-mode is closed")

    async def _load_local(self) -> None:
        """Load policy from a local file (YAML or JSON bundle)."""
        path = self.config.policy_path
        if not Path(path).exists():
            if self.config.fail_mode == "open":
                logger.warning(f"Policy file not found: {path} - fail-open")
                return
            raise FileNotFoundError(f"Policy file not found: {path}")

        if path.endswith(".json"):
            self._bundle = load_bundle_from_file(path)
            if not validate_bundle(
                self._bundle,
                public_key_pem=self.config.bundle_public_key or None,
                require_signature=self.config.require_signed_bundles,
            ):
                raise ValueError(f"Bundle validation failed: {path}")
            logger.info(f"Loaded bundle from {path} (id={self._bundle.bundle_id})")
        else:
            # YAML mode
            loader = PolicyLoader(path)
            self._policy_resolver = PolicyResolver(loader)
            logger.info(f"Loaded YAML policy from {path}")

    # Exponential-backoff schedule for first-boot bundle fetch. Steady-state
    # polling keeps a single-shot model via `_fetch_bundle`. Override via
    # ZNYX_BUNDLE_BOOT_RETRY_DELAYS (comma-separated seconds).
    from znyx_core.config.tunables import BUNDLE_BOOT_RETRY_DELAYS_SECONDS as _BOOT_RETRY_DELAYS

    async def _load_managed(self) -> None:
        """Fetch the latest bundle from the control plane on first boot, with
        exponential-backoff retries. Falls back to a disk-cached bundle when
        the control plane is unreachable across every attempt."""
        last_err: Optional[Exception] = None
        for attempt, delay in enumerate(self._BOOT_RETRY_DELAYS, start=1):
            try:
                await self._fetch_bundle()
                return
            except Exception as e:  # noqa: BLE001 - any error should fall through to retry/cache
                last_err = e
                logger.warning(
                    "Bundle fetch attempt %d/%d failed: %s",
                    attempt, len(self._BOOT_RETRY_DELAYS), e,
                )
                if attempt < len(self._BOOT_RETRY_DELAYS):
                    await asyncio.sleep(delay)

        logger.warning(
            "Failed to fetch bundle from control plane after %d attempts: %s",
            len(self._BOOT_RETRY_DELAYS), last_err,
        )

        # Fall back to cached bundle on disk. Validate it with the SAME checks
        # as a freshly-fetched bundle — anyone who can write the cache file must
        # not be able to poison the policy (e.g. disable every detector). An
        # invalid/tampered cache is treated as "no bundle" and falls through to
        # the fail-mode handling below rather than being trusted.
        cache_path = self._cache_path()
        if Path(cache_path).exists():
            cached = load_bundle_from_file(cache_path)
            if validate_bundle(
                cached,
                public_key_pem=self.config.bundle_public_key or None,
                require_signature=self.config.require_signed_bundles,
            ):
                self._bundle = cached
                logger.info(f"Loaded cached bundle from {cache_path}")
                return
            logger.error("Cached bundle failed validation (ignoring): %s", cache_path)

        # No bundle available.
        if self.config.fail_mode == "open":
            logger.warning("No bundle available - fail-open")
        else:
            raise RuntimeError("Cannot fetch bundle and no cache available (fail-closed)")

    async def _fetch_bundle(self) -> bool:
        """Fetch latest bundle from control plane. Returns True if updated.

        Single-attempt, intentional - the steady-state poll loop calls this on
        a timer, so one failure just delays the next poll by the poll interval.
        First-boot retries are handled by `_load_managed` instead."""
        import httpx

        url = f"{self.config.control_plane_url.rstrip('/')}/v1/bundles/latest"
        headers = {"X-API-Key": self.config.runtime_token}
        if self._etag:
            headers["If-None-Match"] = self._etag

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)

        if resp.status_code == 304:
            logger.debug("Bundle unchanged (304)")
            return False

        resp.raise_for_status()
        data = resp.json()
        bundle = PolicyBundle.from_dict(data)

        # Validate
        if not validate_bundle(
            bundle,
            public_key_pem=self.config.bundle_public_key or None,
            require_signature=self.config.require_signed_bundles,
        ):
            raise ValueError("Fetched bundle failed validation")

        self._bundle = bundle
        self._etag = resp.headers.get("ETag")

        # Cache to disk
        save_bundle_to_file(bundle, self._cache_path())
        logger.info(f"Fetched new bundle: {bundle.bundle_id}")
        return True

    async def _poll_loop(self) -> None:
        """Background task that periodically polls for bundle updates."""
        while True:
            await asyncio.sleep(self.config.bundle_poll_interval)
            try:
                await self._fetch_bundle()
            except Exception as e:
                logger.warning(f"Bundle poll failed: {e}")

    def _cache_path(self) -> str:
        # 0700: the cached bundle is trusted policy; don't let other local users
        # read the runtime token path or write a poisoned bundle into it.
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
        return str(Path(self.config.cache_dir) / "bundle_cache.json")

    @staticmethod
    def _empty_policy() -> Dict[str, Any]:
        """Return a policy with all detectors disabled (fail-open)."""
        return {
            "secrets": {"enabled": False},
            "exfiltration": {"enabled": False},
            "abuse": {"enabled": False},
            "pii": {"enabled": False},
            "jailbreak": {"enabled": False},
            "toxicity": {"enabled": False},
            "topic_restriction": {"enabled": False},
            "competitor": {"enabled": False},
            "tools": {"enabled": False},
            "structure": {"enabled": False},
        }

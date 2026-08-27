"""
Bundle Manager: fetches, caches, and serves policy bundles for the runtime.

In local mode, loads from a YAML or JSON file on disk.
In managed mode, polls the control plane for the latest bundle.
"""
import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

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
        # Called after EVERY poll, changed or not (see _poll_loop). Used to push model
        # pins to the inference sidecar and to report state upstream.
        self._on_cycle: List[Callable[[Optional[Dict[str, Any]]], Awaitable[None]]] = []
        # The in-flight listener run, so a slow cycle skips the next tick instead of stacking.
        self._cycle_task: Optional[asyncio.Task] = None

    def add_cycle_listener(self, callback) -> None:
        """Register an async callback invoked once per poll cycle with the active policy.

        Deliberately fired on unchanged cycles too. That is what makes downstream
        reconciliation self-healing: a sidecar whose model download failed, or which
        restarted and lost its variants, is repopulated on the next tick without needing a
        republish. Gating this on "bundle changed" is exactly the bug that left a failed
        model install permanently unserved.
        """
        self._on_cycle.append(callback)

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
        # Fire once on boot so downstream state converges immediately rather than after a
        # full poll interval. Matters most in local/YAML mode, which has no poll loop at
        # all and would otherwise never push pins to the sidecar.
        #
        # Spawned, not awaited: the sidecar may not be listening yet (containers start
        # together) and a first-time model download takes minutes. Blocking here would hold
        # the runtime out of service behind ML provisioning, which is exactly backwards —
        # the deterministic rules path is ready and should start serving immediately.
        self._spawn_cycle()

    async def stop(self) -> None:
        """Stop background polling and any in-flight listener run."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        # The cycle listeners run detached (see _spawn_cycle), so without this an in-flight
        # push would be left dangling at shutdown and asyncio would warn about a pending task.
        if self._cycle_task is not None and not self._cycle_task.done():
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutting down anyway
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
                require_signature_v2=self.config.require_bundle_sig_v2,
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
                require_signature_v2=self.config.require_bundle_sig_v2,
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
            require_signature_v2=self.config.require_bundle_sig_v2,
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
            # Fire listeners even when the fetch failed or returned 304. They converge
            # downstream state (sidecar models, reported status) and must keep running
            # against the policy we already hold, otherwise a transient failure anywhere
            # downstream would never be retried.
            self._spawn_cycle()

    def _spawn_cycle(self) -> None:
        """Run the cycle listeners OFF the poll loop.

        Deliberately not awaited by the caller. Listeners do network I/O to the inference
        sidecar and the control plane, and the sidecar reconcile can legitimately block for
        minutes behind a large model download. Awaiting it here would make a slow or
        unreachable sidecar delay the next POLICY fetch, so a model download could hold up a
        security policy update. Policy freshness must never depend on ML provisioning.

        Guarded against pile-up: if the previous cycle is still in flight the tick is skipped
        rather than queued, so a sidecar stuck for ten minutes leaves one pending task, not
        twenty. The skipped work is not lost, because the next tick re-pushes the full desired
        state anyway.
        """
        if self._cycle_task is not None and not self._cycle_task.done():
            logger.debug("Bundle cycle listeners still running; skipping this tick")
            return
        self._cycle_task = asyncio.create_task(self._notify_cycle())

    async def _notify_cycle(self) -> None:
        policy = self._active_policy_for_listeners()
        for callback in self._on_cycle:
            try:
                await callback(policy)
            except Exception as e:  # noqa: BLE001 - a listener must never kill the poll loop
                logger.debug("Bundle cycle listener failed: %s", e)

    def _active_policy_for_listeners(self) -> Optional[Dict[str, Any]]:
        """The policy dict listeners should act on, or None in YAML-resolver mode.

        Bundle mode hands over the bundle's own policies, which is where the control plane
        injects runtime_policy. YAML mode resolves the default scope so a self-hosted
        operator who declared runtime_policy.inference in policies.yaml gets the same
        behaviour with no control plane involved.
        """
        if self._bundle is not None:
            return self._bundle.policies or {}
        if self._policy_resolver is not None:
            try:
                return self.get_policy()
            except Exception:  # noqa: BLE001 - resolution is best-effort here
                return None
        return None

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

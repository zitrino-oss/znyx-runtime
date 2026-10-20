"""
Bundle Manager: fetches, caches, and serves policy bundles for the runtime.

In local mode, loads from a YAML or JSON file on disk.
In managed mode, polls the control plane for the latest bundle set: one bundle per
environment the runtime serves (one, for a runtime whose key serves one environment;
several, for one gateway serving several projects). Each request is answered with the
policy of the environment it belongs to: a gateway or app key's own, or the one the
request names. A control plane that predates sets is read as a set of one.
"""
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from znyx_core.policy.bundle import (
    PolicyBundle, load_bundle_from_file, save_bundle_to_file,
    validate_bundle,
)
from znyx_core.policy.bundle_set import (
    BUNDLE_SET_FORMAT, BundleSet, BundleSetMember, load_bundle_set_from_file,
    save_bundle_set_to_file, validate_bundle_set,
)
from znyx_core.policy.loader import PolicyLoader
from znyx_core.policy.resolver import PolicyResolver
from znyx_runtime.config import RuntimeConfig

logger = logging.getLogger(__name__)


class ScopeNotServed(LookupError):
    """The call belongs to a project or environment this runtime does not serve, or,
    on a runtime serving several, names none of them."""


class BundleManager:
    """Manages the active policy bundles for the runtime."""

    # An httpx transport for the control plane calls (tests); None: the network.
    _transport = None
    # After the control plane answered with single bundles only, the bundle-set route
    # is tried again this often (an upgraded control plane is picked up).
    LEGACY_RECHECK_SECONDS = 600.0

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._set: Optional[BundleSet] = None
        # Environments the control plane lists as missing (it could not serve them, for
        # instance because their secrets need references) whose last good bundle this
        # runtime still holds: they keep it, as a runtime keeps its bundle when the
        # control plane answers with an error.
        self._carried: List[BundleSetMember] = []
        self._legacy_until = 0.0        # until then, read single bundles only
        self._on_change: List[Callable[[], None]] = []
        self._policy_resolver: Optional[PolicyResolver] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._etag: Optional[str] = None
        # Called after EVERY poll, changed or not (see _poll_loop). Used to push model
        # pins to the inference sidecar and to report state upstream.
        self._on_cycle: List[Callable[[Optional[Dict[str, Any]]], Awaitable[None]]] = []
        # The in-flight listener run, so a slow cycle skips the next tick instead of stacking.
        self._cycle_task: Optional[asyncio.Task] = None

    def add_change_listener(self, callback: Callable[[], None]) -> None:
        """Register a callback run at once whenever the bundles held change (the gateway
        configuration, say: a key disabled in the console must stop working on the
        next fetch, not after the slower cycle listeners)."""
        self._on_change.append(callback)

    def _changed(self) -> None:
        for callback in self._on_change:
            try:
                callback()
            except Exception as e:  # noqa: BLE001 - a listener must never break a fetch
                logger.error("Bundle change listener failed: %s", e)

    def add_cycle_listener(self, callback) -> None:
        """Register an async callback invoked once per poll cycle with the active policy.

        Deliberately fired on unchanged cycles too. That is what makes downstream
        reconciliation self-healing: a sidecar whose model download failed, or which
        restarted and lost its variants, is repopulated on the next tick without needing a
        republish. Gating this on "bundle changed" is exactly the bug that left a failed
        model install permanently unserved.
        """
        self._on_cycle.append(callback)

    # -- what is held ------------------------------------------------------------

    @property
    def _bundle(self) -> Optional[PolicyBundle]:
        """The bundle, when the runtime serves exactly one environment (a local JSON
        bundle, or a key for one environment)."""
        members = self.members
        if len(members) == 1 and not self.serves_several:
            return members[0].bundle
        return None

    @_bundle.setter
    def _bundle(self, bundle: Optional[PolicyBundle]) -> None:
        self._set = BundleSet.single(bundle) if bundle is not None else None
        self._carried = []

    @property
    def members(self) -> List[BundleSetMember]:
        if self._set is None:
            return []
        return list(self._set.members) + list(self._carried)

    @property
    def serves_several(self) -> bool:
        """The key serves several environments (said by the signed set), or several are
        held. Then a call that names none of them is never answered with one of them,
        even when only one is left."""
        if self._set is None:
            return False
        environments = {m.key for m in self.members}
        environments |= {e.get("environment_id") for e in self._set.missing if e.get("environment_id")}
        return self._set.serves_several or len(environments) > 1

    @property
    def is_ready(self) -> bool:
        """True if a policy is loaded and available."""
        return bool(self.members) or self._policy_resolver is not None

    @property
    def effective_env(self) -> str | None:
        """The environment this runtime serves, when it serves exactly one; None in YAML
        mode and on a runtime serving several (each call then has its own)."""
        if self._bundle:
            return self._bundle.environment or None
        return None

    @staticmethod
    def _member_info(member: BundleSetMember) -> Dict[str, Any]:
        bundle = member.bundle
        info = {"bundle_id": bundle.bundle_id, "policy_hash": bundle.policy_hash,
                "published_at": bundle.published_at, "org_id": bundle.org_id,
                "project_id": bundle.project_id, "environment": bundle.environment}
        if member.environment_id:
            info["environment_id"] = member.environment_id
        if member.project_slug:
            info["project_slug"] = member.project_slug
        return info

    @property
    def bundle_info(self) -> Dict[str, Any]:
        """Metadata about the bundles held (never policy content)."""
        if self._bundle is not None:
            return self._member_info(self.members[0])
        if self._set is not None:
            carried = {m.key for m in self._carried}
            return {"bundle_set": True, "set_id": self._set.set_id, "org_id": self._set.org_id,
                    "members": [self._member_info(m) for m in self.members],
                    "missing": [e for e in self._set.missing
                                if (e.get("environment_id") or "") not in carried],
                    "kept_from_earlier": sorted(carried)}
        return {"mode": "yaml", "policy_path": self.config.policy_path}

    def report_infos(self) -> List[Dict[str, Any]]:
        """One entry per environment served, for the per-environment runtime report."""
        if self._set is not None:
            return [self._member_info(m) for m in self.members]
        return [self.bundle_info]

    # -- which environment a call belongs to --------------------------------------

    def member_by_key(self, member_key: Optional[str]) -> Optional[BundleSetMember]:
        """The member a gateway or app key belongs to. No member key (the instance's own
        key): the only member, when there is one. None in YAML mode, or when nothing is
        held (the fail mode then applies)."""
        members = self.members
        if self._policy_resolver is not None or self._set is None or not members:
            return None
        if member_key:
            for member in members:
                if member.key == member_key:
                    return member
            raise ScopeNotServed("This key's project is not served by this runtime")
        if len(members) == 1 and not self.serves_several:
            return members[0]
        raise ScopeNotServed("This runtime serves several projects: call it with an app key, or name "
                             "the project (app_id: its id or slug) and the environment (env)")

    def member_for(self, member_key: Optional[str] = None, app_id: Optional[str] = None,
                   env: Optional[str] = None) -> Optional[BundleSetMember]:
        """The member a call belongs to: its key's, else the only one, else the one the
        request names (``app_id`` the project's id or slug, ``env`` the environment;
        a project served in one environment only needs no ``env``)."""
        if member_key or not self.serves_several:
            return self.member_by_key(member_key)
        if self._policy_resolver is not None:
            return None
        project = [m for m in self.members if app_id and app_id in (m.project_id, m.project_slug)]
        exact = [m for m in project if env and m.environment == env]
        if len(exact) == 1:
            return exact[0]
        if len(project) == 1:
            return project[0]
        raise ScopeNotServed("This runtime serves several projects: call it with an app key, or name "
                             "the project (app_id: its id or slug) and the environment (env)")

    def environment_for(self, member: Optional[BundleSetMember]) -> Optional[str]:
        return (member.environment if member is not None else None) or self.effective_env

    def resolve(self, request, key=None) -> Tuple[Any, Dict[str, Any]]:
        """``(request, policy)`` for an evaluate call: the request's environment set to
        the one it belongs to, and that environment's policy. Raises ScopeNotServed."""
        member = self.member_for(getattr(key, "member", "") or None, getattr(request, "app_id", None),
                                 getattr(request, "env", None))
        env = self.environment_for(member)
        if env:
            request = request.model_copy(update={"env": env})
        policy = self.get_policy(tenant_id=request.tenant_id, app_id=request.app_id,
                                 agent_id=request.agent_id, env=request.env, member=member)
        return request, policy

    def gateway_documents(self) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """Each member's gateway section, with the member it belongs to."""
        return [(m.key, (m.bundle.policies or {}).get("gateway")) for m in self.members]

    async def start(self) -> None:
        """Initialize the bundle manager and load the initial policy."""
        if self.config.mode == "local":
            await self._load_local()
            # No policy to poll in local mode, but the listeners still run on the poll
            # interval: they push the policy's model pins to the model server, which keeps
            # them only in memory, so a model server restarted by the supervisor (or not
            # yet up at boot) is repopulated within one interval instead of never.
            self._poll_task = asyncio.create_task(self._local_cycle_loop())
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
                   agent_id: str = "default", env: str = "prod",
                   member: Optional[BundleSetMember] = None) -> Dict[str, Any]:
        """
        Get the resolved policy for the given scope.

        Returns the cached policy dict. In YAML mode, uses the PolicyResolver
        for hierarchical resolution. In managed/bundle mode, returns the
        pre-resolved policies of ``member`` (by default the member the scope names,
        see ``member_for``).
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
        # lookup - scope_policies keys were built from UUIDs at publish time,
        # while the request tenant_id/app_id/env may use slug strings or a
        # different env name (e.g. SDK default "prod" vs published "dev").
        if member is None:
            member = self.member_for(None, app_id, env)
        bundle = member.bundle if member is not None else None
        if bundle and (bundle.policies or bundle.scope_policies):
            policy = bundle.resolve_scope(
                tenant_id=bundle.org_id or tenant_id,
                app_id=bundle.project_id or app_id,
                agent_id=agent_id,
                env=bundle.environment or env,
            )
            return {**policy, "policy_version": bundle.bundle_id or bundle.policy_hash}

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
        the control plane is unreachable across every attempt.

        The validated cache is held first, so a fetch that lists an environment as
        missing keeps that environment's last good bundle across a restart too."""
        try:
            self._load_cache()
        except OSError as exc:      # an unusable cache directory must not stop a boot
            logger.warning("Bundle cache unavailable: %s", exc)
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
        if self._set is not None:
            logger.info("Serving the cached bundles (%d environment(s))", len(self.members))
            return

        # No bundle available.
        if self.config.fail_mode == "open":
            logger.warning("No bundle available - fail-open")
        else:
            raise RuntimeError("Cannot fetch bundle and no cache available (fail-closed)")

    def _load_cache(self) -> None:
        """Hold the cached bundles, when they validate. Validated with the SAME checks
        as freshly-fetched ones - anyone who can write the cache file must
        not be able to poison the policy (e.g. disable every detector). An
        invalid/tampered cache is treated as "no bundle" rather than being trusted."""
        set_path = self._set_cache_path()
        if Path(set_path).exists():
            try:
                cached_set = load_bundle_set_from_file(set_path)
                ok, reason = self._validate(cached_set)
            except (OSError, ValueError, TypeError) as exc:
                ok, reason = False, str(exc)
            if ok:
                self._set = cached_set
                self._carried = self._cached_carried(set_path, cached_set)
                logger.info("Loaded cached bundle set from %s (%d environment(s))", set_path,
                            len(self.members))
                return
            logger.error("Cached bundle set failed validation (ignoring): %s (%s)", set_path, reason)
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

    def _carry_valid(self, bundle_set: BundleSet, member: BundleSetMember) -> bool:
        """Whether an earlier member may be kept for an environment the signed set lists
        as missing: the set names exactly that environment as missing, and the member's
        own bundle validates and was published for it."""
        named = any((e.get("environment_id") or "") == member.environment_id
                    and (e.get("project_id") or "") == member.project_id
                    and (e.get("environment") or "") == member.environment
                    for e in bundle_set.missing)
        if not named or not member.environment_id:
            return False
        bundle = member.bundle
        if (bundle.project_id or "") != member.project_id or (bundle.environment or "") != member.environment:
            return False
        if bundle.org_id and bundle_set.org_id and bundle.org_id != bundle_set.org_id:
            return False
        return validate_bundle(bundle, public_key_pem=self.config.bundle_public_key or None,
                               require_signature=self.config.require_signed_bundles,
                               require_signature_v2=self.config.require_bundle_sig_v2)

    def _cached_carried(self, path: str, bundle_set: BundleSet) -> List[BundleSetMember]:
        try:
            with open(path) as f:
                raw = json.load(f).get("carried") or []
            carried = [BundleSetMember.from_dict(m) for m in raw]
        except (OSError, ValueError, TypeError, AttributeError):
            return []
        return [m for m in carried if self._carry_valid(bundle_set, m)]

    def _validate(self, bundle_set: BundleSet) -> Tuple[bool, str]:
        return validate_bundle_set(
            bundle_set,
            public_key_pem=self.config.bundle_public_key or None,
            require_signature=self.config.require_signed_bundles,
            require_signature_v2=self.config.require_bundle_sig_v2,
        )

    async def _fetch_bundle(self) -> bool:
        """Fetch the latest bundle set from the control plane. Returns True if updated.

        Single-attempt, intentional - the steady-state poll loop calls this on
        a timer, so one failure just delays the next poll by the poll interval.
        First-boot retries are handled by `_load_managed` instead."""
        import httpx

        base = self.config.control_plane_url.rstrip('/')
        headers = {"X-API-Key": self.config.runtime_token}
        if self._etag:
            headers["If-None-Match"] = self._etag
        options = {"transport": self._transport} if self._transport is not None else {}

        async with httpx.AsyncClient(timeout=10.0, **options) as client:
            resp = None
            probing = time.monotonic() >= self._legacy_until
            if probing:
                resp = await client.get(f"{base}/v1/bundle-sets/latest", headers=headers)
                if resp.status_code == 404:
                    headers.pop("If-None-Match", None)
                    resp = None
            if resp is None:
                resp = await client.get(f"{base}/v1/bundles/latest", headers=headers)
                if probing and resp.status_code in (200, 304):
                    # Only an answer from the single-bundle route makes this an older
                    # control plane; a 404 alone may be transient.
                    logger.info("The control plane serves single bundles only; reading "
                                "/v1/bundles/latest (bundle sets are tried again later)")
                    self._legacy_until = time.monotonic() + self.LEGACY_RECHECK_SECONDS

        if resp.status_code == 304:
            logger.debug("Bundle unchanged (304)")
            return False

        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("format") == BUNDLE_SET_FORMAT:
            bundle_set = BundleSet.from_dict(data)
        else:
            bundle_set = BundleSet.single(PolicyBundle.from_dict(data))

        ok, reason = self._validate(bundle_set)
        if not ok:
            raise ValueError(f"Fetched bundle failed validation: {reason}")

        carried = [] if bundle_set.legacy else [m for m in self.members if self._carry_valid(bundle_set, m)]
        for member in carried:
            reason = next((e.get("reason") for e in bundle_set.missing
                           if e.get("environment_id") == member.environment_id), "unknown")
            logger.error("The control plane could not serve environment %s of project %s (%s); "
                         "keeping its last good bundle", member.environment,
                         member.project_slug or member.project_id, reason)
        self._set = bundle_set
        self._carried = carried
        self._etag = resp.headers.get("ETag")
        self._save_cache(bundle_set)
        self._changed()
        if bundle_set.legacy:
            logger.info(f"Fetched new bundle: {bundle_set.members[0].bundle.bundle_id}")
        else:
            logger.info("Fetched new bundle set: %d environment(s)%s", len(bundle_set.members),
                        f", {len(bundle_set.missing)} without a published bundle" if bundle_set.missing else "")
        return True

    def _save_cache(self, bundle_set: BundleSet) -> None:
        """One cache, of the shape last fetched (the other is removed, so a boot without
        the control plane never falls back to an older shape)."""
        try:
            if bundle_set.legacy:
                save_bundle_to_file(bundle_set.members[0].bundle, self._cache_path())
                stale = self._set_cache_path()
            else:
                save_bundle_set_to_file(bundle_set, self._set_cache_path(),
                                        extra={"carried": [m.to_dict() for m in self._carried]})
                stale = self._cache_path()
            if os.path.exists(stale):
                os.remove(stale)
        except OSError as exc:
            logger.warning("Could not cache the bundles on disk: %s", exc)

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

    async def _local_cycle_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.bundle_poll_interval)
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
        if self._set is not None:
            return self._merged_listener_policy()
        if self._policy_resolver is not None:
            try:
                return self.get_policy()
            except Exception:  # noqa: BLE001 - resolution is best-effort here
                return None
        return None

    def _merged_listener_policy(self) -> Dict[str, Any]:
        """On a runtime serving several environments: the inference settings of all of
        them, for the one model server the runtime has. Pins are merged; when two
        environments pin a task to different models, the first (in set order) wins and
        the conflict is logged."""
        inference: Dict[str, Any] = {}
        pins: Dict[str, Any] = {}
        stated = False
        for member in self.members:
            rp = (member.bundle.policies or {}).get("runtime_policy")
            mine = rp.get("inference") if isinstance(rp, dict) else None
            if not isinstance(mine, dict):
                continue
            for name, value in mine.items():
                if name != "pins":
                    inference.setdefault(name, value)
            if isinstance(mine.get("pins"), dict):
                stated = True
                for task, pin in mine["pins"].items():
                    if task in pins and pins[task] != pin:
                        logger.warning("Environments served by this runtime pin %s to different models; "
                                       "using the first", task)
                        continue
                    pins[task] = pin
        if stated:
            inference["pins"] = pins
        return {"runtime_policy": {"inference": inference}} if inference else {}

    def _set_cache_path(self) -> str:
        return str(Path(self._cache_path()).with_name("bundle_set_cache.json"))

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

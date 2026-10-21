"""Gateway configuration: providers, per-app and per-agent keys, limits, prices.

Local mode reads ``gateway.yaml`` (``ZNYX_GATEWAY_CONFIG``, default ``./gateway.yaml``);
managed mode will receive the same document inside the signed bundle. With neither,
the gateway still works for one-command use: a provider is configured for each of
``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY`` that is set, and the instance's own API
key acts as the single, unrestricted gateway key.

    providers:
      openai:
        format: openai                 # openai | anthropic | azure_openai
        base_url: https://api.openai.com/v1
        api_key: env:OPENAI_API_KEY    # a reference (env:, file:), or `client`: forward the
                                       # caller's own provider key
        region: us                     # for data residency (runtime_policy.allowed_regions)
    keys:
      - id: support-bot
        key_sha256: 9f2c...            # `znyx keys create` prints this; or key: env:NAME
        tenant_id: acme
        app_id: support-bot
        agent_id: triage               # optional: an agent key
        providers: [openai]            # empty: every provider
        allowed_models: [gpt-4o-mini, "claude-3-5-haiku*"]   # empty: any model
        limits: {requests_per_minute: 600, tokens_per_day: 2000000, spend_usd_per_day: 50}

Keys are stored as SHA-256 digests: the key itself never needs to be in a file.
"""
from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from znyx_core import secret_refs
from znyx_runtime.gateway.optimise import PROFILES, KeyOptimisation

logger = logging.getLogger(__name__)

FORMATS = ("openai", "anthropic", "azure_openai")
CLIENT_KEY = "client"           # api_key value: forward the caller's own provider key

DEFAULT_BASE_URLS = {"openai": "https://api.openai.com/v1",
                     "anthropic": "https://api.anthropic.com"}
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_CONCURRENT = 256
# Every role is checked: a client can send any history it likes (forged assistant turns
# are a known jailbreak), and apps often place user content in system messages.
DEFAULT_CHECK_ROLES = ("system", "developer", "user", "assistant", "tool", "function")
DEFAULT_BODY_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_PARALLEL_CHECKS = 8
UNCHECKABLE_MODES = ("allow", "block")

# Managed mode: where a provider from the bundle may send requests. The control plane
# chooses providers, but the runtime decides where its prompts and secrets may go, so a
# compromised control plane cannot point a provider at itself. Other hosts are added by
# the operator (ZNYX_GATEWAY_ALLOWED_HOSTS, comma-separated, "*.example.com" for
# subdomains); hosts inside the network are always allowed.
DEFAULT_MANAGED_HOSTS = ("api.openai.com", "api.anthropic.com", "*.openai.azure.com",
                         "*.cognitiveservices.azure.com", "*.services.ai.azure.com")
# The standard provider key variables only ever go to their own provider.
KEY_HOSTS = {
    "OPENAI_API_KEY": ("api.openai.com",),
    "ANTHROPIC_API_KEY": ("api.anthropic.com",),
    "AZURE_OPENAI_API_KEY": ("*.openai.azure.com", "*.cognitiveservices.azure.com",
                             "*.services.ai.azure.com"),
    "GOOGLE_API_KEY": ("*.googleapis.com",),
}


class GatewayConfigError(ValueError):
    """The gateway configuration cannot be used; the message says where and why."""


@dataclass(frozen=True)
class Provider:
    name: str
    format: str
    base_url: str
    api_key: Optional[str] = None          # a reference, CLIENT_KEY, or None (no key)
    region: Optional[str] = None
    in_boundary: Optional[bool] = None     # a self-hosted model server inside the network
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    api_version: Optional[str] = None      # azure_openai
    headers: Tuple[Tuple[str, str], ...] = ()
    blocked: Optional[str] = None          # managed mode: why this runtime refuses it
    member: str = ""                       # the environment (bundle set member) it belongs to

    @property
    def uses_client_key(self) -> bool:
        return self.api_key == CLIENT_KEY

    def credential(self) -> Optional[str]:
        """The provider key, resolved now (so a rotated secret is picked up). Never for
        a provider this runtime refuses."""
        if self.blocked:
            raise secret_refs.SecretRefError(self.blocked)
        if self.api_key in (None, CLIENT_KEY):
            return None
        return secret_refs.resolve(self.api_key, what=f"provider {self.name} api_key")


@dataclass(frozen=True)
class Limits:
    requests_per_minute: Optional[int] = None
    tokens_per_day: Optional[int] = None
    spend_usd_per_day: Optional[float] = None


@dataclass(frozen=True)
class GatewayKey:
    id: str
    key_sha256: str
    tenant_id: str = "default"
    app_id: str = "default"
    agent_id: str = "default"
    environment: Optional[str] = None
    providers: Tuple[str, ...] = ()         # allowed provider names; empty = all
    default_provider: Optional[str] = None
    allowed_models: Tuple[str, ...] = ()     # glob patterns; empty = any model
    limits: Limits = field(default_factory=Limits)
    profile: str = "balanced"                # optimisation profile (gateway.optimise.PROFILES)
    profile_chosen: bool = False             # set for this key (else stage A and named techniques only)
    optimisation: KeyOptimisation = field(default_factory=KeyOptimisation)
    disabled: bool = False
    instance_key: bool = False               # the instance's RUNTIME_API_KEY
    member: str = ""                         # the environment (bundle set member) it belongs to

    def may_use(self, provider: str) -> bool:
        return not self.providers or provider in self.providers

    def model_allowed(self, model: Optional[str]) -> bool:
        if not self.allowed_models:
            return True
        if not model:
            return False
        return any(fnmatch.fnmatchcase(model, pattern) for pattern in self.allowed_models)


@dataclass(frozen=True)
class GatewaySettings:
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    check_roles: Tuple[str, ...] = DEFAULT_CHECK_ROLES
    normalise_before_send: bool = True
    allow_instance_key: bool = True
    block_message: str = "This request was blocked by policy."
    output_block_message: str = "This response was withheld by policy."
    body_timeout_seconds: float = DEFAULT_BODY_TIMEOUT_SECONDS
    # Checks run at once for one call; the rest wait, so one long conversation cannot
    # take the whole check pool from other callers.
    max_parallel_checks: int = DEFAULT_MAX_PARALLEL_CHECKS
    # Content the gateway cannot read (images, audio, files): "allow" forwards it
    # (counted), "block" refuses the request.
    uncheckable_content: str = "allow"
    # Anthropic's MCP connector makes the provider call MCP servers itself, outside the
    # gateway's tool policy and egress gate: refused unless allowed here.
    allow_provider_mcp: bool = False
    # End-user ids in usage events (X-Znyx-User, OpenAI `user`, Anthropic
    # metadata.user_id): "hash" (a pseudonym, the default), "plain", or "off".
    user_ids: str = "hash"
    # Technique 27: add a marker unique to this deployment to every system prompt, so
    # a leaked prompt is caught in the answer and can be traced back here. Off by
    # default: it changes what the provider receives.
    canary_marker: bool = False


@dataclass
class GatewayConfig:
    providers: Dict[str, Provider]
    keys_by_digest: Dict[str, GatewayKey]
    settings: GatewaySettings = field(default_factory=GatewaySettings)
    prices: Dict[str, Dict[str, float]] = field(default_factory=dict)
    source: str = "defaults"
    # The environments (bundle set members) this configuration covers; empty outside
    # managed mode. Each provider and key belongs to one of them.
    members: List[str] = field(default_factory=list)
    # The runtime's key serves several environments: the instance key is then never
    # taken for one of them, even when only one is left.
    shared: bool = False
    # Each environment's own price table (its bundle's), by member.
    member_prices: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    def _providers_of(self, key: GatewayKey) -> Dict[str, Provider]:
        """The providers a key can reach: those of its own project and environment.
        The instance's own key belongs to no project, so it reaches the providers of the
        only environment, and none when the runtime serves several."""
        member = key.member
        if not member:
            if len(self.members) > 1 or self.shared:
                raise LookupError("this runtime serves several projects: call it with an app key "
                                  "from the console")
            member = self.members[0] if self.members else ""
        return {p.name: p for p in self.providers.values() if p.member == member}

    def provider_for(self, key: GatewayKey, fmt_family: str,
                     requested: Optional[str] = None) -> Provider:
        """The provider this call goes to: the one the caller named (X-Znyx-Provider),
        else the key's default, else the only allowed provider of the right family.
        Never a choice among providers: the gateway forwards, it does not route."""
        family = {"openai": ("openai",), "azure": ("azure_openai",),
                  "anthropic": ("anthropic",)}[fmt_family]
        providers = self._providers_of(key)
        if requested:
            provider = providers.get(requested)
            if provider is None or not key.may_use(requested):
                raise LookupError(f"provider {requested!r} is not available to this key")
            if provider.format not in family:
                raise LookupError(f"provider {requested!r} does not speak this API")
            return provider
        if key.default_provider:
            provider = providers.get(key.default_provider)
            if provider is not None and provider.format in family:
                return provider
        candidates = [p for name, p in providers.items()
                      if p.format in family and key.may_use(name)]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise LookupError("no provider for this API is configured for this key")
        raise LookupError("several providers serve this API for this key: name one with the "
                          "X-Znyx-Provider header or set the key's default_provider")


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _require(cond: bool, where: str, message: str) -> None:
    if not cond:
        raise GatewayConfigError(f"{where}: {message}")


def _str_list(value: Any, where: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    _require(isinstance(value, list) and all(isinstance(v, str) for v in value), where,
             "expected a list of names")
    return tuple(value)


def _host_matches(host: str, patterns) -> bool:
    host = host.rstrip(".").lower()
    for pattern in patterns:
        pattern = pattern.strip().lower()
        if not pattern:
            continue
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]):
                return True
        elif host == pattern:
            return True
    return False


def _managed_refusal(name: str, base_url: str, api_key: Optional[str]) -> Optional[str]:
    """Why this runtime refuses a provider that came from a bundle (None: allowed)."""
    from urllib.parse import urlparse
    from znyx_core.engine.egress import in_network

    host = (urlparse(base_url).hostname or "").lower()
    if not in_network(host):
        extra = [h for h in (os.getenv("ZNYX_GATEWAY_ALLOWED_HOSTS") or "").split(",") if h.strip()]
        if not _host_matches(host, DEFAULT_MANAGED_HOSTS + tuple(extra)):
            return (f"provider {name} sends to {host}, which this runtime does not allow "
                    "(add it to ZNYX_GATEWAY_ALLOWED_HOSTS)")
    if isinstance(api_key, str) and api_key.lower().startswith("env:"):
        variable = api_key[4:].strip()
        bound = KEY_HOSTS.get(variable)
        if bound and not _host_matches(host, bound):
            return f"provider {name} would send {variable} to {host}, which is not its provider"
    return None


def _provider(name: str, raw: Mapping[str, Any], managed: bool = False) -> Provider:
    where = f"providers.{name}"
    _require(isinstance(raw, Mapping), where, "expected a mapping")
    unknown = set(raw) - {"format", "base_url", "api_key", "region", "in_boundary",
                          "timeout_seconds", "api_version", "headers"}
    _require(not unknown, where, f"unknown setting(s) {', '.join(sorted(unknown))}")
    fmt = str(raw.get("format") or ("anthropic" if name == "anthropic" else "openai")).lower()
    _require(fmt in FORMATS, where, f"format must be one of {', '.join(FORMATS)}")
    base_url = raw.get("base_url") or DEFAULT_BASE_URLS.get(fmt)
    _require(isinstance(base_url, str) and base_url.startswith(("https://", "http://")), where,
             "base_url must be an http(s) URL")
    if base_url.startswith("http://"):
        from urllib.parse import urlparse
        from znyx_core.engine.egress import in_network
        _require(in_network(urlparse(base_url).hostname), where,
                 "plain http:// is only allowed for providers inside your network")
    api_key = raw.get("api_key")
    if api_key is not None and api_key != CLIENT_KEY:
        _require(isinstance(api_key, str) and secret_refs.is_reference(api_key), where,
                 "api_key must be a reference (env:NAME, file:/path), `client` or omitted; "
                 "never the key itself")
    _require(fmt != "azure_openai" or raw.get("api_version"), where,
             "azure_openai needs api_version")
    headers = raw.get("headers") or {}
    _require(isinstance(headers, Mapping), where, "headers must be a mapping")
    for header in headers:
        _require(str(header).lower() not in ("authorization", "x-api-key", "api-key"), where,
                 "credentials belong in api_key, not headers")
    timeout = raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    _require(isinstance(timeout, (int, float)) and timeout > 0, where, "timeout_seconds must be > 0")
    in_boundary = raw.get("in_boundary")
    blocked = None
    if managed:
        # "Inside the network" skips the egress gate, so from a bundle it is believed only
        # when the host really is inside the network.
        from urllib.parse import urlparse
        from znyx_core.engine.egress import in_network
        if in_boundary and not in_network(urlparse(base_url).hostname):
            logger.warning("Gateway: provider %s claims to be inside the network but %s is not; "
                           "treating it as external", name, urlparse(base_url).hostname)
            in_boundary = False
        blocked = _managed_refusal(name, base_url, api_key)
        if blocked:
            logger.error("Gateway: %s", blocked)
    return Provider(name=name, format=fmt, base_url=base_url.rstrip("/"), api_key=api_key,
                    region=raw.get("region"), in_boundary=in_boundary,
                    timeout_seconds=float(timeout), api_version=raw.get("api_version"),
                    headers=tuple((str(k), str(v)) for k, v in headers.items()), blocked=blocked)


def _limits(raw: Any, where: str) -> Limits:
    if raw is None:
        return Limits()
    _require(isinstance(raw, Mapping), where, "expected a mapping")
    unknown = set(raw) - {"requests_per_minute", "tokens_per_day", "spend_usd_per_day"}
    _require(not unknown, where, f"unknown limit(s) {', '.join(sorted(unknown))}")
    for name, value in raw.items():
        _require(value is None or (isinstance(value, (int, float)) and value > 0), where,
                 f"{name} must be a positive number")
    return Limits(**{k: raw.get(k) for k in ("requests_per_minute", "tokens_per_day",
                                             "spend_usd_per_day")})


def _key(raw: Mapping[str, Any], index: int, providers: Mapping[str, Provider]) -> GatewayKey:
    where = f"keys[{index}]"
    _require(isinstance(raw, Mapping), where, "expected a mapping")
    allowed = {"id", "key", "key_sha256", "tenant_id", "app_id", "agent_id", "environment",
               "providers", "default_provider", "allowed_models", "limits", "profile", "optimisation",
               "disabled"}
    unknown = set(raw) - allowed
    _require(not unknown, where, f"unknown setting(s) {', '.join(sorted(unknown))}")
    key_id = raw.get("id")
    _require(isinstance(key_id, str) and key_id, where, "needs an id")
    where = f"keys.{key_id}"
    digest = raw.get("key_sha256")
    if raw.get("key") is not None:
        _require(digest is None, where, "give key or key_sha256, not both")
        try:
            plain = secret_refs.resolve(raw["key"], what=f"{where}.key")
        except secret_refs.SecretRefError as exc:
            raise GatewayConfigError(str(exc)) from None
        _require(bool(plain), where, "key resolved to nothing")
        digest = key_digest(plain)
    _require(isinstance(digest, str) and len(digest) == 64
             and all(c in "0123456789abcdef" for c in digest.lower()), where,
             "key_sha256 must be the 64-character hex SHA-256 of the key")
    names = _str_list(raw.get("providers"), f"{where}.providers")
    for name in names:
        _require(name in providers, where, f"unknown provider {name!r}")
    default = raw.get("default_provider")
    _require(default is None or default in providers, where, f"unknown default_provider {default!r}")
    profile = raw.get("profile") or "balanced"
    chosen = isinstance(profile, str) and profile in PROFILES and bool(raw.get("profile"))
    if not isinstance(profile, str) or profile not in PROFILES:
        # Older consoles stored free text here: such a key keeps working, on the default.
        logger.warning("gateway: %s has unknown profile %r; using balanced (known: %s)", where, profile,
                       ", ".join(PROFILES))
        profile = "balanced"
    return GatewayKey(
        id=key_id, key_sha256=digest.lower(),
        tenant_id=str(raw.get("tenant_id") or "default"), app_id=str(raw.get("app_id") or key_id),
        agent_id=str(raw.get("agent_id") or "default"), environment=raw.get("environment"),
        providers=names, default_provider=default,
        allowed_models=_str_list(raw.get("allowed_models"), f"{where}.allowed_models"),
        limits=_limits(raw.get("limits"), f"{where}.limits"),
        profile=profile, profile_chosen=chosen,
        optimisation=_optimisation(raw.get("optimisation"), f"{where}.optimisation"),
        disabled=bool(raw.get("disabled", False)))


def _optimisation(raw: Any, where: str) -> KeyOptimisation:
    """A key's technique choices. Technique ids are not checked here: one this runtime
    does not know (from a newer console) is ignored, and so are settings it does not
    know yet, so a newer control plane never makes the whole gateway configuration
    unloadable."""
    if raw is None:
        return KeyOptimisation()
    _require(isinstance(raw, Mapping), where, "expected a mapping")
    lists = {}
    for name in ("enable", "disable", "approved"):
        value = raw.get(name)
        if value is None:
            value = []
        elif isinstance(value, str):
            value = [value]
        _require(isinstance(value, list), where, f"{name} must be a list of technique ids")
        kept = [v for v in value if isinstance(v, str)]
        if len(kept) != len(value):
            logger.warning("gateway: %s.%s has entries that are not technique ids; they are ignored",
                           where, name)
        lists[name] = tuple(kept)
    return KeyOptimisation(**lists)


def _settings(raw: Any) -> GatewaySettings:
    if raw is None:
        return GatewaySettings()
    _require(isinstance(raw, Mapping), "settings", "expected a mapping")
    fields = GatewaySettings.__dataclass_fields__
    unknown = set(raw) - set(fields)
    _require(not unknown, "settings", f"unknown setting(s) {', '.join(sorted(unknown))}")
    values = dict(raw)
    if "check_roles" in values:
        values["check_roles"] = _str_list(values["check_roles"], "settings.check_roles")
    for name in ("max_request_bytes", "max_concurrent", "max_parallel_checks"):
        if name in values:
            _require(isinstance(values[name], int) and not isinstance(values[name], bool)
                     and values[name] > 0, "settings", f"{name} must be a positive integer")
    if "body_timeout_seconds" in values:
        _require(isinstance(values["body_timeout_seconds"], (int, float))
                 and values["body_timeout_seconds"] > 0, "settings",
                 "body_timeout_seconds must be > 0")
    if "user_ids" in values:
        _require(values["user_ids"] in ("hash", "plain", "off"), "settings",
                 "user_ids must be hash, plain or off")
    if "uncheckable_content" in values:
        _require(values["uncheckable_content"] in UNCHECKABLE_MODES, "settings",
                 f"uncheckable_content must be one of {', '.join(UNCHECKABLE_MODES)}")
    return GatewaySettings(**values)


def _prices(raw: Any) -> Dict[str, Dict[str, float]]:
    if raw is None:
        return {}
    _require(isinstance(raw, Mapping), "prices", "expected a mapping of model to prices")
    out: Dict[str, Dict[str, float]] = {}
    for model, entry in raw.items():
        _require(isinstance(entry, Mapping), f"prices.{model}", "expected a mapping")
        unknown = set(entry) - {"input_per_mtok", "output_per_mtok", "cached_input_per_mtok",
                                "cache_write_per_mtok"}
        _require(not unknown, f"prices.{model}", f"unknown field(s) {', '.join(sorted(unknown))}")
        for name, value in entry.items():
            _require(isinstance(value, (int, float)) and value >= 0, f"prices.{model}",
                     f"{name} must be a non-negative number")
        _require("input_per_mtok" in entry and "output_per_mtok" in entry, f"prices.{model}",
                 "needs input_per_mtok and output_per_mtok")
        out[str(model)] = {k: float(v) for k, v in entry.items()}
    return out


def from_document(doc: Mapping[str, Any], *, source: str = "document",
                  managed: bool = False) -> GatewayConfig:
    """Validate and build a configuration from a parsed document (file or bundle).
    ``managed``: the document came from a bundle, so this runtime's own limits on
    provider hosts apply (see ``DEFAULT_MANAGED_HOSTS``)."""
    _require(isinstance(doc, Mapping), source, "expected a mapping at the top level")
    unknown = set(doc) - {"version", "providers", "keys", "settings", "prices"}
    _require(not unknown, source, f"unknown section(s) {', '.join(sorted(unknown))}")
    raw_providers = doc.get("providers") or {}
    _require(isinstance(raw_providers, Mapping), "providers", "expected a mapping")
    providers = {str(name): _provider(str(name), raw, managed) for name, raw in raw_providers.items()}
    raw_keys = doc.get("keys") or []
    _require(isinstance(raw_keys, list), "keys", "expected a list")
    keys: Dict[str, GatewayKey] = {}
    ids = set()
    for i, raw in enumerate(raw_keys):
        key = _key(raw, i, providers)
        _require(key.id not in ids, f"keys.{key.id}", "duplicate id")
        _require(key.key_sha256 not in keys, f"keys.{key.id}", "the same key is configured twice")
        ids.add(key.id)
        keys[key.key_sha256] = key
    return GatewayConfig(providers=providers, keys_by_digest=keys,
                         settings=_settings(doc.get("settings")), prices=_prices(doc.get("prices")),
                         source=source)


def from_environment(env: Optional[Mapping[str, str]] = None) -> GatewayConfig:
    """The zero-configuration gateway: one provider per well-known key variable."""
    env = os.environ if env is None else env
    providers: Dict[str, Provider] = {}
    if env.get("OPENAI_API_KEY"):
        providers["openai"] = Provider(name="openai", format="openai",
                                       base_url=env.get("OPENAI_BASE_URL", DEFAULT_BASE_URLS["openai"]).rstrip("/"),
                                       api_key="env:OPENAI_API_KEY")
    if env.get("ANTHROPIC_API_KEY"):
        providers["anthropic"] = Provider(name="anthropic", format="anthropic",
                                          base_url=DEFAULT_BASE_URLS["anthropic"],
                                          api_key="env:ANTHROPIC_API_KEY")
    return GatewayConfig(providers=providers, keys_by_digest={}, source="environment")


def from_bundle_members(documents: List[Tuple[str, Optional[Mapping[str, Any]]]],
                        shared: bool = False) -> GatewayConfig:
    """Managed mode: one configuration from the gateway sections of every environment
    the runtime serves. Each environment's providers and keys stay its own: a key only
    reaches its own environment's providers and policy, and provider names may repeat
    across environments. A section that does not validate is logged and leaves only
    that environment without a gateway; the same gateway key in two environments is
    refused in both."""
    import dataclasses

    providers: Dict[str, Provider] = {}
    keys: Dict[str, GatewayKey] = {}
    clashes: set = set()
    prices: Dict[str, Dict[str, float]] = {}
    member_prices: Dict[str, Dict[str, Dict[str, float]]] = {}
    members: List[str] = []
    for member, doc in documents:
        members.append(member)
        if not doc:
            continue
        try:
            part = from_document(doc, source=f"bundle ({member or 'single'})", managed=True)
        except GatewayConfigError as exc:
            logger.error("Gateway: the section for environment %s was rejected, its keys are not "
                         "served: %s", member or "-", exc)
            continue
        except Exception as exc:  # noqa: BLE001 - one environment's section must not stop the others
            logger.error("Gateway: the section for environment %s could not be read, its keys are not "
                         "served: %s: %s", member or "-", type(exc).__name__, exc)
            continue
        for name, provider in part.providers.items():
            providers[f"{member}/{name}" if member else name] = dataclasses.replace(provider, member=member)
        for digest, key in part.keys_by_digest.items():
            if digest in keys or digest in clashes:
                clashes.add(digest)
                keys.pop(digest, None)
                logger.error("Gateway: key %s is configured in two environments and is refused", key.id)
                continue
            keys[digest] = dataclasses.replace(key, member=member)
        member_prices[member] = part.prices
        for model, entry in part.prices.items():
            prices.setdefault(model, entry)
    return GatewayConfig(providers=providers, keys_by_digest=keys, prices=prices, members=members,
                         shared=shared, member_prices=member_prices,
                         source=f"bundle ({len(members)} environment(s))")


def from_bundle_policy(policies: Optional[Mapping[str, Any]]) -> GatewayConfig:
    """Managed mode: the gateway document the control plane put in the signed bundle
    (``policies["gateway"]``, inside the signed content). No document: no gateway."""
    doc = (policies or {}).get("gateway") if isinstance(policies, Mapping) else None
    if not doc:
        return GatewayConfig(providers={}, keys_by_digest={}, source="bundle (no gateway section)")
    return from_document(doc, source="bundle", managed=True)


def load(path: Optional[str] = None) -> GatewayConfig:
    """Local mode: the gateway file if there is one, else the zero-configuration default."""
    import yaml

    explicit = path or os.getenv("ZNYX_GATEWAY_CONFIG")
    candidate = Path(explicit).expanduser() if explicit else Path("gateway.yaml")
    if not candidate.is_file():
        if explicit:
            raise GatewayConfigError(f"gateway configuration not found: {candidate}")
        return from_environment()
    try:
        doc = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise GatewayConfigError(f"cannot read {candidate}: {exc}") from None
    config = from_document(doc, source=str(candidate))
    logger.info("Gateway: %d provider(s), %d key(s) from %s", len(config.providers),
                len(config.keys_by_digest), candidate)
    return config


def instance_key(settings: GatewaySettings) -> Optional[GatewayKey]:
    """The instance's own RUNTIME_API_KEY as an unrestricted gateway key (when allowed)."""
    raw = (os.getenv("RUNTIME_API_KEY") or "").strip()
    if not raw or not settings.allow_instance_key:
        return None
    return GatewayKey(id="instance", key_sha256=key_digest(raw), instance_key=True)


def model_pattern_list(models: List[str]) -> Tuple[str, ...]:
    return tuple(models)

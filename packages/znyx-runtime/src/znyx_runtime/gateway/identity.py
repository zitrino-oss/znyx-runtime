"""Delegated tokens (Phase 5, part 3): an agent acting for a signed-in person.

``POST /v1/znyx/token``: the agent authenticates with its gateway key and sends the person's
token from the company's identity provider (``subject_token``, a JWT). The gateway verifies it
(``znyx_core.identity``), grants the tools the agent's rules give the person's groups, and
answers a short-lived opaque token (``zat_...``). Only the token's hash is kept, in the shared
state store, with what it stands for; it expires on its own and ``DELETE /v1/znyx/token``
revokes it.

Using it: wherever a gateway key is accepted. The call runs as the agent's key for that person,
with only the granted tools; the rules, the provider and the stops are read again at every use,
so a change in the console applies to tokens already issued.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from znyx_core import identity

logger = logging.getLogger(__name__)

TOKEN_PREFIX = "zat_"
_STORE_PREFIX = "pin:gw:agent-token:"
MIN_REMAINING_SECONDS = 30
MAX_EXCHANGES_PER_MINUTE = 60          # per agent key, across workers: bounds the tokens kept


@dataclass(frozen=True)
class Delegation:
    """What a delegated token stands for, as read at this use."""
    key_sha256: str
    member: str
    agent_id: str
    person: str
    groups: Tuple[str, ...]
    provider: str
    tools: Tuple[str, ...]           # tool name patterns granted now
    expires_at: float


class TokenRefused(Exception):
    def __init__(self, status: int, error: str, description: str):
        super().__init__(description)
        self.status, self.error, self.description = status, error, description


def _minute() -> int:
    return int(time.time() // 60)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class DelegationService:
    def __init__(self, gateway) -> None:
        self.gw = gateway
        self._providers_memo: Optional[tuple] = None
        self._agents_memo: Optional[tuple] = None

    # -- configuration (console controls, else the gateway file) ---------------------

    def providers(self) -> Tuple[identity.Provider, ...]:
        raw = self.gw._controls().get("identity_providers")
        memo = self._providers_memo
        if memo is not None and memo[0] is raw and memo[1] is self.gw.config:
            return memo[2]
        out: List[identity.Provider] = list(self.gw.config.identity_providers)
        for entry in raw if isinstance(raw, list) else []:
            provider, problem = identity.clean_provider(entry)
            if provider is None:
                logger.warning("gateway: an identity provider from the console is ignored: %s", problem)
                continue
            out.append(provider)
        result = tuple(out)
        self._providers_memo = (raw, self.gw.config, result)
        return result

    def agent_rules(self, key, member) -> Optional[identity.AgentRules]:
        raw = self.gw._controls().get("agents")
        memo = self._agents_memo
        if memo is None or memo[0] is not raw or memo[1] is not self.gw.config:
            table: Dict[Tuple[str, str], identity.AgentRules] = {
                (r.project_id, r.agent_id): r for r in self.gw.config.agents}
            for entry in raw if isinstance(raw, list) else []:
                rules, problem = identity.clean_agent(entry)
                if rules is None:
                    logger.warning("gateway: agent rules from the console are ignored: %s", problem)
                    continue
                table[(rules.project_id, rules.agent_id)] = rules
            memo = self._agents_memo = (raw, self.gw.config, table)
        project = member.project_id if member is not None else ""
        return memo[2].get((project, key.agent_id)) or memo[2].get(("", key.agent_id))

    # -- the token store ---------------------------------------------------------------

    @staticmethod
    def _store():
        from znyx_core.state_store import get_state_store
        return get_state_store()

    def resolve(self, token: str) -> Optional[Dict[str, Any]]:
        """The stored record of a token, or None (unknown, revoked or expired)."""
        if not token.startswith(TOKEN_PREFIX) or len(token) > 100:
            return None
        raw = self._store().get_value(_STORE_PREFIX + _digest(token))
        try:
            entry = json.loads(raw) if raw else None
        except ValueError:
            return None
        if not isinstance(entry, dict) or float(entry.get("exp") or 0) <= time.time():
            return None
        return entry

    def revoke(self, token: str) -> bool:
        entry = self.resolve(token)
        if entry is None:
            return False
        self._store().delete(_STORE_PREFIX + _digest(token))
        self._audit("token.revoked", token_ref=_digest(token)[:16], agent_id=entry.get("agent"),
                    person=self.gw._person_id(str(entry.get("person") or "")),
                    identity_provider=entry.get("provider"))
        return True

    def _audit(self, kind: str, **fields: Any) -> None:
        """A governance record (part 4); never the tokens themselves, only a short reference."""
        try:
            self.gw.governance.record(kind, **fields)
        except Exception:  # noqa: BLE001 - the audit never fails a call
            logger.debug("gateway: governance record failed", exc_info=True)

    def delegation_now(self, entry: Dict[str, Any], key, member) -> Delegation:
        """What the token may do at this use: the provider must still be configured and the
        agent's rules still grant something to the person's groups (narrowed to what was
        granted when the token was made)."""
        if entry.get("provider") not in {p.id for p in self.providers()}:
            raise TokenRefused(401, "invalid_token", "the identity provider of this token is no longer configured")
        rules = self.agent_rules(key, member)
        if rules is None:
            raise TokenRefused(401, "invalid_token", "this agent no longer has delegation rules")
        if float(entry.get("iat") or 0) + rules.max_ttl_seconds + 5 < time.time():
            raise TokenRefused(401, "invalid_token", "this token is older than the agent's maximum token lifetime")
        granted_then = [t for t in entry.get("tools") or [] if isinstance(t, str)]
        now = rules.grant(entry.get("groups") or [], granted_then)
        if not now:
            raise TokenRefused(401, "invalid_token", "the agent's rules no longer grant this person any tool")
        return Delegation(key_sha256=entry["key"], member=entry.get("member") or "", agent_id=entry.get("agent") or "",
                          person=str(entry.get("person") or ""), groups=tuple(entry.get("groups") or ()),
                          provider=str(entry.get("provider")), tools=tuple(now), expires_at=float(entry["exp"]))

    # -- the exchange --------------------------------------------------------------------

    def exchange(self, key, member, body: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the person's token and make a delegated one. TokenRefused otherwise."""
        scope = {"org_scope": key.tenant_id, "key_id": key.id, "app_id": key.app_id, "agent_id": key.agent_id,
                 "project_id": getattr(member, "project_id", None),
                 "environment_id": getattr(member, "environment_id", None)}
        try:
            out, record = self._exchange(key, member, body)
        except TokenRefused as refused:
            if refused.status != 429:                     # past the rate limit: counted, not recorded
                self._audit("token.refused", **scope, error=refused.error)
            raise
        self._audit("token.issued", **scope, **record)
        return out

    def _exchange(self, key, member, body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rules = self.agent_rules(key, member)
        if rules is None:
            raise TokenRefused(403, "unauthorized_client", "this agent has no delegation rules")
        # Pinned, so a store full of other entries never evicts the count; it expires on its own.
        made = self._store().incr(f"pin:gw:token-rate:{key.key_sha256[:32]}:{_minute()}", {"n": 1}, 120).get("n", 0)
        if made > MAX_EXCHANGES_PER_MINUTE:
            raise TokenRefused(429, "temporarily_unavailable",
                               f"at most {MAX_EXCHANGES_PER_MINUTE} tokens a minute for one agent; reuse a token "
                               f"until it expires")
        subject = body.get("subject_token")
        if not isinstance(subject, str) or not subject:
            raise TokenRefused(400, "invalid_request", "subject_token is required")
        kind = body.get("subject_token_type")
        if kind not in (None, "", "urn:ietf:params:oauth:token-type:jwt", "urn:ietf:params:oauth:token-type:id_token",
                        "urn:ietf:params:oauth:token-type:access_token"):
            raise TokenRefused(400, "invalid_request", "subject_token_type must name a JWT")
        providers = self.providers()
        if not providers:
            raise TokenRefused(403, "unauthorized_client", "no identity provider is configured")
        try:
            provider, claims = identity.verify_token(subject, providers)
            person, groups = identity.person_of(provider, claims)
        except identity.TokenError as exc:
            raise TokenRefused(400, exc.code, str(exc)) from None
        scope = body.get("scope")
        if isinstance(scope, list) and all(isinstance(s, str) for s in scope):
            scope = " ".join(scope)
        if scope is not None and not isinstance(scope, str):
            raise TokenRefused(400, "invalid_request", "scope must be a space-separated string or a list of tool names")
        asked = [s for s in scope.split() if s][:100] if isinstance(scope, str) and scope.strip() else None
        granted = rules.grant(groups, asked)
        if not granted:
            raise TokenRefused(403, "access_denied", "the agent's rules grant no tools to this person")
        stops = self.gw.stops_for(key, member, person=person)
        if stops.stopped is not None:
            raise TokenRefused(403, "access_denied", stops.stopped.message())
        now = time.time()
        wanted = body.get("expires_in")
        if isinstance(wanted, str) and re.fullmatch(r"\s*\d{1,9}\s*", wanted):
            wanted = int(wanted)                           # a form field
        if wanted is not None and (isinstance(wanted, bool) or not isinstance(wanted, (int, float))
                                   or not math.isfinite(wanted) or wanted <= 0):
            raise TokenRefused(400, "invalid_request", "expires_in must be a positive number of seconds")
        wanted = min(wanted, rules.max_ttl_seconds) if wanted is not None else rules.max_ttl_seconds
        ttl = min(float(wanted), float(rules.max_ttl_seconds), float(claims["exp"]) - now)
        if ttl < MIN_REMAINING_SECONDS:
            raise TokenRefused(400, "invalid_grant", "the person's token expires too soon; sign in again")
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        entry = {"v": 1, "key": key.key_sha256, "member": key.member, "agent": key.agent_id, "person": person,
                 "groups": list(groups)[:identity.MAX_GROUPS], "provider": provider.id, "tools": list(granted),
                 "iat": now, "exp": now + ttl}
        self._store().set_value(_STORE_PREFIX + _digest(token), json.dumps(entry), ttl + 5)
        logger.info("gateway: delegated token for agent %s acting for a person of %s, %d tool pattern(s), %d s",
                    key.agent_id, provider.id, len(granted), int(ttl))
        return ({"access_token": token, "token_type": "Bearer",
                 "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                 "expires_in": int(ttl), "scope": " ".join(granted)},
                {"token_ref": _digest(token)[:16], "person": self.gw._person_id(person), "identity_provider": provider.id,
                 "tools": list(granted)[:50], "expires_in": int(ttl)})

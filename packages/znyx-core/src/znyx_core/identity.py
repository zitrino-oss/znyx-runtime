"""Agent identity (AI gateway Phase 5, part 3): verifying a person's identity-provider token,
and what an agent may do for that person.

An agent exchanges its gateway key and a person's token from the company's identity provider
(OIDC: an ID or access token, a JWT) for a short-lived gateway token that acts for that person
with the tools the agent's rules grant to the person's groups.

Verification uses ``cryptography`` only (no JWT library): the provider's public keys (JWKS),
asymmetric algorithms only (RS/PS/ES 256 to 512, EdDSA; never ``none`` or a shared secret),
the issuer, an accepted audience, and the token's time claims with a small leeway.

Shared by the runtime (verification, grants) and the control plane (validating what it stores).
"""
from __future__ import annotations

import base64
import fnmatch
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ALGORITHMS = ("RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512", "EdDSA")
LEEWAY_SECONDS = 60
MAX_TOKEN_BYTES = 16 * 1024
MAX_KEYS = 50
MAX_GROUPS = 200
DEFAULT_TTL_SECONDS = 900
MAX_TTL_SECONDS = 3600
MIN_TTL_SECONDS = 60
_PRIVATE_MEMBERS = frozenset({"d", "p", "q", "dp", "dq", "qi", "oth", "k"})
_CURVES = {"ES256": "P-256", "ES384": "P-384", "ES512": "P-521"}
_ID = re.compile(r"[A-Za-z0-9._:@/-]{1,200}")
RULE_NOT_DELEGATED = "identity.tool_not_delegated"


class TokenError(ValueError):
    """A token that is not accepted; ``code`` is the OAuth error to answer with."""

    def __init__(self, message: str, code: str = "invalid_grant"):
        super().__init__(message)
        self.code = code


def _b64(segment: str) -> bytes:
    if not isinstance(segment, str) or not re.fullmatch(r"[A-Za-z0-9_-]*", segment):
        raise TokenError("the token is not base64url")
    try:
        return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
    except ValueError:                                     # binascii.Error: a length base64 cannot have
        raise TokenError("the token is not base64url") from None


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")              # NaN and Infinity: never in a real token


# Times in a token: seconds since 1970, finite and before the year 5000.
MAX_TIME = 95_000_000_000


def _int(value: Any) -> int:
    return int.from_bytes(_b64(value), "big")


# -- keys --------------------------------------------------------------------------------------

def clean_jwks(raw: Any) -> Dict[str, Any]:
    """A provider's key set as it may be stored and shipped: public signing keys only
    (``kty`` RSA, EC or OKP); a key with any private part refuses the whole set."""
    keys = raw.get("keys") if isinstance(raw, Mapping) else raw
    if not isinstance(keys, list) or not keys:
        raise ValueError("the key set has no keys")
    out = []
    for i, key in enumerate(keys[:MAX_KEYS]):
        if not isinstance(key, Mapping):
            raise ValueError(f"keys[{i}] is not an object")
        if _PRIVATE_MEMBERS & set(key):
            raise ValueError("the key set holds a private key: give public keys only")
        if key.get("kty") not in ("RSA", "EC", "OKP") or key.get("use", "sig") != "sig":
            continue                                   # encryption keys and others are not ours
        kept = {k: v for k, v in key.items() if k in ("kty", "kid", "alg", "use", "n", "e", "crv", "x", "y")}
        public_key(kept)                               # refuses a malformed key now, not at first use
        out.append(kept)
    if not out:
        raise ValueError("the key set has no public signing keys")
    return {"keys": out}


def public_key(jwk: Mapping[str, Any]):
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, ed448, rsa
    kty = jwk.get("kty")
    try:
        if kty == "RSA":
            numbers = rsa.RSAPublicNumbers(_int(jwk["e"]), _int(jwk["n"]))
            key = numbers.public_key()
            if key.key_size < 2048:
                raise ValueError("RSA keys under 2048 bits are not accepted")
            return key
        if kty == "EC":
            curve = {"P-256": ec.SECP256R1(), "P-384": ec.SECP384R1(), "P-521": ec.SECP521R1()}.get(jwk.get("crv"))
            if curve is None:
                raise ValueError(f"unsupported curve {jwk.get('crv')!r}")
            return ec.EllipticCurvePublicNumbers(_int(jwk["x"]), _int(jwk["y"]), curve).public_key()
        if kty == "OKP":
            raw = _b64(jwk["x"])
            if jwk.get("crv") == "Ed25519":
                return ed25519.Ed25519PublicKey.from_public_bytes(raw)
            if jwk.get("crv") == "Ed448":
                return ed448.Ed448PublicKey.from_public_bytes(raw)
            raise ValueError(f"unsupported curve {jwk.get('crv')!r}")
    except (KeyError, TypeError, TokenError) as exc:
        raise ValueError(f"malformed {kty} key") from exc
    raise ValueError(f"unsupported key type {kty!r}")


def _verify_signature(alg: str, jwk: Mapping[str, Any], signing_input: bytes, signature: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, ed448, ed25519, padding, rsa
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
    key = public_key(jwk)
    digest = {"256": hashes.SHA256(), "384": hashes.SHA384(), "512": hashes.SHA512()}.get(alg[-3:])
    try:
        if alg.startswith("RS") and isinstance(key, rsa.RSAPublicKey):
            key.verify(signature, signing_input, padding.PKCS1v15(), digest)
        elif alg.startswith("PS") and isinstance(key, rsa.RSAPublicKey):
            key.verify(signature, signing_input, padding.PSS(mgf=padding.MGF1(digest), salt_length=digest.digest_size),
                       digest)
        elif alg.startswith("ES") and isinstance(key, ec.EllipticCurvePublicKey):
            if jwk.get("crv") != _CURVES[alg]:
                return False
            size = (key.curve.key_size + 7) // 8
            if len(signature) != 2 * size:
                return False
            r, s = int.from_bytes(signature[:size], "big"), int.from_bytes(signature[size:], "big")
            key.verify(encode_dss_signature(r, s), signing_input, ec.ECDSA(digest))
        elif alg == "EdDSA" and isinstance(key, (ed25519.Ed25519PublicKey, ed448.Ed448PublicKey)):
            key.verify(signature, signing_input)
        else:
            return False
        return True
    except InvalidSignature:
        return False


# -- providers ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Provider:
    id: str
    issuer: str
    audiences: Tuple[str, ...]
    jwks: Mapping[str, Any]
    user_claim: str = "email"
    groups_claim: str = "groups"
    algorithms: Tuple[str, ...] = ALGORITHMS


def clean_provider(raw: Any) -> Tuple[Optional[Provider], Optional[str]]:
    """(the provider, None) or (None, why it cannot be used)."""
    if not isinstance(raw, Mapping):
        return None, "not a mapping"
    pid, issuer = raw.get("id"), raw.get("issuer")
    if not isinstance(pid, str) or not _ID.fullmatch(pid):
        return None, "id must be text"
    if not isinstance(issuer, str) or not issuer.startswith("https://") or len(issuer) > 500:
        return None, "issuer must be an https URL"
    audiences = raw.get("audiences")
    if isinstance(audiences, str):
        audiences = [audiences]
    if not isinstance(audiences, list) or not audiences or not all(isinstance(a, str) and a for a in audiences):
        return None, "audiences must name at least one audience"
    algorithms = raw.get("algorithms") or list(ALGORITHMS)
    if not isinstance(algorithms, list) or not algorithms or any(a not in ALGORITHMS for a in algorithms):
        return None, f"algorithms must be among {', '.join(ALGORITHMS)}"
    try:
        jwks = clean_jwks(raw.get("jwks"))
    except ValueError as exc:
        return None, f"jwks: {exc}"
    user_claim = raw.get("user_claim") or "email"
    groups_claim = raw.get("groups_claim") or "groups"
    if not all(isinstance(c, str) and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,100}", c) for c in (user_claim, groups_claim)):
        return None, "claim names must be plain text"
    return Provider(id=pid, issuer=issuer.rstrip("/"), audiences=tuple(audiences[:20]), jwks=jwks,
                    user_claim=user_claim, groups_claim=groups_claim, algorithms=tuple(algorithms)), None


def _claim(claims: Mapping[str, Any], path: str) -> Any:
    value: Any = claims
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return claims.get(path)                       # a claim whose name contains dots (URLs)
        value = value[part]
    return value


def verify_token(token: str, providers: Iterable[Provider], now: Optional[float] = None) -> Tuple[Provider, Dict[str, Any]]:
    """The provider that issued ``token`` and its claims; TokenError when it is not accepted."""
    if not isinstance(token, str) or len(token) > MAX_TOKEN_BYTES or token.count(".") != 2:
        raise TokenError("subject_token must be a signed JWT", "invalid_request")
    head_b64, body_b64, sig_b64 = token.split(".")
    try:
        header = json.loads(_b64(head_b64), parse_constant=_no_constant)
        claims = json.loads(_b64(body_b64), parse_constant=_no_constant)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise TokenError("the token cannot be read") from None
    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise TokenError("the token cannot be read")
    alg = header.get("alg")
    if alg not in ALGORITHMS:
        raise TokenError(f"algorithm {alg!r} is not accepted")
    if header.get("crit"):
        raise TokenError("the token requires extensions this gateway does not support")
    issuer = str(claims.get("iss") or "").rstrip("/")
    provider = next((p for p in providers if p.issuer == issuer), None)
    if provider is None:
        raise TokenError("the token's issuer is not a configured identity provider")
    if alg not in provider.algorithms:
        raise TokenError(f"algorithm {alg} is not accepted for this provider")
    kid = header.get("kid")
    candidates = [k for k in provider.jwks.get("keys", []) if k.get("alg") in (None, alg)
                  and (kid is None or k.get("kid") == kid)]
    if kid is None and len(candidates) != 1:
        raise TokenError("the token names no key and the provider has several")
    signing_input = f"{head_b64}.{body_b64}".encode("ascii")
    signature = _b64(sig_b64)
    if not any(_verify_signature(alg, jwk, signing_input, signature) for jwk in candidates[:5]):
        raise TokenError("the token's signature does not verify")
    now = time.time() if now is None else now
    for name in ("exp", "nbf", "iat"):
        value = claims.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not 0 <= value < MAX_TIME):
            raise TokenError(f"{name} is not a time")
    if claims.get("exp") is None or claims["exp"] <= now - LEEWAY_SECONDS:
        raise TokenError("the token has expired")
    if claims.get("nbf") is not None and claims["nbf"] > now + LEEWAY_SECONDS:
        raise TokenError("the token is not valid yet")
    if claims.get("iat") is not None and claims["iat"] > now + LEEWAY_SECONDS:
        raise TokenError("the token was issued in the future")
    audience = claims.get("aud")
    audiences = [audience] if isinstance(audience, str) else audience if isinstance(audience, list) else []
    if not set(a for a in audiences if isinstance(a, str)) & set(provider.audiences):
        raise TokenError("the token is not meant for this gateway (audience)")
    return provider, claims


def person_of(provider: Provider, claims: Mapping[str, Any]) -> Tuple[str, List[str]]:
    """(who the person is, their groups). The person comes from the provider's user claim only:
    a token without it is refused (falling back to another claim would give one person two
    identities, and a stop on one would miss the other). An email is taken only when the provider
    does not say it is unverified, and in lower case, as stops match it."""
    person = _claim(claims, provider.user_claim)
    if not isinstance(person, str) or not person.strip():
        raise TokenError(f"the token has no {provider.user_claim} claim; give the identity provider a user claim "
                         f"every token carries (for example sub)")
    if provider.user_claim == "email" and claims.get("email_verified") is False:
        raise TokenError("the token's email is not verified by the identity provider")
    person = person.strip()[:200]
    if "@" in person:
        person = person.lower()                               # an email: not case-sensitive in practice
    groups = _claim(claims, provider.groups_claim)
    if isinstance(groups, str):
        groups = [groups]
    groups = [g for g in groups if isinstance(g, str)][:MAX_GROUPS] if isinstance(groups, list) else []
    return person, groups


# -- what an agent may do for a person ---------------------------------------------------------

@dataclass(frozen=True)
class AgentRules:
    agent_id: str
    project_id: str = ""
    groups: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    all_users: Tuple[str, ...] = ()
    max_ttl_seconds: int = DEFAULT_TTL_SECONDS
    require_delegation: bool = False

    def grant(self, groups: Iterable[str], scope: Optional[Iterable[str]] = None) -> Tuple[str, ...]:
        """The tool patterns this agent may use for a person in ``groups``, narrowed to
        ``scope`` (names or patterns the agent asks for; each must fall within the grant)."""
        allowed: List[str] = list(self.all_users)
        for group in groups:
            allowed += list(self.groups.get(group, ()))
        allowed = list(dict.fromkeys(allowed))
        if scope is None:
            return tuple(allowed)
        granted = []
        for asked in scope:
            if not asked:
                continue
            if any(ch in asked for ch in "*?["):
                # A pattern asked for counts only as itself: matching it against an allowed
                # pattern as a name could grant more than allowed ("a*" against "a?").
                if asked in allowed:
                    granted.append(asked)
            elif any(fnmatch.fnmatchcase(asked, p) for p in allowed):
                granted.append(asked)
        return tuple(dict.fromkeys(granted))


def _patterns(value: Any, where: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) and 0 < len(v) <= 200 for v in value):
        raise ValueError(f"{where} must be a list of tool name patterns")
    return tuple(value[:200])


def clean_agent(raw: Any) -> Tuple[Optional[AgentRules], Optional[str]]:
    if not isinstance(raw, Mapping):
        return None, "not a mapping"
    agent_id = raw.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id or len(agent_id) > 200:
        return None, "agent_id is required"
    try:
        groups_raw = raw.get("groups") or {}
        if not isinstance(groups_raw, Mapping):
            raise ValueError("groups maps a group name to tool name patterns")
        groups = {str(g)[:200]: _patterns(p, f"groups.{g}") for g, p in list(groups_raw.items())[:200]}
        all_users = _patterns(raw.get("all_users"), "all_users")
    except ValueError as exc:
        return None, str(exc)
    ttl = raw.get("max_ttl_seconds", DEFAULT_TTL_SECONDS)
    if isinstance(ttl, bool) or not isinstance(ttl, int) or not MIN_TTL_SECONDS <= ttl <= MAX_TTL_SECONDS:
        return None, f"max_ttl_seconds must be from {MIN_TTL_SECONDS} to {MAX_TTL_SECONDS}"
    return AgentRules(agent_id=agent_id, project_id=str(raw.get("project_id") or ""), groups=groups,
                      all_users=all_users, max_ttl_seconds=ttl,
                      require_delegation=raw.get("require_delegation") is True), None


def tool_granted(patterns: Iterable[str], name: Optional[str]) -> bool:
    return bool(name) and any(fnmatch.fnmatchcase(name, p) for p in patterns)

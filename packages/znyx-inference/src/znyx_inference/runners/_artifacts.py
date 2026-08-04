"""Pinned-artifact helpers for heavy runners (no implicit downloads).

A runner must load weights ONLY from operator-supplied local files, and verify the
pinned sha256 before use — failing closed (RunnerUnavailable) on a missing dir or a
hash mismatch. Dependency-free (stdlib hashlib only) so it's testable without torch.
"""
from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

from znyx_inference.runners.base import RunnerUnavailable

_CHUNK = 1 << 20  # 1 MiB


def _hash_file(path: Path, h) -> None:
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)


def artifact_sha256(path: str) -> str:
    """Deterministic sha256 of a model artifact — a single file, or a directory hashed
    over its sorted relative paths + contents (so a whole HF model dir has one stable
    digest)."""
    p = Path(path)
    h = sha256()
    if p.is_file():
        _hash_file(p, h)
        return h.hexdigest()
    if p.is_dir():
        for f in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(str(f.relative_to(p)).encode("utf-8"))
            h.update(b"\0")
            _hash_file(f, h)
        return h.hexdigest()
    raise RunnerUnavailable(f"model artifact not found: {path}")


def resolve_artifact_dir(spec: dict) -> str:
    """Local directory holding the pinned model. Resolution order:
    spec['artifacts_dir'] → $ZNYX_INFERENCE_ARTIFACTS_DIR/<model_id> → ~/.znyx/models/<model_id>.
    Never a network location — the service must not pull weights at startup."""
    model_id = spec.get("model_id") or "model"
    safe = model_id.replace("/", "__")
    explicit = spec.get("artifacts_dir")
    if explicit:
        return explicit
    base = os.getenv("ZNYX_INFERENCE_ARTIFACTS_DIR") or str(Path.home() / ".znyx" / "models")

    # Containment check. Replacing "/" neutralises the common case but NOT a model_id of
    # ".." (which resolves to the parent of the artifact root) or a backslash form on
    # Windows. Normally unreachable, because resolve_fetch_target validates model_id against
    # the vetted shortlist first — but that check is bypassable with
    # ZNYX_INFERENCE_ALLOW_UNVETTED, so this is the backstop that keeps a hostile pin from
    # writing weights outside the artifact directory. Deliberately a containment assertion
    # rather than extra sanitising, so legitimate names keep resolving to the SAME directory
    # and no already-cached model is orphaned.
    base_path = Path(base).resolve()
    candidate = (base_path / safe).resolve()
    if candidate != base_path and base_path not in candidate.parents:
        raise ValueError(
            f"refusing model_id {model_id!r}: it resolves outside the artifact directory")
    return str(base_path / safe)


def verify_pinned(path: str, expected_sha256: str | None) -> None:
    """Fail closed unless the artifact exists and (if pinned) matches its sha256."""
    p = Path(path)
    if not p.exists():
        raise RunnerUnavailable(f"model artifact not found (no implicit download): {path}")
    if expected_sha256:
        actual = artifact_sha256(path)
        if actual != expected_sha256:
            raise RunnerUnavailable(
                f"sha256 mismatch for {path}: expected {expected_sha256}, got {actual}")

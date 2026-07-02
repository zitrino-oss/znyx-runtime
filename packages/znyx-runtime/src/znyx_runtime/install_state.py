"""
Shared install state for ZNYX Runtime.

Manages ~/.znyx/state.json - a persistent JSON file that tracks:
- install_id: random UUID (stable across runs)
- first_run_at: ISO timestamp of the first launch
- last_run_at: ISO timestamp of the most recent launch
- run_count: total number of times the runtime has been started
- mode: last used runtime mode ("local" or "managed")

Migration: if the legacy ~/.guardrails_install_id file exists its UUID is
imported into state.json so existing installs keep the same identifier.

Opt-out: telemetry reads are gated by the caller; this module only persists state.
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_DIR = Path.home() / ".znyx"
STATE_FILE = STATE_DIR / "state.json"
LEGACY_INSTALL_ID_FILE = Path.home() / ".guardrails_install_id"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrate_legacy_install_id() -> Optional[str]:
    """Read the legacy .guardrails_install_id file and return its UUID, or None."""
    try:
        if LEGACY_INSTALL_ID_FILE.exists():
            install_id = LEGACY_INSTALL_ID_FILE.read_text().strip()
            if install_id:
                logger.debug(f"Migrating legacy install ID from {LEGACY_INSTALL_ID_FILE}")
                return install_id
    except OSError:
        pass
    return None


def load_state() -> dict:
    """Load state.json, creating it (with migration) if it doesn't exist."""
    try:
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            # Ensure all expected keys are present (forward-compat for older state files)
            data.setdefault("run_count", 0)
            data.setdefault("mode", "local")
            return data
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"Could not read state file, creating new one: {e}")

    # No state file yet - create fresh state, migrating legacy install ID if present
    legacy_id = _migrate_legacy_install_id()
    now = _now_iso()
    state = {
        "install_id": legacy_id or str(uuid.uuid4()),
        "first_run_at": now,
        "last_run_at": now,
        "run_count": 0,
        "mode": "local",
    }
    save_state(state)
    return state


def save_state(state: dict) -> None:
    """Persist state.json atomically (best-effort)."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except OSError as e:
        logger.debug(f"Could not save state file (non-fatal): {e}")


def record_run(mode: str = "local") -> dict:
    """
    Increment run_count, update last_run_at and mode, save, and return the
    updated state dict.  Call this once per runtime startup.
    """
    state = load_state()
    state["run_count"] = state.get("run_count", 0) + 1
    state["last_run_at"] = _now_iso()
    state["mode"] = mode
    save_state(state)
    return state


def get_install_id() -> str:
    """Return the persistent install ID, creating state if needed."""
    return load_state()["install_id"]


def get_run_count() -> int:
    """Return the current run count without modifying state."""
    return load_state().get("run_count", 0)

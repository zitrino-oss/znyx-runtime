"""Inspectable, durable, hash-only contributions, sent only while consent is active."""
import json
import hashlib
from pathlib import Path

from znyx_core import defence
from znyx_core.utils.spool import append_lines, spool_lock


def active(feed):
    from datetime import datetime, timezone
    try:
        return feed.get("enabled") is True and datetime.fromisoformat(feed["expires_at"].replace("Z", "+00:00")) > datetime.now(timezone.utc)
    except (TypeError, KeyError, ValueError):
        return False


def feed_state(feed):
    """Expiry and renewal must invalidate an ALLOW cached under the same revision."""
    if not feed:
        return ""
    revision = feed.get("revision") or hashlib.sha256(json.dumps(feed, sort_keys=True).encode()).hexdigest()
    return str(revision) + (":active" if active(feed) else ":inactive")


class DefenceOutbox:
    def __init__(self, path):
        self.path = Path(path)
        self.history = self.path.with_suffix(".history")

    def record(self, text, rule_ids, feed):
        if not active(feed):
            return
        category = next((category for marker, category in (
            ("jailbreak", "jailbreak"), ("injection", "prompt_injection"), ("exfiltration", "exfiltration"))
            if any(marker in str(rule) for rule in rule_ids)), None)
        if category is None:
            return
        value = defence.signature(text, category)
        from znyx_core.state_store import get_state_store
        store = get_state_store()
        if not store.set_value("defence:contribution:" + value["digest"], "1", 86400, only_if_absent=True):
            return
        # Both files contain the exact public schema, never the matched text or rule messages.
        encoded = json.dumps(value, separators=(",", ":"))
        append_lines(self.history, [encoded])
        append_lines(self.path, [encoded])

    def inspect(self, limit=500):
        if not self.history.exists():
            return []
        from collections import deque
        with spool_lock(self.history), self.history.open() as handle:
            return [defence.clean_signature(json.loads(line)) for line in deque(handle, maxlen=limit)]

    def inspect_page(self, cursor=0, limit=500):
        """Byte cursors expose the full append-only history with bounded page memory."""
        if not self.history.exists():
            return {"contributions": [], "next_cursor": None}
        with spool_lock(self.history), self.history.open("rb") as handle:
            if cursor:
                handle.seek(cursor - 1)
                if handle.read(1) != b"\n":
                    raise ValueError("cursor must point to a record boundary")
            handle.seek(cursor)
            entries = []
            for _ in range(limit):
                line = handle.readline(4096)
                if not line:
                    break
                entries.append(defence.clean_signature(json.loads(line)))
            next_cursor = handle.tell()
            return {"contributions": entries, "next_cursor": next_cursor if handle.read(1) else None}

    def sender(self, gateway, post):
        async def send(url, events):
            if not active(gateway._controls().get("shared_defence") or {}):
                return True  # opted out: discard queued contributions without transmitting them
            return await post(url, [defence.clean_signature(event) for event in events])
        return send

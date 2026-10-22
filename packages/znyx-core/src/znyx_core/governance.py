"""Stops: the kill switch (AI gateway Phase 5).

An administrator stops or quarantines an agent, app or key, or stops a model, provider, tool
or everything in an org, project or environment. The control plane delivers the active stops
in the signed bundle set's controls (``controls.stops``), built from its database at every
poll, so a stop reaches every runtime with its next poll and no publish. A runtime without a
console reads the same list from its gateway file (``stops:``).

- ``stop``: calls in scope are refused.
- ``quarantine`` (app, agent, key): the agent can still talk but cannot act: no tools are
  offered to the model, tool calls in answers are withheld, and the evaluate API refuses tool,
  plan, step and memory checks.

Shared by the runtime (enforcement) and the control plane (validation), so both read a stop
the same way. Lenient on read: a stop with a target or mode this version does not know (a
newer control plane) is ignored with a warning, never an error.
"""
from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

TARGETS = ("everything", "app", "agent", "key", "model", "provider", "tool")
MODES = ("stop", "quarantine")
QUARANTINABLE = frozenset({"app", "agent", "key"})
GLOB_TARGETS = frozenset({"model", "tool"})       # match is a glob pattern
MAX_MATCH = 200
MAX_REASON = 200
MAX_ID = 64
MAX_STOPS = 1000

RULE_STOPPED = "kill_switch.stopped"
RULE_QUARANTINED = "kill_switch.quarantined"
RULE_TOOL_STOPPED = "kill_switch.tool_stopped"

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class Stop:
    id: str
    target: str
    mode: str = "stop"
    match: str = ""
    project_id: str = ""
    environment_id: str = ""
    reason: str = ""
    until: Optional[datetime] = None

    def active(self, now: datetime) -> bool:
        return self.until is None or self.until > now

    def in_scope(self, project_id: Optional[str], environment_id: Optional[str]) -> bool:
        return ((not self.project_id or self.project_id == (project_id or ""))
                and (not self.environment_id or self.environment_id == (environment_id or "")))

    def matches(self, value: Optional[str]) -> bool:
        if not value:
            return False
        if self.target in GLOB_TARGETS:
            return fnmatch.fnmatchcase(value, self.match)
        return value == self.match

    def message(self) -> str:
        what = {"everything": "All AI calls here", "app": f"App {self.match}", "agent": f"Agent {self.match}",
                "key": f"Gateway key {self.match}", "model": f"Model {self.match}",
                "provider": f"Provider {self.match}", "tool": f"Tool {self.match}"}.get(self.target, "This call")
        verb = "quarantined (it may not use tools)" if self.mode == "quarantine" else "stopped"
        text = f"{what} {'are' if self.target == 'everything' else 'is'} {verb} by an administrator"
        return f"{text}: {self.reason}" if self.reason else text


def parse_time(value: Any) -> Optional[datetime]:
    """An ISO 8601 time as an aware UTC datetime; ValueError when it is not one."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise ValueError("not a time")
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


def _text(value: Any, limit: int) -> Optional[str]:
    if value is None:
        return ""
    if not isinstance(value, str):
        return None
    return _CONTROL.sub(" ", value).strip()[:limit]


def clean_stop(raw: Any) -> Tuple[Optional[Stop], Optional[str]]:
    """(the stop, None) or (None, why it cannot be used)."""
    if not isinstance(raw, dict):
        return None, "not a mapping"
    target = raw.get("target")
    if target not in TARGETS:
        return None, f"unknown target {target!r}"
    mode = raw.get("mode", "stop")
    if mode not in MODES:
        return None, f"unknown mode {mode!r}"
    if mode == "quarantine" and target not in QUARANTINABLE:
        return None, f"{target} cannot be quarantined, only stopped"
    match = raw.get("match")
    if target == "everything":
        if match not in (None, ""):
            return None, "a stop of everything takes no match"
        match = ""
    else:
        if not isinstance(match, str) or not match.strip() or len(match) > MAX_MATCH or _CONTROL.search(match):
            return None, f"a {target} stop needs a match of 1 to {MAX_MATCH} characters"
        match = match.strip()
    ids = {}
    for name in ("id", "project_id", "environment_id"):
        value = _text(raw.get(name), MAX_ID)
        if value is None:
            return None, f"{name} must be text"
        ids[name] = value
    reason = _text(raw.get("reason"), MAX_REASON)
    if reason is None:
        return None, "reason must be text"
    until = None
    if raw.get("until") not in (None, ""):
        try:
            until = parse_time(raw.get("until"))
        except (TypeError, ValueError):
            return None, "until is not a time"
    return Stop(id=ids["id"], target=target, mode=mode, match=match, project_id=ids["project_id"],
                environment_id=ids["environment_id"], reason=reason, until=until), None


def clean_stops(raw: Any, source: str = "controls") -> Tuple[Stop, ...]:
    """The usable stops of a list; the rest dropped with a warning."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        logger.warning("%s: stops is not a list and is ignored", source)
        return ()
    out: List[Stop] = []
    for i, entry in enumerate(raw[:MAX_STOPS]):
        stop, problem = clean_stop(entry)
        if stop is None:
            logger.warning("%s: stops[%d] is ignored: %s", source, i, problem)
            continue
        out.append(stop)
    if len(raw) > MAX_STOPS:
        logger.warning("%s: only the first %d stops are read", source, MAX_STOPS)
    return tuple(out)


@dataclass(frozen=True)
class CallStops:
    """The stops that concern one caller: one that refuses its calls, one that quarantines
    it, and the tool stops in its scope."""
    stopped: Optional[Stop] = None
    quarantined: Optional[Stop] = None
    tools: Tuple[Stop, ...] = ()

    def tool_stop(self, name: Optional[str]) -> Optional[Stop]:
        return next((s for s in self.tools if s.matches(name or "")), None)

    @property
    def restricts_tools(self) -> bool:
        return self.quarantined is not None or bool(self.tools)


NO_STOPS = CallStops()


class StopSet:
    """Active stops, asked per caller, model or provider."""

    def __init__(self, stops: Iterable[Stop] = ()):
        self.stops: Tuple[Stop, ...] = tuple(stops)

    def __bool__(self) -> bool:
        return bool(self.stops)

    def _live(self, project_id: Optional[str], environment_id: Optional[str], now: Optional[datetime]):
        now = now or datetime.now(timezone.utc)
        return (s for s in self.stops if s.active(now) and s.in_scope(project_id, environment_id))

    def for_caller(self, *, project_id: Optional[str], environment_id: Optional[str], app_id: Optional[str],
                   agent_id: Optional[str], key_id: Optional[str], now: Optional[datetime] = None) -> CallStops:
        if not self.stops:
            return NO_STOPS
        stopped = quarantined = None
        tools: List[Stop] = []
        values = {"app": app_id, "agent": agent_id, "key": key_id}
        for s in self._live(project_id, environment_id, now):
            if s.target == "everything":
                stopped = stopped or s
            elif s.target in values:
                if s.matches(values[s.target]):
                    if s.mode == "stop":
                        stopped = stopped or s
                    else:
                        quarantined = quarantined or s
            elif s.target == "tool":
                tools.append(s)
        if stopped is None and quarantined is None and not tools:
            return NO_STOPS
        return CallStops(stopped, quarantined, tuple(tools))

    def _first(self, target: str, value: Optional[str], project_id: Optional[str],
               environment_id: Optional[str], now: Optional[datetime]) -> Optional[Stop]:
        return next((s for s in self._live(project_id, environment_id, now)
                     if s.target == target and s.matches(value)), None)

    def model(self, model: Optional[str], project_id: Optional[str] = None,
              environment_id: Optional[str] = None, now: Optional[datetime] = None) -> Optional[Stop]:
        return self._first("model", model, project_id, environment_id, now)

    def provider(self, name: Optional[str], project_id: Optional[str] = None,
                 environment_id: Optional[str] = None, now: Optional[datetime] = None) -> Optional[Stop]:
        return self._first("provider", name, project_id, environment_id, now)


def to_document(stop: Stop) -> Dict[str, Any]:
    """A stop as the control plane puts it in the signed controls (only the fields set)."""
    out: Dict[str, Any] = {"id": stop.id, "target": stop.target, "mode": stop.mode}
    for name in ("match", "project_id", "environment_id", "reason"):
        if getattr(stop, name):
            out[name] = getattr(stop, name)
    if stop.until is not None:
        out["until"] = stop.until.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return out

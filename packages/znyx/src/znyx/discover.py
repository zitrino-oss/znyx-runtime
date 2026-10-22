"""``znyx discover``: find AI services reached around the gateway, in network logs.

Runs inside the customer's network on logs it already collects: Squid access logs, W3C
extended logs (IIS, many secure web gateways), Zeek ``dns``/``http``/``ssl`` logs (TSV or JSON),
and any CSV or JSON lines export with a header (a column mapping picks the fields). Each line's
destination host is matched against ``znyx_core.ai_services``; the result is aggregates per
service and host (how many users and requests, first and last seen). Log lines never leave the
machine; ``--upload`` sends only the aggregates to the console.

User names are pseudonymised (a keyed hash, stable on this install) unless
``--keep-user-names`` is given; a line without a user counts its client address as the user,
pseudonymised the same way. ``--exclude-source`` leaves out the gateway's own traffic (its
outbound addresses), which is sanctioned.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import hmac
import io
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

FORMATS = ("auto", "squid", "w3c", "zeek", "csv", "jsonl")
MAX_TRACKED_USERS = 100_000          # per finding; beyond it the count is a lower bound
TOP_USERS = 20

HOST_KEYS = ("host", "hostname", "dest_host", "destination_host", "domain", "server_name", "sni", "query",
             "cs-host", "cs(host)", "r-host", "url", "uri", "cs-uri", "dest", "destination")
USER_KEYS = ("user", "username", "user_name", "src_user", "login", "cs-username", "userid", "user_id")
TIME_KEYS = ("time", "timestamp", "datetime", "ts", "_time", "date_time", "@timestamp", "eventtime")
CLIENT_KEYS = ("client", "client_ip", "src_ip", "source_ip", "c-ip", "id.orig_h", "src", "source")


@dataclass
class Line:
    host: Optional[str]
    user: Optional[str] = None
    client: Optional[str] = None
    at: Optional[datetime] = None


@dataclass
class _Finding:
    service: str
    vendor: str
    category: str
    host: str
    requests: int = 0
    users: Dict[str, int] = field(default_factory=dict)
    capped: bool = False
    first: Optional[datetime] = None
    last: Optional[datetime] = None


_HOST = re.compile(r"[a-z0-9_-]+(\.[a-z0-9_-]+)*")


def _host_of(value: Optional[str]) -> Optional[str]:
    """A destination host from a host, ``host:port``, or URL field; None for anything that is
    not a plain host name (so no user name in ``user@host``, path or other text can end up
    reported as a host)."""
    if not value or value == "-":
        return None
    value = value.strip()[:2048]
    if "://" in value:
        try:
            value = urlsplit(value).hostname or ""
        except ValueError:
            return None
    else:
        value = value.split("/", 1)[0].rsplit("@", 1)[-1]
        if value.count(":") == 1:
            value = value.split(":", 1)[0]
    value = value.lower().rstrip(".")
    return value if 0 < len(value) <= 253 and _HOST.fullmatch(value) else None


def _time(value: Any) -> Optional[datetime]:
    if value in (None, "", "-"):
        return None
    try:
        number = float(value)
        if number > 1e12:                      # milliseconds
            number /= 1000.0
        return datetime.fromtimestamp(number, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    text = str(value).strip().replace("Z", "+00:00")
    for candidate in (text, text.replace(" ", "T", 1)):
        try:
            moment = datetime.fromisoformat(candidate)
            return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _pick(record: Dict[str, Any], keys: Sequence[str], explicit: Optional[str]) -> Optional[str]:
    if explicit:
        value = record.get(explicit)
        return str(value) if value not in (None, "") else None
    lowered = {str(k).lower(): v for k, v in record.items()}
    for key in keys:
        value = lowered.get(key)
        if value not in (None, "", "-"):
            return str(value)
    return None


@dataclass
class Columns:
    host: Optional[str] = None
    user: Optional[str] = None
    time: Optional[str] = None
    client: Optional[str] = None

    def line(self, record: Dict[str, Any]) -> Line:
        return Line(host=_host_of(_pick(record, HOST_KEYS, self.host)), user=_pick(record, USER_KEYS, self.user),
                    client=_pick(record, CLIENT_KEYS, self.client), at=_time(_pick(record, TIME_KEYS, self.time)))


# -- readers ----------------------------------------------------------------------------

def _squid(lines: Iterable[str]) -> Iterator[Line]:
    for raw in lines:
        parts = raw.split()
        if len(parts) < 7:
            continue
        user = parts[7] if len(parts) > 7 and parts[7] != "-" else None
        yield Line(host=_host_of(parts[6]), user=user, client=parts[2], at=_time(parts[0]))


def _w3c(lines: Iterable[str]) -> Iterator[Line]:
    fields: List[str] = []
    for raw in lines:
        if raw.startswith("#"):
            if raw.lower().startswith("#fields:"):
                fields = [f.lower() for f in raw.split(":", 1)[1].split()]
            continue
        if not fields:
            continue
        values = raw.split()
        record = dict(zip(fields, values))
        if "date" in record and "time" in record:
            # W3C splits the moment in two (UTC by the standard): read as one.
            record["datetime"] = f"{record.pop('date')}T{record.pop('time')}"
        yield Columns().line(record)


def _zeek(lines: Iterable[str]) -> Iterator[Line]:
    fields: List[str] = []
    separator = "\t"
    for raw in lines:
        if raw.startswith("#"):
            if raw.startswith("#separator"):
                token = raw.split(None, 1)[1].strip() if " " in raw.strip() else "\\x09"
                try:
                    separator = bytes(token, "utf-8").decode("unicode_escape") or "\t"
                except (UnicodeDecodeError, ValueError):
                    separator = "\t"                 # an unreadable header: Zeek's default
            elif raw.startswith("#fields"):
                fields = raw.rstrip("\n").split(separator)[1:]
            continue
        if not fields:
            continue
        yield Columns().line(dict(zip(fields, raw.rstrip("\n").split(separator))))


def _csv(lines: Iterable[str], columns: Columns) -> Iterator[Line]:
    """One line at a time, so a malformed line (an unmatched quote, a huge field) is skipped
    rather than ending the run or swallowing the lines after it."""
    header: Optional[List[str]] = None
    for raw in lines:
        try:
            row = next(csv.reader([raw]), None)
        except csv.Error:
            continue
        if not row:
            continue
        if header is None:
            header = [h.strip() for h in row]
            continue
        yield columns.line(dict(zip(header, row)))


def _jsonl(lines: Iterable[str], columns: Columns) -> Iterator[Line]:
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except (ValueError, RecursionError):
            continue
        if isinstance(record, dict):
            yield columns.line(_flatten(record))


def _flatten(record: Dict[str, Any], prefix: str = "", depth: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and depth < 8 and len(out) < 500:
            out.update(_flatten(value, name + ".", depth + 1))
        else:
            out[name] = value
    return out


def sniff(first: str) -> str:
    first = first.lstrip("﻿")
    if first.startswith("#separator") or first.startswith("#fields"):
        return "zeek"
    if first.startswith("#"):
        return "w3c"
    if first.lstrip().startswith("{"):
        return "jsonl"
    parts = first.split()
    if parts and re.fullmatch(r"\d{9,11}(\.\d+)?", parts[0]) and len(parts) >= 7:
        return "squid"
    return "csv"


def open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace", newline="")


def read_lines(path: Path, fmt: str = "auto", columns: Optional[Columns] = None) -> Iterator[Line]:
    columns = columns or Columns()
    with open_text(path) as handle:
        first = handle.readline()
        if fmt == "auto":
            fmt = sniff(first)
        lines = _chain(first, handle)
        if fmt == "squid":
            yield from _squid(lines)
        elif fmt == "w3c":
            yield from _w3c(lines)
        elif fmt == "zeek":
            yield from _zeek(lines)
        elif fmt == "jsonl":
            yield from _jsonl(lines, columns)
        else:
            yield from _csv(lines, columns)


def _chain(first: str, rest: Iterable[str]) -> Iterator[str]:
    if first:
        yield first
    yield from rest


# -- aggregation ------------------------------------------------------------------------

class Discovery:
    def __init__(self, *, keep_user_names: bool = False, exclude_sources: Sequence[str] = (),
                 user_key: Optional[bytes] = None):
        self.keep_user_names = keep_user_names
        self.exclude = {s.strip() for s in exclude_sources if s.strip()}
        self._key = user_key or _install_key()
        self.findings: Dict[Tuple[str, str], _Finding] = {}
        self.lines_read = 0
        self.first: Optional[datetime] = None
        self.last: Optional[datetime] = None

    def _who(self, line: Line) -> str:
        raw = line.user or line.client or "unknown"
        if self.keep_user_names and line.user:
            return line.user[:120]
        digest = hmac.new(self._key, raw.encode("utf-8", "surrogatepass"), hashlib.sha256).hexdigest()
        return "u_" + digest[:12]

    def add(self, line: Line) -> None:
        from znyx_core.ai_services import match_host
        self.lines_read += 1
        if line.at is not None:
            self.first = line.at if self.first is None or line.at < self.first else self.first
            self.last = line.at if self.last is None or line.at > self.last else self.last
        if line.client and line.client in self.exclude:
            return
        service = match_host(line.host)
        if service is None:
            return
        finding = self.findings.setdefault((service.name, line.host), _Finding(
            service.name, service.vendor, service.category, line.host))
        finding.requests += 1
        who = self._who(line)
        if who in finding.users or len(finding.users) < MAX_TRACKED_USERS:
            finding.users[who] = finding.users.get(who, 0) + 1
        else:
            finding.capped = True
        if line.at is not None:
            finding.first = line.at if finding.first is None or line.at < finding.first else finding.first
            finding.last = line.at if finding.last is None or line.at > finding.last else finding.last

    def report(self, source: str, tool_version: Optional[str] = None) -> Dict[str, Any]:
        def iso(moment: Optional[datetime]) -> Optional[str]:
            return moment.isoformat().replace("+00:00", "Z") if moment else None
        findings = sorted(self.findings.values(), key=lambda f: (-f.requests, f.host))
        return {
            "source": source[:120], "period_start": iso(self.first), "period_end": iso(self.last),
            "lines_read": self.lines_read, "users_pseudonymised": not self.keep_user_names,
            "tool_version": tool_version,
            "findings": [{
                "service": f.service, "vendor": f.vendor, "category": f.category, "host": f.host,
                "users": len(f.users), "requests": f.requests, "first_seen": iso(f.first),
                "last_seen": iso(f.last), "users_capped": f.capped,
                "top_users": [{"user": u, "requests": n} for u, n in
                              sorted(f.users.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_USERS]],
            } for f in findings[:500]],
        }


def _install_key() -> bytes:
    """The key user names are hashed with: this install's (so the same user is the same
    pseudonym in every report from here)."""
    try:
        from znyx_runtime.install_state import user_id_key
        return hmac.new(user_id_key(), b"znyx-discover", hashlib.sha256).digest()
    except Exception:  # noqa: BLE001 - no install state: a key for this run only
        import os
        return os.urandom(32)


def upload(report: Dict[str, Any], control_plane: str, org_id: str, token: str, timeout: float = 30.0) -> Dict[str, Any]:
    import httpx
    url = f"{control_plane.rstrip('/')}/v1/orgs/{org_id}/inventory/shadow-ai"
    response = httpx.post(url, json=report, headers={"Authorization": f"Bearer {token}"}, timeout=timeout)
    if response.status_code >= 300:
        raise RuntimeError(f"the console refused the report ({response.status_code}): {response.text[:300]}")
    return response.json()


def print_summary(report: Dict[str, Any], out=sys.stdout) -> None:
    findings = report["findings"]
    print(f"Read {report['lines_read']:,} log lines"
          + (f" from {report['period_start']} to {report['period_end']}" if report["period_start"] else "")
          + f": {len(findings)} AI service host(s) reached directly.", file=out)
    if not findings:
        return
    width = max(len(f["service"]) for f in findings)
    for f in findings:
        users = f"{f['users']} user{'' if f['users'] == 1 else 's'}"
        requests = f"{f['requests']:,} request{'' if f['requests'] == 1 else 's'}"
        print(f"  {f['service']:<{width}}  {f['host']:<40}  {users:>12}  {requests:>16}  ({f['category']})", file=out)

# Telemetry

The ZNYX Runtime is **opt-in and off by default**: out of the box it sends
nothing — no heartbeats, no usage pings, no phone-home of any kind. The runtime
runs inside your security perimeter; an outbound call you didn't ask for is
treated as a bug.

Opting in is a single flag:

```bash
export ZNYX_TELEMETRY=true
```

Accepted values are `true`, `1`, `yes`, or `on` (case-insensitive). Anything
else — including an empty value or a typo — leaves telemetry off.

## What is sent when enabled

With `ZNYX_TELEMETRY=true`, the runtime sends an anonymous heartbeat once at
startup and then every 24 hours while the process runs. In local mode, an
install's very first startup also sends a one-shot `first_run` event with the
same shape. Each ping is a single JSON object:

| Field | Example | Notes |
|-------|---------|-------|
| `install_id` | `"1c9e4a02-…"` | Random UUID, generated locally on first run and stored in `~/.znyx/state.json`. Not derived from hardware, network, or account data. Delete the file to rotate it. |
| `version` | `"1.0.0"` | Runtime version |
| `event_type` | `"heartbeat"` | `"heartbeat"` (periodic) or `"first_run"` (once, on an install's first local-mode startup) |
| `mode` | `"local"` | `"local"` or `"managed"` |
| `source` | `"runtime"` | Fixed string |
| `os` | `"Linux"` | Operating system name |
| `os_version` | `"6.8.0"` | OS release |
| `arch` | `"x86_64"` | CPU architecture |
| `python_version` | `"3.12.4"` | Python version |
| `detector_count` | `0` | Number of configured detectors (this version always reports `0`) |
| `eval_count` | `3481` | Evaluations served since this process started — a counter, never content |
| `run_count` | `7` | Number of runtime startups, from local state |
| `timestamp` | `"2026-07-10T12:00:00+00:00"` | When the ping was built (UTC) |

**Never sent:** PII, prompts or any request/response content, detection
results, policy contents, tenant or customer data, hostnames, usernames, or
network identifiers.

Delivery is fire-and-forget with a 5-second timeout. Telemetry failures are
logged at debug level and never affect evaluation or startup.

Heartbeats exist only in the runtime server process. Using the engine
in-process as a library never starts the heartbeat, regardless of environment
variables.

## Where it goes

Heartbeats are POSTed to:

    https://cp.znyx.ai/v1/install-telemetry

That is the only telemetry destination, and **only when opted in** — the runtime
heartbeat is off unless `ZNYX_TELEMETRY=true`, so an unconfigured install sends
nothing. Easy to verify on the wire by watching outbound traffic to `cp.znyx.ai`.

Setting `ZNYX_TELEMETRY_URL` to an empty string removes the destination entirely,
so nothing is sent even if the flag is on.

To send heartbeats to a self-hosted receiver instead, override the URL:

```bash
export ZNYX_TELEMETRY=true
export ZNYX_HEARTBEAT_URL=https://telemetry.internal.example.com/ingest
```

`ZNYX_HEARTBEAT_URL` only changes the destination; it never enables or
disables sending.

## Note for existing deployments

Earlier versions required both `ZNYX_TELEMETRY=true` and `ZNYX_HEARTBEAT_URL`
to send anything — the flag alone was a silent no-op. The URL now defaults to
the endpoint above, so `ZNYX_TELEMETRY=true` by itself starts sending. If you
had the flag set without a URL and relied on it doing nothing, unset it.

## Separate channel: per-evaluation telemetry (managed mode)

Distinct from the anonymous heartbeat, the runtime can stream metadata-only
evaluation events to a control plane over the authenticated runtime-token
channel. That is controlled by `ZNYX_TELEMETRY_ENABLED` (on by default only in
managed mode, where you have explicitly configured a control plane URL and
token; off in local mode), goes to the control plane *you* configured, and is
unaffected by `ZNYX_TELEMETRY`. This document covers only the anonymous
install heartbeat.

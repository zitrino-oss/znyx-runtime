"""Benchmark a model-backed detector against a labelled dataset and (optionally) stamp an
enforcement-tier scorecard into a YAML policy — the console-less path to ML enforcement.

    znyx-runtime benchmark --bundle policy.yaml --detector toxicity --dataset data.jsonl

A model-backed detector's BLOCK/REDACT is pinned to advisory WARN by the runtime until its
policy carries a passing ``_scorecard_gate`` (see ``orchestrator._apply_scorecard_gate``).
Without this tool a self-hosted deployment could be held at WARN by that gate with no way to
satisfy it — the gate, the metrics and the signature check all ship here, so the tool that
drives them belongs here too rather than beside the control plane.

It runs the SAME evaluation pipeline the runtime uses, computes the SAME metrics the hosted
console computes (``znyx_core.engine.eval_metrics``), evaluates the SAME gate
(``znyx_core.engine.scorecard_gate``), and writes the verdict back into your YAML. No
database, no control plane.

Workflow (per detector + pinned model):
    1. Serve the model. Point the detector's ``backends.local_ml.endpoint_url`` at your
       inference sidecar and make sure the model is pinned there (``POST /v1/models/desired``,
       or ``ZNYX_INFERENCE_TASKS`` at startup).
    2. Benchmark and read the verdict (writes nothing):
           znyx-runtime benchmark --bundle policy.yaml --detector toxicity \
               --dataset toxicity_eval.jsonl
    3. If it passes and you accept the result, stamp it:
           znyx-runtime benchmark --bundle policy.yaml --detector toxicity \
               --dataset toxicity_eval.jsonl --stamp

Dataset format — JSONL (one object/line), a JSON array, or CSV with a header row. Fields:
    text | input_text | prompt           the content to evaluate                   (required)
    expected_decision | expected | label ALLOW / WARN / BLOCK / REDACT / TRANSFORM (required)
    language | lang                      BCP-47-ish language tag                   (optional)
    output_text                          for --stage output                        (optional)

The gate needs at least 100 samples PER LANGUAGE, and a set of only positives cannot produce
AUROC at all (one class). Include realistic negatives — especially hard ones that share
surface features with the positives — or ``fp_rate`` reads 0 by construction and certifies
nothing.

SECURITY NOTE: ``_scorecard_gate`` is not covered by the bundle signature, so by default the
runtime TRUSTS whatever the YAML says and ``--stamp`` asserts "I ran this benchmark and accept
the result" — treat the file as you would any enforcement config.

Pass ``--signing-key <ed25519.pem>`` to sign the stamp instead. A runtime started with
ZNYX_SCORECARD_PUBLIC_KEY then verifies it and ignores any stamp whose signature is missing or
invalid, failing closed to advisory WARN — so a hand-edited ``enforcement_passed: true`` no
longer grants enforcement. Unsigned stamps still work on runtimes with no key configured.

``--stamp`` writes the gate verdict this run MEASURED; it cannot be set by hand through this
tool, and it is refused outright when the model layer never fired.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from znyx_core.engine import eval_metrics
from znyx_core.engine.scorecard_gate import (
    ADVISORY,
    ENFORCEMENT,
    evaluate_gate,
    is_model_backed,
    model_versions_for,
)

# Decisions the runtime can emit; what a dataset's expected_decision may be.
_VALID_DECISIONS = {"ALLOW", "WARN", "BLOCK", "REDACT", "TRANSFORM"}
# Reserved (non-detector) policy keys passed through for egress-redaction parity.
_PASSTHROUGH = ("runtime_policy",)
_REDACTOR_KEYS = ("pii", "secrets")


# --------------------------------------------------------------------------- policy locate

def _load_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _base_policy(raw: Any, scope: Optional[str]) -> Dict[str, Any]:
    """The flat detector-keyed policy dict to evaluate against, from the raw YAML.

    Supports the three shapes the runtime/loader accept: a hierarchical file with a
    ``default:`` policy, the pack-authoring ``detectors:`` list, or an already-flat detector
    dict. ``--scope`` (dotted path, e.g. ``tenants.acme.apps.bot``) selects a sub-tree first.
    """
    node = raw
    if scope:
        for part in scope.split("."):
            if not isinstance(node, dict) or part not in node:
                raise SystemExit(f"error: --scope '{scope}' not found (missing '{part}').")
            node = node[part]
    if isinstance(node, dict) and isinstance(node.get("default"), dict):
        return node["default"]
    if isinstance(node, dict) and isinstance(node.get("detectors"), list):
        flat: Dict[str, Any] = {}
        for item in node["detectors"]:
            if isinstance(item, dict) and item.get("name"):
                flat[item["name"]] = {"enabled": item.get("enabled", True),
                                      **(item.get("config") or {})}
        return flat
    if isinstance(node, dict):
        return node
    raise SystemExit("error: could not interpret the bundle as a policy (expected a mapping).")


def _stamp_ref(raw: Any, detector: str, scope: Optional[str]) -> Optional[Dict[str, Any]]:
    """The mutable dict INSIDE raw where ``_scorecard_gate`` must be written so the runtime's
    resolved config for ``detector`` picks it up. None if the detector isn't found."""
    node = raw
    if scope:
        for part in scope.split("."):
            node = node[part]
    if isinstance(node, dict) and isinstance(node.get("default"), dict):
        d = node["default"]
        return d.get(detector) if isinstance(d.get(detector), dict) else None
    if isinstance(node, dict) and isinstance(node.get("detectors"), list):
        for item in node["detectors"]:
            if isinstance(item, dict) and item.get("name") == detector:
                return item.setdefault("config", {})
        return None
    if isinstance(node, dict) and isinstance(node.get(detector), dict):
        return node[detector]
    return None


# --------------------------------------------------------------------------- dataset

def _load_dataset(path: str, default_language: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: dataset not found: {path}")
    rows: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    elif p.suffix.lower() == ".jsonl":
        with open(p, encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
    else:  # .json (array) or fall back to jsonl-of-one
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else [data]

    samples: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        text = r.get("text") or r.get("input_text") or r.get("prompt")
        expected = (r.get("expected_decision") or r.get("expected") or r.get("label") or "")
        expected = str(expected).strip().upper()
        if not text or expected not in _VALID_DECISIONS:
            raise SystemExit(
                f"error: dataset row {i} needs a non-empty text and an expected_decision in "
                f"{sorted(_VALID_DECISIONS)} (got text={bool(text)}, expected={expected!r}).")
        samples.append({
            "text": text,
            "output_text": r.get("output_text"),
            "expected": expected,
            "language": (r.get("language") or r.get("lang") or default_language),
        })
    if not samples:
        raise SystemExit("error: dataset is empty.")
    return samples


# --------------------------------------------------------------------------- run

def _build_eval_policy(base: Dict[str, Any], detector: str,
                       eval_entry: Dict[str, Any]) -> Dict[str, Any]:
    """A policy that runs ONLY the target detector (so the aggregate decision is its own),
    while still supplying runtime_policy + pii/secrets configs (disabled, never executed) so
    the detector's egress redaction matches production."""
    policy: Dict[str, Any] = {detector: {**eval_entry, "enabled": True}}
    for k in _PASSTHROUGH:
        if isinstance(base.get(k), dict):
            policy[k] = base[k]
    for k in _REDACTOR_KEYS:
        if k != detector and isinstance(base.get(k), dict):
            policy[k] = {**base[k], "enabled": False}  # config available, detector not run
    return policy


async def _run(samples: List[Dict[str, Any]], detector: str, stage: str,
               eval_policy: Dict[str, Any]) -> Dict[str, Any]:
    from znyx_core.core.models import EvaluationRequest
    from znyx_core.engine.evaluator import GuardrailsEvaluator

    evaluator = GuardrailsEvaluator()
    confusion: Dict[str, Dict[str, int]] = {}
    scores: List[float] = []
    labels: List[int] = []
    latencies: List[int] = []
    per_language: Dict[str, int] = {}
    escalated = 0  # samples where the model layer actually ran (not a deterministic fallback)
    errors = 0

    for idx, s in enumerate(samples):
        text = s["output_text"] if (stage == "output" and s.get("output_text")) else s["text"]
        req = EvaluationRequest(
            request_id=f"scorecard-{idx}", tenant_id="scorecard-cli",
            app_id="scorecard", text=text,
        )
        try:
            # certify_model=True: this run EXISTS to measure the model and decide whether it may
            # enforce, so the scorecard gate must not downgrade its BLOCK/REDACT to WARN before
            # the decision is recorded — that would log a verdict the model never produced.
            # Scores are unaffected either way (eval_metrics counts WARN as flagged, same as
            # BLOCK); only the recorded decision changes.
            resp = await evaluator.evaluate(req, context=stage, policy=eval_policy,
                                            certify_model=True)
            actual = resp.decision.value if resp.decision else "NONE"
            risk = resp.risk_score or 0
            for d in (resp.detector_results or []):
                if getattr(d, "detector_name", None) == detector:
                    mode = getattr(d, "execution_mode", None)
                    if mode and mode != "local_deterministic":
                        escalated += 1
                    break
        except Exception as exc:  # noqa: BLE001 — a failed sample is an ERROR result, not a crash
            actual, risk = "ERROR", 0
            errors += 1
            print(f"  ! sample {idx} errored: {exc}", file=sys.stderr)

        expected = s["expected"]
        confusion.setdefault(expected, {})
        confusion[expected][actual] = confusion[expected].get(actual, 0) + 1
        scores.append(risk)
        labels.append(0 if expected in eval_metrics.ALLOW_DECISIONS else 1)
        latencies.append(getattr(resp, "latency_ms", 0) if actual != "ERROR" else 0)
        per_language[s["language"]] = per_language.get(s["language"], 0) + 1

    return {
        "confusion": confusion, "scores": scores, "labels": labels,
        "latencies": latencies, "per_language": per_language,
        "escalated": escalated, "errors": errors, "n": len(samples),
    }


def _build_scorecard(run: Dict[str, Any], model_version: str, task: Optional[str],
                     category: Optional[str], threshold: Optional[float]) -> Dict[str, Any]:
    cls = eval_metrics.classification_metrics(run["confusion"])
    probs01 = [min(max(s, 0), 100) / 100.0 for s in run["scores"]]
    pct = eval_metrics.percentiles(run["latencies"])
    return {
        # gate-consumed metrics (keys must match scorecard_gate.evaluate_gate)
        "f1": cls["f1"],
        "auroc": eval_metrics.roc_auc(run["scores"], run["labels"]),
        "ece": eval_metrics.expected_calibration_error(probs01, run["labels"]),
        "fp_rate": cls["fp_rate"],
        "p95_latency_ms": pct["p95"],
        "per_language": {lang: {"samples": n} for lang, n in run["per_language"].items()},
        "validated_at": datetime.now(timezone.utc).isoformat(),
        # provenance / context (informational; not gated)
        "precision": cls["precision"],
        "recall": cls["recall"],
        "auprc": eval_metrics.pr_auc(run["scores"], run["labels"]),
        "binary_confusion": cls["binary_confusion"],
        "sample_count": run["n"],
        "model_version": model_version,
        "task": task,
        "category": category,
        "threshold": threshold,
    }


# --------------------------------------------------------------------------- report

def _fmt(v: Any) -> str:
    return "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _print_report(detector: str, model_version: str, scorecard: Dict[str, Any],
                  run: Dict[str, Any], advisory, enforcement) -> None:
    print(f"\nScorecard — detector '{detector}', model {model_version}")
    print(f"  samples: {run['n']}   errors: {run['errors']}   "
          f"model-layer fired: {run['escalated']}/{run['n']}")
    print("  metrics:")
    for k in ("precision", "recall", "f1", "fp_rate", "auroc", "auprc", "ece"):
        print(f"    {k:<10} {_fmt(scorecard.get(k))}")
    print(f"    {'p95_ms':<10} {_fmt(scorecard.get('p95_latency_ms'))}")
    cm = scorecard.get("binary_confusion", {})
    print(f"    confusion  tp={cm.get('tp')} fp={cm.get('fp')} "
          f"fn={cm.get('fn')} tn={cm.get('tn')}")
    print("  per-language samples: " +
          ", ".join(f"{k}={v['samples']}" for k, v in scorecard["per_language"].items()))

    for name, res in ((ADVISORY, advisory), (ENFORCEMENT, enforcement)):
        verdict = "PASS" if res.passed else "FAIL"
        print(f"\n  {name.upper()} gate: {verdict}")
        for fail in res.failures:
            print(f"    - {fail['metric']}: need {fail['op']} {fail['required']}, "
                  f"got {_fmt(fail['actual'])}")

    if run["escalated"] == 0:
        print("\n  WARNING: the model layer never fired (every sample fell back to the "
              "deterministic\n  base). Is the inference sidecar running and the backend "
              "endpoint_url reachable?\n  This scorecard reflects DETERMINISTIC behaviour, "
              "not the pinned model.")


# --------------------------------------------------------------------------- argparse

def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach this tool's flags. Shared so ``znyx-runtime benchmark`` and a direct
    ``python -m znyx_runtime.benchmark`` expose exactly the same interface."""
    ap.add_argument("--bundle", required=True, help="Path to the YAML policy/bundle file.")
    ap.add_argument("--detector", required=True, help="Detector key to certify (e.g. toxicity).")
    ap.add_argument("--dataset", required=True, help="Labelled dataset (.jsonl / .json / .csv).")
    ap.add_argument("--stage", default="input",
                    help="Evaluation stage (input/output/...; default: input).")
    ap.add_argument("--category", help="Vertical for stricter gates (healthcare/legal/finance).")
    ap.add_argument("--scope", help="Dotted path to the policy sub-tree (e.g. tenants.acme).")
    ap.add_argument("--default-language", default="en",
                    help="Language bucket for samples without a language field (default: en).")
    ap.add_argument("--stamp", action="store_true",
                    help="Write the enforcement-gate verdict (_scorecard_gate) back into --bundle.")
    ap.add_argument("--signing-key",
                    help="Path to an Ed25519 private-key PEM. When given, the stamp is signed "
                         "so a runtime configured with ZNYX_SCORECARD_PUBLIC_KEY can verify it "
                         "(tamper-evident). Without it the stamp is unsigned (trusted as today).")
    ap.add_argument("--out", help="Also write the full scorecard JSON to this path.")
    return ap


# --------------------------------------------------------------------------- run

def run_benchmark(args: argparse.Namespace) -> int:
    """Execute a parsed benchmark invocation. Returns the process exit code."""
    raw = _load_yaml(args.bundle)
    base = _base_policy(raw, args.scope)
    entry = base.get(args.detector)
    if not isinstance(entry, dict):
        raise SystemExit(f"error: detector '{args.detector}' not found in the bundle policy.")
    if not is_model_backed(entry):
        raise SystemExit(
            f"error: detector '{args.detector}' is not model-backed (no strategy with a model "
            "mode). The scorecard gate only applies to model-backed detectors; a purely "
            "deterministic detector enforces without one.")

    versions = model_versions_for(entry)
    model_version = ", ".join(versions) if versions else "default"
    task = None
    threshold = None
    backends = entry.get("backends") or {}
    for mode in (entry.get("strategy") or {}).get("order") or []:
        b = backends.get(mode) if isinstance(backends, dict) else None
        if isinstance(b, dict):
            task = task or b.get("task")
            threshold = threshold if threshold is not None else b.get("threshold")

    samples = _load_dataset(args.dataset, args.default_language)
    print(f"Evaluating '{args.detector}' (model {model_version}) on {len(samples)} samples "
          f"[stage={args.stage}] ...")
    eval_policy = _build_eval_policy(base, args.detector, entry)
    run = asyncio.run(_run(samples, args.detector, args.stage, eval_policy))

    scorecard = _build_scorecard(run, model_version, task, args.category, threshold)
    advisory = evaluate_gate(scorecard, tier=ADVISORY, category=args.category)
    enforcement = evaluate_gate(scorecard, tier=ENFORCEMENT, category=args.category)
    _print_report(args.detector, model_version, scorecard, run, advisory, enforcement)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"detector": args.detector, "scorecard": scorecard,
             "advisory": advisory.to_dict(), "enforcement": enforcement.to_dict()}, indent=2),
            encoding="utf-8")
        print(f"\nWrote scorecard JSON → {args.out}")

    if args.stamp:
        # Refuse to write a stamp for a run the MODEL sat out. An unreachable endpoint (wrong
        # port, sidecar down, a band no sample entered) makes every sample fall back to the
        # deterministic layer, and the run still completes and prints a full set of metrics —
        # metrics describing the deterministic rules, not the pinned model. Stamping that grants
        # the model permission to BLOCK on evidence it never produced, and the stamp records only
        # "passed", never which layer earned it, so nothing downstream can tell.
        #
        # Refused outright rather than warned (the warning above is easy to scroll past) and with
        # no override flag: a purely deterministic detector is not gated at all, so it never needs
        # a stamp — which leaves no legitimate case for stamping an unescalated run.
        if run["escalated"] == 0:
            raise SystemExit(
                f"error: --stamp refused: the model layer never fired "
                f"(0/{run['n']} samples). This scorecard describes the deterministic "
                f"layer, not {model_version}. Check that the sidecar is running and that "
                f"backends.local_ml.endpoint_url is reachable, then re-run.")
        if run["escalated"] < run["n"]:
            print(f"\n  NOTE: the model layer fired on {run['escalated']}/{run['n']} samples; "
                  "the rest fell back to the deterministic layer and are scored as such.")

        ref = _stamp_ref(raw, args.detector, args.scope)
        if ref is None:
            raise SystemExit("error: --stamp could not locate the detector config to write into "
                             "(try --scope, or check the bundle shape).")
        if args.signing_key:
            from znyx_core.engine.scorecard_stamp import sign_stamp
            key_pem = Path(args.signing_key).read_text(encoding="utf-8")
            ref["_scorecard_gate"] = sign_stamp(
                args.detector, enforcement_passed=enforcement.passed,
                model_version=model_version, validated_at=scorecard["validated_at"],
                private_key_pem=key_pem)
        else:
            ref["_scorecard_gate"] = {"enforcement_passed": enforcement.passed}
        # Additive provenance the runtime ignores but an auditor can read.
        ref["_scorecard"] = {
            "model_version": model_version, "validated_at": scorecard["validated_at"],
            "sample_count": run["n"], "dataset": Path(args.dataset).name,
            "f1": scorecard["f1"], "auroc": scorecard["auroc"], "ece": scorecard["ece"],
            "fp_rate": scorecard["fp_rate"], "p95_latency_ms": scorecard["p95_latency_ms"],
            "category": args.category,
        }
        with open(args.bundle, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False, default_flow_style=False)
        state = "ENFORCING" if enforcement.passed else "advisory (WARN) — enforcement gate not met"
        print(f"\nStamped _scorecard_gate into {args.bundle}: '{args.detector}' is now {state}.")
        if not enforcement.passed:
            print("  (BLOCK/REDACT stays pinned to WARN until the enforcement gate passes.)")

    # Exit non-zero if enforcement was requested-but-not-met, so CI can gate on it.
    return 0 if (enforcement.passed or not args.stamp) else 3


def main(argv=None) -> int:
    ap = add_arguments(argparse.ArgumentParser(
        prog="znyx-runtime benchmark",
        description="Benchmark a model-backed detector and stamp an enforcement scorecard "
                    "into a YAML policy (console-less ML enforcement)."))
    return run_benchmark(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

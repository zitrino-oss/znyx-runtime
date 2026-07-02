"""Pure, dependency-free evaluation metrics (shared by the control-plane benchmark
service and the console-less scorecard CLI).

No numpy / sklearn / sqlalchemy — stdlib only — so the SAME metric definitions are
usable in the lightweight runtime/CLI and the DB control plane. This is the single
source of truth: a scorecard produced offline by `scripts/scorecard_benchmark.py`
must clear `scorecard_gate.evaluate_gate` with the exact numbers the console would
compute, so enforcement is consistent whichever path stamped the gate.

Two families:
  * Probability-based (need per-sample score 0..1 + binary label): AUROC, AUPRC, ECE,
    score histogram, PSI drift. Most meaningful for model-backed (calibrated) runs;
    for all-deterministic runs they're honest but degenerate (heavy ties).
  * Decision-confusion-based: precision/recall/F1/FP-rate from an expected→actual
    decision confusion matrix (positive class = "flagged", i.e. decision != ALLOW),
    plus latency percentiles.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

# Decisions that count as "allowed through" — anything else is a guardrail flag.
ALLOW_DECISIONS = frozenset({"ALLOW"})
# Non-decisions: never counted as a positive prediction.
NON_DECISIONS = frozenset({"ERROR", "NONE"})


def _average_ranks(values: Sequence[float]) -> List[float]:
    """1-based ranks with ties averaged (used by the rank-based AUROC)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Area under the ROC curve via the Mann–Whitney U statistic (tie-aware).
    None when only one class is present (AUROC is undefined)."""
    n = len(scores)
    pos = sum(1 for lab in labels if lab)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None
    ranks = _average_ranks(scores)
    sum_pos = sum(r for r, lab in zip(ranks, labels) if lab)
    auc = (sum_pos - pos * (pos + 1) / 2) / (pos * neg)
    return round(auc, 4)


def pr_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Average precision (area under the precision-recall curve). None with no positives."""
    pos = sum(1 for lab in labels if lab)
    if pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    for i in order:
        if labels[i]:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return round(ap, 4)


def expected_calibration_error(probs: Sequence[float], labels: Sequence[int],
                               bins: int = 10) -> Optional[float]:
    """Expected Calibration Error: binned |confidence − accuracy|, weighted by bin size.
    Confidence is the predicted-class probability (max(p, 1−p)); prediction is positive
    when p ≥ 0.5. None for empty input."""
    n = len(probs)
    if n == 0:
        return None
    confs: List[float] = []
    correct: List[int] = []
    for p, lab in zip(probs, labels):
        pred = 1 if p >= 0.5 else 0
        confs.append(p if pred == 1 else 1 - p)
        correct.append(1 if pred == lab else 0)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [i for i in range(n) if (confs[i] > lo or b == 0) and confs[i] <= hi]
        if not members:
            continue
        avg_conf = sum(confs[i] for i in members) / len(members)
        acc = sum(correct[i] for i in members) / len(members)
        ece += (len(members) / n) * abs(avg_conf - acc)
    return round(ece, 4)


def reliability_curve(probs: Sequence[float], labels: Sequence[int],
                      bins: int = 10) -> Optional[List[Dict[str, Any]]]:
    """Per-bin calibration breakdown — the data behind a reliability diagram (and the same
    bins ECE summarizes). For each confidence bin: the average predicted-class confidence vs
    the observed accuracy and the sample count. A well-calibrated model tracks the diagonal
    (avg_confidence ~= accuracy in every bin). Confidence is the predicted-class probability
    (max(p, 1-p)); prediction is positive when p >= 0.5. None for empty input; empty bins are
    included with count 0 and null avg/accuracy so the diagram can show gaps honestly."""
    n = len(probs)
    if n == 0:
        return None
    confs: List[float] = []
    correct: List[int] = []
    for p, lab in zip(probs, labels):
        pred = 1 if p >= 0.5 else 0
        confs.append(p if pred == 1 else 1 - p)
        correct.append(1 if pred == lab else 0)
    curve: List[Dict[str, Any]] = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [i for i in range(n) if (confs[i] > lo or b == 0) and confs[i] <= hi]
        if members:
            avg_conf = sum(confs[i] for i in members) / len(members)
            acc = sum(correct[i] for i in members) / len(members)
            curve.append({"lo": round(lo, 3), "hi": round(hi, 3),
                          "avg_confidence": round(avg_conf, 4), "accuracy": round(acc, 4),
                          "count": len(members)})
        else:
            curve.append({"lo": round(lo, 3), "hi": round(hi, 3),
                          "avg_confidence": None, "accuracy": None, "count": 0})
    return curve


def score_histogram(scores_0_100: Sequence[float], bins: int = 10) -> List[int]:
    """Counts of 0..100 risk scores per equal-width bin (default 10 → 0-10, 10-20, …).
    The distribution a later version-to-version drift (PSI) compares against."""
    counts = [0] * bins
    width = 100.0 / bins
    for s in scores_0_100:
        s = max(0.0, min(100.0, float(s)))
        idx = min(int(s / width), bins - 1)
        counts[idx] += 1
    return counts


def population_stability_index(expected: Sequence[int], actual: Sequence[int]) -> Optional[float]:
    """PSI between two same-length histograms (e.g. score_histogram of two runs). A standard
    distribution-shift measure: <0.1 no shift, 0.1–0.25 moderate, >0.25 significant. None on
    mismatched/empty inputs. Uses a small epsilon to avoid divide-by-zero on empty bins."""
    if not expected or len(expected) != len(actual):
        return None
    e_tot, a_tot = sum(expected), sum(actual)
    if e_tot == 0 or a_tot == 0:
        return None
    eps = 1e-6
    psi = 0.0
    for e, a in zip(expected, actual):
        e_frac = max(e / e_tot, eps)
        a_frac = max(a / a_tot, eps)
        psi += (a_frac - e_frac) * math.log(a_frac / e_frac)
    return round(psi, 4)


def classification_metrics(confusion: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """Binary precision/recall/F1 + FP-rate from an expected→actual decision confusion
    matrix. Positive class = "flagged" (expected/actual not in ALLOW_DECISIONS). Pure label
    arithmetic — no probabilities. A non-decision actual (ERROR/NONE) never counts as a
    positive prediction. Mirrors the control-plane benchmark definition exactly."""
    tp = fp = fn = tn = 0
    for expected, actuals in confusion.items():
        exp_pos = expected not in ALLOW_DECISIONS
        for actual, cnt in actuals.items():
            act_pos = actual not in ALLOW_DECISIONS and actual not in NON_DECISIONS
            if exp_pos and act_pos:
                tp += cnt
            elif exp_pos and not act_pos:
                fn += cnt
            elif not exp_pos and act_pos:
                fp += cnt
            else:
                tn += cnt
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None
    fp_rate = fp / (fp + tn) if (fp + tn) else None
    r = lambda x: round(x, 4) if x is not None else None  # noqa: E731
    return {
        "precision": r(precision),
        "recall": r(recall),
        "f1": r(f1),
        "fp_rate": r(fp_rate),
        "binary_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def percentiles(values: List[int]) -> Dict[str, float]:
    """p50/p95/p99 via linear interpolation on the sorted sample (stdlib-free, robust for
    small n). Empty input → zeros."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(values)

    def pct(p: float) -> float:
        if len(s) == 1:
            return float(s[0])
        k = (len(s) - 1) * p
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return round(s[f] + (s[c] - s[f]) * (k - f), 1)

    return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)}

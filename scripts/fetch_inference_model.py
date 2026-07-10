"""Operator-run model fetch-and-pin CLI for the ZNYX inference sidecar.

This is the ONE deliberate, network-touching step in the model supply chain. It
exports (or snapshots) a vetted checkpoint into the local artifact dir and prints
the sha256 you then PIN via env/compose. The serving path never downloads anything;
it loads only the local, sha256-verified artifact.

Usage (run from the repo root):

    # List the vetted candidates (optionally for one task / open-license only):
    python -m scripts.fetch_inference_model --list
    python -m scripts.fetch_inference_model --list --task safety --open-only

    # Fetch the catalog default for a task and print the sha256 to pin:
    python -m scripts.fetch_inference_model --task prompt_injection

    # Fetch a specific model on the task's vetted shortlist:
    python -m scripts.fetch_inference_model --task safety --model-id allenai/wildguard

Requires the offline export stack:  pip install 'znyx-inference[export]'
(For generative guard_llm models, raw weights are snapshotted instead of exported.)
"""
from __future__ import annotations

import argparse
import logging
import sys

from znyx_inference.runners._fetch import (
    fetch_model,
    list_candidates,
    resolve_fetch_target,
)


def _cmd_list(args: argparse.Namespace) -> int:
    rows = list_candidates(task=args.task, open_only=args.open_only)
    if not rows:
        print("No candidates match the given filters.", file=sys.stderr)
        return 1
    for r in rows:
        print(
            f"{r.get('task', '?'):<16} {r.get('model_id', '?'):<55} "
            f"{r.get('license', '?')}"
        )
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    target = resolve_fetch_target(
        args.task,
        model_id=args.model_id,
        revision=args.revision,
        allow_unvetted=args.allow_unvetted,
    )
    print(
        f"Fetching task='{target['task']}' model='{target['model_id']}'"
        f"@{target['revision']} (runner={target['runner']})\n"
        f"  -> {target['dest_dir']}",
        file=sys.stderr,
    )
    digest = fetch_model(
        target["model_id"],
        target["revision"],
        target["dest_dir"],
        runner=target["runner"],
    )
    print(
        "\nDone. Pin these values for the task (env / compose):\n"
        f"  model_id: {target['model_id']}\n"
        f"  revision: {target['revision']}\n"
        f"  sha256:   {digest}\n"
        f"  dir:      {target['dest_dir']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.fetch_inference_model",
        description="Fetch and pin a vetted inference model artifact.",
    )
    parser.add_argument("--task", help="Inference task (e.g. prompt_injection, safety).")
    parser.add_argument(
        "--model-id",
        help="Override the default model. Must be on the task's vetted shortlist "
        "unless --allow-unvetted is set.",
    )
    parser.add_argument("--revision", help="Model revision/commit to pin (default: catalog).")
    parser.add_argument(
        "--allow-unvetted",
        action="store_true",
        help="Permit an off-shortlist model_id (bring-your-own-model escape hatch).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List vetted candidates instead of fetching.",
    )
    parser.add_argument(
        "--open-only",
        action="store_true",
        help="With --list, show only OSI-open-licensed models.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list:
        return _cmd_list(args)
    if not args.task:
        parser.error("--task is required (or pass --list to browse candidates).")
    try:
        return _cmd_fetch(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

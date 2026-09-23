"""Console entry point for the Znyx runtime.

Installed as the ``znyx-runtime`` command (see pyproject ``[project.scripts]``).
Gives a Docker-free way to start the service locally:

    znyx-runtime serve --port 8080

and to certify a model-backed detector so the scorecard gate lets it enforce:

    znyx-runtime benchmark --bundle policy.yaml --detector toxicity \\
        --dataset labelled.jsonl --stamp
"""
import argparse
import os
import sys


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="znyx-runtime",
        description="Znyx guardrails runtime: evaluate LLM traffic against policies.",
    )
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Run the runtime HTTP service.")
    serve.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    serve.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev).")

    # Flags are declared by the module that implements them, so this subcommand and
    # `python -m znyx_runtime.benchmark` can never drift apart.
    from znyx_runtime.benchmark import add_arguments as _benchmark_args
    _benchmark_args(sub.add_parser(
        "benchmark",
        help="Benchmark a model-backed detector against a labelled dataset and stamp its "
             "enforcement scorecard.",
        description="Runs the same evaluation pipeline, metrics and gate the runtime uses, then "
                    "optionally writes the measured verdict into the policy YAML so the detector "
                    "may enforce. Without this, a model-backed detector stays at advisory WARN.",
    ))

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "znyx_runtime.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    elif args.command == "benchmark":
        from znyx_runtime.benchmark import run_benchmark

        # argparse subcommands share one namespace, so hand the parsed args straight over.
        sys.exit(run_benchmark(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

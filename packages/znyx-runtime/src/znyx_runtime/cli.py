"""Console entry point for the Znyx runtime.

Installed as the ``znyx-runtime`` command (see pyproject ``[project.scripts]``).
Gives a Docker-free way to start the service locally:

    znyx-runtime serve --port 8080
"""
import argparse
import os


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

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "znyx_runtime.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

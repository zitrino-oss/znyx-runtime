"""``python -m znyx``, and the entry point of the standalone binary."""
import multiprocessing

if __name__ == "__main__":
    # The supervisor starts worker processes with the spawn method; a frozen binary
    # must hand those re-executions to multiprocessing before doing anything else.
    multiprocessing.freeze_support()
    from znyx.cli import main

    main()

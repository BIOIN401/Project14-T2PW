"""CLI for the unattended overnight batch run.

Two roles in one file, on purpose. Invoked normally it is the *parent*: it plans
the night and walks every paper+mode pair. Invoked with ``--single`` it is the
*child* that the parent re-invokes for exactly one pair -- which is why the child
command is ``[sys.executable, __file__, "--single", ...]`` and why there is no
second entry point to keep in sync. See ``t2pw.batch.runner`` for why a child
process (rather than a thread or SIGALRM) is the only way to kill a hung run on
Windows.

    python scripts/batch_run.py
    python scripts/batch_run.py --topics topics.txt --limit 10 --modes strict,research \
                               --out runs --timeout 2700 --deadline 8
    python scripts/batch_run.py --status
    python scripts/batch_run.py --fresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from t2pw.batch import runner  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch_run",
        description=(
            "Run every fetched paper through the real Streamlit app twice (Strict PWML "
            "and Research relaxed), one paper+mode per child process, writing artifacts "
            "and a ranked fix-list into runs/TIMESTAMP/."
        ),
        epilog=(
            "Resuming is automatic: an incomplete runs/TIMESTAMP/ is continued and its "
            "finished paper+mode pairs are skipped. Pass --fresh to start a new one."
        ),
    )
    parser.add_argument(
        "--topics",
        default=None,
        metavar="PATH",
        help="topics file (default: topics.txt in the project root)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="cap the total number of papers fetched across all topics",
    )
    parser.add_argument(
        "--modes",
        default=None,
        metavar="LIST",
        help="comma-separated: strict, research (default: both)",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="where run directories live (default: runs/, relative to the project root)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=runner.DEFAULT_PAPER_TIMEOUT,
        metavar="SECONDS",
        help=(
            "wall-clock ceiling for ONE paper+mode; the child process tree is killed "
            f"past it and the pair is recorded as a timeout (default: {runner.DEFAULT_PAPER_TIMEOUT:.0f})"
        ),
    )
    parser.add_argument(
        "--deadline",
        type=float,
        default=runner.DEFAULT_DEADLINE_HOURS,
        metavar="HOURS",
        help=(
            "whole-night ceiling; once it passes, no further pair is started, the "
            "summary is written and the run stays resumable so the rest can finish "
            f"later (default: {runner.DEFAULT_DEADLINE_HOURS:.0f})"
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="print one progress line for the current/most recent run and exit",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="start a new run directory instead of continuing an incomplete one",
    )

    internal = parser.add_argument_group("internal (used by the parent, not by humans)")
    internal.add_argument(
        "--single",
        action="store_true",
        help="run exactly one paper+mode and print its manifest row as one JSON line",
    )
    internal.add_argument("--run-dir", default=None, metavar="DIR", help="the run directory to write into")
    internal.add_argument("--slug", default=None, metavar="SLUG", help="which paper (its folder name)")
    internal.add_argument("--mode", default=None, metavar="MODE", help="strict or research")
    return parser


def main(argv: list[str] | None = None) -> int:
    # FIRST, before anything can print. The child's stdout is always a pipe, and a
    # piped stream encodes with the ANSI codepage (cp1252 here), so a single Greek
    # letter in an entity name or an em dash in a title raises UnicodeEncodeError
    # -- after the artifacts and RESULT.txt are on disk. The parent then records a
    # crash for a pair that passed, and never retries it.
    runner.force_utf8_stdio()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.status:
        return runner.print_status(args.out)

    if args.single:
        if not args.run_dir or not args.slug or not args.mode:
            parser.error("--single requires --run-dir, --slug and --mode")
        row = runner.run_single(
            Path(args.run_dir),
            args.slug,
            args.mode,
            timeout=args.timeout,
        )
        # The parent parses exactly this one line off stdout.
        runner.emit_outcome(row)
        return 0 if str(row.get("status", "")).lower() == "pass" else 1

    return runner.run_batch(
        topics_path=args.topics,
        out=args.out,
        modes=args.modes,
        limit=args.limit,
        timeout=args.timeout,
        deadline_hours=args.deadline,
        fresh=args.fresh,
        script_path=Path(__file__).resolve(),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Ctrl+C is a clean exit: the manifest was flushed after every pair, so
        # rerunning the same command continues exactly where this stopped.
        print("\ninterrupted -- rerun the same command to continue where this stopped")
        raise SystemExit(1)

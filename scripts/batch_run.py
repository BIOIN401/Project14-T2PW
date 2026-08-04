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

Exit codes (also in docs/batch_runner.md section 2d, and echoed by
``run_overnight.bat`` as ERRORLEVEL):

    0  every attempted pair passed -- ALL GREEN, or PASSED WITH WARNINGS
    1  something did not pass: a failure, a reached --deadline with pairs still
       pending, a Ctrl+C, "nothing ran at all", or an unexpected error in the
       batch loop. A run directory exists; read its SUMMARY.txt.
    2  a usage error (argparse), or -- from run_overnight.bat only -- no
       virtualenv at .venv\\Scripts\\python.exe
    3  PREFLIGHT FAILED: this interpreter cannot import what the child processes
       need, so nothing was fetched, nothing was run and NO RUN DIRECTORY WAS
       CREATED. There is no SUMMARY.txt to read; the console message names the
       missing module and the interpreter to rerun with. See
       ``runner.EXIT_PREFLIGHT`` and ``runner.check_preflight``.
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
            "finished paper+mode pairs are skipped. Pass --fresh to start a new one. "
            "Exit codes: 0 everything passed (possibly with warnings), 1 something did "
            "not pass or the run is unfinished, 2 usage error, 3 preflight failed -- the "
            "interpreter cannot import what the child processes need, so nothing ran and "
            "no run directory was created."
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
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help=(
            "acquire the papers and write plan.json / skipped.json / SUMMARY.txt, then stop "
            "BEFORE any paper+mode leg. Executes zero Streamlit/LLM legs by construction: it "
            "returns before the run loop, so no manifest row and no leg directory can be "
            "created regardless of how fast the paper cache is. Use this to stage a run for "
            "inspection -- never a small --deadline, which only decides INSIDE the loop and "
            "will start a leg when acquisition comes back from cache"
        ),
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

    # PARENT PATH ONLY, and the last thing that happens before disk is touched.
    #
    # It is deliberately not run for --status (documented as always exiting 0 so
    # a second terminal can poll it safely -- an environment check has no business
    # blocking a read of yesterday's manifest) and not run for --single (the child
    # is spawned only after the parent has already vouched for the environment, and
    # a second check there would spend 0.72s per leg re-proving it while breaking
    # the one-row-per-pair contract if it ever disagreed).
    #
    # Its position here, above run_batch, is the entire point: run_batch's first
    # acts are mkdir(out_dir), new_run_dir() and a 43-second fetch, and once those
    # have happened a failed night is indistinguishable on disk from a real one.
    # See runs/2026-07-27_2135, which is still in runs/ looking like a night that
    # ran 56 papers and failed all of them.
    preflight_code = runner.run_preflight()
    if preflight_code:
        return preflight_code

    return runner.run_batch(
        topics_path=args.topics,
        out=args.out,
        modes=args.modes,
        limit=args.limit,
        timeout=args.timeout,
        deadline_hours=args.deadline,
        fresh=args.fresh,
        stage_only=args.stage_only,
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

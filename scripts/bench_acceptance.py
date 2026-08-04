"""CLI for the pinned scientific benchmark.

Three jobs, none of which touches the network or an LLM:

    # 1. check the gold set loads and is internally consistent
    python scripts/bench_acceptance.py --validate-gold

    # 2. write the pinned topics file that makes an acceptance batch reproducible
    python scripts/bench_acceptance.py --write-topics bench_topics.txt

    # 3. score a run directory against the gold set
    python scripts/bench_acceptance.py --run-dir runs/2026-07-28_2122
    python scripts/bench_acceptance.py                       # newest run under runs/
    python scripts/bench_acceptance.py --json out.json --out out.txt

Exit codes:

    0  scored successfully and acceptance priorities 1-3 all hold
    1  scored successfully but at least one of priorities 1-3 is violated
    2  usage error, or the gold set / run directory could not be read

The split between 1 and 0 is the point of the tool: priorities 1-3 are the
absolute ones (no false real identifiers, no unsupported reactions, no
referential-integrity violations), so a non-zero exit means the run produced
science that should not be trusted, regardless of how many papers exported.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from t2pw.bench.acceptance import score_run  # noqa: E402
from t2pw.bench.goldset import GoldSetError, load_gold_set  # noqa: E402
from t2pw.bench.preflight import PreflightError, verify_plan, verify_specs  # noqa: E402
from t2pw.bench.render import render_text  # noqa: E402


_RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}")


def _force_utf8_stdio() -> None:
    """Same defence ``batch_run`` takes, for the same reason.

    Entity names carry Greek letters and titles carry em dashes. A piped stdout
    encodes with the ANSI codepage on Windows, so printing the report would raise
    ``UnicodeEncodeError`` -- after the scoring is done and before anything is
    shown.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - defensive
                pass


def _newest_run(out_dir: Path) -> Path | None:
    if not out_dir.is_dir():
        return None
    candidates = [
        child
        for child in out_dir.iterdir()
        if child.is_dir() and _RUN_DIR_RE.match(child.name) and (child / "manifest.jsonl").is_file()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_acceptance",
        description=(
            "Score a batch run against the pinned scientific gold set. Deterministic and "
            "offline: it reads stored artifacts only, calls no LLM, and writes nothing into "
            "the run directory."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        metavar="DIR",
        help="run directory to score (default: the newest one under runs/)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help="where run directories live, used to find the newest (default: runs/)",
    )
    parser.add_argument(
        "--gold",
        default=None,
        metavar="PATH",
        help="gold-set JSON (default: the checked-in pinned set)",
    )
    parser.add_argument("--json", default=None, metavar="PATH", help="also write the report as JSON")
    parser.add_argument("--out", default=None, metavar="PATH", help="also write the text report to a file")
    parser.add_argument(
        "--validate-gold",
        action="store_true",
        help="load and validate the gold set, print a summary, and exit",
    )
    parser.add_argument(
        "--write-topics",
        default=None,
        metavar="PATH",
        help=(
            "write a topics file pinning exactly the gold-set papers, in the "
            "PAPERID | pathway | organism form (fetched by id, zero search calls, "
            "eligibility score bypassed, requested scope preserved), and exit"
        ),
    )
    parser.add_argument(
        "--verify-plan",
        default=None,
        metavar="PATH",
        help=(
            "check a staged run directory (or plan.json) against the gold set and exit; "
            "refuses if any gold case lost or changed its requested pathway/organism, "
            "if a search call was recorded, or if a paper was not admitted by pinned override"
        ),
    )
    parser.add_argument(
        "--verify-topics",
        default=None,
        metavar="PATH",
        help="parse a topics file and check its scope against the gold set before any fetch, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        gold = load_gold_set(args.gold)
    except GoldSetError as exc:
        print(f"gold set error: {exc}", file=sys.stderr)
        return 2

    if args.validate_gold:
        mechanistic = [c for c in gold if not c.is_negative_control]
        controls = [c for c in gold if c.is_negative_control]
        print(f"gold set {gold.version} loaded from {gold.source_path}")
        print(f"  {len(gold)} case(s): {len(mechanistic)} mechanistic, {len(controls)} negative control")
        print(f"  strict-exportable expected: {len(gold.strict_expected_ids)}")
        print("")
        for case in gold:
            tag = " [negative control]" if case.is_negative_control else ""
            print(
                f"  {case.paper_id:<14} {case.requested_pathway:<26} "
                f"rel={case.mechanistic_relevance:<12} export={case.expected_export:<17} "
                f"min_core={case.min_connected_reactions}{tag}"
            )
            print(
                f"                 anchors={len(case.expected_pathway_anchors)} "
                f"enzymes={len(case.expected_enzymes)} metabolites={len(case.expected_metabolites)} "
                f"forbidden_ids={len(case.forbidden_identifiers)} "
                f"forbidden_organisms={len(case.forbidden_organisms)}"
            )
        return 0

    if args.write_topics:
        target = Path(args.write_topics)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(gold.topics_lines()) + "\n", encoding="utf-8")
        print(f"wrote {len(gold)} scoped pinned paper id(s) to {target}")
        print("stage it (acquisition + plan only, zero legs) with:")
        print(
            f"  python scripts/batch_run.py --topics {target} --modes strict,research "
            "--stage-only --fresh"
        )
        print("then ALWAYS verify the plan before running any leg:")
        print("  python scripts/bench_acceptance.py --verify-plan runs/TIMESTAMP")
        return 0

    if args.verify_topics:
        from t2pw.batch.fetch import parse_topics

        try:
            text = Path(args.verify_topics).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"cannot read topics file: {exc}", file=sys.stderr)
            return 2
        report = verify_specs(parse_topics(text), gold)
        print(report.render())
        return 0 if report.ok else 1

    if args.verify_plan:
        try:
            report = verify_plan(args.verify_plan, gold)
        except PreflightError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(report.render())
        if not report.ok:
            print(
                "REFUSED: this plan is not a valid acceptance run. Do not execute or resume it; "
                "correct the topics file and stage a fresh directory.",
                file=sys.stderr,
            )
        return 0 if report.ok else 1

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "runs"
        found = _newest_run(out_dir)
        if found is None:
            print(f"no run directory with a manifest found under {out_dir}", file=sys.stderr)
            return 2
        run_dir = found

    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 2

    report = score_run(run_dir, gold)
    text = render_text(report)
    print(text)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ACCEPTANCE REQUIRES COMPLETENESS, and it is checked before the priorities.
    #
    # Every scientific error count is computed over the legs that ran, so a run
    # that ran fewer legs has fewer chances to be wrong. Taken to its limit, a run
    # that attempted nothing has zero of everything and clears priorities 1-3
    # perfectly. A run that attempted 19 of 20 is the same failure in miniature:
    # the twentieth leg is exactly where the missing error could be, and exiting 0
    # would report "accepted" for a benchmark that was never finished.
    #
    # The text and JSON reports are still written above -- a partial run is worth
    # reading, it is just not worth quoting -- but the process must not say
    # accepted.
    completion = report.completion()
    if not report.is_complete:
        print(
            "NOT ACCEPTED: this run is INCOMPLETE "
            f"({completion['rendered']}). Priority checks are computed only over the legs "
            "that ran, so an unfinished run cannot be accepted however clean those legs are. "
            "Reports were still written.",
            file=sys.stderr,
        )
        missing = completion["papers_with_no_attempted_leg"]
        if missing:
            print(f"  gold cases with no attempted leg: {', '.join(missing)}", file=sys.stderr)
        partial = completion["papers_with_only_one_mode_attempted"]
        if partial:
            print(f"  gold cases with only one mode attempted: {', '.join(partial)}", file=sys.stderr)
        return 1

    absolute = [entry for entry in report.priorities() if entry["rank"] <= 3]
    return 0 if all(entry["ok"] for entry in absolute) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""``python -m t2pw.mining`` -- the headless route.

Calls the same controller and artifact layer the Streamlit page calls. There is
no parallel "CLI implementation"; the difference between the two front ends is
where the arguments come from and where progress is printed.

Commands::

    run       start a mining run
    resume    continue an interrupted run from its checkpoint
    status    print a run's manifest, stop reason and counters
    report    regenerate report.md / summary.json / graph views from artifacts
    preflight check services and configuration without running anything
    list      list runs under the runs directory

Exit codes::

    0   ok
    2   bad usage / unparseable argument
    3   feature disabled
    4   preflight failed
    5   the run stopped without completing normally (cancelled, failed, limit)

A limit-stop is reported as 5 rather than 0 on purpose: a run that stopped at
its budget is a legitimate outcome, but a caller scripting around it must be able
to tell it from a run that finished its work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DISABLED = 3
EXIT_PREFLIGHT = 4
EXIT_INCOMPLETE = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m t2pw.mining",
        description=(
            "Experimental literature-driven pathway mining. Produces a research graph, "
            "NOT PWML. Disabled by default; enable with T2PW_PATHWAY_MINING_ENABLED=true "
            "or --enable-pathway-mining."
        ),
    )
    parser.add_argument(
        "--enable-pathway-mining",
        action="store_true",
        help="enable the experimental feature for this invocation only",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start a mining run")
    run.add_argument("--species", required=True, help="target species (required, explicit)")
    run.add_argument("--topic", default="", help="topic to mine, e.g. 'jasmonate biosynthesis'")
    run.add_argument("--seed-metabolite", action="append", default=[], metavar="NAME")
    run.add_argument("--seed-enzyme", action="append", default=[], metavar="NAME")
    run.add_argument("--seed-reaction", action="append", default=[], metavar="TEXT")
    run.add_argument(
        "--paper",
        action="append",
        default=[],
        metavar="REF",
        help="PMID:123, PMC123, a DOI, or a supported URL (repeatable)",
    )
    run.add_argument(
        "--paper-file",
        action="append",
        default=[],
        metavar="PATH",
        help="a local PDF or text file to ingest (repeatable)",
    )
    run.add_argument(
        "--starting-graph",
        default="",
        metavar="PATH",
        help="an existing pathway/research graph JSON to start from (read-only)",
    )
    run.add_argument("--config", default="", metavar="PATH", help="JSON config file of overrides")
    run.add_argument("--run-id", default="", help="explicit run id (default: timestamped)")
    run.add_argument("--runs-dir", default="", help="where run directories are written")
    run.add_argument("--model", default="", help="model slug (default deepseek/deepseek-v4-flash)")
    run.add_argument("--offline", action="store_true", help="never contact a literature service")
    run.add_argument("--no-screen", action="store_true", help="skip the eligibility screen")
    run.add_argument("--vector-backend", default="", choices=["", "memory", "chroma"])
    run.add_argument("--max-rounds", type=int, default=None)
    run.add_argument("--max-papers-discovered", type=int, default=None)
    run.add_argument("--max-papers-processed", type=int, default=None)
    run.add_argument("--max-llm-calls", type=int, default=None)
    run.add_argument("--max-input-tokens", type=int, default=None)
    run.add_argument("--max-output-tokens", type=int, default=None)
    run.add_argument("--max-cost-usd", type=float, default=None)
    run.add_argument("--max-seconds", type=float, default=None, dest="max_wall_clock_seconds")
    run.add_argument("--max-frontier-size", type=int, default=None)
    run.add_argument("--no-probe", action="store_true", help="skip live preflight probes")
    run.add_argument("--json", action="store_true", help="print the summary as JSON")

    resume = sub.add_parser("resume", help="continue an interrupted run")
    resume.add_argument("run_path", help="path to the run directory")
    resume.add_argument("--max-rounds", type=int, default=None)
    resume.add_argument("--max-cost-usd", type=float, default=None)
    resume.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="print a run's state")
    status.add_argument("run_path")
    status.add_argument("--json", action="store_true")

    report = sub.add_parser("report", help="regenerate report/summary/views from artifacts")
    report.add_argument("run_path")

    pre = sub.add_parser("preflight", help="check services without running anything")
    pre.add_argument("--species", required=True)
    pre.add_argument("--topic", default="")
    pre.add_argument("--runs-dir", default="")
    pre.add_argument("--no-probe", action="store_true")
    pre.add_argument("--json", action="store_true")

    listing = sub.add_parser("list", help="list runs")
    listing.add_argument("--runs-dir", default="")
    listing.add_argument("--json", action="store_true")

    return parser


def _budget_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in (
        "max_rounds",
        "max_papers_discovered",
        "max_papers_processed",
        "max_llm_calls",
        "max_input_tokens",
        "max_output_tokens",
        "max_cost_usd",
        "max_wall_clock_seconds",
        "max_frontier_size",
    ):
        value = getattr(args, name, None)
        if value is not None:
            out[name] = value
    return out


def _starting_inputs(args: argparse.Namespace) -> List[Any]:
    from t2pw.mining.acquisition import PaperReferenceError, make_paper_starting_input
    from t2pw.mining.config import StartingInput

    items: List[Any] = []
    for kind, values in (
        ("seed_metabolite", args.seed_metabolite),
        ("seed_enzyme", args.seed_enzyme),
        ("seed_reaction", args.seed_reaction),
    ):
        for value in values:
            items.append(StartingInput(kind=kind, raw=value, value=value))
    for ref in args.paper:
        try:
            items.append(make_paper_starting_input(ref))
        except PaperReferenceError as exc:
            raise SystemExit(f"error: {exc}") from exc
    if args.topic:
        items.append(StartingInput(kind="topic", raw=args.topic, value=args.topic))
    if args.starting_graph:
        path = Path(args.starting_graph)
        if not path.exists():
            raise SystemExit(f"error: starting graph not found: {path}")
        items.append(
            StartingInput(
                kind="pathway_graph",
                raw=str(path),
                value=path.stem,
                source_path=str(path.resolve()),
            )
        )
    return items


def _load_uploads(paths: Sequence[str]) -> List[Any]:
    from t2pw.mining.acquisition import record_from_upload

    uploads = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise SystemExit(f"error: paper file not found: {path}")
        uploads.append(record_from_upload(data=path.read_bytes(), file_name=path.name))
    return uploads


def _make_config(args: argparse.Namespace, *, enabled_via: str) -> Any:
    from t2pw.mining.config import MiningConfig

    overrides: Dict[str, Any] = {}
    if getattr(args, "config", ""):
        path = Path(args.config)
        if not path.exists():
            raise SystemExit(f"error: config file not found: {path}")
        overrides.update(json.loads(path.read_text(encoding="utf-8")))
    budgets = dict(overrides.get("budgets") or {})
    budgets.update(_budget_overrides(args))
    if budgets:
        overrides["budgets"] = budgets
    model = dict(overrides.get("model") or {})
    if getattr(args, "model", ""):
        model["model"] = args.model
    if model:
        overrides["model"] = model
    if getattr(args, "runs_dir", ""):
        overrides["runs_dir"] = args.runs_dir
    if getattr(args, "vector_backend", ""):
        overrides["vector_backend"] = args.vector_backend
    if getattr(args, "offline", False):
        overrides["offline"] = True
    if getattr(args, "no_screen", False):
        overrides["screen_papers"] = False
    overrides["run_id"] = getattr(args, "run_id", "") or ""
    overrides["enabled_via"] = enabled_via

    return MiningConfig.build(
        target_species=args.species,
        topic=getattr(args, "topic", "") or "",
        starting_inputs=_starting_inputs(args) if hasattr(args, "paper") else [],
        overrides=overrides,
    )


def _print_progress(event: str, payload: Dict[str, Any]) -> None:
    if event == "round_start":
        print(f"[round {payload.get('round')}] starting, frontier={payload.get('frontier')}",
              flush=True)
    elif event == "round_end":
        metrics = payload.get("metrics") or {}
        print(
            f"[round {payload.get('round')}] papers={metrics.get('papers_processed')} "
            f"claims={metrics.get('claims')} primary={metrics.get('primary_reactions')} "
            f"candidates={metrics.get('candidate_reactions')} "
            f"useful={'yes' if metrics.get('useful') else 'no'}",
            flush=True,
        )
    elif event == "finished":
        print(f"[run] {payload.get('state')} ({payload.get('stop_reason')})", flush=True)


def _cmd_run(args: argparse.Namespace) -> int:
    from t2pw.mining.controller import MiningController

    override = True if args.enable_pathway_mining else None
    config = _make_config(
        args, enabled_via="cli_flag" if args.enable_pathway_mining else "environment"
    )
    uploads = _load_uploads(args.paper_file)
    controller = MiningController(
        config, progress=_print_progress, enabled_override=override
    )
    outcome = controller.run(
        probe_model=not args.no_probe,
        probe_embeddings=not args.no_probe,
        uploaded=uploads,
    )
    print(f"\nrun directory: {outcome.run_dir.root}")
    if args.json:
        print(json.dumps(outcome.summary, indent=2, sort_keys=True, default=str))
    else:
        print(_render_status(outcome.summary))
    if outcome.manifest.completion_state == "preflight_failed":
        return EXIT_PREFLIGHT
    return EXIT_OK if outcome.completed_normally else EXIT_INCOMPLETE


def _cmd_resume(args: argparse.Namespace) -> int:
    from t2pw.mining.controller import MiningController

    override = True if args.enable_pathway_mining else None
    overrides = {k: v for k, v in _budget_overrides(args).items()}
    controller = MiningController.resume(
        args.run_path,
        progress=_print_progress,
        enabled_override=override,
        overrides=overrides,
    )
    outcome = controller.continue_run()
    print(f"\nrun directory: {outcome.run_dir.root}")
    if args.json:
        print(json.dumps(outcome.summary, indent=2, sort_keys=True, default=str))
    else:
        print(_render_status(outcome.summary))
    return EXIT_OK if outcome.completed_normally else EXIT_INCOMPLETE


def _render_status(summary: Dict[str, Any]) -> str:
    graph = summary.get("graph") or {}
    papers = summary.get("papers") or {}
    costs = summary.get("costs") or {}
    lines = [
        f"run_id            {summary.get('run_id')}",
        f"completion_state  {summary.get('completion_state')}"
        f"{'' if summary.get('completed_normally') else '  <- did NOT complete normally'}",
        f"stop_reason       {summary.get('stop_reason')}",
        f"rounds_completed  {summary.get('rounds_completed')}",
        f"papers            {papers.get('included', 0)} included / "
        f"{papers.get('excluded', 0)} excluded / {papers.get('processed', 0)} processed",
        f"primary graph     {graph.get('primary_reactions', 0)} reactions, "
        f"{graph.get('chemical_connections', 0)} chemical connections",
        f"candidates        {graph.get('candidate_reactions', 0)} reactions in "
        f"{graph.get('candidate_components', 0)} components",
        f"conflicts         {(summary.get('conflicts') or {}).get('unresolved', 0)} unresolved",
        f"model calls       {costs.get('llm_calls', 0)}"
        f"  est. ${costs.get('estimated_cost_usd', 0.0):.6f} (estimate, not a bill)",
        "PWML              none - this mode produces a research graph only",
        "database check    deferred (not_evaluated)",
    ]
    return "\n".join(lines)


def _cmd_status(args: argparse.Namespace) -> int:
    from t2pw.mining.artifacts import RunDirectory, read_json

    run_dir = RunDirectory.open(Path(args.run_path))
    summary = read_json(run_dir.path("summary.json"))
    if not summary:
        manifest = run_dir.read_manifest()
        if manifest is None:
            print(f"error: no manifest under {run_dir.root}", file=sys.stderr)
            return EXIT_USAGE
        summary = {
            "run_id": manifest.run_id,
            "completion_state": manifest.completion_state,
            "completed_normally": manifest.completed_normally,
            "stop_reason": manifest.stop_reason,
            "rounds_completed": manifest.rounds_completed,
        }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(_render_status(summary))
    return EXIT_OK if summary.get("completed_normally") else EXIT_INCOMPLETE


def _cmd_report(args: argparse.Namespace) -> int:
    """Regenerate report.md, summary.json and the graph views from artifacts."""
    from t2pw.mining.artifacts import RunDirectory
    from t2pw.mining.regenerate import regenerate

    run_dir = RunDirectory.open(Path(args.run_path))
    written = regenerate(run_dir)
    for path in written:
        print(path)
    return EXIT_OK


def _cmd_preflight(args: argparse.Namespace) -> int:
    from t2pw.mining.preflight import run_preflight

    override = True if args.enable_pathway_mining else None
    config = _make_config(
        args, enabled_via="cli_flag" if args.enable_pathway_mining else "environment"
    )
    report = run_preflight(
        config,
        probe_model=not args.no_probe,
        probe_embeddings=not args.no_probe,
        enabled_override=override,
    )
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())
    return EXIT_OK if report.ok else EXIT_PREFLIGHT


def _cmd_list(args: argparse.Namespace) -> int:
    from t2pw.mining.artifacts import list_runs
    from t2pw.mining.config import DEFAULT_RUNS_DIR
    from t2pw.paths import PROJECT_ROOT

    runs_dir = Path(args.runs_dir) if args.runs_dir else (PROJECT_ROOT / DEFAULT_RUNS_DIR)
    rows = list_runs(runs_dir)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return EXIT_OK
    if not rows:
        print(f"no runs under {runs_dir}")
        return EXIT_OK
    for row in rows:
        print(
            f"{row['run_id']}  {row['completion_state']:<18} rounds={row['rounds_completed']:<3} "
            f"{row['species']} / {row['topic']}"
        )
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    from t2pw.mining.flags import (
        FEATURE_ENV_VAR,
        PathwayMiningDisabled,
        mining_enabled,
    )

    override = True if args.enable_pathway_mining else None
    if args.command in {"run", "resume", "preflight"} and not mining_enabled(override):
        print(
            "Pathway mining is disabled (experimental feature, default off).\n"
            f"Enable it with {FEATURE_ENV_VAR}=true, or pass --enable-pathway-mining "
            "before the subcommand.",
            file=sys.stderr,
        )
        return EXIT_DISABLED

    handlers = {
        "run": _cmd_run,
        "resume": _cmd_resume,
        "status": _cmd_status,
        "report": _cmd_report,
        "preflight": _cmd_preflight,
        "list": _cmd_list,
    }
    try:
        return handlers[args.command](args)
    except PathwayMiningDisabled as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DISABLED
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return EXIT_INCOMPLETE
    except SystemExit:
        raise
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


__all__ = ["main", "EXIT_OK", "EXIT_USAGE", "EXIT_DISABLED", "EXIT_PREFLIGHT", "EXIT_INCOMPLETE"]

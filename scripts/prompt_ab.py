"""Run one paper through Stage-1 extraction under several system prompts, side by side.

For HAND comparison: it scores nothing and decides nothing. Each variant gets the same
paper, the same model and the same settings, and its output is written twice -- the raw
payload, and the readable summary in the format the app produces -- so the outputs can be
read against each other.

Why not promptfoo: it would work, and `npx promptfoo` is available here, but every
LLM-backed command in this project has to run inside the bounded wrapper with zero
surviving processes (TEST_MATRIX section 0), and a Node runner spawning its own workers
breaks that ownership model. Its real value-add is assertions and scoring, which is the
part being done by eye here anyway.

The Stage-1 call mirrors ``pipeline.run_extraction_pipeline`` exactly, through the same
private stage runner, so retries and JSON repair behave as they do in production. The one
difference is the system prompt, which is the point. If that function's arguments change,
this harness must be updated with it.

Usage, under the bounded wrapper::

    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \
      --timeout 3600 --label orch747-prompt-ab --json <evidence>.json -- \
      .venv/Scripts/python.exe scripts/prompt_ab.py \
        --text C:/t/eval-fulltext/PMC319880.r3.txt \
        --variant committed=git:HEAD \
        --variant working=src/t2pw/llm/prompts/pwml_system.txt \
        --pathway "Ara operon inactivation (autoregulation of AraC)" \
        --organism "Escherichia coli" \
        --out C:/t/prompt-ab/pw701720 --runs 1

``--variant NAME=git:<rev>`` reads ``pwml_system.txt`` as committed at that revision, which
is how a working-tree edit is compared against the baseline it departed from.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

PROMPT_REL = "src/t2pw/llm/prompts/pwml_system.txt"


def _app_extraction_max_tokens() -> int:
    """The output budget the APP gives Stage 1, resolved the same way it does.

    A harness that under-budgets the model does not measure the prompt, it measures the
    budget. At 12000 the longer variants came back EMPTY with ``finish_reason=length``:
    the model spent the whole allowance before emitting any content, six of ten rows
    errored, and it looked like a prompt failure. streamlit_app.py reads
    OPENROUTER_EXTRACTION_MAX_TOKENS with a 64000 default; this file reads the same.
    """

    import os

    raw = (os.getenv("OPENROUTER_EXTRACTION_MAX_TOKENS") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 64000
    return value if 500 <= value <= 128000 else 64000


def _load_variant(spec: str) -> Tuple[str, str, str]:
    """``NAME=PATH`` or ``NAME=git:<rev>`` to ``(name, origin, prompt_text)``."""

    if "=" not in spec:
        raise SystemExit(f"--variant needs NAME=PATH or NAME=git:<rev>, got {spec!r}")
    name, _, where = spec.partition("=")
    name, where = name.strip(), where.strip()
    if not name or not where:
        raise SystemExit(f"--variant needs both a name and a source, got {spec!r}")
    if where.startswith("git:"):
        rev = where[4:] or "HEAD"
        out = subprocess.run(["git", "show", f"{rev}:{PROMPT_REL}"], cwd=REPO,
                             capture_output=True, text=True, encoding="utf-8")
        if out.returncode != 0:
            raise SystemExit(f"{name}: cannot read {PROMPT_REL} at {rev}: {out.stderr.strip()}")
        return name, f"git:{rev}:{PROMPT_REL}", out.stdout
    p = Path(where)
    if not p.is_absolute():
        p = REPO / p
    if not p.is_file():
        raise SystemExit(f"{name}: no such prompt file: {p}")
    return name, str(p), p.read_text(encoding="utf-8")


def _extract(prompt_text: str, paper_text: str, *, pathway: str, organism: str,
             scope: str, temperature: float, max_tokens: int,
             max_attempts: int) -> Dict[str, Any]:
    """One Stage-1 draw with ``prompt_text`` as the system prompt."""

    from t2pw.pipeline.pipeline import (_build_extraction_prompt, _run_json_stage,
                                        is_ambiguous_multi_example_review_context)

    context: Dict[str, Any] = {}
    if pathway:
        context["name"] = pathway
    if organism:
        context["organism"] = organism
    if is_ambiguous_multi_example_review_context(context or None):
        raise SystemExit("the pathway context reads as an ambiguous multi-example review")

    payload, _logs = _run_json_stage(
        stage_name="extraction",
        model_env_var="OPENROUTER_EXTRACTION_MODEL",
        system_prompt=prompt_text,
        build_user_prompt=lambda prev_output, last_error: _build_extraction_prompt(
            paper_text, prev_output, last_error,
            pathway_context=context or None,
            pathway_scope=scope or None,
            user_task_context=None,
        ),
        max_attempts=max_attempts,
        temperature=temperature,
        max_tokens=max_tokens,
        retry_on_empty_payload=True,
    )
    return payload


def _summary(payload: Dict[str, Any]) -> str:
    """The payload in the readable format the app writes, or the reason it could not be."""

    try:
        from t2pw.pipeline.draft_graph import build_draft_graph
        from t2pw.pipeline.reaction_summary import generate_reaction_summary
        try:
            from t2pw.pipeline.qa_graph import generate_qa_report
        except Exception:  # noqa: BLE001 - the summary degrades without QA flags
            generate_qa_report = None  # type: ignore[assignment]
        graph = build_draft_graph(payload)
        qa = {}
        if generate_qa_report is not None:
            try:
                qa = generate_qa_report(graph)
            except Exception:  # noqa: BLE001
                qa = {}
        return generate_reaction_summary(graph, qa)
    except Exception:  # noqa: BLE001 - a broken render must not lose the payload
        return "could not render a summary for this payload:\n" + traceback.format_exc()


def _counts(payload: Dict[str, Any]) -> Dict[str, int]:
    processes = payload.get("processes") or {}
    entities = payload.get("entities") or {}
    out = {k: len(v) for k, v in processes.items() if isinstance(v, list)}
    for key in ("compounds", "proteins", "protein_complexes", "nucleic_acids"):
        out[key] = len(entities.get(key) or [])
    rows = [r for r in (processes.get("interactions") or []) if isinstance(r, dict)]
    out["interactions_without_evidence"] = sum(
        1 for r in rows if not str(r.get("evidence") or "").strip())
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--text", required=True, help="the paper text to extract from")
    ap.add_argument("--variant", action="append", required=True, metavar="NAME=SRC",
                    help="NAME=PATH or NAME=git:<rev>; repeatable")
    ap.add_argument("--out", required=True, help="output directory (outside the repo)")
    ap.add_argument("--runs", type=int, default=1,
                    help="draws per variant. Identical legs differ even at temperature 0, "
                         "so use 3 before trusting a difference")
    ap.add_argument("--pathway", default="")
    ap.add_argument("--organism", default="")
    ap.add_argument("--scope", default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default: the same budget the app gives Stage 1")
    ap.add_argument("--max-attempts", type=int, default=2)
    args = ap.parse_args(list(argv) if argv is not None else None)

    from t2pw.config import ensure_dotenv_loaded
    ensure_dotenv_loaded()   # the CLI is not the app: without this LLM_PROVIDER is unset

    paper = Path(args.text)
    if not paper.is_file():
        raise SystemExit(f"no such paper text: {paper}")
    paper_text = paper.read_text(encoding="utf-8")
    variants = [_load_variant(v) for v in args.variant]
    names = [n for n, _, _ in variants]
    if len(set(names)) != len(names):
        raise SystemExit(f"variant names must be unique, got {names}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"paper      : {paper} ({len(paper_text)} chars, "
          f"sha256 {hashlib.sha256(paper.read_bytes()).hexdigest()[:16]})")
    print(f"variants   : {', '.join(names)}")
    print(f"draws      : {len(variants)} x {args.runs} = {len(variants) * args.runs} "
          f"billed extraction call(s)")
    print()

    index: List[Dict[str, Any]] = []
    for name, origin, prompt_text in variants:
        for run in range(1, args.runs + 1):
            stem = f"{name}__run{run}"
            print(f"--- {stem}: extracting ...", flush=True)
            record: Dict[str, Any] = {
                "variant": name, "run": run, "prompt_origin": origin,
                "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "prompt_chars": len(prompt_text),
            }
            try:
                payload = _extract(prompt_text, paper_text, pathway=args.pathway,
                                   organism=args.organism, scope=args.scope,
                                   temperature=args.temperature, max_tokens=args.max_tokens or _app_extraction_max_tokens(),
                                   max_attempts=args.max_attempts)
            except BaseException as exc:  # noqa: BLE001 - one variant must not kill the rest
                record["error"] = f"{type(exc).__name__}: {exc}"
                (out_dir / f"{stem}__error.txt").write_text(traceback.format_exc(),
                                                            encoding="utf-8")
                print(f"    FAILED: {record['error']}")
                index.append(record)
                if isinstance(exc, KeyboardInterrupt):
                    break
                continue

            (out_dir / f"{stem}__payload.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            (out_dir / f"{stem}__summary.txt").write_text(_summary(payload), encoding="utf-8")
            record["counts"] = _counts(payload)
            index.append(record)
            print(f"    {record['counts']}")

    manifest = {
        "schema": "t2pw.prompt_ab/1",
        "ran_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "paper": str(paper),
        "paper_sha256": hashlib.sha256(paper.read_bytes()).hexdigest(),
        "pathway": args.pathway, "organism": args.organism, "scope": args.scope,
        "temperature": args.temperature, "max_tokens": args.max_tokens or _app_extraction_max_tokens(),
        "max_attempts": args.max_attempts,
        "note": "Hand comparison only. Nothing here is scored, and identical legs differ "
                "even at temperature 0, so a single-draw difference is not a result.",
        "draws": index,
    }
    (out_dir / "index.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                        encoding="utf-8")

    keys: List[str] = []
    for row in index:
        for k in (row.get("counts") or {}):
            if k not in keys:
                keys.append(k)
    width = max([len("variant / run")] + [len(f"{r['variant']} run{r['run']}") for r in index])
    print()
    print("  " + "variant / run".ljust(width) + "  " + "  ".join(k[:12].rjust(12) for k in keys))
    for row in index:
        label = f"{row['variant']} run{row['run']}".ljust(width)
        if row.get("error"):
            print("  " + label + "  FAILED: " + row["error"][:60])
            continue
        counts = row["counts"]
        print("  " + label + "  " + "  ".join(str(counts.get(k, 0)).rjust(12) for k in keys))
    print()
    print(f"outputs: {out_dir}")
    print("read side by side:  *__summary.txt   (payloads in *__payload.json)")
    return 0 if all(not r.get("error") for r in index) else 1


if __name__ == "__main__":
    sys.exit(main())

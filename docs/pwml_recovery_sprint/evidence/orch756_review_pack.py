"""ORCH-756/759: gather one finished paper leg into the product owner's review folder.

Copy of ``orch755_review_pack.py`` (ORCH-755) with two changes: it copies EVERY ``*.pwml`` and every
``research_pathway_*`` report of the leg, and the index title names the task. Called by
``orch756_paper_runs.sh`` after every leg. Presentation only: no model, network or pipeline call.

``ROOT`` is this script's own tree, so run the copy that lives in the tree the legs ran in: it reads
that tree's ``tmp/extraction_diagnostics`` and imports that tree's ``src``. ``--review`` may point
anywhere (ORCH-756 writes straight into the primary checkout's ``runs_papers/ORCH-756/REVIEW/``).

    python orch756_review_pack.py --task ORCH-756 --leg-out <batch --out dir> --cond <name>
        --mode <strict|research> --started <epoch> --finished <epoch> --exit <code> --review <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from t2pw.app.reaction_evidence import build_report, load_run_artifacts, render_text  # noqa: E402

DIAG_ROOT = ROOT / "tmp" / "extraction_diagnostics"
SIDE_FILES = ("stage1_ensemble_report.json", "stage1_ensemble_report.txt",
              "ensemble_identity_merge_report.json", "ensemble_identity_merge_report.txt")


def _leg_dir(leg_out: Path, mode: str) -> Path | None:
    hits = sorted(glob.glob(str(leg_out / "*" / "papers" / "*" / mode)))
    return Path(hits[-1]) if hits else None


def _side_files(started: float, finished: float) -> list[Path]:
    """Ensemble side files written during this leg's window (legs run one at a time)."""
    found = []
    for name in SIDE_FILES:
        for f in glob.glob(str(DIAG_ROOT / "**" / name), recursive=True):
            if started - 5 <= os.path.getmtime(f) <= finished + 5:
                found.append(Path(f))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ORCH-756")
    ap.add_argument("--leg-out", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--started", type=float, required=True)
    ap.add_argument("--finished", type=float, required=True)
    ap.add_argument("--exit", default="")
    ap.add_argument("--review", required=True)
    a = ap.parse_args()

    review = Path(a.review)
    dest = review / a.cond / a.mode
    dest.mkdir(parents=True, exist_ok=True)
    row = {"cond": a.cond, "mode": a.mode, "exit": a.exit,
           "minutes": round((a.finished - a.started) / 60, 1)}

    leg = _leg_dir(Path(a.leg_out), a.mode)
    row["leg_dir"] = str(leg) if leg else "NOT FOUND"
    if leg:
        pwmls = sorted(p.name for p in leg.glob("*.pwml"))
        for p in [leg / "RESULT.txt", *leg.glob("*.pwml"), *leg.glob("research_pathway_*")]:
            if p.is_file():
                shutil.copy2(p, dest / p.name)
        row["pwml"] = ", ".join(pwmls) or "none written"
        try:
            report = build_report(load_run_artifacts(leg))
            (dest / "reaction_evidence.txt").write_text(render_text(report), encoding="utf-8")
            row.update(reactions=len(report.get("reactions") or []),
                       interactions=report.get("interaction_count"),
                       quarantined=report.get("quarantined_processes"),
                       release=report.get("release_status"))
        except Exception as exc:  # a broken leg must not stop the run
            row["reaction_evidence"] = f"render failed: {type(exc).__name__}: {exc}"
        result = leg / "RESULT.txt"
        if result.is_file():
            row["result"] = result.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0][:160]
    sides = _side_files(a.started, a.finished)
    for f in sides:
        shutil.copy2(f, dest / f.name)
    row["ensemble_side_files"] = [str(f) for f in sides] or "none found in this leg's window"

    rows_path = review / "index.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if rows_path.exists() else []
    rows = [r for r in rows if (r["cond"], r["mode"]) != (a.cond, a.mode)] + [row]
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    lines = [f"{a.task} paper runs: review index (rewritten after every leg)", "",
             "Each folder <condition>/<mode>/ holds:",
             "  reaction_evidence.txt        the exact record: every reaction and interaction with its quotes",
             "  *.pwml                       every PathWhiz file the leg wrote, if any",
             "  research_pathway_*           the research-mode report (txt, csv, citations), if any",
             "  stage1_ensemble_report.txt   the Stage-1 pooling record (support marks, merge flags)",
             "  RESULT.txt                   the leg's own outcome", ""]
    for r in rows:
        lines.append(f"{r['cond']:>13} / {r['mode']:<8} exit={r.get('exit')} {r.get('minutes')} min | "
                     f"pwml: {r.get('pwml', '-')} | reactions {r.get('reactions', '-')}, "
                     f"interactions {r.get('interactions', '-')} | release {r.get('release', '-')}")
        if r.get("result"):
            lines.append(f"{'':>15}{r['result']}")
        if r.get("reaction_evidence"):
            lines.append(f"{'':>15}{r['reaction_evidence']}")
        if isinstance(r.get("ensemble_side_files"), str):
            lines.append(f"{'':>15}ensemble report: {r['ensemble_side_files']}")
    (review / "INDEX.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

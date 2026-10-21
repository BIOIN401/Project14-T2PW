"""ORCH-755: gather one finished paper leg into the product owner's review folder.

Called by ``orch755_paper_runs.sh`` after every leg, so the folder fills in as the night goes
and a run cut short still leaves everything that finished. Presentation only: it reads a leg's
existing artifacts, renders the same deterministic Reaction Evidence text the app's ``.txt``
download produces (``t2pw.app.reaction_evidence``), copies the PWML and the Stage-1 ensemble
side files, and rewrites ``INDEX.txt``. No model, network or pipeline call.

    python orch755_review_pack.py --leg-out <batch --out dir> --cond <name> --mode <strict|research>
        --started <epoch> --finished <epoch> --exit <code> --review <review dir>
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
PWMLS = ("pathway.pwml", "pathway.review_required.pwml")


def _leg_dir(leg_out: Path, mode: str) -> Path | None:
    hits = sorted(glob.glob(str(leg_out / "*" / "papers" / "*" / mode)))
    return Path(hits[-1]) if hits else None


def _side_files(started: float, finished: float) -> list[Path]:
    """Ensemble side files written during this leg's window. Legs run one at a time, so the
    window identifies them; every match is listed in the index so a wrong pick is visible."""
    found = []
    for name in SIDE_FILES:
        for f in glob.glob(str(DIAG_ROOT / "**" / name), recursive=True):
            m = os.path.getmtime(f)
            if started - 5 <= m <= finished + 5:
                found.append(Path(f))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
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
        for name in ("RESULT.txt",) + PWMLS:
            if (leg / name).is_file():
                shutil.copy2(leg / name, dest / name)
        row["pwml"] = next((n for n in PWMLS if (leg / n).is_file()), "none written")
        try:
            report = build_report(load_run_artifacts(leg))
            (dest / "reaction_evidence.txt").write_text(render_text(report), encoding="utf-8")
            row.update(reactions=len(report.get("reactions") or []),
                       interactions=report.get("interaction_count"),
                       quarantined=report.get("quarantined_processes"),
                       release=report.get("release_status"))
        except Exception as exc:  # a broken leg must not stop the night
            row["reaction_evidence"] = f"render failed: {type(exc).__name__}: {exc}"
        result = (leg / "RESULT.txt")
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

    lines = ["ORCH-755 paper runs: review index (rewritten after every leg)", "",
             "Each folder <condition>/<mode>/ holds:",
             "  reaction_evidence.txt        the exact record: every reaction and interaction with its quotes",
             "  pathway*.pwml                the PathWhiz file, if one was written",
             "  stage1_ensemble_report.txt   the 3-draw merge: support 3/3, 2/3, 1/3 and every merge flag",
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

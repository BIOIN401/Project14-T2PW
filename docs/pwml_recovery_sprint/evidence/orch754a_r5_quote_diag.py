"""ORCH-754a round 5 diagnostic: which draw quotes are lost, where they sat, and whether the
JSON side file or only the .txt misses them."""
import json
import sys
from copy import deepcopy
from pathlib import Path

W = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch754a")
sys.path.insert(0, str(W / "src"))
from t2pw.pipeline.stage_one_ensemble import merge_draws, render_report_text  # noqa: E402

E = W / "docs/pwml_recovery_sprint/evidence"
COHORT = {"chlorophyll": (E / "orch752_chlorophyll_payloads", "PMC5130987"),
          "enterobactin": (E / "orch754a_inputs/PMC12096016_enterobactin", "PMC12096016"),
          "menaquinone": (E / "orch754a_inputs/PMC12421875_menaquinone", "PMC12421875")}


def quotes(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "evidence" and v:
                yield path, v
            else:
                yield from quotes(v, path + "/" + k)
    elif isinstance(x, list):
        for v in x:
            yield from quotes(v, path)


def strings(x):
    if isinstance(x, dict):
        return {s for v in x.values() for s in strings(v)}
    if isinstance(x, list):
        return {s for v in x for s in strings(v)}
    return {x} if isinstance(x, str) else set()


for case, (folder, pmc) in COHORT.items():
    draws = [json.loads((folder / f"control__run{i}__payload.json").read_text(encoding="utf-8")) for i in (1, 2, 3)]
    paper = (E / "orch754a_inputs/papers" / f"{pmc}.txt").read_text(encoding="utf-8")
    merged, report = merge_draws(deepcopy(draws), paper_text=paper)
    kept, side, text = strings(merged), strings(report), render_report_text(report)
    rows = {}
    for d, x in enumerate(draws, 1):
        for p, q in quotes(x):
            if q in kept:
                continue
            where = "JSON+txt" if q in side and repr(q) in text else ("txt only missing" if q in side else "LOST")
            rows.setdefault((where, p), set()).add(q)
    print(f"== {case}")
    for (where, p), qs in sorted(rows.items()):
        print(f"  {where:17s} {p:45s} {len(qs)}  e.g. {sorted(qs)[0][:70]!r}")

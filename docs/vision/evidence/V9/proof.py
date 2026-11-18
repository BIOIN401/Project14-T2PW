"""V9 proof: score the stored runs of the gold papers through the benchmark scorer.

Run (through bw.sh; read-only on every run directory and on the eval_v2 gold files)::

    PYTHONPATH=<tree>/src python docs/vision/evidence/V9/proof.py <out dir>

Writes <out dir>/proof.json. Paper set: PAPER_ONLY and DEFAULT views of every stored run of the
certified gold papers; DEFAULT is cross-checked against eval_v2 ``score_leg`` (same rows, same
gold) and against the committed sample_report.json where the same run is in it. Full set: PW701720
(committed extract) against the stored Ara runs, plus the other cohort pathways when the read-only
PathBank database answers.
"""

import json
import re
import sys
from pathlib import Path

from t2pw.bench.eval_v2 import reactions as R
from t2pw.bench.eval_v2.gold import load_supplement
from t2pw.vision.bench import TYPES, build_full_gold, gold_from_pathbank, gold_from_supplement, run_benchmark, score_view
from t2pw.vision.payload import big_from_run
from t2pw.vision.view import DEFAULT, FULL, PAPER_ONLY

TREE = Path(__file__).resolve().parents[4]
RUNLIST = Path("C:/t/ev-cm/runlist.json")
PAPER_CASES = ("PMC12096016", "PMC12312563", "PMC12421875", "PMC12657337")
FULL_CASES = {"PW701720": "PMC11983282", "PW739181": "PMC7059786", "PW495535": "PMC10425814",
              "PW488575": "PMC10425814", "PW766996": "PMC5130987"}


def short(d):
    return re.sub(r"^.*Project14-T2PW/", "", d.replace("\\", "/"))


def stored_runs(paper):
    out = {}
    for r in json.loads(RUNLIST.read_text(encoding="utf-8")):
        d = r["dir"].replace("\\", "/")
        if f"/papers/{paper}/" in d + "/" and (Path(r["dir"]) / "final_mapped.json").is_file():
            out[short(d)] = r["dir"]
    return out


def compact(s):
    t = s["types"]
    return {k: {x: t[k][x] for x in ("gold", "recalled", "predicted", "correct", "recall", "precision")} for k in TYPES}


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sup = load_supplement()
    raw = json.loads((TREE / "src/t2pw/bench/eval_v2/gold_v2_supported_reactions.json").read_text(encoding="utf-8"))["cases"]
    sample = {short(l["leg_dir"]): l["metrics"]["table1_paper_extraction"]
              for l in json.loads((TREE / "docs/evaluation_v2/sample_report.json").read_text(encoding="utf-8"))["reaction_metrics"]["legs"]}
    report = {"paper": {}, "full": {}}

    # ---- paper set -------------------------------------------------------------------------
    cases, runs_by_case, bigs = [], {}, {}
    for pid in PAPER_CASES:
        cases.append(gold_from_supplement(raw[pid], pid))
        runs = stored_runs(pid)
        runs_by_case[pid] = {}
        for label, d in runs.items():
            big = big_from_run(d)
            runs_by_case[pid][label] = big
            payload = big["base"]
            leg = R.score_leg(sup[pid], payload, [], None)["table1_paper_extraction"]
            mine = score_view(big, DEFAULT, cases[-1])["types"]["reactions"]
            same = (mine["recalled"], mine["gold"], mine["correct"], mine["predicted"]) == (
                len(leg["gold_reactions_matched"]), leg["gold_reactions"], leg["true_positive_rows"], leg["rows_claimed_as_paper_extraction"])
            ref = sample.get(label)
            report["paper"].setdefault(pid, {})[label] = {
                "default_equals_eval_v2_score_leg": same,
                "in_committed_sample_report": ref is not None,
                "recall_equals_sample_report": None if ref is None else mine["recall"] == ref["recall"],
                "tp_equals_sample_report": None if ref is None else mine["correct"] == ref["true_positive_rows"],
                "sample_report_rows_claimed": None if ref is None else ref["rows_claimed_as_paper_extraction"],
                "default": compact(score_view(big, DEFAULT, cases[-1])),
                "paper_only": compact(score_view(big, PAPER_ONLY, cases[-1]))}
    report["paper_summary"] = {
        "runs_scored": {pid: len(r) for pid, r in runs_by_case.items()},
        "default_equals_eval_v2_on_all_legs": all(v["default_equals_eval_v2_score_leg"] for p in report["paper"].values() for v in p.values()),
        "legs_in_sample_report": sum(v["in_committed_sample_report"] for p in report["paper"].values() for v in p.values()),
        "legs_recall_equals_sample_report": sum(bool(v["recall_equals_sample_report"]) for p in report["paper"].values() for v in p.values()),
        "legs_tp_equals_sample_report": sum(bool(v["tp_equals_sample_report"]) for p in report["paper"].values() for v in p.values())}
    for name, scope in (("paper_only", PAPER_ONLY), ("default", DEFAULT)):
        agg = run_benchmark([c for c in cases if runs_by_case[c["id"]]], runs_by_case, scope)
        report["paper_aggregate_" + name] = agg["aggregate"]["paper"]["types"]

    # ---- full set --------------------------------------------------------------------------
    gold_cases = {"PW701720": gold_from_pathbank(
        json.loads((TREE / "docs/evaluation_v2/pathbank_gold/PW701720__extract.json").read_text(encoding="utf-8")), "PW701720")}
    for pw in FULL_CASES:
        if pw not in gold_cases:
            try:
                gold_cases[pw] = build_full_gold(pw)
                (out / f"gold_{pw}.json").write_text(json.dumps(gold_cases[pw], indent=1, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                report["full"][pw] = {"gold": f"not built: {type(exc).__name__}: {str(exc)[:200]}"}
    for pw, g in gold_cases.items():
        runs = {label: big_from_run(d) for label, d in stored_runs(FULL_CASES[pw]).items()}
        report["full"][pw] = {"paper": FULL_CASES[pw], "gold_counts": {t: len(g[t]) for t in TYPES}, "runs": {
            label: {"full": compact(score_view(big, FULL, g)), "default": compact(score_view(big, DEFAULT, g)),
                    "paper_only_coverage": compact(score_view(big, PAPER_ONLY, g)),
                    "extension": score_view(big, FULL, g)["extension"]["all"],
                    "provenance_reactions": score_view(big, FULL, g)["provenance"]["reactions"]["primary"]}
            for label, big in runs.items()}}
    (out / "proof.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["paper_summary"], indent=1))
    print(json.dumps({"paper_only": report["paper_aggregate_paper_only"]["reactions"],
                      "default": report["paper_aggregate_default"]["reactions"]}, indent=1))
    print({pw: (v.get("gold") if "runs" not in v else f"{len(v['runs'])} runs") for pw, v in report["full"].items()})


if __name__ == "__main__":
    main(sys.argv[1])

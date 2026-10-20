"""ORCH-754a: pooling three saved Stage-1 draws with the paper as the dictionary (D-104).

Offline only: no model, no network, no pipeline run. For each case the three saved draws
are pooled four ways and the pooled reactions are counted -- and, for the two certified
gold cases, scored with the evaluator's own ``reactions.score_leg`` (as
``scripts/score_prompt_grid.py`` and ORCH-751 § 2.1 do):

* ``base``         -- ``stage_one_ensemble.merge_draws`` AS COMMITTED at ``--base-rev``
                      (ORCH-750 + ORCH-752's cofactor-blind key), loaded from ``git show``;
* ``tip_no_paper`` -- the tip's merge with no paper text: finding 1's compatibility rule only;
* ``tip``          -- the tip's merge with the paper text, which is what
                      ``run_extraction_pipeline`` now does at N>1 (DELIVERED);
* ``counterfactual_undefined_state_is_no_state`` -- ``tip`` with ONE rule relaxed, for the
                      open question in the RESULT doc: a biological_state label its draw never
                      defines is treated as no state instead of failing closed. NOT delivered.

Inputs are committed beside this file: the chlorophyll draws in
``orch752_chlorophyll_payloads/``, the enterobactin and menaquinone draws and all three
paper texts in ``orch754a_inputs/``. Every paper's sha256 is checked against the
``paper_sha256`` its draws' ``index.json`` recorded; payload hashes are taken over
LF-normalised bytes so a CRLF checkout reproduces them.

Usage::

    PYTHONPATH=src python docs/pwml_recovery_sprint/evidence/orch754a_ensemble_merge.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from t2pw.bench.eval_v2 import reactions as R  # noqa: E402
from t2pw.bench.eval_v2.gold import load_fulltext, load_supplement  # noqa: E402
from t2pw.pipeline import stage_one_ensemble as tip  # noqa: E402

INPUTS = HERE / "orch754a_inputs"
CASES = {
    "chlorophyll": {"paper": "PMC5130987", "draws": HERE / "orch752_chlorophyll_payloads", "gold": False},
    "enterobactin": {"paper": "PMC12096016", "draws": INPUTS / "PMC12096016_enterobactin", "gold": True},
    "menaquinone": {"paper": "PMC12421875", "draws": INPUTS / "PMC12421875_menaquinone", "gold": True},
}
ORCH751 = {"enterobactin": {"matched": 4, "gold": 5, "union_rows": 6},
           "menaquinone": {"matched": 9, "gold": 9, "union_rows": 9}}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _base_module(rev: str) -> Any:
    text = subprocess.run(["git", "show", f"{rev}:src/t2pw/pipeline/stage_one_ensemble.py"], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8", check=True).stdout
    path = Path(tempfile.mkdtemp(prefix="orch754a_base_")) / "stage_one_ensemble_base.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("stage_one_ensemble_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _score(payload: Dict[str, Any], case: Any, text: Optional[str]) -> Dict[str, Any]:
    metrics = R.score_leg(case, payload, [], text, expected_species=case.species)
    t1 = metrics.get("table1_paper_extraction") or {}
    return {"gold_reactions": t1.get("gold_reactions"),
            "matched": len(t1.get("gold_reactions_matched") or []),
            "gold_reactions_matched": t1.get("gold_reactions_matched"),
            "rows_claimed": t1.get("rows_claimed_as_paper_extraction"),
            "precision": t1.get("precision"), "recall": t1.get("recall"), "f1": t1.get("f1"),
            "row_classes": metrics.get("row_classes")}


def _rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (report.get("buckets") or {}).get("/processes/reactions", {}).get("rows") or []


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--base-rev", default="f9892cdf")
    ap.add_argument("--out", default=str(HERE / "orch754a_ensemble_merge.json"))
    args = ap.parse_args()

    base = _base_module(args.base_rev)
    gold = dict(load_supplement())
    delivered = tip._state_profile

    def undefined_is_none(label: Any, states: Dict[str, Any]) -> Dict[str, str]:
        return delivered(label, states) if tip._strict_key(label) in states else {}

    result: Dict[str, Any] = {"task": "ORCH-754a", "base_rev": args.base_rev, "cases": {}}
    lines = ["ORCH-754a pooled Stage-1 draws, paper as the dictionary (offline; 3 saved draws each)", ""]
    for name, spec in CASES.items():
        files = [spec["draws"] / f"control__run{i}__payload.json" for i in (1, 2, 3)]
        draws = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        paper_path = INPUTS / "papers" / f"{spec['paper']}.txt"
        paper_bytes = paper_path.read_bytes()
        recorded = json.loads((spec["draws"] / "index.json").read_text(encoding="utf-8"))["paper_sha256"]
        assert _sha(paper_bytes) == recorded, f"{paper_path} is not the paper these draws read"
        paper = paper_bytes.decode("utf-8")

        pooled = {"base": base.merge_draws(draws), "tip_no_paper": tip.merge_draws(draws),
                  "tip": tip.merge_draws(draws, paper_text=paper)}
        with patch.object(tip, "_state_profile", undefined_is_none):
            pooled["counterfactual_undefined_state_is_no_state"] = tip.merge_draws(draws, paper_text=paper)

        entry: Dict[str, Any] = {
            "paper": spec["paper"], "paper_sha256": recorded,
            "payloads": [{"file": str(f.relative_to(HERE)), "sha256_lf": _sha(f.read_bytes().replace(b"\r\n", b"\n"))}
                         for f in files],
            "reactions_per_draw": [len(d["processes"]["reactions"]) for d in draws],
            "pooled_reactions": {k: len(m["processes"]["reactions"]) for k, (m, _r) in pooled.items()},
        }
        if spec["gold"]:
            case = gold[spec["paper"]]
            text = load_fulltext(case, INPUTS / "papers")
            entry["orch751_union"] = ORCH751[name]
            entry["per_draw_score"] = [_score(d, case, text) for d in draws]
            entry["score"] = {k: _score(m, case, text) for k, (m, _r) in pooled.items()}
        tip_merged, tip_report = pooled["tip"]
        entry["tip_lexicon"] = tip_report["paper_lexicon"]
        entry["tip_names_rewritten"] = tip_report["names_rewritten"]
        entry["tip_rows"] = _rows(tip_report)
        entry["tip_payload_reactions"] = [
            {k: r.get(k) for k in ("name", "inputs", "outputs", "biological_state", "evidence")}
            for r in tip_merged["processes"]["reactions"]]
        result["cases"][name] = entry

        # The human-facing hand-comparison text (ruling 2026-10-20): the side file rendered,
        # never reaction_summary.txt, which a model reads.
        (HERE / f"orch754a_{name}_side_file.txt").write_text(tip.render_report_text(tip_report), encoding="utf-8")

        lines.append(f"== {name} ({spec['paper']})  reactions per draw {entry['reactions_per_draw']}")
        for key, count in entry["pooled_reactions"].items():
            score = (entry.get("score") or {}).get(key)
            extra = (f"  matched {score['matched']}/{score['gold_reactions']}  recall {score['recall']}"
                     f"  precision {score['precision']}  rows claimed {score['rows_claimed']}") if score else ""
            lines.append(f"   {key:44s} pooled reactions {count:3d}{extra}")
        if spec["gold"]:
            lines.append(f"   ORCH-751 union (reference)                    matched {ORCH751[name]['matched']}/"
                         f"{ORCH751[name]['gold']}")
        lines.append("   tip rows (support | key | merges: draw->rule | kept apart because | cofactors withheld):")
        for row in entry["tip_rows"]:
            merges = ", ".join(f"d{m['draw']}->{'+'.join(m['bridged_by'])}" for m in row.get("merges", [])) or "-"
            apart = "; ".join(k["reason"] for k in row.get("kept_apart_from", [])) or "-"
            withheld = ", ".join(w["cofactor"] for w in row.get("cofactors_withheld", [])) or "-"
            lines.append(f"     {row['support']} | {row['key'][:70]} | {merges} | {apart} | {withheld}")
        lines.append("   names rewritten at the tip (draw: from -> to [rules]):")
        for r in entry["tip_names_rewritten"]:
            lines.append(f"     d{r['draw']}: {r['from']!r} -> {r['to']!r} {r['rules']}"
                         + (f" ({r['pattern']})" if r.get("pattern") else ""))
        lines.append("")

    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    text_out = "\n".join(lines) + "\n"
    Path(args.out).with_suffix(".txt").write_text(text_out, encoding="utf-8")
    sys.stdout.buffer.write(text_out.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

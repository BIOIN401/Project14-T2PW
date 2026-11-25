"""FIX-S2SCOPE offline replay (read-only over stored runs). usage: replay.py <tree> <out.json>

For every stored run with stage1_payload.json and merged_payload.json: the candidate Stage-2 additions are the
merged rows whose inputs/outputs (merge's own _reaction_io_key) match no in-scope Stage-1 row; that includes RAG
rows, so it over-counts (an upper bound). drop_stage_two_scope_repeats is applied with a neutral pathway context
(no subject escapes, so the most currency and the loosest match: again an upper bound) and, separately, with the
run's own scope when known. Gold: every dropped row scored alone with accscore (matched gold ids), and
final_mapped scored with and without its rows carrying a dropped signature (gold lost = the difference).
"""
import json
import sys
from pathlib import Path

tree, outp = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(Path(tree) / "src"))
sys.path.insert(0, "C:/t/acc-runs/scripts")
import accscore as S  # noqa: E402
from t2pw.pipeline.cofactor_policy import PathwayContext  # noqa: E402
from t2pw.pipeline.pipeline import _reaction_io_key  # noqa: E402
from t2pw.service.extraction import drop_stage_two_scope_repeats, step_signature  # noqa: E402

NEUTRAL = PathwayContext("requested pathway", "")
SCOPE = {"PMC12096016": PathwayContext("enterobactin biosynthesis", "Escherichia coli")}
ENTERO_TEXT = next(Path("C:/t/acc-runs/legs5g").glob("base1-PMC12096016/*/papers/PMC12096016/01_source_text.txt"))


def load(p):
    return json.load(open(p, encoding="utf-8"))


def runs():
    for s1 in sorted(Path("C:/t/acc-runs").glob("legs*/**/papers/*/strict/stage1_payload.json")):
        sd = s1.parent
        if (sd / "merged_payload.json").exists():
            tx = sd.parent / "01_source_text.txt"
            yield str(sd), sd.parent.name, sd, tx
    for i in range(1, 10):
        sd = Path(f"C:/t/t2pw-server-data/work/item-{i}/attempt-1/out")
        if (sd / "stage1_payload.json").exists() and (sd / "merged_payload.json").exists():
            yield str(sd), "PMC12096016", sd, ENTERO_TEXT


def rows(p):
    return [r for r in ((p.get("processes") or {}).get("reactions") or []) if isinstance(r, dict)]


def fsig(row):
    from t2pw.pipeline.cofactor_policy import COFACTOR_OR_CURRENCY, classify_entity
    from t2pw.pipeline.name_keys import strict_key
    return tuple(frozenset(strict_key(n) for n in (row.get(k) or []) if isinstance(n, str)
                           and classify_entity(n, NEUTRAL).verdict != COFACTOR_OR_CURRENCY) for k in ("inputs", "outputs"))


def has_gold(paper):
    return paper not in (S.NEG, S.SOUR) and (paper == S.CHL or paper in S.cases())


res = {"runs": [], "scanned": 0, "with_out_of_scope": 0, "drops_neutral": 0, "drops_scope": 0,
       "gold_lost": 0, "gold_matching_drops": 0}
for name, paper, sd, tx in runs():
    res["scanned"] += 1
    s1, merged = load(sd / "stage1_payload.json"), load(sd / "merged_payload.json")
    oos = [r for r in rows(s1) if str(r.get("scope_membership") or "").strip().casefold() == "out_of_scope"]
    in_keys = {_reaction_io_key(r) for r in rows(s1) if r not in oos}
    cands = [r for r in rows(merged) if _reaction_io_key(r) not in in_keys]
    rec = {"run": name, "paper": paper, "stage1_out_of_scope": len(oos), "candidates": len(cands)}
    if oos:
        res["with_out_of_scope"] += 1
    env = {"additions": {"processes": {"reactions": cands}}}
    vr = load(sd / "reaction_verifier_report.json") if (sd / "reaction_verifier_report.json").exists() else None
    rec["verifier_report"] = vr is not None
    _, rep = drop_stage_two_scope_repeats(env, s1, NEUTRAL, vr)
    rec["verifier_removed_rows_ignored"] = rep["verifier_removed_rows_ignored"]
    rec["drops"] = [{"stage2": S.label(d["stage2_row"]), "stage1": S.label(d["stage1_row"]),
                     "stage1_index": d["stage1_index"], "match": d["match"],
                     "rag_lineage": any("rag" in json.dumps(x).casefold()
                                        for x in d["stage2_row"].get("provenance_lineage") or [])}
                    for d in rep["drops"]]
    res["drops_neutral"] += len(rep["drops"])
    if paper in SCOPE:
        _, rep2 = drop_stage_two_scope_repeats(env, s1, SCOPE[paper], vr)
        rec["drops_with_run_scope"] = len(rep2["drops"])
        res["drops_scope"] += len(rep2["drops"])
    if rep["drops"] and has_gold(paper):
        text = tx.read_text(encoding="utf-8", errors="replace") if tx.exists() else ""
        fake = {"processes": {"reactions": [{"inputs": d["stage2_row"].get("inputs"),
                                              "outputs": d["stage2_row"].get("outputs")} for d in rep["drops"]]}}
        rec["dropped_rows_matching_gold"] = S.score_payload(paper, fake, text, [], tree)["matched"]
        res["gold_matching_drops"] += len(rec["dropped_rows_matching_gold"])
        fm = sd / "final_mapped.json"
        if fm.exists() and text:
            full = load(fm)
            # final_mapped names are mapped ("Pyruvic acid"): match them with the evaluator's strict_key
            # (-ic acid == -ate), currency left out as in step_signature. Replay-side only.
            sigs = {fsig(d["stage2_row"]) for d in rep["drops"]}
            cut = json.loads(json.dumps(full))
            cut["processes"]["reactions"] = [r for r in rows(full) if fsig(r) not in sigs]
            lin = (lambda p: S.leg_lineage(tree, sd, p)) if paper not in (S.CHL,) else (lambda p: [])
            a = S.score_payload(paper, full, text, lin(full), tree)
            b = S.score_payload(paper, cut, text, lin(cut), tree)
            rec["final_rows_removed"] = len(rows(full)) - len(rows(cut))
            rec["final_matched_before"], rec["final_matched_after"] = a["matched"], b["matched"]
            rec["gold_lost"] = sorted(set(a["matched"]) - set(b["matched"]))
            rec["precision_before"], rec["precision_after"] = a["precision"], b["precision"]
            res["gold_lost"] += len(rec["gold_lost"])
    res["runs"].append(rec)

json.dump(res, open(outp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"scanned {res['scanned']} with_oos {res['with_out_of_scope']} drops(neutral) {res['drops_neutral']} "
      f"drops(run scope, enterobactin only) {res['drops_scope']} dropped-rows-matching-gold {res['gold_matching_drops']} "
      f"gold_lost {res['gold_lost']}")
for r in res["runs"]:
    for d in r["drops"]:
        print(f"DROP {r['paper']} {r['run']}\n     s2: {d['stage2']}\n     s1[{d['stage1_index']}]: {d['stage1']} rag={d['rag_lineage']}")
    if "gold_lost" in r:
        print(f"GOLD {r['run']} matching={r.get('dropped_rows_matching_gold')} lost={r['gold_lost']} "
              f"rows_removed={r['final_rows_removed']} prec {r['precision_before']} -> {r['precision_after']}")
print("REPLAY_DONE")

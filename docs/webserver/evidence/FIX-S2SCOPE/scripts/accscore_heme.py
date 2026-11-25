"""Accuracy-lane scorer (offline, no model calls). Import with PYTHONPATH=<tree>/src.

score_payload(paper, payload, text, lineage) -> {recall, precision, matched:[gold ids], rows, tp_rows, fps:[...]}
  eval_v2 cases: eval_v2 score_leg TABLE 1 (the autoscorer's own numbers).
  PMC5130987 (chlorophyll): vision.bench score_view, PAPER_ONLY, gold PW766996 with pheophytin (R2146856) and
    phytylation (R2146854) flagged in_paper=false (strategy: 12 paper-stated steps). precision = match rate.
  PMC13231680 (negative control): recall None, rows = reaction count (target 0).
leg_lineage(tree, leg_dir, payload) -> lineage records for final_mapped (eval_v2 external classification).
"""
import copy
import json
from pathlib import Path

CHL = "PMC5130987"
NEG = "PMC13231680"
SOUR = "PMC7059786"
CHL_NOT_IN_PAPER = {"PW_R2146856", "PW_R2146854"}
_cases = None
_chl = None


def cases():
    global _cases
    if _cases is None:
        from t2pw.bench.eval_v2.gold import load_supplement
        _cases = load_supplement()
    return _cases


def chl_gold(tree):
    global _chl
    if _chl is None:
        g = json.load(open(Path(tree) / "docs/vision/evidence/V9/gold_PW766996.json", encoding="utf-8"))
        g = copy.deepcopy(g)
        for r in g["reactions"]:
            if r["id"] in CHL_NOT_IN_PAPER:
                r["in_paper"] = False
            # FIX-S2SCOPE review F1 (copy, not accscore.py): PW_R2146849 ferrochelatase, Protoporphyrin IX +
            # Fe2+ -> H+ + Heme (CHEBI:17627). Runs name the product "protoheme" (protoheme IX = heme b) and
            # Stage-2 cleaning splits "Fe2+" into "Fe 2"; neither is a gold alias, so the original misses it.
            if r["id"] == "PW_R2146849":
                for x in r["inputs"] + r["outputs"]:
                    if x.get("name") == "Heme":
                        x["aliases"] = list(x.get("aliases") or []) + ["Protoheme", "Protoheme IX"]
                    if x.get("name") == "Fe2+":
                        x["aliases"] = list(x.get("aliases") or []) + ["Fe 2", "Fe2", "Fe(2+)", "ferrous iron"]
        _chl = g
    return _chl


def rows_of(payload):
    return [r for r in ((payload.get("processes") or {}).get("reactions") or []) if isinstance(r, dict)]


def label(row):
    from t2pw.bench.eval_v2 import matcher as M
    return " + ".join(M.participant_names(row.get("inputs"))) + " -> " + " + ".join(M.participant_names(row.get("outputs")))


def leg_lineage(tree, leg_dir, payload):
    from t2pw.bench.eval_v2 import reactions as R
    rd = R.load_lineage(Path(tree))
    idx = rd.entity_provenance_index(payload)
    legd = {"leg_dir": str(leg_dir), "run": str(Path(leg_dir).parents[2]), "population": "g3",
            "payload_file": "final_mapped.json", "target_paper": Path(leg_dir).parent.name}
    return [rd.build_record(r, legd, i, idx, {}) for i, r in enumerate(rows_of(payload))]


def score_payload(paper, payload, text, lineage=(), tree=None):
    rows = rows_of(payload)
    if paper == SOUR:  # PW739181 has 0 reactions and 1 curated interaction: count rows, score the interaction
        from t2pw.vision.bench import score_view
        from t2pw.vision.payload import build_big_payload
        from t2pw.vision.view import PAPER_ONLY
        g = json.load(open(Path(tree) / "docs/vision/evidence/V9/gold_PW739181.json", encoding="utf-8"))
        ix = score_view(build_big_payload(copy.deepcopy(payload)), PAPER_ONLY, g)["types"]["interactions"]
        return {"recall": None, "precision": None, "matched": [], "rows": len(rows), "tp_rows": 0, "fps": [],
                "n_gold": 0, "gold_interactions": ix["gold"], "gold_interactions_recalled": ix["recalled"],
                "gold_interactions_weak": ix.get("recalled_weak")}
    if paper == NEG:
        return {"recall": None, "precision": None, "matched": [], "rows": len(rows), "tp_rows": 0,
                "fps": [{"label": label(r), "cls": "negative_control"} for r in rows], "n_gold": 0}
    if paper == CHL:
        from t2pw.vision.bench import score_view
        from t2pw.vision.payload import build_big_payload
        from t2pw.vision.view import PAPER_ONLY
        import t2pw.vision.bench as B
        gold = chl_gold(tree)
        big = build_big_payload(copy.deepcopy(payload))
        s = score_view(big, PAPER_ONLY, gold)["types"]["reactions"]
        # which gold ids: recompute hits the same way score_view does
        from t2pw.vision.view import view
        vw = view(big, PAPER_ONLY)
        ctx = B._Ctx(vw)
        golds = B._gold_elems(gold, "reactions", PAPER_ONLY)
        from t2pw.bench.eval_v2.gold import Reaction
        objs = {g["id"]: Reaction(g["id"], g.get("label") or "", B.scoring_terms(g["inputs"], True),
                                  B.scoring_terms(g["outputs"], True), None, bool(g.get("reversible")), "") for g in golds}
        matched, fps = set(), []
        for r in rows:
            hit = [g["id"] for g in golds if B._m_reaction(g, objs, r, ctx)]
            matched |= set(hit)
            if not hit:
                fps.append({"label": label(r), "cls": "no_gold_match"})
        return {"recall": s["recall"], "precision": s.get("match_rate"), "matched": sorted(matched),
                "rows": s["predicted"], "tp_rows": s["correct"], "fps": fps, "n_gold": s["gold"],
                "check_recalled": s["recalled"]}
    from t2pw.bench.eval_v2 import reactions as R
    case = cases()[paper]
    m = R.score_leg(case, payload, list(lineage), text, expected_species=case.species)
    t1 = m["table1_paper_extraction"]
    fps = []
    for rec, row in zip(m["rows"], rows):
        if rec["duplicate_of"] is None and rec["table1"] and rec["class"] != R.TP:
            fps.append({"label": label(row), "cls": rec["class"], "evidence": rec["evidence"]})
    return {"recall": t1["recall"], "precision": t1["precision"], "matched": sorted(t1["gold_reactions_matched"]),
            "rows": t1["rows_claimed_as_paper_extraction"], "tp_rows": t1["true_positive_rows"], "fps": fps,
            "n_gold": len(case.reactions)}


def interactions_of(payload):
    out = []
    for r in ((payload.get("processes") or {}).get("interactions") or []):
        if isinstance(r, dict):
            out.append({k: r.get(k) for k in ("entity_1", "entity_2", "relationship", "name", "evidence")})
    return out


def leg_paths(leg_root):
    """leg_root = C:/t/acc-runs/legs/<LEG> -> (paper, strict dir, text) of its newest run."""
    runs = sorted(p for p in Path(leg_root).glob("*") if p.is_dir() and (p / "papers").is_dir())
    if not runs:
        return None
    pdir = next((p for p in (runs[-1] / "papers").iterdir() if p.is_dir()), None)
    if pdir is None:
        return None
    sd = pdir / "strict"
    tx = pdir / "01_source_text.txt"
    text = tx.read_text(encoding="utf-8", errors="replace") if tx.exists() else None
    return pdir.name, sd, text

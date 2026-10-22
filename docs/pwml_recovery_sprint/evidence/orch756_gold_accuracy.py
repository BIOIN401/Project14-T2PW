"""ORCH-756: score the two PWML-writing legs against the certified paper gold, at three levels.

Offline only: no model, no network, no pipeline run. The instrument is the evaluator own
``reactions.score_leg``, called exactly as ``scripts/score_prompt_grid.py`` and
``orch754a_ensemble_merge.py`` call it (ORCH-751 section 2.1):
``score_leg(case, payload, [], text, expected_species=case.species)``.

Levels per paper:
  1 stage1   ``stage1_payload.json``  what the LLM extracted (scored as-is)
  2 final    ``final_mapped.json``    what the pipeline kept (scored as-is)
  3 pwml     ``pwml_ir.json``         the reactions drawn in the PWML

Input-loading adaptation, level 3 only (no matching rule is changed): IR reactions name
participants by ``entity_key``. Each ``left``/``right``/``enzymes`` key is replaced by that
IR entity ``name``; a protein complex ``components`` (``protein_key``) by the component
protein name and ``gene_name``; a biological state ``species_key`` by the species name;
and since IR reactions carry no evidence, the evidence string of the ONE ``final_mapped.json``
reaction with the same name (none if zero or several). Each IR reaction is cross-checked
against the ``.pwml`` itself (left/right ``element-id`` resolved through the PWML own
``<compound>`` table, compared by ``matcher.strict_key``); only IR rows found in the PWML
are scored at level 3.

Secondary column, level 2 only: ``report.evaluate`` (unlike the two scripts above) feeds
``score_leg`` R-D092-1 lineage records. They are built here the same way, with
``admission_index`` given this leg own ``rag_admission_report.json`` instead of the
committed corpus. It changes precision only when a row with external lineage matches no gold.

Identifier checks: ``reactions.id_correctness`` against ``build_reference_index`` (the three
reference tables as committed at HEAD), levels 2 and 3. Interactions are counted, not scored
(ORCH-752 section 1.2).

A leg that wrote NO PWML (ORCH-759) is still scored: the newest leg with a
``stage1_payload.json`` is taken, level 3 is ``pwml_ir.json`` marked "IR, not exported" if it
exists and "not written" otherwise. With a Stage-1 ensemble report (N>1 draws), each Stage-1
row is annotated with the draws that found it.

Usage::

    PYTHONPATH=src python -u docs/pwml_recovery_sprint/evidence/orch756_gold_accuracy.py
    PYTHONPATH=src python -u docs/pwml_recovery_sprint/evidence/orch756_gold_accuracy.py --task ORCH-759 \
        --legs runs_papers/ORCH-759/legs --review runs_papers/ORCH-759/REVIEW --papers enterobactin \
        --out docs/pwml_recovery_sprint/evidence/orch759_gold_accuracy.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from t2pw.bench.eval_v2 import matcher as M  # noqa: E402
from t2pw.bench.eval_v2 import reactions as R  # noqa: E402
from t2pw.bench.eval_v2.gold import load_fulltext, load_supplement  # noqa: E402

LEGS = REPO / ".claude/worktrees/orch756/runs_papers/ORCH-756/legs"
REVIEW = REPO / "runs_papers/ORCH-756/REVIEW"
FULLTEXT = Path("C:/t/eval-fulltext")
OUT_TXT = HERE / "orch756_gold_accuracy.txt"
PAPERS = {"enterobactin": "PMC12096016", "menaquinone": "PMC12421875"}
BASELINES = {
    "enterobactin": "ORCH-747 round 4 per-draw recall (ORCH-747-RESULT.md section 12): control 5,5,4,5,5 /5; "
                    "ORCH-751 3-draw union (ORCH-751-RESULT.md section 2.1): 4/5 (per draw 4,4,4)",
    "menaquinone": "ORCH-747 round 4 per-draw recall (ORCH-747-RESULT.md section 12): 4,7,4,9,7 /9; "
                   "ORCH-751 3-draw union (ORCH-751-RESULT.md section 2.1): 9/9 (per draw 2,9,7)",
}
EXTRA_BASELINE = {
    "ORCH-759": "; ORCH-756 1-draw leg (orch756_gold_accuracy.txt): stage1 5/5 P 1.000 F1 1.000, "
                "final 5/5 P 0.714 R 1.000 F1 0.833 (with lineage P 0.833 F1 0.909), PWML 5/5 P 0.714",
}
#: The reader reading of each unmatched row (ORCH-756 test-runner, by hand against the
#: paper text). NOT evaluator output. Keyed by (paper, level, row name).
READINGS: Dict[tuple, str] = {}


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def pick_leg(legs: Path, name: str, pid: str) -> tuple:
    """Newest leg with a PWML; failing that, newest leg with a Stage-1 payload."""
    cands = sorted(legs.glob(f"{name}*/*/papers/{pid}/strict"), key=lambda p: p.parents[2].name)
    with_pwml = [c for c in cands if any(c.glob("*.pwml"))]
    if with_pwml:
        return with_pwml[-1], cands, True
    with_s1 = [c for c in cands if (c / "stage1_payload.json").is_file()]
    return (with_s1[-1] if with_s1 else None), cands, False


def ensemble_support(path: Path) -> Optional[Dict[str, str]]:
    """Pooled Stage-1 reaction name -> its draw support, from stage1_ensemble_report.json."""
    if not path.is_file():
        return None
    doc = _load(path)
    doc = doc[0] if isinstance(doc, list) and doc else doc
    rows = ((doc.get("buckets") or {}).get("/processes/reactions") or {}).get("rows") or []
    n = doc.get("draws_succeeded")
    # ponytail: keyed by row name; two pooled rows sharing a name would collide (none here).
    return {r.get("name"): f"support {len(r.get("draws") or [])}/{n} draws {r.get("draws")}" for r in rows}
_ENT_LDH = ("Real paper statement, but it is the coupled-assay reporter (gold O2, off-pathway LDH assay), "
            "not enterobactin biosynthesis. Added at Stage 2: present in merged_payload.json, absent from "
            "stage1_payload.json, no provenance_lineage. Drawn in the PWML.")
_ENT_DUP = ("A duplicate of gold R3 (EntA), not a new reaction. RAG-transcribed from ANOTHER paper, PMC11258883 "
            "(rag_provenance chunk 83dab4d0..., admitted for gap-dangling_reaction-bbd16481), so its quote is "
            "correctly not in the target paper. The strict matcher has no gold alias for the substrate spelling "
            "2,3-dihydro-DHB (U+2010 hyphens), hence unsupported rather than a second R3 row. It also puts a second "
            "substrate node (2,3-dihydro-DHB beside 2,3-diDHB) into the PWML. With R-D092-1 lineage it is "
            "indeterminate_external and leaves the precision denominator.")
_MEN_MENA = ("Gold R8 itself, damaged downstream. stage1_payload.json and merged_payload.json carry the paper name "
             "prenyl diphosphate; in final_mapped.json the compound is renamed Dimethylallylpyrophosphate "
             "(raw_name prenyl diphosphate, mapping_meta.chosen_rule novel_compound). DMAPP is the C5 prenyl "
             "diphosphate; the paper names only the class (gold R8 quote: MenA joins DHNA and prenyl diphosphate "
             "to produce demethylmenaquinone (DMK)). A renaming after Stage 1, not an extraction error; the PWML "
             "draws DMAPP.")
READINGS.update({
    ("PMC12096016", "final", "LDH-catalyzed reduction of pyruvate to lactate"): _ENT_LDH,
    ("PMC12096016", "pwml", "LDH-catalyzed reduction of pyruvate to lactate"): _ENT_LDH,
    ("PMC12096016", "final", "oxidation of 2,3-dihydro-DHB to 2,3-DHB"): _ENT_DUP,
    ("PMC12096016", "pwml", "oxidation of 2,3-dihydro-DHB to 2,3-DHB"): _ENT_DUP,
    ("PMC12421875", "stage1", "MenF reaction"): (
        "Duplicate of row 0: the same MenF step re-extracted from the kinetic-model term k2[MenF][A] "
        "(dump_only evidence). Collapsed by the evaluator; later quarantined as duplicate_locked_reaction."),
    ("PMC12421875", "stage1", "MenD reaction"): (
        "Duplicate of row 1, from the model term kcatD[MenD][B]. Quarantined as duplicate_locked_reaction "
        "(rxn_lock_005); the original MenD row (rxn_lock_004) was then ALSO quarantined, see the notes."),
    ("PMC12421875", "final", "MenA reaction"): _MEN_MENA,
    ("PMC12421875", "pwml", "MenA reaction"): _MEN_MENA,
})
#: Reader notes per paper (ORCH-756 test-runner, from the leg artifacts). NOT evaluator output.
NOTES: Dict[str, List[str]] = {
    "PMC12096016": [
        "Stage 1 extracted exactly the 5 gold reactions, one enzyme each, every quote verbatim in the paper.",
        "Final/PWML rows carry extra catalysts that Stage 1 did not: EntC row also lists isochorismatase, "
        "EntA row lists isochorismatase and EntCB complex, EntF row lists EntE complex. The evaluator scores "
        "an enzyme correct when ANY listed enzyme matches, so these extras are not penalised.",
        "Interactions: 6 in Stage 1, 7 in final and IR; all 7 are withheld from the PWML (untyped_interaction, "
        "pwml_ir_report.json), so the PWML draws none.",
        "Identifier check flags EntCB complex (pathbank_complex_id 1143, record name Isochorismate synthase) at "
        "the IR level only, resting on the mapper cache; the final_mapped.json entity is not flagged.",
    ],
    "PMC12421875": [
        "Stage 1 extracted all 9 gold reactions (11 rows; 2 duplicates from the kinetic model collapsed).",
        "R2 MenD was LOST between Stage 1 and final: final_mapped.json quarantined_locked_reactions holds "
        "rxn_lock_005 (duplicate_locked_reaction) AND rxn_lock_004, the paper-quoted original, with reason "
        "protein_missing_external_identity, missing_entities [MenD], quarantine_stage "
        "pre_export_strict_quarantine. No MenD row survives to the PWML.",
        "R8 MenA was damaged by renaming prenyl diphosphate to Dimethylallylpyrophosphate (see the row reading).",
        "MenF reverse reaction (isochorismate -> chorismate) was added at Stage 2 from the paper (quote: MenF is "
        "also known to be capable of catalyzing both forward and reverse conversion of chorismate to "
        "isochorismate). Gold R1 is reversible, so the evaluator counts it as a second true-positive row for R1: "
        "it raises precision and adds no recall.",
        "Most Stage-1 quotes are not_in_paper because they elide the paper letter labels, e.g. the paper reads "
        "converting chorismate (A) to isochorismate (B) and the quote drops (A)/(B); a quote fidelity note, "
        "not a different source text.",
    ],
}
#: ORCH-759 (3-draw enterobactin leg) readings and notes, by the same rule: NOT evaluator output.
READINGS_759: Dict[tuple, str] = {}
NOTES_759: Dict[str, List[str]] = {
    "PMC12096016": [
        "Draw support of the matched gold reactions (stage1_ensemble_report): R1 EntC, R2 EntB, R3 EntA, R4 EntE "
        "each 3/3. R5 EntF is carried by TWO pooled rows: 2,3-DHB + L-serine -> enterobactin (2/3, draws 2 and 3) "
        "and 2,3-dihydroxybenzoyl-AMP + L-serine -> enterobactin (1/3, draw 1 only, flagged REVIEW by the "
        "ensemble report).",
        "The one single-draw row is NOT an extra reaction: it is a second rendering of R5, with the AMP ester as "
        "substrate. Gold R5 accepts that ester (its note: a row that feeds 2,3-DHB or its AMP ester straight into "
        "EntF represents the paper own chemistry and matches), so both rows are true positives and precision "
        "stays 1.000. Pooling kept them apart because their input keys differ; both survive to final_mapped.json.",
        "Unlike the ORCH-756 1-draw leg, final_mapped.json has no LDH assay row and no RAG-transcribed EntA "
        "duplicate: 6 rows, all target_paper_supported.",
        "Final rows carry catalysts Stage 1 did not; EntD complex appears on the EntB and on one EntF row. EntD is "
        "the phosphopantetheinyl transferase (gold excludes it as a protein-modification statement), not a "
        "catalyst of these conversions. Not penalised: the evaluator credits an enzyme when ANY listed one matches.",
        "No PWML and no pwml_ir.json were written (leg failed final_pre_export_stage3_gates); level 3 is not scored.",
    ],
}
READINGS_BY_TASK = {"ORCH-756": READINGS, "ORCH-759": READINGS_759}
NOTES_BY_TASK = {"ORCH-756": NOTES, "ORCH-759": NOTES_759}


def pwml_equations(path: Path) -> List[tuple]:
    root = ET.parse(path).getroot()
    names = {}
    for c in root.iter("compound"):
        i, n = c.find("id"), c.find("name")
        if i is not None and n is not None and i.text:
            names[i.text.strip()] = (n.text or "").strip()

    def side(rx, tag):
        out = []
        for el in rx.iter(tag):
            t, e = el.findtext("element-type"), el.findtext("element-id")
            out.append(names.get((e or "").strip(), f"<{t} {e} unresolved>"))
        return frozenset(M.strict_key(n) for n in out), out

    return [(side(rx, "reaction-left-element"), side(rx, "reaction-right-element")) for rx in root.iter("reaction")]


def ir_payload(ir: Dict[str, Any], final: Dict[str, Any], pwml: Optional[List[tuple]]) -> tuple:
    """IR rows as a payload. ``pwml`` None (no PWML written): every IR row is kept, unfiltered."""
    ents = {e["key"]: e for b in (ir.get("entities") or {}).values() if isinstance(b, list)
            for e in b if isinstance(e, dict) and e.get("key")}
    species = {s["key"]: s.get("name") for s in ir.get("species") or [] if isinstance(s, dict)}
    states = {b["key"]: b for b in ir.get("biological_states") or [] if isinstance(b, dict)}
    ev: Dict[str, List[Any]] = {}
    for r in (final.get("processes") or {}).get("reactions") or []:
        ev.setdefault(r.get("name"), []).append(r.get("evidence"))

    def names(slot):
        return [{"name": ents[p["entity_key"]]["name"]} for p in slot or [] if p.get("entity_key") in ents]

    pw_keys = Counter((l[0], r[0]) for l, r in pwml or [])
    rows, check = [], []
    for r in (ir.get("processes") or {}).get("reactions") or []:
        ins, outs = names(r.get("left")), names(r.get("right"))
        key = (frozenset(M.strict_key(x["name"]) for x in ins), frozenset(M.strict_key(x["name"]) for x in outs))
        in_pwml = None
        if pwml is not None:
            in_pwml = pw_keys[key] > 0
            if in_pwml:
                pw_keys[key] -= 1
        check.append((r.get("key"), r.get("name"), in_pwml))
        if in_pwml is False:
            continue
        e = ev.get(r.get("name")) or []
        rows.append({"name": r.get("name"), "inputs": ins, "outputs": outs, "enzymes": names(r.get("enzymes")),
                     "biological_state": (states.get(r.get("biological_state_key")) or {}).get("name"),
                     "evidence": e[0] if len(e) == 1 else None})
    complexes = []
    for c in (ir.get("entities") or {}).get("protein_complexes") or []:
        comps = [ents[x["protein_key"]] for x in c.get("components") or [] if x.get("protein_key") in ents]
        complexes.append({**c, "components": [{"name": p.get("name"),
                                               "gene_name": (p.get("mapped_ids") or {}).get("gene_name")} for p in comps]})
    payload = {"entities": {**ir["entities"], "protein_complexes": complexes},
               "biological_states": [{"name": b.get("name"), "species": species.get(b.get("species_key"))}
                                     for b in states.values()],
               "processes": {"reactions": rows, "interactions": (ir.get("processes") or {}).get("interactions") or []}}
    return payload, check, sum(pw_keys.values())


def lineage(payload: Dict[str, Any], leg: Path, pid: str) -> List[Dict[str, Any]]:
    rd = R.load_lineage(REPO)
    rel = leg.relative_to(REPO).as_posix()
    adm = rd.admission_index([f"{rel}/rag_admission_report.json"], REPO, [])
    rec = {"leg_dir": rel, "target_paper": pid, "run": rel.split("/papers/")[0],
           "payload_file": "final_mapped.json", "population": "canonical"}
    idx = rd.entity_provenance_index(payload)
    rows = [r for r in (payload.get("processes") or {}).get("reactions") or [] if isinstance(r, dict)]
    return [rd.build_record(r, rec, i, idx, adm) for i, r in enumerate(rows)]


def eq(ins, outs) -> str:
    return " + ".join(ins) + " -> " + " + ".join(outs)


def gold_eq(g) -> str:
    def side(ts):
        return " + ".join(t.name + ("" if t.role == "defining" else " [cof]") for t in ts)
    arrow = "<->" if g.reversible else "->"
    return f"{side(g.inputs)} {arrow} {side(g.outputs)}"


def evidence_of(payload, i) -> str:
    rows = [r for r in (payload.get("processes") or {}).get("reactions") or [] if isinstance(r, dict)]
    spans = R._evidence_spans(rows[i]) if i < len(rows) else []
    return " | ".join(spans) if spans else "(no evidence string)"


def fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO / p


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--task", default="ORCH-756")
    ap.add_argument("--legs", default=str(LEGS))
    ap.add_argument("--review", default=str(REVIEW))
    ap.add_argument("--out", default=str(OUT_TXT))
    ap.add_argument("--papers", default=",".join(PAPERS), help="comma list of condition names")
    args = ap.parse_args()
    legs_root, review, out_txt = _path(args.legs), _path(args.review), _path(args.out)
    readings = READINGS_BY_TASK.get(args.task, {})
    notes = NOTES_BY_TASK.get(args.task, {})
    cases = load_supplement()
    ref = R.build_reference_index(REPO)
    header = ("ORCH-756 gold accuracy: the two PWML-writing legs, three levels, eval-v2 score_leg"
              if args.task == "ORCH-756" else
              f"{args.task} gold accuracy: legs under {legs_root.relative_to(REPO)}, three levels, eval-v2 score_leg")
    out: List[str] = [header, f"reference tables (committed at HEAD): {ref["tables"]}", ""]
    summary: List[str] = []
    for name in args.papers.split(","):
        pid = PAPERS[name]
        case = cases[pid]
        assert case.complete, f"{pid} gold is not certified complete"
        text = load_fulltext(case, FULLTEXT)
        assert text is not None, f"{FULLTEXT}/{pid}.txt missing"
        leg, cands, has_pwml = pick_leg(legs_root, name, pid)
        assert leg is not None, f"no scorable leg for {name}"
        pw = sorted(leg.glob("*.pwml"))[0] if has_pwml else None
        final = _load(leg / "final_mapped.json")
        ens_path = review / name / "strict" / "stage1_ensemble_report.json"
        support = ensemble_support(ens_path)
        levels = {"1 stage1 (stage1_payload.json)": _load(leg / "stage1_payload.json"),
                  "2 final (final_mapped.json)": final}
        check, leftover, withheld = [], 0, 0
        if (leg / "pwml_ir.json").is_file():
            ir = _load(leg / "pwml_ir.json")
            ir_pl, check, leftover = ir_payload(ir, final, pwml_equations(pw) if pw else None)
            ir_pl["processes"]["transports"] = (ir.get("processes") or {}).get("transports") or []
            rep = leg / "pwml_ir_report.json"
            withheld = sum(1 for w in (_load(rep).get("warnings") or [] if rep.is_file() else [])
                           if w.get("code") == "interaction_withheld_from_pwml")
            label3 = ("3 pwml (pwml_ir.json rows drawn in the PWML)" if pw
                      else "3 ir (pwml_ir.json, IR, not exported: no PWML written)")
            levels[label3] = ir_pl
        rel_leg = leg.relative_to(REPO)
        pw_line = f"PWML: {pw.name}" if pw else "PWML: none written"
        P: List[str] = [f"=== {name} {pid}  gold: {len(case.reactions)} certified in-scope reactions "
                        f"(+{len(case.off_pathway)} off-pathway, {len(case.summaries)} summaries)",
                        f"leg candidates: {[str(c.relative_to(REPO)) for c in cands]}",
                        (f"scored leg (newest with a PWML): {rel_leg}" if pw
                         else f"scored leg (newest with a Stage-1 payload; NO PWML written): {rel_leg}"),
                        f"{pw_line}; paper text {FULLTEXT}/{pid}.txt sha256 matches gold text_sha256 "
                        f"{case.source["text_sha256"][:16]}"]
        if pw:
            P.append(f"PWML cross-check: IR rows found in the PWML {sum(bool(c[2]) for c in check)}/{len(check)}; "
                     f"PWML reactions with no IR row {leftover}")
            for k, n, ok in check:
                P.append(f"   IR {k} {"in PWML" if ok else "NOT IN PWML"}: {n}")
        P.append(f"baseline: {BASELINES[name]}{EXTRA_BASELINE.get(args.task, "")}")
        if support is not None:
            P.append(f"Stage-1 ensemble report: {ens_path.relative_to(REPO)} ({len(support)} pooled reactions)")
        P.append("")
        for label, payload in levels.items():
            lvl = label.split()[1]
            m = R.score_leg(case, payload, [], text, expected_species=case.species)
            t1 = m["table1_paper_extraction"]
            proc = payload.get("processes") or {}
            n_int = len(proc.get("interactions") or [])
            drawn = f", {n_int - withheld} drawn / {withheld} withheld from the PWML" if lvl == "pwml" else ""
            line = (f"  {name:12s} {lvl:7s} P {fmt(t1["precision"])}  R {fmt(t1["recall"])}  F1 {fmt(t1["f1"])}  "
                    f"matched {len(t1["gold_reactions_matched"])}/{t1["gold_reactions"]}  "
                    f"predicted {t1["rows_claimed_as_paper_extraction"]} (rows {m["retained_reactions"]}, "
                    f"dups collapsed {m["duplicate_rows_collapsed"]})  interactions {n_int}{drawn} (not scored)")
            summary.append(line)
            P += [f"--- level {label}", line.strip(),
                  f"row classes: {m["row_classes"]}",
                  f"transports (not scored): {len(proc.get("transports") or [])}"]
            if lvl == "final":
                ml = R.score_leg(case, payload, lineage(payload, leg, pid), text, expected_species=case.species)
                l1 = ml["table1_paper_extraction"]
                lin_line = (f"with R-D092-1 lineage (as report.evaluate): P {fmt(l1["precision"])}  "
                            f"R {fmt(l1["recall"])}  F1 {fmt(l1["f1"])}  "
                            f"matched {len(l1["gold_reactions_matched"])}/{l1["gold_reactions"]}  "
                            f"predicted {l1["rows_claimed_as_paper_extraction"]}  classes {ml["row_classes"]}  "
                            f"lineage per row {[r["lineage_class"] for r in ml["rows"]]}")
                P.append(lin_line)
                summary.append(" " * 23 + lin_line)
            rows = m["rows"]

            def sup(r):
                return f"  [{support.get(r["name"], "support unknown")}]" if (lvl == "stage1" and support) else ""

            P.append("gold reactions:")
            for g in case.reactions:
                hits = [r for r in rows if r["class"] == R.TP and g.id in r["gold_ids"]]
                near = [r for r in rows if r["class"] in (R.PARTIAL, R.CONFLICT) and g.id in r["gold_ids"]]
                enz = f" [{g.enzyme.name}]" if g.enzyme else ""
                flag = "MATCHED" if hits else "missed "
                P.append(f"  {g.id} {flag}  {g.label}{enz}: {gold_eq(g)}")
                for r in hits:
                    dup = f"  DUPLICATE of {r["duplicate_of"]}" if r["duplicate_of"] is not None else ""
                    P.append(f"      by row {r["row_index"]} {repr(r["name"])}: {eq(r["inputs"], r["outputs"])}"
                             f"  enzymes {r["enzymes"]} enzyme_verdict {r.get("enzyme_verdict")}{dup}{sup(r)}")
                for r in near:
                    P.append(f"      near ({r["class"]}) row {r["row_index"]} {repr(r["name"])}: "
                             f"{eq(r["inputs"], r["outputs"])}  missing_defining {r.get("missing_defining")}{sup(r)}")
            extras = [r for r in rows if not (r["class"] == R.TP and r["duplicate_of"] is None)]
            P.append(f"unmatched / non-TP rows ({len(extras)}):")
            for r in extras:
                tag = f"DUPLICATE of row {r["duplicate_of"]}" if r["duplicate_of"] is not None else r["class"]
                gid = f" gold_ids {r["gold_ids"]}" if r["gold_ids"] else ""
                P.append(f"  row {r["row_index"]} [{tag}]{gid} {repr(r["name"])}: {eq(r["inputs"], r["outputs"])}"
                         f"  enzymes {r["enzymes"]}  evidence {r["evidence"]}{sup(r)}")
                if r.get("missing_defining"):
                    P.append(f"      missing_defining {r["missing_defining"]}")
                P.append(f"      quote: {evidence_of(payload, r["row_index"])[:700]}")
                reading = readings.get((pid, lvl, r["name"]))
                if reading:
                    P.append(f"      READER READING (not the evaluator): {reading}")
            P.append(f"name conflicts for hand review: {m["name_conflicts_for_hand_review"] or "none"}")
            P.append(f"minor cofactors missing on matched reactions: "
                     f"{m["minor_cofactors_missing_on_matched_reactions"] or "none"}")
            ea = m["entity_accuracy"]
            P.append(f"entity accuracy: substrate {ea["substrate_hits"]}/{ea["substrate_total"]}  product "
                     f"{ea["product_hits"]}/{ea["product_total"]}  enzyme {fmt(ea["enzyme_accuracy"])} "
                     f"{ea["enzyme_verdicts"]}  species {ea["species_verdicts"]}")
            P.append(f"evidence verdicts (vs pinned gold text): {m["evidence"]["verdicts"]}")
            om = m["old_lenient_matcher_same_rows"]
            P.append(f"old lenient matcher, same rows: recall {fmt(om["recall"])}  disagreements "
                     f"{om["rows_where_old_and_v2_disagree"] or "none"}")
            if lvl != "stage1":
                idc = R.id_correctness(payload, ref)
                P.append(f"identifier checks: {idc["entities"]} entities {idc["counts"]}  resting on mapper cache only "
                         f"{idc["counts_resting_on_mapper_cache_only"]}")
                for w in idc["wrong"]:
                    refs = [p.get("reference_names") for p in w.get("per_id", []) if p["verdict"] == "wrong"]
                    P.append(f"      wrong_identity {w["bucket"]} {repr(w["name"])} {w.get("contradicting_ids")} "
                             f"independent={w.get("independent")} ref={refs}")
            P.append("")
        if not any(k.startswith("3 ") for k in levels):
            P += ["--- level 3 pwml: NOT WRITTEN (no PWML and no pwml_ir.json in the leg)", ""]
            summary.append(f"  {name:12s} {"pwml":7s} NOT WRITTEN (no PWML, no pwml_ir.json)")
        P.append(f"READER NOTES ({args.task} test-runner reading of the leg artifacts, not evaluator output):")
        P += [f"  - {n}" for n in notes.get(pid, [])]
        P.append("")
        out += P
        (review / name / "strict" / "gold_comparison.txt").write_text("\n".join(P) + "\n", encoding="utf-8")
    out = ["SUMMARY", *summary, "", *out]
    text_out = "\n".join(out) + "\n"
    out_txt.write_text(text_out, encoding="utf-8")
    sys.stdout.buffer.write(text_out.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

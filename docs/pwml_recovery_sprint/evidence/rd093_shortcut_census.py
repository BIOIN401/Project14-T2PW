"""ORCH-723 F-179 census -- does the `glycine -> heme` failure mechanism recur?

EVALUATION-ONLY and READ-ONLY. Re-runs nothing. D-093 section 3 forbids recreating
information already on disk, and this module obeys that literally: it reads archived
payloads, gold and release reports, and it never launches a pipeline leg. **No
T-107/T-108/T-109 re-run and no redraw of any kind** -- every number is a property of
artifacts that already existed before this module was written.

=============================================================================
THE MECHANISM UNDER INVESTIGATION
=============================================================================

On ``runs_verify/2026-09-02_2052/papers/PMC12180156/strict`` the delivered canonical
payload contains exactly one reaction, ``glycine -> heme``. Heme biosynthesis is an
EIGHT-step pathway; that row collapses all of it into a single step that no paper
states. The gold case is a ``context_only`` negative-ish control whose
``export_rationale`` reads "With zero heme-biosynthesis reactions recoverable, nothing
about heme biosynthesis is exportable", and its ``relevance_note`` records "Zero
heme-biosynthesis reactions have both sides named anywhere in the file". The leg
nevertheless produced a PWML and the runtime recorded ``semantic_evaluation: passed``.

The question this census exists to answer is NOT "is that row wrong" -- it is wrong.
It is **"is this one paper, or is it a mechanism?"** Those have different remedies: an
isolated limitation can be recorded and lived with under the D-090 freeze, while a
recurring product-contract violation needs a narrow product-owner unfreeze.

=============================================================================
FIVE CRITERIA, COUNTED SEPARATELY, NEVER SUMMED
=============================================================================

  C1 ``exact_glycine_heme``        the literal shortcut, inputs {glycine} outputs {heme}
  C2 ``only_identifier_mapping``   the row's OWN lineage names identifier_mapping and
                                   nothing else -- no stage ever claimed a paper
  C3 ``no_paper_and_no_rag``       no paper-stated origin AND no reaction-specific RAG
                                   evidence: nothing anywhere asserts this reaction
  C4 ``precursor_terminal_shortcut`` produces the pathway's TERMINAL product from an
                                   input gold never links directly to it
  C5 ``passed_gates``              of the above, how many the runtime released or
                                   marked semantically passed

**C2 HAS TWO READINGS AND BOTH ARE REPORTED**, because conflating them produced a wrong
statement during the diagnosis this census follows. ``own_lineage`` asks what the ROW's
own ``provenance_lineage`` says. ``inherited`` asks what its PARTICIPANTS' lineage says
when the row carries none. The `glycine -> heme` row carries NO lineage of its own; the
``identifier_mapping`` stage seen in its record is INHERITED from its participants. A
census reporting only the first would report 0 and miss the case entirely.

=============================================================================
C4's DEFINITION, AND WHY IT IS EVIDENCE-BASED RATHER THAN A GUESS
=============================================================================

A "shortcut" is defined against the COMMITTED GOLD SET, not against intuition:

  * the reaction matches NO gold ``supported_reactions`` signature (via the production
    matcher ``bench.semantic._signature_matches``), AND
  * one of its outputs matches the terminal product of the leg's requested pathway
    (``heme biosynthesis`` -> ``heme``, ``enterobactin biosynthesis`` -> ``enterobactin``,
    and so on -- taken from the pathway name itself, not invented), AND
  * no gold signature for that case links this reaction's input directly to that
    terminal product.

So C4 says: *the row produces the pathway's end product from something the paper's own
verified chemistry never connects to it.* That is checkable against committed data.
Where a case states no signatures at all, every terminal-product reaction qualifies --
correctly, because gold has recorded that nothing is exportable.

=============================================================================
POPULATIONS
=============================================================================

``committed`` (in ``git ls-files``) and ``preserved_untracked`` (on disk only) are
counted APART and never summed. F-178 cost a wave because a helper measured the working
tree while claiming to measure the committed corpus, and the 2026-09-02 run -- the one
that started this -- is untracked, so it MUST be visible here while being clearly
labelled as resting on a single-disk artifact.

Usage:
  python rd093_shortcut_census.py <repo-root> [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

COMMITTED = "committed"
PRESERVED = "preserved_untracked"
POPULATION_ORDER: Tuple[str, ...] = (COMMITTED, PRESERVED)

CRITERIA: Tuple[str, ...] = (
    "exact_glycine_heme",
    "only_identifier_mapping_own_lineage",
    "only_identifier_mapping_inherited",
    "no_paper_and_no_rag",
    "precursor_terminal_shortcut",
)

#: Terminal product token for a requested pathway. Derived from the pathway NAME, so it
#: cannot be tuned to flatter a result: "<X> biosynthesis" produces "<X>".
_PATHWAY_TAIL = re.compile(r"\s*(biosynthesis|biosynthetic pathway|synthesis|pathway)\s*$", re.I)


def terminal_product(requested_pathway: Any) -> str:
    if not isinstance(requested_pathway, str) or not requested_pathway.strip():
        return ""
    return _PATHWAY_TAIL.sub("", requested_pathway.strip()).strip().lower()


def _norm(v: Any) -> str:
    return str(v).strip().lower()


def _names(seq: Any) -> List[str]:
    out: List[str] = []
    for v in seq or []:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
        elif isinstance(v, dict):
            for k in ("name", "entity"):
                if isinstance(v.get(k), str) and v[k].strip():
                    out.append(v[k].strip())
                    break
    return out


def matches_terminal(outputs: Sequence[str], terminal: str) -> bool:
    """Whether any output IS the pathway's terminal product.

    Substring containment in the OUTPUT direction only, so ``heme`` matches ``heme``
    and ``heme b`` but ``heme`` does not match an output merely mentioning it inside a
    longer unrelated name; the reverse containment is deliberately not accepted.
    """

    if not terminal:
        return False
    for o in outputs:
        n = _norm(o)
        if n == terminal or n.startswith(terminal + " ") or n.endswith(" " + terminal):
            return True
    return False


def lineage_stage_sets(reaction: Dict[str, Any]) -> Tuple[set, bool]:
    """(stages named by the row's OWN lineage, whether it carries any)."""

    lin = reaction.get("provenance_lineage")
    if not isinstance(lin, list) or not lin:
        return set(), False
    return {e.get("stage") for e in lin if isinstance(e, dict) and e.get("stage")}, True


def has_paper_stated(reaction: Dict[str, Any]) -> bool:
    """Whether any stage typed this row as stated by the paper."""

    for e in reaction.get("provenance_lineage") or []:
        if not isinstance(e, dict):
            continue
        if e.get("origin") == "paper_stated" or e.get("paper_explicit") == "explicit":
            return True
    return False


def release_facts(leg_dir: str) -> Dict[str, Any]:
    """What the RUNTIME concluded for this leg. Reported, never recomputed."""

    out = {"release_status": "unavailable", "semantic_evaluation": "unavailable",
           "strict_gates_passed": None, "pwml_files": []}
    q = os.path.join(leg_dir, "quarantine_report.json")
    if os.path.isfile(q):
        try:
            rel = (json.load(open(q, encoding="utf-8")).get("release") or {})
            out["release_status"] = rel.get("status", "unavailable")
            out["semantic_evaluation"] = rel.get("semantic_evaluation", "unavailable")
            out["strict_gates_passed"] = rel.get("strict_gates_passed")
        except (OSError, ValueError):
            out["release_status"] = "artifact_malformed"
    out["pwml_files"] = sorted(os.path.basename(p)
                               for p in glob.glob(os.path.join(leg_dir, "*.pwml")))
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_root")
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from t2pw.bench.goldset import load_gold_set          # noqa: E402
    from t2pw.bench.semantic import _signature_matches    # noqa: E402

    cases = {c.paper_id: c for c in load_gold_set().cases}

    tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace").stdout.split())
    payloads = sorted({p.replace("\\", "/") for p in
                       glob.glob("runs/**/final_mapped.json", recursive=True) +
                       glob.glob("runs_verify/**/final_mapped.json", recursive=True)}
                      | {f for f in tracked if f.endswith("final_mapped.json")})

    hits: List[Dict[str, Any]] = []
    counts: Dict[str, collections.Counter] = {p: collections.Counter() for p in POPULATION_ORDER}
    rows_seen: Dict[str, int] = {p: 0 for p in POPULATION_ORDER}
    legs_seen: Dict[str, set] = {p: set() for p in POPULATION_ORDER}
    papers_hit: Dict[str, set] = collections.defaultdict(set)

    for rel_path in payloads:
        if not os.path.isfile(rel_path):
            continue
        pop = COMMITTED if rel_path in tracked else PRESERVED
        m = re.search(r"/papers/(PMC\d+)", rel_path)
        paper = m.group(1) if m else None
        case = cases.get(paper or "")
        leg_dir = os.path.dirname(rel_path)
        try:
            doc = json.load(open(rel_path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        legs_seen[pop].add(leg_dir)
        rfacts = release_facts(leg_dir)
        sigs = list(getattr(case, "supported_reactions", ()) or ()) if case else []
        term = terminal_product(getattr(case, "requested_pathway", "") if case else "")

        for i, r in enumerate((doc.get("processes") or {}).get("reactions") or []):
            if not isinstance(r, dict):
                continue
            rows_seen[pop] += 1
            ins = _names(r.get("inputs"))
            outs = _names(r.get("outputs"))
            own_stages, has_own = lineage_stage_sets(r)
            flags: Dict[str, bool] = {}

            flags["exact_glycine_heme"] = ({_norm(x) for x in ins} == {"glycine"}
                                           and {_norm(x) for x in outs} == {"heme"})
            flags["only_identifier_mapping_own_lineage"] = has_own and own_stages == {"identifier_mapping"}
            flags["only_identifier_mapping_inherited"] = (not has_own) and _inherited_only_idmap(doc, r)
            gold_match = any(_signature_matches(s, r) for s in sigs)
            flags["no_paper_and_no_rag"] = (not has_paper_stated(r)) and not _has_rag_literature(r)
            flags["precursor_terminal_shortcut"] = (
                bool(term) and matches_terminal(outs, term) and not gold_match)

            if not any(flags.values()):
                continue
            for k, v in flags.items():
                if v:
                    counts[pop][k] += 1
                    papers_hit[k].add(paper or "?")
            hits.append({
                "population": pop, "run": rel_path.split("/papers/")[0], "leg_dir": leg_dir,
                "paper": paper, "row_index": i, "name": r.get("name"),
                "inputs": ins, "outputs": outs,
                "gold_matches_a_signature": gold_match,
                "gold_signature_count": len(sigs),
                "own_lineage_stages": sorted(own_stages) if has_own else None,
                "criteria": sorted(k for k, v in flags.items() if v),
                **rfacts,
            })

    report = {
        "instrument": "rd093_shortcut_census",
        "charter": "ORCH-723 follow-up to the 2026-09-02 diagnosis",
        "evaluation_only": True,
        "reran_nothing": True,
        "criteria_definitions": {
            "exact_glycine_heme": "inputs == {glycine} and outputs == {heme}",
            "only_identifier_mapping_own_lineage": "row's OWN provenance_lineage names only identifier_mapping",
            "only_identifier_mapping_inherited": "row carries NO lineage; participants' lineage names only identifier_mapping",
            "no_paper_and_no_rag": "no paper_stated/paper_explicit origin AND no rag_literature origin on the row",
            "precursor_terminal_shortcut": "produces the requested pathway's terminal product and matches no gold signature",
        },
        "populations": {p: {"legs": len(legs_seen[p]), "reaction_rows": rows_seen[p],
                            "criteria": {c: counts[p][c] for c in CRITERIA}}
                        for p in POPULATION_ORDER},
        "distinct_papers_per_criterion": {c: sorted(papers_hit[c]) for c in CRITERIA},
        "hits": hits,
    }

    print("ORCH-723 shortcut census -- EVALUATION-ONLY, RE-RAN NOTHING")
    print("populations counted APART and never summed (F-178)\n")
    for p in POPULATION_ORDER:
        d = report["populations"][p]
        print(f"== {p}: {d['legs']} canonical legs / {d['reaction_rows']} reaction rows ==")
        for c in CRITERIA:
            n = d["criteria"][c]
            print(f"   {c:38s} {n:5d}   distinct papers: "
                  f"{len([x for x in papers_hit[c]]) if n else 0}")
        print()
    print("distinct papers per criterion (corpus-wide):")
    for c in CRITERIA:
        print(f"   {c:38s} {report['distinct_papers_per_criterion'][c]}")

    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


def _has_rag_literature(reaction: Dict[str, Any]) -> bool:
    for e in reaction.get("provenance_lineage") or []:
        if isinstance(e, dict) and e.get("origin") == "rag_literature":
            return True
    return bool(reaction.get("rag_provenance"))


def _inherited_only_idmap(doc: Dict[str, Any], reaction: Dict[str, Any]) -> bool:
    """Participants' lineage names identifier_mapping and nothing else.

    The reading that catches the `glycine -> heme` row: it carries no lineage of its
    own, so its only recorded stage comes from its participants.
    """

    want = {_norm(n) for n in _names(reaction.get("inputs")) + _names(reaction.get("outputs"))}
    if not want:
        return False
    stages: set = set()
    found = False
    for rows in ((doc.get("entities") or {}).values()):
        for row in rows or []:
            if not isinstance(row, dict) or _norm(row.get("name")) not in want:
                continue
            for e in row.get("provenance_lineage") or []:
                if isinstance(e, dict) and e.get("stage"):
                    stages.add(e["stage"])
                    found = True
    return found and stages == {"identifier_mapping"}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

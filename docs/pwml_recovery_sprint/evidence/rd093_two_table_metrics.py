"""D-093 section 5.5 -- TWO TABLES, TWO DENOMINATORS, NEVER ONE NUMBER.

EVALUATION-ONLY. Read-only over archived runs and the pinned gold set. Writes nothing
into any run, produces NO acceptance verdict, changes no runtime behaviour, and does
not re-score T-107/T-108/T-109.

=============================================================================
THE CLAUSE THIS MODULE EXISTS TO OBEY
=============================================================================

D-093 section 1, verbatim: *"External RAG support does not count toward claims that
the target paper itself was exhaustively extracted. Stage-1 paper-extraction recall and
final-system biological support are SEPARATE METRICS and must never be summed."*

So there are two questions, two populations of row, and two denominators:

  TABLE 1 -- PAPER EXTRACTION.  Did we recover the reactions the TARGET PAPER
      supports?  Scored against the paper-specific gold signatures.
      recall    = matched signatures / VERIFIED gold signatures
      precision = matched rows / rows we CLAIM the paper supports
      The precision denominator is the lineage-aware part, and the whole point: it
      counts only rows classified ``target_paper_supported``. A row we have
      attributed to another paper is not a claim about this paper's extraction, so
      charging it as a paper-extraction false positive is the D-091 error.

  TABLE 2 -- FINAL PATHWAY SUPPORT.  Does every retained reaction have defensible
      evidence, from the target paper OR properly attributed external RAG?
      unsupported rate = rows classified ``unsupported`` / ALL retained rows.
      A different question over a different denominator. ``external_rag_supported``
      counts as SUPPORTED here and is excluded from Table 1 entirely.

Adding these two numbers, or reporting either as the other, is the specific mistake
D-093 was written to stop.

=============================================================================
WHAT IS REUSED RATHER THAN REBUILT
=============================================================================

``MASTER_PLAN`` section 2 forbids rebuilding what exists, and the signature matcher
exists and is good. ``bench.semantic._signature_matches`` already handles alias
spellings, reversibility and the rule that an omitted enzyme still matches while a
CONTRADICTED one does not; ``bench.goldset`` already loads and validates the pinned
set, and ``fold_for_quote`` already implements gold quote verification. All are used
here as-is, so this instrument's matching is the SAME matching the production scorer
performs -- a lineage-aware layer over the existing scorer, not a second opinion.

WHAT IS NEW is only the partition: rows are split by support class BEFORE either
table is computed, which is the abstraction D-091's failure revealed was missing.

QUOTE VERIFICATION IS INHERITED AND NOT WEAKENED. A gold signature whose quote does
not occur in the stored paper text is a GOLD DEFECT, and ``semantic.py`` already
excludes it from scoring rather than charging it to the run. This module keeps that
rule and reports the excluded count, because silently scoring against an unverifiable
signature is how a gold defect becomes a pipeline regression.

Usage:
  python rd093_two_table_metrics.py <repo-root> [--run <substr> ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
LINEAGE_MODULE = HERE / "rd092_1_reaction_lineage.py"


def _load_lineage() -> Any:
    """The R-D092-1 classifier, loaded by path.

    It lives under ``docs/.../evidence/`` and is deliberately not importable
    production code, exactly like ``eval_semantic_populations.py``.
    """

    spec = importlib.util.spec_from_file_location("rd092_1_reaction_lineage", LINEAGE_MODULE)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise SystemExit(f"cannot load {LINEAGE_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rd092_1_reaction_lineage"] = module
    spec.loader.exec_module(module)
    return module


def _load_gold_api(repo_root: Path) -> Tuple[Any, Any, Any]:
    """``(load_gold_set, fold_for_quote, _signature_matches)`` from production bench.

    Imported rather than reimplemented so this instrument matches signatures exactly
    as the scorer does. Read-only use: nothing here mutates a gold case or a payload.
    """

    src = str(repo_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from t2pw.bench.goldset import fold_for_quote, load_gold_set  # noqa: E402
    from t2pw.bench.semantic import _signature_matches  # noqa: E402
    return load_gold_set, fold_for_quote, _signature_matches


def _paper_text(repo_root: Path, leg_dir: str) -> Optional[str]:
    """The stored paper text for a leg, for gold quote verification.

    ``01_source_text.txt`` sits at the PAPER directory, one level above the mode
    directory (``.../papers/<paper>/strict`` -> ``.../papers/<paper>``).
    """

    paper_dir = (repo_root / leg_dir).parent
    for name in ("01_source_text.txt", "00_PAPER.txt"):
        p = paper_dir / name
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
    return None


def score_leg(rd: Any, records: Sequence[Dict[str, Any]], case: Any, payload: Dict[str, Any],
              paper_text: Optional[str], fold: Any, matches: Any) -> Dict[str, Any]:
    """Both tables for one leg. Neither number is derived from the other."""

    reactions = (payload.get("processes") or {}).get("reactions") or []
    by_class = collections.Counter(r["support_class"] for r in records)

    # ---- TABLE 1: paper extraction -------------------------------------------
    # Rows we CLAIM the paper supports. This is the lineage-aware denominator.
    claimed_idx = [r["row_index"] for r in records
                   if r["support_class"] == rd.TARGET_PAPER_SUPPORTED]
    claimed_rows = [reactions[i] for i in claimed_idx if 0 <= i < len(reactions)]

    signatures = list(getattr(case, "supported_reactions", ()) or ())
    folded = fold(paper_text) if paper_text else ""

    verified: List[Any] = []
    unverifiable = 0
    for sig in signatures:
        # A gold quote that cannot be verified is an assertion. semantic.py excludes
        # it from scoring rather than charging it to the run; so does this.
        if folded and fold(sig.quote) in folded:
            verified.append(sig)
        else:
            unverifiable += 1

    matched_sigs = 0
    matched_rows: set = set()
    for sig in verified:
        hits = [i for i, row in zip(claimed_idx, claimed_rows) if matches(sig, row)]
        if hits:
            matched_sigs += 1
            matched_rows.update(hits)

    recall_den = len(verified)
    prec_den = len(claimed_rows)
    table1 = {
        "gold_signatures_stated": len(signatures),
        "gold_signatures_unverifiable_excluded": unverifiable,
        "recall_numerator_matched_signatures": matched_sigs,
        "recall_denominator_verified_signatures": recall_den,
        "recall": (matched_sigs / recall_den) if recall_den else None,
        "precision_numerator_matched_rows": len(matched_rows),
        "precision_denominator_rows_claimed_target_paper_supported": prec_den,
        "precision": (len(matched_rows) / prec_den) if prec_den else None,
        "evaluable": bool(recall_den) and paper_text is not None,
        "inapplicable_reason": (
            "" if recall_den and paper_text is not None else
            ("the gold case states no supported_reactions" if not signatures else
             ("no stored paper text, so no gold quote could be verified"
              if paper_text is None else
              "every gold signature quote failed verification against the stored text"))
        ),
    }

    # ---- TABLE 2: final pathway support --------------------------------------
    # A DIFFERENT denominator: every retained row, whatever its origin.
    retained = len(records)
    table2 = {
        "retained_rows": retained,
        "unsupported_rows": by_class.get(rd.UNSUPPORTED, 0),
        "unsupported_rate": (by_class.get(rd.UNSUPPORTED, 0) / retained) if retained else None,
        "target_paper_supported": by_class.get(rd.TARGET_PAPER_SUPPORTED, 0),
        "external_rag_supported": by_class.get(rd.EXTERNAL_RAG_SUPPORTED, 0),
        "indeterminate": by_class.get(rd.INDETERMINATE, 0),
    }
    return {"table1_paper_extraction": table1, "table2_final_support": table2}


def f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def aggregate(legs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Corpus totals, per population, with both denominators carried through.

    Micro-averaged from the numerators and denominators rather than averaging the
    per-leg rates: a leg with one signature and a leg with three are not equal
    evidence, and averaging rates would silently weight them the same.
    """

    out: Dict[str, Any] = {}
    for pop in ("canonical", "fallback"):
        rows = [l for l in legs if l["population"] == pop]
        ev = [l for l in rows if l["scores"]["table1_paper_extraction"]["evaluable"]]
        rn = sum(l["scores"]["table1_paper_extraction"]["recall_numerator_matched_signatures"] for l in ev)
        rd_ = sum(l["scores"]["table1_paper_extraction"]["recall_denominator_verified_signatures"] for l in ev)
        pn = sum(l["scores"]["table1_paper_extraction"]["precision_numerator_matched_rows"] for l in ev)
        pd_ = sum(l["scores"]["table1_paper_extraction"]["precision_denominator_rows_claimed_target_paper_supported"] for l in ev)
        recall = (rn / rd_) if rd_ else None
        precision = (pn / pd_) if pd_ else None

        retained = sum(l["scores"]["table2_final_support"]["retained_rows"] for l in rows)
        unsup = sum(l["scores"]["table2_final_support"]["unsupported_rows"] for l in rows)
        distinct_papers = len({l["target_paper"] for l in ev})
        distinct_sigs = sum(
            n for n in {l["target_paper"]: l["scores"]["table1_paper_extraction"]
                        ["gold_signatures_stated"] for l in ev}.values())
        out[pop] = {
            "legs_total": len(rows),
            "legs_evaluable_for_table1": len(ev),
            "legs_inapplicable_for_table1": len(rows) - len(ev),
            "table1_paper_extraction": {
                "recall": recall, "recall_numerator": rn,
                # Named for what it is. The unit is a (signature, leg) pair, not a
                # distinct signature: micro-averaging weights a paper by how many
                # legs it was run in, which is the right weighting for "how often did
                # extraction recover this?" and the WRONG number to quote as "the
                # gold set has N signatures". Both are reported so neither can be
                # mistaken for the other.
                "recall_denominator": rd_,
                "recall_denominator_unit": "verified (signature, leg) pairs",
                "distinct_signatures_in_gold": distinct_sigs,
                "distinct_papers_scored": distinct_papers,
                "precision": precision, "precision_numerator": pn, "precision_denominator": pd_,
                "precision_denominator_unit": "rows classified target_paper_supported",
                "f1": f1(precision, recall),
            },
            "table2_final_support": {
                "unsupported_rate": (unsup / retained) if retained else None,
                "unsupported_rows": unsup, "retained_rows_denominator": retained,
                "external_rag_supported": sum(
                    l["scores"]["table2_final_support"]["external_rag_supported"] for l in rows),
                "indeterminate": sum(
                    l["scores"]["table2_final_support"]["indeterminate"] for l in rows),
            },
        }
    return out


def render(agg: Dict[str, Any]) -> str:
    lines = ["D-093 s.5.5 -- TWO TABLES, TWO DENOMINATORS. Never summed, never interchanged.",
             "EVALUATION-ONLY. No acceptance verdict. No runtime touched.", ""]
    for pop in ("canonical", "fallback"):
        a = agg[pop]
        lines.append(f"== population: {pop} ==")
        lines.append(f"   legs {a['legs_total']}   evaluable for table 1: "
                     f"{a['legs_evaluable_for_table1']}   inapplicable: "
                     f"{a['legs_inapplicable_for_table1']}")
        t1, t2 = a["table1_paper_extraction"], a["table2_final_support"]
        lines.append("   TABLE 1 -- PAPER EXTRACTION (did we recover what the TARGET PAPER states?)")
        lines.append(f"      recall    {_pct(t1['recall'])}  = {t1['recall_numerator']} of "
                     f"{t1['recall_denominator']} verified SIGNATURE-LEG PAIRS")
        lines.append(f"                (the gold set states {t1['distinct_signatures_in_gold']} "
                     f"distinct signatures across {t1['distinct_papers_scored']} scored papers; "
                     f"the denominator above is micro-averaged over legs, so a paper run more "
                     f"often weighs more -- it is NOT a count of distinct signatures)")
        lines.append(f"      precision {_pct(t1['precision'])}  = {t1['precision_numerator']} of "
                     f"{t1['precision_denominator']} rows claimed target_paper_supported")
        lines.append(f"      F1        {_pct(t1['f1'])}")
        lines.append("   TABLE 2 -- FINAL PATHWAY SUPPORT (does every RETAINED row have defensible evidence?)")
        lines.append(f"      unsupported rate {_pct(t2['unsupported_rate'])}  = "
                     f"{t2['unsupported_rows']} of {t2['retained_rows_denominator']} retained rows")
        lines.append(f"      external_rag_supported {t2['external_rag_supported']}   "
                     f"indeterminate {t2['indeterminate']}")
        lines.append("   (the two tables have DIFFERENT denominators and are not addable)")
        lines.append("")
    return "\n".join(lines)


def _pct(v: Optional[float]) -> str:
    return "  n/a " if v is None else f"{100.0*v:5.1f}%"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_root")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    rd = _load_lineage()
    load_gold_set, fold, matches = _load_gold_api(repo_root)

    gold = load_gold_set()
    cases = {c.paper_id: c for c in gold.cases}

    paths = rd.committed_paths(repo_root)
    legs_meta = rd.discover_legs(paths, args.run)
    adm = rd.admission_index(paths, repo_root, [])

    legs: List[Dict[str, Any]] = []
    for leg in legs_meta:
        case = cases.get(leg["target_paper"])
        if case is None:
            continue  # not a gold paper: it has no paper-extraction ground truth
        payload = rd.load_json(repo_root / leg["leg_dir"] / leg["payload_file"])
        if not isinstance(payload, dict):
            continue
        index = rd.entity_provenance_index(payload)
        reactions = (payload.get("processes") or {}).get("reactions") or []
        records = [rd.build_record(r, leg, i, index, adm)
                   for i, r in enumerate(reactions) if isinstance(r, dict)]
        legs.append({
            "leg_dir": leg["leg_dir"], "population": leg["population"],
            "target_paper": leg["target_paper"],
            "scores": score_leg(rd, records, case, payload,
                                _paper_text(repo_root, leg["leg_dir"]), fold, matches),
        })

    agg = aggregate(legs)
    print(render(agg))
    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "instrument": "rd093_two_table_metrics",
            "charter": "D-093 s.5.5",
            "evaluation_only": True,
            "gold_version": getattr(gold, "version", "unknown"),
            "corpus": "committed (git ls-files)",
            "aggregate": agg, "legs": legs,
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

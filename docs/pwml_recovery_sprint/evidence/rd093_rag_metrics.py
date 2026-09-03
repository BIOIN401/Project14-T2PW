"""D-093 section 5.7 -- core RAG metrics from the EXISTING archived artifacts.

EVALUATION-ONLY. Read-only over committed admission reports and the pinned gold set.
No pipeline leg is re-run: D-093 section 3 is explicit that information already on disk
must not be recreated by re-running expensive legs.

=============================================================================
D-093 FORBIDS ONE "RAG ACCURACY" NUMBER. THESE SEVEN STAY APART.
=============================================================================

The ruling names them and requires they not be collapsed, because each has a different
fix and averaging them hides which one is broken:

  retrieval_did_not_find_it                 the gold reaction is in no candidate at all
  found_but_ranked_poorly                   relevant candidate present, outside top-5
  found_but_not_admitted                    relevant, well-ranked, and not admitted
  correct_candidate_rejected                a gold-matching candidate was REJECTED
  unsupported_candidate_admitted            a non-matching candidate was ACCEPTED
  unsupported_candidate_correctly_rejected  the system working as intended
  rejected_candidate_reintroduced           a rejected candidate survives in the payload

=============================================================================
THE RELEVANCE LABEL, AND WHY IT IS NOT THE GATE'S OWN VERDICT
=============================================================================

A candidate is RELEVANT when its chemistry matches a gold ``supported_reactions``
signature for that leg's target paper, using ``bench.semantic._signature_matches`` --
the same matcher the production scorer uses.

**Labelling by the admission gate's own accept/reject would be circular**: it would make
"correct candidate rejected" unmeasurable by construction, since a rejected candidate
would be incorrect by definition. The gold set is independent of the gate, so the gate
can be scored against it.

WHAT THAT LABEL DOES AND DOES NOT COVER, stated plainly. Gold signatures describe the
TARGET PAPER's reactions, while RAG candidates are drawn from OTHER papers to fill gaps.
So a "relevant" candidate is one where retrieval reproduced a reaction the target paper
itself states. That is a real and checkable notion, and it is NARROW: a candidate
supplying legitimate external biology the target paper never states is NOT relevant
here and is not thereby wrong. It is exactly D-093's ``external_rag_supported``, which
the R-D092-1 instrument classifies and this one does not re-litigate. **These counts are
retrieval quality against the paper's own chemistry, not a verdict on external
enrichment.**

=============================================================================
TWO MEASURED PROPERTIES OF THE ARCHIVE THAT LIMIT THESE NUMBERS
=============================================================================

(1) THERE IS NO RETRIEVAL RANK ON DISK. No candidate record carries a rank field.
    Rank is DERIVED by sorting a gap's candidates by ``evidence.score`` descending, and
    the score belongs to the CHUNK, not the candidate -- many candidates share one
    chunk and therefore one score. Measured on a representative gap: 7 candidates, 4
    distinct scores. **Ties are resolved PESSIMISTICALLY** (a tied candidate takes the
    WORST rank in its tie group), because the alternative flatters the system and this
    sprint has paid for flattering measurements. ``ties_in_top5`` is reported so a
    reader can judge how much the convention is doing.

(2) THE PERSISTED CANDIDATE LIST IS TRUNCATED. ``policy.max_report_entries`` is 200 and
    five of T-109's nineteen legs hit it: 3,261 rejections were COUNTED and only 2,076
    PERSISTED, so 1,185 candidates exist only as a tally. A rank metric computed over a
    truncated list silently assumes the missing candidates were irrelevant. **Truncated
    legs are therefore a SEPARATE POPULATION and are never summed with clean ones** --
    the same discipline F-177 established for canonical and fallback payloads.

Usage:
  python rd093_rag_metrics.py <repo-root> [--run <substr> ...] [--json OUT]
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

#: Populations. A truncated leg's ranks are unreliable by construction, so its numbers
#: live apart rather than being folded into a corpus average.
CLEAN = "untruncated"
TRUNCATED = "truncated"
POPULATION_ORDER: Tuple[str, ...] = (CLEAN, TRUNCATED)

#: The seven outcomes D-093 s.6 requires be kept apart, in a fixed print order.
OUTCOMES: Tuple[str, ...] = (
    "retrieval_did_not_find_it",
    "found_but_ranked_poorly",
    "found_but_not_admitted",
    "correct_candidate_rejected",
    "unsupported_candidate_admitted",
    "unsupported_candidate_correctly_rejected",
    "rejected_candidate_reintroduced",
)

#: The UNIT each outcome is counted in. Printed on every row, because three different
#: units share one table and a reader who adds them gets a meaningless number.
OUTCOME_UNIT: Dict[str, str] = {
    "retrieval_did_not_find_it": "per gold signature",
    "found_but_ranked_poorly": "per gap",
    "found_but_not_admitted": "per gap",
    "correct_candidate_rejected": "per gap",
    "unsupported_candidate_admitted": "per gap",
    "unsupported_candidate_correctly_rejected": "per gap",
    "rejected_candidate_reintroduced": "per reaction row",
}

K_VALUES: Tuple[int, ...] = (1, 3, 5)


def _load_lineage() -> Any:
    spec = importlib.util.spec_from_file_location(
        "rd092_1_reaction_lineage", HERE / "rd092_1_reaction_lineage.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["rd092_1_reaction_lineage"] = m
    spec.loader.exec_module(m)
    return m


def pessimistic_ranks(scores: Sequence[Optional[float]]) -> List[int]:
    """1-based ranks, ties taking the WORST rank in their group.

    Scores sort descending; a missing score sorts last rather than first, because an
    absent score is not evidence of a good match.

    The convention is conservative on purpose. With average- or best-rank tie handling
    a gap whose seven candidates all share one score would report Recall@1 = 1.0 for a
    retriever that expressed no preference at all.
    """

    order = sorted(range(len(scores)),
                   key=lambda i: (scores[i] is None, -(scores[i] or 0.0), i))
    ranks = [0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        worst = j + 1                      # 1-based worst rank in this tie group
        for k in range(i, j + 1):
            ranks[order[k]] = worst
        i = j + 1
    return ranks


def score_gap(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Rank-based metrics for ONE gap, treated as one retrieval query.

    Returns ``relevant=False`` for a gap with no gold-matching candidate; such a gap is
    a NEGATIVE query, where the correct behaviour is to admit nothing, and it is scored
    that way rather than counted as a recall failure.
    """

    ranks = pessimistic_ranks([c.get("score") for c in candidates])
    rel = [(ranks[i], c) for i, c in enumerate(candidates) if c["relevant"]]
    ties = sum(1 for i, c in enumerate(candidates)
               if ranks[i] <= 5 and sum(1 for s in candidates
                                        if s.get("score") == c.get("score")) > 1)
    out: Dict[str, Any] = {
        "candidates": len(candidates),
        "relevant_candidates": len(rel),
        "is_positive_query": bool(rel),
        "accepted": sum(1 for c in candidates if c["accepted"]),
        "ties_in_top5": ties,
    }
    if rel:
        best = min(r for r, _ in rel)
        out["best_relevant_rank"] = best
        out["reciprocal_rank"] = 1.0 / best
        for k in K_VALUES:
            out[f"hit_at_{k}"] = best <= k
        top5 = [c for i, c in enumerate(candidates) if ranks[i] <= 5]
        out["precision_at_5_numerator"] = sum(1 for c in top5 if c["relevant"])
        out["precision_at_5_denominator"] = len(top5)
        out["relevant_admitted"] = any(c["accepted"] for _, c in rel)
    else:
        # Negative query: correct behaviour is to admit nothing.
        out["correctly_rejected_all"] = out["accepted"] == 0
    return out


def classify_outcomes(gaps: Sequence[Dict[str, Any]], signatures_never_retrieved: int,
                      reintroduced: int) -> collections.Counter:
    """The seven-way taxonomy. Never averaged into one rate.

    THE UNITS DIFFER AND ARE NAMED, because forcing them to one unit is what makes a
    taxonomy lie. Five outcomes are counted per GAP (one retrieval query).
    ``retrieval_did_not_find_it`` is counted per GOLD SIGNATURE -- a signature no
    candidate anywhere matched produces no gap at all, so counting it per gap would
    report a structural zero, which is this project's standing "missing key read as
    zero" defect. ``rejected_candidate_reintroduced`` is counted per REACTION ROW and
    comes from the R-D092-1 lineage classifier, since it is a property of the delivered
    payload rather than of a retrieval query.
    """

    c: collections.Counter = collections.Counter()
    c["retrieval_did_not_find_it"] = signatures_never_retrieved
    c["rejected_candidate_reintroduced"] = reintroduced
    for g in gaps:
        if not g["is_positive_query"]:
            c["unsupported_candidate_correctly_rejected" if g.get("correctly_rejected_all")
              else "unsupported_candidate_admitted"] += 1
            continue
        if g.get("relevant_admitted"):
            continue                       # the system worked; not one of the failures
        if g["best_relevant_rank"] > 5:
            c["found_but_ranked_poorly"] += 1
        elif g["accepted"] > 0:
            # Well-ranked, not admitted, and the gate DID admit something else from
            # this gap: the relevant candidate was passed over rather than refused.
            # D-093's "found but the LLM ignored it".
            c["found_but_not_admitted"] += 1
        else:
            c["correct_candidate_rejected"] += 1
    return c


def aggregate(legs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pop in POPULATION_ORDER:
        rows = [l for l in legs if l["population"] == pop]
        gaps = [g for l in rows for g in l["gaps"]]
        pos = [g for g in gaps if g["is_positive_query"]]
        neg = [g for g in gaps if not g["is_positive_query"]]
        agg: Dict[str, Any] = {
            "legs": len(rows),
            "gaps_total": len(gaps),
            "positive_queries": len(pos),
            "negative_queries": len(neg),
            "ties_in_top5": sum(g["ties_in_top5"] for g in gaps),
        }
        for k in K_VALUES:
            agg[f"recall_at_{k}"] = (sum(1 for g in pos if g[f"hit_at_{k}"]) / len(pos)) if pos else None
            agg[f"recall_at_{k}_numerator"] = sum(1 for g in pos if g[f"hit_at_{k}"])
        agg["recall_denominator_positive_queries"] = len(pos)
        pn = sum(g["precision_at_5_numerator"] for g in pos)
        pdn = sum(g["precision_at_5_denominator"] for g in pos)
        agg["precision_at_5"] = (pn / pdn) if pdn else None
        agg["precision_at_5_numerator"] = pn
        agg["precision_at_5_denominator_retrieved_in_top5"] = pdn
        agg["mrr"] = (sum(g["reciprocal_rank"] for g in pos) / len(pos)) if pos else None
        agg["negative_query_rejection_rate"] = (
            sum(1 for g in neg if g.get("correctly_rejected_all")) / len(neg)) if neg else None
        agg["negative_query_denominator"] = len(neg)
        counts = collections.Counter()
        for l in rows:
            counts.update(l["outcomes"])
        agg["outcomes"] = {o: counts.get(o, 0) for o in OUTCOMES}
        out[pop] = agg
    return out


def render(agg: Dict[str, Any], caveats: Dict[str, Any]) -> str:
    L = ["D-093 s.5.7 -- CORE RAG METRICS from archived artifacts. EVALUATION-ONLY.",
         "Seven outcomes kept APART; no single 'RAG accuracy' number is produced.", ""]
    L.append(f"relevance label : a candidate matching a GOLD supported_reactions signature")
    L.append(f"                  for its leg's target paper (bench.semantic matcher).")
    L.append(f"                  NARROW BY CONSTRUCTION -- legitimate external biology the")
    L.append(f"                  target paper never states is NOT 'relevant' and is NOT")
    L.append(f"                  thereby wrong; that is external_rag_supported, scored by")
    L.append(f"                  R-D092-1, not here.")
    L.append(f"rank            : DERIVED from chunk score (no rank on disk); ties take the")
    L.append(f"                  WORST rank in their group.")
    L.append(f"truncation      : {caveats['truncated_legs']} of {caveats['legs_total']} legs hit "
             f"max_report_entries; {caveats['candidates_counted_not_persisted']} candidates were "
             f"COUNTED but never PERSISTED.")
    L.append("")
    for pop in POPULATION_ORDER:
        a = agg[pop]
        L.append(f"== population: {pop} ==")
        L.append(f"   legs {a['legs']}   gaps {a['gaps_total']}   "
                 f"positive {a['positive_queries']}   negative {a['negative_queries']}")
        if not a["gaps_total"]:
            L.append("   0 gaps -- itself a fact about the corpus, not a skip")
            L.append("")
            continue
        for k in K_VALUES:
            L.append(f"   Recall@{k}   {_p(a[f'recall_at_{k}'])}  = {a[f'recall_at_{k}_numerator']} "
                     f"of {a['recall_denominator_positive_queries']} positive queries")
        L.append(f"   Precision@5 {_p(a['precision_at_5'])}  = {a['precision_at_5_numerator']} "
                 f"of {a['precision_at_5_denominator_retrieved_in_top5']} candidates in top-5")
        L.append(f"   MRR         {_f(a['mrr'])}  over {a['recall_denominator_positive_queries']} positive queries")
        L.append(f"   negative-query rejection {_p(a['negative_query_rejection_rate'])}  = of "
                 f"{a['negative_query_denominator']} negative queries")
        L.append(f"   ties inside top-5: {a['ties_in_top5']} (pessimistic tie handling)")
        L.append("   outcomes (each a different defect with a different fix).")
        L.append("   UNITS DIFFER AND ARE NOT ADDABLE -- the unit is named on every row:")
        for o in OUTCOMES:
            L.append(f"      {o:44s} {a['outcomes'][o]:5d}  [{OUTCOME_UNIT[o]}]")
        L.append("")
    return "\n".join(L)


def _p(v: Optional[float]) -> str:
    return "  n/a " if v is None else f"{100.0*v:5.1f}%"


def _f(v: Optional[float]) -> str:
    return "  n/a " if v is None else f"{v:5.3f}"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_root")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--json", dest="json_path", default=None,
                    help="aggregate plus a per-leg summary. THIS is the committed artifact")
    ap.add_argument("--detail-json", dest="detail_path", default=None,
                    help="every per-gap record (~1.1 MB, regenerable, not committed)")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    rd = _load_lineage()
    src = str(repo_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from t2pw.bench.goldset import load_gold_set          # noqa: E402
    from t2pw.bench.semantic import _signature_matches    # noqa: E402

    cases = {c.paper_id: c for c in load_gold_set().cases}
    paths = rd.committed_paths(repo_root)
    tracked = set(paths)
    adm_index = rd.admission_index(paths, repo_root, [])

    legs: List[Dict[str, Any]] = []
    truncated_legs = 0
    counted_not_persisted = 0
    legs_total = 0

    for p in paths:
        if not p.endswith("/rag_admission_report.json"):
            continue
        if args.run and not any(f in p for f in args.run):
            continue
        leg_dir = p.rpartition("/")[0]
        target = rd.target_paper_id(leg_dir)
        case = cases.get(target or "")
        doc = rd.load_json(repo_root / p)
        if not isinstance(doc, dict) or case is None:
            continue
        legs_total += 1

        trunc = doc.get("truncated") or {}
        is_trunc = bool(trunc.get("accepted") or trunc.get("rejected"))
        if is_trunc:
            truncated_legs += 1
        counted_not_persisted += max(
            0, (doc.get("counts") or {}).get("rejected", 0) - len(doc.get("rejected") or []))

        sigs = list(getattr(case, "supported_reactions", ()) or ())
        by_gap: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for group in ("accepted", "rejected"):
            for rec in doc.get(group) or []:
                if not isinstance(rec, dict):
                    continue
                row = {"inputs": rec.get("inputs") or [], "outputs": rec.get("outputs") or [],
                       "enzymes": [{"protein": e} for e in (rec.get("enzymes") or [])]}
                by_gap[rec.get("gap_id") or "unknown"].append({
                    "score": (rec.get("evidence") or {}).get("score"),
                    "accepted": group == "accepted",
                    "relevant": any(_signature_matches(s, row) for s in sigs),
                    "name": rec.get("name"),
                })

        gaps = [score_gap(c) for c in by_gap.values()]

        # Per-SIGNATURE, not per-gap: a gold reaction that no candidate anywhere in
        # this leg matched was never retrieved at all, and produces no gap to count.
        all_rows = [{"inputs": r.get("inputs") or [], "outputs": r.get("outputs") or [],
                     "enzymes": [{"protein": e} for e in (r.get("enzymes") or [])]}
                    for group in ("accepted", "rejected") for r in (doc.get(group) or [])
                    if isinstance(r, dict)]
        never_retrieved = sum(
            1 for s in sigs if not any(_signature_matches(s, row) for row in all_rows))

        # Per-ROW, from the lineage classifier: a rejected candidate that nonetheless
        # survives into the delivered payload. A property of the payload, not of a
        # query, so it is counted in its own unit rather than folded into a gap rate.
        reintroduced = 0
        payload_file = next((f for f in rd.PAYLOAD_FILES
                             if f"{leg_dir}/{f}" in tracked), None)
        if payload_file:
            payload = rd.load_json(repo_root / leg_dir / payload_file)
            if isinstance(payload, dict):
                meta = {"leg_dir": leg_dir, "payload_file": payload_file,
                        "population": rd.POPULATION_BY_FILE[payload_file],
                        "run": leg_dir.split("/papers/")[0],
                        "target_paper": target or rd.UNAVAILABLE}
                idx = rd.entity_provenance_index(payload)
                for i, r in enumerate((payload.get("processes") or {}).get("reactions") or []):
                    if not isinstance(r, dict):
                        continue
                    rec = rd.build_record(r, meta, i, idx, adm_index)
                    if (rec.get("admission_result") == "rejected"
                            and rec.get("chunk_join_reaction_specific") is True
                            and rec.get("survives_in_payload") is True):
                        reintroduced += 1

        legs.append({
            "leg_dir": leg_dir, "target_paper": target,
            "population": TRUNCATED if is_trunc else CLEAN,
            "gold_signatures": len(sigs), "signatures_never_retrieved": never_retrieved,
            "gaps": gaps,
            "outcomes": classify_outcomes(gaps, never_retrieved, reintroduced),
        })

    caveats = {"legs_total": legs_total, "truncated_legs": truncated_legs,
               "candidates_counted_not_persisted": counted_not_persisted}
    agg = aggregate(legs)
    print(render(agg, caveats))
    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "instrument": "rd093_rag_metrics", "charter": "D-093 s.5.7",
            "evaluation_only": True, "corpus": "committed (git ls-files)",
            "caveats": caveats, "aggregate": agg,
            # Per-leg summary only. The per-GAP detail is ~1.1 MB of derived data that
            # one command reproduces, and the sprint's .git is already 158 MB, so it is
            # written only when --detail-json asks for it.
            "legs": [{"leg_dir": l["leg_dir"], "target_paper": l["target_paper"],
                      "population": l["population"],
                      "gold_signatures": l["gold_signatures"],
                      "signatures_never_retrieved": l["signatures_never_retrieved"],
                      "gaps_total": len(l["gaps"]),
                      "positive_queries": sum(1 for g in l["gaps"] if g["is_positive_query"]),
                      "outcomes": dict(l["outcomes"])} for l in legs],
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}")
    if args.detail_path:
        out = Path(args.detail_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            [{k: v for k, v in l.items() if k != "outcomes"} |
             {"outcomes": dict(l["outcomes"])} for l in legs],
            indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

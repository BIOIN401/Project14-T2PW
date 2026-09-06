"""ORCH-724 -- the final bounded admission-gate audit.

EVALUATION-ONLY, READ-ONLY. Scores the RAG admission gate's REJECTIONS against the
41-reaction CURATION corpus (``curation/expected_core_*.json``) rather than against the
19-signature recovery gold. Re-runs no leg, writes into no run directory, imports no
pipeline stage that mutates anything.

=============================================================================
THE ONE QUESTION THIS ANSWERS
=============================================================================

  Is there a clear, REPEATED admission rule rejecting genuinely correct,
  already-curated reactions?

A repeated mechanism is one reason code refusing many curated reactions for the same
wrong cause. This instrument does not adjudicate all 1,123 rejections and is not meant
to: it produces (a) the credibility screen on the curated positives, (b) the census of
curated-matching rejections BY REASON CODE, and (c) a stratified sample with the full
record a human needs to classify each case. The classification itself stays human.

=============================================================================
WHY THE CURATED POSITIVES ARE SCREENED FIRST
=============================================================================

A curated reaction is a POSITIVE in this audit only if its own quote is verifiable in
the stored paper text, using ``bench.goldset.fold_for_quote`` -- the same folding the
production scorer uses, which drops punctuation because the cached full text carries
italic-stripping artifacts that split tokens. A curated entry whose quote cannot be
found is EXCLUDED from the decision set and reported separately. An unverifiable
expectation is an assertion, and an audit that lets one drive a production change is
the ``gold_data_defect`` failure the sprint has already paid for once.

=============================================================================
WHAT "MATCHES A CURATED REACTION" MEANS, AND WHAT IT DOES NOT
=============================================================================

Chemistry matching uses ``bench.semantic._signature_matches`` against a
``SupportedReaction`` built from the curated substrates/products/enzyme -- the SAME
matcher the production scorer and ``rd093_rag_metrics.py`` use, so this audit and the
committed RAG metrics cannot disagree about what "the same reaction" means.

The match is one-directional on the participant side: a candidate may carry EXTRA
participants (cofactors the curator did not spell out) and still match, but it may not
be missing one the curator states. The enzyme is optional evidence -- a candidate that
names no enzyme still matches; one that names a DIFFERENT enzyme does not.

**This is a claim about chemistry, not a verdict that the candidate should have been
admitted.** The gate may still be right to refuse a chemically-correct candidate --
because the span does not state the relation, because the organism is wrong, because
the candidate is the wrong TYPE to fill the gap it was proposed against. Separating
those is exactly what the reason-code census is for.

=============================================================================
TRUNCATION, AND WHY THE TWO POPULATIONS STAY APART
=============================================================================

``policy.max_report_entries`` caps the persisted candidate list. A truncated leg's
rejection census is a sample of unknown bias, so truncated and untruncated legs are
counted apart and never summed -- the discipline F-177 established and
``rd093_rag_metrics.py`` follows.

Usage:
  python orch724_admission_audit.py <repo-root> [--json OUT] [--sample-per-reason N]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent

CURATION_DIRNAME = Path("docs/pwml_recovery_sprint/curation")
FULLTEXT_DIRNAME = Path("data/rag_index/acquire_cache/fulltext")

CLEAN = "untruncated"
TRUNCATED = "truncated"


# ---------------------------------------------------------------------------
# Repo-relative loading.
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fulltext_of(root: Path, cache_file: str) -> str:
    """The stored paper text for a curation record's ``source_cache_file``.

    Returns "" when the cache entry is absent or carries no text; the caller
    reports that as UNVERIFIABLE rather than as a quote failure, because the two
    have different owners.
    """

    if not cache_file:
        return ""
    path = root / FULLTEXT_DIRNAME / cache_file
    if not path.is_file():
        return ""
    try:
        blob = _load_json(path)
    except Exception:
        return ""
    if isinstance(blob, str):
        return blob
    if isinstance(blob, dict):
        for key in ("text", "fulltext", "full_text", "body", "content"):
            value = blob.get(key)
            if isinstance(value, str) and value.strip():
                return value
        # Some cache entries hold the text under a nested record.
        for value in blob.values():
            if isinstance(value, str) and len(value) > 2000:
                return value
    return ""


# ---------------------------------------------------------------------------
# Candidate records.
# ---------------------------------------------------------------------------
def _candidate_row(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """A candidate rendered in the shape ``_signature_matches`` reads.

    The admission report stores participants as plain name strings under
    ``inputs``/``outputs`` and catalysts under ``enzymes``; ``_names`` accepts
    strings directly, and ``_enzyme_names`` reads the schema's enzyme slots, of
    which ``enzymes`` is one.
    """

    return {
        "inputs": list(candidate.get("inputs") or []),
        "outputs": list(candidate.get("outputs") or []),
        "enzymes": list(candidate.get("enzymes") or []),
    }


def _reason_codes(candidate: Dict[str, Any]) -> List[str]:
    """Bare reason codes; the persisted strings are ``code: prose``."""

    out: List[str] = []
    for reason in (candidate.get("reasons") or []):
        if isinstance(reason, str):
            out.append(reason.split(":", 1)[0].strip())
    return out


def _leg_paper_and_mode(path: Path) -> Tuple[str, str]:
    parts = path.as_posix().split("/")
    return parts[-3], parts[-2]


def _is_truncated(report: Dict[str, Any]) -> bool:
    trunc = report.get("truncated")
    if isinstance(trunc, dict):
        return bool(trunc.get("rejected") or trunc.get("accepted") or trunc.get("any"))
    return bool(trunc)


# ---------------------------------------------------------------------------
# The audit.
# ---------------------------------------------------------------------------
def build_signatures(root: Path) -> Tuple[Dict[str, List[Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(signatures_by_paper, admitted_records, excluded_records)``.

    ``admitted_records`` are the curated reactions whose quote verified and which
    therefore form the decision set; ``excluded_records`` are reported and never
    used to justify anything.
    """

    from t2pw.bench.goldset import GoldTerm, SupportedReaction, fold_for_quote  # noqa: E402

    signatures: Dict[str, List[Any]] = {}
    admitted: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []

    for path in sorted((root / CURATION_DIRNAME).glob("expected_core_*.json")):
        record = _load_json(path)
        paper = str(record.get("paper_id") or "")
        text = _fulltext_of(root, str(record.get("source_cache_file") or ""))
        folded_text = fold_for_quote(text) if text else ""
        for reaction in (record.get("expected_core_reactions") or []):
            quote = str(reaction.get("quote") or "")
            entry = {
                "paper": paper,
                "id": reaction.get("id"),
                "substrates": list(reaction.get("substrates") or []),
                "products": list(reaction.get("products") or []),
                "enzymes": list(reaction.get("enzymes") or []),
                "confidence": reaction.get("confidence"),
                "quote": quote,
                "curated_by": record.get("curated_by"),
            }
            if not folded_text:
                entry["screen"] = "unverifiable_no_stored_text"
                excluded.append(entry)
                continue
            if not quote:
                entry["screen"] = "excluded_no_quote"
                excluded.append(entry)
                continue
            if fold_for_quote(quote) not in folded_text:
                entry["screen"] = "excluded_quote_not_found_in_stored_text"
                excluded.append(entry)
                continue
            if not entry["substrates"] or not entry["products"]:
                # A signature with an empty side matches everything on that side.
                entry["screen"] = "excluded_one_sided_signature"
                excluded.append(entry)
                continue
            entry["screen"] = "admitted"
            admitted.append(entry)
            signatures.setdefault(paper, []).append(
                SupportedReaction(
                    inputs=tuple(GoldTerm(name=n) for n in entry["substrates"]),
                    outputs=tuple(GoldTerm(name=n) for n in entry["products"]),
                    enzyme=GoldTerm(name=entry["enzymes"][0]) if entry["enzymes"] else None,
                    reversible=False,
                    quote=quote,
                    label=f"{paper}:{entry['id']}",
                )
            )
    return signatures, admitted, excluded


def audit(root: Path, sample_per_reason: int) -> Dict[str, Any]:
    from t2pw.bench.semantic import _signature_matches  # noqa: E402

    signatures, admitted, excluded = build_signatures(root)

    reports = sorted(
        list(root.glob("runs/*/papers/*/*/rag_admission_report.json"))
        + list(root.glob("runs_verify/*/papers/*/*/rag_admission_report.json"))
    )

    # reason -> population -> count, over candidates that MATCH a curated reaction
    matched_by_reason: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    matched_by_paper: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    accepted_matched: collections.Counter = collections.Counter()
    totals: collections.Counter = collections.Counter()
    # distinct curated reactions that were matched by at least one REJECTED candidate
    curated_hit_rejected: Dict[str, set] = collections.defaultdict(set)
    curated_hit_accepted: Dict[str, set] = collections.defaultdict(set)
    samples: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    legs_seen = 0

    for path in reports:
        paper, mode = _leg_paper_and_mode(path)
        sigs = signatures.get(paper)
        if not sigs:
            continue
        try:
            report = _load_json(path)
        except Exception:
            continue
        legs_seen += 1
        population = TRUNCATED if _is_truncated(report) else CLEAN
        totals[f"legs_{population}"] += 1

        for status, bucket in (("accepted", report.get("accepted")), ("rejected", report.get("rejected"))):
            for candidate in (bucket or []):
                if not isinstance(candidate, dict):
                    continue
                totals[f"{status}_{population}"] += 1
                row = _candidate_row(candidate)
                hits = [s for s in sigs if _signature_matches(s, row)]
                if not hits:
                    continue
                totals[f"{status}_matched_curated_{population}"] += 1
                if status == "accepted":
                    accepted_matched[population] += 1
                    for s in hits:
                        curated_hit_accepted[population].add(s.label)
                    continue
                for s in hits:
                    curated_hit_rejected[population].add(s.label)
                matched_by_paper[paper][population] += 1
                codes = _reason_codes(candidate) or ["(no_reason_recorded)"]
                for code in codes:
                    matched_by_reason[code][population] += 1
                    if population == CLEAN and len(samples[code]) < sample_per_reason:
                        samples[code].append(
                            {
                                "run": path.as_posix().split("/")[1],
                                "target_paper": paper,
                                "mode": mode,
                                "curated_matches": [s.label for s in hits],
                                "curated_signature": [s.describe() for s in hits],
                                "candidate_name": candidate.get("name"),
                                "inputs": row["inputs"],
                                "outputs": row["outputs"],
                                "enzymes": row["enzymes"],
                                "evidence_source_paper": (candidate.get("source_paper") or {}).get("source_id"),
                                "evidence_section": (candidate.get("evidence") or {}).get("section"),
                                "evidence_score": (candidate.get("evidence") or {}).get("score"),
                                "evidence_span": (candidate.get("evidence") or {}).get("span"),
                                "gap_id": candidate.get("gap_id"),
                                "requested_pathway": candidate.get("requested_pathway"),
                                "requested_pathway_match": candidate.get("requested_pathway_match"),
                                "requested_organism": candidate.get("requested_organism"),
                                "organism_match": candidate.get("organism_match"),
                                "all_reasons": candidate.get("reasons"),
                            }
                        )

    return {
        "instrument": "orch724_admission_audit",
        "charter": "final bounded admission audit: is a repeated rule rejecting curated-correct reactions?",
        "evaluation_only": True,
        "corpus": "committed run and runs_verify admission reports",
        "legs_scored": legs_seen,
        "curation_screen": {
            "admitted_positives": len(admitted),
            "excluded": len(excluded),
            "excluded_records": excluded,
            "admitted_records": admitted,
        },
        "totals": dict(totals),
        "matched_rejections_by_reason": {k: dict(v) for k, v in sorted(
            matched_by_reason.items(), key=lambda kv: -kv[1][CLEAN])},
        "matched_rejections_by_paper": {k: dict(v) for k, v in sorted(matched_by_paper.items())},
        "accepted_matching_curated": dict(accepted_matched),
        "distinct_curated_reactions_rejected": {
            k: sorted(v) for k, v in curated_hit_rejected.items()},
        "distinct_curated_reactions_accepted": {
            k: sorted(v) for k, v in curated_hit_accepted.items()},
        "samples_by_reason": {k: v for k, v in samples.items()},
    }


def render(result: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("=" * 78)
    L.append("ORCH-724 ADMISSION AUDIT -- curated positives vs the gate's rejections")
    L.append("=" * 78)
    screen = result["curation_screen"]
    L.append("")
    L.append("CURATION CREDIBILITY SCREEN")
    L.append(f"  curated core reactions admitted as positives : {screen['admitted_positives']}")
    L.append(f"  excluded from the decision set              : {screen['excluded']}")
    reasons = collections.Counter(r["screen"] for r in screen["excluded_records"])
    for k, v in reasons.most_common():
        L.append(f"      {k:<45} {v}")
    L.append("")
    L.append(f"LEGS SCORED (target paper has curated positives): {result['legs_scored']}")
    t = result["totals"]
    for pop in (CLEAN, TRUNCATED):
        L.append(f"  [{pop}] legs={t.get('legs_' + pop, 0)} "
                 f"accepted={t.get('accepted_' + pop, 0)} rejected={t.get('rejected_' + pop, 0)} "
                 f"rejected_matching_curated={t.get('rejected_matched_curated_' + pop, 0)} "
                 f"accepted_matching_curated={t.get('accepted_matched_curated_' + pop, 0)}")
    L.append("")
    L.append("REJECTIONS OF CURATED-CORRECT CHEMISTRY, BY REASON CODE")
    L.append("  (a chemistry claim, NOT a verdict that the gate was wrong)")
    L.append(f"  {'reason code':<45} {CLEAN:>12} {TRUNCATED:>11}")
    for code, counts in result["matched_rejections_by_reason"].items():
        L.append(f"  {code:<45} {counts.get(CLEAN, 0):>12} {counts.get(TRUNCATED, 0):>11}")
    L.append("")
    L.append("DISTINCT CURATED REACTIONS REFUSED AT LEAST ONCE")
    for pop in (CLEAN, TRUNCATED):
        got = result["distinct_curated_reactions_rejected"].get(pop, [])
        L.append(f"  [{pop}] {len(got)}: {', '.join(got)}")
    L.append("")
    L.append("DISTINCT CURATED REACTIONS ADMITTED AT LEAST ONCE")
    for pop in (CLEAN, TRUNCATED):
        got = result["distinct_curated_reactions_accepted"].get(pop, [])
        L.append(f"  [{pop}] {len(got)}: {', '.join(got) if got else '(none)'}")
    L.append("")
    L.append("PER-PAPER matched rejections")
    for paper, counts in result["matched_rejections_by_paper"].items():
        L.append(f"  {paper:<14} untruncated={counts.get(CLEAN, 0):<6} truncated={counts.get(TRUNCATED, 0)}")
    return "\n".join(L)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root")
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--sample-per-reason", type=int, default=12)
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    result = audit(root, args.sample_per_reason)
    print(render(result))
    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

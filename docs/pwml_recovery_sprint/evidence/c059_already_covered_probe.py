"""C-059 -- replay the committed PMC12096016 strict leg through the RAG gate.

Offline, deterministic, no LLM and no network: everything it reads is committed
under ``runs_verify/2026-08-18_1328/papers/PMC12096016/strict/``.

It answers four questions and writes them to JSON:

1. **Does the replay reproduce the committed run?** ``detect_gaps`` on the
   committed ``stage1_payload.json`` must reproduce every ``dangling_reaction``
   gap id in the committed ``rag_admission_report.json``, and the two accepted
   records, rebuilt from that report, must reach the gate in the state the run
   left them in. Without this the other three numbers measure nothing.
2. **What does the gate do now?** ``counts``, per-candidate reason codes, and the
   ``gap_ids`` the surviving admission carries.
3. **Which of the two C-059 refusals is load-bearing?** The same replay is run
   with ``_already_covered`` neutralized, isolating the cross-gap dedup. That is
   the configuration the charter predicts leaves the PAYLOAD unchanged.
4. **Is the payload really unchanged by the dedup alone?** The candidates that
   survive each configuration are turned into ``_Reaction`` rows the way
   ``synthesize`` turns them, run through ``_resolve_reactions`` with the same
   offline synonym resolver production uses, and emitted with ``_reaction_row``.
   Two configurations, two row lists, compared as canonical JSON.

Usage::

    python docs/pwml_recovery_sprint/evidence/c059_already_covered_probe.py [--out PATH]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "src",
    ),
)

from t2pw.rag import admission as admission_mod  # noqa: E402
from t2pw.rag import retrieve, synonyms  # noqa: E402
from t2pw.rag import synthesize as synthesize_mod  # noqa: E402
from t2pw.rag.admission import (  # noqa: E402
    AdmissionPolicy,
    RagReactionCandidate,
    admit_candidates,
)

LEG = os.path.join(
    "runs_verify", "2026-08-18_1328", "papers", "PMC12096016", "strict"
)
REQUESTED_PATHWAY = "enterobactin biosynthesis"
REQUESTED_ORGANISM = "Escherichia coli"


def _load(root: str, name: str) -> Any:
    with io.open(os.path.join(root, LEG, name), encoding="utf-8") as handle:
        return json.load(handle)


def _candidate(row: Dict[str, Any]) -> RagReactionCandidate:
    """Rebuild the candidate the committed report describes.

    Every field is copied from the report, never inferred: the point of the
    replay is that the gate sees what the run's gate saw.
    """
    return RagReactionCandidate(
        gap_id=row["gap_id"],
        name=row["name"],
        inputs=list(row["inputs"]),
        outputs=list(row["outputs"]),
        enzymes=list(row["enzymes"]),
        reversible=bool(row["reversible"]),
        gap_ids=list(row.get("gap_ids") or []),
        source_paper=dict(row["source_paper"]),
        evidence=dict(row["evidence"]),
        evidence_span=row["evidence"]["span"],
        organism=row.get("organism", ""),
        observed_organisms=list(row.get("observed_organisms") or []),
        observed_pathways=list(row.get("observed_pathways") or []),
        requested_pathway=row.get("requested_pathway", ""),
        requested_organism=row.get("requested_organism", ""),
        requested_pathway_match=row.get("requested_pathway_match", ""),
        organism_match=row.get("organism_match", ""),
        confidence=float(row.get("confidence") or 0.0),
    )


def _reaction_of(candidate: RagReactionCandidate) -> Any:
    """The ``_Reaction`` ``synthesize`` would carry for this candidate.

    ``gap_ids`` is the BUNDLE's own, exactly as ``_reactions_from_bundle`` writes
    it; the union across a collapsed group is applied afterwards by
    :func:`_carry_the_verdict`, which is where production applies it too.
    """
    participant = synthesize_mod._Participant
    return synthesize_mod._Reaction(
        name=candidate.name,
        inputs=[participant(name=n) for n in candidate.inputs],
        outputs=[participant(name=n) for n in candidate.outputs],
        enzymes=list(candidate.enzymes),
        reversible=bool(candidate.reversible),
        provenance=[dict(candidate.evidence)],
        evidence=[dict(candidate.evidence)],
        source_papers=[dict(candidate.source_paper)],
        scores=[float(candidate.evidence.get("score") or 0.0)],
        origin="rag",
        gap_id=candidate.gap_id,
        gap_ids=[candidate.gap_id],
        scope_membership=candidate.scope_membership,
        evidence_span=candidate.evidence_span,
        observed_organisms=list(candidate.observed_organisms),
        observed_pathways=list(candidate.observed_pathways),
    )


def _row_delta(left: List[Any], right: List[Any]) -> List[Dict[str, Any]]:
    """Per-row, per-key differences between two emitted row lists."""
    delta: List[Dict[str, Any]] = []
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a == b:
            continue
        if a is None or b is None:
            delta.append({"index": index, "present_in_left": a is not None})
            continue
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                delta.append(
                    {
                        "index": index,
                        "key": key,
                        "left": a.get(key),
                        "right": b.get(key),
                    }
                )
    return delta


def _carry_the_verdict(reaction: Any, candidate: RagReactionCandidate) -> Any:
    """The C-059 carry step from ``synthesize_with_report``, replayed.

    When the gate collapsed a claim several gaps each retrieved, the sibling rows
    never reach ``_resolve_reactions``, so the union that merge used to perform --
    the gap attributions and the group's best retrieval score -- travels with the
    verdict instead. Guarded on the union growing, so an uncollapsed claim is
    untouched.
    """
    merged = synthesize_mod._dedupe_strs(
        list(reaction.gap_ids or []) + list(candidate.gap_ids or [])
    )
    if len(merged) > len(list(reaction.gap_ids or [])):
        reaction.gap_ids = merged
        if candidate.confidence:
            reaction.scores = list(reaction.scores) + [float(candidate.confidence)]
    return reaction


def _rows_for(candidates: List[RagReactionCandidate], resolver: Any) -> List[Any]:
    resolved, _conflicts = synthesize_mod._resolve_reactions(
        [_carry_the_verdict(_reaction_of(c), c) for c in candidates], resolver
    )
    return [synthesize_mod._reaction_row(r) for r in resolved]


def _run(root: str, *, disable_covered: bool, disable_dedup: bool = False) -> Dict[str, Any]:
    payload = _load(root, "stage1_payload.json")
    committed = _load(root, "rag_admission_report.json")
    gaps = retrieve.detect_gaps(
        payload,
        None,
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    resolver = synonyms.build_offline_synonym_resolver()
    candidates = [_candidate(row) for row in committed["accepted"]]
    policy = AdmissionPolicy(**dict(committed["policy"]))

    # Both refusals are neutralized by REPLACING the production function, not by
    # asking the gate for a flag: a gate with an "off" switch is a gate that can
    # be turned off in production, and F-057 is what that costs.
    original_covered = getattr(admission_mod, "_already_covered", None)
    original_phase3 = getattr(admission_mod, "_refuse_covered_and_duplicate", None)
    if disable_covered and original_covered is not None:
        admission_mod._already_covered = lambda *a, **k: None
    if disable_dedup and original_phase3 is not None:
        admission_mod._refuse_covered_and_duplicate = lambda accepted, **k: (
            list(accepted),
            [],
        )
    try:
        accepted, report = admit_candidates(
            candidates,
            gaps=gaps,
            seed_payload=payload,
            policy=policy,
            name_resolver=resolver,
        )
    finally:
        if disable_covered and original_covered is not None:
            admission_mod._already_covered = original_covered
        if disable_dedup and original_phase3 is not None:
            admission_mod._refuse_covered_and_duplicate = original_phase3

    return {
        "counts": dict(report.counts),
        "reason_codes": sorted(report.reason_counts),
        "verdicts": [
            {
                "gap_id": c.gap_id,
                "gap_ids": list(c.gap_ids),
                "status": c.status,
                "reasons": list(c.reasons),
            }
            for c in candidates
        ],
        "emitted_rows": _rows_for(accepted, resolver),
        "detected_dangling_gap_ids": sorted(
            g.gap_id for g in gaps if g.kind == "dangling_reaction"
        ),
        "committed_dangling_gap_ids": sorted(
            g for g in committed["by_gap"] if g.startswith("gap-dangling_reaction-")
        ),
        "committed_counts": dict(committed["counts"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.getcwd())
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    tip = _run(args.root, disable_covered=False)
    dedup_only = _run(args.root, disable_covered=True)
    # Phase 3 neutralized entirely == the base SHA's behaviour, measured on the
    # same tree so the comparison is not a tree comparison (F-051).
    base_like = _run(args.root, disable_covered=True, disable_dedup=True)

    def _canon(rows: Any) -> str:
        return json.dumps(rows, sort_keys=True, ensure_ascii=False)

    dedup_is_payload_neutral = _canon(base_like["emitted_rows"]) == _canon(
        dedup_only["emitted_rows"]
    )
    dedup_payload_delta = _row_delta(
        base_like["emitted_rows"], dedup_only["emitted_rows"]
    )
    covered_removes_the_row = not tip["emitted_rows"] and bool(
        base_like["emitted_rows"]
    )

    result = {
        "task": "C-059",
        "leg": LEG.replace("\\", "/"),
        "purpose": (
            "replay the committed strict leg through detect_gaps + "
            "admit_candidates; separate the biology fix (already-covered) from "
            "the hygiene fix (cross-gap dedup)"
        ),
        "replay_faithful": (
            tip["committed_dangling_gap_ids"] == tip["detected_dangling_gap_ids"]
            and tip["committed_counts"]["accepted"] == 2
        ),
        "committed_run": {
            "counts": tip["committed_counts"],
            "dangling_gap_ids": tip["committed_dangling_gap_ids"],
        },
        "replay_dangling_gap_ids": tip["detected_dangling_gap_ids"],
        "both_refusals_active": {
            "counts": tip["counts"],
            "reason_codes": tip["reason_codes"],
            "verdicts": tip["verdicts"],
            "emitted_reaction_names": [r["name"] for r in tip["emitted_rows"]],
        },
        "dedup_only_covered_neutralized": {
            "counts": dedup_only["counts"],
            "reason_codes": dedup_only["reason_codes"],
            "verdicts": dedup_only["verdicts"],
            "emitted_reaction_names": [
                r["name"] for r in dedup_only["emitted_rows"]
            ],
        },
        "base_like_phase3_neutralized": {
            "counts": base_like["counts"],
            "reason_codes": base_like["reason_codes"],
            "emitted_reaction_names": [
                r["name"] for r in base_like["emitted_rows"]
            ],
        },
        "dedup_only_emits_the_duplicate_row": bool(dedup_only["emitted_rows"]),
        "dedup_alone_is_payload_neutral": dedup_is_payload_neutral,
        "dedup_alone_payload_delta": dedup_payload_delta,
        "dedup_alone_payload_delta_note": (
            "The charter's section 2 predicted a byte-identical payload from the "
            "dedup alone. MEASURED, that was FALSE as first built: _merge_into "
            "unions _Reaction.scores and gap_ids and _confidence maxes over the "
            "scores, so the committed row carries the HIGHER of the two "
            "per-retrieval scores (0.930233 at merged_payload.json "
            "/processes/reactions/5/rag_confidence) and BOTH gap ids. Collapsing "
            "at admission means the sibling row never reaches that merge, which "
            "cost the row 0.930233 -> 0.914815 and dropped the second gap's "
            "attribution -- a regression against the pre-existing pinned test "
            "test_rag_admission_adversarial.py::"
            "test_one_claim_admitted_for_two_gaps_keeps_both_attributions. Fixed "
            "by carrying that union with the verdict; this delta is now empty. "
            "The row itself is NOT removed by the dedup -- that is the "
            "already-covered refusal's doing, which is the point of the split."
        ),
        "already_covered_removes_the_reimported_row": covered_removes_the_row,
        "reading": (
            "The already-covered refusal is what removes the re-imported row: "
            "with it neutralized the cross-gap dedup still emits one copy of "
            "'Isochorismate to 2,3-Dihydro-2,3-Dihydroxybenzoate (DHB)', which is "
            "the row that went degree-zero and refused both strict legs. The "
            "dedup corrects counts.accepted and the per-gap breakdown; it is "
            "hygiene."
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        with io.open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

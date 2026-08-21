"""C-071 G9 probe: is an actor that its own cited span does not name still releasable?

Run at the base SHA and again at the tip; the two JSON reports are the behavioural delta.
**Symbol absence is not proof** -- no source is read and no tip-only symbol is touched on
the base arm. Every arm calls the shipped ``evaluate_production_semantics`` ->
``semantic_verdict`` -> ``classify_release_status`` chain, all three of which exist at
base, and records what they returned.

The input is the COMMITTED ``final_mapped.json`` of F-079's leg,
``runs_verify/2026-08-18_1328/papers/PMC12096016/research``, read by absolute path from
the branch worktree on both arms with its sha256 recorded on both -- so the report itself
proves the arms scored identical bytes and only the code differed.

Arms. ``f079_leg`` is the G9 delta. ``neutralized_*`` are **RULING 13 non-vacuity**: two
ways to make the detector examine zero rows without removing it, each recording
``ok`` beside ``applicable`` because that pair is what distinguishes a detector that
stopped looking from one that looked and found nothing. ``match_rule`` is the over-fire /
under-fire table as verdicts of the shipped predicate rather than as prose. The last two
are tip-only and record that fact at base rather than skipping silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

#: Read from the BRANCH worktree on both arms: ``c045b_base_tree.py``'s PATHSPEC does not
#: carry ``runs_verify/`` (Finding H-3) and this card changes nothing under it, so one
#: file serves both halves and the recorded hash proves it.
LEG = os.path.join(
    ROOT, "runs_verify", "2026-08-18_1328", "papers", "PMC12096016", "research",
    "final_mapped.json",
)
#: ``observed_pathways[0]`` / ``observed_organisms[0]`` of that run's committed
#: ``manifest.jsonl`` row, restated under the two keys the PINNED derivation reads
#: (D-042 section 2).
OBSERVED_CONTEXT = {
    "pathway_name": "enterobactin biosynthesis",
    "likely_organism": "Escherichia coli",
}
CHECK_NAME = "actor_named_in_its_own_cited_span"

SP: Any = None
RS: Any = None
EA: Any = None


def _bind_source(src: str) -> str:
    """Import ``t2pw`` from ``src`` and prove that is where it came from."""

    global SP, RS, EA
    src = os.path.abspath(src)
    sys.path.insert(0, src)
    from t2pw.bench import semantic_production as sp_module
    from t2pw.pipeline import entity_admission as ea_module
    from t2pw.pipeline import release_status as rs_module

    for module in (sp_module, rs_module, ea_module):
        loaded = os.path.abspath(module.__file__)
        if not loaded.startswith(src):
            raise SystemExit(f"REFUSED: {module.__name__} loaded from {loaded}, not {src}")
    SP, RS, EA = sp_module, rs_module, ea_module
    return os.path.abspath(sp_module.__file__)


def _payload() -> Tuple[Dict[str, Any], str]:
    raw = open(LEG, "rb").read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _coverage() -> Dict[str, Any]:
    """Clears on its own, so the SEMANTIC cap is the only thing that can move a status."""

    return {
        "requested_core_declared": True, "surviving_processes": 3,
        "minimum_core_satisfied": True, "reasons": [],
    }


def _score(payload: Dict[str, Any]) -> Dict[str, Any]:
    """One pass of the shipped production chain. Arity-tolerant across base and tip."""

    requested = EA.pathway_context_from_stage_zero(dict(OBSERVED_CONTEXT))
    report = SP.evaluate_production_semantics(
        payload,
        requested_pathway=requested.requested_pathway,
        requested_organism=requested.organism,
        mode="research",
    )
    verdict = RS.semantic_verdict(report)
    extra = {"semantic_check_evaluability": verdict[3]} if len(verdict) > 3 else {}
    status = RS.classify_release_status(
        _coverage(), strict_gates_passed=True,
        semantic_evaluation=verdict[0], semantic_not_evaluated_reason=verdict[1],
        semantic_failed_checks=verdict[2], **extra,
    )
    check = report.checks.get(CHECK_NAME)
    return {
        "requested_pathway": requested.requested_pathway,
        "requested_organism": requested.organism,
        "evaluation": verdict[0],
        "failed_checks": list(verdict[2]),
        "status": status.status,
        "strict_acceptance_eligible": status.strict_acceptance_eligible,
        "reasons": list(status.reasons),
        "check_present": check is not None,
        "check_ok": None if check is None else bool(check.ok),
        "check_applicable": None if check is None else bool(check.applicable),
        "check_summary": "" if check is None else check.summary,
        "findings": [] if check is None else [
            {"pointer": f.get("pointer"), "entity": f.get("entity"),
             "cited_span": str(f.get("cited_span"))[:200]}
            for f in check.findings
        ],
    }


def _neutralized(attribute: str, value: Any) -> Dict[str, Any]:
    """Score the same leg with one of the detector's own parameters disarmed."""

    if not hasattr(SP, attribute):
        return {"skipped": f"{attribute} does not exist on this source"}
    original = getattr(SP, attribute)
    setattr(SP, attribute, value)
    try:
        return _score(_payload()[0])
    finally:
        setattr(SP, attribute, original)


#: (label, actor name, cited span, expected verdict). ``under_*`` is what a raw substring
#: test accepts, shipping the fabricated actor; ``over_*`` is what a whole-name test
#: rejects, demoting a leg over a naming convention. Verbatim committed rows except the
#: two marked synthetic, which cover the three-character-symbol floor in both directions.
MATCH_RULE_CASES: List[Tuple[str, str, str, bool]] = [
    ("under_f079_transporter", "EntE",
     "Enterobactin is produced in the cytoplasm and secreted to the extracellular "
     "environment by a TolC-dependent process", False),
    ("under_f079_self_referential", "EntE",
     "EntF-catalyzed enterobactin assembly with L-serine Enterobactin production from "
     "activated 2,3-DHB t", False),
    ("under_tonb_uptake", "EntE",
     "the ferric-enterobactin complex is then imported via TonB-dependent uptake", False),
    ("under_no_symbol_at_all", "EntA",
     "2,3-DHB is produced from chorismate by sequential reactions catalyzed by three "
     "enzymes", False),
    ("over_uniprot_recommended_name", "Serine hydroxymethyltransferase, mitochondrial",
     "serine is cleaved to glycine in the reaction catalyzed by serine "
     "hydroxymethyltransferase", True),
    ("over_generated_complex_wrapper", "ALAS2 complex",
     "ALAS2 mediates the condensation of glycine and succinyl-CoA", True),
    ("over_parenthesised_symbol", "oxidoreductase (entA)",
     "EntA (2,3-dihydro-2,3-dihydroxybenzoate dehydrogenase; EC 1.3.1.28)", True),
    ("over_hyphen_and_case", "EntC",
     "EntC-catalyzed isomerization of chorismate to isochorismate", True),
    ("true_positive_synthetic_three_letter_symbol", "Fur",
     "expression is repressed by Lon under iron-replete conditions", False),
    ("true_negative_synthetic_three_letter_symbol", "Fur",
     "expression is repressed by Fur under iron-replete conditions", True),
]


def _match_rule() -> Dict[str, Any]:
    if not hasattr(SP, "_actor_named_in_span"):
        return {"skipped": "the predicate does not exist on this source"}
    rows = {}
    for label, name, span, expected in MATCH_RULE_CASES:
        observed = SP._actor_named_in_span(name, [span])
        naive = name.casefold() in span.casefold()
        rows[label] = {
            "entity": name, "expected": expected, "observed": observed,
            "agrees": observed is expected, "raw_substring_would_say": naive,
        }
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", required=True, choices=("base", "tip"))
    args = parser.parse_args(argv)

    module_path = _bind_source(args.src)
    payload, digest = _payload()
    report: Dict[str, Any] = {
        "arm": args.arm,
        "semantic_production_loaded_from": module_path,
        "gating_set": list(RS.SEMANTIC_GATING_CHECKS),
        "payload": os.path.relpath(LEG, ROOT).replace("\\", "/"),
        "payload_sha256": digest,
        "f079_leg": _score(payload),
        "neutralized_no_actor_keys": _neutralized("_ACTOR_KEYS", ()),
        "neutralized_token_floor": _neutralized("_MIN_IDENTIFYING_TOKEN", 99),
        "match_rule": _match_rule(),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ("arm", "payload_sha256")}))
    print(json.dumps(report["f079_leg"], indent=2)[:1400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

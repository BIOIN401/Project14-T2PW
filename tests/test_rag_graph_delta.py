"""C-021 — validating one RAG round's graph delta (PRODUCT_CONTRACT §10/§3/§11, D-008).

NEW acceptance tests. ``t2pw.rag.graph_delta`` does not exist at the base SHA, so these
replace NO prior observable behaviour and are NOT G9 regression evidence.
"""

from __future__ import annotations

import ast
import inspect
from copy import deepcopy

import pytest

from t2pw.pipeline.lineage import LINEAGE_KEY
from t2pw.rag.admission import SCOPE_ADMITTED, SCOPE_REJECTED
from t2pw.rag.graph_delta import (
    FALSE, NOT_EVALUATED, REENTRY_STAGES, RULE_ADMISSION_VERDICT, RULE_DUPLICATE_CLAIM,
    RULE_EVIDENCE_RETENTION, RULE_GAP_ATTRIBUTION, RULE_GAP_NOT_DETECTED,
    RULE_MUTATED_ELEMENT, RULE_PROVENANCE, RULE_REJECTED_CLAIM, RULE_REMOVED_ELEMENT,
    RULE_ROUND_REENTRY, RULE_UNCLASSIFIED, RULES, TRUE, RoundRecord, validate_graph_delta,
)
from t2pw.rag.loop_policy import SeenClaims, claim_identity_key

GAP = "gap-missing_step-1a2b3c4d"
SPAN = "LpxC deacetylates UDP-3-O-acyl-GlcNAc."
PAPER = "PMC11046580"
#: A valid C-015 entry: the six § 3 facts, validated by ``LineageEntry`` itself.
ENTRY = {"stage": "rag_admission", "origin": "rag_literature", "support": "direct",
         "paper_explicit": "not_explicit", "reason": "fills the detected gap",
         "review_required": False, "sources": [{"source_id": PAPER}]}
CORE = {"name": "core step", "inputs": ["x", "w"], "outputs": ["y"], "enzymes": ["LpxA"]}
OTHER = {"name": "second step", "inputs": ["y"], "outputs": ["z"]}
ELEMENT = "processes.reactions[deacetylation]"


def added(**over):
    """One admissible RAG-added reaction. ``over`` changes exactly one thing; a value of
    ``None`` DROPS that key, which is how the "arrives without X" cases are built."""
    row = {"name": "deacetylation", "inputs": ["a"], "outputs": ["b"],
           "enzymes": ["LpxC"], "scope_membership": SCOPE_ADMITTED, "evidence": SPAN,
           "source_papers": [{"source_id": PAPER}], LINEAGE_KEY: [dict(ENTRY)],
           "rag_provenance": {"source_id": PAPER, "chunk_id": "c1", "gap_id": GAP}}
    row.update(over)
    return {k: v for k, v in row.items() if v is not None}


def graphs(*rows, before=(CORE,)):
    """``(payload before the round, payload after it)``; ``rows`` are what it ADDED."""
    seed = {"entities": {"compounds": [{"name": "x"}]},
            "processes": {"reactions": [dict(r) for r in before]}}
    after = deepcopy(seed)
    after["processes"]["reactions"].extend(rows)
    return seed, after


def record(**over):  # a round that detected GAP and re-entered all five D-008 stages
    return RoundRecord(**{"detected_gap_ids": frozenset({GAP}),
                          "reentered": dict.fromkeys(REENTRY_STAGES, True), **over})


def verdict(*rows, rec=None, **kw):
    return validate_graph_delta(*graphs(*rows, **kw), rec if rec is not None else record())


def test_an_admissible_delta_names_its_gap_passes_admission_and_keeps_its_evidence():
    result = verdict(added())
    assert (result.admissible, result.violations, result.removed) == (True, (), ())
    assert result.added == (ELEMENT,)
    assert set(result.reentry.values()) == {TRUE}
    # §4.2: the pre-existing core rows carry no lineage, and that is NOT a violation.
    assert LINEAGE_KEY not in CORE


ONE_RULE = [
    (RULE_GAP_ATTRIBUTION, {"rag_provenance": {"source_id": PAPER, "chunk_id": "c1"}}),
    (RULE_GAP_NOT_DETECTED, {"rag_provenance": {"source_id": PAPER, "chunk_id": "c1",
                                                "gap_id": "gap-never-detected"}}),
    (RULE_UNCLASSIFIED, {"scope_membership": None}),
    (RULE_ADMISSION_VERDICT, {"scope_membership": SCOPE_REJECTED}),
    (RULE_PROVENANCE, {LINEAGE_KEY: None}),
    # A stage outside C-015's closed vocabulary: ``LineageError`` must NOT leak.
    (RULE_PROVENANCE, {LINEAGE_KEY: [dict(ENTRY, stage="pwml_export")]}),
    (RULE_EVIDENCE_RETENTION, {"evidence": None,
                               "rag_provenance": {"source_id": PAPER, "gap_id": GAP}}),
    (RULE_EVIDENCE_RETENTION, {"source_papers": None,
                               "rag_provenance": {"chunk_id": "c1", "gap_id": GAP}}),
]


@pytest.mark.parametrize("rule, override", ONE_RULE)
def test_each_added_reaction_rule_fires_alone_on_the_element_that_broke_it(rule, override):
    """One rule per mutation. A rule with no failing case is a rule not really enforced."""
    result = verdict(added(**override))
    assert (result.admissible, result.rules()) == (False, (rule,))
    assert {v.element for v in result.violations} == {ELEMENT}
    assert all(v.observed and v.required for v in result.violations)


def test_a_row_that_lost_what_its_own_claim_carried_is_reported_as_a_loss():
    """§10 retention across the delta: the row is well-formed, but its claim said more."""
    claim = {"inputs": ["a"], "outputs": ["b"], "enzymes": ["LpxC"],
             "evidence_span": SPAN, "source_paper": {"source_id": "PMC12452463"}}
    result = verdict(added(evidence="a paraphrase"), rec=record(claims=[claim]))
    assert result.rules() == (RULE_EVIDENCE_RETENTION,)
    assert sorted(v.detail for v in result.violations) == ["evidence_span", "source_paper"]


def test_a_seen_or_a_rejected_claim_is_refused_however_the_reaction_is_re_worded():
    """§10 convergence: dedupe is against ALL claims ever seen, rejected ones included."""
    row = added(name="R4 demethylation")  # same chemistry, another paper's wording
    key = claim_identity_key(row)
    seen, _ = SeenClaims().observe([{"inputs": ["a"], "outputs": ["b"],
                                     "enzymes": ["LpxC"]}])
    assert verdict(row, rec=record(seen_claims=seen)).rules() == (RULE_DUPLICATE_CLAIM,)
    rejected = verdict(row, rec=record(seen_claims=seen,
                                       rejected_claim_keys=frozenset({key})))
    assert rejected.rules() == (RULE_REJECTED_CLAIM,)  # the specific rule, reported alone
    assert rejected.violations[0].detail == key


def test_an_added_entity_is_held_to_provenance_and_evidence_but_not_to_a_gap():
    before, after = graphs()
    after["entities"]["compounds"].append({"name": "UDP-3-O-acyl-GlcNAc"})
    result = validate_graph_delta(before, after, record())
    assert result.added == ("entities.compounds[udp3oacylglcnac]",)
    assert result.rules() == (RULE_EVIDENCE_RETENTION, RULE_PROVENANCE)


def test_removing_a_pre_existing_reaction_is_a_violation_and_is_never_repaired():
    before, after = graphs(before=(CORE, OTHER))
    after["processes"]["reactions"] = [dict(CORE)]
    result = validate_graph_delta(before, after, record())
    assert result.rules() == (RULE_REMOVED_ELEMENT,)
    assert result.removed == ("processes.reactions[second step]",)
    assert len(after["processes"]["reactions"]) == 1  # reported, never put back


@pytest.mark.parametrize("field, value", [
    ("inputs", ["x", "CONTAMINANT"]), ("outputs", ["z"]), ("enzymes", ["LpxC"]),
    ("mapped_ids", {"uniprot": "P0A722"}), ("scope_membership", "out_of_scope")])
def test_mutating_a_pre_existing_rows_biology_is_a_violation(field, value):
    before, after = graphs()
    after["processes"]["reactions"][0][field] = value
    result = validate_graph_delta(before, after, record())
    assert result.rules() == (RULE_MUTATED_ELEMENT,)
    fired = result.violations[0]  # observed vs required, in canonical order-free form
    assert (fired.detail, fired.observed != fired.required) == (field, True)
    assert (fired.required == "absent") is (field not in CORE)


def test_reordering_participants_or_attaching_evidence_is_not_a_biology_change():
    before, after = graphs()
    row = after["processes"]["reactions"][0]
    row["inputs"] = list(reversed(row["inputs"]))
    row["evidence"], row["rag_confidence"] = "a passage the round attached", 0.9
    assert validate_graph_delta(before, after, record()).admissible is True


def test_d008_reentry_is_three_valued_and_not_evaluated_is_never_false():
    silent = verdict(added(), rec=record(reentered={}))
    assert set(silent.reentry.values()) == {NOT_EVALUATED}
    assert silent.admissible is True and NOT_EVALUATED != FALSE  # §11: never collapsed
    missed = verdict(added(), rec=record(
        reentered=dict(dict.fromkeys(REENTRY_STAGES, True), gates=False)))
    assert missed.rules() == (RULE_ROUND_REENTRY,)
    assert missed.reentry == dict(dict.fromkeys(REENTRY_STAGES, TRUE), gates=FALSE)
    assert missed.violations[0].detail == "gates"


def test_the_verdict_is_deterministic_and_neither_input_is_mutated():
    before, after = graphs(added(scope_membership=SCOPE_REJECTED),
                           added(name="two", **{LINEAGE_KEY: None}))
    rec = record(reentered={"gates": False})
    frozen = deepcopy((before, after))
    first, second = (validate_graph_delta(before, after, rec) for _ in range(2))
    assert first == second and len(first.violations) > 2
    order = [RULES.index(v.rule) for v in first.violations]
    assert order == sorted(order)  # deterministic: RULES order, then element
    assert (before, after) == frozen  # read-only: neither graph was touched


def test_the_module_imports_no_clock_no_io_and_no_retrieval_or_llm_stack():
    """Also pins the reuse: claim identity and provenance are IMPORTED, not re-written."""
    tree = ast.parse(inspect.getsource(inspect.getmodule(validate_graph_delta)))
    imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                 for a in n.names}
    assert imported == {"__future__", "collections.abc", "dataclasses", "typing",
                        "t2pw.pipeline.lineage", "t2pw.pipeline.process_normalizer",
                        "t2pw.rag.admission", "t2pw.rag.loop_policy"}

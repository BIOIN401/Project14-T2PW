"""C-021 — validating one RAG round's graph delta (PRODUCT_CONTRACT §10/§3/§11, D-008).

NEW acceptance tests. ``t2pw.rag.graph_delta`` does not exist at the C-021 base SHA, so
these replace NO prior observable behaviour and are NOT G9 regression evidence.

The H-008 block at the end is the exception: ``validate_graph_delta`` DOES exist at that
task's base SHA and returns ``admissible=True`` for both deltas it pins, so those two are
genuine G9 regression evidence.
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


# ======================================================================================
# H-008 -- the two fail-open retention defects (§10 retention, §3 lineage).
# ======================================================================================

#: A pre-existing row already carrying everything § 10 says must be retained. Every key it
#: adds to ``CORE`` except ``evidence_span`` is in ``NON_BIOLOGICAL``, so ``_biology``
#: discards it and the mutation rule cannot see it leave -- that is defect D1 exactly.
RICH = dict(CORE, evidence=[{"text": SPAN}, {"text": "and the fold is Zn-dependent"}],
            evidence_span=SPAN, source_papers=[{"source_id": PAPER}],
            rag_provenance={"source_id": PAPER, "chunk_id": "c1"},
            **{LINEAGE_KEY: [dict(ENTRY)]})
#: Dropping all four at once is silent under every rule the base implementation has.
CARRIERS = {"evidence": None, "source_papers": None, "rag_provenance": None,
            LINEAGE_KEY: None}
#: Unrelated text put where the claim's passage was.
SUBSTITUTE = "Ribosome assembly proceeds through a 30S intermediate."
RICH_ELEMENT = "processes.reactions[core step]"


def rewritten(**over):
    """``(before, after)`` where the round rewrote the ONE pre-existing ``RICH`` row.
    ``over`` changes what that row says; a value of ``None`` DROPS that key."""
    before, after = graphs(before=(RICH,))
    row = dict(RICH, **over)
    after["processes"]["reactions"][0] = {k: v for k, v in row.items() if v is not None}
    return before, after


def test_a_pre_existing_row_may_not_silently_lose_its_evidence_and_lineage():
    """D1, G9. ``_biology`` drops every key in ``NON_BIOLOGICAL``, so stripping the
    passage, the source identifier, the chunk pointer AND the lineage off a pre-existing
    row changes nothing the mutation rule compares. Base verdict: admissible, no
    violations. § 10 retention and § 3 attribution both say otherwise."""
    result = validate_graph_delta(*rewritten(**CARRIERS), record())
    assert (result.admissible, result.rules(), len(result.violations)) == (
        False, (RULE_EVIDENCE_RETENTION, RULE_PROVENANCE), 4)
    assert sorted(v.detail for v in result.violations) == [
        "chunk_id", "evidence", LINEAGE_KEY, "source_paper"]
    assert {v.element for v in result.violations} == {RICH_ELEMENT}
    assert (result.added, result.removed) == ((), ())  # a PAIRED row, not a removal


def test_an_added_row_may_not_substitute_an_unrelated_passage_for_its_claims():
    """D2, G9. ``_evidence_violations`` throws the claim's passage and chunk pointer away
    (``_, _, claim_ids, claim_span``) and asks only whether SOME text exists, so a row
    that keeps the claim's source id and span passes at the base with a swapped passage
    -- the retention § 10 requires reduced to the presence of any string at all."""
    claim = {"inputs": ["a"], "outputs": ["b"], "enzymes": ["LpxC"], "evidence_span": SPAN,
             "evidence": [{"text": SPAN}], "source_paper": {"source_id": PAPER},
             "rag_provenance": {"source_id": PAPER, "chunk_id": "c1"}}
    result = verdict(added(evidence=SUBSTITUTE, evidence_span=SPAN,
                           rag_provenance={"source_id": PAPER, "chunk_id": "c99",
                                           "gap_id": GAP}),
                     rec=record(claims=[claim]))
    assert (result.admissible, result.rules(), len(result.violations)) == (
        False, (RULE_EVIDENCE_RETENTION,), 2)
    assert sorted(v.detail for v in result.violations) == ["chunk_id", "evidence"]
    assert [SPAN in v.observed for v in result.violations] == [False, True]  # names it


def test_a_pre_existing_row_that_loses_its_evidence_span_is_a_retention_loss():
    """§ 10's fourth fact. ``evidence_span`` is NOT in ``NON_BIOLOGICAL``, so the mutation
    rule co-fires -- that existing rule is untouched. The retention loss is what is new,
    and it is reported under its own rule rather than folded into that one."""
    result = validate_graph_delta(*rewritten(evidence_span=None), record())
    assert result.rules() == (RULE_MUTATED_ELEMENT, RULE_EVIDENCE_RETENTION)
    assert [v.detail for v in result.violations] == ["evidence_span", "evidence_span"]


@pytest.mark.parametrize("lineage", [
    [],                                  # the round emptied it
    [dict(ENTRY, stage="pwml_export")],  # outside C-015's closed vocabulary
])
def test_a_pre_existing_rows_lineage_may_be_neither_dropped_nor_corrupted(lineage):
    """§ 3 through C-015: a lineage is append-only. ``LineageError`` is REPORTED, never
    leaked -- a validator that raised would fail OPEN for the caller that caught it."""
    result = validate_graph_delta(*rewritten(**{LINEAGE_KEY: lineage}), record())
    assert (result.admissible, result.rules()) == (False, (RULE_PROVENANCE,))
    assert result.violations[0].detail == LINEAGE_KEY
    assert all(v.observed and v.required for v in result.violations)


def test_reordering_re_spacing_or_adding_evidence_is_never_reported_as_a_loss():
    """§ 10 asks for retention, not immutability, so the comparison is DIRECTIONAL and
    order-free: neither a reordered evidence list, nor re-spaced passage text, nor an
    appended source, pointer or lineage entry is a loss. ``_evidence_facts`` joins texts
    with ``" ".join``, so comparing THAT would have manufactured a false loss here."""
    before, after = rewritten(
        evidence=[{"text": "and the fold is\n   Zn-dependent"},
                  {"text": SPAN.replace(" ", "\n  ")},
                  {"text": "a passage this round attached"}],
        source_papers=[{"source_id": "PMC12452463"}, {"source_id": PAPER}],
        rag_provenance={"source_id": PAPER, "chunk_id": "c1", "gap_id": GAP},
        **{LINEAGE_KEY: [dict(ENTRY, reason="a second stage also touched it"),
                         dict(ENTRY)]})
    assert validate_graph_delta(before, after, record()).admissible is True


#: The list shape ``synthesize._merge_into`` unions, and the ONE string
#: ``conform._coerce_evidence_in_place`` joins it into for ``ReactionModel.evidence: str``.
#: Under a set of WHOLE values that join is a single element, so appending a passage would
#: replace it and read as a loss -- retention is containment in the join instead.
PASSAGES = [{"text": SPAN}, {"text": "and the fold is Zn-dependent"}]
JOINED = f"{SPAN} and the fold is Zn-dependent"
APPENDED = f"{JOINED} A third passage this round found."
REJOINED = f"and the fold is Zn-dependent {SPAN}"


@pytest.mark.parametrize("was, now, lost", [
    (JOINED, APPENDED, False),                    # a passage APPENDED to the join
    (PASSAGES, REJOINED, False),                  # the same passages re-joined, reordered
    (JOINED, JOINED.replace(" ", "\n  "), False),  # the same join, re-spaced
    (JOINED, SPAN, True),                         # one passage genuinely DROPPED
])
def test_a_joined_evidence_string_is_compared_by_containment(was, now, lost):
    """A pre-existing row in the coerced string shape. Adding to the join is not a loss;
    dropping a passage out of it still is."""
    before, after = graphs(before=(dict(RICH, evidence=was),))
    after["processes"]["reactions"][0]["evidence"] = now
    result = validate_graph_delta(before, after, record())
    assert result.admissible is not lost
    assert [v.detail for v in result.violations] == (["evidence"] if lost else [])


@pytest.mark.parametrize("row_evidence, lost", [
    (JOINED, False), (APPENDED, False), (REJOINED, False),
    (JOINED.replace(" ", "\n  "), False), (SPAN, True),
])
def test_a_coerced_evidence_string_is_measured_against_its_claim(row_evidence, lost):
    """The same fact on the added-row path over the REAL seam: the claim holds the passage
    LIST ``_merge_into`` unioned, the row the string ``conform`` joined it into. A passage
    present verbatim in that join is retained -- the false loss fired on exactly the
    multi-passage claim § 10 exists to protect -- and one dropped out of it still fires."""
    claim = {"inputs": ["a"], "outputs": ["b"], "enzymes": ["LpxC"], "evidence": PASSAGES,
             "rag_provenance": {"source_id": PAPER, "chunk_id": "c1"}}
    result = verdict(added(evidence=row_evidence), rec=record(claims=[claim]))
    assert result.admissible is not lost
    assert [v.detail for v in result.violations] == (["evidence"] if lost else [])


def test_a_lineage_that_was_already_unreadable_may_still_not_be_dropped():
    """A pre-existing row whose lineage does not parse still HAS an attribution, and the
    round may not answer that by deleting the carrier. REPAIRING it is allowed, which is
    why the branch cannot be "unreadable before, always report"."""
    before, after = rewritten()
    before["processes"]["reactions"][0][LINEAGE_KEY] = [dict(ENTRY, stage="pwml_export")]
    after["processes"]["reactions"][0].pop(LINEAGE_KEY)
    result = validate_graph_delta(before, after, record())
    assert (result.admissible, result.rules()) == (False, (RULE_PROVENANCE,))
    assert result.violations[0].observed == "dropped an unreadable lineage"
    after["processes"]["reactions"][0][LINEAGE_KEY] = [dict(ENTRY)]  # repaired, not dropped
    assert validate_graph_delta(before, after, record()).admissible is True

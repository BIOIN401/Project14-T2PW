"""Locked-reaction accounting, requested scope, and how a refusal is reported.

Three things that only look like bookkeeping:

* **Locks.** ``locked_reaction_filter_report`` is what the Stage 3 gate reads to
  decide whether every locked reaction is accounted for. Quarantining one while
  that report still counts it as exported makes the gate certify a reaction that
  is no longer in the payload.
* **Scope.** The final mapped payload carries no Stage-0 context, so a coverage
  check that only looks at the payload silently falls back to "relevance
  unjudgeable" -- the regime in which an unrelated survivor passes. The context
  has to arrive explicitly.
* **Refusals.** "issues: 3" tells a reviewer nothing. An empty core, a stranded
  lock and a surviving type overlap need different fixes and get different names.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.process_normalizer import (  # noqa: E402
    GateValidationError,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    AUXILIARY_ACCEPTED,
    CORE_ACCEPTED,
    collect_requested_core_terms,
    quarantine_and_close,
)

from test_strict_quarantine_contract_alignment import _base  # noqa: E402


# ── Locked reactions ────────────────────────────────────────────────────────


def _locked(payload: Dict[str, Any], index: int, lock_id: str) -> None:
    payload["processes"]["reactions"][index]["locked_reaction_id"] = lock_id


def _with_locks() -> Dict[str, Any]:
    payload = _base()
    _locked(payload, 0, "rxn_lock_001")
    _locked(payload, 1, "rxn_lock_002")
    payload["locked_reaction_filter_report"] = {
        "locked_reactions_found": 2,
        "exported_locked_reactions": 2,
        "quarantined_locked_reactions": 0,
        "unaccounted_locked_reactions": 0,
    }
    return payload


def test_all_locked_reactions_active_leaves_the_report_saying_so() -> None:
    result = quarantine_and_close(_with_locks(), strict_db=True)

    assert result.payload["locked_reaction_filter_report"] == {
        "locked_reactions_found": 2,
        "exported_locked_reactions": 2,
        "quarantined_locked_reactions": 0,
        "unaccounted_locked_reactions": 0,
    }
    assert result.ok is True
    run_strict_post_normalization_gates(result.payload)


def test_a_quarantined_locked_reaction_is_moved_into_the_canonical_accounting() -> None:
    """It may leave, but it may not leave the report claiming it was exported."""

    payload = _with_locks()
    # Reaction 1's only input is now undeclared, so it cannot be exported.
    payload["processes"]["reactions"][1]["inputs"] = ["never declared"]

    result = quarantine_and_close(payload, strict_db=True)
    report = result.payload["locked_reaction_filter_report"]

    assert report == {
        "locked_reactions_found": 2,
        "exported_locked_reactions": 1,
        "quarantined_locked_reactions": 1,
        "unaccounted_locked_reactions": 0,
    }
    records = result.payload["quarantined_locked_reactions"]
    assert [row["locked_reaction_id"] for row in records] == ["rxn_lock_002"]
    assert records[0]["quarantine_stage"] == "pre_export_strict_quarantine"
    # The row itself, so the lock can be restored rather than just mourned.
    assert records[0]["original_reaction"]["locked_reaction_id"] == "rxn_lock_002"
    # And the gate that reads this report is satisfied by it.
    run_strict_post_normalization_gates(result.payload)


def test_a_locked_reaction_never_vanishes_while_the_report_claims_it_exported() -> None:
    """The failure the recomputation exists to prevent, asserted directly."""

    payload = _with_locks()
    payload["processes"]["reactions"][1]["inputs"] = ["never declared"]

    result = quarantine_and_close(payload, strict_db=True)
    report = result.payload["locked_reaction_filter_report"]
    active = {
        row.get("locked_reaction_id")
        for row in result.payload["processes"]["reactions"]
        if row.get("locked_reaction_id")
    }
    quarantined = {
        row.get("locked_reaction_id") for row in result.payload["quarantined_locked_reactions"]
    }

    assert report["exported_locked_reactions"] == len(active)
    assert report["quarantined_locked_reactions"] == len(quarantined)
    assert not (active & quarantined)


def test_a_lock_that_is_neither_active_nor_quarantined_is_refused() -> None:
    """An unaccounted lock blocks, with its own refusal reason.

    ``locked_reactions_found`` only grows, so a lock this stage never saw shows up
    as unaccounted rather than being quietly forgotten -- and the Stage 3 gate
    would abort on it anyway, so refusing here is the honest place to say it.
    """

    payload = _with_locks()
    payload["locked_reaction_filter_report"]["locked_reactions_found"] = 5

    result = quarantine_and_close(payload, strict_db=True)

    assert result.ok is False
    assert "unaccounted_locked_reactions:3" in result.refusal_reasons
    assert result.payload["locked_reaction_filter_report"]["unaccounted_locked_reactions"] == 3
    with pytest.raises(GateValidationError):
        run_strict_post_normalization_gates(result.payload)


def test_a_payload_with_no_locks_grows_no_lock_report() -> None:
    """The accounting must stay invisible on the runs that never locked anything."""

    result = quarantine_and_close(_base(), strict_db=True)

    assert "locked_reaction_filter_report" not in result.payload
    assert "quarantined_locked_reactions" not in result.payload
    assert result.quarantine_report["locked_reactions"]["enabled"] is False


# ── Requested scope ─────────────────────────────────────────────────────────


STAGE_ZERO_CONTEXT = {
    "pathway_name": "Glutathione biosynthesis",
    "likely_organism": "Homo sapiens",
    "key_compounds": ["L-glutamate", "glutathione"],
    "key_proteins": ["glutamate-cysteine ligase"],
}


def _production_shaped_payload() -> Dict[str, Any]:
    """A final mapped payload as production actually presents one: no metadata.

    Stage 6 rebuilds rows from field whitelists that carry neither ``metadata``
    nor ``pathway_context``, so payload-only core discovery finds nothing here.
    That is not a contrived omission -- it is the shape the boundary receives.
    """

    payload = _base()
    payload.pop("metadata", None)
    payload.pop("pathway_context", None)
    return payload


def test_the_production_payload_really_carries_no_core_of_its_own() -> None:
    assert collect_requested_core_terms(_production_shaped_payload()) == []


def test_stage_zero_context_declares_the_core_when_the_payload_cannot() -> None:
    payload = _production_shaped_payload()

    result = quarantine_and_close(
        payload, strict_db=True, pathway_context=STAGE_ZERO_CONTEXT
    )

    assert result.coverage["requested_core_declared"] is True
    assert result.coverage["requested_core_source"] == "pathway_context"
    assert result.coverage["requested_core_terms"] == [
        "L-glutamate",
        "glutathione",
        "glutamate-cysteine ligase",
    ]
    # Preserved unchanged, as evidence for the derived terms above.
    assert result.coverage["requested_context"] == STAGE_ZERO_CONTEXT
    assert result.ok is True


def test_an_unrelated_survivor_cannot_produce_success_once_the_context_arrives() -> None:
    """The regression the explicit hand-off exists for.

    Without the context this payload passes -- ``requested_core_declared`` is
    False and only emptiness is refused, so one off-topic reaction is enough.
    With it, coverage is measured and the run is correctly refused. Both halves
    are asserted so the difference is attributable to the context alone.
    """

    payload = _production_shaped_payload()
    payload["entities"]["compounds"].extend([{"name": "citrate"}, {"name": "isocitrate"}])
    payload["element_locations"]["compound_locations"].extend(
        [
            {"compound": "citrate", "biological_state": "cytosol"},
            {"compound": "isocitrate", "biological_state": "cytosol"},
        ]
    )
    payload["processes"]["reactions"].append(
        {
            "name": "citrate isomerisation",
            "inputs": ["citrate"],
            "outputs": ["isocitrate"],
            "enzymes": [],
            "biological_state": "cytosol",
        }
    )
    for reaction in payload["processes"]["reactions"][:2]:
        reaction["inputs"] = ["never declared"]

    without_context = quarantine_and_close(deepcopy(payload), strict_db=True)
    assert without_context.coverage["requested_core_declared"] is False
    assert without_context.ok is True, "this is the hole the explicit hand-off closes"

    with_context = quarantine_and_close(
        deepcopy(payload), strict_db=True, pathway_context=STAGE_ZERO_CONTEXT
    )
    assert with_context.coverage["requested_core_declared"] is True
    # MOVED BY C-041a (D-002, LOCKED). Base SHA: ``ok is False``. What the
    # explicit hand-off closes is unchanged -- without the context the run is
    # judged in the undeclared regime and passes outright; with it the shortfall
    # is seen. What changed is what a seen shortfall does: it produces
    # review_required PWML, which is not success, instead of no PWML at all.
    assert with_context.ok is True
    assert with_context.quarantine_report["release"]["status"] == "review_required"
    assert with_context.quarantine_report["release"]["strict_acceptance_eligible"] is False
    assert with_context.coverage["core_accepted_processes"] == 0
    assert with_context.states()["citrate isomerisation"] == AUXILIARY_ACCEPTED
    assert any(
        reason.startswith("minimum_core:core_process_count_below_minimum")
        for reason in with_context.quarantine_report["review_reasons"]
    )


def test_the_requested_core_is_never_derived_from_what_survived() -> None:
    """Terms taken from the survivors would match the survivors. Not a test."""

    payload = _production_shaped_payload()
    result = quarantine_and_close(payload, strict_db=True)

    assert result.coverage["requested_core_terms"] == []
    assert result.coverage["requested_core_source"] == "none"


def test_an_explicit_requested_core_outranks_both_context_and_payload() -> None:
    result = quarantine_and_close(
        _base(),
        strict_db=True,
        requested_core=["urea cycle"],
        pathway_context=STAGE_ZERO_CONTEXT,
    )

    assert result.coverage["requested_core_terms"] == ["urea cycle"]
    assert result.coverage["requested_core_source"] == "explicit_argument"
    # MOVED BY C-041a (D-002, LOCKED). Base SHA: ``ok is False``. Which input
    # wins is what this test is about and is unchanged.
    assert result.ok is True
    assert result.quarantine_report["release"]["status"] == "review_required"
    assert result.quarantine_report["release"]["strict_acceptance_eligible"] is False


def test_the_payload_is_used_only_when_no_context_was_handed_over() -> None:
    result = quarantine_and_close(_base(), strict_db=True)

    assert result.coverage["requested_core_source"] == "payload"
    assert result.coverage["requested_core_terms"] == ["L-glutamate", "glutathione"]


# ── Refusal reporting ───────────────────────────────────────────────────────


def test_each_refusal_class_gets_its_own_reason() -> None:
    """A reviewer must be able to tell these four apart from the report alone."""

    empty = _base()
    for reaction in empty["processes"]["reactions"]:
        reaction["scope_membership"] = "out_of_scope"
    assert any(
        reason.startswith("minimum_core:")
        for reason in quarantine_and_close(empty, strict_db=True).refusal_reasons
    )

    overlap = _base()
    overlap["entities"]["compounds"].append({"name": "GPX1"})
    overlap["entities"]["proteins"].append(
        {"name": "GPX1", "species": "Homo sapiens", "mapped_ids": {"uniprot": "P07203"}}
    )
    overlap["entities"]["protein_complexes"][0]["components"].append(
        {"name": "GPX1", "stoichiometry": 1}
    )
    overlap_result = quarantine_and_close(overlap, strict_db=True)
    assert any(r.startswith("entity_type_overlap:") for r in overlap_result.refusal_reasons)

    locks = _with_locks()
    locks["locked_reaction_filter_report"]["locked_reactions_found"] = 9
    assert any(
        r.startswith("unaccounted_locked_reactions:")
        for r in quarantine_and_close(locks, strict_db=True).refusal_reasons
    )

    stalled = quarantine_and_close(_base(), strict_db=True, max_iterations=0)
    assert any(r.startswith("closure_not_converged:") for r in stalled.refusal_reasons)


def test_strict_invariant_failures_are_visible_as_refusals_not_only_as_state() -> None:
    """``ok`` and ``refusal_reasons`` must agree; a silent invariant is no gate."""

    payload = _base()
    payload["locked_reaction_filter_report"] = {
        "locked_reactions_found": 4,
        "exported_locked_reactions": 0,
        "quarantined_locked_reactions": 0,
        "unaccounted_locked_reactions": 4,
    }

    result = quarantine_and_close(payload, strict_db=True)

    assert result.ok is False
    assert result.refusal_reasons
    assert result.quarantine_report["strict_invariants"]["ok"] is False
    # Every refusal reason names a class, never a bare count.
    assert all(":" in reason for reason in result.refusal_reasons)


def test_a_clean_payload_refuses_nothing() -> None:
    result = quarantine_and_close(_base(), strict_db=True)

    assert result.ok is True
    assert result.refusal_reasons == []
    assert result.quarantine_report["strict_invariants"] == {
        "entity_type_overlaps": [],
        "degree_zero_exports": [],
        "unexportable_entities": [],
        "unaccounted_locked_reactions": 0,
        "closure_converged": True,
        "ok": True,
    }


def test_a_clean_pathway_of_only_interactions_is_not_refused() -> None:
    """A false negative found by replaying real legs, not by imagining shapes.

    ``runs/2026-07-28_2122`` PMC12624714/strict is four interactions and no
    reactions. It failed no gate and nothing was quarantined, and the coverage
    check refused it anyway: with no declared core every accepted process falls
    to AUXILIARY unless it is a reaction, so ``core_accepted`` was zero. Undeclared
    means relevance is unjudgeable, and the only rule left is "not empty".
    """

    payload = _base()
    payload.pop("metadata", None)
    payload["processes"]["reactions"] = []
    payload["processes"]["interactions"] = [
        {
            "name": "product inhibition",
            "left": "glutathione",
            "right": "glutathione synthetase",
            "relationship": "inhibits",
        }
    ]

    result = quarantine_and_close(payload, strict_db=True)

    assert result.coverage["requested_core_declared"] is False
    assert result.coverage["core_accepted_processes"] == 0
    assert result.coverage["auxiliary_accepted_processes"] == 1
    assert result.ok is True
    assert result.refusal_reasons == []


def test_the_undeclared_regime_still_refuses_an_empty_graph() -> None:
    """The relaxation above must not turn "not empty" off as well."""

    payload = _base()
    payload.pop("metadata", None)
    for reaction in payload["processes"]["reactions"]:
        reaction["scope_membership"] = "out_of_scope"

    result = quarantine_and_close(payload, strict_db=True)

    assert result.ok is False
    assert "minimum_core:no_surviving_process" in result.refusal_reasons


def test_a_declared_core_still_counts_only_core_accepted_processes() -> None:
    """The relaxation is scoped to the undeclared regime and nowhere else."""

    payload = _base()
    payload.pop("metadata", None)
    payload["processes"]["reactions"] = []
    payload["processes"]["interactions"] = [
        {
            "name": "unrelated interaction",
            "left": "glutathione",
            "right": "glutathione synthetase",
            "relationship": "inhibits",
        }
    ]

    result = quarantine_and_close(
        payload, strict_db=True, pathway_context={"key_compounds": ["ornithine"]}
    )

    assert result.coverage["requested_core_declared"] is True
    # MOVED BY C-041a (D-002, LOCKED). Base SHA: ``ok is False``. The relaxation
    # being scoped to the undeclared regime is what this test asserts and is
    # unchanged: the declared regime still counts only core_accepted, still sees
    # the shortfall, and still refuses to call it success.
    assert result.ok is True
    assert result.quarantine_report["release"]["strict_acceptance_eligible"] is False
    assert any(
        reason.startswith("minimum_core:core_process_count_below_minimum")
        for reason in result.quarantine_report["review_reasons"]
    )

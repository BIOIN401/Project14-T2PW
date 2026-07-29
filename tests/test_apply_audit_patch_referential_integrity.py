"""Referential integrity of /entities removals in apply_audit_patch.

These tests pin the guard added after the 2026-07-28 batch review. Two runs on
disk motivated it:

  * runs/2026-07-27_1623/.../PMC13231680__mechanistic-insights-into-
    phthalylsulfacetamide/strict/final_stage3_gate_report.json failed with
    "/processes/reactions/2/inputs/0 unknown entity: phthalylsulfacetamide (PSA)"
    while merged_payload.json for that leg still listed five near-duplicate
    compound rows ("phthalylsulfacetamide" twice, "phthalylsulfacetamide (PSA)",
    "PSA", "sulfacetamide"). A curation step deleted the redundant parenthesised
    row and left the reaction input naming it.
  * runs/2026-07-28_0919 PMC12444477/research burned 2778s and then failed with
    25 Stage 3 errors, 24 of them unknown_protein_modifier_reference pointing at
    reaction enzymes/modifiers.

The two halves of the guard are equally important and are both asserted here:
a removal that strands a reference must be refused, and a removal that strands
nothing -- the duplicate cleanup audit_json_llm.py's prompt explicitly solicits
at confidence >= 0.95 -- must still go through.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation.apply_audit_patch import (  # noqa: E402
    REFERENTIAL_INTEGRITY_REASON_PREFIX,
    apply_patch_with_policy,
)
from t2pw.pipeline.process_normalizer import validate_registry_references  # noqa: E402


def _psa_payload() -> dict:
    """A trimmed stand-in for the PMC13231680 merged payload.

    Keeps what mattered: duplicate compound rows for the same drug under three
    spellings, and a reaction input that uses the parenthesised spelling.
    """
    return {
        "entities": {
            "compounds": [
                {"name": "phthalylsulfacetamide"},
                {"name": "phthalylsulfacetamide (PSA)"},
                {"name": "meropenem"},
                {"name": "hydrolyzed meropenem"},
            ],
            "proteins": [{"name": "NDM-1"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "Meropenem hydrolysis",
                    "inputs": ["meropenem"],
                    "outputs": ["hydrolyzed meropenem"],
                    "enzymes": [{"entity": "NDM-1"}],
                },
                {
                    "name": "PSA inhibition of NDM-1",
                    "inputs": ["phthalylsulfacetamide (PSA)"],
                    "outputs": ["hydrolyzed meropenem"],
                    "modifiers": [{"entity": "NDM-1"}],
                },
            ]
        },
    }


def _remove(path: str, confidence: float = 0.99) -> dict:
    return {
        "op": "remove",
        "path": path,
        "confidence": confidence,
        "evidence": "Redundant duplicate row.",
    }


def test_refuses_removing_compound_still_named_by_a_reaction_input() -> None:
    """The exact PMC13231680 shape: delete the row reactions[1].inputs[0] names."""
    payload = _psa_payload()

    patched, report = apply_patch_with_policy(payload, [_remove("/entities/compounds/1")])

    assert patched == payload
    assert report["summary"] == {"accepted_count": 0, "rejected_count": 1, "total": 1}
    reason = report["rejected"][0]["reason"]
    assert reason.startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)
    assert "phthalylsulfacetamide (PSA)" in reason
    assert "/processes/reactions/1/inputs/0" in reason
    # The payload the guard protected must still satisfy the gate it agrees with.
    validate_registry_references(patched)


def test_refuses_removing_protein_still_named_by_an_enzyme_row() -> None:
    """The PMC12444477/research shape: 24 of its 25 gate errors were enzyme and
    modifier references, not inputs, so the guard has to walk actor rows too."""
    payload = _psa_payload()

    patched, report = apply_patch_with_policy(payload, [_remove("/entities/proteins/0")])

    assert patched == payload
    assert report["rejected"][0]["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)
    assert "NDM-1" in report["rejected"][0]["reason"]


def test_accepts_removing_a_genuinely_unreferenced_compound() -> None:
    """Do not over-block: an orphan row nothing points at is still deletable."""
    payload = _psa_payload()
    payload["entities"]["compounds"].append({"name": "sulfacetamide"})

    patched, report = apply_patch_with_policy(payload, [_remove("/entities/compounds/4")])

    assert report["summary"] == {"accepted_count": 1, "rejected_count": 0, "total": 1}
    assert [row["name"] for row in patched["entities"]["compounds"]] == [
        "phthalylsulfacetamide",
        "phthalylsulfacetamide (PSA)",
        "meropenem",
        "hydrolyzed meropenem",
    ]


def test_accepts_true_duplicate_cleanup_because_a_surviving_row_keeps_coverage() -> None:
    """Duplicate cleanup is a real, wanted behaviour and must keep working.

    Two rows spell the drug identically; deleting one leaves the reference
    resolvable through the other, so no cascade and no refusal is needed.
    """
    payload = _psa_payload()
    payload["entities"]["compounds"].insert(1, {"name": "phthalylsulfacetamide"})
    payload["processes"]["reactions"][1]["inputs"] = ["phthalylsulfacetamide"]

    patched, report = apply_patch_with_policy(payload, [_remove("/entities/compounds/1")])

    assert report["summary"] == {"accepted_count": 1, "rejected_count": 0, "total": 1}
    assert [row["name"] for row in patched["entities"]["compounds"]].count("phthalylsulfacetamide") == 1
    validate_registry_references(patched)


def test_accepts_removal_when_a_declared_synonym_on_another_row_keeps_coverage() -> None:
    """Identity must be the gate's, not a home-grown one.

    process_normalizer.validate_registry_references resolves a reference against
    names AND declared synonyms, so a row whose name is duplicated as another
    row's synonym is removable without stranding anything. A guard that only
    compared `name` fields would refuse this and block a legitimate merge.
    """
    payload = _psa_payload()
    payload["entities"]["compounds"][0] = {
        "name": "phthalylsulfacetamide",
        "synonyms": ["phthalylsulfacetamide (PSA)"],
    }

    patched, report = apply_patch_with_policy(payload, [_remove("/entities/compounds/1")])

    assert report["summary"] == {"accepted_count": 1, "rejected_count": 0, "total": 1}
    validate_registry_references(patched)


def test_refuses_removing_the_synonym_that_was_the_only_coverage() -> None:
    """The mirror of the previous case: the reference resolves only through a
    synonym, so deleting that synonym array is as destructive as deleting a row.
    Checking the coverage diff rather than the pointer shape catches both."""
    payload = _psa_payload()
    payload["entities"]["compounds"] = [
        {"name": "phthalylsulfacetamide", "synonyms": ["phthalylsulfacetamide (PSA)"]},
        {"name": "meropenem"},
        {"name": "hydrolyzed meropenem"},
    ]

    patched, report = apply_patch_with_policy(payload, [_remove("/entities/compounds/0/synonyms")])

    assert patched == payload
    assert report["rejected"][0]["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)


def test_refuses_removing_the_whole_compounds_bucket() -> None:
    payload = _psa_payload()

    patched, report = apply_patch_with_policy(payload, [_remove("/entities/compounds")])

    assert patched == payload
    assert report["rejected"][0]["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)


def test_transport_cargo_and_transporter_references_are_protected() -> None:
    """validate_registry_references also checks transports; so must the guard."""
    payload = {
        "entities": {
            "compounds": [{"name": "colistin"}],
            "proteins": [{"name": "McrB"}],
        },
        "processes": {
            "transports": [
                {
                    "cargo": "colistin",
                    "from_biological_state": "extracellular",
                    "to_biological_state": "cytoplasm",
                    "transporters": [{"entity": "McrB"}],
                }
            ]
        },
    }

    _, cargo_report = apply_patch_with_policy(payload, [_remove("/entities/compounds/0")])
    _, transporter_report = apply_patch_with_policy(payload, [_remove("/entities/proteins/0")])

    assert cargo_report["rejected"][0]["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)
    assert transporter_report["rejected"][0]["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)


def test_cofactor_relocation_from_proteins_to_compounds_is_not_blocked() -> None:
    """audit_json_llm.py's prompt asks for exactly this two-op repair:
    "A small-molecule cofactor ... listed under entities.proteins[] is a type
    error. Propose removing it from proteins[] and adding it to
    entities.compounds[]". Judged in isolation the remove strands the modifier
    reference, so the guard has to see the compensating add later in the batch.
    """
    payload = {
        "entities": {
            "compounds": [{"name": "L-serine"}],
            "proteins": [{"name": "PLP"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "Serine turnover",
                    "inputs": ["L-serine"],
                    "outputs": ["L-serine"],
                    "modifiers": [{"entity": "PLP"}],
                }
            ]
        },
    }
    patch = [
        _remove("/entities/proteins/0"),
        {
            "op": "add",
            "path": "/entities/compounds/-",
            "value": {"name": "PLP", "class": "cofactor"},
            "confidence": 0.95,
            "evidence": "PLP is a small-molecule cofactor, not a protein.",
        },
    ]

    patched, report = apply_patch_with_policy(payload, patch)

    assert report["summary"] == {"accepted_count": 2, "rejected_count": 0, "total": 2}
    assert patched["entities"]["proteins"] == []
    assert patched["entities"]["compounds"][-1]["name"] == "PLP"
    validate_registry_references(patched)


def test_compensating_add_below_its_own_threshold_does_not_license_the_removal() -> None:
    """A look-ahead that counted ops _should_accept will reject would just move
    the dangling reference one step later, so only ops that clear their own
    confidence bar count as compensation. An add sits at 0.75 (_threshold_for_op)."""
    payload = {
        "entities": {"compounds": [{"name": "L-serine"}], "proteins": [{"name": "PLP"}]},
        "processes": {
            "reactions": [
                {
                    "name": "Serine turnover",
                    "inputs": ["L-serine"],
                    "outputs": ["L-serine"],
                    "modifiers": [{"entity": "PLP"}],
                }
            ]
        },
    }
    patch = [
        _remove("/entities/proteins/0"),
        {
            "op": "add",
            "path": "/entities/compounds/-",
            "value": {"name": "PLP"},
            "confidence": 0.40,
            "evidence": "Unsure.",
        },
    ]

    patched, report = apply_patch_with_policy(payload, patch)

    assert report["summary"]["accepted_count"] == 0
    assert report["rejected"][0]["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)
    assert patched["entities"]["proteins"] == [{"name": "PLP"}]


def test_index_drift_across_two_removals_is_judged_against_the_live_payload() -> None:
    """Why the guard runs post-application rather than inside _should_accept:
    that function is handed the pre-batch payload, so after the first removal
    every later index addresses a different row than it did originally. Here the
    second op says /entities/compounds/1, which is the unreferenced "spectator"
    row before the first removal and the referenced "meropenem" row after it.
    """
    payload = {
        "entities": {
            "compounds": [
                {"name": "orphan-a"},
                {"name": "spectator"},
                {"name": "meropenem"},
                {"name": "hydrolyzed meropenem"},
            ]
        },
        "processes": {
            "reactions": [
                {
                    "name": "Meropenem hydrolysis",
                    "inputs": ["meropenem"],
                    "outputs": ["hydrolyzed meropenem"],
                }
            ]
        },
    }
    patch = [_remove("/entities/compounds/0"), _remove("/entities/compounds/1")]

    patched, report = apply_patch_with_policy(payload, patch)

    assert report["summary"] == {"accepted_count": 1, "rejected_count": 1, "total": 2}
    assert report["rejected"][0]["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)
    assert [row["name"] for row in patched["entities"]["compounds"]] == [
        "spectator",
        "meropenem",
        "hydrolyzed meropenem",
    ]


def test_non_entity_and_non_shrinking_entity_ops_are_untouched() -> None:
    """The guard is scoped to /entities and to coverage-shrinking ops only.

    A process-side rewrite that introduces an undeclared reference is the Stage 3
    gate's business, not this guard's -- widening the scope here would change
    long-standing accepted behaviour (see
    test_without_locked_manifest_preserves_existing_reactions_array_replace_behavior).
    """
    payload = _psa_payload()
    patch = [
        {
            "op": "add",
            "path": "/entities/compounds/0/mapped_ids",
            "value": {"hmdb": "HMDB0000001"},
            "confidence": 0.90,
            "evidence": "Mapped.",
        },
        {
            "op": "replace",
            "path": "/processes/reactions",
            "value": [{"name": "Replacement", "inputs": ["A"], "outputs": ["B"]}],
            "confidence": 0.99,
            "evidence": "High-confidence rewrite.",
        },
    ]

    patched, report = apply_patch_with_policy(payload, patch)

    assert report["summary"] == {"accepted_count": 2, "rejected_count": 0, "total": 2}
    assert patched["entities"]["compounds"][0]["mapped_ids"] == {"hmdb": "HMDB0000001"}


def test_rejection_reason_is_recorded_in_the_rejected_patch_log() -> None:
    """The refusal has to be legible in rejected_patch_log.json, since that is
    where the next audit round is supposed to read it and propose a synonym add
    instead of a second deletion."""
    payload = _psa_payload()
    manifest = [
        {
            "locked_reaction_id": "rxn_lock_001",
            "source_reaction_id": "r1",
            "name": "Meropenem hydrolysis",
            "inputs": ["meropenem"],
            "outputs": ["hydrolyzed meropenem"],
            "modifiers_or_enzymes": ["NDM-1"],
            "locked": True,
        }
    ]

    _, report = apply_patch_with_policy(
        payload,
        [_remove("/entities/compounds/1")],
        locked_manifest=manifest,
        stage="audit_round_2",
    )

    assert len(report["rejected_patch_log"]) == 1
    entry = report["rejected_patch_log"][0]
    assert entry["stage"] == "audit_round_2"
    assert entry["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)
    assert report["applied_patch_log"] == []


def _gate_errors(payload: dict) -> set:
    """The Stage 3 registry errors a payload currently has, as a comparable set."""
    try:
        validate_registry_references(apply_patch_with_policy(payload, [])[0])
    except ValueError as exc:
        return set(str(exc).splitlines()[1:])
    return set()


def test_guard_agrees_with_the_stage3_gate_on_every_single_row_removal() -> None:
    """The stated risk of this fix was inventing a second notion of entity
    identity that disagrees with the gate it is supposed to protect. This asserts
    they cannot disagree: for every compound and protein row, the guard's verdict
    must equal "does deleting this row introduce a Stage 3 registry error that was
    not already there".

    The oracle is deliberately phrased as *newly introduced* errors, not "does the
    gate pass". Both reference payloads on disk arrive already failing for an
    unrelated reason -- PMC13231680/strict on
    "/processes/interactions/2/entity_2 unknown entity: MEM" and
    PMC12444477/research on
    "/processes/interactions/4/entity_2 unknown entity: LapB-FtsH proteolysis" --
    and a guard that refused every removal on an already-broken payload would be
    useless exactly when the audit loop is most needed. Replaying all 89 entity
    rows of the PMC12444477/research merged_payload.json through this oracle gives
    47 accepted and 42 refused with zero disagreements.
    """
    payload = _psa_payload()
    payload["entities"]["compounds"].append({"name": "sulfacetamide"})
    payload["entities"]["compounds"][0]["synonyms"] = ["PSA"]
    payload["processes"]["reactions"][1]["modifiers"].append({"entity": "PSA"})
    baseline = _gate_errors(payload)

    for bucket in ("compounds", "proteins"):
        for index in range(len(payload["entities"][bucket])):
            _, report = apply_patch_with_policy(
                payload, [_remove(f"/entities/{bucket}/{index}")]
            )
            accepted = report["summary"]["accepted_count"] == 1

            probe = apply_patch_with_policy(payload, [])[0]
            del probe["entities"][bucket][index]
            introduces_new_error = bool(_gate_errors(probe) - baseline)

            assert accepted == (not introduces_new_error), (
                f"guard and Stage 3 gate disagree on removing /entities/{bucket}/{index}"
            )


def test_pre_existing_dangling_reference_is_not_blamed_on_an_unrelated_removal() -> None:
    """Both reference payloads arrive already failing the registry gate on an
    interaction participant. If the guard reported total gate health instead of
    the damage the op itself caused, every entity removal on those payloads would
    be refused and the audit loop could never make progress on a broken pathway.
    """
    payload = _psa_payload()
    payload["entities"]["compounds"].append({"name": "sulfacetamide"})
    payload["processes"]["interactions"] = [
        {"entity_1": "meropenem", "entity_2": "MEM", "relationship": "SAME_AS"}
    ]
    assert _gate_errors(payload), "fixture must start out already gate-failing"

    _, report = apply_patch_with_policy(payload, [_remove("/entities/compounds/4")])

    assert report["summary"] == {"accepted_count": 1, "rejected_count": 0, "total": 1}

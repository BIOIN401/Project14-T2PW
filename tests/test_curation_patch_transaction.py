"""Curation patches commit as a set, or not at all.

``tests/test_apply_audit_patch_referential_integrity.py`` pins the per-op guard.
These tests pin the two holes that guard cannot see, both of which end the same
way -- a dangling name in the payload and a Stage 3 abort at export time:

* **A bucket the guard did not count.** ``_REGISTRY_ENTITY_BUCKETS`` was three
  buckets while ``validate_registry_references`` had grown to four, so coverage
  never registered a nucleic-acid loss and every such removal was waved through.
* **A promise the batch never kept.** The guard clears a removal on the strength
  of a *later* add. Nothing used to check that the add actually landed, so a
  malformed one authorised a removal it could not compensate for.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation.apply_audit_patch import (  # noqa: E402
    REFERENTIAL_INTEGRITY_REASON_PREFIX,
    ROLLED_BACK_REASON_PREFIX,
    apply_patch_with_policy,
    committed_change_count,
)
from t2pw.pipeline.process_normalizer import validate_registry_references  # noqa: E402


def _operon_payload() -> dict:
    """A reaction consuming a nucleic acid, alongside ordinary compounds.

    'pmrHFIJKLM operon' is a reaction input of PMC13278307 in run 2026-07-28_0919
    and lives in entities.nucleic_acids because enforce_entity_type_consistency
    relocated it out of compounds.
    """

    return {
        "entities": {
            "compounds": [
                {"name": "UDP-4-amino-4-deoxy-L-arabinose"},
                {"name": "lipid A"},
                {"name": "modified lipid A"},
                {"name": "unused cofactor"},
            ],
            "proteins": [{"name": "ArnT", "mapped_ids": {"uniprot": "P76473"}}],
            "protein_complexes": [],
            "nucleic_acids": [
                {"name": "pmrHFIJKLM operon"},
                {"name": "unused transcript"},
            ],
        },
        "processes": {
            "reactions": [
                {
                    "name": "lipid A modification",
                    "inputs": ["lipid A", "UDP-4-amino-4-deoxy-L-arabinose"],
                    "outputs": ["modified lipid A"],
                    "enzymes": [{"entity": "ArnT"}],
                },
                {
                    "name": "operon transcription",
                    "inputs": ["pmrHFIJKLM operon"],
                    "outputs": ["modified lipid A"],
                },
            ]
        },
    }


def _remove(path: str, confidence: float = 0.99) -> dict:
    return {
        "op": "remove",
        "path": path,
        "confidence": confidence,
        "evidence": "curation cleanup",
    }


def _add(path: str, value, confidence: float = 0.99) -> dict:
    return {
        "op": "add",
        "path": path,
        "value": value,
        "confidence": confidence,
        "evidence": "curation cleanup",
    }


# ── The missing bucket ──────────────────────────────────────────────────────


def test_removing_a_referenced_nucleic_acid_is_rejected() -> None:
    payload = _operon_payload()

    result, report = apply_patch_with_policy(payload, [_remove("/entities/nucleic_acids/0")])

    assert report["summary"]["accepted_count"] == 0
    assert report["rejected"][0]["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)
    assert "pmrHFIJKLM operon" in report["rejected"][0]["reason"]
    assert [row["name"] for row in result["entities"]["nucleic_acids"]] == [
        "pmrHFIJKLM operon",
        "unused transcript",
    ]
    validate_registry_references(result)


def test_removing_an_unreferenced_nucleic_acid_still_goes_through() -> None:
    """The guard must not turn into a blanket ban on the bucket."""

    payload = _operon_payload()

    result, report = apply_patch_with_policy(payload, [_remove("/entities/nucleic_acids/1")])

    assert report["summary"]["accepted_count"] == 1
    assert report["transaction"]["committed"] is True
    assert [row["name"] for row in result["entities"]["nucleic_acids"]] == ["pmrHFIJKLM operon"]
    validate_registry_references(result)


def test_the_guard_agrees_with_the_stage3_gate_on_every_nucleic_acid_removal() -> None:
    payload = _operon_payload()

    for index in range(len(payload["entities"]["nucleic_acids"])):
        _result, report = apply_patch_with_policy(
            deepcopy(payload), [_remove(f"/entities/nucleic_acids/{index}")]
        )
        accepted = report["summary"]["accepted_count"] == 1

        # What the gate would say if the removal went through unguarded. Asking
        # it about the *guarded* result would be circular: a refused removal
        # leaves the payload untouched, so the gate always passes.
        forced = deepcopy(payload)
        del forced["entities"]["nucleic_acids"][index]
        gate_would_pass = True
        try:
            validate_registry_references(forced)
        except ValueError:
            gate_would_pass = False

        # Accepting a removal the gate then rejects is the bug; refusing one the
        # gate would have allowed is over-blocking. Neither is permitted.
        assert accepted == gate_would_pass


# ── The promise the batch never kept ────────────────────────────────────────


def test_a_malformed_later_add_cannot_authorize_an_earlier_removal() -> None:
    """The look-ahead is a promise; this is the check that it was kept.

    The relocation repair audit_json_llm.py prompts for arrives as
    "remove from proteins" + "add to compounds". The remove is only safe because
    of the add, so the look-ahead credits any list of strings an /entities op
    would introduce -- without checking which leaf it lands on. Here it lands on
    ``aliases``, a field no registry reads, so the name is credited to the
    removal and then restores nothing: exactly the reference the Stage 3 gate
    aborts the export over.
    """

    payload = _operon_payload()
    ops = [
        _remove("/entities/compounds/1"),  # lipid A, named by a reaction input
        _add("/entities/compounds/0/aliases", ["lipid A"]),
    ]

    result, report = apply_patch_with_policy(payload, ops)

    assert report["transaction"]["committed"] is False
    assert report["transaction"]["rolled_back"] is True
    assert report["transaction"]["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)
    assert "lipid A" in report["transaction"]["reason"]
    assert result == payload
    validate_registry_references(result)


def test_a_later_add_that_fails_to_apply_cannot_authorize_an_earlier_removal() -> None:
    payload = _operon_payload()
    ops = [
        _remove("/entities/compounds/1"),
        # Index 40 does not exist; _apply_single_op raises and the op is rejected.
        _add("/entities/compounds/40", {"name": "lipid A"}),
    ]

    result, report = apply_patch_with_policy(payload, ops)

    assert report["transaction"]["committed"] is False
    assert result == payload


def test_a_well_formed_later_add_still_authorizes_the_removal_and_commits() -> None:
    """The relocation repair the pipeline actually wants must keep working."""

    payload = _operon_payload()
    ops = [
        _remove("/entities/compounds/1"),
        _add("/entities/compounds/-", {"name": "lipid A", "class": "compound"}),
    ]

    result, report = apply_patch_with_policy(payload, ops)

    assert report["summary"]["accepted_count"] == 2
    assert report["transaction"]["committed"] is True
    assert report["transaction"]["applied_count"] == 2
    assert "lipid A" in {row["name"] for row in result["entities"]["compounds"]}
    validate_registry_references(result)


def test_rollback_marks_every_accepted_op_and_records_the_orphans() -> None:
    payload = _operon_payload()
    ops = [
        _remove("/entities/compounds/1"),
        _add("/entities/compounds/0/aliases", ["lipid A"]),
    ]

    _result, report = apply_patch_with_policy(payload, ops)

    accepted = report["accepted"]
    assert accepted, "the removal was accepted per-op; that is what makes rollback necessary"
    assert all(record["rolled_back"] is True for record in accepted)
    assert report["transaction"]["applied_count"] == 0
    assert report["summary"]["accepted_count"] == len(accepted)
    assert any("lipid A" in entry for entry in report["transaction"]["orphaned_references"])


def test_a_pre_existing_dangling_reference_does_not_roll_back_an_unrelated_batch() -> None:
    """Rolling back over a defect the patch never touched would disable curation
    on exactly the payloads that need it most."""

    payload = _operon_payload()
    payload["processes"]["reactions"][0]["inputs"].append("never declared anywhere")

    result, report = apply_patch_with_policy(payload, [_remove("/entities/compounds/3")])

    assert report["transaction"]["committed"] is True
    assert report["summary"]["accepted_count"] == 1
    assert "unused cofactor" not in {row["name"] for row in result["entities"]["compounds"]}


def test_a_clean_batch_reports_a_committed_transaction() -> None:
    payload = _operon_payload()

    _result, report = apply_patch_with_policy(payload, [])

    assert report["transaction"] == {
        "committed": True,
        "rolled_back": False,
        "reason": "",
        "orphaned_references": [],
        "accepted_before_rollback": 0,
        "applied_count": 0,
    }


def test_rollback_is_recorded_in_the_rejected_patch_log(tmp_path: Path) -> None:
    payload = _operon_payload()
    rejected_log = tmp_path / "rejected_patch_log.json"
    ops = [
        _remove("/entities/compounds/1"),
        _add("/entities/compounds/0/aliases", ["lipid A"]),
    ]

    _result, report = apply_patch_with_policy(
        payload, ops, rejected_log_path=rejected_log, stage="audit"
    )

    assert rejected_log.exists()
    reasons = [record["reason"] for record in report["rejected_patch_log"]]
    assert any(reason.startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX) for reason in reasons)


@pytest.mark.parametrize("bucket", ["compounds", "proteins", "protein_complexes", "nucleic_acids"])
def test_every_registry_bucket_is_covered_by_the_guard(bucket: str) -> None:
    """All four buckets the Stage 3 registry unions, not three of them."""

    payload = {
        "entities": {
            "compounds": [{"name": "referenced compound"}],
            "proteins": [{"name": "referenced protein", "mapped_ids": {"uniprot": "P00001"}}],
            "protein_complexes": [
                {"name": "referenced complex", "components": [{"name": "referenced protein"}]}
            ],
            "nucleic_acids": [{"name": "referenced operon"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "everything at once",
                    "inputs": ["referenced compound", "referenced operon"],
                    "outputs": ["referenced compound"],
                    "enzymes": [{"entity": "referenced protein"}],
                    "modifiers": [{"entity": "referenced complex"}],
                }
            ]
        },
    }

    _result, report = apply_patch_with_policy(payload, [_remove(f"/entities/{bucket}/0")])

    assert report["summary"]["accepted_count"] == 0
    assert report["rejected"][0]["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)


# ── Truthful reporting of what actually committed ───────────────────────────


def _rolled_back_ops() -> list:
    """A remove authorised by a later add that lands on a field no registry reads."""

    return [
        _remove("/entities/compounds/1"),
        _add("/entities/compounds/0/aliases", ["lipid A"]),
    ]


def test_rollback_writes_no_accepted_entry_to_the_applied_patch_log(tmp_path: Path) -> None:
    """The applied log is the record of what CHANGED the payload.

    After a rollback nothing did. Leaving these entries in it -- each stamped
    "accepted" -- would have the run's own audit trail assert edits that were
    undone before anyone saw them, and every downstream reader of that log
    (batch reports, the completeness audit) would repeat the claim.
    """

    applied_log = tmp_path / "applied_patch_log.json"
    rejected_log = tmp_path / "rejected_patch_log.json"

    _result, report = apply_patch_with_policy(
        _operon_payload(),
        _rolled_back_ops(),
        applied_log_path=applied_log,
        rejected_log_path=rejected_log,
        stage="audit",
    )

    assert report["transaction"]["committed"] is False
    assert report["applied_patch_log"] == []
    assert json.loads(applied_log.read_text(encoding="utf-8")) == []

    reasons = [record["reason"] for record in report["rejected_patch_log"]]
    assert any(reason.startswith(ROLLED_BACK_REASON_PREFIX) for reason in reasons), reasons
    rolled = [record for record in report["rejected_patch_log"] if record.get("rolled_back")]
    assert rolled, "the undone op must be recorded, not merely deleted from the applied log"


def test_a_committed_batch_still_writes_its_applied_log(tmp_path: Path) -> None:
    """The rollback path must not become a blanket suppression."""

    applied_log = tmp_path / "applied_patch_log.json"

    _result, report = apply_patch_with_policy(
        _operon_payload(),
        [_remove("/entities/nucleic_acids/1")],
        applied_log_path=applied_log,
        stage="audit",
    )

    assert report["transaction"]["committed"] is True
    assert len(report["applied_patch_log"]) == 1
    assert report["applied_patch_log"][0]["reason"] == "accepted"
    assert len(json.loads(applied_log.read_text(encoding="utf-8"))) == 1


def test_committed_change_count_is_zero_on_rollback_and_accurate_otherwise() -> None:
    _result, rolled = apply_patch_with_policy(_operon_payload(), _rolled_back_ops())
    assert committed_change_count(rolled) == 0
    assert rolled["summary"]["accepted_count"] >= 1, "the per-op verdicts are unchanged"

    _result, committed = apply_patch_with_policy(
        _operon_payload(), [_remove("/entities/nucleic_acids/1")]
    )
    assert committed_change_count(committed) == 1


def test_committed_change_count_falls_back_for_reports_written_before_transactions() -> None:
    """An archived report has no ``transaction`` block and must still read sensibly."""

    legacy = {"summary": {"accepted_count": 4, "rejected_count": 1, "total": 5}}

    assert committed_change_count(legacy) == 4
    assert committed_change_count({}) == 0
    assert committed_change_count(None) == 0


def test_every_production_consumer_reports_zero_changes_after_a_rollback() -> None:
    """The five call sites that report "how many patches landed", together.

    Each one used ``summary["accepted_count"]``, which after a rollback is
    non-zero while the payload is byte-identical to the input. Asserting them as
    a group is deliberate: they are one claim made in five places, and a future
    sixth site should be added here rather than left to drift.
    """

    payload = _operon_payload()
    before = deepcopy(payload)
    result, report = apply_patch_with_policy(payload, _rolled_back_ops())

    assert result == before, "nothing changed, so every count below must be zero"

    # pathway_curator.report["summary"]["patches_accepted"]
    assert committed_change_count(report) == 0
    # gap_resolver.report["summary"]["enrichment_patches_accepted"]
    assert committed_change_count(report) == 0
    # focused_repair's returned applied-count
    assert committed_change_count(report) == 0
    # streamlit candidate scoring (_audit_objective_score accepted_count)
    assert committed_change_count(report) == 0
    # apply_audit_patch CLI "Patch applied:"
    assert committed_change_count(report) == 0


def test_focused_repair_returns_the_committed_count() -> None:
    from t2pw.pipeline.focused_repair import _apply_pass_patches

    payload = _operon_payload()
    updated, applied, rejected = _apply_pass_patches(
        payload, _rolled_back_ops(), [], stage="focused_repair"
    )

    assert applied == 0
    assert rejected == 0, "both ops passed their per-op checks; the SET is what failed"
    assert updated == _operon_payload()


def test_pathway_curator_reports_zero_patches_accepted_after_a_rollback() -> None:
    from t2pw.curation import pathway_curator

    report: dict = {"summary": {}}
    _patched, apply_report = apply_patch_with_policy(_operon_payload(), _rolled_back_ops())
    report["summary"]["patches_accepted"] = pathway_curator.committed_change_count(apply_report)

    assert report["summary"]["patches_accepted"] == 0

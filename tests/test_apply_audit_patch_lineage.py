"""Lineage emission at the audit-patch stage.

R-004 asked whether three reactions in a committed payload had been re-added by
the audit and could not tell from the artifacts on disk. Every test here is
black-box, through the module's public entry points only: the point is what a
reader of a committed payload can conclude, not how the stage computes it.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation.apply_audit_patch import (  # noqa: E402
    REFERENTIAL_INTEGRITY_REASON_PREFIX,
    apply_patch_with_policy,
    run_apply,
)
from t2pw.pipeline.lineage import LINEAGE_KEY  # noqa: E402


# A record written by an earlier stage, in the exact serialized form lineage.record
# re-emits, so an equality check proves it survived rather than merely resembling.
_PAPER_ENTRY = {
    "stage": "paper_extraction",
    "origin": "paper_stated",
    "support": "direct",
    "paper_explicit": "explicit",
    "reason": "the paper names this compound",
    "review_required": False,
    "uncertainty": "",
    "sources": [{"source_id": "PMC12421875", "source_type": "", "uri": "", "locator": ""}],
}


def _payload() -> dict:
    return {
        "entities": {
            "compounds": [{"name": "DHNA"}, {"name": "DMK"}],
            "proteins": [{"name": "MenA"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "DHNA prenylation",
                    "inputs": ["DHNA"],
                    "outputs": ["DMK"],
                    "enzymes": [{"entity": "MenA"}],
                }
            ]
        },
    }


def _op(action: str, path: str, value=None, *, confidence: float = 0.99,
        evidence: str = "the paper states it") -> dict:
    op = {"op": action, "path": path, "confidence": confidence, "evidence": evidence}
    if value is not None:
        op["value"] = value
    return op


def _entries(row: dict) -> list:
    return row.get(LINEAGE_KEY, [])


def _audit_entries(row: dict) -> list:
    return [entry for entry in _entries(row) if entry["stage"] == "audit_repair"]


def _strip_lineage(value):
    if isinstance(value, dict):
        return {k: _strip_lineage(v) for k, v in value.items() if k != LINEAGE_KEY}
    if isinstance(value, list):
        return [_strip_lineage(item) for item in value]
    return value


def _locked() -> tuple:
    payload = _payload()
    payload["processes"]["reactions"][0]["locked_reaction_id"] = "rxn_lock_001"
    manifest = [
        {
            "locked_reaction_id": "rxn_lock_001",
            "name": "DHNA prenylation",
            "inputs": ["DHNA"],
            "outputs": ["DMK"],
            "locked": True,
        }
    ]
    return payload, manifest


# ---------------------------------------------------------------------------
# NEW ACCEPTANCE -- genuinely new emission. These fail on base b5bbf08 because
# no record is written there, which is the capability being added, not a
# pre-existing behaviour being corrected.
# ---------------------------------------------------------------------------


def test_a_row_the_audit_added_says_the_audit_wrote_it() -> None:
    patched, report = apply_patch_with_policy(
        _payload(), [_op("add", "/entities/compounds/-", {"name": "MK-4"})]
    )

    assert report["transaction"]["applied_count"] == 1
    added = patched["entities"]["compounds"][-1]
    assert added["name"] == "MK-4"
    assert len(_audit_entries(added)) == 1
    assert _audit_entries(added)[0]["origin"] == "audit_modified"


def test_a_reaction_the_audit_added_is_distinguishable_from_one_it_carried_through() -> None:
    """R-004's question, asked of the payload itself.

    Both reactions are present when the stage finishes and the patch log records
    only an op. The records are what separate "the audit put this here" from
    "the audit found this here".
    """
    patched, _report = apply_patch_with_policy(
        _payload(),
        [
            _op(
                "add",
                "/processes/reactions/-",
                {"name": "DMK methylation", "inputs": ["DMK"], "outputs": ["MK-4"]},
            )
        ],
    )

    carried, added = patched["processes"]["reactions"]
    assert carried["name"] == "DHNA prenylation"
    assert added["name"] == "DMK methylation"
    assert not _audit_entries(carried), "a row the patch never touched must claim nothing"
    assert len(_audit_entries(added)) == 1


def test_an_edit_inside_a_row_records_on_the_row_that_owns_it() -> None:
    """Attribution is positional: LineageEntry carries no reaction id, so a
    record is true of the row it sits on and of nothing else."""
    patched, _report = apply_patch_with_policy(
        _payload(), [_op("add", "/processes/reactions/0/inputs/-", "prenyl diphosphate")]
    )

    reaction = patched["processes"]["reactions"][0]
    assert reaction["inputs"] == ["DHNA", "prenyl diphosphate"]
    assert len(_audit_entries(reaction)) == 1
    assert all(not _entries(row) for row in patched["entities"]["compounds"]), (
        "the entity rows the edit merely names must not carry the reaction's record"
    )


def test_the_record_claims_no_source_and_no_paper_support() -> None:
    """An op's ``evidence`` is free text from the auditing model. Accepting it as
    a source would be accepting an identifier for its shape, so direct/indirect
    are unavailable and the content is provisional."""
    patched, _report = apply_patch_with_policy(
        _payload(), [_op("add", "/entities/compounds/-", {"name": "MK-4"})]
    )

    entry = _audit_entries(patched["entities"]["compounds"][-1])[0]
    assert entry["support"] == "unsupported"
    assert entry["sources"] == []
    assert entry["paper_explicit"] == "not_evaluated", (
        "this stage never evaluates whether the paper stated it, and "
        "not_evaluated is never not_explicit"
    )
    assert entry["review_required"] is False
    assert entry["uncertainty"]


def test_an_earlier_stages_record_is_kept_and_the_audit_appends_to_it() -> None:
    payload = _payload()
    payload["entities"]["compounds"][0][LINEAGE_KEY] = [dict(_PAPER_ENTRY)]

    patched, _report = apply_patch_with_policy(
        payload,
        [_op("add", "/entities/compounds/0/synonyms", ["1,4-dihydroxy-2-naphthoate"])],
    )

    row = patched["entities"]["compounds"][0]
    assert _PAPER_ENTRY in _entries(row), (
        "a row this stage did not originate keeps the origin it already had"
    )
    assert len(_audit_entries(row)) == 1


def test_the_records_reach_the_audited_json_on_disk(tmp_path: Path) -> None:
    """R-004 could not answer its question from committed artifacts. This is the
    assertion that the answer is now inside one."""
    input_path = tmp_path / "final.json"
    patch_path = tmp_path / "audit_patch.json"
    output_path = tmp_path / "final.audited.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    patch_path.write_text(
        json.dumps(
            [
                _op(
                    "add",
                    "/processes/reactions/-",
                    {"name": "DMK methylation", "inputs": ["DMK"], "outputs": ["MK-4"]},
                )
            ]
        ),
        encoding="utf-8",
    )

    run_apply(input_path, patch_path, output_path)

    audited = json.loads(output_path.read_text(encoding="utf-8"))
    carried, added = audited["processes"]["reactions"]
    assert not _audit_entries(carried)
    assert len(_audit_entries(added)) == 1
    assert _audit_entries(added)[0]["stage"] == "audit_repair"


# ---------------------------------------------------------------------------
# PRESERVATION -- the instrumented stage's decisions and outputs are unchanged.
# Each of these passes on base b5bbf08 AND at the tip.
# ---------------------------------------------------------------------------


def test_the_audited_payload_is_unchanged_once_the_records_are_stripped() -> None:
    ops = [
        _op("add", "/entities/compounds/-", {"name": "MK-4"}),
        _op("add", "/processes/reactions/0/inputs/-", "prenyl diphosphate"),
    ]

    patched, report = apply_patch_with_policy(_payload(), ops)

    expected = _payload()
    expected["entities"]["compounds"].append({"name": "MK-4"})
    expected["processes"]["reactions"][0]["inputs"].append("prenyl diphosphate")
    assert _strip_lineage(patched) == expected
    assert report["summary"] == {"accepted_count": 2, "rejected_count": 0, "total": 2}
    assert report["transaction"]["applied_count"] == 2


def test_a_rejected_op_changes_nothing_and_records_nothing() -> None:
    payload = _payload()

    patched, report = apply_patch_with_policy(
        payload, [_op("remove", "/processes/reactions/0", evidence="unsupported duplicate")]
    )

    assert report["summary"] == {"accepted_count": 0, "rejected_count": 1, "total": 1}
    assert patched == payload


def test_a_rolled_back_batch_changes_nothing_and_records_nothing() -> None:
    """A batch that changed nothing must claim nothing: on rollback the payload
    handed back is a fresh copy of the input, records included."""
    payload = _payload()
    before = deepcopy(payload)

    patched, report = apply_patch_with_policy(
        payload,
        [
            _op("remove", "/entities/compounds/1", evidence="duplicate of DMK"),
            _op("add", "/entities/compounds/0/aliases", ["DMK"]),
        ],
    )

    assert report["transaction"]["rolled_back"] is True
    assert report["transaction"]["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)
    assert patched == before


def test_a_record_on_a_locked_reaction_does_not_read_as_a_second_lock() -> None:
    """The post-op lock validator walks every value in a reaction looking for
    locked ids, so it sees the record the previous op left on that row. Two ops
    on one locked reaction is what puts a record in front of that walk."""
    payload, manifest = _locked()

    patched, report = apply_patch_with_policy(
        payload,
        [
            _op("add", "/processes/reactions/0/inputs/-", "prenyl diphosphate"),
            _op("add", "/processes/reactions/0/outputs/-", "MK-4"),
        ],
        locked_manifest=manifest,
    )

    assert report["summary"] == {"accepted_count": 2, "rejected_count": 0, "total": 2}
    assert patched["processes"]["reactions"][0]["locked_reaction_id"] == "rxn_lock_001"


def test_a_record_on_an_entity_row_does_not_disturb_the_integrity_guard() -> None:
    """Registry coverage is computed from name and synonyms. A record added to an
    entity row by one op must neither create nor destroy coverage for the next."""
    patched, report = apply_patch_with_policy(
        _payload(),
        [
            _op("add", "/entities/compounds/0/synonyms", ["1,4-dihydroxy-2-naphthoate"]),
            _op("remove", "/entities/compounds/1", evidence="duplicate of DMK"),
        ],
    )

    assert report["summary"] == {"accepted_count": 1, "rejected_count": 1, "total": 2}
    assert report["rejected"][0]["reason"].startswith(REFERENTIAL_INTEGRITY_REASON_PREFIX)
    assert [row["name"] for row in patched["entities"]["compounds"]] == ["DHNA", "DMK"]

"""FIX-NONAME: a curation patch must never leave an entity row without a name.

Under Gemma 4 31B, 2 of 35 legs aborted at the post_audit contract with
``entity_missing_name`` (legs4g vt1-PMC12096016, base1-PMC13231680). Stage-1 and
merged payloads had no nameless rows; the only writers of entity rows between
them and post_audit that take model output are the audit and gap-resolver
enrichment patches, and both go through ``apply_patch_with_policy``. The shapes
below are the ones that slip past every other guard there: they touch only
entities no process references, so the referential-integrity guard has nothing
to protect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402
from t2pw.pipeline.stage_contracts import StageContractError, validate_post_audit  # noqa: E402
from t2pw.service import postpipeline  # noqa: E402


def _lipid_a_payload() -> dict:
    """Trimmed from the PMC13231680 merged payload (lipid A, flags off)."""
    return {
        "entities": {
            "species": [{"name": "Escherichia coli"}],
            "compounds": [
                {"name": "lipid A"},  # orphan: no process names it
                {"name": "UDP-3-O-acyl-GlcNAc"},
                {"name": "myr-UDP-GlcN"},
            ],
            "proteins": [{"name": "LpxC"}],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "LpxC reaction",
                    "inputs": ["UDP-3-O-acyl-GlcNAc"],
                    "outputs": ["myr-UDP-GlcN"],
                    "enzymes": [{"protein": "LpxC"}],
                }
            ]
        },
    }


def _op(op: str, path: str, value=None) -> dict:
    row = {"op": op, "path": path, "confidence": 0.97, "evidence": "species lookup returned id 1"}
    if op != "remove":
        row["value"] = value
    return row


# Each is one Gemma-shaped op that, on 1a52fb55, is accepted and leaves a nameless row.
GEMMA_SHAPES = {
    "row_replaced_by_id_only": _op("replace", "/entities/species/0", {"pathbank_species_id": "1"}),
    "row_inserted_at_index": _op("add", "/entities/species/0", {"pathbank_species_id": "1"}),
    "name_blanked": _op("replace", "/entities/compounds/0/name", ""),
    "name_nulled": _op("replace", "/entities/compounds/0/name", None),
    "name_removed": _op("remove", "/entities/compounds/0/name"),
    "nameless_row_appended": _op(
        "add", "/entities/protein_complexes/-", {"complex_name": "LpxC complex", "components": [{"name": "LpxC"}]}
    ),
}


@pytest.mark.parametrize("shape", sorted(GEMMA_SHAPES))
def test_gemma_shape_no_longer_reaches_post_audit_nameless(shape: str) -> None:
    source = _lipid_a_payload()
    good = _op("add", "/entities/proteins/0/uniprot_id", "P0A725")
    patched, report = apply_patch_with_policy(source, [GEMMA_SHAPES[shape], good])

    # The run would have died here on 1a52fb55.
    validate_post_audit(patched)

    # Nothing with a valid name was lost or renamed, and nothing nameless was added.
    for bucket, rows in source["entities"].items():
        assert [row["name"] for row in patched["entities"][bucket]] == [row["name"] for row in rows]
    # The unrelated op in the same batch still applied.
    assert patched["entities"]["proteins"][0]["uniprot_id"] == "P0A725"
    # The refusal is recorded, never silent.
    assert [r["index"] for r in report["rejected"]] == [0]
    assert report["rejected"][0]["reason"].startswith("entity_name_required")
    assert report["transaction"]["committed"] is True


def test_blank_then_rename_in_one_batch_is_applied_exactly_as_before() -> None:
    """A batch whose end state names every row is not judged op by op."""
    ops = [
        _op("replace", "/entities/compounds/0/name", ""),
        _op("replace", "/entities/compounds/0/name", "lipid A disaccharide"),
    ]
    patched, report = apply_patch_with_policy(_lipid_a_payload(), ops)
    assert patched["entities"]["compounds"][0]["name"] == "lipid A disaccharide"
    assert report["rejected"] == []


def test_preexisting_nameless_row_does_not_block_unrelated_ops() -> None:
    source = _lipid_a_payload()
    source["entities"]["compounds"].append({"hmdb_id": "HMDB0000001"})
    patched, report = apply_patch_with_policy(source, [_op("add", "/entities/proteins/0/uniprot_id", "P0A725")])
    assert patched["entities"]["proteins"][0]["uniprot_id"] == "P0A725"
    assert report["rejected"] == []


def test_post_audit_failure_snapshot_names_bucket_and_index(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _lipid_a_payload()
    payload["entities"]["species"][0] = {"pathbank_species_id": "1"}
    with pytest.raises(StageContractError) as caught:
        validate_post_audit(payload)

    recorded: list = []

    class _Recorder:
        def record_mapped_failure(self, snapshot):
            recorded.append(snapshot)

    monkeypatch.setattr(postpipeline, "current_extraction_diagnostics", lambda: _Recorder())
    postpipeline._record_mapped_failure_snapshot(
        payload, stage=caught.value.stage, reason=str(caught.value), report=caught.value.report
    )

    (snapshot,) = recorded
    assert snapshot["failing_codes"] == ["entity_missing_name"]
    (issue,) = snapshot["failing_issues"]
    assert issue["code"] == "entity_missing_name"
    assert issue["bucket"] == "species"
    assert issue["pointer"] == "/entities/species/0/name"

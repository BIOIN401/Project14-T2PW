"""ORCH-769d (D-113 orchestrator ruling (a)): audit-patch index rebasing.

A patch is written against one payload, the pre-patch snapshot. apply_patch_with_policy
used to apply each op at its written index even after an earlier op in the same patch
had removed a row, so every later op on that array landed one row late (KI-018).

Every test here is a G9 correction proof: each FAILS behaviourally on base 3ef3fa9c
(the wrong row is written, or a neighbour is silently edited or deleted) and passes at
the tip. None imports a symbol the fix adds; reason strings are matched as literals.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402


def _op(action: str, path: str, value=None, confidence: float = 0.97) -> dict:
    op = {"op": action, "path": path, "confidence": confidence, "evidence": "ORCH-769d test"}
    if action != "remove":
        op["value"] = value
    return op


def _rows(payload: dict) -> list:
    return payload["entities"]["proteins"]


def _four_proteins() -> dict:
    """'dup' is declared as a protein and as a complex, the shape the Wnt/Ca2+ auditor removed."""
    return {
        "entities": {
            "proteins": [{"name": n, "note": ""} for n in ("A", "dup", "C", "D")],
            "protein_complexes": [{"name": "dup"}],
        },
        "processes": {},
    }


def test_g9_a_later_op_hits_the_row_it_was_written_for_after_a_remove() -> None:
    """[remove /x/1, replace /x/2/...]: /x/2 was C when the patch was written."""
    patched, report = apply_patch_with_policy(
        _four_proteins(),
        [_op("remove", "/entities/proteins/1"), _op("replace", "/entities/proteins/2/note", "written for C")],
    )

    assert {row["name"]: row["note"] for row in _rows(patched)} == {"A": "", "C": "written for C", "D": ""}
    assert report["summary"] == {"accepted_count": 2, "rejected_count": 0, "total": 2}
    record = report["accepted"][1]
    assert record["op"]["path"] == "/entities/proteins/2/note"  # recorded as written
    assert record["applied_path"] == "/entities/proteins/1/note"  # and where it landed


def test_g9_an_op_aimed_at_the_removed_row_is_rejected_not_redirected() -> None:
    patched, report = apply_patch_with_policy(
        _four_proteins(),
        [_op("remove", "/entities/proteins/1"), _op("replace", "/entities/proteins/1/note", "written for dup")],
    )

    # Base wrote this onto C, the row that slid into index 1.
    assert {row["name"]: row["note"] for row in _rows(patched)} == {"A": "", "C": "", "D": ""}
    assert report["summary"] == {"accepted_count": 1, "rejected_count": 1, "total": 2}
    assert report["rejected"][0]["index"] == 1
    assert report["rejected"][0]["reason"].startswith("removed_target:")
    assert report["transaction"]["committed"] is True


def test_g9_a_repeated_remove_does_not_delete_the_neighbour() -> None:
    patched, report = apply_patch_with_policy(
        _four_proteins(),
        [_op("remove", "/entities/proteins/1"), _op("remove", "/entities/proteins/1")],
    )

    # Base deleted C as well.
    assert [row["name"] for row in _rows(patched)] == ["A", "C", "D"]
    assert report["summary"] == {"accepted_count": 1, "rejected_count": 1, "total": 2}
    assert report["rejected"][0]["reason"].startswith("removed_target:")


def test_g9_ops_before_the_removed_row_and_inserts_are_rebased_correctly() -> None:
    """Indices below a remove do not move; an insert moves the rows at and after it."""
    patched, report = apply_patch_with_policy(
        _four_proteins(),
        [
            _op("remove", "/entities/proteins/1"),
            _op("add", "/entities/proteins/3", {"name": "new", "note": ""}),  # before D, as written
            _op("replace", "/entities/proteins/0/note", "written for A"),
            _op("replace", "/entities/proteins/3/note", "written for D"),
        ],
    )

    assert [(row["name"], row["note"]) for row in _rows(patched)] == [
        ("A", "written for A"),
        ("C", ""),
        ("new", ""),
        ("D", "written for D"),
    ]
    assert report["summary"] == {"accepted_count": 4, "rejected_count": 0, "total": 4}


def test_g9_the_integrity_guard_judges_the_rebased_row() -> None:
    """The referential-integrity guard must be handed the op as applied.

    /entities/compounds/2 is meropenem in the snapshot. Base removed
    'hydrolyzed meropenem' instead and refused it for the wrong reference.
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
                {"name": "Meropenem hydrolysis", "inputs": ["meropenem"], "outputs": ["hydrolyzed meropenem"]}
            ]
        },
    }

    patched, report = apply_patch_with_policy(
        payload, [_op("remove", "/entities/compounds/0"), _op("remove", "/entities/compounds/2")]
    )

    reason = report["rejected"][0]["reason"]
    assert reason.startswith("referential_integrity")
    assert "'meropenem' at /processes/reactions/0/inputs/0" in reason
    assert [row["name"] for row in patched["entities"]["compounds"]] == [
        "spectator",
        "meropenem",
        "hydrolyzed meropenem",
    ]


# ── ORCH-767 Wnt/Ca2+ (KI-018): a faithful minimal reconstruction ─────────────
#
# RECONSTRUCTION, not a recording. The leg (ORCH-767 wnt_ca-strict 2026-10-24_0314,
# PMC10425814) persisted neither the audit patch nor the audit's input payload. What
# it did persist is final_mapped.json, whose rows carry an audit_repair lineage entry
# on exactly the rows an accepted audit op wrote (ORCH-769d G11 02). Those rows are:
# proteins 2,3,4,6,7,8,9 with a synonym that is their own (short) name, and proteins
# 13,14,17,18,19,20 each carrying the NAME OF THE ROW BEFORE IT. The audit rationale
# says it removed the protein row 'calcineurin', declared also as a complex, and
# stage1_payload.json puts that row directly after SERCA, i.e. at index 10.
#
# The rows are listed below in that order, with 'calcineurin' put back at index 10;
# the patch removes it and adds each recorded synonym at its pre-patch index. On base
# this reproduces every recorded synonym in final_mapped.json, cross-row ones included
# (G11 evidence in ORCH-769d-RESULT.md); the processes are reduced to two interactions.

_WNT_CA_PROTEINS = [
    "Wnt5a",
    "Frizzled2",
    "G protein",
    "phospholipase C (PLC)",
    "PKC",
    "Cdc42",
    "inositol-1,4,5-trisphosphate receptor (InsP3R)",
    "STIM1/2",
    "Orai",
    "Sarcoplasmic/ER Ca2+ ATPases",
    "calcineurin",
    "NFAT",
    "Nemo-like kinase (NLK)",
    "TCF",
    "β-catenin",
    "calmodulin",
    "TCF4",
    "calcineurin A",
    "calcineurin B",
    "phosphorylated CAMKII",
    "phosphorylated TCF4",
    "NF-ATp",
    "Unknown",
]

_WNT_CA_SYNONYM_ADDS = {
    2: "G protein",
    3: "PLC",
    4: "PKC",
    6: "InsP3R",
    7: "STIM1/2",
    8: "Orai",
    9: "SERCA",
    13: "TCF",
    14: "β-catenin",
    17: "calcineurin A",
    18: "calcineurin B",
    19: "phosphorylated CAMKII",
    20: "phosphorylated TCF4",
}


def _wnt_ca_payload() -> dict:
    return {
        "entities": {
            "proteins": [{"name": name} for name in _WNT_CA_PROTEINS],
            "protein_complexes": [{"name": "calcineurin", "components": [{"name": "calcineurin B"}]}],
        },
        "processes": {
            "interactions": [
                {"entity_1": "calmodulin", "entity_2": "calcineurin", "type": "activation"},
                {"entity_1": "calcineurin B", "entity_2": "NFAT", "type": "activation"},
            ]
        },
    }


def _wnt_ca_patch() -> list:
    remove = _op("remove", "/entities/proteins/10")
    remove["evidence"] = "'calcineurin' is declared as both a protein and a protein_complex."
    adds = [
        _op("add", f"/entities/proteins/{index}/synonyms", [synonym], confidence=0.9)
        for index, synonym in sorted(_WNT_CA_SYNONYM_ADDS.items())
    ]
    return [remove, *adds]


def test_g9_wnt_ca_reconstruction_every_synonym_lands_on_its_own_row() -> None:
    patched, report = apply_patch_with_policy(_wnt_ca_payload(), _wnt_ca_patch())

    synonyms = {row["name"]: row.get("synonyms") for row in _rows(patched)}
    # The row that kept an unidentified protein alive on the leg (diagnosis variant C).
    assert synonyms["phosphorylated CAMKII"] == ["phosphorylated CAMKII"]
    # Each synonym is on the row whose pre-patch index the auditor wrote.
    assert synonyms == {
        name: ([_WNT_CA_SYNONYM_ADDS[index]] if index in _WNT_CA_SYNONYM_ADDS else None)
        for index, name in enumerate(_WNT_CA_PROTEINS)
        if index != 10
    }
    # The diagnosis's criterion (variant B): no protein is named by another row's synonym.
    names = {name.casefold() for name in synonyms}
    for name, values in synonyms.items():
        assert not [v for v in values or [] if v.casefold() in names and v.casefold() != name.casefold()], name
    assert report["summary"] == {"accepted_count": 14, "rejected_count": 0, "total": 14}
    assert report["transaction"]["committed"] is True

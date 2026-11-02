"""NEW ACCEPTANCE TEST (ORCH-799a, single-source refactor; no G9 base failure claimed).

``row_is_referenced`` is the one name-UNION-synonym membership test the strict
quarantine pruner and degree-zero detector share (FINDINGS F-208 row 1).
"""

from t2pw.pipeline.process_normalizer import row_is_referenced
from t2pw.pipeline.strict_quarantine import (
    CORE_ACCEPTED,
    _degree_zero_exports,
    _prune_entities,
)


def test_predicate_reads_name_and_synonyms_and_never_the_empty_norm() -> None:
    referenced = {"enzymex", "exa", ""}
    assert row_is_referenced({"name": "Enzyme X"}, referenced)
    assert row_is_referenced({"name": "Other", "synonyms": ["ExA"]}, referenced)
    assert not row_is_referenced({"name": "Enzyme Y", "synonyms": ["EyB"]}, referenced)
    # A punctuation-only name normalizes to "": it is never a reference.
    assert not row_is_referenced({"name": "-"}, referenced)
    assert not row_is_referenced({"name": ""}, referenced)


def _payload() -> dict:
    return {
        "entities": {
            "compounds": [{"name": "A"}, {"name": "B"}],
            "proteins": [
                {"name": "Enzyme X", "synonyms": ["ExA"]},
                {"name": "Enzyme Y", "synonyms": ["EyB"]},
            ],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "processes": {
            "reactions": [
                {"name": "A to B", "inputs": ["A"], "outputs": ["B"], "enzymes": [{"entity": "ExA"}]}
            ],
            "transports": [],
            "interactions": [],
        },
    }


def test_pruner_and_detector_keep_synonym_reference_and_flag_unreferenced() -> None:
    admissions = [{"state": CORE_ACCEPTED, "bucket": "reactions", "index": 0}]

    payload = _payload()
    assert _degree_zero_exports(payload, admissions, process_snapshot=payload["processes"]) == [
        {"bucket": "proteins", "name": "Enzyme Y"}
    ]

    removed = _prune_entities(payload, {"a", "b", "exa"})
    assert [row["name"] for row in payload["entities"]["proteins"]] == ["Enzyme X"]
    assert [(row["bucket"], row["name"]) for row in removed] == [("proteins", "Enzyme Y")]

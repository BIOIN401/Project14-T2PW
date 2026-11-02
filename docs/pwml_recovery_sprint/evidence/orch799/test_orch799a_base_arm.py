from t2pw.pipeline.strict_quarantine import (
    CORE_ACCEPTED,
    _degree_zero_exports,
    _prune_entities,
)


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

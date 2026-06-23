import json

from t2pw.pipeline.reaction_lock_manifest import (
    MANIFEST_FILENAME,
    RAW_STAGE1_FILENAME,
    REPORT_FILENAME,
    apply_locked_reaction_ids,
    build_locked_reaction_manifest,
    write_stage1_lock_artifacts,
    write_locked_reaction_manifest,
)


def test_manifest_excludes_out_of_scope_and_keeps_incomplete_reactions():
    payload = {
        "processes": {
            "reactions": [
                {
                    "reaction_id": "R2",
                    "name": "out of scope reaction",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "evidence": "A becomes B",
                    "scope_membership": "out_of_scope",
                    "confidence": 0.9,
                },
                {
                    "reaction_id": "R1",
                    "name": "incomplete reaction",
                    "inputs": ["C"],
                    "outputs": [],
                    "scope_membership": "core",
                    "confidence": 0.5,
                },
                {
                    "reaction_id": "R3",
                    "name": "complete reaction",
                    "inputs": ["D"],
                    "outputs": ["E"],
                    "modifiers": [{"entity": "Enzyme X", "role": "catalyst"}],
                    "evidence": "Enzyme X converts D to E",
                    "scope_membership": "auxiliary",
                    "confidence": 1.0,
                },
            ]
        }
    }

    manifest, report = build_locked_reaction_manifest(payload)

    assert [entry["locked_reaction_id"] for entry in manifest] == ["rxn_lock_001", "rxn_lock_002"]
    assert [entry["source_reaction_id"] for entry in manifest] == ["R1", "R3"]
    assert manifest[0]["status"] == "incomplete"
    assert manifest[0]["issues"] == ["missing_outputs", "missing_evidence"]
    assert manifest[1]["status"] == "valid"
    assert manifest[1]["modifiers_or_enzymes"] == ["Enzyme X"]

    assert report["locked_reaction_count"] == 2
    assert report["valid_count"] == 1
    assert report["incomplete_count"] == 1
    assert report["out_of_scope_excluded_count"] == 1
    assert report["duplicate_candidates"] == []
    assert report["incomplete_reactions"] == [
        {
            "locked_reaction_id": "rxn_lock_001",
            "source_reaction_id": "R1",
            "name": "incomplete reaction",
            "issues": ["missing_outputs", "missing_evidence"],
        }
    ]


def test_manifest_uses_chunk_fallback_and_source_refs_for_evidence():
    payloads = [
        {
            "payload": {
                "processes": {
                    "reactions": [
                        {
                            "name": "second alphabetically",
                            "inputs": ["B"],
                            "outputs": ["C"],
                            "source_refs": ["B is converted to C"],
                            "scope_membership": "core",
                        }
                    ]
                }
            },
            "source_chunk": "chunk_002",
        },
        {
            "payload": {
                "processes": {
                    "reactions": [
                        {
                            "name": "first alphabetically",
                            "inputs": ["A"],
                            "outputs": ["B"],
                            "evidence": "A is converted to B",
                            "scope_membership": "core",
                            "source_chunk": "source paragraph 1",
                        }
                    ]
                }
            },
            "source_chunk": "chunk_001",
        },
    ]

    manifest, report = build_locked_reaction_manifest(payloads)

    assert report["locked_reaction_count"] == 2
    assert [entry["locked_reaction_id"] for entry in manifest] == ["rxn_lock_001", "rxn_lock_002"]
    assert manifest[0]["name"] == "first alphabetically"
    assert manifest[0]["source_chunk"] == "source paragraph 1"
    assert manifest[1]["evidence_quote"] == "B is converted to C"
    assert manifest[1]["source_chunk"] == "chunk_002"


def test_write_locked_reaction_manifest_writes_expected_artifacts(tmp_path):
    payload = {
        "processes": {
            "reactions": [
                {
                    "name": "reaction",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "evidence": "A produces B",
                    "scope_membership": "core",
                }
            ]
        }
    }

    result = write_locked_reaction_manifest(payload, tmp_path)

    manifest_path = tmp_path / MANIFEST_FILENAME
    report_path = tmp_path / REPORT_FILENAME
    assert result["manifest_path"] == str(manifest_path)
    assert result["report_path"] == str(report_path)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == result["manifest"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == result["report"]


def test_write_stage1_lock_artifacts_includes_raw_extraction(tmp_path):
    payload = {
        "processes": {
            "reactions": [
                {
                    "name": "reaction",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "evidence": "A produces B",
                    "scope_membership": "core",
                }
            ]
        }
    }

    result = write_stage1_lock_artifacts(payload, tmp_path)

    raw_path = tmp_path / RAW_STAGE1_FILENAME
    assert result["raw_stage1_path"] == str(raw_path)
    assert json.loads(raw_path.read_text(encoding="utf-8")) == payload
    assert (tmp_path / MANIFEST_FILENAME).exists()
    assert (tmp_path / REPORT_FILENAME).exists()


def test_manifest_accepts_iterable_of_raw_payloads():
    payload = {
        "processes": {
            "reactions": [
                {
                    "name": "reaction",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "evidence": "A produces B",
                    "scope_membership": "core",
                }
            ]
        }
    }

    manifest, report = build_locked_reaction_manifest([payload])

    assert report["locked_reaction_count"] == 1
    assert manifest[0]["name"] == "reaction"


def test_apply_locked_reaction_ids_annotates_matching_raw_reactions():
    payload = {
        "processes": {
            "reactions": [
                {
                    "reaction_id": "R1",
                    "name": "reaction",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "evidence": "A produces B",
                    "scope_membership": "core",
                }
            ]
        }
    }

    manifest, _report = build_locked_reaction_manifest(payload)
    apply_locked_reaction_ids(payload, manifest)

    reaction = payload["processes"]["reactions"][0]
    assert reaction["locked_reaction_id"] == "rxn_lock_001"
    assert reaction["locked"] is True

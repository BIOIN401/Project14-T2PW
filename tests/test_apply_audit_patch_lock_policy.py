from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation.apply_audit_patch import apply_patch_with_policy, run_apply  # noqa: E402
from t2pw.pipeline.reaction_lock_manifest import MANIFEST_FILENAME  # noqa: E402


def _payload() -> dict:
    return {
        "entities": {
            "compounds": [{"name": "ATP"}, {"name": "ADP"}],
            "proteins": [{"name": "Kinase"}],
        },
        "processes": {
            "reactions": [
                {
                    "locked_reaction_id": "rxn_lock_001",
                    "source_reaction_id": "r1",
                    "name": "ATP conversion",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "enzymes": [{"entity": "Kinase"}],
                    "evidence": "ATP conversion was observed.",
                }
            ]
        },
    }


def _manifest() -> list[dict]:
    return [
        {
            "locked_reaction_id": "rxn_lock_001",
            "source_reaction_id": "r1",
            "name": "ATP conversion",
            "inputs": ["ATP"],
            "outputs": ["ADP"],
            "modifiers_or_enzymes": ["Kinase"],
            "evidence_quote": "ATP conversion was observed.",
            "locked": True,
        }
    ]


def test_locked_policy_rejects_locked_reaction_removal() -> None:
    payload = _payload()
    patch = [
        {
            "op": "remove",
            "path": "/processes/reactions/0",
            "confidence": 0.99,
            "evidence": "Unsupported duplicate.",
        }
    ]

    patched, report = apply_patch_with_policy(payload, patch, locked_manifest=_manifest(), stage="audit_round_1")

    assert patched == payload
    assert report["summary"] == {"accepted_count": 0, "rejected_count": 1, "total": 1}
    assert report["rejected"][0]["reason"] == "attempted_to_delete_locked_reaction"
    assert report["rejected"][0]["locked_reaction_id"] == "rxn_lock_001"
    assert report["rejected_patch_log"] == [
        {
            "stage": "audit_round_1",
            "patch": patch[0],
            "reason": "attempted_to_delete_locked_reaction",
            "locked_reaction_id": "rxn_lock_001",
        }
    ]


def test_locked_policy_accepts_missing_compound_addition() -> None:
    payload = _payload()
    patch = [
        {
            "op": "add",
            "path": "/entities/compounds/-",
            "value": {"name": "AMP"},
            "confidence": 0.90,
            "evidence": "AMP is needed as a missing compound.",
        }
    ]

    patched, report = apply_patch_with_policy(payload, patch, locked_manifest=_manifest(), stage="curator")

    assert report["summary"] == {"accepted_count": 1, "rejected_count": 0, "total": 1}
    assert patched["entities"]["compounds"][-1] == {"name": "AMP"}
    assert report["applied_patch_log"][0]["reason"] == "accepted"
    assert report["rejected_patch_log"] == []


def test_locked_policy_rejects_replacing_entire_reactions_array() -> None:
    payload = _payload()
    patch = [
        {
            "op": "replace",
            "path": "/processes/reactions",
            "value": [],
            "confidence": 0.99,
            "evidence": "Rewrite reactions.",
        }
    ]

    patched, report = apply_patch_with_policy(payload, patch, locked_manifest=_manifest())

    assert patched == payload
    assert report["summary"]["accepted_count"] == 0
    assert report["rejected"][0]["reason"] == "attempted_to_replace_reactions_array"


def test_without_locked_manifest_preserves_existing_reactions_array_replace_behavior() -> None:
    payload = _payload()
    replacement = [
        {
            "name": "Replacement",
            "inputs": ["A"],
            "outputs": ["B"],
        }
    ]
    patch = [
        {
            "op": "replace",
            "path": "/processes/reactions",
            "value": replacement,
            "confidence": 0.99,
            "evidence": "High-confidence rewrite.",
        }
    ]

    patched, report = apply_patch_with_policy(payload, patch)

    assert report["summary"] == {"accepted_count": 1, "rejected_count": 0, "total": 1}
    assert patched["processes"]["reactions"] == replacement
    assert "lock_policy" not in report


def test_locked_policy_rejects_deleting_compound_used_by_locked_reaction() -> None:
    payload = _payload()
    patch = [
        {
            "op": "remove",
            "path": "/entities/compounds/0",
            "confidence": 0.99,
            "evidence": "Remove ATP.",
        }
    ]

    patched, report = apply_patch_with_policy(payload, patch, locked_manifest=_manifest())

    assert patched == payload
    assert report["rejected"][0]["reason"] == "attempted_to_delete_locked_reaction_entity"
    assert report["rejected"][0]["locked_reaction_id"] == "rxn_lock_001"


def test_run_apply_discovers_locked_manifest_and_writes_patch_logs(tmp_path: Path) -> None:
    input_path = tmp_path / "final.json"
    patch_path = tmp_path / "audit_patch.json"
    output_path = tmp_path / "final.audited.json"
    report_path = tmp_path / "audit_apply_report.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    patch_path.write_text(
        json.dumps(
            [
                {
                    "op": "replace",
                    "path": "/processes/reactions",
                    "value": [],
                    "confidence": 0.99,
                    "evidence": "Rewrite reactions.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(_manifest()), encoding="utf-8")

    report = run_apply(
        input_path,
        patch_path,
        output_path,
        apply_report_path=report_path,
        stage="audit_round_1",
    )

    assert report["summary"] == {"accepted_count": 0, "rejected_count": 1, "total": 1}
    assert json.loads(output_path.read_text(encoding="utf-8")) == _payload()
    rejected_log = json.loads((tmp_path / "rejected_patch_log.json").read_text(encoding="utf-8"))
    applied_log = json.loads((tmp_path / "applied_patch_log.json").read_text(encoding="utf-8"))
    assert applied_log == []
    assert rejected_log[0]["stage"] == "audit_round_1"
    assert rejected_log[0]["reason"] == "attempted_to_replace_reactions_array"

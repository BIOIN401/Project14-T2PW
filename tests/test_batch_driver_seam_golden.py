"""Golden driver-observable equivalence across the C-012 ``_drive`` seam extraction.

C-012 lifts three terminal paths out of :func:`t2pw.batch.driver._drive` --
``_finalize_gate_failure``, ``_finalize_pwml_export`` and ``_finalize_timeout`` -- as a
PURE MOVE, so the acceptance criterion is an EMPTY golden diff, not a test count.

:data:`GOLDEN` was captured on this branch's BASE SHA before a line of ``driver.py``
moved. Its last slot is SHA-256 over the canonicalized WHOLE of :func:`_observable`,
which is what makes the comparison byte-for-byte over the entire field list
``TEST_MATRIX.md`` requires of a ``driver.py`` branch -- exit classification, the whole
``RunOutcome.to_dict()``, the artifact key set in write order and sorted, ``files[]``
name and bytes after ``runner._relocate_files``, ``issue_codes``, ``detail``,
``message``, ``warnings`` and ``canonical_payload_sha256`` as it reaches disk inside
``final_mapped.json`` -- rather than over a payload hash, which would let a refactor
drop a quarantine report or misclassify a timeout unnoticed. The two readable slots
beside it name the drift a reviewer looks for first, so a failure says what moved.

Leg fixtures are the throwaway app scripts below, built from
``tests/test_batch_driver.py``'s helpers so the two files cannot disagree about the
app's contract, and written to pytest's ``tmp_path``: C-012 checks in no fixture and
no evidence artifact.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.batch import driver, runner  # noqa: E402
from t2pw.batch.driver import RESEARCH, STRICT, RunOutcome, run_one  # noqa: E402
from test_batch_driver import (  # noqa: E402
    PAPER,
    _artifacts,
    _post_pipeline_body,
    _write_app,
    real_streamlit,  # noqa: F401 -- autouse fixture, re-exported on purpose
)

#: The canonical export payload, so ``canonical_payload_sha256`` is observable on the
#: disk artifact the way production writes it (``_add_identity_artifacts`` prefers
#: ``canonical_export_payload`` over ``final_mapped_db``).
_CANON = {"canonical_payload_sha256": "c" * 64, "processes": {"reactions": [{"name": "r"}]}}
_PWML = {
    "ok": True,
    "_xml": "<pathway><name>menaquinone</name></pathway>",
    "pwml_ir": {"pathway": {"name": "menaquinone"}},
    "counts": {"reactions": 1},
}
_GATE_ERR = {"path": "/entities/protein_complexes/0/name", "reason": "Forbidden complex reference detected: x"}
_GATE_FAIL = {"status": "failed", "stage": "post_normalization_hard_gates", "errors": [_GATE_ERR]}
_CONTRACT_ERR = {"code": "reaction_enzyme_unresolved", "pointer": "/processes/reactions/0", "message": "no identity"}
_CONTRACT_FAIL = {"ok": False, "stage": "post_extraction", "errors": [_CONTRACT_ERR], "warnings": []}
_HANG = '\nif submitted:\n    import time\n    time.sleep(4)\n    st.session_state["x"] = 1\n'


def _legs() -> dict:
    """The committed leg fixtures: ``{name: (app body, export mode, app timeout)}``.

    One per extracted terminal path, plus the neighbours that must not move with them:
    both ways into the gate finalizer, the two PWML exits that are not the success
    path, and the research deliverable, which shares no code with any of the three.
    """

    common = {"canonical_export_payload": _CANON}
    clean = _artifacts("pathwhiz", **common)
    gate = _artifacts("pathwhiz", normalization_gate_failed=True, gate_fail_report=_GATE_FAIL, **common)
    contract = _artifacts("pathwhiz", post_extraction_contract_report=_CONTRACT_FAIL, **common)
    no_xml = {"ok": False, "error": "the gate rejected the payload"}
    return {
        "strict_pwml_export": (_post_pipeline_body(clean, pwml=_PWML), STRICT, 45.0),
        "strict_export_not_ok": (_post_pipeline_body(clean, pwml=no_xml), STRICT, 45.0),
        "strict_no_pwml_button": (_post_pipeline_body(clean), STRICT, 45.0),
        "strict_gate_failure": (_post_pipeline_body(gate), STRICT, 45.0),
        "strict_contract_failure": (_post_pipeline_body(contract), STRICT, 45.0),
        "research_pass": (_post_pipeline_body(_artifacts("research", **common)), RESEARCH, 45.0),
        "input_timeout": (_HANG, STRICT, 1.0),
    }


def _sha(blob: object) -> str:
    data = blob if isinstance(blob, (bytes, bytearray)) else str(blob).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _observable(outcome: RunOutcome) -> dict:
    """Everything a driver caller or an artifact consumer can see, order preserved."""

    row = outcome.to_dict()
    assert isinstance(row.pop("seconds"), float)  # wall clock is not behaviour
    blobs = outcome.artifacts
    mapped = blobs.get("final_mapped.json")
    return {
        "row": row,
        "artifact_write_order": list(blobs),
        "artifact_keys_sorted": sorted(blobs),
        "artifact_digests": {name: [len(blob), _sha(blob)] for name, blob in blobs.items()},
        "relocated_files": runner._relocate_files(outcome.to_dict(), "pmc1", outcome.mode)["files"],
        "canonical_payload_sha256": json.loads(mapped)["canonical_payload_sha256"] if mapped else None,
        "release_status_absent": "release_status" not in row,
    }


def _observe(tmp_path: Path, leg: str) -> tuple:
    body, mode, app_timeout = _legs()[leg]
    app = _write_app(tmp_path, leg, body)
    outcome = run_one(PAPER, mode, app_path=app, timeout=120.0, app_timeout=app_timeout)
    seen = _observable(outcome)
    row = seen["row"]
    return (
        f"{row['status']}|{row['stage']}|{row['failure_kind']}",
        row["message"],
        _sha(json.dumps(seen, sort_keys=True)),
    )


#: Captured on the BASE SHA, before driver.py changed:
#: ``(status|stage|failure_kind, message, sha256 of the canonicalized observable)``.
GOLDEN: dict = {
    "input_timeout": ("timeout|input|timeout", "extraction did not finish inside the time budget", "5f49be0ff60ffa75b32762a7f5aa577fcabee6d0885137b657f71baa450df096"),
    "research_pass": ("pass|research_report|", "research run completed; no RAG synthesis in this run, so no citation report was produced", "f0c0ca68385547a671a94d9ae57fe39a4864ffbc03c8874261c89a8cc028c7e2"),
    "strict_contract_failure": ("fail|post_pipeline|contract", "post-pipeline validation failed: 1 blocking issue(s) at a stage boundary", "7c74c7b62e93dbc7924e356b9e46db3099a6903eb2071ecd41497f7efa7dc399"),
    "strict_export_not_ok": ("fail|pwml_export|unknown", "PWML export failed: the gate rejected the payload", "7e13ded1e51efb6f86e26346942a07e525d614e7cb03022d562bb021784d9058"),
    "strict_gate_failure": ("fail|post_pipeline|contract", "post-pipeline validation failed: 1 blocking issue(s) at post_normalization_hard_gates", "9ad6c2721e827c0ae844b4bb8939630b70f216975942436ea5e902cc47cf44a4"),
    "strict_no_pwml_button": ("fail|pwml_export|unknown", "the app never rendered the \"refinement_generate_pwml\" button ('refinement_generate_pwml'), so the review step that unlocks PWML export was not reached", "a39147d3cfc5c0f345628d648e3d02d534f352dd14e964c12c7011c6e093f7bf"),
    "strict_pwml_export": ("pass|pwml_export|", "strict run completed; pathway.pwml is 43 bytes", "eb5ded64c2afa8b0db64f11e4059ec6e1d811773b0da84e31b6428959061dd0d"),
}


def test_driver_observable_behaviour_is_byte_identical_to_the_base_sha(tmp_path: Path) -> None:
    """C-012's acceptance criterion: the golden driver-observable diff is EMPTY."""

    assert {leg: _observe(tmp_path, leg) for leg in sorted(_legs())} == GOLDEN


def test_the_three_terminal_paths_are_addressable_module_level_seams() -> None:
    """Fails on the BASE SHA: the seams C-031/C-032/C-041/C-053 need did not exist."""

    for name in ("_finalize_gate_failure", "_finalize_pwml_export", "_finalize_timeout"):
        assert callable(getattr(driver, name, None)), f"{name} is not an addressable seam"

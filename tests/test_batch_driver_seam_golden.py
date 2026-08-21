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
        # C-053 / MASTER_PLAN 3 hotspot 10, GROWTH-ONLY. ``release_status_absent``
        # made ABSENCE the invariant, which was right only while no card owned the
        # row. D-004 puts the classification and the artifact name IN the row, so
        # the invariant becomes PRESENCE and both are recorded BY VALUE -- one
        # field replaced by two, none dropped. The rule this encodes: a digest is
        # never stabilised by deleting a field from this list.
        "release_status": row.get("release_status"),
        "pwml_artifact": row.get("pwml_artifact"),
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
#:
#: **RE-BASELINED by C-053 under merge rule 4 / hotspot 10.** All seven digests
#: move, which is expected and was budgeted: ``_observable`` folds its whole field
#: list into one digest, so replacing ``release_status_absent`` with the two D-004
#: fields moves every slot at once. The move is derived, not asserted -- the base
#: capture is committed at ``docs/pwml_recovery_sprint/evidence/c053_golden_base.json``
#: (it reproduced the seven pre-C-053 tuples exactly), the tip capture beside it,
#: and the slot-by-slot difference at ``c053_golden_delta.json``. Reading down that
#: delta:
#:
#: * **slot 0** (``status|stage|failure_kind``) moves on **no** leg. No leg changed
#:   its exit classification.
#: * **slot 1** (``message``) moves on exactly ONE leg, ``strict_pwml_export``:
#:   ``pathway.pwml`` -> ``pathway.review_required.pwml``, because that fixture's
#:   PWML result carries no frozen record and D-038 3 names that export honestly.
#: * **slot 2** (digest) moves on all seven, and the per-leg field difference says
#:   why: the four legs that never reached a classification differ in
#:   ``release_status_absent`` alone (the replaced field); the two gate-blocked legs
#:   add ``release_status`` to the row, which C-041 built and deliberately left out
#:   of it; and ``strict_pwml_export`` additionally moves its artifact name, hence
#:   ``artifact_*``, ``relocated_files`` and ``pwml_artifact``.
#:
#: **RE-BASELINED AGAIN by C-056b under merge rule 4.** ``ReleaseStatus.to_dict``
#: grew ``semantic_failed_checks`` (schema 4 -> 5, D-039 section 5), and the row
#: carries the record verbatim, so any leg whose row HAS a classification moves its
#: digest. The move is derived, not asserted: base capture
#: ``evidence/c056b_golden_base.json`` (taken in a real worktree at ``01bb7ef``),
#: tip capture ``c056b_golden_tip.json``, slot-by-slot difference
#: ``c056b_golden_delta.json``. The exact delta:
#:
#: * **slot 0** (``status|stage|failure_kind``) moves on **0 of 7** legs.
#: * **slot 1** (``message``) moves on **0 of 7** legs.
#: * **slot 2** (digest) moves on **exactly 2 of 7** -- ``strict_gate_failure`` and
#:   ``strict_contract_failure``, the only two fixtures that reach
#:   ``_finalize_gate_failure`` and therefore the only two whose row carries a
#:   ``release_status`` at all. The other five never produce a classification and
#:   are byte-identical.
#: * on both moving legs the ONLY differing observable field is ``release_status``
#:   (and ``row``, which contains it): **one key added, ``semantic_failed_checks``,
#:   with value ``[]``; zero keys removed; zero value changes on any shared key.**
#:   ``fields_added`` and ``fields_dropped`` are both empty and the capture's own
#:   GROWTH-ONLY guard is True -- no digest was stabilised by dropping a field.
#:
#: **RE-BASELINED AGAIN by C-056c under merge rule 4**, authorized in advance by
#: D-054 section 8. ``ReleaseStatus.to_dict`` grew ``semantic_check_evaluability``
#: (quarantine report schema 5 -> 6), and the row carries the record verbatim, so
#: the same two legs move for the same reason. Derived, not asserted: base capture
#: ``evidence/c056c_golden_base.json`` (taken in a real git worktree at the base
#: SHA ``1cbfa01``, with ``.env`` copied in as an F-051 control on BOTH sides),
#: tip capture ``c056c_golden_tip.json``, slot-by-slot difference
#: ``c056c_golden_delta.json``. The base capture reproduced all seven digests
#: above BYTE-IDENTICALLY before the tip was taken, which is what makes the move
#: attributable. The exact delta:
#:
#: * **slot 0** (``status|stage|failure_kind``) moves on **0 of 7** legs.
#: * **slot 1** (``message``) moves on **0 of 7** legs.
#: * **slot 2** (digest) moves on **exactly 2 of 7** -- ``strict_contract_failure``
#:   (``b1a7a743…`` -> ``d102035d…``) and ``strict_gate_failure``
#:   (``50d3c8f5…`` -> ``a55b08af…``). They are the only two fixtures reaching
#:   ``_finalize_gate_failure``, so the only two whose row carries a
#:   ``release_status``. The other five carry none and are byte-identical --
#:   measured, not assumed: the capture reports ``release_status`` as ``None`` on
#:   ``input_timeout``, ``research_pass``, ``strict_export_not_ok``,
#:   ``strict_no_pwml_button`` and ``strict_pwml_export``.
#: * on both moving legs the ONLY differing observable field is ``release_status``
#:   (and ``row``, which contains it): **one key added,
#:   ``semantic_check_evaluability``, with value ``[]``; zero keys removed; zero
#:   value changes on any shared key.** ``[]`` is right there and not a gap:
#:   ``driver.py:1770`` classifies a gate failure without the new input, taking
#:   the byte-preserving default, and the field's documented meaning for empty is
#:   "not recorded" -- which is exactly true of a run that never reached the
#:   semantic seam. ``fields_added``/``fields_dropped`` are both empty and the
#:   capture's own GROWTH-ONLY guard is True.
#:
#: **RE-BASELINED AGAIN by C-056d under merge rule 4** (F-055), by the same
#: mechanism D-054 section 8 ratified for C-056c. C-056d makes
#: ``_finalize_gate_failure`` CARRY the quarantine boundary's semantic verdict
#: instead of re-deriving one from nothing, and corrects the text of
#: ``SEMANTIC_INPUT_NOT_WIRED`` (``release_status.py:58-61``), which named C-056a
#: as pending after C-056a had merged at ``93594aa``. Derived, not asserted: base
#: capture ``evidence/c056d_golden_base.json`` (taken in a real git worktree at the
#: base SHA ``40fdb23``, with ``.env`` copied in as an F-051 control on BOTH
#: sides), tip capture ``c056d_golden_tip.json``, slot-by-slot difference
#: ``c056d_golden_delta.json``. **The base capture reproduced all seven tuples
#: above BYTE-IDENTICALLY before the tip was taken**, which is what makes the move
#: attributable. The exact delta:
#:
#: * **slot 0** (``status|stage|failure_kind``) moves on **0 of 7** legs.
#: * **slot 1** (``message``) moves on **0 of 7** legs.
#: * **slot 2** (digest) moves on **exactly 2 of 7** -- ``strict_contract_failure``
#:   (``d102035d…`` -> ``8a5c7a80…``) and ``strict_gate_failure``
#:   (``a55b08af…`` -> ``3279e51b…``). The same two as C-056b and C-056c, for the
#:   same structural reason: they are the only fixtures reaching
#:   ``_finalize_gate_failure`` and therefore the only rows carrying a
#:   ``release_status`` at all. The other five carry ``None`` and are
#:   byte-identical -- measured, not assumed.
#: * on both moving legs: **zero keys added, zero keys dropped, and exactly ONE
#:   value changed -- ``semantic_not_evaluated_reason``**, the corrected constant.
#:   The GROWTH-ONLY guard is True; no digest was stabilised by dropping a field.
#: * **the carry itself contributes nothing to this delta, and that is the point.**
#:   Neither fixture publishes a ``quarantine_artifacts`` map, so no boundary
#:   record exists for these legs to carry and both take the honest
#:   ``not_evaluated`` fallback. The carried-verdict behaviour is proved on real
#:   committed boundary records in
#:   ``tests/test_c056d_gate_failure_semantic_carry.py``, which is red on the base
#:   SHA with ``assert 'not_evaluated' == 'failed'``.
GOLDEN: dict = {
    "input_timeout": ("timeout|input|timeout", "extraction did not finish inside the time budget", "382cc778b455d0c776c58455b2db22d6eba86740350b63edec2039350721efe5"),
    "research_pass": ("pass|research_report|", "research run completed; no RAG synthesis in this run, so no citation report was produced", "cfb25c20c9fee6cbf0e60254d3ec0a9db37214df5906e53b264b73da944485b3"),
    "strict_contract_failure": ("fail|post_pipeline|contract", "post-pipeline validation failed: 1 blocking issue(s) at a stage boundary", "8a5c7a80e4f97e7e62ab21005c6eb21f10d05a7e5980845b497d9f2620595205"),
    "strict_export_not_ok": ("fail|pwml_export|unknown", "PWML export failed: the gate rejected the payload", "beed6d1d332465d944dd2447914acdf630618b7d94ace556e1062f63f9f9f2f4"),
    "strict_gate_failure": ("fail|post_pipeline|contract", "post-pipeline validation failed: 1 blocking issue(s) at post_normalization_hard_gates", "3279e51b71a3548bae18b0b1360916f8b4287bda852f10f790f57da0a1a2852b"),
    "strict_no_pwml_button": ("fail|pwml_export|unknown", "the app never rendered the \"refinement_generate_pwml\" button ('refinement_generate_pwml'), so the review step that unlocks PWML export was not reached", "8ab600ff81d56ec1c9540ab79f1a822079e417b8772a638db89b23b0e8672e14"),
    "strict_pwml_export": ("pass|pwml_export|", "strict run completed; pathway.review_required.pwml is 43 bytes", "a768148079da3a9ff9de8312ac9255ebe05c26318739fbe6fa2e97543e93e027"),
}


def test_driver_observable_behaviour_is_byte_identical_to_the_base_sha(tmp_path: Path) -> None:
    """C-012's acceptance criterion: the golden driver-observable diff is EMPTY."""

    assert {leg: _observe(tmp_path, leg) for leg in sorted(_legs())} == GOLDEN


def test_the_three_terminal_paths_are_addressable_module_level_seams() -> None:
    """Fails on the BASE SHA: the seams C-031/C-032/C-041/C-053 need did not exist."""

    for name in ("_finalize_gate_failure", "_finalize_pwml_export", "_finalize_timeout"):
        assert callable(getattr(driver, name, None)), f"{name} is not an addressable seam"

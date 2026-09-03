"""C-031: the quarantine boundary's four artifacts must survive the batch seam.

THE FAILURE THIS CLOSES. ``write_quarantine_artifacts`` produces four artifacts --
``quarantine_report.json``, ``removed_entity_report.json``,
``graph_closure_iterations.json``, ``coverage_summary.json`` -- and the batch driver
never carried a single one of them into the leg directory:
``find runs runs_verify -name quarantine_report.json`` returned **zero** across all 15
committed run directories, and ``bench.acceptance``'s ``_QUARANTINE_FILES`` lookup has
therefore never once found a file. The consequence is not cosmetic. A strict leg that
the boundary REFUSED writes its pre-quarantine fallback payload and no record of the
refusal, so "which processes were dropped, and why" is not a question that gets a wrong
answer after the run -- it is a question that cannot be asked at all.

The producer is not reimplemented here and its output is not reshaped. These tests run
the REAL ``quarantine_and_close`` + ``write_quarantine_artifacts`` and then assert that
what reaches the leg directory is character-for-character what the producer emitted.

Fixture apps are built from ``tests/test_batch_driver.py``'s helpers, and the payload
from ``tests/test_strict_quarantine.py``'s, so this file cannot disagree with either
about the app's contract or about what a quarantinable pathway looks like. Everything
these tests write goes under pytest's ``tmp_path``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.batch import runner  # noqa: E402
from t2pw.batch.driver import STRICT, RunOutcome  # noqa: E402
from t2pw.bench import acceptance  # noqa: E402
from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    CLOSURE_REPORT_FILENAME,
    COVERAGE_DIAGNOSTICS_FILENAME,
    COVERAGE_REPORT_FILENAME,
    QUARANTINE_REPORT_FILENAME,
    REMOVED_ENTITY_REPORT_FILENAME,
    quarantine_and_close,
    write_quarantine_artifacts,
)
from test_batch_driver import (  # noqa: E402
    PAPER,
    _RELEASE_READY,
    _artifacts,
    _lit,
    _post_pipeline_body,
    _run,
    _write_app,
    real_streamlit,  # noqa: F401 -- autouse fixture, re-exported on purpose
)
from test_strict_quarantine import _add_reaction, _glutathione_payload  # noqa: E402


#: The four ``write_quarantine_artifacts`` RETURNS, in the order it writes them.
#: Still exactly four: F-175 did not widen the returned map, and
#: ``test_f175_the_returned_map_pin_is_untouched`` holds it to that.
FOUR: Tuple[str, ...] = (
    QUARANTINE_REPORT_FILENAME,
    REMOVED_ENTITY_REPORT_FILENAME,
    CLOSURE_REPORT_FILENAME,
    COVERAGE_REPORT_FILENAME,
)

#: The five the BATCH DRIVER CARRIES into a leg directory. These are different
#: sets and the difference is the whole of F-175: the producer writes five
#: documents to disk and returns four, so the fifth has to be carried by a
#: sibling-path resolution rather than by reading the map.
CARRIED: Tuple[str, ...] = FOUR + (COVERAGE_DIAGNOSTICS_FILENAME,)

_PWML = {
    "ok": True,
    "_xml": "<pathway><name>glutathione</name></pathway>",
    "pwml_ir": {"pathway": {"name": "glutathione"}},
    "counts": {"reactions": 2},
    # C-053 / D-004: the export carries the boundary's frozen decision, and the
    # PWML filename is derived from it. These legs are ordinary passing strict
    # runs, so the record says so; a fixture omitting it would silently become a
    # classification-unavailable leg, which is a different behaviour tested in
    # ``tests/test_batch_pwml_artifact_naming.py``.
    "quarantine_report": {"release": _RELEASE_READY},
}
_GATE_FAIL = {
    "status": "failed",
    "stage": "post_normalization_hard_gates",
    "errors": [
        {
            "path": "/entities/protein_complexes/0/name",
            "reason": "Forbidden complex reference detected: x",
        }
    ],
}

#: The one dropped process. Its essential participant is declared nowhere, which is
#: the exact shape that used to fail the required-field gate for a whole pathway.
_PERIPHERAL = {
    "name": "peripheral oxidation",
    "inputs": ["glutathione"],
    "outputs": ["glutathione disulfide"],
    "biological_state": "cytosol",
}


def _produce(tmp_path: Path, name: str = "outputs") -> Tuple[Path, Dict[str, str]]:
    """Run the REAL producer once. Returns ``(outputs_dir, {name: path})``.

    Deliberately the production call, not a hand-built fixture: the claim under test
    is that the driver persists *what the boundary emitted*, and a mimicked document
    could not fail if the producer's shape moved.
    """

    payload = _add_reaction(_glutathione_payload(), dict(_PERIPHERAL))
    outputs = tmp_path / name
    written = write_quarantine_artifacts(quarantine_and_close(payload, strict_db=True), outputs)
    assert set(written) == set(FOUR), written
    return outputs, written


def _row(outcome: RunOutcome) -> Dict[str, Any]:
    """The manifest row minus the two fields this card is allowed to move."""

    row = outcome.to_dict()
    row.pop("seconds")  # wall clock is not behaviour
    row.pop("files")  # artifact names/bytes: the deliberate, enumerated move
    return row


# ---------------------------------------------------------------------------
# G9 regression: the artifacts did not reach the leg directory.
# ---------------------------------------------------------------------------
def test_the_four_quarantine_artifacts_survive_the_batch_seam(tmp_path: Path) -> None:
    """G9 REGRESSION. Fails on the base SHA: none of the four is in the leg.

    Asserted on artifact CONTENT, not on a symbol: ``write_quarantine_artifacts``
    exists on the base SHA and produced these very files there too. What did not
    happen on the base SHA is the batch boundary carrying them across.
    """

    outputs, written = _produce(tmp_path)
    app = _write_app(
        tmp_path,
        "quarantine_pass",
        _post_pipeline_body(_artifacts("pathwhiz", quarantine_artifacts=written), pwml=_PWML),
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "pass", outcome.detail
    missing = [name for name in FOUR if name not in outcome.artifacts]
    assert not missing, f"the batch boundary dropped {missing}"
    for name in FOUR:
        assert outcome.artifacts[name] == (outputs / name).read_text(encoding="utf-8"), (
            f"{name} is not what write_quarantine_artifacts emitted"
        )


def test_a_refused_leg_lands_the_evidence_naming_what_was_dropped(tmp_path: Path) -> None:
    """G9 REGRESSION. The failing leg is the one the evidence exists for.

    A leg blocked at the gate never reaches the PWML export, which is exactly the
    case in which the run directory used to hold no quarantine record at all. The
    dropped process and its undeclared participant have to be readable afterwards
    from the file, not inferred from a ``RESULT.txt`` sentence.
    """

    outputs, written = _produce(tmp_path)
    blocked = _artifacts(
        "pathwhiz",
        normalization_gate_failed=True,
        gate_fail_report=_GATE_FAIL,
        quarantine_artifacts=written,
    )
    app = _write_app(tmp_path, "quarantine_gate_fail", _post_pipeline_body(blocked))

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert "pathway.pwml" not in outcome.artifacts
    report = json.loads(outcome.artifacts[QUARANTINE_REPORT_FILENAME])
    assert [row["name"] for row in report["quarantined"]] == ["peripheral oxidation"]
    assert report["quarantined"][0]["essential_participant"] == "glutathione disulfide"
    # The other three carry the rest of the decision, unaltered by this driver.
    assert json.loads(outcome.artifacts[COVERAGE_REPORT_FILENAME]) == json.loads(
        (outputs / COVERAGE_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    assert json.loads(outcome.artifacts[CLOSURE_REPORT_FILENAME])["converged"] is True
    assert "removed_entities" in json.loads(outcome.artifacts[REMOVED_ENTITY_REPORT_FILENAME])


def test_the_evidence_survives_a_run_that_died_after_the_boundary(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. The boundary ran; the artifact set was never stamped.

    ``post_pipeline_artifacts`` is written at the END of the post-pipeline step, so a
    run that takes the boundary and then dies inside the same step publishes nothing
    through it. The session key the boundary itself writes is the only surviving
    hand-off, and losing the decision because the run died after taking it is the
    worst case for a refusal record.
    """

    outputs, written = _produce(tmp_path)
    app = _write_app(
        tmp_path,
        "died_after_boundary",
        f'''
if submitted:
    st.session_state["pipeline_ready"] = True
    st.session_state["final_payload"] = {{"processes": {{"reactions": [{{"name": "r"}}]}}}}

if st.session_state.get("pipeline_ready"):
    if st.button("Run audit and DB mapping", key="pwml_generate_btn"):
        st.session_state["quarantine_artifacts"] = {_lit(written)}
        st.error("Stage contract failed: Payload must include a processes object.")
        st.stop()
''',
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    for name in FOUR:
        assert outcome.artifacts[name] == (outputs / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# New acceptance: the consumer can finally find the file.
# ---------------------------------------------------------------------------
def test_bench_acceptance_finds_the_quarantine_report_in_the_leg_directory(
    tmp_path: Path,
) -> None:
    """NEW ACCEPTANCE. ``acceptance._QUARANTINE_FILES`` resolves for the first time.

    Goes through ``runner.write_artifacts`` rather than asserting on the in-memory
    dict, because the lookup ``bench/acceptance.py`` performs is against files on
    disk in ``papers/<slug>/<mode>/`` and nothing else.
    """

    outputs, written = _produce(tmp_path)
    app = _write_app(
        tmp_path,
        "acceptance_reachable",
        _post_pipeline_body(_artifacts("pathwhiz", quarantine_artifacts=written), pwml=_PWML),
    )
    outcome = _run(app, STRICT)

    leg_dir = tmp_path / "run" / "papers" / "pmc1" / "strict"
    files = runner.write_artifacts(leg_dir, outcome.artifacts)
    assert not [entry for entry in files if entry.get("error")]

    found, source, path = acceptance._first_existing(leg_dir, acceptance._QUARANTINE_FILES)
    assert source == QUARANTINE_REPORT_FILENAME, "acceptance still cannot find the report"
    assert Path(path) == leg_dir / QUARANTINE_REPORT_FILENAME
    # Byte-identical to the producer's file after the round trip through the runner.
    assert (leg_dir / QUARANTINE_REPORT_FILENAME).read_text(encoding="utf-8") == (
        outputs / QUARANTINE_REPORT_FILENAME
    ).read_text(encoding="utf-8")
    assert found == json.loads((outputs / QUARANTINE_REPORT_FILENAME).read_text(encoding="utf-8"))


def test_only_a_named_artifact_at_its_own_path_is_persisted(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. A session-state map is not a trusted source of filenames.

    Two independent guards, both required. The KEY must be one of the four the
    producer emits, or a session key could name any file in the run directory; and
    the PATH's basename must be that same name, or the driver would copy arbitrary
    file contents into the run under a filename an auditor trusts.
    """

    decoy = tmp_path / "not_a_quarantine_report.json"
    decoy.write_text('{"unrelated": true}', encoding="utf-8")
    hostile = {
        QUARANTINE_REPORT_FILENAME: str(decoy),
        "../../escaped.json": str(decoy),
        "final_mapped.json": str(decoy),
    }
    app = _write_app(
        tmp_path,
        "hostile_map",
        _post_pipeline_body(_artifacts("pathwhiz", quarantine_artifacts=hostile), pwml=_PWML),
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "pass", outcome.detail
    assert QUARANTINE_REPORT_FILENAME not in outcome.artifacts
    assert "escaped.json" not in outcome.artifacts
    assert "../../escaped.json" not in outcome.artifacts
    # final_mapped.json is written by this driver from the canonical payload; the
    # map must never be able to substitute a file for it.
    assert json.loads(outcome.artifacts["final_mapped.json"]) != {"unrelated": True}


# ---------------------------------------------------------------------------
# Preservation: nothing else moved.
# ---------------------------------------------------------------------------
def test_only_the_carried_artifacts_move_when_a_leg_produces_quarantine_output(
    tmp_path: Path,
) -> None:
    """PRESERVATION. Same leg with and without the map: the carried set, nothing else.

    The manifest row is compared whole, minus ``seconds`` (wall clock) and ``files``
    (the artifact names and byte counts, which is the deliberate pinned-baseline
    move this card makes). Every other driver-observable field -- status, stage,
    failure_kind, message, detail, issue_codes, counts, warnings -- must be equal,
    and every artifact the leg already produced must be byte-identical.

    PINNED BASELINE MOVED DELIBERATELY, ORCH-722, merge rule 4, exact delta:
    ``FOUR`` -> ``CARRIED``, i.e. ``+1`` name, ``coverage_diagnostics.json``, under
    the narrow D-090 exception of 2026-09-03 authorising F-175 artifact
    persistence. Renamed from ``..._four_artifacts_...`` because the number in the
    old name became false, and a test whose name misdescribes its assertion is the
    F-172 defect in miniature.

    NOTHING ELSE ABOUT THIS TEST WEAKENS. The whole-row equality and the
    byte-identity of every pre-existing artifact are unchanged, and they are what
    make this the preservation proof: exactly one name appears, and no other
    driver-observable field moves with it.
    """

    _outputs, written = _produce(tmp_path)
    plain = _run(
        _write_app(tmp_path, "plain", _post_pipeline_body(_artifacts("pathwhiz"), pwml=_PWML)),
        STRICT,
    )
    with_quarantine = _run(
        _write_app(
            tmp_path,
            "with_quarantine",
            _post_pipeline_body(_artifacts("pathwhiz", quarantine_artifacts=written), pwml=_PWML),
        ),
        STRICT,
    )

    assert _row(with_quarantine) == _row(plain)
    assert set(with_quarantine.artifacts) - set(plain.artifacts) == set(CARRIED)
    assert {
        name: blob for name, blob in with_quarantine.artifacts.items() if name not in CARRIED
    } == plain.artifacts


def test_a_leg_that_never_reached_the_boundary_is_untouched(tmp_path: Path) -> None:
    """PRESERVATION. No map means no new files and no new failure mode."""

    app = _write_app(
        tmp_path,
        "no_boundary",
        _post_pipeline_body(_artifacts("pathwhiz"), pwml=_PWML),
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "pass", outcome.detail
    assert set(outcome.artifacts).isdisjoint(FOUR)
    assert outcome.warnings == []


def test_an_artifact_the_driver_cannot_read_is_a_diagnosis_not_a_crash(
    tmp_path: Path,
) -> None:
    """PRESERVATION. A deleted or locked artifact must not fail an otherwise good leg.

    The boundary's ``outputs/`` directory outlives no run: ``reset_quarantine_state``
    clears it when the next pipeline run starts. A map pointing at a file that is
    gone by the time the driver reads it therefore has to degrade to "no record",
    the same way a run that never reached mapping has no ``final_mapped.json``.
    """

    absent = {name: str(tmp_path / "swept_away" / name) for name in FOUR}
    app = _write_app(
        tmp_path,
        "unreadable_artifacts",
        _post_pipeline_body(_artifacts("pathwhiz", quarantine_artifacts=absent), pwml=_PWML),
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "pass", outcome.detail
    assert set(outcome.artifacts).isdisjoint(FOUR)
    assert outcome.warnings == []
    assert outcome.issue_codes == []


# ---------------------------------------------------------------------------
# F-175 (ORCH-722) -- the D-088 coverage diagnostics must reach a BATCH LEG.
#
# THE TEST IS THE POINT, and it is the whole lesson of the finding. C-116 shipped
# ELEVEN passing tests for these diagnostics and not one of them ran the batch
# path, so a file that reached zero benchmark legs was covered by a green suite.
# Everything below therefore goes through the REAL driver -- ``_run(app, STRICT)``
# executes the batch leg, and ``runner.write_artifacts`` puts the result on disk
# in ``papers/<slug>/<mode>/`` -- and never asserts on an in-memory dict alone.
#
# WHY THE FILE NEEDS A SEPARATE CARRY AT ALL. ``write_quarantine_artifacts``
# writes FIVE documents and deliberately RETURNS four: its own comment says the
# returned map "stays the four-name set two unowned pinned consumers assert by
# equality". Those consumers are still asserted below, unchanged. So the driver
# resolves the fifth as a SIBLING of the coverage report rather than by widening
# the map, and ``_produce``'s ``set(written) == set(FOUR)`` still holds.
# ---------------------------------------------------------------------------
def test_f175_the_diagnostics_reach_a_real_batch_leg_directory(tmp_path: Path) -> None:
    """G9 REGRESSION. Fails on the base SHA: the file reaches no leg directory.

    Asserted on CONTENT and on DISK, not on a symbol.
    ``COVERAGE_DIAGNOSTICS_FILENAME`` and its producer both exist on the base SHA
    and the file is written to the app's ``outputs/`` there too -- what does not
    happen on base is the batch boundary carrying it across, which is precisely
    why a symbol-absence proof would be worthless here.
    """

    outputs, written = _produce(tmp_path)
    # The producer wrote five files and handed back four. That asymmetry is the
    # defect's whole mechanism, so it is pinned here rather than assumed.
    assert COVERAGE_DIAGNOSTICS_FILENAME not in written
    assert (outputs / COVERAGE_DIAGNOSTICS_FILENAME).is_file()

    app = _write_app(
        tmp_path,
        "f175_diagnostics",
        _post_pipeline_body(_artifacts("pathwhiz", quarantine_artifacts=written), pwml=_PWML),
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "pass", outcome.detail
    assert COVERAGE_DIAGNOSTICS_FILENAME in outcome.artifacts, (
        "the batch boundary dropped the D-088 coverage diagnostics"
    )
    # ...and it is the producer's bytes, not something the driver re-authored.
    assert outcome.artifacts[COVERAGE_DIAGNOSTICS_FILENAME] == (
        outputs / COVERAGE_DIAGNOSTICS_FILENAME
    ).read_text(encoding="utf-8")

    # THROUGH THE RUNNER AND ONTO DISK. An offline evaluator reads the leg
    # directory, so an in-memory dict is not where this claim can be settled.
    leg_dir = tmp_path / "run" / "papers" / "pmc1" / "strict"
    files = runner.write_artifacts(leg_dir, outcome.artifacts)
    assert not [entry for entry in files if entry.get("error")]
    landed = leg_dir / COVERAGE_DIAGNOSTICS_FILENAME
    assert landed.is_file(), "the runner did not put the diagnostics in the leg directory"

    # REAL LEG-DERIVED VALUES, not an empty shell. The peripheral reaction this
    # payload exists to exercise is the one the boundary quarantined, so the
    # diagnostics have to be about THIS leg.
    document = json.loads(landed.read_text(encoding="utf-8"))
    assert isinstance(document, dict) and document, "diagnostics landed empty"
    assert document == json.loads(
        (outputs / COVERAGE_DIAGNOSTICS_FILENAME).read_text(encoding="utf-8")
    ), "the document changed crossing the seam"


def test_f175_the_returned_map_pin_is_untouched(tmp_path: Path) -> None:
    """The carry must not widen the RETURNED MAP. That pin still holds exactly.

    ``write_quarantine_artifacts`` returns four names and
    ``test_d088_coverage_diagnostics`` asserts the fifth is absent from that map.
    The fix works around that contract instead of breaking it.

    SCOPED PRECISELY, because the first draft of this test was named "the four
    pinned consumers are untouched" and that was an OVERCLAIM. One pin DID move:
    the driver's carried-artifact set went from four names to five, deliberately,
    in ``test_only_the_carried_artifacts_move_...``. What is untouched is the
    producer's returned map. Two different pins, and only one of them held.
    """

    _outputs, written = _produce(tmp_path)
    assert set(written) == set(FOUR)
    assert COVERAGE_DIAGNOSTICS_FILENAME not in written


def test_f175_a_hostile_or_absent_diagnostics_path_persists_nothing(tmp_path: Path) -> None:
    """The sibling resolution inherits the same two guards as the four-name loop.

    The carry derives its path from the COVERAGE REPORT's recorded path, so a
    session-state map naming a coverage report somewhere else must not cause an
    arbitrary sibling file to be persisted under a filename an auditor trusts --
    and a leg whose diagnostics are simply absent must persist nothing rather than
    crash. Absence stays a diagnosis.
    """

    # A directory holding a decoy coverage report and NO diagnostics beside it.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / COVERAGE_REPORT_FILENAME).write_text('{"decoy": true}', encoding="utf-8")
    app = _write_app(
        tmp_path,
        "f175_absent",
        _post_pipeline_body(
            _artifacts(
                "pathwhiz",
                quarantine_artifacts={
                    COVERAGE_REPORT_FILENAME: str(elsewhere / COVERAGE_REPORT_FILENAME)
                },
            ),
            pwml=_PWML,
        ),
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "pass", outcome.detail
    assert COVERAGE_DIAGNOSTICS_FILENAME not in outcome.artifacts


def test_f175_release_status_and_pwml_are_BYTE_IDENTICAL_across_the_change(
    tmp_path: Path,
) -> None:
    """THE AUTHORIZATION BOUNDARY, asserted rather than asserted-about.

    The narrow D-090 exception permits artifact PERSISTENCE and nothing else. So
    the leg's manifest row -- status, release record, codes, every field except
    the deliberately-moved artifact list -- and the exported PWML bytes must be
    exactly what they were. ``_row`` drops ``seconds`` (wall clock is not
    behaviour) and ``files`` (the enumerated move); everything else must match a
    leg run with NO quarantine artifacts published at all, which is the shape the
    driver saw before this carry existed.
    """

    _outputs, written = _produce(tmp_path)

    with_diagnostics = _run(
        _write_app(
            tmp_path,
            "f175_with",
            _post_pipeline_body(
                _artifacts("pathwhiz", quarantine_artifacts=written), pwml=_PWML
            ),
        ),
        STRICT,
    )
    without_any = _run(
        _write_app(
            tmp_path,
            "f175_without",
            _post_pipeline_body(_artifacts("pathwhiz"), pwml=_PWML),
        ),
        STRICT,
    )

    assert _row(with_diagnostics) == _row(without_any), (
        "publishing the diagnostics moved something other than the artifact set"
    )
    # PWML bytes, compared directly. This leg is deterministic -- no model stage
    # runs -- so byte equality is available here and is the strongest form.
    assert with_diagnostics.artifacts["pathway.pwml"] == without_any.artifacts["pathway.pwml"]
    # ...and the release record specifically, because that is the field the
    # authorization names first.
    assert with_diagnostics.to_dict().get("release") == without_any.to_dict().get("release")


def test_f175_an_offline_evaluator_can_consume_what_landed(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. The artifact is not evidence until a reader can resolve it.

    Mirrors how ``bench/acceptance.py`` finds the quarantine set: a filename
    lookup against the leg directory on disk. A file that lands under a name no
    consumer looks for would satisfy every assertion above and still be useless.
    """

    _outputs, written = _produce(tmp_path)
    app = _write_app(
        tmp_path,
        "f175_consumable",
        _post_pipeline_body(_artifacts("pathwhiz", quarantine_artifacts=written), pwml=_PWML),
    )
    outcome = _run(app, STRICT)

    leg_dir = tmp_path / "run" / "papers" / "pmc1" / "strict"
    runner.write_artifacts(leg_dir, outcome.artifacts)

    found, source, path = acceptance._first_existing(
        leg_dir, (COVERAGE_DIAGNOSTICS_FILENAME,)
    )
    assert source == COVERAGE_DIAGNOSTICS_FILENAME, "an evaluator cannot find the diagnostics"
    assert Path(path) == leg_dir / COVERAGE_DIAGNOSTICS_FILENAME
    assert isinstance(found, dict) and found

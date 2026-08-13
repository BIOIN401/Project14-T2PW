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
    COVERAGE_REPORT_FILENAME,
    QUARANTINE_REPORT_FILENAME,
    REMOVED_ENTITY_REPORT_FILENAME,
    quarantine_and_close,
    write_quarantine_artifacts,
)
from test_batch_driver import (  # noqa: E402
    PAPER,
    _artifacts,
    _lit,
    _post_pipeline_body,
    _run,
    _write_app,
    real_streamlit,  # noqa: F401 -- autouse fixture, re-exported on purpose
)
from test_strict_quarantine import _add_reaction, _glutathione_payload  # noqa: E402


#: The four, in the order ``write_quarantine_artifacts`` writes them.
FOUR: Tuple[str, ...] = (
    QUARANTINE_REPORT_FILENAME,
    REMOVED_ENTITY_REPORT_FILENAME,
    CLOSURE_REPORT_FILENAME,
    COVERAGE_REPORT_FILENAME,
)

_PWML = {
    "ok": True,
    "_xml": "<pathway><name>glutathione</name></pathway>",
    "pwml_ir": {"pathway": {"name": "glutathione"}},
    "counts": {"reactions": 2},
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
def test_only_the_four_artifacts_move_when_a_leg_produces_quarantine_output(
    tmp_path: Path,
) -> None:
    """PRESERVATION. Same leg with and without the map: four files, nothing else.

    The manifest row is compared whole, minus ``seconds`` (wall clock) and ``files``
    (the artifact names and byte counts, which is the deliberate pinned-baseline
    move this card makes). Every other driver-observable field -- status, stage,
    failure_kind, message, detail, issue_codes, counts, warnings -- must be equal,
    and every artifact the leg already produced must be byte-identical.
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
    assert set(with_quarantine.artifacts) - set(plain.artifacts) == set(FOUR)
    assert {
        name: blob for name, blob in with_quarantine.artifacts.items() if name not in FOUR
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

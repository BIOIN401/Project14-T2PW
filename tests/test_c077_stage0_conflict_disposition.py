"""C-077 / F-095 / D-062: a Stage-0 scope conflict is CLASSIFIED, not left blank.

**G9 arm: CORRECTION.** Nothing here is a new capability. ``_finalize_gate_failure``
has attached a first-class ``ReleaseStatus`` to a stopped run since C-053, and
``_release_status_row``'s own docstring already states the rule -- the classification
travels into the manifest row so an offline reader never has to re-derive it. The
Stage-0 conflict path was the one terminal path that stopped a run without answering
the second question.

The pre-existing observable behaviour being corrected, measured on the six T-105
conflict legs committed in ``runs_verify/2026-08-22_2147``: every one of the six
manifest rows carries NO ``release_status`` key at all, while the same six rows carry
``stage1_payload.json`` and ``merged_payload.json`` on disk. With no classification on
the row, ``batch.report``'s only surviving reading of ``status="scope_conflict"`` is
``STATUS_INELIGIBLE``, whose definition says *"nothing was attempted, so nothing
failed"*. That is false on the evidence and is the drop D-062 reverses.

Every test drives ``run_one`` -- the production entry point -- against a fixture app
that speaks the real app's contract, and asserts on ROW AND ARTIFACT CONTENT. Every
symbol driven exists unchanged on the base SHA, so the base failures are behavioural,
never symbol absence.

WHERE THIS DEVIATES FROM THE C-077 CHARTER, AND WHY
---------------------------------------------------
The charter's section 3 requires the attached status to be ``REVIEW_REQUIRED``. It is
not constructible at this seam, and the charter asked for exactly this to be reported
rather than faked:

* ``PRODUCT_CONTRACT`` section 4 defines ``review_required`` as "**Valid, useful PWML
  produced**, but one or more important biological uncertainties are explicitly
  identified", and ``ReleaseStatus.produced_pwml`` encodes it -- ``diagnostic_only`` is
  documented as "the one state with no final PWML". This seam is reached before the
  audit, the DB mapping, the freeze and the export, so no PWML exists. Measured: not
  one of the six committed conflict legs has a ``.pwml`` file of any name.
* The only route to ``REVIEW_REQUIRED`` through ``classify_release_status`` runs
  through ``strict_gates_passed=True``. Those gates never ran here. Asserting they
  passed would fabricate a measurement.

So the honest classification is ``diagnostic_only`` with an explicit reason naming the
scope conflict -- the same shape, and the same "this is about SERIALIZATION, not about
the biology" distinction, that ``_finalize_gate_failure`` already ships. Delivering
D-062's literal ``review_required`` requires driving the run onward under an organism
known to be wrong; that is a product decision nobody has taken and is escalated, not
assumed here. ``test_the_classification_is_not_a_strict_success`` pins the property
that actually matters to D-062's second prohibition either way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.batch import driver  # noqa: E402
from t2pw.batch import report as batch_report  # noqa: E402
from t2pw.batch.driver import (  # noqa: E402
    RESEARCH,
    SCOPE_CONFLICT_NAME,
    STRICT,
    RunOutcome,
)
from t2pw.pipeline.release_status import DIAGNOSTIC_ONLY, RELEASE_READY  # noqa: E402
from t2pw.rag.eligibility import thresholds_from_config  # noqa: E402
from test_batch_driver import (  # noqa: E402
    _artifacts,
    _lit,
    _Paper,
    _post_pipeline_body,
    _run,
    _write_app,
    real_streamlit,  # noqa: F401 -- autouse fixture, re-exported on purpose
)

#: The committed evidence F-095 / D-062 were registered from. Six legs, three
#: papers x two modes, every one of them a real Stage-0 ORGANISM conflict.
RUN_DIR = ROOT / "runs_verify" / "2026-08-22_2147"
CONFLICT_PAPERS = ("PMC12657337", "PMC12421875", "PMC12312563")

#: The exact T-105 shapes for ``PMC12421875``, lifted from that leg's committed
#: ``scope_conflict.json`` and asserted against it in
#: ``test_the_fixture_reproduces_the_committed_t105_shapes`` -- so a fixture that
#: drifts away from what production actually produced fails loudly instead of
#: quietly testing something else.
T105_PAPER_ID = "PMC12421875"
T105_REQUESTED_PATHWAY = "menaquinone biosynthesis"
T105_REQUESTED_ORGANISM = "Bacillus subtilis"
T105_OBSERVED_ORGANISM = "Lactococcus lactis"
T105_STAGE0_CONTEXT = {
    "pathway_name": T105_REQUESTED_PATHWAY,
    "likely_organism": T105_OBSERVED_ORGANISM,
    "document_type": "primary_research",
}


# ---------------------------------------------------------------------------
# Fixtures: the real driver, driven through the real app boundary.
# ---------------------------------------------------------------------------
def _stage0_stanza(context: object) -> str:
    """Make the fixture app store a Stage-0 context, as the real app does."""

    return f'    st.session_state["pathway_context"] = {_lit(context)}\n'


def _scoped_paper(pathway: str, organism: str, paper_id: str) -> _Paper:
    paper = _Paper(paper_id=paper_id, topic=pathway, organism=organism)
    paper.requested_pathway = pathway
    paper.requested_organism = organism
    return paper


def _t105_leg(tmp_path: Path, mode: str = STRICT, name: str = "c077_t105") -> RunOutcome:
    """One leg reproducing the real ``PMC12421875`` organism trap.

    The batch asks for menaquinone biosynthesis in *Bacillus subtilis*; Stage 0
    reads the same pathway in *Lactococcus lactis*. The pathway agrees, so the
    ONLY conflict is the organism -- which is what makes it the D-062 case rather
    than a general scope mismatch.
    """

    app = _write_app(
        tmp_path,
        name,
        _post_pipeline_body(
            _artifacts("research"), extra=_stage0_stanza(T105_STAGE0_CONTEXT)
        ),
    )
    paper = _scoped_paper(T105_REQUESTED_PATHWAY, T105_REQUESTED_ORGANISM, T105_PAPER_ID)
    return _run(app, mode, paper)


def _committed_conflict_rows() -> list:
    """The six committed T-105 ``scope_conflict`` manifest rows."""

    rows = []
    for line in (RUN_DIR / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") == "scope_conflict":
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 1. The correction itself. G9 arm.
# ---------------------------------------------------------------------------
def test_the_committed_conflict_legs_carry_no_classification_at_all() -> None:
    """The measured defect, read off the artifacts D-062 was registered from.

    Not a base-SHA proof (it reads committed bytes, which do not move with the
    code) but the statement of WHAT is wrong: six legs, six blank
    classifications, and an extracted payload on disk beside every one of them.
    """

    rows = _committed_conflict_rows()
    assert len(rows) == 6, [r.get("paper_id") for r in rows]
    assert sorted({r["paper_id"] for r in rows}) == sorted(CONFLICT_PAPERS)
    for row in rows:
        assert "release_status" not in row, row["paper_id"]
        names = {entry["name"].rsplit("/", 1)[-1] for entry in row.get("files", [])}
        # Something WAS attempted, and it got far enough to write both payloads.
        assert "stage1_payload.json" in names, row["paper_id"]
        assert "merged_payload.json" in names, row["paper_id"]
        # And nothing was ever serialized: no PWML of any name.
        assert not [n for n in names if n.endswith(".pwml")], row["paper_id"]


def test_a_stage0_organism_conflict_row_carries_a_release_classification(
    tmp_path: Path,
) -> None:
    """G9: on base ``129d9b2`` this row has no ``release_status`` key at all.

    The base failure is behavioural: ``run_one`` exists unchanged on base, the
    fixture app is unchanged, and the assertion is on the CONTENT of the manifest
    row the production driver produces.
    """

    outcome = _t105_leg(tmp_path)
    row = outcome.to_dict()

    assert row["status"] == "scope_conflict"
    assert "release_status" in row, "the run was stopped but never classified"
    release = row["release_status"]
    # The pipeline DID execute. This is the machine-readable refutation of
    # ``STATUS_INELIGIBLE``'s "nothing was attempted, so nothing failed".
    assert release["pipeline_executed"] is True
    # See the module docstring: review_required is not constructible at a seam
    # that reaches no exporter. This is the honest classification.
    assert release["status"] == DIAGNOSTIC_ONLY
    assert release["status"] != RELEASE_READY
    assert driver.REASON_STAGE0_SCOPE_CONFLICT in release["reasons"]
    # A real classify_release_status product, not a hand-built dict: it carries
    # the classifier's whole record, including fields this seam never sets.
    for key in (
        "strict_gates_passed",
        "semantic_evaluation",
        "coverage_evaluated",
        "missing_anchors",
    ):
        assert key in release, key
    assert outcome.release_status.to_dict() == release


def test_the_classification_survives_the_batch_report_reader(tmp_path: Path) -> None:
    """The row is readable by the consumer that used to have nothing to read."""

    outcome = _t105_leg(tmp_path)
    run = batch_report._to_run(outcome.to_dict())

    assert run.release_status is not None
    assert run.release_status["status"] == DIAGNOSTIC_ONLY
    # Unchanged, deliberately: the leg is still not a failure. Fixing the
    # STATUS_INELIGIBLE fold would need a new report-level status vocabulary
    # member, which is a widening this card did not take -- see the report.
    assert run.failed is False


# ---------------------------------------------------------------------------
# 2. Strict export stays forbidden. D-062's second prohibition.
# ---------------------------------------------------------------------------
def test_the_classification_is_not_a_strict_success(tmp_path: Path) -> None:
    """No PWML of any name, and never a strict-benchmark numerator."""

    for mode in (STRICT, RESEARCH):
        outcome = _t105_leg(tmp_path, mode=mode, name=f"c077_strict_{mode[:4].lower()}")
        row = outcome.to_dict()

        assert driver.PWML_RELEASE_READY_NAME not in outcome.artifacts
        assert driver.PWML_REVIEW_REQUIRED_NAME not in outcome.artifacts
        assert not [n for n in outcome.artifacts if str(n).endswith(".pwml")]
        assert outcome.pwml_artifact == ""
        assert "pwml_artifact" not in row

        assert row["release_status"]["strict_acceptance_eligible"] is False
        assert outcome.release_status.strict_acceptance_eligible is False
        assert outcome.release_status.produced_pwml is False
        assert outcome.status != "pass"


# ---------------------------------------------------------------------------
# 3. The extracted payload survives.
# ---------------------------------------------------------------------------
def test_the_extracted_payload_is_preserved_not_discarded(tmp_path: Path) -> None:
    """Merge rule 7: an incomplete-but-correct pathway is preserved, not dropped."""

    outcome = _t105_leg(tmp_path)

    stage1 = json.loads(outcome.artifacts["stage1_payload.json"])
    merged = json.loads(outcome.artifacts["merged_payload.json"])
    assert stage1["processes"]["reactions"], stage1
    assert merged["processes"]["reactions"], merged
    # The reaction the fixture extracted is still there, with its chemistry.
    reaction = merged["processes"]["reactions"][0]
    assert reaction["inputs"] and reaction["outputs"]

    # And it reaches the manifest row's file list, which is what an offline
    # reader consults.
    names = {entry["name"] for entry in outcome.to_dict()["files"]}
    assert {"stage1_payload.json", "merged_payload.json", SCOPE_CONFLICT_NAME} <= names


# ---------------------------------------------------------------------------
# 4. Both organisms, distinguishably.
# ---------------------------------------------------------------------------
def test_the_fixture_reproduces_the_committed_t105_shapes() -> None:
    """The constants above are the real leg's, not invented ones."""

    committed = json.loads(
        (RUN_DIR / "papers" / T105_PAPER_ID / "strict" / SCOPE_CONFLICT_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert committed["requested"]["requested_pathway"] == T105_REQUESTED_PATHWAY
    assert committed["requested"]["requested_organism"] == T105_REQUESTED_ORGANISM
    assert committed["observed"]["observed_organisms"] == [T105_OBSERVED_ORGANISM]
    assert committed["stage0_context"] == T105_STAGE0_CONTEXT


def test_observed_and_requested_organisms_both_survive_in_separate_fields(
    tmp_path: Path,
) -> None:
    """Neither organism overwrites the other, anywhere on the record.

    On base the requested ORGANISM reaches the manifest row nowhere: ``topic``
    carries the requested pathway alone and ``scope_conflicts`` carries the
    organism only inside a prose sentence. This is the second behavioural base
    failure in this file.
    """

    outcome = _t105_leg(tmp_path)
    row = outcome.to_dict()

    # Observed: what Stage 0 read.
    assert row["observed_context"]["observed_organisms"] == [T105_OBSERVED_ORGANISM]
    # Requested: what the batch asked for. Its own field on the row.
    assert row["requested_scope"]["requested_organism"] == T105_REQUESTED_ORGANISM
    assert row["requested_scope"]["requested_pathway"] == T105_REQUESTED_PATHWAY
    assert row["requested_scope"]["pinned"] is False

    # Neither overwrote the other, in either direction.
    assert T105_OBSERVED_ORGANISM not in json.dumps(row["requested_scope"])
    assert T105_REQUESTED_ORGANISM not in json.dumps(row["observed_context"])
    # The request that came back is the request that went in.
    assert outcome.warnings == []

    # And the on-disk artifact still says the same thing, unchanged.
    artifact = json.loads(outcome.artifacts[SCOPE_CONFLICT_NAME])
    assert artifact["requested"]["requested_organism"] == T105_REQUESTED_ORGANISM
    assert artifact["observed"]["observed_organisms"] == [T105_OBSERVED_ORGANISM]


# ---------------------------------------------------------------------------
# 5. The guard is NOT weakened. The non-vacuity arm.
# ---------------------------------------------------------------------------
def test_the_stage0_guard_still_detects_and_still_stops_the_run(tmp_path: Path) -> None:
    """A patch that made the guard inert would pass everything above but not this.

    Three separate things are asserted, because "the conflict was detected" and
    "the run stopped" and "the run stopped BEFORE the audit" are three claims and
    only the third rules out the shape D-062 forbids.
    """

    # The abort default is untouched. Flipping it is what "weakening the guard"
    # would mean, and D-062 forbids it outright.
    assert thresholds_from_config().stage0_conflict_aborts is True

    outcome = _t105_leg(tmp_path)

    # Detected.
    assert outcome.scope_conflicts, "the conflict was not detected at all"
    assert T105_OBSERVED_ORGANISM in outcome.scope_conflicts[0]
    assert T105_REQUESTED_ORGANISM in outcome.scope_conflicts[0]
    assert outcome.counts["scope_conflicts"] == 1
    assert "scope_conflict" in outcome.issue_codes
    assert SCOPE_CONFLICT_NAME in outcome.artifacts

    # Stopped, with the terminal status and no failure kind.
    assert outcome.status == "scope_conflict"
    assert outcome.scope_conflicted is True
    assert outcome.failure_kind == ""

    # Stopped BEFORE the audit and the DB mapping. The fixture app only publishes
    # ``post_pipeline_artifacts`` when the "Run audit and DB mapping" button is
    # clicked, so the absence of every one of these IS the proof the driver never
    # clicked it.
    assert outcome.stage == "stage1"
    for name in (
        "final_mapped.json",
        "final_mapped_db.json",
        "mapping_report.json",
        "pwml_ir.json",
    ):
        assert name not in outcome.artifacts, name


def test_a_pathway_only_conflict_is_still_stopped_and_still_classified(
    tmp_path: Path,
) -> None:
    """The detection is unchanged for the non-organism conflicts too."""

    app = _write_app(
        tmp_path,
        "c077_pathway_conflict",
        _post_pipeline_body(
            _artifacts("research"),
            extra=_stage0_stanza(
                {"pathway_name": "cholesterol biosynthesis", "likely_organism": "Homo sapiens"}
            ),
        ),
    )
    outcome = _run(
        app, RESEARCH, _scoped_paper("lipid A biosynthesis", "Escherichia coli", "PMC1")
    )

    assert outcome.status == "scope_conflict"
    assert outcome.counts["scope_conflicts"] == 2
    assert outcome.to_dict()["release_status"]["status"] == DIAGNOSTIC_ONLY
    assert outcome.to_dict()["requested_scope"]["requested_organism"] == "Escherichia coli"


# ---------------------------------------------------------------------------
# 6. No collateral on the non-conflict paths.
# ---------------------------------------------------------------------------
def test_an_agreeing_stage0_reading_gains_no_new_row_keys(tmp_path: Path) -> None:
    """A leg with no conflict writes exactly the row it wrote before."""

    app = _write_app(
        tmp_path,
        "c077_agreeing",
        _post_pipeline_body(
            _artifacts("research"),
            extra=_stage0_stanza(
                {"pathway_name": "lipid A biosynthesis", "likely_organism": "Escherichia coli"}
            ),
        ),
    )
    outcome = _run(
        app, RESEARCH, _scoped_paper("lipid A biosynthesis", "Escherichia coli", "PMC1")
    )
    row = outcome.to_dict()

    # Asserted on the ROW, not on the attribute: this arm exists to prove NO
    # collateral, so it must be green on the base SHA too. An attribute check
    # would fail there for symbol absence and prove nothing about behaviour.
    assert outcome.status != "scope_conflict"
    assert "requested_scope" not in row
    assert "release_status" not in row
    assert "scope_conflicts" not in row
    assert "observed_context" in row  # the observation is still recorded


def test_a_pinned_paper_is_untouched(tmp_path: Path) -> None:
    """No request to contradict, so nothing is reconciled and nothing is added."""

    app = _write_app(
        tmp_path,
        "c077_pinned",
        _post_pipeline_body(
            _artifacts("research"), extra=_stage0_stanza(T105_STAGE0_CONTEXT)
        ),
    )
    outcome = _run(app, RESEARCH, _Paper(paper_id="PMC1"))
    row = outcome.to_dict()

    assert outcome.status != "scope_conflict"
    assert "requested_scope" not in row


def test_the_default_outcome_row_is_unchanged() -> None:
    """The new field is empty by default, so it adds no key to any other row."""

    row = RunOutcome().to_dict()
    assert "requested_scope" not in row
    assert "release_status" not in row
    assert set(row) == {
        "paper_id",
        "mode",
        "status",
        "stage",
        "seconds",
        "failure_kind",
        "message",
        "detail",
        "issue_codes",
        "counts",
        "files",
        "warnings",
    }


def test_the_annotate_only_conflict_path_is_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    """``stage0_conflict_aborts=false`` still annotates and carries on, unclassified.

    The classification is attached on the ABORT branch only. A run that carries
    on reaches the exporter and gets the boundary's own frozen record, so
    attaching one here would be the merge-rule-8 defect of classifying twice.
    """

    monkeypatch.setenv("RAG_ELIGIBILITY_STAGE0_CONFLICT_ABORTS", "false")
    assert thresholds_from_config().stage0_conflict_aborts is False
    outcome = _t105_leg(tmp_path, name="c077_annotate_only")

    row = outcome.to_dict()
    assert outcome.status != "scope_conflict"
    assert any(w.startswith(driver.WARN_SCOPE_CONFLICT_PREFIX) for w in outcome.warnings)
    # Row-level, so this arm is green on base as well -- see the note above.
    assert "requested_scope" not in row
    assert "stage0_scope_conflict" not in json.dumps(row)


# ---------------------------------------------------------------------------
# 7. The gold plan is untouched. D-062 requires this to keep holding.
# ---------------------------------------------------------------------------
def test_verify_plan_still_returns_ok_with_ten_pinned_overrides() -> None:
    """No topics file was edited, so the staged acceptance plan still verifies."""

    from t2pw.bench.goldset import load_gold_set
    from t2pw.bench.preflight import verify_plan

    report = verify_plan(RUN_DIR, load_gold_set(None))
    rendered = report.render()

    assert report.ok is True, rendered
    assert rendered.count("[pinned_override]") == 10, rendered

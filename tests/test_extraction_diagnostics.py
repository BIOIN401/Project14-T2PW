"""The diagnostics contract: five failure categories, bounded, always on disk.

WHY THIS FILE EXISTS. Before 2026-07-29 a Stage-0/Stage-1 death produced one
sentence and nothing else. Both legs lost in ``runs/2026-07-28_0919`` carry
``files: []`` in the manifest, so the postmortem in ``t2pw/llm/client.py`` had to
*infer* the cause from the shape of the failure rather than read it. These tests
pin the three properties that make that impossible to repeat:

  1. the five failure categories are actually distinguished, and the EARLIEST
     cause wins rather than the last symptom (:func:`classify_outcome`);
  2. what cleaning discards is reported with the rule that discarded it, so
     "the model produced nothing" stops being confusable with "we deleted what it
     produced";
  3. the artifacts exist when there is no final payload, and nothing in them is
     unbounded -- a diagnostic that embeds a 4.7 MB payload's evidence fields is
     one nobody opens and one that costs real money to write on every attempt.

Everything here is offline: no provider, no network, no streamlit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.extraction_diagnostics import (  # noqa: E402
    BOUNDARY_REPORT_NAME,
    BOUNDARY_STAGE1_EXTRACTION,
    CLEANING_REPORT_NAME,
    MAPPED_FAILURE_SNAPSHOT_NAME,
    MAX_ATTEMPTS_KEPT,
    MAX_SAMPLE,
    OUTCOME_CONTRACT_FAILED,
    OUTCOME_DISCARDED_BY_CLEANING,
    OUTCOME_EMPTY_COMPLETION,
    OUTCOME_INVALID_JSON,
    OUTCOME_OK,
    OUTCOME_ZERO_PROCESSES,
    PREVIEW_CHARS,
    STAGE0_ATTEMPTS_NAME,
    DiscardLedger,
    ExtractionDiagnostics,
    activate,
    classify_outcome,
    count_entities,
    count_processes,
    current,
    deactivate,
    payload_hash,
)
from t2pw.pipeline.pipeline import clean_stage_one, clean_stage_one_with_report  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_recorder():
    """Every test starts with no ambient recorder and leaves none behind.

    The recorder is module-level by design (the facts are produced deep in the
    call stack and consumed at the top), so leaking one between tests would let
    one test's boundaries appear in another's artifacts -- exactly the cross-paper
    contamination ``activate`` exists to prevent in the batch.
    """

    deactivate()
    yield
    deactivate()


# ---------------------------------------------------------------------------
# 1. The five categories, and which one wins.
# ---------------------------------------------------------------------------
def test_the_five_failure_categories_are_distinct() -> None:
    """Each of the five faults maps to its own outcome, not to a shared one."""

    assert classify_outcome(empty_completion=True) == OUTCOME_EMPTY_COMPLETION
    assert classify_outcome(parsed=False) == OUTCOME_INVALID_JSON
    assert classify_outcome(raw_process_count=0) == OUTCOME_ZERO_PROCESSES
    assert (
        classify_outcome(raw_process_count=3, cleaned_process_count=0)
        == OUTCOME_DISCARDED_BY_CLEANING
    )
    assert (
        classify_outcome(raw_process_count=3, cleaned_process_count=3, contract_ok=False)
        == OUTCOME_CONTRACT_FAILED
    )
    assert classify_outcome(raw_process_count=3, cleaned_process_count=3, contract_ok=True) == (
        OUTCOME_OK
    )


def test_the_earliest_cause_wins_not_the_last_symptom() -> None:
    """An empty reply also has zero processes and also fails the contract.

    Reporting that as ``contract_failed_after_cleaning`` would name the symptom
    and bury the cause, which is the exact confusion the taxonomy exists to end.
    """

    assert (
        classify_outcome(
            empty_completion=True,
            parsed=False,
            raw_process_count=0,
            cleaned_process_count=0,
            contract_ok=False,
        )
        == OUTCOME_EMPTY_COMPLETION
    )
    # And one step down: unparseable outranks every later stage too.
    assert (
        classify_outcome(parsed=False, raw_process_count=0, contract_ok=False)
        == OUTCOME_INVALID_JSON
    )


def test_unobserved_facts_never_decide_an_outcome() -> None:
    """``None`` means "not measured yet" and must not be read as zero.

    A boundary is recorded before cleaning has run -- it has to be, because the
    stage can raise in between -- so a not-yet-known cleaned count must leave the
    verdict alone rather than reporting everything as discarded.
    """

    assert classify_outcome(raw_process_count=4, cleaned_process_count=None) == OUTCOME_OK
    assert classify_outcome(raw_process_count=None, cleaned_process_count=None) == OUTCOME_OK


def test_a_boundary_is_reclassified_when_cleaning_reports_back() -> None:
    """Rows in, none out, on a record that was already written as ``ok``."""

    diagnostics = ExtractionDiagnostics()
    record = diagnostics.record_boundary(
        stage="extraction",
        boundary=BOUNDARY_STAGE1_EXTRACTION,
        outcome=OUTCOME_OK,
        raw_process_counts={"reactions": 3, "total": 3},
    )
    assert record["outcome"] == OUTCOME_OK

    updated = diagnostics.attach_cleaning_to_boundary(
        record, cleaned_process_counts={"total": 0}
    )

    assert updated["outcome"] == OUTCOME_DISCARDED_BY_CLEANING
    # Same record, updated in place -- not a duplicate row in the report.
    assert len(diagnostics.boundaries) == 1


# ---------------------------------------------------------------------------
# 2. Discards are reported with reasons.
# ---------------------------------------------------------------------------
def test_valid_reactions_discarded_by_cleaning_are_reported_with_reasons() -> None:
    """THE headline case: three good-looking reactions, zero survivors, named rules.

    Every one of these rows is dropped by a bare ``continue`` in
    ``_clean_processes``. Downstream, the resulting payload is byte-identical to
    one where the model returned nothing at all: both arrive as "Payload must
    include a processes object". The report is the only thing that says which.
    """

    payload = {
        "entities": {"compounds": [{"name": "ATP"}, {"name": "ADP"}]},
        "processes": {
            "reactions": [
                # Outputs are present as a key but every entry is blank.
                {"name": "r1", "inputs": ["ATP"], "outputs": ["   "], "evidence": "e1"},
                # Participants arrived as objects where the schema wants strings.
                {"name": "r2", "inputs": [{"name": "ATP"}], "outputs": [{"name": "ADP"}]},
                # Inputs missing entirely.
                {"name": "r3", "outputs": ["ADP"]},
            ]
        },
    }

    cleaned, report = clean_stage_one_with_report(payload, label="unit")

    assert cleaned.get("processes", {}).get("reactions") is None
    assert report["raw_process_counts"]["reactions"] == 3
    assert report["cleaned_process_counts"]["total"] == 0
    assert report["all_processes_discarded"] is True

    reasons = report["discarded_by_reason"]
    assert reasons["reaction_no_usable_outputs"] == 1
    # r2 loses both sides (objects survive neither) and r3 has no inputs at all.
    assert reasons["reaction_no_usable_participants"] == 1
    assert reasons["reaction_no_usable_inputs"] == 1
    assert report["discarded_total"] == 3

    # The sample names the rows so a reviewer can find them in the payload.
    sampled = [
        entry.get("name")
        for rows in report["discarded_sample"].values()
        for entry in rows
    ]
    assert {"r1", "r2", "r3"} <= set(sampled)


def test_an_empty_extraction_is_not_reported_as_a_cleaning_discard() -> None:
    """The other side of the same coin: nothing in, nothing out, no discards.

    Without this the flag would be useless -- it has to separate the two cases,
    not fire on both.
    """

    _cleaned, report = clean_stage_one_with_report({"entities": {}, "processes": {}})

    assert report["raw_process_counts"]["total"] == 0
    assert report["all_processes_discarded"] is False
    assert report["discarded_total"] == 0


def test_an_enzyme_dropped_by_the_actor_whitelist_is_reported() -> None:
    """A reaction that ships without its catalyst, and why.

    ``_clean_enzymes`` accepts ``protein``, ``protein_complex`` or a typed
    ``entity``. A model that writes the obvious ``{"name": ...}`` has its enzyme
    deleted -- the reaction survives, uncatalysed, and nothing said so. This is a
    cleaning problem with a one-line fix, and it used to look identical to the
    model never naming an enzyme.
    """

    payload = {
        "entities": {"compounds": [{"name": "ATP"}]},
        "processes": {
            "reactions": [
                {
                    "name": "hydrolysis",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "enzymes": [
                        {"name": "ATPase"},
                        {"protein": "kept", "evidence": "e"},
                        {"entity": "Mg2+", "entity_type": "cofactor"},
                    ],
                }
            ]
        },
    }

    cleaned, report = clean_stage_one_with_report(payload)

    # The reaction itself survives -- only the unusable actor rows are gone.
    assert cleaned["processes"]["reactions"][0]["enzymes"] == [
        {"protein": "kept", "evidence": "e"}
    ]
    reasons = report["discarded_by_reason"]
    assert reasons["actor_missing_protein_reference"] == 1
    assert reasons["actor_entity_type_is_not_an_actor"] == 1
    sample = report["discarded_sample"]["actor_missing_protein_reference"][0]
    assert sample["name"] == "ATPase"
    assert sample["pointer"] == "/processes/reactions/0/enzymes/0"


def test_entity_rows_lost_to_an_unrecognised_bucket_are_counted() -> None:
    """A whole bucket outside the whitelist is dropped silently, and always was.

    It is the largest invisible loss in ``_clean_entities``: nothing downstream
    can tell that four enzymes arrived under a key cleaning does not read.
    """

    payload = {
        "entities": {
            "enzymes": [{"name": "LpxA"}, {"name": "LpxD"}],
            "compounds": [{"name": "ATP"}],
        },
        "processes": {},
    }

    _cleaned, report = clean_stage_one_with_report(payload)

    assert report["discarded_by_reason"]["entity_bucket_not_recognized"] == 2
    assert "LpxA" in json.dumps(report["discarded_sample"])


def test_nameless_and_duplicate_entity_rows_get_their_own_reasons() -> None:
    payload = {
        "entities": {
            "compounds": [
                {"name": "ATP"},
                {"name": ""},
                {"name": "atp"},
                "not-an-object",
            ]
        },
        "processes": {},
    }

    _cleaned, report = clean_stage_one_with_report(payload)

    reasons = report["discarded_by_reason"]
    assert reasons["entity_missing_name"] == 1
    assert reasons["entity_duplicate_name"] == 1
    assert reasons["entity_not_an_object"] == 1


def test_clean_stage_one_behaviour_is_unchanged_by_the_ledger() -> None:
    """The reporting variant must be a pure observer of the historical function.

    ``clean_stage_one`` has dozens of call sites; if adding a ledger changed what
    it returns, this whole change would be a silent behavioural rewrite of the
    pipeline rather than an addition to it.
    """

    payload = {
        "entities": {"compounds": [{"name": "ATP"}, {"name": ""}]},
        "processes": {
            "reactions": [
                {"name": "keep", "inputs": ["ATP"], "outputs": ["ADP"]},
                {"name": "drop", "inputs": ["ATP"], "outputs": []},
            ],
            "interactions": [{"entity_1": "A", "entity_2": ""}],
        },
    }

    plain = clean_stage_one(payload)
    reporting, report = clean_stage_one_with_report(payload)

    assert plain == reporting
    assert report["discarded_total"] == 3


# ---------------------------------------------------------------------------
# 3. Artifacts exist when there is no final payload.
# ---------------------------------------------------------------------------
def test_failure_artifacts_exist_when_no_final_payload_exists(tmp_path: Path) -> None:
    """The 2026-07-28 property, inverted.

    Those legs wrote ``files: []`` and only ``RESULT.txt``. Here nothing produces
    a payload at all -- the Stage-0 draw was empty, the Stage-1 reply did not
    parse -- and the diagnostics are on disk anyway, because every ``record_*``
    call flushes as it is made rather than at the end of a run that never comes.
    """

    diagnostics = activate(run_id="replay", artifact_dir=tmp_path)
    diagnostics.record_stage0_attempt(
        {"boundary": "stage0_preprocess", "status": "empty_reply", "attempts": 3}
    )
    diagnostics.record_boundary(
        stage="extraction",
        boundary=BOUNDARY_STAGE1_EXTRACTION,
        outcome=OUTCOME_INVALID_JSON,
        model="test-model",
        finish_reason="length",
        attempts=2,
        response_status="ok",
        error="JSONDecodeError: Unterminated string",
    )
    diagnostics.record_cleaning({"label": "stage_one_merged", "all_processes_discarded": False})

    written = sorted(p.name for p in tmp_path.iterdir())
    assert STAGE0_ATTEMPTS_NAME in written
    assert BOUNDARY_REPORT_NAME in written
    assert CLEANING_REPORT_NAME in written
    # Mapping never ran, so its snapshot is absent -- and that absence is itself
    # the diagnosis. Writing an empty file here would destroy that signal.
    assert MAPPED_FAILURE_SNAPSHOT_NAME not in written

    report = json.loads((tmp_path / BOUNDARY_REPORT_NAME).read_text(encoding="utf-8"))
    boundary = report["boundaries"][0]
    for field in ("model", "finish_reason", "attempts", "response_status", "stage", "boundary"):
        assert field in boundary, field
    assert report["outcomes"] == [OUTCOME_INVALID_JSON]


def test_the_mapped_snapshot_appears_only_once_mapping_has_run(tmp_path: Path) -> None:
    diagnostics = activate(artifact_dir=tmp_path)
    assert not (tmp_path / MAPPED_FAILURE_SNAPSHOT_NAME).exists()

    diagnostics.record_mapped_failure({"stage": "post_mapping", "reason": "species_required"})

    snapshot = json.loads(
        (tmp_path / MAPPED_FAILURE_SNAPSHOT_NAME).read_text(encoding="utf-8")
    )
    assert snapshot["stage"] == "post_mapping"


def test_a_write_failure_is_reported_and_never_raises(tmp_path: Path) -> None:
    """A diagnostics writer that can kill the run it is diagnosing is worse than
    none. A directory where the file belongs makes the write fail; the recorder
    keeps the finding and carries on."""

    (tmp_path / BOUNDARY_REPORT_NAME).mkdir()
    diagnostics = activate(artifact_dir=tmp_path)

    diagnostics.record_boundary(stage="extraction", boundary=BOUNDARY_STAGE1_EXTRACTION)

    assert diagnostics.write_errors
    assert BOUNDARY_REPORT_NAME in diagnostics.write_errors[0]


def test_current_always_returns_a_recorder_so_recording_is_never_conditional() -> None:
    """Call sites must not have to branch on "is there a recorder".

    That branch is how recording ends up absent from exactly the failure paths
    that need it.
    """

    deactivate()
    recorder = current()
    assert isinstance(recorder, ExtractionDiagnostics)
    assert recorder.artifact_dir is None  # detached: collects, writes nothing
    recorder.record_boundary(stage="extraction", boundary=BOUNDARY_STAGE1_EXTRACTION)
    assert len(recorder.boundaries) == 1


def test_activate_replaces_rather_than_appends(tmp_path: Path) -> None:
    """One paper's diagnostics must never appear in the next paper's artifacts."""

    first = activate(run_id="paper-a", artifact_dir=tmp_path)
    first.record_boundary(stage="extraction", boundary=BOUNDARY_STAGE1_EXTRACTION)

    second = activate(run_id="paper-b", artifact_dir=tmp_path)

    assert second is not first
    assert second.boundaries == []
    assert second.run_id == "paper-b"


# ---------------------------------------------------------------------------
# 4. Bounds.
# ---------------------------------------------------------------------------
_HUGE_EVIDENCE = "Lipid A biosynthesis proceeds via LpxA. " * 4000


def test_diagnostic_evidence_is_bounded(tmp_path: Path) -> None:
    """No diagnostic artifact may carry a repeated evidence blob.

    Payload ``evidence`` fields reach six figures of characters and these files
    are rewritten on every attempt. The bound is asserted on the FILE, not on an
    in-memory object, because the file is what costs disk and what a reviewer
    opens.
    """

    diagnostics = activate(artifact_dir=tmp_path)
    diagnostics.record_boundary(
        stage="extraction",
        boundary=BOUNDARY_STAGE1_EXTRACTION,
        outcome=OUTCOME_INVALID_JSON,
        raw_preview=_HUGE_EVIDENCE,
        error=_HUGE_EVIDENCE,
        raw_chars=len(_HUGE_EVIDENCE),
        attempt_log=[{"attempt": 1, "raw": _HUGE_EVIDENCE, "status": "ok"}],
    )

    text = (tmp_path / BOUNDARY_REPORT_NAME).read_text(encoding="utf-8")
    assert len(_HUGE_EVIDENCE) > 150_000
    assert len(text) < 4_000
    # The size is still reported, so nothing is hidden -- only elided.
    boundary = json.loads(text)["boundaries"][0]
    assert boundary["raw_chars"] == len(_HUGE_EVIDENCE)
    assert len(boundary["raw_preview"]) <= PREVIEW_CHARS + 40
    assert "raw" not in boundary["attempt_log"][0]
    assert boundary["attempt_log"][0]["raw_chars"] == len(_HUGE_EVIDENCE)
    assert boundary["attempt_log"][0]["raw_hash"].startswith("sha256:")


def test_a_discard_sample_is_capped_but_the_count_is_not() -> None:
    """The count is the finding; the sample only has to make the shape visible."""

    ledger = DiscardLedger()
    for index in range(500):
        ledger.record(
            reason="reaction_no_usable_outputs",
            pointer=f"/processes/reactions/{index}",
            name=f"reaction {index}",
        )

    report = ledger.to_dict()
    assert report["discarded_by_reason"]["reaction_no_usable_outputs"] == 500
    assert len(report["discarded_sample"]["reaction_no_usable_outputs"]) == MAX_SAMPLE
    assert report["sample_cap_per_reason"] == MAX_SAMPLE


def test_a_runaway_attempt_loop_cannot_grow_the_artifact_without_bound(
    tmp_path: Path,
) -> None:
    diagnostics = activate(artifact_dir=tmp_path)
    for index in range(200):
        diagnostics.record_stage0_attempt({"attempt": index, "status": "empty_reply"})

    stored = json.loads((tmp_path / STAGE0_ATTEMPTS_NAME).read_text(encoding="utf-8"))
    assert stored["attempt_count"] == 200  # the truth is reported
    assert len(stored["attempts"]) == MAX_ATTEMPTS_KEPT  # the file stays small


def test_a_long_sample_string_is_clipped_not_stored_whole() -> None:
    ledger = DiscardLedger()
    ledger.record(reason="reaction_no_usable_outputs", name=_HUGE_EVIDENCE)

    entry = ledger.to_dict()["discarded_sample"]["reaction_no_usable_outputs"][0]
    assert len(entry["name"]) < 200
    assert "elided" in entry["name"]


# ---------------------------------------------------------------------------
# 5. The small primitives everything above leans on.
# ---------------------------------------------------------------------------
def test_payload_hash_is_stable_across_key_order_and_detects_change() -> None:
    """Two attempts returning identical content must produce identical hashes.

    That equality is what proves "the retry bought nothing" without storing
    either body twice.
    """

    left = {"a": 1, "b": [1, 2]}
    right = {"b": [1, 2], "a": 1}
    assert payload_hash(left) == payload_hash(right)
    assert payload_hash(left) != payload_hash({"a": 1, "b": [1, 3]})
    assert payload_hash(None) == ""
    assert payload_hash("text").startswith("sha256:")


@pytest.mark.parametrize(
    "payload",
    [
        {"entities": [], "processes": None},
        {"entities": "nope"},
        {},
        None,
        "not a payload",
    ],
)
def test_counting_never_raises_on_the_malformed_payloads_it_exists_for(payload: Any) -> None:
    """These counters run on exactly the payloads that caused the failure.

    One that raises on a payload whose ``entities`` is a list would take out the
    diagnostics for the run it was meant to explain.
    """

    assert count_entities(payload)["total"] == 0
    assert count_processes(payload)["total"] == 0


def test_counting_reports_per_bucket_totals() -> None:
    payload: Dict[str, Any] = {
        "entities": {"compounds": [{}, {}], "proteins": [{}]},
        "processes": {"reactions": [{}, {}, {}], "transports": [{}]},
    }

    assert count_entities(payload) == {"compounds": 2, "proteins": 1, "total": 3}
    assert count_processes(payload) == {"reactions": 3, "transports": 1, "total": 4}

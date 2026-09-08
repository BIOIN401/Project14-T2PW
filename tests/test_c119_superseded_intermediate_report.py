"""C-119 seams 1 and 2 -- the hard prohibitions and the nine deterministic regressions.

**A REGRESSION FIX (G9 class A), NOT NEW FUNCTIONALITY.** The base-failure proof
G9 requires for these two seams is in ``tests/test_c119_g9_base_failure_proof.py``,
which is written entirely in base-existing symbols so it RUNS at base SHA
``6746a8d31e993d8dc6968ad46534b4e48b9c7bee`` and fails there on VALUES. This file is
the other half: it fixes the boundaries of the change (only ``audit_round``, only
with a later boundary, never in a pre-stamp artifact set) and replays the nine legs
C-119 section 4 requires.

Seam 3 -- the sixth cap in ``classify_release_status`` -- is a genuinely NEW
capability and is tested separately, and labelled as such, in
``tests/test_c119_release_status_superseded_cap.py``. Neither file claims the other
kind of proof.

THE FIXTURES ARE ARCHIVED PAYLOADS, REPLAYED. Nothing here runs a pipeline leg,
takes an LLM draw, touches a cache or reaches a network. Each file in
``tests/fixtures/c119/`` is four objects copied verbatim out of one archived strict
leg by ``docs/pwml_recovery_sprint/evidence/c119_build_fixtures.py``, which proves
rather than assumes that ``final_mapped.json`` is the payload the final gate report
validated (it asserts both canonical digests match before writing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.batch import driver  # noqa: E402
from t2pw.batch.driver import (  # noqa: E402
    PWML_RELEASE_READY_NAME,
    PWML_REVIEW_REQUIRED_NAME,
    STRICT,
    run_one,
)
from t2pw.pipeline.gate_reports import (  # noqa: E402
    CANONICAL_PAYLOAD_KEY,
    FINAL_GATE_REPORT_KEY,
    PHASE_AUDIT_ROUND,
    PHASE_FINAL_PRE_EXPORT,
    PHASE_INITIAL_POST_NORMALIZATION,
    gate_verdict,
)
from t2pw.pipeline.reaction_support import (  # noqa: E402
    CODE_NO_DEFENSIBLE_REACTION_SUPPORT,
    reaction_support_issue,
)
from t2pw.pipeline.release_status import (  # noqa: E402
    DIAGNOSTIC_ONLY,
    REASON_SUPERSEDED_INTERMEDIATE_REPORT,
    RELEASE_READY,
    REVIEW_REQUIRED,
)
from t2pw.pipeline.stage_contracts import (  # noqa: E402
    StageContractError,
    validate_pre_export,
)
from helpers_c119 import SNAPSHOT, artifacts as _artifacts, bundle as _bundle  # noqa: E402
import test_c119_g9_base_failure_proof as g9  # noqa: E402
from test_batch_driver import (  # noqa: E402
    PAPER,
    _write_app,
    real_streamlit,  # noqa: F401 -- autouse fixture, re-exported on purpose
)


def _contract_error_count(reassembled: dict) -> int:
    """The driver's own blocking arithmetic, through its own two functions."""

    _codes, _lines, error_count = driver._collect_issue_codes(
        driver._blocking_reports(reassembled)
    )
    return error_count


def test_the_g9_proof_module_is_pinned_to_the_real_constants() -> None:
    """The two strings the G9 module writes as literals ARE the production ones.

    ``tests/test_c119_g9_base_failure_proof.py`` cannot import them -- importing a
    symbol this card introduced would turn its base run into an ImportError, and
    **symbol absence is not proof**. This test is where the duplication is kept
    honest, and it lives here because this file is not required to import at base.
    """

    assert g9.REASON_LITERAL == REASON_SUPERSEDED_INTERMEDIATE_REPORT
    assert g9.WARN_PREFIX_LITERAL == driver.WARN_SUPERSEDED_INTERMEDIATE_PREFIX


# ===========================================================================
# THE HARD PROHIBITIONS. Each one is a named constraint of C-119 section 3.
# ===========================================================================


@pytest.mark.parametrize(
    "phase",
    [
        "post_extraction",
        "post_mapping",
        "post_audit",
        "post_remap",
        PHASE_INITIAL_POST_NORMALIZATION,
        PHASE_FINAL_PRE_EXPORT,
        "some_phase_that_does_not_exist_yet",
    ],
)
def test_a_live_report_at_any_other_phase_still_blocks(phase: str) -> None:
    """C-119 section 3.4. ONLY ``audit_round`` is excluded. Nothing else is.

    Note ``initial_post_normalization`` in the list. It is the PRE-audit snapshot
    and the app documents it in almost the same words -- but the card excludes one
    phase and one phase only, and widening the exclusion is a product decision
    nobody has taken.
    """

    artifacts = _artifacts("PMC7232280_strict_2026-09-06")
    artifacts[SNAPSHOT] = {**artifacts[SNAPSHOT], "phase": phase}
    assert _contract_error_count(artifacts) == 2
    assert SNAPSHOT in driver._blocking_reports(artifacts)


def test_a_report_with_no_phase_key_at_all_still_blocks() -> None:
    """C-119 section 3.4. An unstamped report inside a STAMPED set is live.

    The set is still recognisably post-stamp -- four of its five contract reports
    carry a phase -- so the legacy branch does not fire, and a report whose
    boundary this code cannot identify is read as live. Absence of a phase is never
    read as ``audit_round``.
    """

    artifacts = _artifacts("PMC7232280_strict_2026-09-06")
    artifacts[SNAPSHOT] = {
        key: value for key, value in artifacts[SNAPSHOT].items() if key != "phase"
    }
    assert "phase" not in artifacts[SNAPSHOT]
    assert _contract_error_count(artifacts) == 2
    assert SNAPSHOT in driver._blocking_reports(artifacts)


def test_absence_of_every_later_boundary_still_blocks() -> None:
    """C-119 section 3.5. **Absence of a final boundary is never success.**

    With nothing left to supersede it, the ``audit_round`` snapshot IS the latest
    verdict available for this leg, and it must still destroy the export.
    """

    artifacts = _artifacts("PMC7232280_strict_2026-09-06")
    for key in ("post_audit_contract_report", "post_remap_contract_report", FINAL_GATE_REPORT_KEY):
        artifacts.pop(key)
    assert _contract_error_count(artifacts) == 2
    assert SNAPSHOT in driver._blocking_reports(artifacts)


@pytest.mark.parametrize(
    "keep",
    ["post_audit_contract_report", "post_remap_contract_report", FINAL_GATE_REPORT_KEY],
)
def test_any_one_later_boundary_is_enough_to_supersede(keep: str) -> None:
    """The other half of the fail-closed rule: one authoritative boundary suffices."""

    artifacts = _artifacts("PMC7232280_strict_2026-09-06")
    for key in ("post_audit_contract_report", "post_remap_contract_report", FINAL_GATE_REPORT_KEY):
        if key != keep:
            artifacts.pop(key)
    assert _contract_error_count(artifacts) == 0


def test_a_later_boundary_that_is_itself_an_audit_round_snapshot_is_not_a_boundary() -> None:
    """A snapshot cannot supersede a snapshot -- nor, therefore, itself."""

    artifacts = _artifacts("PMC7232280_strict_2026-09-06")
    artifacts.pop(FINAL_GATE_REPORT_KEY)
    for key in ("post_audit_contract_report", "post_remap_contract_report"):
        artifacts[key] = {**artifacts[key], "phase": PHASE_AUDIT_ROUND}
    assert _contract_error_count(artifacts) == 2


def test_a_final_gate_report_at_the_wrong_phase_is_not_a_boundary() -> None:
    """The same rule ``gate_verdict`` applies to itself, applied here.

    A report parked under :data:`FINAL_GATE_REPORT_KEY` that does not claim
    :data:`PHASE_FINAL_PRE_EXPORT` is exactly what ``gate_verdict`` fails closed
    on. It cannot license a contract exclusion either.
    """

    artifacts = _artifacts("PMC7232280_strict_2026-09-06")
    artifacts.pop("post_audit_contract_report")
    artifacts.pop("post_remap_contract_report")
    artifacts[FINAL_GATE_REPORT_KEY] = {
        **artifacts[FINAL_GATE_REPORT_KEY],
        "phase": PHASE_AUDIT_ROUND,
    }
    assert _contract_error_count(artifacts) == 2


def test_an_artifact_set_that_predates_the_phase_stamp_keeps_the_old_arithmetic() -> None:
    """C-119 section 2 clause 3. Archived runs keep their recorded verdicts.

    Every phase key is stripped from every contract report, which is what a
    pre-``stamp_report`` artifact set looks like. Such a set cannot tell a
    superseded snapshot from a live verdict, so it is judged by the pre-change
    rules and the errors block -- even though the later boundaries are all present.
    """

    artifacts = _artifacts("PMC7232280_strict_2026-09-06")
    for key, value in list(artifacts.items()):
        if key.endswith("_contract_report") and isinstance(value, dict):
            artifacts[key] = {k: v for k, v in value.items() if k != "phase"}
    assert _contract_error_count(artifacts) == 2
    assert SNAPSHOT in driver._blocking_reports(artifacts)


def test_the_superseded_reason_never_vanishes_from_the_release_record() -> None:
    """C-119 seam 2 clause 3. What was superseded, at which phase, how many errors."""

    artifacts = _artifacts("PMC8510960_strict_2026-09-06")
    superseded = driver._superseded_contract_reports(artifacts)
    assert superseded == [
        {
            "report": SNAPSHOT,
            "phase": PHASE_AUDIT_ROUND,
            "errors": 7,
            "superseded_by": [
                "post_audit_contract_report",
                "post_remap_contract_report",
                FINAL_GATE_REPORT_KEY,
            ],
        }
    ]

    release = _bundle("PMC8510960_strict_2026-09-06")["quarantine_release"]
    record = driver._frozen_release_record({"quarantine_report": {"release": release}}, superseded)
    assert f"{REASON_SUPERSEDED_INTERMEDIATE_REPORT}:{SNAPSHOT}@{PHASE_AUDIT_ROUND}=7" in record["reasons"]
    # The record already read review_required on its ordinary caps, and the reason
    # is recorded ANYWAY -- which is the whole point: on every leg C-119 was
    # derived from, recording "from release_ready only" would have dropped it.
    assert release["status"] == REVIEW_REQUIRED
    assert record["status"] == REVIEW_REQUIRED

    # The other, independent channel: a warning line, which the manifest row
    # carries unconditionally even when the frozen record is unreadable.
    warning = driver._superseded_warning(superseded)
    assert warning.startswith(driver.WARN_SUPERSEDED_INTERMEDIATE_PREFIX)
    assert SNAPSHOT in warning and "7 error(s)" in warning


def test_the_superseded_count_reaches_the_row_on_a_leg_that_is_still_refused(
    tmp_path: Path,
) -> None:
    """The finding does not vanish on the paths that DO refuse.

    ``PMC12452463`` @ 2026-09-02 is refused by the GATE channel, which this card
    does not touch, so it never reaches the export seam where the review reason and
    the warning are written. The neutral count is written BEFORE the blocking
    decision instead, so a reader of that leg's manifest row still learns that its
    contract channel carried 8 superseded errors -- which at the base SHA arrived as
    8 blocking contract errors in ``codes`` and ``detail``.

    It is a COUNT and not the export path's sentence on purpose: that sentence says
    the leg was serialized, and here it was not. And it is written only when there
    is something to say, so a leg with no superseded snapshot has a byte-identical
    row.
    """

    data = _bundle("PMC12452463_strict_2026-09-02")
    reassembled = _artifacts("PMC12452463_strict_2026-09-02")
    assert gate_verdict(reassembled).failed is True  # still refused, by the gate

    artifacts_path = tmp_path / "artifacts.json"
    artifacts_path.write_text(
        json.dumps(
            {
                **data["contract_reports"],
                FINAL_GATE_REPORT_KEY: data["final_stage3_gate_report"],
                CANONICAL_PAYLOAD_KEY: data["canonical_export_payload"],
                "artifact_set_version": 1,
                "export_mode": "pathwhiz",
                "final_mapped": data["canonical_export_payload"],
                "final_mapped_db": data["canonical_export_payload"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    body = f'''
import json
if submitted:
    st.session_state["pipeline_ready"] = True
    st.session_state["stage_one"] = {{"entities": {{}}, "processes": {{"reactions": []}}}}
    with open(r"{artifacts_path}", encoding="utf-8") as fh:
        st.session_state["final_payload"] = json.load(fh)["final_mapped"]
if st.session_state.get("pipeline_ready"):
    if st.button("Run audit and DB mapping", key="pwml_generate_btn"):
        with open(r"{artifacts_path}", encoding="utf-8") as fh:
            st.session_state["post_pipeline_artifacts"] = json.load(fh)
'''
    app = _write_app(tmp_path, "c119_pmc12452463_0902", body)
    outcome = run_one(PAPER, STRICT, app_path=app, timeout=180.0, app_timeout=90.0)

    assert outcome.status == "fail"
    assert outcome.pwml_artifact == ""
    assert PWML_RELEASE_READY_NAME not in outcome.artifacts
    assert PWML_REVIEW_REQUIRED_NAME not in outcome.artifacts
    row = outcome.to_dict()
    assert row["counts"]["superseded_contract_errors"] == 8
    assert row["release_status"]["status"] == DIAGNOSTIC_ONLY


def test_the_cap_can_never_create_release_ready() -> None:
    """C-119 section 3.3. Monotone: one step down, from one status, or nothing."""

    superseded = driver._superseded_contract_reports(_artifacts("PMC7232280_strict_2026-09-06"))
    assert superseded

    def record(status: str) -> dict:
        return driver._frozen_release_record(
            {"quarantine_report": {"release": {"status": status, "strict_acceptance_eligible": status == RELEASE_READY}}},
            superseded,
        )

    demoted = record(RELEASE_READY)
    assert demoted["status"] == REVIEW_REQUIRED
    assert demoted["strict_acceptance_eligible"] is False
    assert driver._pwml_artifact_name(demoted) == PWML_REVIEW_REQUIRED_NAME

    assert record(REVIEW_REQUIRED)["status"] == REVIEW_REQUIRED
    # Never PROMOTED: a diagnostic_only record still gets no final PWML.
    assert record(DIAGNOSTIC_ONLY)["status"] == DIAGNOSTIC_ONLY
    assert driver._pwml_artifact_name(record(DIAGNOSTIC_ONLY)) == ""

    # And with NO superseded reports the record is returned untouched, which is
    # what keeps every leg this card does not reach byte-identical.
    untouched = driver._frozen_release_record(
        {"quarantine_report": {"release": {"status": RELEASE_READY, "strict_acceptance_eligible": True}}}
    )
    assert untouched == {"status": RELEASE_READY, "strict_acceptance_eligible": True}
    assert driver._pwml_artifact_name(untouched) == PWML_RELEASE_READY_NAME


def test_f179_stays_a_floor_and_never_becomes_survivable() -> None:
    """C-119 section 3.1. The anti-invention floor does not move.

    PMC12180156's 2026-08-28 archive carries BOTH a superseded ``audit_round``
    snapshot (so this card's exclusion fires on it) and an F-179 refusal. The
    exclusion changes the CONTRACT CHANNEL's arithmetic and nothing else: the live
    pre-export predicate still answers ``no_defensible_reaction_support`` on the
    archived payload, and ``validate_pre_export`` still raises with that code on
    the inner ``pwml_contract_report`` the production caller branches on.
    """

    bundle = _bundle("PMC12180156_strict_2026-08-28")
    artifacts = _artifacts("PMC12180156_strict_2026-08-28")
    assert driver._superseded_contract_reports(artifacts)  # the exclusion DOES fire
    assert _contract_error_count(artifacts) == 0

    payload = bundle["canonical_export_payload"]
    issue = reaction_support_issue(payload)
    assert issue is not None
    assert issue["code"] == CODE_NO_DEFENSIBLE_REACTION_SUPPORT

    with pytest.raises(StageContractError) as raised:
        validate_pre_export(payload)
    inner = raised.value.report["pwml_contract_report"]
    assert inner["ok"] is False
    assert CODE_NO_DEFENSIBLE_REACTION_SUPPORT in {
        issue.get("code") for issue in inner["errors"]
    }


# ===========================================================================
# THE NINE REQUIRED DETERMINISTIC REGRESSIONS -- C-119 section 4.
# Replays over ARCHIVED payloads. No pipeline leg, no LLM draw, no network.
# ===========================================================================

#: ``stem -> (superseded error count or None, contract channel blocks, gate fails,
#: F-179 code or None, the pre-export code that refuses this leg or None)``.
#:
#: ``pathway_missing_name`` / ``pathway_missing_subject`` appear on EVERY leg's
#: ``validate_pre_export`` replay and are deliberately not asserted: the pathway
#: name and subject are supplied by the exporter at export time and are not fields
#: of ``final_mapped.json``, so they are an artefact of replaying the contract over
#: the archived payload alone. The codes asserted here are the ones that describe
#: the BIOLOGY of the archived payload.
REGRESSIONS = {
    # 1. Recovered: review_required PWML where the base SHA wrote none.
    "PMC7232280_strict_2026-09-06": (2, False, False, None, None),
    # 2. Recovered: review_required PWML where the base SHA wrote none.
    "PMC8510960_strict_2026-09-06": (7, False, False, None, None),
    # 3. THE CONTROL. It really shipped pathway.review_required.pwml, and it lands
    #    there with NO superseded-report cap at all -- on the ordinary caps alone.
    "PMC12071552_strict_2026-09-06": (None, False, False, None, None),
    # 4/5. glycine -> heme. Refused by F-179 on BOTH archives, with and without a
    #      superseded snapshot present. The gold agrees: "nothing about heme
    #      biosynthesis is exportable".
    "PMC12180156_strict_2026-09-02": (None, False, False, CODE_NO_DEFENSIBLE_REACTION_SUPPORT, CODE_NO_DEFENSIBLE_REACTION_SUPPORT),
    "PMC12180156_strict_2026-08-28": (1, False, False, CODE_NO_DEFENSIBLE_REACTION_SUPPORT, CODE_NO_DEFENSIBLE_REACTION_SUPPORT),
    # 6/7. Refused by the LIVE pre-export gate: a resolved enzyme used without its
    #      single-protein PathWhiz wrapper. Untouched by this card.
    "PMC12376012_strict_2026-09-06": (10, False, False, None, "reaction_enzyme_must_be_protein_complex"),
    "PMC11405693_strict_2026-09-06": (None, False, False, None, "reaction_enzyme_must_be_protein_complex"),
    # 8. Refused while the live Fe3+ plus-token failure exists -- the GATE channel,
    #    which this card does not touch, so the leg is refused even though its
    #    superseded snapshot no longer blocks.
    "PMC12452463_strict_2026-09-02": (8, False, True, None, None),
    # 9. The qualifying archive. May serialize, and ONLY as review_required
    #    (PRODUCT_CONTRACT 13: "Never strict success").
    "PMC12452463_strict_2026-08-28": (3, False, False, None, None),
}


@pytest.mark.parametrize("stem", sorted(REGRESSIONS))
def test_archived_leg_reaches_its_required_outcome(stem: str) -> None:
    """One row of C-119 section 4, through the production seams."""

    expected_superseded, expect_contract_block, expect_gate_fail, expect_f179, expect_preexport = (
        REGRESSIONS[stem]
    )
    bundle = _bundle(stem)
    artifacts = _artifacts(stem)
    payload = bundle["canonical_export_payload"]

    superseded = driver._superseded_contract_reports(artifacts)
    if expected_superseded is None:
        assert superseded == [], stem
    else:
        assert [item["errors"] for item in superseded] == [expected_superseded], stem
        assert [item["report"] for item in superseded] == [SNAPSHOT], stem

    assert (_contract_error_count(artifacts) > 0) is expect_contract_block, stem
    assert gate_verdict(artifacts).failed is expect_gate_fail, stem

    issue = reaction_support_issue(payload)
    assert (issue or {}).get("code") == expect_f179, stem

    try:
        validate_pre_export(payload)
        preexport_codes: set = set()
    except StageContractError as exc:
        preexport_codes = {
            item.get("code") for item in exc.report["pwml_contract_report"]["errors"]
        }
    if expect_preexport is None:
        assert expect_f179 is None
        assert "reaction_enzyme_must_be_protein_complex" not in preexport_codes, stem
        assert CODE_NO_DEFENSIBLE_REACTION_SUPPORT not in preexport_codes, stem
    else:
        assert expect_preexport in preexport_codes, stem


@pytest.mark.parametrize("stem", sorted(REGRESSIONS))
def test_no_archived_leg_becomes_release_ready(stem: str) -> None:
    """C-119 section 3.3 and section 4's last row, over BOTH cohorts.

    Not one case, in either cohort, may newly produce ``release_ready``. The
    archived frozen record is run through the production record seam with this
    leg's own superseded set; the status that comes out is what names the PWML
    file, and none of them earns the reserved ``pathway.pwml`` name.
    """

    bundle = _bundle(stem)
    artifacts = _artifacts(stem)
    record = driver._frozen_release_record(
        {"quarantine_report": {"release": bundle["quarantine_release"]}},
        driver._superseded_contract_reports(artifacts),
    )
    assert record["status"] != RELEASE_READY, stem
    assert record.get("strict_acceptance_eligible") is False, stem
    assert driver._pwml_artifact_name(record) != PWML_RELEASE_READY_NAME, stem

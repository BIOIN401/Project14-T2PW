"""C-126 -- the boundaries of the widened seam, and every negative control.

**A REGRESSION FIX (G9 class A), NOT NEW FUNCTIONALITY.** The base-failure proof G9
requires is in ``tests/test_c126_g9_base_failure_proof.py``, written entirely in
base-existing symbols so it RUNS at base SHA
``c6159bc27989f5bc865d4b6946b0e8eb93fb5c12`` and fails there on VALUES. This file is
the other half: it fixes the boundaries of the change and runs every negative control
C-126 section 4 requires.

WHAT C-126 CHANGED, IN ONE PREDICATE. ``_superseded_contract_reports`` gained two
conditions on top of C-119's two:

3. the report's own ``effect_on_failure`` must be ``feed_audit`` -- set by exactly one
   contract in ``stage_contracts.py`` (``validate_post_normalization``, ``:219``);
   every other uses ``abort``, and ``relax_report:186`` rewrites it to
   ``annotate_only`` in research mode;
4. its ``phase`` must be in the CLOSED allow-list ``_NON_AUTHORITATIVE_PHASES``.

**CONDITION 4 IS AN ALLOW-LIST, DELIBERATELY NOT ``!= PHASE_FINAL_PRE_EXPORT``.** The
negated form would silently enrol every future phase name, every typo and every
malformed string into the excluded set, against ``_report_phase``'s own stated
doctrine that an unidentified boundary reads as LIVE. Several tests below exist only
to fail if someone ever rewrites it as a negation.

ONE BASELINE MOVED, DELIBERATELY, AND IT IS RECORDED HERE.
``tests/test_c119_superseded_intermediate_report.py::test_a_live_report_at_any_other
_phase_still_blocks`` was parametrized with ``PHASE_INITIAL_POST_NORMALIZATION`` and
asserted it still blocks, with the comment *"widening the exclusion is a product
decision nobody has taken"*. The product owner took that decision on 2026-09-10 on the
evidence of ``F-147-RECURRENCE-DIAGNOSIS.md``. That one parameter is removed there and
its replacement -- the same mutation, now expected to be superseded -- is
:func:`test_the_initial_post_normalization_phase_is_now_superseded_too` below. Every
other parameter of that test is unchanged and still passing.

THE FIXTURES ARE ARCHIVED PAYLOADS, REPLAYED. Nothing here runs a pipeline leg, takes
an LLM draw, touches a cache or reaches a network.
"""

from __future__ import annotations

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
)
from t2pw.pipeline import canonical_hash  # noqa: E402
from t2pw.pipeline.export_mode import relax_report  # noqa: E402
from t2pw.pipeline.gate_reports import (  # noqa: E402
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
    validate_post_normalization,
)
from helpers_c119 import FIXTURES, SNAPSHOT, artifacts as _artifacts, bundle as _bundle  # noqa: E402
import test_c126_g9_base_failure_proof as g9  # noqa: E402

#: The F-147 recurrence leg committed as the tenth fixture.
NEW = "PMC13089919_strict_2026-09-10_initial_post_normalization"

#: The nine fixtures C-119 shipped. Every one of them is ``audit_round``; ZERO are
#: ``initial_post_normalization``, which is exactly why C-119's evidence base could
#: not have caught F-147.
C119_NINE = (
    "PMC11405693_strict_2026-09-06",
    "PMC12071552_strict_2026-09-06",
    "PMC12180156_strict_2026-08-28",
    "PMC12180156_strict_2026-09-02",
    "PMC12376012_strict_2026-09-06",
    "PMC12452463_strict_2026-08-28",
    "PMC12452463_strict_2026-09-02",
    "PMC7232280_strict_2026-09-06",
    "PMC8510960_strict_2026-09-06",
)

#: The three later boundaries every one of these archives carries.
BOUNDARIES = ("post_audit_contract_report", "post_remap_contract_report", FINAL_GATE_REPORT_KEY)


def _count(reassembled: dict) -> int:
    """The driver's own blocking arithmetic, through its own two functions."""

    _codes, _lines, error_count = driver._collect_issue_codes(
        driver._blocking_reports(reassembled)
    )
    return error_count


# ===========================================================================
# The literals the G9 module cannot import, kept honest here.
# ===========================================================================


def test_the_g9_proof_module_is_pinned_to_the_real_constants() -> None:
    """The strings the G9 module writes as literals ARE the production ones.

    ``tests/test_c126_g9_base_failure_proof.py`` cannot import them -- importing a
    symbol this card introduced would turn its base run into an ImportError, and
    **symbol absence is not proof**. This test is where the duplication is kept
    honest, and it lives here because this file is not required to import at base.
    """

    assert g9.FEED_AUDIT_LITERAL == driver._EFFECT_FEED_AUDIT
    assert g9.REASON_LITERAL == REASON_SUPERSEDED_INTERMEDIATE_REPORT
    assert g9.WARN_PREFIX_LITERAL == driver.WARN_SUPERSEDED_INTERMEDIATE_PREFIX


# ===========================================================================
# The new fixture: what dimension it adds, and that it is a faithful copy.
# ===========================================================================


def test_the_new_fixture_adds_the_only_dimension_c119_had_no_instance_of() -> None:
    """Nine ``audit_round``, one ``initial_post_normalization``. That is the gap.

    Also re-proves the fixture's payload binding INSIDE the test rather than trusting
    the builder script: ``canonical_export_payload`` must reproduce both digests the
    archived ``final_stage3_gate_report`` recorded, so the pairing this fixture
    reassembles is the pairing the live run had.
    """

    phases = {stem: _artifacts(stem)[SNAPSHOT].get("phase") for stem in C119_NINE}
    assert set(phases.values()) == {PHASE_AUDIT_ROUND}, phases

    data = _bundle(NEW)
    gate = data["final_stage3_gate_report"]
    payload = data["canonical_export_payload"]
    assert canonical_hash.canonical_graph_sha256(payload) == gate["canonical_graph_sha256"]
    assert canonical_hash.canonical_payload_sha256(payload) == gate["canonical_payload_sha256"]

    snapshot = data["contract_reports"][SNAPSHOT]
    assert snapshot["phase"] == PHASE_INITIAL_POST_NORMALIZATION
    assert snapshot["effect_on_failure"] == driver._EFFECT_FEED_AUDIT
    assert snapshot["ok"] is False
    assert len(snapshot["errors"]) == 2
    # The pointers address a PRE-REMAP protein list the shipped payload cannot host:
    # two named proteins became one ``Unknown`` row backing two functional complexes.
    assert [error["path"] for error in snapshot["errors"]] == [
        "/entities/proteins/0",
        "/entities/proteins/1",
    ]
    assert len(payload["entities"]["proteins"]) == 1
    assert payload["entities"]["proteins"][0]["name"] == "Unknown"
    assert payload["protein_export_policy"]["summary"]["unknown_backed_functional_complexes"] == 2
    # The tenth file really is the tenth file.
    assert len(sorted(FIXTURES.glob("*.json"))) == len(C119_NINE) + 1


# ===========================================================================
# CONDITION 4 -- THE ALLOW-LIST. Each of these fails if it is ever negated.
# ===========================================================================


def test_the_non_authoritative_phase_set_is_exactly_the_two_the_module_defines() -> None:
    """A CLOSED set of two, and ``final_pre_export`` is not in it.

    ``gate_reports.py`` defines exactly three phases. Two of them it documents as
    non-authoritative; the third it calls *"THE authoritative phase"*. The set is
    complete by construction, which is what makes an allow-list safe here.
    """

    assert driver._NON_AUTHORITATIVE_PHASES == frozenset(
        {PHASE_INITIAL_POST_NORMALIZATION, PHASE_AUDIT_ROUND}
    )
    assert PHASE_FINAL_PRE_EXPORT not in driver._NON_AUTHORITATIVE_PHASES


@pytest.mark.parametrize(
    "phase",
    [
        PHASE_FINAL_PRE_EXPORT,
        "post_extraction",
        "post_mapping",
        "post_audit",
        "post_remap",
        # THE NEGATION TRAP. Every one of these is ``!= PHASE_FINAL_PRE_EXPORT`` and
        # would therefore be SILENTLY EXCLUDED by a negated condition 4. They are not
        # hypothetical shapes: a fourth phase added to ``gate_reports``, a typo in a
        # future stamp call, a British spelling, a capitalisation drift, and an empty
        # string from a truncated write are each one commit away.
        "some_phase_that_does_not_exist_yet",
        "initial_post_normalisation",
        "INITIAL_POST_NORMALIZATION",
        "Audit_Round",
        "audit_round_2",
        "post_normalization",
        "final_pre_export_stage3_gates",
        "  ",
    ],
)
def test_a_report_at_an_unrecognised_or_authoritative_phase_still_blocks(phase: str) -> None:
    """C-126 section 4, control 6. **Condition 4 is an ALLOW-LIST.**

    An unidentified boundary reads as LIVE -- ``_report_phase``'s own docstring.
    """

    artifacts = _artifacts(NEW)
    artifacts[SNAPSHOT] = {**artifacts[SNAPSHOT], "phase": phase}
    assert driver._superseded_contract_reports(artifacts) == [], phase
    assert _count(artifacts) == 2, phase
    assert SNAPSHOT in driver._blocking_reports(artifacts)


def test_a_report_with_no_phase_key_at_all_still_blocks() -> None:
    """Absence of a phase is never read as a non-authoritative phase.

    The set is still recognisably post-stamp -- four of its five contract reports
    carry a phase -- so the legacy branch does not fire, and a report whose boundary
    this code cannot identify is read as live.
    """

    artifacts = _artifacts(NEW)
    artifacts[SNAPSHOT] = {
        key: value for key, value in artifacts[SNAPSHOT].items() if key != "phase"
    }
    assert "phase" not in artifacts[SNAPSHOT]
    assert driver._superseded_contract_reports(artifacts) == []
    assert _count(artifacts) == 2
    assert SNAPSHOT in driver._blocking_reports(artifacts)


def test_surrounding_whitespace_on_a_phase_string_is_stripped_as_it_always_was() -> None:
    """PRE-EXISTING normalization, recorded so it is not mistaken for new laxity.

    ``_report_phase`` runs the raw value through ``_text``, which ``strip()``s it
    (``driver.py:452``). It did so at the base SHA too, where ``" audit_round "``
    compared equal to :data:`PHASE_AUDIT_ROUND` for exactly the same reason. C-126
    changes the comparison from one name to a CLOSED SET of two; it does not touch
    the normalization. A whitespace-ONLY phase still collapses to ``""``, which is in
    neither set, and still blocks.
    """

    artifacts = _artifacts(NEW)
    padded = {**artifacts[SNAPSHOT], "phase": f"  {PHASE_INITIAL_POST_NORMALIZATION}	"}
    artifacts[SNAPSHOT] = padded
    assert driver._report_phase(padded) == PHASE_INITIAL_POST_NORMALIZATION
    assert _count(artifacts) == 0

    artifacts = _artifacts(NEW)
    artifacts[SNAPSHOT] = {**artifacts[SNAPSHOT], "phase": "   "}
    assert driver._report_phase(artifacts[SNAPSHOT]) == ""
    assert _count(artifacts) == 2


def test_the_initial_post_normalization_phase_is_now_superseded_too() -> None:
    """THE REPLACEMENT for the removed C-119 parameter, stated as its own test.

    ``tests/test_c119_superseded_intermediate_report.py`` asserted this exact
    mutation still blocks. The product owner's 2026-09-10 decision reverses it, and
    this is where the reversal is recorded: the same C-119 archive, its snapshot
    restamped to the OTHER non-authoritative phase, is now superseded -- and the
    descriptor reports that other phase, not a hard-coded one.
    """

    artifacts = _artifacts("PMC7232280_strict_2026-09-06")
    artifacts[SNAPSHOT] = {
        **artifacts[SNAPSHOT],
        "phase": PHASE_INITIAL_POST_NORMALIZATION,
    }
    assert _count(artifacts) == 0
    assert [item["phase"] for item in driver._superseded_contract_reports(artifacts)] == [
        PHASE_INITIAL_POST_NORMALIZATION
    ]


def test_the_descriptor_reports_the_reports_actual_phase_not_a_constant() -> None:
    """C-126 section 3. A hard-coded phase would make the review metadata LIE.

    C-119 wrote ``"phase": PHASE_AUDIT_ROUND`` into the descriptor because that was
    the only phase that could reach the line. With two phases reachable, the release
    reason and the manifest row would name the WRONG boundary as superseded.
    """

    for stem, expected in ((NEW, PHASE_INITIAL_POST_NORMALIZATION),
                           ("PMC7232280_strict_2026-09-06", PHASE_AUDIT_ROUND)):
        descriptors = driver._superseded_contract_reports(_artifacts(stem))
        assert [item["phase"] for item in descriptors] == [expected], stem
        reason = driver._frozen_release_record(
            {"quarantine_report": {"release": {"status": REVIEW_REQUIRED}}}, descriptors
        )["reasons"][-1]
        assert f"{SNAPSHOT}@{expected}=" in reason, stem


# ===========================================================================
# CONDITION 3 -- effect_on_failure. Only ``feed_audit`` may be set aside.
# ===========================================================================


@pytest.mark.parametrize("phase", [PHASE_INITIAL_POST_NORMALIZATION, PHASE_AUDIT_ROUND])
@pytest.mark.parametrize("effect", ["abort", "annotate_only", "", "FEED_AUDIT", "feed audit"])
def test_a_non_feed_audit_report_with_errors_still_blocks_at_every_phase(
    phase: str, effect: str
) -> None:
    """C-126 section 4, control 7. **An ``abort`` contract blocks at any phase.**

    ``validate_post_normalization`` ESCALATES itself to ``abort`` and raises when
    ``_validate_payload_container`` fails (``stage_contracts.py:221-224``), so an
    ``abort`` report reaching this seam may be carrying structural garbage. It is
    never set aside.
    """

    artifacts = _artifacts(NEW)
    artifacts[SNAPSHOT] = {**artifacts[SNAPSHOT], "phase": phase, "effect_on_failure": effect}
    assert driver._superseded_contract_reports(artifacts) == []
    assert _count(artifacts) == 2
    assert SNAPSHOT in driver._blocking_reports(artifacts)


def test_a_report_with_no_effect_on_failure_key_at_all_still_blocks() -> None:
    """Unstated is not ``feed_audit``. The safe reading of "unstated" is "blocking"."""

    artifacts = _artifacts(NEW)
    artifacts[SNAPSHOT] = {
        key: value
        for key, value in artifacts[SNAPSHOT].items()
        if key != "effect_on_failure"
    }
    assert driver._superseded_contract_reports(artifacts) == []
    assert _count(artifacts) == 2


def test_feed_audit_is_still_the_only_value_validate_post_normalization_writes() -> None:
    """Condition 3 rests on production, so production is asserted, not quoted.

    If ``stage_contracts.validate_post_normalization`` ever stops writing
    ``feed_audit``, condition 3 goes inert and this seam silently stops firing -- so
    the dependency is pinned rather than left as a comment.
    """

    report = validate_post_normalization(
        {"entities": {"proteins": [], "compounds": []}, "processes": {"reactions": []}}
    )
    assert report["effect_on_failure"] == driver._EFFECT_FEED_AUDIT == "feed_audit"


# ===========================================================================
# RESEARCH MODE -- inert by construction, and the mechanism is pinned.
# ===========================================================================


def test_a_research_mode_relaxed_report_is_never_superseded() -> None:
    """C-126 section 4, last row. ``relax_report`` rewrites ``effect_on_failure``.

    ``export_mode.relax_report:186`` stamps ``annotate_only`` (or ``abort`` when a
    structural guard still blocks), so a research-mode
    ``post_normalization_contract_report`` NEVER carries ``feed_audit`` and condition
    3 makes this seam inert there. Measured across all 75 archived research legs on
    disk: 0 changed, and all 75 carry ``annotate_only`` -- 5 of them at
    ``initial_post_normalization``, which is the combination that would otherwise
    have moved.
    """

    artifacts = _artifacts(NEW)
    relaxed = relax_report(artifacts[SNAPSHOT], stage="post_normalization")
    assert relaxed["effect_on_failure"] == "annotate_only"
    artifacts[SNAPSHOT] = relaxed
    assert driver._superseded_contract_reports(artifacts) == []
    # And it does not block either -- relax_report already moved its errors to
    # warnings. Research mode is untouched in BOTH directions.
    assert _count(artifacts) == 0
    assert relaxed["ok"] is True
    assert relaxed["errors"] == []


# ===========================================================================
# CONDITIONS 1 AND 2 -- C-119's, preserved unchanged, re-proved on the NEW phase.
# ===========================================================================


def test_absence_of_every_later_boundary_still_blocks() -> None:
    """C-126 section 4, control 2. **Absence of a final boundary is never success.**"""

    artifacts = _artifacts(NEW)
    for key in BOUNDARIES:
        artifacts.pop(key)
    assert driver._superseding_boundaries(artifacts) == []
    assert driver._superseded_contract_reports(artifacts) == []
    assert _count(artifacts) == 2
    assert SNAPSHOT in driver._blocking_reports(artifacts)


@pytest.mark.parametrize("keep", BOUNDARIES)
def test_any_one_later_boundary_is_enough_to_supersede(keep: str) -> None:
    """The other half of the fail-closed rule, re-proved at the new phase."""

    artifacts = _artifacts(NEW)
    for key in BOUNDARIES:
        if key != keep:
            artifacts.pop(key)
    assert _count(artifacts) == 0


def test_a_legacy_artifact_set_with_no_phase_anywhere_keeps_the_old_arithmetic() -> None:
    """C-126 section 4, control 5. **Byte-identical.** Archived runs keep verdicts.

    Every phase key is stripped from every contract report, which is what a
    pre-``stamp_report`` artifact set looks like. Such a set cannot tell a superseded
    snapshot from a live verdict, so it is judged by the pre-change rules -- even
    though the later boundaries are all present and all clean.
    """

    artifacts = _artifacts(NEW)
    for key, value in list(artifacts.items()):
        if key.endswith("_contract_report") and isinstance(value, dict):
            artifacts[key] = {k: v for k, v in value.items() if k != "phase"}
    assert driver._artifact_set_is_phase_stamped(artifacts) is False
    assert driver._superseded_contract_reports(artifacts) == []
    assert _count(artifacts) == 2
    assert SNAPSHOT in driver._blocking_reports(artifacts)


# ===========================================================================
# THE GATE CHANNEL -- untouched, and still authoritative in both directions.
# ===========================================================================


def test_a_dirty_final_pre_export_gate_report_still_blocks_via_gate_verdict() -> None:
    """C-126 section 4, control 3. The authoritative report is never ignored.

    The contract channel stops objecting -- the snapshot is superseded -- and the leg
    is refused anyway, by ``gate_verdict``, which this card does not touch.
    """

    artifacts = _artifacts(NEW)
    artifacts[FINAL_GATE_REPORT_KEY] = {
        **artifacts[FINAL_GATE_REPORT_KEY],
        "ok": False,
        "errors": [
            {"path": "/entities/proteins/0", "reason": "Protein 'Unknown' is missing a UniProt or DrugBank identifier."}
        ],
    }
    assert _count(artifacts) == 0
    verdict = gate_verdict(artifacts)
    assert verdict.failed is True
    assert len(verdict.errors) == 1


def test_a_missing_final_stage3_gate_report_fails_closed() -> None:
    """C-126 section 4, control 4. Unchanged ``gate_verdict`` behaviour.

    With the two contract boundaries still present the snapshot is still superseded,
    so the contract channel is silent -- and the GATE channel fails closed on the
    missing final report, which is what refuses the leg.
    """

    artifacts = _artifacts(NEW)
    artifacts.pop(FINAL_GATE_REPORT_KEY)
    assert _count(artifacts) == 0
    verdict = gate_verdict(artifacts)
    assert verdict.failed is True
    assert verdict.source == "fail_closed"


def test_a_final_gate_report_at_the_wrong_phase_is_not_a_superseding_boundary() -> None:
    """C-119's rule, unchanged, re-proved with the new phase in play."""

    artifacts = _artifacts(NEW)
    artifacts.pop("post_audit_contract_report")
    artifacts.pop("post_remap_contract_report")
    artifacts[FINAL_GATE_REPORT_KEY] = {
        **artifacts[FINAL_GATE_REPORT_KEY],
        "phase": PHASE_AUDIT_ROUND,
    }
    assert driver._superseding_boundaries(artifacts) == []
    assert _count(artifacts) == 2


# ===========================================================================
# F-179 -- the anti-invention floor, untouched. The glycine -> heme case.
# ===========================================================================


@pytest.mark.parametrize(
    "stem", ["PMC12180156_strict_2026-08-28", "PMC12180156_strict_2026-09-02"]
)
@pytest.mark.parametrize("phase", [PHASE_INITIAL_POST_NORMALIZATION, PHASE_AUDIT_ROUND])
def test_f179_glycine_to_heme_stays_refused_at_either_non_authoritative_phase(
    stem: str, phase: str
) -> None:
    """C-126 section 4, control 1. **F-179 must not be bypassable.**

    Both archives, and with the snapshot restamped to EITHER non-authoritative phase
    so the widened exclusion definitely fires. F-179 is a reaction-support invariant
    on a different channel: the live pre-export predicate still answers
    ``no_defensible_reaction_support`` on the archived payload, and
    ``validate_pre_export`` still raises with that code.
    """

    artifacts = _artifacts(stem)
    artifacts[SNAPSHOT] = {**artifacts[SNAPSHOT], "phase": phase}
    assert _count(artifacts) == 0

    payload = _bundle(stem)["canonical_export_payload"]
    issue = reaction_support_issue(payload)
    assert issue is not None
    assert issue["code"] == CODE_NO_DEFENSIBLE_REACTION_SUPPORT

    with pytest.raises(StageContractError) as raised:
        validate_pre_export(payload)
    inner = raised.value.report["pwml_contract_report"]
    assert inner["ok"] is False
    assert CODE_NO_DEFENSIBLE_REACTION_SUPPORT in {i.get("code") for i in inner["errors"]}


# ===========================================================================
# MERGE RULE 6 / C-119 section 3.3 -- nothing becomes release_ready.
# ===========================================================================


def test_the_cap_fires_on_the_new_phase_and_can_never_create_release_ready() -> None:
    """C-126 section 3. **Prove the C-119 cap follows, do not assume it.**

    The cap is reached through the same descriptor list, so widening the descriptor
    predicate should carry it for free. "Should" is not evidence.
    """

    superseded = driver._superseded_contract_reports(_artifacts(NEW))
    assert [item["phase"] for item in superseded] == [PHASE_INITIAL_POST_NORMALIZATION]

    def record(status: str) -> dict:
        return driver._frozen_release_record(
            {
                "quarantine_report": {
                    "release": {
                        "status": status,
                        "strict_acceptance_eligible": status == RELEASE_READY,
                    }
                }
            },
            superseded,
        )

    demoted = record(RELEASE_READY)
    assert demoted["status"] == REVIEW_REQUIRED
    assert demoted["strict_acceptance_eligible"] is False
    assert driver._pwml_artifact_name(demoted) == PWML_REVIEW_REQUIRED_NAME
    assert (
        f"{REASON_SUPERSEDED_INTERMEDIATE_REPORT}:{SNAPSHOT}@{PHASE_INITIAL_POST_NORMALIZATION}=2"
        in demoted["reasons"]
    )

    assert record(REVIEW_REQUIRED)["status"] == REVIEW_REQUIRED
    # Never PROMOTED: a diagnostic_only record still gets no final PWML.
    assert record(DIAGNOSTIC_ONLY)["status"] == DIAGNOSTIC_ONLY
    assert driver._pwml_artifact_name(record(DIAGNOSTIC_ONLY)) == ""


@pytest.mark.parametrize("stem", (*C119_NINE, NEW))
def test_no_archived_leg_becomes_release_ready(stem: str) -> None:
    """Merge rule 6, over all ten fixtures including the new one."""

    record = driver._frozen_release_record(
        {"quarantine_report": {"release": _bundle(stem)["quarantine_release"]}},
        driver._superseded_contract_reports(_artifacts(stem)),
    )
    assert record["status"] != RELEASE_READY, stem
    assert record.get("strict_acceptance_eligible") is False, stem
    assert driver._pwml_artifact_name(record) != PWML_RELEASE_READY_NAME, stem


# ===========================================================================
# THE NINE C-119 FIXTURES -- verdicts pinned, before and after.
# ===========================================================================

#: ``stem -> (contract error_count, superseded descriptor count, snapshot phase,
#: descriptor errors or None, gate_verdict.failed)``. **MEASURED AT BASE SHA
#: ``c6159bc2`` AND AT THE TIP AND FOUND IDENTICAL.** Every one of the nine carries an
#: ``audit_round`` / ``feed_audit`` snapshot, so C-126's two new conditions are both
#: already satisfied on all nine and neither can move them.
C119_VERDICTS = {
    "PMC11405693_strict_2026-09-06": (0, 0, PHASE_AUDIT_ROUND, None, False),
    "PMC12071552_strict_2026-09-06": (0, 0, PHASE_AUDIT_ROUND, None, False),
    "PMC12180156_strict_2026-08-28": (0, 1, PHASE_AUDIT_ROUND, 1, False),
    "PMC12180156_strict_2026-09-02": (0, 0, PHASE_AUDIT_ROUND, None, False),
    "PMC12376012_strict_2026-09-06": (0, 1, PHASE_AUDIT_ROUND, 10, False),
    "PMC12452463_strict_2026-08-28": (0, 1, PHASE_AUDIT_ROUND, 3, False),
    "PMC12452463_strict_2026-09-02": (0, 1, PHASE_AUDIT_ROUND, 8, True),
    "PMC7232280_strict_2026-09-06": (0, 1, PHASE_AUDIT_ROUND, 2, False),
    "PMC8510960_strict_2026-09-06": (0, 1, PHASE_AUDIT_ROUND, 7, False),
}


@pytest.mark.parametrize("stem", sorted(C119_VERDICTS))
def test_the_nine_c119_fixtures_keep_their_verdicts_exactly(stem: str) -> None:
    """C-126 section 4. **This widens C-119; it must not move it.**"""

    expect_count, expect_n, expect_phase, expect_errors, expect_gate = C119_VERDICTS[stem]
    artifacts = _artifacts(stem)
    assert artifacts[SNAPSHOT].get("phase") == expect_phase, stem
    assert artifacts[SNAPSHOT].get("effect_on_failure") == driver._EFFECT_FEED_AUDIT, stem
    assert _count(artifacts) == expect_count, stem
    superseded = driver._superseded_contract_reports(artifacts)
    assert len(superseded) == expect_n, stem
    if expect_errors is None:
        assert superseded == [], stem
    else:
        assert [item["errors"] for item in superseded] == [expect_errors], stem
        assert [item["report"] for item in superseded] == [SNAPSHOT], stem
        assert [item["phase"] for item in superseded] == [expect_phase], stem
    assert gate_verdict(artifacts).failed is expect_gate, stem

"""C-110 -- NEW ACCEPTANCE TEST for a NEW CAPABILITY: ``PASS_NEGATIVE_CONTROL``.

G9 LABEL, STATED PLAINLY AND FIRST
----------------------------------
**Everything in this file is a NEW capability, not a regression fix.** The status
``PASS_NEGATIVE_CONTROL`` did not exist before C-110, in any spelling, and no
behaviour asserted here was ever observable at the base SHA. There is therefore
**no base failure to fabricate and none is claimed**. The one thing this file
pins as *pre-existing* is the MOTIVATING MEASUREMENT -- that a leg which declined
exactly as the gold demands still carries a failure token as its raw verdict --
and that pin asserts the raw token is **preserved**, which is a property of the
tip, not a base regression.

WHAT THE RULING SAYS (ORCH-717 Q1, product owner)
-------------------------------------------------
A gold-designated negative control -- or an equivalent ``context_only`` case --
passes its semantic expectation when **all three** hold:

1. the pipeline releases **no pathway reactions**;
2. it provides the **required rejection or empty-pathway reason**;
3. the empty result is **not** caused by a timeout, crash, missing artifact or
   infrastructure failure.

WHY CONDITION 3 IS THE WHOLE CARD
---------------------------------
An empty result may be rewarded only when it is a **DECISION**, never when it is
a **CASUALTY**. Getting it wrong permissively would convert every timeout on a
negative control into a pass -- on exactly the papers where "nothing" looks
normal. F-148 is the standing evidence: a killed leg preserves the stop reason
and little else, ``files: []`` and ``counts: {}`` are the signature of a child
killed with its finalization reserve already spent, and the artifact needed to
rule out retry amplification is the artifact the kill destroyed.

So the five populations are tested **SEPARATELY, one test each**, and for the
three that must NOT earn the status the test asserts the exact reason it was
withheld -- not merely that something went wrong.

Every fixture is constructed here. **No committed run is scored by this file**,
and T-107's artifacts are neither read nor re-scored: its verdict is
``NOT ACCEPTED`` and that is a fact about the artifacts it produced.

Offline and deterministic: no network, no LLM, no Streamlit at scoring time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench.acceptance import (  # noqa: E402
    MODES,
    NC_ARM_CONTEXT_ONLY,
    NC_ARM_NEGATIVE_CONTROL,
    NC_BLOCK_CODES,
    NC_BLOCK_DELIVERABLE_PRODUCED,
    NC_BLOCK_EXECUTION_FAILURE,
    NC_BLOCK_INDETERMINATE,
    NC_BLOCK_NOT_ATTEMPTED,
    NC_BLOCK_NO_ARTIFACTS,
    NC_BLOCK_NO_STATED_REASON,
    NC_BLOCK_REACTIONS_RELEASED,
    NEGATIVE_CONTROL_NOT_AWARDED,
    NEGATIVE_CONTROL_PASS,
    PRIORITY1_TARGET,
    PRIORITY1_VARIANCE_CEILING,
    _empty_is_correct,
    negative_control_outcome,
    score_run,
)
from t2pw.bench.goldset import RELEVANCE_CONTEXT_ONLY, load_gold_set  # noqa: E402
from t2pw.bench.metrics import BLOCKER_SCOPES  # noqa: E402
from t2pw.bench.render import render_text  # noqa: E402
from t2pw.bench.semantic import ERR_UNSUPPORTED_REACTIONS  # noqa: E402
from t2pw.pipeline.deadline import (  # noqa: E402
    BUDGET_EXHAUSTED,
    OPERATION_TIMEOUT,
    OPERATIONAL_TERMINATION_REASONS,
)

GOLD = load_gold_set()

#: The declared negative control -- ``max_retained_reactions == 0``. Arm 1 of
#: ``_empty_is_correct``. This is the paper the ruling was written about.
NEG_CONTROL = "PMC13231680"
#: ``context_only`` with ``min_connected_reactions == 0`` but a NON-zero
#: retention ceiling, so ``is_negative_control`` is False. Arm 2, and it is a
#: real case in the pinned gold rather than a hypothetical.
CONTEXT_ONLY = "PMC12180156"
#: A mechanistically relevant paper with a real connected-core floor. The
#: positive control: this rule must not touch it at all.
POSITIVE = "PMC12096016"


# ---------------------------------------------------------------------------
# Fixtures. Rows are modelled on the shapes the batch runner and driver
# actually write -- see ``batch/runner.py::_timeout_row`` / ``_crash_row`` and
# ``batch/driver.py::RunOutcome.to_dict``.
# ---------------------------------------------------------------------------
def _payload(reactions: int):
    rows = [
        {
            "name": f"step {i}",
            "inputs": [f"m{i}"],
            "outputs": [f"m{i + 1}"],
            "evidence": "quoted",
        }
        for i in range(reactions)
    ]
    names = {n for r in rows for n in (*r["inputs"], *r["outputs"])}
    return {
        "entities": {
            "compounds": [{"name": n} for n in sorted(names)],
            "proteins": [],
            "protein_complexes": [],
        },
        "processes": {"reactions": rows, "transports": [], "interactions": []},
    }


def _declined_row(paper_id: str, mode: str = "strict") -> dict:
    """A leg that produced nothing and SAID SO. The shape the ruling rewards.

    Modelled on the measured ``PMC13231680/strict`` outcome: ``reactions=0``,
    ``failure_kind=no_reactions``, a stated reason -- and, per PRODUCT_CONTRACT
    4, a preserved diagnostic bundle.
    """

    return {
        "paper_id": paper_id,
        "slug": paper_id,
        "mode": mode,
        "status": "fail",
        "stage": "stage1",
        "failure_kind": "no_reactions",
        "message": (
            "extraction produced no reactions: nothing lipid-A-related is present in "
            "this paper at any level of partiality, so no pathway was exported"
        ),
        "issue_codes": [],
        "counts": {"reactions": 0, "transports": 0, "entities": 0},
        "files": [{"name": "extraction_diagnostics.json", "bytes": 812}],
    }


def _timeout_row(paper_id: str, mode: str = "strict", *, message: str = "") -> dict:
    """``runner._timeout_row``'s shape: an OUTER parent kill. F-148 § 1."""

    return {
        "paper_id": paper_id,
        "slug": paper_id,
        "mode": mode,
        "status": "timeout",
        "stage": "unknown",
        "failure_kind": "timeout",
        "termination_reason": BUDGET_EXHAUSTED,
        "operational_failure": True,
        "message": message
        or (
            "the child process was still running after 1800s and was killed, so this "
            f"paper+mode produced nothing ({BUDGET_EXHAUSTED})"
        ),
        "issue_codes": [],
        "counts": {},
        "files": [],
    }


def _run(tmp_path: Path, rows, payloads=None) -> object:
    """Score a constructed run directory. Nothing committed is read."""

    run_dir = tmp_path / "2026-01-01_0000"
    (run_dir / "papers").mkdir(parents=True)
    for key, payload in (payloads or {}).items():
        paper_id, mode = key
        leg_dir = run_dir / "papers" / paper_id / mode
        leg_dir.mkdir(parents=True, exist_ok=True)
        (leg_dir / "merged_payload.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    (run_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return score_run(run_dir, GOLD)


def _leg(report, paper_id: str, mode: str = "strict"):
    paper = next(p for p in report.papers if p.paper_id == paper_id)
    return paper.leg(mode)


def _record(tmp_path: Path, row: dict, payloads=None) -> dict:
    report = _run(tmp_path, [row], payloads=payloads)
    return _leg(report, row["paper_id"], row["mode"]).negative_control


# ===========================================================================
# THE FIVE POPULATIONS -- five tests, five assertions, no parameterised blur.
# ===========================================================================
def test_population_1_a_positive_control_still_requires_reaction_production(tmp_path):
    """POPULATION 1. A mechanistically relevant paper is untouched by this rule.

    Handed the EXACT row that earns the status on a negative control, a positive
    control gets no record at all -- so nothing about it can be read as a pass,
    and it still has to produce reactions to succeed at anything.
    """

    report = _run(tmp_path, [_declined_row(POSITIVE)])
    leg = _leg(report, POSITIVE)

    assert leg.negative_control is None, (
        "a relevant paper must carry NO negative-control record: producing nothing "
        "is not the right outcome for it, and the rule must not even apply"
    )
    assert not leg.extracted
    assert not leg.passed
    # The other gold cases have no row in this run, so they carry NOT_AWARDED
    # records for `not_attempted` -- which is itself the right answer. What must
    # not happen is any leg earning the status off a positive control's row.
    assert NEGATIVE_CONTROL_PASS not in report.negative_control_outcomes
    assert set(report.negative_control_outcomes) <= {NEGATIVE_CONTROL_NOT_AWARDED}


def test_population_2_a_true_negative_control_earns_the_status(tmp_path):
    """POPULATION 2. Empty, a stated reason, no infrastructure failure.

    This is the case the ruling exists for, and the only shape that earns the
    status. All three conditions are affirmatively true and ``blocked_by`` is
    empty.
    """

    record = _record(tmp_path, _declined_row(NEG_CONTROL))

    assert record is not None
    assert record["status"] == NEGATIVE_CONTROL_PASS
    assert record["blocked_by"] == []
    assert record["arm"] == NC_ARM_NEGATIVE_CONTROL
    assert record["conditions"] == {
        "no_reactions_released": True,
        "declared_decline_reason": True,
        "no_execution_failure": True,
    }


def test_population_3_a_timeout_does_not_earn_the_status(tmp_path):
    """POPULATION 3, and the one that matters most. **PROVED, not assumed.**

    A timed-out leg on a negative control is empty for an OPERATIONAL reason. If
    this ever passes, every timeout on a negative control becomes a ``PASS`` on
    exactly the papers where "nothing" looks normal.
    """

    record = _record(tmp_path, _timeout_row(NEG_CONTROL))

    assert record is not None, "the case still applies -- the rule must judge it, not skip it"
    assert record["status"] == NEGATIVE_CONTROL_NOT_AWARDED
    assert NC_BLOCK_EXECUTION_FAILURE in record["blocked_by"]
    assert record["conditions"]["no_execution_failure"] is False
    # The row's own operational verdict survives into the record, so a reader
    # sees WHY it was withheld without leaving the row.
    assert record["raw"]["operational_failure"] is True
    assert record["raw"]["termination_reason"] == BUDGET_EXHAUSTED


def test_population_4_a_missing_artifact_does_not_earn_the_status(tmp_path):
    """POPULATION 4. Empty, a stated reason -- and NOTHING PRESERVED.

    F-148: ``files: []`` is the signature of a child killed with its
    finalization reserve already spent, and PRODUCT_CONTRACT 4 requires a
    no-PWML outcome to preserve its diagnostic artifacts. A leg that preserved
    nothing cannot show it decided anything, so it does not earn the status even
    though its ``failure_kind`` and message look exactly like a clean decline.
    """

    row = _declined_row(NEG_CONTROL)
    row["files"] = []
    record = _record(tmp_path, row)

    assert record["status"] == NEGATIVE_CONTROL_NOT_AWARDED
    assert NC_BLOCK_NO_ARTIFACTS in record["blocked_by"]
    assert record["conditions"]["no_execution_failure"] is False
    # And it is ONLY the artifact condition that failed -- the other two held,
    # which is what makes this population distinct from populations 3 and 5.
    assert record["conditions"]["no_reactions_released"] is True
    assert record["conditions"]["declared_decline_reason"] is True
    assert record["blocked_by"] == [NC_BLOCK_NO_ARTIFACTS]


def test_population_5_an_accidental_empty_output_does_not_earn_the_status(tmp_path):
    """POPULATION 5. Empty, artifacts preserved, no execution failure -- and SILENT.

    Condition 2 is a POSITIVE requirement. A leg that produced nothing and said
    nothing is a silence, not a decline, and a silence is indistinguishable from
    an extractor that simply returned an empty structure.
    """

    row = _declined_row(NEG_CONTROL)
    row["failure_kind"] = ""
    row["message"] = ""
    row["issue_codes"] = []
    record = _record(tmp_path, row)

    assert record["status"] == NEGATIVE_CONTROL_NOT_AWARDED
    assert NC_BLOCK_NO_STATED_REASON in record["blocked_by"]
    assert record["conditions"]["declared_decline_reason"] is False
    assert record["rejection_reason"] == ""
    # Nothing else failed: it released nothing and it was not a casualty. The
    # missing reason alone is what withheld the status.
    assert record["conditions"]["no_reactions_released"] is True
    assert record["conditions"]["no_execution_failure"] is True


# ===========================================================================
# Condition 3, beyond the timeout: crash, not-attempted, indeterminate.
# ===========================================================================
def test_a_crash_does_not_earn_the_status(tmp_path):
    """``runner._crash_row``: exited without printing a result line.

    Separate from the timeout population because it is a separate mechanism --
    an outer kill and a crash are different epistemic situations (F-148 § 1) and
    a rule that caught only one of them would be half a rule.
    """

    row = _declined_row(NEG_CONTROL)
    row.update(
        {
            "status": "error",
            "stage": "unknown",
            "failure_kind": "crash",
            "message": "the child process exited with code 1 without printing a result line",
            "counts": {},
            "files": [],
        }
    )
    record = _record(tmp_path, row)

    assert record["status"] == NEGATIVE_CONTROL_NOT_AWARDED
    assert NC_BLOCK_EXECUTION_FAILURE in record["blocked_by"]
    assert NC_BLOCK_NO_ARTIFACTS in record["blocked_by"]


def test_a_leg_with_no_manifest_row_does_not_earn_the_status(tmp_path):
    """The other shape of "missing artifact": the leg's record is not there.

    An unattempted leg produced nothing in the most literal sense. Rewarding it
    would turn a run that never executed into a run of clean declines.
    """

    # The research leg of the negative control has no row at all.
    report = _run(tmp_path, [_declined_row(NEG_CONTROL, "strict")])
    leg = _leg(report, NEG_CONTROL, "research")

    assert leg.attempted is False
    assert leg.negative_control["status"] == NEGATIVE_CONTROL_NOT_AWARDED
    assert NC_BLOCK_NOT_ATTEMPTED in leg.negative_control["blocked_by"]


@pytest.mark.parametrize("kind", ["", "unknown"])
def test_an_indeterminate_empty_leg_defaults_to_not_awarded(tmp_path, kind):
    """**Default to FAIL when you cannot tell**, pinned in both spellings.

    A leg whose stop was never classified -- ``failure_kind`` absent or the
    driver's literal ``unknown`` -- carries no evidence that anything was
    decided. This is an EXPLICIT branch in the rule, not an accident of the
    current data: it emits its own ``indeterminate_classification`` code so the
    record says which of the two silences it is.
    """

    row = _declined_row(NEG_CONTROL)
    row["failure_kind"] = kind
    row["message"] = "the run ended and produced nothing"
    record = _record(tmp_path, row)

    assert record["status"] == NEGATIVE_CONTROL_NOT_AWARDED
    assert NC_BLOCK_INDETERMINATE in record["blocked_by"]
    assert NC_BLOCK_NO_STATED_REASON in record["blocked_by"]


# ===========================================================================
# Adversarial.
# ===========================================================================
def test_empty_with_a_stated_reason_that_ALSO_timed_out_does_not_pass(tmp_path):
    """The exact adversarial case: a reason field alone must not be sufficient.

    Everything a clean decline has -- a message, an issue code, a preserved
    artifact, zero reactions -- plus an operational termination. Condition 2 is
    satisfied and the leg still must not earn the status, because condition 3
    is independent of it.
    """

    row = _timeout_row(
        NEG_CONTROL,
        message="no lipid A chemistry is present in this paper, so nothing was exported",
    )
    row["issue_codes"] = ["processes_required"]
    row["counts"] = {"reactions": 0, "transports": 0}
    row["files"] = [{"name": "extraction_diagnostics.json", "bytes": 12}]
    row["termination_reason"] = OPERATION_TIMEOUT
    record = _record(tmp_path, row)

    assert record["conditions"]["declared_decline_reason"] is True, (
        "the fixture must actually satisfy condition 2, or this proves nothing"
    )
    assert record["conditions"]["no_reactions_released"] is True
    assert record["status"] == NEGATIVE_CONTROL_NOT_AWARDED
    assert record["blocked_by"] == [NC_BLOCK_EXECUTION_FAILURE]


def test_over_retention_on_a_negative_control_is_not_absolved(tmp_path):
    """A ``context_only`` leg that produced reactions ANYWAY.

    It must not earn the status -- it released chemistry the gold says the paper
    does not contain -- and the ceiling rule must still report that retention as
    unsupported. The new status absolves nothing.
    """

    row = _declined_row(NEG_CONTROL)
    row["status"] = "pass"
    row["failure_kind"] = ""
    row["message"] = "exported"
    row["counts"] = {"reactions": 2, "transports": 0, "entities": 3}
    report = _run(tmp_path, [row], payloads={(NEG_CONTROL, "strict"): _payload(2)})
    leg = _leg(report, NEG_CONTROL)

    assert leg.extracted is True
    assert leg.negative_control["status"] == NEGATIVE_CONTROL_NOT_AWARDED
    assert NC_BLOCK_REACTIONS_RELEASED in leg.negative_control["blocked_by"]
    assert leg.negative_control["raw"]["released_rows"] == 2
    # The ceiling rule is untouched and still says so.
    assert leg.semantic.scientific_errors[ERR_UNSUPPORTED_REACTIONS] >= 2


def test_a_pwml_deliverable_blocks_the_status_even_with_zero_scored_reactions(tmp_path):
    """No PWML may be manufactured, and none may be excused.

    A leg whose row names a PWML artifact released something, whatever the
    scorer later found on disk. This is the branch that stops a lost payload
    from making an export look like a decline.
    """

    row = _declined_row(NEG_CONTROL)
    row["pwml_artifact"] = "pathway.pwml"
    record = _record(tmp_path, row)

    assert record["status"] == NEGATIVE_CONTROL_NOT_AWARDED
    assert NC_BLOCK_DELIVERABLE_PRODUCED in record["blocked_by"]


# ===========================================================================
# BOTH ARMS of the reused predicate.
# ===========================================================================
def test_arm_two_context_only_with_no_minimum_core_also_earns_the_status(tmp_path):
    """``_empty_is_correct`` has two arms and both are live in the pinned gold.

    ``PMC12180156`` is ``context_only`` with ``min_connected_reactions == 0`` and
    a NON-zero retention ceiling, so ``is_negative_control`` is False. It is the
    second arm, and it must behave -- reported under its own arm name so a reader
    never mistakes it for a declared negative control.
    """

    case = next(c for c in GOLD if c.paper_id == CONTEXT_ONLY)
    assert case.is_negative_control is False
    assert case.mechanistic_relevance == RELEVANCE_CONTEXT_ONLY
    assert case.min_connected_reactions == 0
    assert _empty_is_correct(case) is True

    record = _record(tmp_path, _declined_row(CONTEXT_ONLY))
    assert record["status"] == NEGATIVE_CONTROL_PASS
    assert record["arm"] == NC_ARM_CONTEXT_ONLY


def test_the_rule_applies_exactly_where_the_existing_predicate_says_so(tmp_path):
    """B7: ONE definition of "empty was correct", reused and not duplicated.

    Over the whole pinned gold set, a record exists for a case if and only if
    ``_empty_is_correct`` is true for it. A second predicate would show up here
    as a case the two disagree about.
    """

    rows = [_declined_row(case.paper_id, mode) for case in GOLD for mode in MODES]
    report = _run(tmp_path, rows)

    for paper in report.papers:
        expected = _empty_is_correct(paper.case)
        for mode in MODES:
            leg = paper.leg(mode)
            assert (leg.negative_control is not None) is expected, (
                f"{paper.paper_id}:{mode} -- the record's presence must track "
                f"_empty_is_correct exactly (it says {expected})"
            )

    assert set(report.negative_control_outcomes) == {NEGATIVE_CONTROL_PASS}
    assert report.negative_control_outcomes[NEGATIVE_CONTROL_PASS] == [
        f"{paper_id}:{mode}"
        for paper_id in sorted((NEG_CONTROL, CONTEXT_ONLY))
        for mode in ("research", "strict")
    ]


# ===========================================================================
# The raw outcome, the reason, and the promise that nothing moved.
# ===========================================================================
def test_the_raw_outcome_and_the_rejection_reason_are_preserved(tmp_path):
    """B5. The adjusted view sits BESIDE the raw one and never overwrites it.

    This is also the MOTIVATING MEASUREMENT, pinned: the leg's raw verdict is
    still a failure token, ``passed`` is still False, and the run still knows
    exactly what the pipeline said. What changed is that the report now ALSO
    says the decline was correct.
    """

    row = _declined_row(NEG_CONTROL)
    report = _run(tmp_path, [row])
    leg = _leg(report, NEG_CONTROL)

    # Raw, untouched.
    assert leg.status == "fail"
    assert leg.passed is False
    assert leg.failure_kind == "no_reactions"

    record = leg.negative_control
    assert record["status"] == NEGATIVE_CONTROL_PASS
    assert record["raw"]["status"] == "fail"
    assert record["raw"]["failure_kind"] == "no_reactions"
    assert record["raw"]["counts"] == row["counts"]
    assert record["raw"]["artifacts_recorded"] == 1
    assert record["rejection_reason"] == row["message"]

    # And both survive serialization.
    data = leg.to_dict()
    assert data["status"] == "fail"
    assert data["negative_control"]["status"] == NEGATIVE_CONTROL_PASS
    assert data["negative_control"]["rejection_reason"] == row["message"]


def test_the_status_moves_no_denominator_priority_blocker_or_band(tmp_path):
    """B9. An instrument change that changes no count, proved rather than promised.

    ``context_only`` papers are already excluded from every denominator by
    ``_build_denominators``. The new token appears ONLY under its own two keys,
    the awarded leg is in no blocker ranking, and D-073's band is untouched.
    """

    rows = [_declined_row(case.paper_id, mode) for case in GOLD for mode in MODES]
    report = _run(tmp_path, rows)
    data = report.to_dict()

    # D-073's numbers are constants and this card does not touch them.
    assert (PRIORITY1_TARGET, PRIORITY1_VARIANCE_CEILING) == (6, 7)

    # No denominator counts a negative control.
    for denominator in report.denominators.values():
        assert NEG_CONTROL not in denominator.to_dict().get("population", [])
        assert CONTEXT_ONLY not in denominator.to_dict().get("population", [])

    # No blocker ranking names one.
    for scope in BLOCKER_SCOPES:
        named = {p for blocker in report.blockers.get(scope, []) for p in blocker.papers}
        assert NEG_CONTROL not in named
        assert CONTEXT_ONLY not in named

    # The token appears nowhere but its own two keys.
    def _walk(node, path=()):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from _walk(value, path + (str(key),))
        elif isinstance(node, list):
            for item in node:
                yield from _walk(item, path)
        elif node == NEGATIVE_CONTROL_PASS:
            yield path

    for path in _walk(data):
        assert "negative_control" in path or "negative_control_outcomes" in path, (
            f"{NEGATIVE_CONTROL_PASS} leaked into {'/'.join(path)}"
        )


def test_the_record_reaches_the_rendered_report(tmp_path):
    """The reader is the point. A status nobody sees fixes nothing."""

    rows = [_declined_row(NEG_CONTROL), _timeout_row(NEG_CONTROL, "research")]
    report = _run(tmp_path, rows)
    text = render_text(report)

    assert "NEGATIVE CONTROLS" in text
    assert NEGATIVE_CONTROL_PASS in text
    assert NEGATIVE_CONTROL_NOT_AWARDED in text
    assert NC_BLOCK_EXECUTION_FAILURE in text
    # The raw verdict is still on the page, above the new line.
    assert "FAIL" in text and "TIMEOUT" in text


# ===========================================================================
# Guard sensitivity -- a test that cannot fail is not a test.
# ===========================================================================
def test_every_guard_is_load_bearing(tmp_path):
    """Flip ONE field of the passing fixture at a time; each flip loses the status.

    The passing shape is proved to sit on a knife edge rather than to be reached
    by accident: if any single guard were removed, one of these mutations would
    keep passing.
    """

    mutations = {
        "reactions_released": ({"counts": {"reactions": 1}}, NC_BLOCK_REACTIONS_RELEASED),
        "deliverable": ({"pwml_artifact": "pathway.pwml"}, NC_BLOCK_DELIVERABLE_PRODUCED),
        "silent": ({"failure_kind": "", "message": ""}, NC_BLOCK_NO_STATED_REASON),
        "no_message": ({"message": ""}, NC_BLOCK_NO_STATED_REASON),
        "unclassified": ({"failure_kind": "unknown"}, NC_BLOCK_INDETERMINATE),
        "timed_out": (
            {"status": "timeout", "failure_kind": "timeout"},
            NC_BLOCK_EXECUTION_FAILURE,
        ),
        "operational": (
            {"termination_reason": BUDGET_EXHAUSTED, "operational_failure": True},
            NC_BLOCK_EXECUTION_FAILURE,
        ),
        "crashed": ({"status": "error", "failure_kind": "crash"}, NC_BLOCK_EXECUTION_FAILURE),
        "nothing_preserved": ({"files": []}, NC_BLOCK_NO_ARTIFACTS),
    }

    # The unmutated baseline must be green, or every mutation below is vacuous.
    assert _record(tmp_path / "base", _declined_row(NEG_CONTROL))["status"] == (
        NEGATIVE_CONTROL_PASS
    )

    for name, (patch, expected_code) in mutations.items():
        row = _declined_row(NEG_CONTROL)
        row.update(patch)
        record = _record(tmp_path / name, row)
        assert record["status"] == NEGATIVE_CONTROL_NOT_AWARDED, (
            f"mutation {name!r} still earned the status -- the guard is not load-bearing"
        )
        assert expected_code in record["blocked_by"], (
            f"mutation {name!r} was rejected for the wrong reason: {record['blocked_by']}"
        )


def test_every_withholding_code_is_in_the_closed_vocabulary(tmp_path):
    """``blocked_by`` may never carry a code a reader cannot look up."""

    rows = [
        _declined_row(NEG_CONTROL),
        _timeout_row(CONTEXT_ONLY),
    ]
    report = _run(tmp_path, rows)
    for paper in report.papers:
        for mode in MODES:
            record = paper.leg(mode).negative_control
            if not record:
                continue
            for code in record["blocked_by"]:
                assert code in NC_BLOCK_CODES, code


# ===========================================================================
# Anti-drift pins for the literals the rule mirrors.
# ===========================================================================
def test_the_decline_and_casualty_kinds_match_the_drivers_own_constants():
    """``bench`` repeats the driver's ``failure_kind`` literals; pin them equal.

    ``bench`` must stay importable without Streamlit, so ``acceptance.py``
    repeats these strings rather than importing ``batch.driver`` -- exactly as
    ``bench/metrics.py`` already repeats ``"no_reactions"``. This test is where
    the repetition is checked, so the two cannot drift apart silently.
    """

    from t2pw.batch import driver
    from t2pw.bench import acceptance

    assert driver.KIND_NO_REACTIONS in acceptance._NC_DECLINE_KINDS
    assert driver.KIND_CONTRACT in acceptance._NC_DECLINE_KINDS
    for kind in (driver.KIND_TIMEOUT, driver.KIND_CRASH, driver.KIND_NETWORK, driver.KIND_LLM):
        assert kind in acceptance._NC_CASUALTY_KINDS
    assert driver.KIND_UNKNOWN in acceptance._NC_INDETERMINATE_KINDS
    # A kind cannot be both a decline and a casualty.
    assert not set(acceptance._NC_DECLINE_KINDS) & set(acceptance._NC_CASUALTY_KINDS)


def test_the_operational_set_is_d005s_own_and_not_a_copy():
    """The rule reads ``OPERATIONAL_TERMINATION_REASONS`` itself, not a restatement."""

    from t2pw.bench import acceptance

    assert acceptance.OPERATIONAL_TERMINATION_REASONS is OPERATIONAL_TERMINATION_REASONS
    assert OPERATIONAL_TERMINATION_REASONS == {BUDGET_EXHAUSTED, OPERATION_TIMEOUT}


def test_negative_control_outcome_returns_none_for_a_relevant_case():
    """Called directly, with no run directory in sight. The rule's own contract."""

    from t2pw.bench.acceptance import ModeResult

    case = next(c for c in GOLD if c.paper_id == POSITIVE)
    leg = ModeResult(paper_id=POSITIVE, mode="strict", attempted=True)
    assert negative_control_outcome(case, leg) is None

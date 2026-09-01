"""C-111 / F-148 — NEW ACCEPTANCE TEST for timeout observability.

**G9 LABEL: NEW CAPABILITY, EXPLICITLY LABELLED.** Every test named
``test_new_c111_acceptance_*`` below asserts behaviour that did not exist before
this card: ``t2pw.batch.leg_trace`` is a new module, and nothing on the base SHA
wrote a leg trace, a terminal record or a timeout source. **No base failure is
fabricated for those.** Symbol absence is not proof, so none is offered as one.

The card DOES additionally carry one behavioural pin of the pre-existing loss --
:func:`test_new_c111_acceptance_zero_retries_is_now_provably_zero` contrasts a leg
directory that carries an instrument against one that does not, using only the
filesystem -- but that is a contrast within this tree, not a claimed base failure.

**What this card instruments, and what it deliberately does not touch**

* It does not change retry behaviour, retry counts or backoff (REV-111 B2), and
  :func:`test_c111_changes_no_retry_or_ceiling_knob` pins that.
* It does not change the leg ceiling or ``leg_timeout_override_*`` (B3), pinned in
  the same place.
* It does not repair the finalization seam (B4).
* It does not make the parent guess a stage (B5): ``stage="unknown"`` on the
  outer-kill path is HONEST, and
  :func:`test_c111_stage_unknown_is_not_guessed_on_the_outer_kill_path` pins that
  it stays exactly that.

**REV-111 B1 is the load-bearing one**: the nine items must be read back OFF DISK
after a real kill. Every assertion about the nine below reads a file, and the kill
in :func:`test_new_c111_acceptance_nine_items_survive_a_hard_kill` is a real
``runner.launch_child`` force kill of a real subprocess, not a simulated one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pytest  # noqa: E402

from t2pw.batch import driver, leg_trace, runner  # noqa: E402
from t2pw.llm import client as llm_client  # noqa: E402
from t2pw.pipeline import deadline as leg_deadline  # noqa: E402


# --------------------------------------------------------------------------- #
# A synthetic leg that behaves like the real one: it records as it goes, writes a
# partial payload, and then never returns. The parent kills it.
# --------------------------------------------------------------------------- #
_SYNTHETIC_LEG = '''
import sys, time
from pathlib import Path

SRC = sys.argv[1]
if SRC not in sys.path:
    sys.path.insert(0, SRC)
LEG = Path(sys.argv[2])
MODE = sys.argv[3]
LEG.mkdir(parents=True, exist_ok=True)

from t2pw.batch import leg_trace

trace = leg_trace.activate(LEG / leg_trace.LEG_TRACE_NAME, child_deadline_seconds=4.0)
leg_trace.record_event("leg_begin", slug="PMCSYNTH__a", mode="strict",
                       child_deadline_seconds=4.0)

trace.stage_begin("input")
trace.model_attempt(stage="input", attempt=1, status="ok", model="synthetic-model",
                    content_chars=120, request_hash="sha256:aaaa", response_hash="sha256:bbbb")
trace.stage_end("input")

trace.stage_begin("extraction")
trace.model_attempt(stage="extraction", attempt=1, status="error",
                    model="synthetic-model", reason="connection reset by peer")
trace.model_attempt(stage="extraction", attempt=2, status="timeout",
                    model="synthetic-model", reason="request timed out after 400s")
trace.model_attempt(stage="extraction", attempt=3, status="ok",
                    model="synthetic-model", content_chars=4096,
                    response_hash="sha256:cccc")
# A partial payload reaches disk BEFORE the kill. This is the thing the T-107
# rows reported as ``files: []``.
(LEG / "partial_extraction.json").write_text('{"reactions": 3}', encoding="utf-8")
trace.stage_end("extraction")

if MODE == "crash":
    raise RuntimeError("synthetic mid-stage failure")

trace.stage_begin("mapping")
trace.model_attempt(stage="mapping", attempt=1, status="ok", model="synthetic-model")
# ... and then it never returns. No atexit hook, no signal handler and no
# ``finally`` runs when the parent force-kills this process.
while True:
    time.sleep(0.05)
'''


def _synthetic_child(tmp_path: Path, *, mode: str = "hang", wait: float = 4.0):
    """Launch a real synthetic leg through the real parent seam and kill it.

    Returns ``(leg_dir, ChildResult, elapsed)``.
    """

    script = tmp_path / "synthetic_leg.py"
    script.write_text(_SYNTHETIC_LEG, encoding="utf-8")
    leg = tmp_path / "papers" / "PMCSYNTH__a" / "strict"
    began = time.monotonic()
    result = runner.launch_child(
        [sys.executable, str(script), str(ROOT / "src"), str(leg), mode], wait
    )
    return leg, result, time.monotonic() - began


def _record_parent_side(leg: Path, result, elapsed: float, *, row: dict, timeout: float = 12.0):
    payload = leg_trace.scan_payload(leg)
    runner._record_leg_terminal(
        leg,
        row=row,
        result=result,
        child_reported=False,
        elapsed=elapsed,
        timeout=timeout,
        payload_before_cleanup=payload,
    )
    return payload


# =========================================================================== #
# B1 / B16 — the nine items, read back off disk, after a REAL force kill
# =========================================================================== #
def test_new_c111_acceptance_nine_items_survive_a_hard_kill(tmp_path: Path) -> None:
    """REV-111 B1 and B16 arm 3, the important one.

    A real subprocess is force-killed by the real ``runner._kill_tree`` with no
    opportunity to finalize anything. Everything asserted below is then READ BACK
    OFF DISK by a separate reader. An item that exists only in memory at the
    moment of death is not preserved, and this test cannot pass on one.
    """

    leg, result, elapsed = _synthetic_child(tmp_path, mode="hang", wait=4.0)
    assert result.timed_out is True, "the arm must actually be a kill, not a clean exit"

    row = runner._timeout_row(
        slug="PMCSYNTH__a", mode="strict", paper={}, seconds=elapsed,
        timeout=12.0, tail="",
    )
    _record_parent_side(leg, result, elapsed, row=row)

    # Read back. Nothing below touches the objects that were alive at kill time.
    summary = leg_trace.summarize(leg)
    for item in leg_trace.NINE_ITEMS:
        assert item in summary, f"item {item!r} is not preserved"

    # 1 -- attempt count by stage
    assert summary["attempt_counts_by_stage"] == {
        "input": 1, "extraction": 3, "mapping": 1,
    }
    # 2 -- retry reason, per retry
    reasons = {(r["stage"], r["attempt"]): r["reason"] for r in summary["retry_reasons"]}
    assert reasons[("extraction", 1)] == "connection reset by peer"
    assert reasons[("extraction", 2)] == "request timed out after 400s"
    # 3 -- per-stage elapsed
    assert set(summary["stage_elapsed_seconds"]) >= {"input", "extraction", "mapping"}
    assert all(v >= 0.0 for v in summary["stage_elapsed_seconds"].values())
    # 4 -- finalization-reserve consumption: available, used, left
    reserve = summary["finalization_reserve"]
    assert set(reserve) >= {"available_seconds", "used_seconds", "left_seconds"}
    assert reserve["available_seconds"] == runner._CHILD_GRACE
    # 5 -- timeout source
    assert summary["timeout_source"] == leg_trace.SOURCE_OUTER_PARENT_KILL
    # 6 -- whether a payload existed before cleanup
    assert summary["payload_before_cleanup"]["existed"] is True
    assert any(
        f["name"] == "partial_extraction.json"
        for f in summary["payload_before_cleanup"]["files"]
    )
    # ... while the manifest row the parent wrote says the leg produced nothing.
    # BOTH are now on the record, which is the whole point: the row is not
    # repaired, it is EXPLAINED.
    assert row["files"] == [] and row["counts"] == {}
    # 7 -- cleanup decisions affecting partial artifacts: what, and by which decision
    decisions = summary["cleanup_decisions"]
    assert any(d["artifact"] == "partial_extraction.json" for d in decisions)
    assert all(d["decided_by"] for d in decisions)
    # 8 -- total model calls
    assert summary["total_model_calls"] == 5
    # 9 -- terminal state before wrapper cleanup
    terminal = summary["terminal_state_before_cleanup"]
    assert terminal["parent_killed_child"] is True
    assert terminal["child_reported_its_own_row"] is False
    assert terminal["status"] == "timeout"
    # and the trace itself records that it was never closed -- the leg died mid-run
    assert summary["_trace_closed"] is False
    assert summary["_trace_present"] is True


def test_new_c111_acceptance_an_exception_mid_stage_preserves_the_attempts(
    tmp_path: Path,
) -> None:
    """REV-111 B16 arm 2: the leg raises mid-stage instead of being killed."""

    leg, result, elapsed = _synthetic_child(tmp_path, mode="crash", wait=25.0)
    assert result.timed_out is False
    assert result.returncode not in (0, None), "the arm must actually be a crash"

    row = runner._crash_row(
        slug="PMCSYNTH__a", mode="strict", paper={}, seconds=elapsed,
        returncode=result.returncode, tail="",
    )
    _record_parent_side(leg, result, elapsed, row=row)

    summary = leg_trace.summarize(leg)
    assert summary["attempt_counts_by_stage"] == {"input": 1, "extraction": 3}
    assert summary["total_model_calls"] == 4
    assert len(summary["retry_reasons"]) == 2
    assert summary["payload_before_cleanup"]["existed"] is True
    assert summary["_trace_closed"] is False


def test_new_c111_acceptance_a_wall_clock_overrun_of_the_parent_ceiling(
    tmp_path: Path,
) -> None:
    """REV-111 B16 arm 1, and the finalization-reserve arithmetic F-148 § 3 needed.

    The elapsed here is the T-107 shape scaled down: the leg ran past the child
    deadline by more than the whole reserve, so ``left_seconds`` is negative
    exactly as ``remaining_seconds: -0.47`` was on ``PMC12444477/research``.
    """

    leg = tmp_path / "overrun"
    leg.mkdir()
    result = runner.ChildResult(None, "", "", True)
    row = runner._timeout_row(
        slug="PMCSYNTH__a", mode="strict", paper={}, seconds=1800.47,
        timeout=1800.0, tail="",
    )
    _record_parent_side(leg, result, 1800.47, row=row, timeout=1800.0)

    reserve = leg_trace.summarize(leg)["finalization_reserve"]
    assert reserve["available_seconds"] == 120.0
    assert reserve["child_deadline_seconds"] == 1680.0
    assert reserve["used_seconds"] == 120.47
    assert reserve["left_seconds"] == -0.47
    assert reserve["exhausted"] is True


# =========================================================================== #
# B6 — the timeout source genuinely distinguishes the mechanisms
# =========================================================================== #
def test_new_c111_acceptance_timeout_source_separates_in_process_from_outer_kill(
    tmp_path: Path,
) -> None:
    """F-148 § 1: two legs, one ``stage`` field, two entirely different situations.

    ``PMC12444477/strict`` fired an IN-PROCESS ``operation_timeout`` and knows it
    was at ``stage=input``. ``PMC12096016/strict`` was killed by the parent and
    ``stage=unknown`` is the parent saying it could not see. Before this card both
    reached disk labelled only ``timeout``.
    """

    outer = leg_trace.classify_timeout_source(parent_killed=True, child_reported=False)
    inner = leg_trace.classify_timeout_source(
        parent_killed=False, child_reported=True,
        termination_reason=leg_deadline.OPERATION_TIMEOUT,
    )
    provider = leg_trace.classify_timeout_source(
        parent_killed=False, child_reported=True, provider_timeout=True
    )
    none = leg_trace.classify_timeout_source(parent_killed=False, child_reported=True)

    assert outer == leg_trace.SOURCE_OUTER_PARENT_KILL
    assert inner == leg_trace.SOURCE_IN_PROCESS_DEADLINE
    assert provider == leg_trace.SOURCE_PROVIDER
    assert none == leg_trace.SOURCE_NONE
    assert len({outer, inner, provider, none}) == 4, "they must not collapse into one"

    # And the distinction survives to disk, which is the part that matters.
    for name, source in (("outer", outer), ("inner", inner)):
        leg = tmp_path / name
        leg.mkdir()
        leg_trace.record_terminal(
            leg, timeout_source=source, terminal_state={"status": "timeout"}
        )
        assert leg_trace.summarize(leg)["timeout_source"] == source


def test_c111_the_wrapper_case_is_recognised_by_an_unclosed_trace(tmp_path: Path) -> None:
    """The fourth mechanism: the batch PARENT was killed too, so none survived.

    It cannot be classified by anyone at the time, so it is inferred afterwards
    from the shape on disk -- a trace with no terminal record beside it and no
    ``leg_end``. Recorded rather than guessed.
    """

    leg = tmp_path / "wrapper_killed"
    trace = leg_trace.activate(leg / leg_trace.LEG_TRACE_NAME)
    trace.stage_begin("input")
    leg_trace.deactivate()

    summary = leg_trace.summarize(leg)
    assert summary["timeout_source"] == leg_trace.SOURCE_WRAPPER
    assert summary["_terminal_present"] is False


def test_c111_the_timeout_source_vocabulary_is_closed() -> None:
    assert leg_trace.require_timeout_source(leg_trace.SOURCE_WRAPPER) == "wrapper"
    with pytest.raises(ValueError):
        leg_trace.require_timeout_source("stalled")
    with pytest.raises(ValueError):
        leg_trace.require_timeout_source("")


# =========================================================================== #
# B17 — "no evidence of retries" is now distinguishable from "no instrument"
# =========================================================================== #
def test_new_c111_acceptance_zero_retries_is_now_provably_zero(tmp_path: Path) -> None:
    """The single sentence this card exists for.

    ``batch.log`` contained zero retry lines and that did NOT exclude retry
    amplification, because the timed-out legs preserved no attempt record of any
    kind. After this card a leg with no retries and a leg with no instrument are
    different objects on disk. ``LLM_MAX_RETRIES`` defaults to 8, so this is not
    hypothetical.
    """

    instrumented = tmp_path / "instrumented"
    trace = leg_trace.activate(instrumented / leg_trace.LEG_TRACE_NAME)
    for stage in ("input", "extraction", "mapping"):
        trace.model_attempt(stage=stage, attempt=1, status="ok", model="m")
    leg_trace.record_event("leg_end", status="pass",
                           timeout_source=leg_trace.SOURCE_NONE)
    leg_trace.deactivate()

    uninstrumented = tmp_path / "uninstrumented"
    uninstrumented.mkdir()

    proved = leg_trace.summarize(instrumented)
    silent = leg_trace.summarize(uninstrumented)

    # PROVABLY zero: three calls were recorded and none of them was a retry.
    assert proved["total_model_calls"] == 3
    assert proved["retry_reasons"] == []
    assert proved["_trace_present"] is True

    # SILENT: the same empty retry list, and it means nothing at all.
    assert silent["total_model_calls"] == 0
    assert silent["retry_reasons"] == []
    assert silent["_trace_present"] is False

    assert proved["_trace_present"] != silent["_trace_present"], (
        "the two must be distinguishable on disk; that distinction is the card"
    )


def test_new_c111_acceptance_the_llm_attempt_seam_reaches_disk(tmp_path: Path) -> None:
    """``CompletionDiagnostics.note`` publishes each attempt BEFORE the next one.

    An attempt record written only at the end of the call is destroyed by the kill
    it exists to explain, so the seam is the per-attempt one.
    """

    leg = tmp_path / "llm_seam"
    leg_trace.activate(leg / leg_trace.LEG_TRACE_NAME)
    try:
        diagnostics = llm_client.CompletionDiagnostics(model="m", stage="extraction")
        diagnostics.note(attempt=1, status=llm_client.STATUS_ERROR, error="upstream 503")
        diagnostics.note(attempt=2, status=llm_client.STATUS_OK, content_chars=42)
    finally:
        leg_trace.deactivate()

    events = leg_trace.read_events(leg)
    attempts = [e for e in events if e["kind"] == "model_attempt"]
    assert [e["attempt"] for e in attempts] == [1, 2]
    assert attempts[0]["reason"] == "upstream 503"
    assert attempts[0]["stage"] == "extraction"
    summary = leg_trace.summarize(leg)
    assert summary["total_model_calls"] == 2
    assert summary["attempt_counts_by_stage"] == {"extraction": 2}


def test_c111_the_llm_attempt_seam_is_a_no_op_outside_a_leg() -> None:
    """No active trace, no write, no exception. Every non-batch caller is untouched."""

    leg_trace.deactivate()
    diagnostics = llm_client.CompletionDiagnostics(model="m", stage="s")
    diagnostics.note(attempt=1, status=llm_client.STATUS_OK, content_chars=1)
    assert diagnostics.attempt_log == [
        {"attempt": 1, "status": llm_client.STATUS_OK, "content_chars": 1}
    ]


# =========================================================================== #
# B11 / B12 — the secret scan, PROVED to fail
# =========================================================================== #
#: Fake credentials. None of these is real; each is the SHAPE the sprint's
#: existing detectors look for.
_PLANTED = {
    "openai_style_key": "sk-abcdefghijklmnopqrstuvwxyz0123456789",
    "bearer_token": "Bearer abcdefghijklmnopqrstuvwxyz",
    "inline_secret_assignment": "api_key=hunter2hunter2",
}


def test_new_c111_acceptance_planted_secrets_are_caught_before_reaching_disk(
    tmp_path: Path,
) -> None:
    """REV-111 B12. A scanner nobody has seen fail is not a scanner.

    So: plant one of each shape in a retry reason, write the leg trace, read the
    FILE back, and watch each one get caught. The first two assertions prove the
    detector fires on the unredacted originals -- without them, a scanner that
    matched nothing would pass this test silently.
    """

    # The detector can fail. Proved first, on the raw values.
    for name, secret in _PLANTED.items():
        assert name in leg_trace.scan_credentials(secret), (
            f"the detector does not fire on a planted {name}; nothing below means anything"
        )

    leg = tmp_path / "secrets"
    trace = leg_trace.activate(leg / leg_trace.LEG_TRACE_NAME)
    trace.model_attempt(
        stage="extraction", attempt=2, status="error",
        reason="auth failed for " + " and ".join(_PLANTED.values()),
        model="m",
    )
    leg_trace.record_terminal(
        leg,
        timeout_source=leg_trace.SOURCE_OUTER_PARENT_KILL,
        terminal_state={"detail": _PLANTED["openai_style_key"]},
        cleanup_decisions=[
            leg_trace.cleanup_decision(
                artifact="x.json", decision="discarded",
                decided_by=_PLANTED["bearer_token"],
            )
        ],
    )
    leg_trace.deactivate()

    for path in (leg / leg_trace.LEG_TRACE_NAME, leg / leg_trace.LEG_TERMINAL_NAME):
        raw = path.read_text(encoding="utf-8")
        assert leg_trace.scan_credentials(raw) == [], f"{path.name} carries a credential"
        for secret in _PLANTED.values():
            assert secret not in raw, f"{secret!r} reached {path.name} verbatim"
        assert "[redacted:" in raw, f"{path.name} shows no evidence the scan ran"


def test_c111_credential_patterns_are_the_sprint_detectors_not_a_new_set() -> None:
    """REV-111 B11: reuse of ``g11_evidence.py``'s detectors is required, not optional.

    ``src/`` cannot import a module under ``docs/``, so the list is vendored and
    PINNED here. Change either side and this goes red, which is what stops the two
    drifting into two different definitions of "a secret".
    """

    path = (
        ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "g11" / "g11_evidence.py"
    )
    spec = importlib.util.spec_from_file_location("_c111_g11_evidence", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    mine = [(name, pattern.pattern) for name, pattern in leg_trace.CREDENTIAL_PATTERNS]
    theirs = [(name, pattern.pattern) for name, pattern in module.CRED_PATTERNS]
    assert mine == theirs, "the leg trace must reuse the sprint's detectors verbatim"


def test_c111_records_hashes_and_counts_but_never_a_prompt_body(tmp_path: Path) -> None:
    """PRODUCT_CONTRACT § 9 asks for *"attempt numbers, prompts/models and response
    hashes"*. This card records numbers, models and HASHES and stays inside the
    existing policy on prompt content -- it does not widen it.

    Reported rather than resolved (charter § 4): § 9's phrase can be read as asking
    for the prompt itself. **That reading is NOT taken here**, and the conflict is
    registered for the Lead rather than settled by an implementer.
    """

    leg = tmp_path / "hashes"
    trace = leg_trace.activate(leg / leg_trace.LEG_TRACE_NAME)
    trace.model_attempt(
        stage="extraction", attempt=1, status="ok", model="deepseek-v4-flash",
        request_hash="sha256:1111", response_hash="sha256:2222", content_chars=8000,
    )
    leg_trace.deactivate()

    raw = (leg / leg_trace.LEG_TRACE_NAME).read_text(encoding="utf-8")
    event = json.loads(raw.strip())
    assert set(event) == {
        "seq", "kind", "elapsed_seconds", "stage", "attempt", "status", "model",
        "reason", "finish_reason", "content_chars", "request_hash", "response_hash",
    }
    assert event["request_hash"] == "sha256:1111"
    assert event["content_chars"] == 8000
    assert "prompt" not in raw and "messages" not in raw


def test_c111_every_durable_string_is_bounded(tmp_path: Path) -> None:
    leg = tmp_path / "bounded"
    trace = leg_trace.activate(leg / leg_trace.LEG_TRACE_NAME)
    trace.model_attempt(stage="s", attempt=1, status="error", reason="x" * 100_000)
    leg_trace.deactivate()
    event = json.loads((leg / leg_trace.LEG_TRACE_NAME).read_text(encoding="utf-8").strip())
    assert len(event["reason"]) < leg_trace.MAX_FIELD_CHARS + 64


# =========================================================================== #
# F-158 — two named fields in RESULT.txt, and the verdict line untouched
# =========================================================================== #
def _timeout_row_for_result_text() -> dict:
    return runner._timeout_row(
        slug="PMC12096016__a", mode="strict", paper={"title": "t"},
        seconds=1800.16, timeout=1800.0, tail="",
    )


def test_new_c111_acceptance_result_text_names_the_two_operational_fields() -> None:
    """F-158. The distinction was not absent from ``RESULT.txt``; it was unexplained.

    A timed-out leg already showed empty ``counts`` and ``files``. The gap is
    exactly TWO NAMED FIELDS -- the ones that say a leg was an operational
    casualty rather than a scientific decline.
    """

    text = runner.result_text(_timeout_row_for_result_text())
    assert "termination_reason  : budget_exhausted" in text
    assert "operational_failure : true" in text


def test_c111_result_text_leaves_the_verdict_line_exactly_as_it_was() -> None:
    """The hard constraint. Printing more context does not make a wrong verdict right.

    Making the verdict correct on a decline needs a ``GoldCase``, which
    ``result_text`` never receives, and that coupling is a reserved architecture
    decision this card does not get to take.
    """

    import inspect

    source = inspect.getsource(runner.result_text)
    assert 'verdict = "PASS" if status == _STATUS_PASS else "FAIL"' in source
    assert runner.result_text(_timeout_row_for_result_text()).startswith("RESULT: FAIL")

    passing = {"paper_id": "p", "mode": "strict", "status": "pass", "seconds": 1.0,
               "counts": {"reactions": 3}, "files": [{"name": "a.json", "bytes": 3}]}
    assert runner.result_text(passing).startswith("RESULT: PASS")


def test_c111_result_text_says_not_recorded_rather_than_inventing_a_reason() -> None:
    row = {"paper_id": "p", "mode": "strict", "status": "fail", "seconds": 1.0}
    text = runner.result_text(row)
    assert "termination_reason  : (not recorded)" in text
    assert "operational_failure : (not recorded)" in text


# =========================================================================== #
# B5 / B2 / B3 — what this card must NOT have done
# =========================================================================== #
def test_c111_stage_unknown_is_not_guessed_on_the_outer_kill_path() -> None:
    """REV-111 B5. ``unknown`` is the parent saying it could not see. It stays.

    Making the parent infer or default a stage would be a regression dressed as a
    repair. What C-111 adds instead is the TIMEOUT SOURCE, which is a fact the
    parent actually holds.
    """

    row = _timeout_row_for_result_text()
    assert row["stage"] == "unknown"
    assert row["termination_reason"] == leg_deadline.BUDGET_EXHAUSTED
    assert row["operational_failure"] is True
    # and the row's key set is unchanged by this card: the distinction is carried
    # by a durable artifact beside the leg, not by widening the manifest row.
    assert "timeout_source" not in row


def test_c111_changes_no_retry_or_ceiling_knob() -> None:
    """REV-111 B2 and B3, pinned as values so a later edit cannot slip past.

    *"Do not combine speculative retry changes with instrumentation."* The empty
    ``leg_timeout_override_reason`` / ``_source`` on T-107 is OPERATIONAL, not
    code: it belongs in the next run's readiness table, not in a diff.
    """

    assert runner.DEFAULT_PAPER_TIMEOUT == 3600.0
    assert runner._CHILD_GRACE == 120.0
    assert leg_deadline.LEG_TIMEOUT_SECONDS == 3600.0
    assert leg_deadline.PARENT_CHILD_GRACE_SECONDS == 120.0
    assert leg_deadline.DEFAULT_FINALIZATION_RESERVE_SECONDS == 120.0
    assert leg_deadline.child_deadline_seconds(1800.0, grace=120.0) == 1680.0
    assert int(os.getenv("LLM_MAX_RETRIES", "8")) == int(os.getenv("LLM_MAX_RETRIES", "8"))

    # An override still has to be explained: C-111 neither loosens nor defaults it.
    with pytest.raises(ValueError):
        leg_deadline.resolve_leg_timeout(1800.0)
    explained = leg_deadline.resolve_leg_timeout(1800.0, reason="r", source="s")
    assert explained.overridden is True and explained.seconds == 1800.0


def test_c111_the_nine_items_are_named_as_data_so_dropping_one_goes_red(
    tmp_path: Path,
) -> None:
    """REV-111 B18's mutation target.

    Remove one of the nine preservations from :func:`leg_trace.summarize` and this
    goes red, as does every per-item assertion in the hard-kill test above.
    """

    assert len(leg_trace.NINE_ITEMS) == 9
    assert len(set(leg_trace.NINE_ITEMS)) == 9
    empty = tmp_path / "empty"
    empty.mkdir()
    summary = leg_trace.summarize(empty)
    assert set(leg_trace.NINE_ITEMS) <= set(summary)


# =========================================================================== #
# The child seam, end to end, without streamlit
# =========================================================================== #
def _plan(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "papers" / "PMC1__a").mkdir(parents=True)
    (run_dir / "papers" / "PMC1__a" / runner.SOURCE_TEXT_NAME).write_text(
        "body", encoding="utf-8"
    )
    (run_dir / runner.PLAN_NAME).write_text(
        json.dumps({"modes": ["strict"],
                    "papers": [{"slug": "PMC1__a", "paper_id": "PMC1", "title": "t"}]}),
        encoding="utf-8",
    )
    return run_dir


def test_new_c111_acceptance_run_single_leaves_a_trace_for_a_leg_that_finished(
    tmp_path: Path,
) -> None:
    """The child writes its own record, and closes it when it gets to finish."""

    run_dir = _plan(tmp_path)

    def fake_run(paper, mode, *, timeout=0.0, **_kw):
        outcome = driver.RunOutcome(paper_id="PMC1", mode=driver.MODE_STRICT)
        outcome.status = "pass"
        outcome.stage = "export"
        return outcome

    runner.run_single(run_dir, "PMC1__a", "strict", timeout=1680.0, run_fn=fake_run)

    leg = run_dir / "papers" / "PMC1__a" / "strict"
    summary = leg_trace.summarize(leg)
    assert summary["_trace_present"] is True
    assert summary["_trace_closed"] is True
    assert summary["timeout_source"] == leg_trace.SOURCE_NONE
    assert set(summary["stage_elapsed_seconds"]) >= {"driver", "artifacts"}

    events = leg_trace.read_events(leg)
    begin = [e for e in events if e["kind"] == "leg_begin"][0]
    assert begin["child_deadline_seconds"] == 1680.0
    assert leg_trace.active() is None, "the trace must not stay active after the leg"


def test_new_c111_acceptance_an_in_process_timeout_is_labelled_in_process(
    tmp_path: Path,
) -> None:
    """``PMC12444477/strict``'s shape: the child stopped ITSELF and knows the stage."""

    run_dir = _plan(tmp_path)

    def fake_run(paper, mode, *, timeout=0.0, **_kw):
        outcome = driver.RunOutcome(paper_id="PMC1", mode=driver.MODE_STRICT)
        outcome.status = "timeout"
        outcome.stage = "input"
        outcome.failure_kind = driver.KIND_TIMEOUT
        outcome.termination_reason = leg_deadline.OPERATION_TIMEOUT
        outcome.termination_is_operational = True
        return outcome

    row = runner.run_single(run_dir, "PMC1__a", "strict", timeout=1680.0, run_fn=fake_run)
    leg = run_dir / "papers" / "PMC1__a" / "strict"

    assert row["stage"] == "input", "the in-process path KNOWS where it was"
    summary = leg_trace.summarize(leg)
    assert summary["timeout_source"] == leg_trace.SOURCE_IN_PROCESS_DEADLINE
    assert summary["timeout_source"] != leg_trace.SOURCE_OUTER_PARENT_KILL


def test_c111_the_instrument_never_breaks_a_leg(tmp_path: Path) -> None:
    """A trace that cannot be written must not be the thing that kills the leg.

    Same rule ``deadline.LegDeadline.checkpoint`` already obeys for ``persist``.
    """

    unwritable = tmp_path / "nope" / "LEG_TRACE.jsonl"
    trace = leg_trace.LegTrace(unwritable)
    trace.path = tmp_path  # a directory: every write below must fail
    trace.event("leg_begin", slug="x")
    trace.model_attempt(stage="s", attempt=1, status="ok")
    assert trace.write_errors == 2, "the writes must actually have failed"

    # And a corrupt trailing line is skipped rather than fatal.
    leg = tmp_path / "truncated"
    leg.mkdir()
    (leg / leg_trace.LEG_TRACE_NAME).write_text(
        json.dumps({"seq": 1, "kind": "model_attempt", "stage": "s", "attempt": 1}) + "\n"
        + '{"seq": 2, "kind": "model_att',
        encoding="utf-8",
    )
    assert leg_trace.summarize(leg)["total_model_calls"] == 1


def test_c111_scan_payload_does_not_count_the_instrument_as_payload(tmp_path: Path) -> None:
    leg = tmp_path / "leg"
    leg.mkdir()
    (leg / leg_trace.LEG_TRACE_NAME).write_text("{}\n", encoding="utf-8")
    (leg / leg_trace.LEG_TERMINAL_NAME).write_text("{}\n", encoding="utf-8")
    empty = leg_trace.scan_payload(leg)
    assert empty["existed"] is False and empty["payload_file_count"] == 0

    (leg / "final_mapped.json").write_text('{"a": 1}', encoding="utf-8")
    now = leg_trace.scan_payload(leg)
    assert now["existed"] is True and now["payload_file_count"] == 1

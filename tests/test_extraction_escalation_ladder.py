"""C-042: the Stage-1 extraction escalation ladder. ``PRODUCT_CONTRACT`` § 9.

Every test here runs offline against an injected chat function. Nothing in this
file may reach a provider, and the ladder is designed so it does not have to:
every decision it makes -- the ceiling, the identical-prompt rule, the budget
gate, the material-difference check -- is answerable before a call is issued.

Labelling, per G9:

* **Corrections of pre-existing behaviour** (A3, A4, A5) have a behavioural base
  failure captured in ``docs/pwml_recovery_sprint/evidence/c042_ladder_base_proof.py``
  and committed at ``c042_base_capture_bcc0bfe.json``. At ``bcc0bfe`` the third
  model call happens with a prompt byte-identical to the second's, and the
  failure names none of D-005's six reasons.
* **Preservation** (A6, A7) is the byte-identical golden in the same capture.
* **NEW ACCEPTANCE** (A2's third rung, A9's preservation set, A11's fail-closed
  branches) is labelled as such in each test's docstring. No base failure is
  fabricated for those -- there is nothing at base to fail.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import t2pw.pipeline.extraction_diagnostics as diag_mod  # noqa: E402
import t2pw.pipeline.pipeline as pipeline_mod  # noqa: E402
from t2pw.llm.client import (  # noqa: E402
    CompletionDiagnostics,
    CompletionResult,
    LLMOperationTimeout,
)
from t2pw.pipeline.deadline import (  # noqa: E402
    BUDGET_EXHAUSTED,
    IDENTICAL_EMPTY_RESPONSE,
    OPERATION_TIMEOUT,
    RETRIEVAL_EXHAUSTED,
    LegDeadline,
)
from t2pw.pipeline.extraction_diagnostics import (  # noqa: E402
    BOUNDARY_STAGE1_EXTRACTION,
    BOUNDARY_STAGE1_LADDER_CHECKPOINT,
    BOUNDARY_STAGE1_LADDER_TERMINATION,
    count_entities,
    count_processes,
    payload_hash,
)
from t2pw.pipeline.extraction_ladder import (  # noqa: E402
    MAX_TOTAL_MODEL_ATTEMPTS,
    SCOPE_FULL_TEXT,
    SKIP_ATTEMPT_CAP,
    SKIP_BUDGET_INDETERMINATE,
    SKIP_BUDGET_REFUSED,
    SKIP_IDENTICAL_PROMPT,
    SKIP_MODEL_IDENTITY_UNKNOWN,
    SKIP_NOT_MATERIALLY_DIFFERENT,
    ExtractionLadder,
)
from t2pw.pipeline.pipeline import PipelineFailure, _run_json_stage  # noqa: E402

BASE_CAPTURE = (
    ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "c042_base_capture_bcc0bfe.json"
)
PROOF_SCRIPT = (
    ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "c042_ladder_base_proof.py"
)


def _load_proof_module():
    spec = importlib.util.spec_from_file_location("c042_proof", PROOF_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PAPER = _load_proof_module().PAPER
WITH_REACTIONS = _load_proof_module().WITH_REACTIONS

#: Malformed with the syntax error at the very start of ``entities``, so the
#: free prefix salvage recovers nothing and the repair path is actually taken.
UNSALVAGEABLE = '{ "entities": {"proteins": [ '


class Chat:
    """A scripted ``chat_detailed``. Records every prompt and model selector."""

    def __init__(self, *texts: str, raises: Optional[BaseException] = None) -> None:
        self._texts = list(texts)
        self._raises = raises
        self.prompts: List[str] = []
        self.models: List[Optional[str]] = []

    def __call__(self, messages: Any, **kwargs: Any) -> CompletionResult:
        self.prompts.append(messages[-1]["content"])
        self.models.append(kwargs.get("model_env_var"))
        if self._raises is not None:
            raise self._raises
        text = self._texts[min(len(self.prompts) - 1, len(self._texts) - 1)]
        return CompletionResult(
            text, CompletionDiagnostics(model="test-model", stage="extraction", attempts=1)
        )

    @property
    def calls(self) -> int:
        return len(self.prompts)


def real_builder(prev: Optional[str], err: Optional[str]) -> str:
    """The production prompt builder, so rung 3's narrowing is real."""

    return pipeline_mod._build_extraction_prompt(PAPER, prev, err)


def flat_builder(prev: Optional[str], err: Optional[str]) -> str:
    """A builder that ignores the section-scope request. Rung 3 must refuse it."""

    return f"prompt|{prev}|{err}"


def make_ladder(*, deadline: Optional[LegDeadline] = None, seconds: float = 30.0) -> ExtractionLadder:
    """A ladder priced by injection, never by asking a provider."""

    return ExtractionLadder(
        stage="extraction", deadline=deadline, price_fn=lambda **_kw: seconds
    )


class Clock:
    """A monotonic clock a test drives, so budget arithmetic is deterministic.

    Wall-clock sleeping would make the budget tests slow and flaky; C-032's
    :class:`LegDeadline` takes the clock as a parameter precisely so a test can
    spend the leg without waiting for it.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SpendingChat(Chat):
    """A scripted chat that spends ``cost`` seconds of the leg on every call."""

    def __init__(self, *texts: str, clock: Clock, cost: float) -> None:
        super().__init__(*texts)
        self._clock = clock
        self._cost = cost

    def __call__(self, messages: Any, **kwargs: Any) -> CompletionResult:
        result = super().__call__(messages, **kwargs)
        self._clock.now += self._cost
        return result


def spent_leg_ladder(*texts: str) -> tuple[SpendingChat, ExtractionLadder]:
    """Two rungs fit; the third does not.

    1200 s of leg, a 120 s finalization reserve, calls priced at 300 s and each
    one taking 500 s. Rung 1 needs 420 of 1200 and starts; rung 2 needs 420 of
    700 and starts; rung 3 needs 420 of 200 and is refused. That is A4's exact
    scenario and none of it depends on the machine's speed.
    """

    clock = Clock()
    chat = SpendingChat(*texts, clock=clock, cost=500.0)
    deadline = LegDeadline(1200.0, clock=clock)
    return chat, ExtractionLadder(
        stage="extraction", deadline=deadline, price_fn=lambda **_kw: 300.0
    )


def excerpt(prompt: str) -> str:
    """The text a built prompt actually handed the model, between the fences."""

    head, _, rest = prompt.partition("<<<")
    body, _, _tail = rest.rpartition(">>>")
    return body if head else prompt


@pytest.fixture
def recorder():
    rec = diag_mod.activate(run_id="c042-test")
    yield rec
    diag_mod.deactivate()


def run_stage(
    chat: Chat,
    *,
    monkeypatch: pytest.MonkeyPatch,
    max_attempts: int = 3,
    ladder: Optional[ExtractionLadder] = None,
    build: Any = real_builder,
    repair_json: bool = False,
    repair_chat_fn: Any = None,
    stage_name: str = "extraction",
    retry_on_empty_payload: bool = True,
) -> Any:
    monkeypatch.setattr(pipeline_mod, "chat_detailed", chat)
    return _run_json_stage(
        stage_name=stage_name,
        system_prompt="SYSTEM",
        build_user_prompt=build,
        max_attempts=max_attempts,
        temperature=0.0,
        max_tokens=100,
        repair_json=repair_json,
        repair_chat_fn=repair_chat_fn,
        retry_on_empty_payload=retry_on_empty_payload,
        ladder=ladder,
    )


def terminations(rec) -> List[Dict[str, Any]]:
    return [r for r in rec.boundaries if r.get("boundary") == BOUNDARY_STAGE1_LADDER_TERMINATION]


# ---------------------------------------------------------------------------
# A1 -- the hard cap of three, across BOTH escalation paths in one call.
# ---------------------------------------------------------------------------
def test_a1_cap_of_three_holds_with_repair_and_empty_retry_in_the_same_call(
    monkeypatch, recorder
) -> None:
    """CORRECTION. Three model calls total, counting the repair draws.

    Base worst case for this exact configuration is ``max_attempts`` draws plus
    ``max_attempts x MAX_JSON_REPAIR_ATTEMPTS`` repair draws -- 5 + 5 x 3 = 20 --
    because localized repair sat entirely outside ``max_attempts``. The fixture
    exercises both triggers in one call: attempt 1 is unparseable and
    unsalvageable (repair path), attempt 2 is structurally empty (empty-payload
    path).
    """

    monkeypatch.setattr(pipeline_mod, "MAX_JSON_REPAIR_ATTEMPTS", 3)
    draws = Chat(UNSALVAGEABLE, "{}")
    repairs = Chat(UNSALVAGEABLE)
    ladder = make_ladder()

    with pytest.raises(PipelineFailure):
        run_stage(
            draws,
            monkeypatch=monkeypatch,
            max_attempts=5,
            ladder=ladder,
            repair_json=True,
            repair_chat_fn=repairs,
        )

    total = draws.calls + repairs.calls
    assert total == MAX_TOTAL_MODEL_ATTEMPTS == 3, (draws.calls, repairs.calls)
    assert ladder.attempts_used == 3
    assert any(row["skip_cause"] == SKIP_ATTEMPT_CAP for row in ladder.skipped)


def test_a1_repair_budget_never_exceeds_what_the_ceiling_leaves() -> None:
    """CORRECTION. The repair draw is priced against the same ceiling."""

    ladder = make_ladder()
    assert ladder.repair_budget(3) == 3
    ladder.record(rung="normal", model="env:X", request_hash="h1")
    assert ladder.repair_budget(3) == 2
    ladder.record(rung="normal", model="env:X", request_hash="h2", calls=2)
    assert ladder.repair_budget(3) == 0


# ---------------------------------------------------------------------------
# A2 -- rung 3 is materially different, verified rather than asserted.
# ---------------------------------------------------------------------------
def test_a2_third_rung_is_a_narrower_section_scoped_extraction(monkeypatch, recorder) -> None:
    """NEW ACCEPTANCE. There is no third rung at base to compare against.

    The rung is only accepted as different because it *is*: a different request
    hash and a narrower scope, both read back from the prompt that was built.
    """

    draws = Chat("{}", "{}", WITH_REACTIONS)
    ladder = make_ladder(deadline=LegDeadline(3480.0))

    payload = run_stage(draws, monkeypatch=monkeypatch, ladder=ladder)[0]

    assert draws.calls == 3
    assert payload["processes"]["reactions"][0]["name"] == "r1"

    third = ladder.issued[-1]
    assert third.rung == "materially_different_strategy"
    assert third.scope.startswith("sections:")
    assert third.scope != SCOPE_FULL_TEXT
    # Observably different in request hash AND in scope, per A2.
    assert third.request_hash not in {a.request_hash for a in ladder.issued[:-1]}
    assert "<extraction_scope>" in draws.prompts[2]
    # The TEXT the model is asked to read is narrower -- the instruction block
    # around it is longer, which is why the excerpt is what gets measured.
    assert len(excerpt(draws.prompts[2])) < len(excerpt(draws.prompts[0]))
    # And it is a NARROWING, not a re-wording: the references section is gone.
    assert "sequencing core" not in draws.prompts[2]
    assert "LpxC hydrolyses" in draws.prompts[2]


def test_a2_a_prompt_tweak_is_not_a_rung(monkeypatch, recorder) -> None:
    """NEW ACCEPTANCE. A builder that ignores the scope request gets refused.

    ``flat_builder`` produces a *different string* for rung 3 -- so the request
    hash differs -- while being the same strategy over the same text for the same
    model. That is precisely the shape of the defect, and a different hash alone
    must not buy it a third call.
    """

    draws = Chat("{}")
    ladder = make_ladder(deadline=LegDeadline(3480.0))

    with pytest.raises(PipelineFailure):
        run_stage(draws, monkeypatch=monkeypatch, ladder=ladder, build=flat_builder)

    assert draws.calls == 2
    assert any(
        row["skip_cause"] == SKIP_NOT_MATERIALLY_DIFFERENT for row in ladder.skipped
    )


# ---------------------------------------------------------------------------
# A3 -- the identical-empty-hash rule. The PMC12782028 signature, by hash.
# ---------------------------------------------------------------------------
def test_a3_identical_empty_hash_stops_the_same_prompt_going_out_again(
    monkeypatch, recorder
) -> None:
    """CORRECTION. Base issues a third identical call; the tip does not.

    Base capture (``c042_base_capture_bcc0bfe.json``): ``model_calls: 3`` and
    ``third_prompt_equals_second: true``. Pinned by hash rather than by prose:
    the rule fires on two byte-identical structurally empty replies.
    """

    draws = Chat("{}")
    ladder = make_ladder()

    with pytest.raises(PipelineFailure) as excinfo:
        run_stage(draws, monkeypatch=monkeypatch, max_attempts=3, ladder=ladder)

    assert draws.calls == 2
    # The base failure, restated as an assertion: prompts 2 and 3 would have
    # been identical, and the third one never went out.
    assert ladder.identical_empty_hash() == payload_hash("{}")
    assert ladder.identical_empty_hash().startswith("sha256:")
    assert any(row["skip_cause"] == SKIP_IDENTICAL_PROMPT for row in ladder.skipped)
    assert excinfo.value.terminal_reason == IDENTICAL_EMPTY_RESPONSE


def test_a3_when_a_third_call_does_happen_it_carries_a_different_scope(
    monkeypatch, recorder
) -> None:
    """CORRECTION. § 9 permits a third call only if it is not the same ask."""

    draws = Chat("{}")
    ladder = make_ladder(deadline=LegDeadline(3480.0))

    with pytest.raises(PipelineFailure):
        run_stage(draws, monkeypatch=monkeypatch, max_attempts=3, ladder=ladder)

    assert draws.calls == 3
    assert draws.prompts[2] != draws.prompts[1]
    assert ladder.issued[2].scope.startswith("sections:")
    assert len({a.request_hash for a in ladder.issued}) == 3


# ---------------------------------------------------------------------------
# A4 -- rung 3 is budget-gated through LegDeadline.admit.
# ---------------------------------------------------------------------------
def test_a4_third_rung_is_not_issued_when_the_budget_cannot_fit_it(
    monkeypatch, recorder
) -> None:
    """CORRECTION. A refused budget is budget_exhausted and nothing else."""

    draws, ladder = spent_leg_ladder("{}")

    with pytest.raises(PipelineFailure) as excinfo:
        run_stage(draws, monkeypatch=monkeypatch, max_attempts=2, ladder=ladder)

    assert draws.calls == 2
    assert excinfo.value.terminal_reason == BUDGET_EXHAUSTED
    assert excinfo.value.terminal_reason != IDENTICAL_EMPTY_RESPONSE
    assert excinfo.value.terminal_reason != RETRIEVAL_EXHAUSTED
    assert any(row["skip_cause"] == SKIP_BUDGET_REFUSED for row in ladder.skipped)
    row = terminations(recorder)[-1]
    assert row["terminal_reason"] == BUDGET_EXHAUSTED
    assert row["extraction_ladder"]["skipped_next_step"]["skip_cause"] == SKIP_BUDGET_REFUSED


def test_a4_the_refusal_comes_from_the_c032_admission_and_raises_its_error() -> None:
    """CORRECTION. The gate is C-032's, not a second budget notion."""

    ladder = make_ladder(deadline=LegDeadline(200.0), seconds=300.0)
    decision = ladder.admit(
        rung="materially_different_strategy",
        model="env:X",
        request_hash="h3",
        budget_conditional=True,
    )
    assert decision.allowed is False
    assert decision.admission is not None and decision.admission.allowed is False
    assert decision.admission.reason == BUDGET_EXHAUSTED
    with pytest.raises(Exception) as excinfo:
        decision.raise_if_refused()
    assert getattr(type(excinfo.value), "terminal_reason", "") == BUDGET_EXHAUSTED


# ---------------------------------------------------------------------------
# A5 -- the three reasons, one cause each. The cross-product.
# ---------------------------------------------------------------------------
def test_a5_each_reason_is_produced_by_and_only_by_its_own_cause(
    monkeypatch, recorder
) -> None:
    """CORRECTION. At base no boundary carries any of the six reasons at all."""

    observed: Dict[str, str] = {}

    # Cause 1: the leg budget cannot fit the next rung.
    diag_mod.activate(run_id="c042-budget")
    spender, ladder = spent_leg_ladder("{}")
    with pytest.raises(PipelineFailure) as budget_exc:
        run_stage(spender, monkeypatch=monkeypatch, max_attempts=2, ladder=ladder)
    observed["budget"] = budget_exc.value.terminal_reason

    # Cause 2: the model returned the same nothing twice, with budget untouched.
    diag_mod.activate(run_id="c042-identical")
    ladder2 = make_ladder(deadline=LegDeadline(3480.0))
    with pytest.raises(PipelineFailure) as empty_exc:
        run_stage(
            Chat("{}"), monkeypatch=monkeypatch, max_attempts=2, ladder=ladder2,
            build=flat_builder,
        )
    observed["identical"] = empty_exc.value.terminal_reason

    # Cause 3: one external operation exceeded its own configured maximum.
    rec3 = diag_mod.activate(run_id="c042-timeout")
    ladder3 = make_ladder(deadline=LegDeadline(3480.0))
    with pytest.raises(LLMOperationTimeout):
        run_stage(
            Chat(raises=LLMOperationTimeout("call exceeded its per-operation bound")),
            monkeypatch=monkeypatch, max_attempts=2, ladder=ladder3,
        )
    observed["timeout"] = terminations(rec3)[-1]["terminal_reason"]
    diag_mod.deactivate()

    assert observed == {
        "budget": BUDGET_EXHAUSTED,
        "identical": IDENTICAL_EMPTY_RESPONSE,
        "timeout": OPERATION_TIMEOUT,
    }
    # The cross-product: no reason appears for a cause that is not its own.
    assert len(set(observed.values())) == 3
    assert RETRIEVAL_EXHAUSTED not in set(observed.values())


def test_a5_an_operation_timeout_propagates_unchanged_to_the_caller(
    monkeypatch, recorder
) -> None:
    """PRESERVATION. Recording the reason must not swallow or retype the error."""

    boom = LLMOperationTimeout("call exceeded its per-operation bound")
    with pytest.raises(LLMOperationTimeout) as excinfo:
        run_stage(Chat(raises=boom), monkeypatch=monkeypatch, ladder=make_ladder())
    assert excinfo.value is boom


# ---------------------------------------------------------------------------
# A6 -- the cap is a ceiling, not a quota.
# ---------------------------------------------------------------------------
def test_a6_a_successful_first_attempt_issues_exactly_one_model_call(
    monkeypatch, recorder
) -> None:
    """PRESERVATION. Asserted on the call count, not on the result."""

    draws = Chat(WITH_REACTIONS)
    ladder = make_ladder(deadline=LegDeadline(3480.0))
    payload, attempts = run_stage(draws, monkeypatch=monkeypatch, ladder=ladder)

    assert draws.calls == 1
    assert ladder.attempts_used == 1
    assert ladder.attempts_remaining == 2
    assert len(attempts) == 1
    assert payload["processes"]["reactions"][0]["name"] == "r1"


# ---------------------------------------------------------------------------
# A7 -- preservation, against a golden captured at base FIRST.
# ---------------------------------------------------------------------------
def test_a7_non_empty_paths_are_byte_identical_to_the_base_capture() -> None:
    """PRESERVATION / G9. The golden was captured by running at ``bcc0bfe``.

    Four fixtures that end in a usable payload plus the Stage-2 path, compared as
    canonical JSON: return value, attempt log and every boundary record.
    """

    base = json.loads(BASE_CAPTURE.read_text(encoding="utf-8"))
    assert "src" in base["t2pw_file"], "the committed capture must come from a source tree"
    tip = _load_proof_module().capture_preservation()

    dump = lambda value: json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)
    assert set(tip) == set(base["preservation"])
    for name in sorted(base["preservation"]):
        assert dump(tip[name]) == dump(base["preservation"][name]), name


def test_a7_repair_unavailable_behaves_exactly_as_before(monkeypatch, recorder) -> None:
    """PRESERVATION. The documented guarantee in ``_run_json_stage``'s docstring."""

    salvageable = WITH_REACTIONS[:-1] + ', "trailing": '
    draws = Chat(salvageable)
    payload, attempts = run_stage(
        draws, monkeypatch=monkeypatch, max_attempts=2, repair_json=False,
        ladder=make_ladder(),
    )
    assert draws.calls == 1
    assert payload["processes"]["reactions"][0]["name"] == "r1"
    assert attempts[-1]["note"] == "salvaged_json"


def test_a7_stage_two_gets_no_ladder(monkeypatch, recorder) -> None:
    """PRESERVATION. Stage 2's top level is empty by design; a rung there invents."""

    draws = Chat("{}")
    payload, _attempts = run_stage(
        draws, monkeypatch=monkeypatch, max_attempts=2, stage_name="inference",
        retry_on_empty_payload=False, build=flat_builder,
    )
    assert draws.calls == 1
    assert payload == {}
    assert terminations(recorder) == []
    assert [r for r in recorder.boundaries if r["boundary"] == BOUNDARY_STAGE1_LADDER_CHECKPOINT] == []


# ---------------------------------------------------------------------------
# A8 -- nothing invented.
# ---------------------------------------------------------------------------
def test_a8_exhausting_all_three_rungs_invents_no_biology(monkeypatch, recorder) -> None:
    """NEW ACCEPTANCE. § 1's hard limit, asserted on the preserved payload."""

    draws = Chat("{}")
    ladder = make_ladder(deadline=LegDeadline(3480.0))

    with pytest.raises(PipelineFailure) as excinfo:
        run_stage(draws, monkeypatch=monkeypatch, max_attempts=2, ladder=ladder)

    assert draws.calls == 3
    preserved = excinfo.value.ladder["current_structured_payload"]
    assert count_entities(preserved).get("total", 0) == 0
    assert count_processes(preserved).get("total", 0) == 0
    # Nothing anywhere in the record carries a fabricated row either.
    for row in recorder.boundaries:
        assert (row.get("raw_process_counts") or {}).get("total", 0) == 0
        assert (row.get("raw_entity_counts") or {}).get("total", 0) == 0
    assert "structurally empty" in str(excinfo.value)


# ---------------------------------------------------------------------------
# A9 -- checkpoint before each long call; the § 9 preservation set on stop.
# ---------------------------------------------------------------------------
def test_a9_a_checkpoint_is_persisted_before_every_model_call(
    monkeypatch, recorder
) -> None:
    """NEW ACCEPTANCE. There is no checkpoint at base to preserve."""

    draws = Chat("{}", "{}", WITH_REACTIONS)
    ladder = make_ladder(deadline=LegDeadline(3480.0))
    run_stage(draws, monkeypatch=monkeypatch, ladder=ladder)

    names = [row["boundary"] for row in recorder.boundaries]
    assert names.count(BOUNDARY_STAGE1_LADDER_CHECKPOINT) == draws.calls == 3
    # Before, not after: the first row of the run is a checkpoint, and every
    # extraction row is preceded by one.
    assert names[0] == BOUNDARY_STAGE1_LADDER_CHECKPOINT
    for index, name in enumerate(names):
        if name == BOUNDARY_STAGE1_EXTRACTION:
            assert BOUNDARY_STAGE1_LADDER_CHECKPOINT in names[:index]
    assert len(ladder.checkpoints) == 3


def test_a9_termination_records_the_whole_section_9_preservation_set(
    monkeypatch, recorder
) -> None:
    """NEW ACCEPTANCE. Every item § 9 lists, in one record."""

    draws, ladder = spent_leg_ladder("{}")
    with pytest.raises(PipelineFailure) as excinfo:
        run_stage(draws, monkeypatch=monkeypatch, max_attempts=2, ladder=ladder)

    preserved = excinfo.value.ladder
    assert preserved["last_completed_stage"] == "stage0_preprocess"
    assert preserved["current_structured_payload"] == {}
    assert [a["attempt"] for a in preserved["attempts"]] == [1, 2]
    assert all(a["model"] and a["request_hash"] and a["response_hash"] for a in preserved["attempts"])
    assert preserved["budget"]["elapsed_seconds"] >= 0.0
    assert "remaining_seconds" in preserved["budget"]
    assert preserved["skipped_next_step"]["rung"] == "materially_different_strategy"
    assert preserved["termination_reason"] == BUDGET_EXHAUSTED
    assert preserved["attempt_cap"] == MAX_TOTAL_MODEL_ATTEMPTS
    # And the same set reached the artifact, not only the exception.
    assert terminations(recorder)[-1]["extraction_ladder"]["termination_reason"] == BUDGET_EXHAUSTED


# ---------------------------------------------------------------------------
# A10 -- diagnostics stay diagnostics.
# ---------------------------------------------------------------------------
def test_a10_extraction_boundary_rows_keep_their_schema(monkeypatch, recorder) -> None:
    """PRESERVATION. Existing readers filter by boundary name; nothing moved."""

    base = json.loads(BASE_CAPTURE.read_text(encoding="utf-8"))
    base_rows = base["preservation"]["clean_json"]["boundaries"]
    base_keys = {key for row in base_rows for key in row}

    run_stage(Chat(WITH_REACTIONS), monkeypatch=monkeypatch, ladder=make_ladder())
    rows = [r for r in recorder.boundaries if r["boundary"] == BOUNDARY_STAGE1_EXTRACTION]

    assert len(rows) == len(base_rows)
    assert {key for row in rows for key in row} == base_keys
    # The ladder's own rows live on their own boundary names, so a reader that
    # selects ``stage1_extraction`` sees exactly what it always saw.
    assert BOUNDARY_STAGE1_LADDER_CHECKPOINT != BOUNDARY_STAGE1_EXTRACTION
    assert json.loads(json.dumps(recorder.artifacts()))


def test_a10_the_ladder_extends_record_boundary_additively(monkeypatch, recorder) -> None:
    """NEW ACCEPTANCE. Extra keys go through the existing ``**extra`` sanitizer."""

    ladder = make_ladder(deadline=LegDeadline(3480.0))
    draws = Chat("{}", "{}", WITH_REACTIONS)
    run_stage(draws, monkeypatch=monkeypatch, ladder=ladder)

    third = [
        r for r in recorder.boundaries
        if r.get("ladder_rung") == "materially_different_strategy"
    ]
    assert len(third) == 1
    assert third[0]["ladder_scope"].startswith("sections:")
    assert third[0]["boundary"] == BOUNDARY_STAGE1_EXTRACTION
    # Serializable, because everything went through ``bound``.
    assert json.loads(json.dumps(recorder.boundaries))


# ---------------------------------------------------------------------------
# A11 -- no fail-open. Every missing input stops the rung.
# ---------------------------------------------------------------------------
def test_a11_missing_budget_does_not_buy_a_third_attempt(monkeypatch, recorder) -> None:
    """NEW ACCEPTANCE. No deadline in scope means rung 3 does not run."""

    draws = Chat("{}")
    ladder = make_ladder()
    assert ladder.budget_determinable is False

    with pytest.raises(PipelineFailure):
        run_stage(draws, monkeypatch=monkeypatch, max_attempts=2, ladder=ladder)

    assert draws.calls == 2
    assert any(row["skip_cause"] == SKIP_BUDGET_INDETERMINATE for row in ladder.skipped)


@pytest.mark.parametrize(
    "setup,expected",
    [
        ("no_model", SKIP_MODEL_IDENTITY_UNKNOWN),
        ("cap_spent", SKIP_ATTEMPT_CAP),
        ("no_budget", SKIP_BUDGET_INDETERMINATE),
    ],
)
def test_a11_every_undeterminable_input_refuses_rather_than_defaults(
    setup: str, expected: str
) -> None:
    """NEW ACCEPTANCE. The three inputs A11 names, each refused on its own cause."""

    ladder = make_ladder(deadline=LegDeadline(3480.0) if setup != "no_budget" else None)
    model = "" if setup == "no_model" else "env:X"
    if setup == "cap_spent":
        for index in range(MAX_TOTAL_MODEL_ATTEMPTS):
            ladder.record(rung="normal", model="env:X", request_hash=f"h{index}")

    decision = ladder.admit(
        rung="materially_different_strategy",
        model=model,
        request_hash="fresh",
        scope="sections:results",
        budget_conditional=True,
    )
    assert decision.allowed is False
    assert decision.skip_cause == expected


def test_a11_material_difference_is_checked_against_what_actually_ran() -> None:
    """NEW ACCEPTANCE. The truth table behind ``materially_differs``."""

    ladder = make_ladder()
    ladder.record(rung="normal", model="env:A", request_hash="h1")

    assert ladder.materially_differs(model="env:A", request_hash="h1", scope=SCOPE_FULL_TEXT)[0] is False
    assert ladder.materially_differs(model="env:A", request_hash="h2", scope=SCOPE_FULL_TEXT)[0] is False
    assert ladder.materially_differs(model="env:A", request_hash="h2", scope="sections:results")[0] is True
    assert ladder.materially_differs(model="env:B", request_hash="h2", scope=SCOPE_FULL_TEXT)[0] is True
    assert ladder.materially_differs(model="env:B", request_hash="", scope="sections:results")[0] is False


def test_a11_an_unnarrowable_text_yields_no_scope_rather_than_a_fake_one() -> None:
    """NEW ACCEPTANCE. ``("", "")`` is the honest answer, and it blocks the rung."""

    assert pipeline_mod._narrow_to_high_signal_sections("one blob, no headers") == ("", "")
    assert pipeline_mod._narrow_to_high_signal_sections("") == ("", "")
    narrowed, label = pipeline_mod._narrow_to_high_signal_sections(PAPER)
    assert label.startswith("sections:")
    assert len(narrowed) < len(PAPER)
    assert "Acknowledgements" not in narrowed

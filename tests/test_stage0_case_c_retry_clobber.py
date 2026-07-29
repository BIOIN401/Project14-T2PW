"""Regression tests for the Stage-0 "Case C" clobber in the Streamlit driver.

Background — the failure this pins
----------------------------------
Stage 0 has two DIFFERENT states that look identical to ``_has_usable_context``:

* a genuinely failed call — Stage 0 fails CLOSED, so a swallowed LLM error or an
  unparseable reply both come back with every field blank; and
* a deliberate **Case C** — the Stage-0 prompt's verdict for a multi-example
  review with no target selected, which MANDATES that ``pathway_name`` /
  ``likely_organism`` / ``key_compounds`` / ``key_proteins`` / ``gap_terms`` all
  be left blank. The only field that distinguishes it is ``document_type``.

The old inline block in ``streamlit_app.py`` retried Stage 0 on a 20,000-char
head whenever ``not _has_usable_context(...)``, and assigned the retry's result
straight back over ``pathway_context``. So a correct Case C triggered the retry,
and a blank second draw then dropped ``document_type`` — the one field
``is_ambiguous_multi_example_review_context`` requires. With the refusal gate
disarmed the run carried on into an unguided extraction over a multi-example
review, and because the batch driver fills the extraction-focus box it no longer
aborted loudly. That is how leg PMC13278307/strict of ``runs/2026-07-28_0919``
reported PASS in 2098s with 14 reactions and ZERO reactions of the lipid A
pathway named in its own focus box.

What is asserted here
---------------------
1. With a fake ``preprocess`` returning [Case C, then empty], the run ends in
   ``ambiguous_review_scope`` instead of proceeding. The refusal block is not
   re-implemented in the test: it is lifted out of ``streamlit_app.py`` by AST
   (it lives inside the module-level ``if submit:`` script body, which cannot be
   imported and called) and executed against the context the Stage-0 helper
   actually returns.
2. The Case-C exclusion sits BEFORE the retry, so no second LLM call is spent on
   a verdict that is deterministic and that re-running never changes.
3. The retry is non-destructive in every direction: it may still rescue a truly
   failed Stage 0, but it can never lower the rank of the context it was meant to
   improve.

Offline / deterministic. The app module runs a Streamlit script at import and
neither ``streamlit`` nor ``openai`` is present in the test env, so both are
stubbed at the top (mirroring tests/test_stage0_scope_fallback.py).
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *_: object, **__: object) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **__: None)
            )

    openai_stub.OpenAI = _OpenAI
    openai_stub.RateLimitError = RuntimeError
    openai_stub.APIError = RuntimeError
    openai_stub.APITimeoutError = RuntimeError
    openai_stub.AuthenticationError = RuntimeError
    openai_stub.BadRequestError = RuntimeError
    sys.modules["openai"] = openai_stub

if "streamlit" not in sys.modules:
    _st = MagicMock(name="streamlit")
    _st.columns.side_effect = lambda spec=1, **__: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    _st.tabs.side_effect = lambda labels, **__: [MagicMock() for _ in labels]
    _st.form_submit_button.return_value = False
    _st.checkbox.return_value = False
    _st.session_state = MagicMock()
    _st.session_state.get.return_value = None
    _st.session_state.__contains__ = lambda _s, _k: False
    sys.modules["streamlit"] = _st

import t2pw.app.streamlit_app as app  # noqa: E402
from t2pw.pipeline.preprocessor import (  # noqa: E402
    is_ambiguous_multi_example_review_context,
)

APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"

# Longer than _PREPROCESS_RETRY_CHARS (20,000), which is the length condition the
# retry is gated on. Essentially every topic-fetched review clears it, which is
# why the clobber was not a rare path.
LONG_REVIEW_TEXT = "Lipid A biosynthesis is reviewed alongside five other examples. " * 400


# ---------------------------------------------------------------------------
# Stage-0 draws
# ---------------------------------------------------------------------------
def _case_c_context() -> dict[str, Any]:
    """A correct Case C: the prompt REQUIRES the pathway fields be blank."""
    return {
        "document_type": "multi_example_review",
        "context_type": "review",
        "scope_status": "ambiguous",
        "selected_example": "",
        "candidate_examples": [
            {"name": "Lipid A biosynthesis", "organism": "Escherichia coli"},
            {"name": "Menaquinone biosynthesis", "organism": "Lactococcus lactis"},
        ],
        "warning": "No specific example was selected.",
        "pathway_name": "",
        "likely_organism": "",
        "key_compounds": [],
        "key_proteins": [],
    }


def _failed_context() -> dict[str, Any]:
    """Stage 0 failed closed: every field blank, and no document_type."""
    return {
        "document_type": "",
        "pathway_name": "",
        "likely_organism": "",
        "key_compounds": [],
        "key_proteins": [],
    }


def _usable_context() -> dict[str, Any]:
    return {
        "document_type": "primary_research",
        "pathway_name": "lipid A biosynthesis",
        "likely_organism": "Escherichia coli",
        "key_compounds": ["UDP-GlcNAc"],
        "key_proteins": ["LpxA"],
    }


def _fake_preprocess(monkeypatch: pytest.MonkeyPatch, draws: list[dict[str, Any]]) -> list[str]:
    """Patch ``streamlit_app.preprocess`` to return ``draws`` in order.

    Returns the list of texts it was called with, so a test can assert both HOW
    MANY Stage-0 calls happened and that the retry really saw the bounded head.
    """
    seen: list[str] = []
    remaining = list(draws)

    def fake(text: str, **_: Any) -> dict[str, Any]:
        seen.append(text)
        assert remaining, f"preprocess called {len(seen)} times, only {len(draws)} draws staged"
        return remaining.pop(0)

    monkeypatch.setattr(app, "preprocess", fake)
    return seen


# ---------------------------------------------------------------------------
# 1. The refusal gate, lifted out of the module-level script body.
# ---------------------------------------------------------------------------
class _Stopped(Exception):
    """Raised by the fake ``st.stop()`` — the run halting is the pass condition."""


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.errors: list[str] = []
        self.stopped = False

    def error(self, message: str, *_: Any, **__: Any) -> None:
        self.errors.append(str(message))

    def warning(self, *_: Any, **__: Any) -> None:
        pass

    def subheader(self, *_: Any, **__: Any) -> None:
        pass

    def json(self, *_: Any, **__: Any) -> None:
        pass

    def stop(self) -> None:
        raise _Stopped()


def _ambiguous_review_gate() -> ast.If:
    """Extract the ``ambiguous_review_scope`` refusal block from the app source.

    The block lives inside ``if submit:`` at module level, so it cannot be
    imported and invoked; compiling just this node keeps the test honest (it
    executes the shipped statements, not a copy of them) and makes the test fail
    loudly if the gate is ever deleted or renamed.
    """
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and (ast.get_source_segment(source, node.test) or "").strip()
        == "is_ambiguous_multi_example_review_context(pathway_context)"
        and "ambiguous_review_scope" in (ast.get_source_segment(source, node) or "")
    ]
    assert len(blocks) == 1, f"expected exactly one ambiguous_review_scope gate, found {len(blocks)}"
    return blocks[0]


def _run_gate(pathway_context: dict[str, Any]) -> _FakeStreamlit:
    """Execute the shipped refusal block against ``pathway_context``.

    ``stopped`` on the returned recorder is the pass condition: the block ends in
    ``st.stop()``, so a run that reaches the end without stopping is exactly the
    run that walked on into an unguided extraction.
    """
    fake_st = _FakeStreamlit()
    namespace: dict[str, Any] = {
        "st": fake_st,
        "pathway_context": pathway_context,
        "is_ambiguous_multi_example_review_context": (
            is_ambiguous_multi_example_review_context
        ),
    }
    module = ast.Module(body=[_ambiguous_review_gate()], type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        exec(compile(module, str(APP_PATH), "exec"), namespace)  # noqa: S102
    except _Stopped:
        fake_st.stopped = True
    return fake_st


def test_case_c_then_empty_draw_still_ends_in_ambiguous_review_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact 2026-07-28 clobber: [Case C, empty] must NOT proceed.

    Before the fix the blank second draw replaced the Case C, ``document_type``
    was lost, ``is_ambiguous_multi_example_review_context`` went False and the run
    walked on into an unguided extraction that reported PASS.
    """
    _fake_preprocess(monkeypatch, [_case_c_context(), _failed_context()])

    pathway_context = app._run_stage_zero_with_retry(
        LONG_REVIEW_TEXT, temperature=0.0, user_task_context="lipid A biosynthesis"
    )

    # The refusal survives: document_type is what the gate keys on.
    assert pathway_context.get("document_type") == "multi_example_review"
    assert is_ambiguous_multi_example_review_context(pathway_context)

    gate = _run_gate(pathway_context)
    assert gate.stopped, "the run proceeded past the ambiguous-review gate"
    assert gate.session_state["pipeline_error"]["status"] == "ambiguous_review_scope"
    assert gate.session_state["pipeline_ready"] is False
    # The candidate examples the user must choose between survive to the UI.
    assert [
        entry["name"] for entry in gate.session_state["pipeline_error"]["candidate_examples"]
    ] == ["Lipid A biosynthesis", "Menaquinone biosynthesis"]


def test_case_c_does_not_spend_a_retry_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Case-C exclusion must sit BEFORE the retry, not after it.

    A deliberate Case C is deterministic — the second draw can only return the
    same verdict or something weaker, so the second LLM call is pure cost (and,
    before the fix, pure risk).
    """
    seen = _fake_preprocess(monkeypatch, [_case_c_context()])

    pathway_context = app._run_stage_zero_with_retry(
        LONG_REVIEW_TEXT, temperature=0.0, user_task_context=None
    )

    assert len(seen) == 1, "the retry fired on a deliberate Case C"
    assert is_ambiguous_multi_example_review_context(pathway_context)


# ---------------------------------------------------------------------------
# 2. The retry is still allowed to help — it just may never hurt.
# ---------------------------------------------------------------------------
def test_retry_rescues_a_genuinely_failed_stage_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_preprocess(monkeypatch, [_failed_context(), _usable_context()])

    pathway_context = app._run_stage_zero_with_retry(
        LONG_REVIEW_TEXT, temperature=0.0, user_task_context=None
    )

    assert len(seen) == 2
    # The retry really re-read the bounded head, not the whole document.
    assert seen[1] == LONG_REVIEW_TEXT[: app._PREPROCESS_RETRY_CHARS]
    assert pathway_context["pathway_name"] == "lipid A biosynthesis"
    assert app._has_usable_context(pathway_context)


def test_retry_that_recognises_the_review_is_adopted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Case C from the retry outranks a blank first draw and re-arms the gate.

    The head of a review is frequently enough for Stage 0 to classify a document
    it could not classify from the full text, so this direction has to stay open.
    """
    _fake_preprocess(monkeypatch, [_failed_context(), _case_c_context()])

    pathway_context = app._run_stage_zero_with_retry(
        LONG_REVIEW_TEXT, temperature=0.0, user_task_context=None
    )

    assert is_ambiguous_multi_example_review_context(pathway_context)
    assert _run_gate(pathway_context).stopped


def test_two_failed_draws_leave_the_first_context_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No improvement means no swap — the downstream scope fallback still runs."""
    first = _failed_context()
    first["_marker"] = "first draw"
    _fake_preprocess(monkeypatch, [first, _failed_context()])

    pathway_context = app._run_stage_zero_with_retry(
        LONG_REVIEW_TEXT, temperature=0.0, user_task_context=None
    )

    assert pathway_context["_marker"] == "first draw"


def test_usable_first_draw_is_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_preprocess(monkeypatch, [_usable_context()])

    pathway_context = app._run_stage_zero_with_retry(
        LONG_REVIEW_TEXT, temperature=0.0, user_task_context=None
    )

    assert len(seen) == 1
    assert pathway_context["pathway_name"] == "lipid A biosynthesis"


def test_short_document_is_not_re_sent_to_stage_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below the head length the retry would re-send identical input."""
    seen = _fake_preprocess(monkeypatch, [_failed_context()])

    app._run_stage_zero_with_retry(
        "short abstract", temperature=0.0, user_task_context=None
    )

    assert len(seen) == 1


# ---------------------------------------------------------------------------
# 3. The rank that makes "strictly better" well defined.
# ---------------------------------------------------------------------------
def test_stage0_rank_puts_the_refusal_above_everything() -> None:
    """A Case C must outrank a usable context, so no draw can disarm the gate."""
    assert app._stage0_context_rank(_case_c_context()) == 2
    assert app._stage0_context_rank(_usable_context()) == 1
    assert app._stage0_context_rank(_failed_context()) == 0
    assert app._stage0_context_rank(None) == 0
    assert app._stage0_context_rank("not a dict") == 0
    # A Case C and a failed call are indistinguishable to _has_usable_context —
    # that ambiguity is the whole reason the rank exists.
    assert not app._has_usable_context(_case_c_context())
    assert not app._has_usable_context(_failed_context())

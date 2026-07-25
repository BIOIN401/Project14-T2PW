"""Tests for the Stage-0 user-scope fallback.

Background
----------
Stage 0 (the preprocessor LLM) fails CLOSED to an empty context. An empty
context silently disables multi-paper RAG — ``t2pw.rag.acquire.build_query``
returns "" so zero papers are ever fetched — and leaves extraction unguided,
even when the user has typed an obvious scope into the extraction-focus box.

``t2pw.app.streamlit_app._seed_context_from_user_scope`` closes that gap: when
Stage 0 returns no usable context but the user supplied a scope string, it
derives a ``pathway_name`` (and, from a trailing ``" in <organism>"``, a
``likely_organism``) so a flaky preprocessor call can no longer nuke a scope the
user already provided.

Offline / deterministic. The app module runs a Streamlit script at import and
neither ``streamlit`` nor ``openai`` is present in the test env, so both are
stubbed at the top (mirroring tests/test_rag_differential_gate.py).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

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

from t2pw.app.streamlit_app import (  # noqa: E402
    _has_usable_context,
    _seed_context_from_user_scope,
)


def _empty_ctx() -> dict:
    """A Stage-0 context that failed closed — every field blank."""
    return {
        "pathway_name": "",
        "likely_organism": "",
        "key_compounds": [],
        "key_proteins": [],
    }


def test_scope_with_organism_is_split_into_pathway_and_organism() -> None:
    seeded = _seed_context_from_user_scope(
        _empty_ctx(), "menaquinone biosynthesis in Lactococcus lactis"
    )
    assert seeded is not None
    assert seeded["pathway_name"] == "menaquinone biosynthesis"
    assert seeded["likely_organism"] == "Lactococcus lactis"
    # The seeded context is now usable, so RAG/build_query can proceed.
    assert _has_usable_context(seeded)
    # Original keys are preserved (augmented copy, not a fresh dict).
    assert "key_compounds" in seeded


def test_scope_without_organism_becomes_pathway_name() -> None:
    seeded = _seed_context_from_user_scope(_empty_ctx(), "menaquinone biosynthesis")
    assert seeded is not None
    assert seeded["pathway_name"] == "menaquinone biosynthesis"
    assert seeded["likely_organism"] == ""
    assert _has_usable_context(seeded)


def test_long_in_tail_is_not_mistaken_for_organism() -> None:
    # "in vitro reconstitution of the pathway" is 5 words — not an organism.
    seeded = _seed_context_from_user_scope(
        _empty_ctx(), "menaquinone in vitro reconstitution of the pathway"
    )
    assert seeded is not None
    # Whole scope kept as the pathway name; no bogus organism restriction that
    # would zero out the literature query.
    assert seeded["pathway_name"] == "menaquinone in vitro reconstitution of the pathway"
    assert seeded["likely_organism"] == ""


def test_blank_scope_returns_none() -> None:
    assert _seed_context_from_user_scope(_empty_ctx(), "") is None
    assert _seed_context_from_user_scope(_empty_ctx(), "   ") is None
    assert _seed_context_from_user_scope(_empty_ctx(), None) is None


def test_existing_organism_is_not_overwritten() -> None:
    ctx = _empty_ctx()
    ctx["likely_organism"] = "Escherichia coli"
    seeded = _seed_context_from_user_scope(ctx, "menaquinone biosynthesis in Bacillus subtilis")
    assert seeded is not None
    # Pathway name is filled from the scope, but a pre-existing organism wins.
    assert seeded["pathway_name"] == "menaquinone biosynthesis"
    assert seeded["likely_organism"] == "Escherichia coli"

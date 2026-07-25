"""Tests for the differential S3 RAG-merge schema gate + the nameless-interaction
root-cause fix.

Background
----------
The S3 RAG-merge seam used to run the post-extraction runtime schema in
``enforce`` mode against the merged payload only. The rest of the pipeline runs
that same schema in ``report`` mode (``RUNTIME_SCHEMA_MODE = "report"``), so the
single-paper seed can legitimately carry shapes the strict schema flags — e.g.
Stage-2 inference interactions that were emitted without a ``name``. Holding the
merged payload to a stricter bar than the seed it merges into therefore
false-positived: it rejected a valid RAG merge because the *seed's own*
pre-existing nameless interactions violated the strict schema.

Two fixes are proven here:

1. The gate is now DIFFERENTIAL: it rejects only when the merge INTRODUCES a
   new schema violation not already present in the seed
   (``t2pw.app.streamlit_app._post_extraction_new_error_keys``). A merged payload
   whose only violation also exists in the seed is adopted; a merged payload with
   a brand-new violation is rejected.
2. ``clean_inference_output`` now synthesizes a deterministic ``name`` for an
   interaction that has usable ``entity_1``/``entity_2`` endpoints but no name, so
   the payload is schema-clean at its source rather than merely tolerated.

Offline / deterministic. The app module runs a Streamlit script at import and
neither ``streamlit`` nor ``openai`` is present in the test env, so both are
stubbed at the top (mirroring tests/test_rag_triage_orchestration.py).
"""

from __future__ import annotations

import copy
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
    _post_extraction_error_keys,
    _post_extraction_new_error_keys,
)
from t2pw.pipeline.pipeline import clean_inference_output  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _valid_seed() -> dict:
    """A minimal payload that passes the post-extraction runtime schema in
    enforce mode (empty error-key set)."""
    return {
        "metadata": {"pathway_name": "caffeine degradation"},
        "entities": {
            "species": [{"name": "Pseudomonas putida"}],
            "compounds": [{"name": "caffeine"}, {"name": "paraxanthine"}],
            "proteins": [{"name": "NdmA"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "R1 caffeine demethylation",
                    "inputs": ["caffeine"],
                    "outputs": ["paraxanthine"],
                }
            ]
        },
    }


def _nameless_interaction() -> dict:
    """An interaction with usable endpoints but no ``name`` — the exact shape
    Stage-2 inference used to emit, which the strict schema rejects."""
    return {"entity_1": "caffeine", "entity_2": "NdmA"}


# ---------------------------------------------------------------------------
# 1. Differential gate — a seed-shared violation is ADOPTED, not rejected.
# ---------------------------------------------------------------------------
def test_gate_adopts_merge_whose_only_violation_also_exists_in_seed():
    # The seed itself carries a nameless interaction (a pre-existing violation
    # the wider report-mode pipeline tolerates).
    seed = _valid_seed()
    seed["processes"]["interactions"] = [_nameless_interaction()]

    # The merge keeps that same nameless interaction and adds a valid reaction.
    merged = copy.deepcopy(seed)
    merged["processes"]["reactions"].append(
        {
            "name": "R4 theobromine demethylation",
            "inputs": ["theobromine"],
            "outputs": ["3-methylxanthine"],
        }
    )

    # The seed on its own already fails enforce mode...
    assert _post_extraction_error_keys(seed)
    # ...but the merge introduces NO new violation, so it is adopted.
    assert _post_extraction_new_error_keys(seed, merged) == set()


# ---------------------------------------------------------------------------
# 2. Differential gate — a NEW violation absent from the seed is REJECTED.
# ---------------------------------------------------------------------------
def test_gate_rejects_merge_that_introduces_a_new_violation():
    # A clean seed with no schema violations.
    seed = _valid_seed()
    assert _post_extraction_error_keys(seed) == set()

    # The merge introduces a brand-new nameless interaction not on the seed.
    merged = copy.deepcopy(seed)
    merged["processes"]["interactions"] = [_nameless_interaction()]

    new_keys = _post_extraction_new_error_keys(seed, merged)
    assert new_keys, "a genuinely new schema violation must be reported"
    # The new key is the missing interaction name, with array indices normalized.
    assert ("runtime_schema_missing_field", "/processes/interactions/*/name") in new_keys


# ---------------------------------------------------------------------------
# 3. Root cause — clean_inference_output names a nameless interaction.
# ---------------------------------------------------------------------------
def test_clean_inference_output_names_nameless_interaction_with_relationship():
    payload = {
        "additions": {
            "processes": {
                "interactions": [
                    {
                        "entity_1": "caffeine",
                        "entity_2": "NdmA",
                        "relationship": "binds",
                    }
                ]
            }
        }
    }
    cleaned = clean_inference_output(payload)
    interactions = cleaned["additions"]["processes"]["interactions"]
    assert len(interactions) == 1
    # Deterministic "{e1} {relationship} {e2}" name synthesized.
    assert interactions[0]["name"] == "caffeine binds NdmA"
    assert interactions[0]["entity_1"] == "caffeine"
    assert interactions[0]["entity_2"] == "NdmA"


def test_clean_inference_output_names_nameless_interaction_without_relationship():
    payload = {
        "additions": {
            "processes": {
                "interactions": [{"entity_1": "caffeine", "entity_2": "NdmA"}]
            }
        }
    }
    cleaned = clean_inference_output(payload)
    interaction = cleaned["additions"]["processes"]["interactions"][0]
    # Deterministic "{e1} - {e2}" fallback when no relationship is present.
    assert interaction["name"] == "caffeine - NdmA"


def test_clean_inference_output_preserves_existing_interaction_name():
    payload = {
        "additions": {
            "processes": {
                "interactions": [
                    {
                        "name": "explicit interaction name",
                        "entity_1": "caffeine",
                        "entity_2": "NdmA",
                        "relationship": "binds",
                    }
                ]
            }
        }
    }
    cleaned = clean_inference_output(payload)
    interaction = cleaned["additions"]["processes"]["interactions"][0]
    assert interaction["name"] == "explicit interaction name"

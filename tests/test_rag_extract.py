"""Prose→reaction extraction tests (``t2pw.rag.extract``) + synthesis integration.

Offline and deterministic: the LLM is always a fake ``chat_fn``, never the real
client. The guarded ``openai`` stub mirrors the other rag tests because the
integration fixtures import ``t2pw.rag.retrieve`` -> ``t2pw.rag.ingest`` ->
``t2pw.rag.acquire`` -> ``t2pw.mapping`` -> ``t2pw.llm.client`` -> ``openai`` at
import time. Kept at the top so the module passes run alone.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

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

from t2pw.rag import extract  # noqa: E402
from t2pw.rag.retrieve import EvidenceBundle, Gap  # noqa: E402
from t2pw.rag.store import Chunk, Retrieved  # noqa: E402
from t2pw.rag.synthesize import synthesize  # noqa: E402


# ---------------------------------------------------------------------------
# Fake chat_fn helpers (the model is never real here).
# ---------------------------------------------------------------------------
def _fake_chat(payload: Any):
    """Return a ``chat_fn`` that always returns ``payload`` (dict -> json.dumps)."""

    def _fn(_system: str, _user: str) -> str:
        return payload if isinstance(payload, str) else json.dumps(payload)

    return _fn


# ---------------------------------------------------------------------------
# Unit: extract_reactions_from_text.
# ---------------------------------------------------------------------------
def test_extract_parses_reactions_from_prose() -> None:
    chat = _fake_chat(
        {
            "reactions": [
                {
                    "name": "caffeine N1-demethylation",
                    "inputs": [{"name": "caffeine", "stoichiometry": 1}],
                    "outputs": [{"name": "theobromine"}, {"name": "formaldehyde"}],
                    "enzymes": ["NdmA"],
                    "reversible": False,
                }
            ]
        }
    )
    out = extract.extract_reactions_from_text(
        "NdmA catalyzes the N1-demethylation of caffeine to theobromine.", chat_fn=chat
    )
    assert len(out) == 1
    r = out[0]
    assert r["name"] == "caffeine N1-demethylation"
    assert [p["name"] for p in r["inputs"]] == ["caffeine"]
    assert {p["name"] for p in r["outputs"]} == {"theobromine", "formaldehyde"}
    assert r["enzymes"] == ["NdmA"]
    assert r["reversible"] is False


def test_extract_tolerates_markdown_fence() -> None:
    raw = (
        "```json\n"
        '{"reactions": [{"inputs": [{"name": "a"}], "outputs": [{"name": "b"}], '
        '"enzymes": [], "reversible": false}]}\n'
        "```"
    )
    out = extract.extract_reactions_from_text("x", chat_fn=lambda _s, _u: raw)
    assert len(out) == 1
    assert out[0]["inputs"][0]["name"] == "a"
    assert out[0]["outputs"][0]["name"] == "b"


def test_extract_no_reaction_returns_empty() -> None:
    out = extract.extract_reactions_from_text(
        "This paragraph is background context and states no reaction.",
        chat_fn=_fake_chat({"reactions": []}),
    )
    assert out == []


def test_extract_empty_text_makes_no_call() -> None:
    calls: List[str] = []

    def _fn(_s: str, user: str) -> str:
        calls.append(user)
        return "{}"

    assert extract.extract_reactions_from_text("   ", chat_fn=_fn) == []
    assert calls == []  # a blank passage short-circuits before any model call


def test_extract_fails_closed_on_chat_error() -> None:
    def _boom(_s: str, _u: str) -> str:
        raise RuntimeError("no embeddings endpoint / model error")

    # Fails closed: an error yields [] rather than propagating into synthesis.
    assert extract.extract_reactions_from_text("some prose", chat_fn=_boom) == []


def test_extract_garbage_output_returns_empty() -> None:
    assert extract.extract_reactions_from_text("x", chat_fn=lambda _s, _u: "not json") == []


def test_extract_drops_reaction_with_no_participants() -> None:
    # A "reaction" with neither inputs nor outputs is not a reaction.
    out = extract.extract_reactions_from_text(
        "x",
        chat_fn=_fake_chat({"reactions": [{"name": "junk", "inputs": [], "outputs": []}]}),
    )
    assert out == []


# ---------------------------------------------------------------------------
# Integration: the prose extractor flows into synthesize().
# ---------------------------------------------------------------------------
def _prose_bundle() -> EvidenceBundle:
    # A paper chunk of PROSE (no arrow equation) that states the missing step.
    chunk = Chunk(
        id="chunk-prose-ndmb",
        text=(
            "The enzyme NdmB catalyzes the N3-demethylation of theobromine, "
            "producing 7-methylxanthine and formaldehyde."
        ),
        source_id="PMC777001",
        source_title="NdmB continues bacterial caffeine catabolism",
        source_type="paper",
        source_uri="https://europepmc.org/article/PMC/PMC777001",
        organism="Pseudomonas putida",
        section="results",
    )
    gap = Gap(kind="dangling_reaction", label="theobromine")
    return EvidenceBundle(gap=gap, query="q", hits=[Retrieved(chunk=chunk, score=0.9)])


def _seed_with_dangling_theobromine() -> Dict[str, Any]:
    return {
        "entities": {
            "species": [{"name": "Pseudomonas putida"}],
            "compounds": [{"name": "caffeine"}, {"name": "theobromine"}],
            "proteins": [{"name": "NdmA"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "R1 caffeine demethylation",
                    "provenance": "extracted",
                    "inputs": [{"name": "caffeine"}],
                    "outputs": [{"name": "theobromine"}],
                    "enzymes": [{"entity": "NdmA"}],
                }
            ]
        },
    }


# A dict seed context so seed reactions carry provenance (see the D1 fix).
_SEED_CTX = {"source": {"source_id": "seed_paper", "source_type": "paper"}}


def _ndmb_prose_extractor(text: str) -> List[Dict[str, Any]]:
    if "NdmB" in text:
        return [
            {
                "name": "theobromine N3-demethylation",
                "inputs": [{"name": "theobromine", "stoichiometry": None}],
                "outputs": [
                    {"name": "7-methylxanthine", "stoichiometry": None},
                    {"name": "formaldehyde", "stoichiometry": None},
                ],
                "enzymes": ["NdmB"],
                "reversible": False,
            }
        ]
    return []


def _outputs_str(reaction: Dict[str, Any]) -> str:
    return str(reaction.get("outputs"))


def test_prose_extractor_adds_evidence_bound_reaction_to_synthesis() -> None:
    payload = synthesize(
        _seed_with_dangling_theobromine(),
        [_prose_bundle()],
        _SEED_CTX,
        prose_extractor=_ndmb_prose_extractor,
    )
    reactions = payload["processes"]["reactions"]
    # The seed reaction survives AND the prose-extracted NdmB step was added.
    ndmb = [r for r in reactions if "7-methylxanthine" in _outputs_str(r)]
    assert len(ndmb) == 1
    # It is evidence-bound to the exact paper the passage came from.
    assert ndmb[0].get("rag_provenance", {}).get("source_id") == "PMC777001"
    # ...and it is catalyzed by the enzyme named in the prose.
    enzyme_names = {a.get("entity") for a in ndmb[0].get("enzymes", [])}
    assert "NdmB" in enzyme_names


def test_without_extractor_prose_yields_no_new_reaction() -> None:
    # Baseline / proof the extractor is doing the work: arrow-only synthesis of
    # the same prose bundle adds nothing, because the passage has no equation.
    payload = synthesize(
        _seed_with_dangling_theobromine(), [_prose_bundle()], _SEED_CTX
    )
    reactions = payload["processes"]["reactions"]
    assert not any("7-methylxanthine" in _outputs_str(r) for r in reactions)


def test_prose_extractor_called_once_per_chunk() -> None:
    # The two passes over the bundles (main synthesis + unfilled-gap detection)
    # must not double-call the model — extraction is memoized per chunk.
    calls: List[str] = []

    def _counting(text: str) -> List[Dict[str, Any]]:
        calls.append(text)
        return _ndmb_prose_extractor(text)

    synthesize(
        _seed_with_dangling_theobromine(),
        [_prose_bundle()],
        _SEED_CTX,
        prose_extractor=_counting,
    )
    assert len(calls) == 1  # one unique chunk -> exactly one extraction call

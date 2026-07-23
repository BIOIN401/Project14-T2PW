"""Regression: a reversible reaction given as an explicit forward + reverse pair
must keep BOTH directions through conflict resolution and end-to-end synthesis.

Bug (fixed): ``_Reaction.conflict_key`` keyed on the UNORDERED set of ALL
participants (inputs + outputs merged), so a forward reaction and its reverse
(identical participants, inputs/outputs swapped) produced the SAME key.
``_resolve_reactions`` then treated the two directions as conflicting variants of
one reaction, kept only the heavier one and DROPPED the other. In the field this
silently deleted a *locked* reverse reaction, tripping the post-normalization
locked-reaction accounting gate ("1 locked reaction(s) are neither active nor
quarantined").

Fix: ``conflict_key`` is now direction-aware — keyed on the (sorted inputs,
sorted outputs) PAIR — so opposite directions key to distinct reactions and both
survive, mirroring the core ``dedupe_processes`` reaction key. Same-direction
disagreements (identical inputs->outputs) are still grouped and resolved.

Offline / deterministic: no chromadb, no network, no live LLM. The ``openai``
stub mirrors ``tests/test_rag_synthesize.py`` because importing
``t2pw.rag.retrieve`` (for the WP4 ``EvidenceBundle`` / ``Gap`` fixtures) pulls
``openai`` in at import time.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

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

from t2pw.pipeline.stage_contracts import validate_post_extraction  # noqa: E402
from t2pw.rag.synthesize import (  # noqa: E402
    _Participant,
    _Reaction,
    _resolve_reactions,
    synthesize_with_report,
)


def _rxn(name, inputs, outputs, *, scores=None, source_id="seed", chunk_id="c1"):
    return _Reaction(
        name=name,
        inputs=[_Participant(n) for n in inputs],
        outputs=[_Participant(n) for n in outputs],
        enzymes=["LpxA"],
        provenance=[{"source_id": source_id, "chunk_id": chunk_id}],
        scores=list(scores or []),
    )


def _names(participants) -> list:
    return [p if isinstance(p, str) else p.get("name") for p in participants]


# ---------------------------------------------------------------------------
# Unit level — _resolve_reactions.
# ---------------------------------------------------------------------------
def test_forward_and_reverse_pair_both_survive_resolution():
    """A forward reaction and its exact reverse both survive; neither is a conflict."""
    fwd = _rxn(
        "LpxA reaction",
        inputs=["UDP-N-acetylglucosamine", "beta-hydroxymyristoyl-ACP"],
        outputs=["UDP-3-O-acyl-N-acetylglucosamine", "ACP"],
        scores=[0.9, 0.8],
    )
    rev = _rxn(
        "LpxA reverse reaction",
        inputs=["UDP-3-O-acyl-N-acetylglucosamine", "ACP"],
        outputs=["UDP-N-acetylglucosamine", "beta-hydroxymyristoyl-ACP"],
        scores=[],  # lighter evidence — would have been the dropped one pre-fix
    )

    resolved, conflicts = _resolve_reactions([fwd, rev])

    assert len(resolved) == 2, "both directions must survive (reverse was dropped pre-fix)"
    assert {r.name for r in resolved} == {"LpxA reaction", "LpxA reverse reaction"}
    assert conflicts == [], "opposite directions are distinct reactions, not a conflict"

    # Direction-aware keys really are distinct (guards against a merged-set regression).
    assert fwd.conflict_key() != rev.conflict_key()


def test_same_direction_duplicates_still_merge():
    """Control: identical same-direction reactions still dedupe to one (no over-split)."""
    d1 = _rxn("A to B", inputs=["A"], outputs=["B"], scores=[0.5],
              source_id="PMID:1", chunk_id="c1")
    d2 = _rxn("A -> B", inputs=["A"], outputs=["B"], scores=[0.4],
              source_id="PMID:2", chunk_id="c2")

    resolved, conflicts = _resolve_reactions([d1, d2])

    assert len(resolved) == 1, "same inputs->outputs is one reaction — still merged"
    assert conflicts == []
    # Provenance from both duplicates is unioned onto the survivor.
    assert d1.conflict_key() == d2.conflict_key()
    assert len(resolved[0].provenance) == 2


# ---------------------------------------------------------------------------
# End-to-end — synthesize_with_report over a minimal seed payload.
# ---------------------------------------------------------------------------
def _seed_context() -> dict:
    return {
        "text": "LpxA acylation (reversible), seed paper",
        "source": {
            "source_id": "PMID:LPXA",
            "source_title": "LpxA reversible acylation",
            "source_type": "paper",
            "source_uri": "https://example.org/lpxa",
            "organism": "Escherichia coli",
        },
    }


def _reversible_seed_payload() -> dict:
    """Minimal seed carrying an explicit forward + reverse LpxA pair.

    Both rows fall back to the seed source pointer for provenance (see
    ``_seed_reactions``) so neither is omitted as unsupported.
    """
    return {
        "entities": {
            "species": [{"name": "Escherichia coli"}],
            "compounds": [
                {"name": "UDP-N-acetylglucosamine"},
                {"name": "beta-hydroxymyristoyl-ACP"},
                {"name": "UDP-3-O-acyl-N-acetylglucosamine"},
                {"name": "ACP"},
            ],
            "proteins": [{"name": "LpxA"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "LpxA reaction",
                    "inputs": ["UDP-N-acetylglucosamine", "beta-hydroxymyristoyl-ACP"],
                    "outputs": ["UDP-3-O-acyl-N-acetylglucosamine", "ACP"],
                    "enzymes": [{"protein": "LpxA"}],
                    "reversible": False,
                },
                {
                    "name": "LpxA reverse reaction",
                    "inputs": ["UDP-3-O-acyl-N-acetylglucosamine", "ACP"],
                    "outputs": ["UDP-N-acetylglucosamine", "beta-hydroxymyristoyl-ACP"],
                    "enzymes": [{"protein": "LpxA"}],
                    "reversible": False,
                },
            ]
        },
    }


def test_reversible_pair_survives_end_to_end():
    result = synthesize_with_report(
        _reversible_seed_payload(), [], _seed_context()
    )
    payload = result.payload

    # Still a valid seam-S3 payload.
    assert validate_post_extraction(payload)["ok"] is True

    reactions = payload["processes"]["reactions"]
    names = {r["name"] for r in reactions}
    assert "LpxA reaction" in names
    assert "LpxA reverse reaction" in names
    assert len(reactions) == 2, "both directions must be exported, not collapsed"

    # No direction conflict was recorded (the pair is two reactions, not a conflict).
    assert result.conflicts == []

    # The two reactions really are opposite directions of the same participants.
    fwd = next(r for r in reactions if r["name"] == "LpxA reaction")
    rev = next(r for r in reactions if r["name"] == "LpxA reverse reaction")
    assert set(_names(fwd["inputs"])) == set(_names(rev["outputs"]))
    assert set(_names(fwd["outputs"])) == set(_names(rev["inputs"]))

"""Tests for the RAG conform-and-merge seam (multi-paper payload fix).

The multi-paper RAG synthesis used to REPLACE the rich single-paper (Stage-1)
payload with its own thin, reaction-only one, which crashed the post-pipeline
audit / DB-mapping with ``'NoneType' object is not subscriptable``. The fix
CONFORMS the synthesized payload to a Stage-2 additions envelope
(:func:`t2pw.rag.conform.conform_rag_additions_for_merge`) and MERGES it into the
seed with :func:`t2pw.pipeline.pipeline.merge_additions` — the same additive path
Stage 2 uses — so the post-pipeline always sees a Stage-1-shaped payload.

Offline / deterministic: no chromadb, no network, no live LLM. The ``openai`` stub
mirrors the other RAG tests because importing the pipeline pulls the mapping/LLM
stack at import time.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

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

from t2pw.pipeline.payload_models import validate_payload_shape  # noqa: E402
from t2pw.pipeline.pipeline import merge_additions  # noqa: E402
from t2pw.rag.conform import conform_rag_additions_for_merge  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
def _seed_payload() -> dict:
    """A rich single-paper (Stage-1) payload — element_locations, biological_states,
    species scaffolding, and reactions with string participants + string evidence.
    """
    return {
        "metadata": {"pathway_name": "caffeine degradation"},
        "entities": {
            "species": [{"name": "Pseudomonas putida"}],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [
                {"name": "caffeine"},
                {"name": "paraxanthine"},
                {"name": "7-methylxanthine"},
            ],
            "proteins": [{"name": "NdmA"}],
        },
        "biological_states": [
            {
                "name": "cytosol",
                "species": "Pseudomonas putida",
                "subcellular_location": "cytosol",
            }
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": "caffeine", "biological_state": "cytosol"}
            ]
        },
        "processes": {
            "reactions": [
                {
                    "name": "R1 caffeine demethylation",
                    "inputs": ["caffeine"],
                    "outputs": ["paraxanthine"],
                    "evidence": "caffeine is demethylated to paraxanthine",
                },
                {
                    "name": "R2 paraxanthine demethylation",
                    "inputs": ["paraxanthine"],
                    "outputs": ["7-methylxanthine"],
                },
            ]
        },
    }


def _rag_payload() -> dict:
    """A thin RAG-synthesized payload: evidence LISTs, a stoichiometric dict
    participant, and the carried-forward scaffolding buckets synthesis copies from
    the seed (species / subcellular_locations) which must NOT be re-added.
    """
    return {
        "entities": {
            # Scaffolding carried forward by synthesis — must be EXCLUDED from
            # the additions envelope (already on the seed).
            "species": [{"name": "Pseudomonas putida"}],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [
                {
                    "name": "theobromine",
                    "rag_provenance": {"source_id": "PMID:0002"},
                    "source_refs": ["PMID:0002"],
                    "rag_confidence": 0.8,
                },
                {"name": "3-methylxanthine", "rag_provenance": {"source_id": "PMID:0002"}},
            ],
            "proteins": [{"name": "NdmB", "rag_provenance": {"source_id": "PMID:0002"}}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "R4 theobromine demethylation",
                    "inputs": [{"name": "theobromine", "stoichiometry": 2}],
                    "outputs": ["3-methylxanthine", "formaldehyde"],
                    "enzymes": [
                        {
                            "entity": "NdmB",
                            "entity_type": "protein",
                            "role": "catalyst",
                            "evidence": [
                                {"text": "NdmB catalyzes the demethylation"},
                                {"text": "in Pseudomonas putida"},
                            ],
                        }
                    ],
                    "evidence": [
                        {"text": "theobromine is demethylated"},
                        {"text": "to 3-methylxanthine by NdmB"},
                    ],
                    "rag_provenance": {"source_id": "PMID:0002"},
                    "source_refs": ["PMID:0002"],
                    "rag_confidence": 0.9,
                }
            ]
        },
    }


def _has_list_evidence(node: Any) -> bool:
    """True if any ``evidence`` field anywhere in the payload is a list."""
    if isinstance(node, dict):
        if isinstance(node.get("evidence"), list):
            return True
        return any(_has_list_evidence(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_list_evidence(item) for item in node)
    return False


# ---------------------------------------------------------------------------
# Test 1 — conform coerces evidence + participants and excludes scaffolding.
# ---------------------------------------------------------------------------
def test_conform_coerces_evidence_and_participants_and_excludes_scaffolding():
    envelope = conform_rag_additions_for_merge(_rag_payload())
    additions = envelope["additions"]

    # Scaffolding buckets are excluded (already live on the seed).
    entities = additions["entities"]
    assert "species" not in entities
    assert "subcellular_locations" not in entities
    assert {c["name"] for c in entities["compounds"]} == {"theobromine", "3-methylxanthine"}
    assert {p["name"] for p in entities["proteins"]} == {"NdmB"}

    reaction = additions["processes"]["reactions"][0]

    # evidence coerced list -> string, on the reaction AND the enzyme actor.
    assert isinstance(reaction["evidence"], str)
    assert reaction["evidence"]
    assert isinstance(reaction["enzymes"][0]["evidence"], str)
    assert reaction["enzymes"][0]["evidence"]

    # Participants coerced to the core plain-name string shape (dict -> name).
    assert reaction["inputs"] == ["theobromine"]
    assert reaction["outputs"] == ["3-methylxanthine", "formaldehyde"]
    assert all(isinstance(p, str) for p in reaction["inputs"])
    assert all(isinstance(p, str) for p in reaction["outputs"])

    # The source RAG payload is not mutated.
    assert isinstance(_rag_payload()["processes"]["reactions"][0]["evidence"], list)


# ---------------------------------------------------------------------------
# Test 2 — merge injects RAG reactions while retaining the seed's rich shape.
# ---------------------------------------------------------------------------
def test_merge_retains_seed_shape_and_adds_rag_reactions():
    seed = _seed_payload()
    envelope = conform_rag_additions_for_merge(_rag_payload())
    merged = merge_additions(seed, envelope)

    # (a) The seed's rich Stage-1 scaffolding survives.
    assert merged["element_locations"]["compound_locations"]
    assert merged["biological_states"]
    assert any(s["name"] == "Pseudomonas putida" for s in merged["entities"]["species"])

    # (b) Seed reactions PLUS the new RAG reaction.
    names = {r["name"] for r in merged["processes"]["reactions"]}
    assert "R1 caffeine demethylation" in names
    assert "R2 paraxanthine demethylation" in names
    assert "R4 theobromine demethylation" in names

    # (c) No evidence field is typed as a list anywhere in the merged payload.
    assert not _has_list_evidence(merged)

    # (d) Passes the strict runtime-schema validator at the post-extraction boundary.
    report = validate_payload_shape(merged, boundary="post_extraction", mode="report")
    assert report["ok"], report["errors"]


# ---------------------------------------------------------------------------
# Test 3 — a RAG payload that only duplicates seed reactions adds nothing.
# ---------------------------------------------------------------------------
def test_merge_of_duplicate_only_rag_adds_nothing():
    seed = _seed_payload()
    seed_reaction_count = len(seed["processes"]["reactions"])

    rag_dup = {
        "entities": {"compounds": [{"name": "caffeine"}]},
        "processes": {
            "reactions": [
                {
                    "name": "R1 duplicate",
                    "inputs": ["caffeine"],
                    "outputs": ["paraxanthine"],
                    "evidence": [{"text": "same reaction, restated by another paper"}],
                }
            ]
        },
    }

    merged = merge_additions(seed, conform_rag_additions_for_merge(rag_dup))

    # No new reaction row was added (the duplicate keyed to the existing R1).
    assert len(merged["processes"]["reactions"]) == seed_reaction_count
    names = {r["name"] for r in merged["processes"]["reactions"]}
    assert names == {"R1 caffeine demethylation", "R2 paraxanthine demethylation"}

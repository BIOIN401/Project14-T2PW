"""A RAG-sourced REACTION must keep its ``rag_provenance`` through cleaning.

Why this file exists
--------------------
``pipeline._clean_processes`` rebuilds every process row from a key whitelist,
and that whitelist named no RAG carrier. ``_clean_entities``, by contrast, copies
an entity row key-for-key. The asymmetry is measurable in the reference run of
2026-07-28. In
``runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict/merged_payload.json``:

* the key union across all 27 delivered reactions is exactly
  ``('biological_state', 'enzymes', 'evidence', 'inputs', 'locked_reaction_id',
  'name', 'outputs')`` -- plus ``preservation_status`` /
  ``repaired_missing_compound_entities`` on 3 rows, added by a later repair stage.
  ZERO reactions carry ``rag_provenance``;
* 41 of 56 compounds and 18 of 31 proteins DO carry it, 35 of them pointing at
  ``source_id`` PMC12898747 and 2 at PMC11046580.

So the shipped payload proves cross-paper ENTITIES were imported, while making it
impossible to attribute one delivered REACTION to the paper it came from. That is
what blocks the scope-creep work: a gap-relevance filter over imported reactions
cannot be evaluated, and "how much of this pathway came from another paper?" has
no answer -- a question the same run makes urgent, since PMC13278307 (strict)
delivered 14 reactions and none of them belonged to the lipid A pathway named in
its own focus box.

What is pinned here
-------------------
1. A RAG reaction keeps ``rag_provenance`` / ``source_papers`` / ``rag_confidence``
   through ``_clean_processes`` -- directly, and through the real seam
   (``rag.conform`` -> ``merge_additions``) the pipeline actually uses.
2. Transports and interactions keep it too (same structural defect, four buckets).
3. Nothing else moved: a plain non-RAG reaction row gains no key and keeps its
   exact key ORDER; reaction identity, ordering and count are unchanged; and the
   locked-reaction preservation report is byte-identical with and without the
   carriers, which is the machinery most at risk from a new reaction key
   (``reaction_preservation_validator._score_candidate`` mixes an evidence signal
   into its match score).

Offline / deterministic: no chromadb, no network, no live LLM. The ``openai`` stub
mirrors tests/test_rag_conform_merge.py because importing the pipeline pulls the
mapping/LLM stack at import time.
"""

from __future__ import annotations

import copy
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

from t2pw.pipeline.payload_models import validate_payload_shape  # noqa: E402
from t2pw.pipeline.pipeline import (  # noqa: E402
    _RAG_ROW_CARRIER_KEYS,
    _clean_processes,
    clean_stage_one,
    merge_additions,
)
from t2pw.pipeline.reaction_preservation_validator import (  # noqa: E402
    validate_reaction_preservation,
)
from t2pw.rag.conform import conform_rag_additions_for_merge  # noqa: E402


# The pointer shape observed on the imported entities of the reference run: the
# entity rows of PMC12444477's strict merged payload carry exactly this dict under
# ``rag_provenance``, 35 of them with source_id PMC12898747.
_IMPORTED_POINTER = {
    "source_id": "PMC12898747",
    "source_title": "Lipid A modification systems in Gram-negative bacteria",
    "source_type": "paper",
    "source_uri": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12898747/",
    "section": "results",
    "chunk_id": "PMC12898747::results::0007",
}


def _rag_processes() -> dict:
    """One imported reaction, one imported transport, one imported interaction.

    Every row carries the three namespaced carriers plus the core keys the
    whitelist already emitted, so the test can tell "the carrier survived" apart
    from "the row survived".
    """
    return {
        "reactions": [
            {
                "name": "lipid IVA palmitoylation",
                "inputs": ["lipid IVA", "palmitoyl-ACP"],
                "outputs": ["hepta-acylated lipid A", "ACP"],
                "locked_reaction_id": "rxn-0007",
                "biological_state": "outer membrane",
                "evidence": "PagP transfers a palmitate to lipid A.",
                "enzymes": [{"protein": "PagP", "evidence": "PagP transfers a palmitate."}],
                "rag_provenance": dict(_IMPORTED_POINTER),
                "source_papers": [
                    {"source_id": "PMC12898747", "title": "Lipid A modification systems"}
                ],
                "rag_confidence": 0.82,
            },
            {
                # No RAG carrier at all -- the seed-paper reaction. Must come out
                # exactly as it did before this change.
                "name": "seed reaction",
                "inputs": ["KDO2-lipid A"],
                "outputs": ["lipid A"],
                "evidence": "stated by the seed paper",
            },
        ],
        "transports": [
            {
                "name": "lipid A flipping",
                "cargo": "lipid A",
                "from_biological_state": "inner leaflet",
                "to_biological_state": "outer leaflet",
                "transporters": [{"protein": "MsbA"}],
                "rag_provenance": dict(_IMPORTED_POINTER),
                "rag_confidence": 0.6,
            }
        ],
        "interactions": [
            {
                "entity_1": "PhoP",
                "relationship": "activates",
                "entity_2": "pmrD",
                "rag_provenance": dict(_IMPORTED_POINTER),
            }
        ],
    }


def _rag_payload_for_merge() -> dict:
    """A thin synthesis-shaped payload: list evidence, carriers on the reaction."""
    return {
        "entities": {
            "compounds": [
                {"name": "lipid IVA", "rag_provenance": dict(_IMPORTED_POINTER)},
                {"name": "hepta-acylated lipid A", "rag_provenance": dict(_IMPORTED_POINTER)},
            ],
            "proteins": [{"name": "PagP", "rag_provenance": dict(_IMPORTED_POINTER)}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "lipid IVA palmitoylation",
                    "inputs": [{"name": "lipid IVA", "stoichiometry": 1}],
                    "outputs": ["hepta-acylated lipid A"],
                    "evidence": [
                        {"text": "PagP transfers a palmitate to lipid A.", "source_id": "PMC12898747"}
                    ],
                    "enzymes": [{"protein": "PagP", "evidence": [{"text": "PagP transfers"}]}],
                    "rag_provenance": dict(_IMPORTED_POINTER),
                    "source_papers": [{"source_id": "PMC12898747", "title": "Lipid A modification"}],
                    "rag_confidence": 0.82,
                }
            ]
        },
    }


def _seed_payload() -> dict:
    """A rich single-paper (Stage-1) payload for the merge seam."""
    return {
        "metadata": {"pathway_name": "regulation of lipid A biosynthesis"},
        "entities": {
            "species": [{"name": "Escherichia coli"}],
            "subcellular_locations": [{"name": "outer membrane"}],
            "compounds": [{"name": "KDO2-lipid A"}, {"name": "lipid A"}],
            "proteins": [{"name": "LpxC"}],
        },
        "biological_states": [
            {
                "name": "outer membrane",
                "species": "Escherichia coli",
                "subcellular_location": "outer membrane",
            }
        ],
        "processes": {
            "reactions": [
                {
                    "name": "KDO2-lipid A deacylation",
                    "inputs": ["KDO2-lipid A"],
                    "outputs": ["lipid A"],
                    "evidence": "the seed paper states this step",
                }
            ]
        },
    }


# ---------------------------------------------------------------------------
# 1 — the defect itself: a RAG reaction keeps its pointer through cleaning.
# ---------------------------------------------------------------------------
def test_rag_sourced_reaction_keeps_provenance_through_cleaning():
    cleaned = _clean_processes(_rag_processes())

    imported = next(
        r for r in cleaned["reactions"] if r["name"] == "lipid IVA palmitoylation"
    )
    assert imported["rag_provenance"]["source_id"] == "PMC12898747"
    assert imported["rag_provenance"]["chunk_id"] == "PMC12898747::results::0007"
    assert imported["source_papers"][0]["source_id"] == "PMC12898747"
    assert imported["rag_confidence"] == 0.82

    # The shape matches what an entity row carries (a dict, not a string), so a
    # reader does not have to branch on where the row came from.
    assert isinstance(imported["rag_provenance"], dict)


def test_rag_sourced_transport_and_interaction_keep_provenance():
    cleaned = _clean_processes(_rag_processes())

    transport = cleaned["transports"][0]
    assert transport["rag_provenance"]["source_id"] == "PMC12898747"
    assert transport["rag_confidence"] == 0.6
    # The emit guard is unchanged: the transport is still carried by its cargo.
    assert transport["cargo"] == "lipid A"

    interaction = cleaned["interactions"][0]
    assert interaction["rag_provenance"]["source_id"] == "PMC12898747"


def test_rag_reaction_keeps_provenance_through_the_real_conform_merge_seam():
    """The path a real run takes: synthesize -> conform -> merge_additions.

    ``rag.conform`` copies the row key-for-key (``dict(row)``) and only coerces
    evidence/participants, so the carrier reaches ``clean_inference_output`` -- and
    used to die there. This is the assertion that would have failed on the payload
    shipped by run 2026-07-28_0919.
    """
    merged = merge_additions(_seed_payload(), conform_rag_additions_for_merge(_rag_payload_for_merge()))

    imported = next(
        r for r in merged["processes"]["reactions"] if r["name"] == "lipid IVA palmitoylation"
    )
    assert imported["rag_provenance"]["source_id"] == "PMC12898747"
    assert imported["rag_confidence"] == 0.82

    # The seed's own reaction gained nothing.
    seed_reaction = next(
        r for r in merged["processes"]["reactions"] if r["name"] == "KDO2-lipid A deacylation"
    )
    assert not any(key in seed_reaction for key in _RAG_ROW_CARRIER_KEYS)

    # The merged payload still passes the runtime shape validator at the boundary
    # the S3 RAG-merge gate uses (``_RuntimeModel`` is ``extra="allow"``, so the
    # namespaced carriers are accepted rather than merely tolerated).
    report = validate_payload_shape(merged, boundary="post_extraction", mode="report")
    assert report["ok"], report["errors"]


# ---------------------------------------------------------------------------
# 2 — nothing else moved.
# ---------------------------------------------------------------------------
def test_non_rag_reaction_row_gains_no_key_and_keeps_key_order():
    """A seed-paper reaction must be byte-for-byte what it was before.

    Key ORDER is asserted, not just the key set: the carriers are appended last,
    so a row without them serializes exactly as it did in every payload written
    before this change.
    """
    cleaned = _clean_processes(_rag_processes())
    seed_reaction = next(r for r in cleaned["reactions"] if r["name"] == "seed reaction")

    assert list(seed_reaction.keys()) == ["inputs", "outputs", "name", "evidence"]
    assert not any(key in seed_reaction for key in _RAG_ROW_CARRIER_KEYS)


def test_reaction_identity_ordering_and_locked_id_are_untouched():
    """Same reactions, same order, same locked ids -- only the carriers are new."""
    with_carriers = _clean_processes(_rag_processes())

    stripped_input = copy.deepcopy(_rag_processes())
    for bucket in stripped_input.values():
        for row in bucket:
            for key in _RAG_ROW_CARRIER_KEYS:
                row.pop(key, None)
    without_carriers = _clean_processes(stripped_input)

    assert [r["name"] for r in with_carriers["reactions"]] == [
        r["name"] for r in without_carriers["reactions"]
    ]
    for left, right in zip(with_carriers["reactions"], without_carriers["reactions"]):
        assert left["inputs"] == right["inputs"]
        assert left["outputs"] == right["outputs"]
        assert left.get("locked_reaction_id") == right.get("locked_reaction_id")
        assert left.get("enzymes") == right.get("enzymes")
        assert left.get("evidence") == right.get("evidence")
        # The ONLY difference between the two rows is the carrier set.
        assert set(left) - set(right) <= set(_RAG_ROW_CARRIER_KEYS)
        assert set(right) - set(left) == set()

    assert with_carriers["reactions"][0]["locked_reaction_id"] == "rxn-0007"


def test_locked_reaction_preservation_report_is_unchanged_by_the_carriers():
    """The preservation validator must score identically with and without them.

    ``reaction_preservation_validator._score_candidate`` mixes a 0.04-weighted
    evidence signal into its match score, and ``_evidence_text`` falls back to
    ``source_refs`` when a row has no evidence -- which is precisely why
    ``source_refs`` is NOT carried by ``_carry_rag_provenance`` while the three
    namespaced keys are. This pins that the validator cannot see the difference.
    """
    manifest = [
        {
            "locked_reaction_id": "rxn-0007",
            "name": "lipid IVA palmitoylation",
            "inputs": ["lipid IVA", "palmitoyl-ACP"],
            "outputs": ["hepta-acylated lipid A", "ACP"],
            "evidence": "PagP transfers a palmitate to lipid A.",
        }
    ]

    with_carriers = {"processes": _clean_processes(_rag_processes())}

    stripped_input = copy.deepcopy(_rag_processes())
    for bucket in stripped_input.values():
        for row in bucket:
            for key in _RAG_ROW_CARRIER_KEYS:
                row.pop(key, None)
    without_carriers = {"processes": _clean_processes(stripped_input)}

    report_with = validate_reaction_preservation(with_carriers, manifest, stage="test")
    report_without = validate_reaction_preservation(without_carriers, manifest, stage="test")

    assert report_with == report_without
    assert report_with["details"][0]["status"] == "preserved_exactly"


def test_clean_stage_one_carries_the_reaction_pointer_too():
    """``clean_stage_one`` is the other ``_clean_processes`` caller (line 1949).

    The RAG merge reaches ``_clean_processes`` through ``clean_inference_output``,
    but ``run_stage_one_with_chunking`` re-cleans the merged payload through
    ``clean_stage_one`` -- so both doors have to carry the pointer or a chunked run
    would silently drop it again.
    """
    cleaned = clean_stage_one(
        {
            "entities": {"compounds": [{"name": "lipid IVA", "rag_provenance": dict(_IMPORTED_POINTER)}]},
            "processes": _rag_processes(),
        }
    )

    imported = next(
        r for r in cleaned["processes"]["reactions"] if r["name"] == "lipid IVA palmitoylation"
    )
    entity = cleaned["entities"]["compounds"][0]

    # Reaction and entity now agree on the shape of the pointer.
    assert imported["rag_provenance"] == entity["rag_provenance"]

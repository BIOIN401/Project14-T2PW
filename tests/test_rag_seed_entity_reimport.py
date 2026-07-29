"""The seed paper's own entities must not be re-imported as duplicate rows.

The defect, measured on the delivered artifact
``runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/
strict/merged_payload.json``:

* ``entities.proteins``  — 31 rows for 22 distinct normalized names. All nine seed
  enzymes (LpxA, LpxB, LpxC, LpxD, LpxH, LpxK, LpxL, LpxM, WaaA) appear twice.
* ``entities.compounds`` — 56 rows for 43 distinct normalized names, 13 doubled.

The two copies are BYTE-IDENTICAL names, which is why the 2026-07-23 synonym
resolver never touched them — it collapses synonyms, and these are not synonyms.
Every second copy carries ``rag_provenance.source_id == "seed_paper"``: it is the
seed's own entity rebuilt by RAG synthesis (which resolves the seed's reactions
alongside the retrieved ones) and then APPENDED by ``merge_additions``, whose
``_extend_unique`` dedupes on ``json.dumps(row, sort_keys=True)`` — and
``{"name": "LpxA", "class": "protein", "provenance": "extracted", ...}`` is a
different signature from ``{"name": "LpxA", "rag_provenance": {...}, ...}``.

The fix is at the merge boundary, in
:func:`t2pw.rag.conform.conform_rag_additions_for_merge`, which now takes the
payload the envelope is about to be merged INTO and drops any entity row that base
already registers. It is deliberately NOT in synthesis: the synthesized payload is
a standalone artifact whose entity buckets must cover its own reactions and whose
seed rows carry the corroborating citations the citation report prints (see
``_build_entities``' docstring and tests/test_rag_synthesize.py).

Identity is the registry gate's own (``process_normalizer._normalize`` /
``_entity_name_norms``), so "already registered" means exactly what
``validate_registry_references`` means by it.

Offline / deterministic: no chromadb, no network, no live LLM. The ``openai``
stub mirrors the other RAG tests because importing the pipeline pulls the
mapping/LLM stack at import time.
"""

from __future__ import annotations

import ast
import collections
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"
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

from t2pw.pipeline.pipeline import merge_additions  # noqa: E402
from t2pw.pipeline.process_normalizer import _normalize as registry_normalize  # noqa: E402
from t2pw.pipeline.process_normalizer import validate_registry_references  # noqa: E402
from t2pw.rag.conform import conform_rag_additions_for_merge  # noqa: E402
from t2pw.rag.synthesize import synthesize_with_report  # noqa: E402


# Exactly the buckets validate_registry_references unions into its registry.
REGISTRY_BUCKETS = ("compounds", "proteins", "protein_complexes", "nucleic_acids")


# ---------------------------------------------------------------------------
# Fixtures — the real PMC12444477 shape, trimmed to the reactions that doubled.
# ---------------------------------------------------------------------------
def _seed_payload() -> Dict[str, Any]:
    """Stage-1 shape for "The regulation of lipid A biosynthesis" (E. coli).

    Mirrors the real ``stage1_payload.json``: rich entity rows with ``class`` /
    ``confidence`` / ``provenance`` / ``organism``, and one participant
    ('lipid IV_A precursor') that the reactions reference but the entity buckets
    do NOT register. That omission is reproduced deliberately — it is the case
    only the merge base can resolve, and the case a naive "drop anything tagged
    seed_paper" rule would get wrong.
    """
    return {
        "entities": {
            "species": [{"name": "Escherichia coli"}],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [
                {"name": "UDP-GlcNAc", "class": "compound", "confidence": 1.0, "provenance": "extracted"},
                {"name": "UMP", "class": "compound", "confidence": 1.0, "provenance": "extracted"},
                {"name": "Kdo", "class": "compound", "confidence": 1.0, "provenance": "extracted"},
            ],
            "proteins": [
                {
                    "name": "LpxA",
                    "class": "protein",
                    "confidence": 1.0,
                    "provenance": "extracted",
                    "organism": "Escherichia coli",
                },
                {
                    "name": "LpxC",
                    "class": "protein",
                    "confidence": 1.0,
                    "provenance": "extracted",
                    "organism": "Escherichia coli",
                },
            ],
        },
        "biological_states": [
            {"name": "cytosol", "species": "Escherichia coli", "subcellular_location": "cytosol"}
        ],
        "processes": {
            "reactions": [
                {
                    "name": "LpxA reaction",
                    "inputs": ["UDP-GlcNAc"],
                    "outputs": ["UMP"],
                    "enzymes": [{"protein": "LpxA"}],
                    "evidence": "LpxA acylates UDP-GlcNAc",
                },
                {
                    "name": "LpxC reaction",
                    "inputs": ["UMP"],
                    "outputs": ["Kdo"],
                    "enzymes": [{"protein": "LpxC"}],
                    "evidence": "LpxC deacetylates the product",
                },
                {
                    # 'lipid IV_A precursor' is a participant with NO entity row.
                    "name": "LpxH reaction",
                    "inputs": ["Kdo"],
                    "outputs": ["lipid IV_A precursor"],
                    "evidence": "LpxH yields the lipid IV_A precursor",
                },
            ]
        },
    }


def _seed_context() -> Dict[str, Any]:
    return {
        "source": {
            "source_id": "seed_paper",
            "title": "The regulation of lipid A biosynthesis",
            "source_type": "paper",
        }
    }


def _stage_two_additions() -> Dict[str, Any]:
    """What Stage 2 added in the real run: the entity row Stage 1 forgot."""
    return {"additions": {"entities": {"compounds": [{"name": "lipid IV_A precursor"}]}}}


def _duplicate_names(payload: Dict[str, Any]) -> Dict[str, List[str]]:
    """Normalized name -> the raw names sharing it, for names listed twice.

    Unions every bucket ``validate_registry_references`` unions, because a species
    listed once as a compound and once as a protein is still two rows for one
    thing as far as the registry is concerned.
    """
    by_norm: Dict[str, List[str]] = collections.defaultdict(list)
    entities = payload.get("entities") or {}
    for bucket in REGISTRY_BUCKETS:
        for row in entities.get(bucket) or []:
            if isinstance(row, dict) and isinstance(row.get("name"), str):
                by_norm[registry_normalize(row["name"])].append(row["name"])
    return {norm: names for norm, names in by_norm.items() if norm and len(names) > 1}


def _norms(payload: Dict[str, Any]) -> set:
    entities = payload.get("entities") or {}
    return {
        registry_normalize(r["name"])
        for bucket in REGISTRY_BUCKETS
        for r in entities.get(bucket) or []
        if isinstance(r, dict) and isinstance(r.get("name"), str)
    }


# ---------------------------------------------------------------------------
# 1 — the end-to-end regression: seed -> synthesize -> conform -> merge.
# ---------------------------------------------------------------------------
def test_seed_entities_are_not_reimported_as_duplicate_rows():
    """The exact PMC12444477 path, with no retrieved papers at all.

    With zero evidence bundles every entity synthesis rebuilds is the seed's own,
    which is the purest form of the defect. Pre-fix this fixture merges to six
    doubled names (udpglcnac, ump, kdo, lipid iva precursor, lpxa, lpxc); the same
    mechanism produced 22 extra rows in the real strict payload and 55 across the
    nine ``merged_payload.json`` files under ``runs/``.
    """
    seed = _seed_payload()
    base = merge_additions(seed, _stage_two_additions())

    # Sanity: the base is clean before RAG merges into it.
    assert _duplicate_names(base) == {}

    result = synthesize_with_report(seed, [], _seed_context())
    merged = merge_additions(base, conform_rag_additions_for_merge(result.payload, base))

    assert _duplicate_names(merged) == {}, "seed entities were re-imported as second rows"

    # Not merely deduped — the row that survives is the SEED's rich one, with the
    # core ``class`` / ``provenance`` fields the downstream mapper reads. (A fix
    # that kept the thin RAG copy instead would pass a naive count check and then
    # break mapping.)
    lpxa = [p for p in merged["entities"]["proteins"] if p["name"] == "LpxA"]
    assert len(lpxa) == 1
    assert lpxa[0]["class"] == "protein"
    assert lpxa[0]["provenance"] == "extracted"
    assert "rag_provenance" not in lpxa[0]

    # Nothing the base registered was lost.
    assert _norms(base) <= _norms(merged)


def test_the_old_call_shape_is_what_produced_the_duplicates():
    """Guard against a silent regression of the wiring itself.

    ``conform_rag_additions_for_merge(payload)`` — the call shape that shipped run
    2026-07-28_0919 — still doubles every seed entity. This test exists so that if
    the merge seam is ever re-wired back to the one-argument form, a test says so
    instead of a corpus run.
    """
    seed = _seed_payload()
    base = merge_additions(seed, _stage_two_additions())
    result = synthesize_with_report(seed, [], _seed_context())

    unguarded = merge_additions(base, conform_rag_additions_for_merge(result.payload))
    assert set(_duplicate_names(unguarded)) == {
        "udpglcnac",
        "ump",
        "kdo",
        "lipid iva precursor",
        "lpxa",
        "lpxc",
    }


def test_merged_payload_still_passes_the_registry_gate():
    """Dropping rows must not strand a reaction reference.

    The one real hazard of this fix: 'lipid IV_A precursor' is referenced by a
    Stage-1 reaction, so if its row were dropped without the base carrying one,
    ``validate_registry_references`` would report
    ``/processes/reactions/2/outputs/0 unknown entity``.
    """
    seed = _seed_payload()
    base = merge_additions(seed, _stage_two_additions())
    result = synthesize_with_report(seed, [], _seed_context())
    merged = merge_additions(base, conform_rag_additions_for_merge(result.payload, base))

    validate_registry_references(merged)  # raises on failure


def test_the_synthesized_payload_itself_is_left_complete():
    """The pre-merge payload keeps every seed entity — that is not the seam.

    ``rag_result.payload`` feeds ``validate_provenance``, ``assign_tiers`` and the
    citation report, and its entity buckets have to cover its own reactions. The
    guard removes the duplicate ROW at the merge, not the synthesized entity.
    """
    seed = _seed_payload()
    result = synthesize_with_report(seed, [], _seed_context())
    entities = result.payload["entities"]

    assert {c["name"] for c in entities["compounds"]} >= {"UDP-GlcNAc", "UMP", "Kdo"}
    assert {p["name"] for p in entities["proteins"]} == {"LpxA", "LpxC"}
    for row in entities["compounds"] + entities["proteins"]:
        assert row["source_refs"] == ["seed_paper"]


# ---------------------------------------------------------------------------
# 2 — corroboration survives: only the duplicate ROW goes.
# ---------------------------------------------------------------------------
def test_corroborating_second_paper_is_not_dropped_from_the_pre_merge_payload():
    """The seed is in the RAG index on purpose — its claims must stay quotable.

    In the real run 'LpxC' carried ``source_refs == ["seed_paper", "PMC11046580"]``
    and 'WaaA' ``["seed_paper", "PMC12898747"]``: a retrieved paper independently
    corroborated a seed enzyme. Suppressing the duplicate merged row must not
    suppress that citation, which the citation report reads off the pre-merge
    payload.
    """
    seed = _seed_payload()
    base = merge_additions(seed, _stage_two_additions())
    rag_payload = {
        "entities": {
            "proteins": [
                {
                    "name": "LpxC",
                    "rag_provenance": {"source_id": "seed_paper"},
                    "source_refs": ["seed_paper", "PMC11046580"],
                    "rag_confidence": 0.9,
                }
            ]
        },
        "processes": {
            "reactions": [
                {
                    "name": "LpxC reaction",
                    "inputs": ["UMP"],
                    "outputs": ["Kdo"],
                    "enzymes": [{"entity": "LpxC", "entity_type": "protein", "role": "catalyst"}],
                    "rag_provenance": {"source_id": "PMC11046580"},
                    "source_refs": ["seed_paper", "PMC11046580"],
                }
            ]
        },
    }

    merged = merge_additions(base, conform_rag_additions_for_merge(rag_payload, base))

    # One LpxC row, and it is the seed's.
    assert [p["name"] for p in merged["entities"]["proteins"]] == ["LpxA", "LpxC"]
    # The corroborating citation is untouched in the pre-merge payload the
    # citation report renders.
    assert rag_payload["entities"]["proteins"][0]["source_refs"] == [
        "seed_paper",
        "PMC11046580",
    ]


def test_entities_only_a_retrieved_paper_states_are_still_added():
    """Guard against over-dropping — the fix must not swallow real contributions.

    Across ``runs/`` there are ``source_id == "seed_paper"`` rows that are the ONLY
    row for their species: 'PSA' and 'sulfacetamide' (PMC13231680), 'PEtN',
    'mcr genes', 'pmrCAB operon' (PMC13278307), 'α-ketoglutarate' (PMC12312563).
    They carry that tag because a seed reaction registered them first, not because
    the base has them — so a tag-based rule would have deleted all of them. Name
    identity against the base keeps them.
    """
    base = {"entities": {"compounds": [{"name": "Kdo"}], "proteins": [{"name": "LpxA"}]}}
    rag_payload = {
        "entities": {
            "compounds": [
                {"name": "Kdo", "rag_provenance": {"source_id": "seed_paper"}},
                {"name": "PSA", "rag_provenance": {"source_id": "seed_paper"}},
                {"name": "α-ketoglutarate", "rag_provenance": {"source_id": "seed_paper"}},
                {"name": "CDP-DAG", "rag_provenance": {"source_id": "PMC12898747"}},
            ],
            "proteins": [
                {"name": "LpxA", "rag_provenance": {"source_id": "seed_paper"}},
                {"name": "mcr genes", "rag_provenance": {"source_id": "seed_paper"}},
            ],
        }
    }

    entities = conform_rag_additions_for_merge(rag_payload, base)["additions"]["entities"]

    assert [c["name"] for c in entities["compounds"]] == [
        "PSA",
        "α-ketoglutarate",
        "CDP-DAG",
    ]
    assert [p["name"] for p in entities["proteins"]] == ["mcr genes"]


# ---------------------------------------------------------------------------
# 3 — identity is the registry gate's, including declared synonyms.
# ---------------------------------------------------------------------------
def test_a_declared_synonym_on_the_base_row_also_counts_as_registered():
    """``_entity_name_norms`` folds ``synonyms`` in, so this guard does too.

    A base compound named "NAD+" that declares "NAD" already satisfies the gate for
    a reaction that says "NAD"
    (tests/test_process_normalizer.py::test_validate_registry_references_recognizes
    _declared_synonyms), so adding a row named "NAD" would be that same duplicate
    in a different spelling.
    """
    base = {"entities": {"compounds": [{"name": "NAD+", "synonyms": ["NAD"]}]}}
    rag_payload = {"entities": {"compounds": [{"name": "NAD"}, {"name": "NADH"}]}}

    entities = conform_rag_additions_for_merge(rag_payload, base)["additions"]["entities"]
    assert [c["name"] for c in entities["compounds"]] == ["NADH"]


def test_registration_is_checked_across_every_registry_bucket():
    """A name the base filed as a nucleic acid still blocks a compound row.

    ``nucleic_acids`` joined the gate's registry union when
    ``enforce_entity_type_consistency`` began relocating 'pmrHFIJKLM operon' out of
    ``entities.compounds`` (it is a reaction input of PMC13278307 in run
    2026-07-28_0919). A guard that ignored the bucket would re-add it as a
    compound and hand the payload two rows for one operon.
    """
    base = {
        "entities": {
            "nucleic_acids": [{"name": "pmrHFIJKLM operon"}],
            "protein_complexes": [{"name": "LpxA/LpxC complex"}],
        }
    }
    rag_payload = {
        "entities": {
            "compounds": [{"name": "pmrHFIJKLM operon"}, {"name": "PEtN"}],
            "protein_complexes": [{"name": "LpxA/LpxC complex"}],
        }
    }

    entities = conform_rag_additions_for_merge(rag_payload, base)["additions"]["entities"]
    assert [c["name"] for c in entities["compounds"]] == ["PEtN"]
    assert "protein_complexes" not in entities


# ---------------------------------------------------------------------------
# 4 — FALSE-MERGE guard: distinct species must stay distinct.
# ---------------------------------------------------------------------------
def test_charge_states_phospho_forms_and_spacing_are_not_false_merged():
    """The registry normalizer keeps ``+`` and ``:``, and is purely lexical.

    A looser normalizer — ``t2pw.mapping.map_ids._normalize_name`` strips ``+`` —
    would collapse 'NAD+' into 'NAD' and 'Ca2+' into 'Ca2'. Audited against every
    ``merged_payload.json`` under ``runs/``: with this normalizer every collision
    in the whole corpus is between BYTE-IDENTICAL names; zero distinct names
    collapse. It under-merges instead ('lipid IV_A' vs 'lipid IV A' stay two rows),
    which is the safe direction here and the synonym resolver's problem, not this
    guard's.
    """
    base = {
        "entities": {
            "compounds": [{"name": "NAD+"}, {"name": "lipid IV_A"}, {"name": "Ca2+"}],
            "proteins": [{"name": "PhoP"}],
        }
    }
    rag_payload = {
        "entities": {
            "compounds": [
                {"name": "NAD"},
                {"name": "NADH"},
                {"name": "lipid IV A"},
                {"name": "Ca2"},
                {"name": "NAD+"},  # the identical one — dropped
            ],
            "proteins": [
                {"name": "phosphorylated PhoP"},
                {"name": "PhoP-P"},
                {"name": "  PhoP "},  # the identical one, padded — dropped
            ],
        }
    }

    entities = conform_rag_additions_for_merge(rag_payload, base)["additions"]["entities"]
    assert [c["name"] for c in entities["compounds"]] == [
        "NAD",
        "NADH",
        "lipid IV A",
        "Ca2",
    ]
    assert [p["name"] for p in entities["proteins"]] == ["phosphorylated PhoP", "PhoP-P"]


# ---------------------------------------------------------------------------
# 5 — shape / robustness of the guard itself.
# ---------------------------------------------------------------------------
def test_conform_without_a_base_still_dedupes_the_envelope_against_itself():
    """``seed_payload`` omitted = the old call shape, minus self-duplication.

    Nothing in the envelope should be able to introduce two rows for one
    normalized name even when the caller cannot supply the merge base.
    """
    rag_payload = {
        "entities": {
            "compounds": [
                {"name": "Kdo", "rag_provenance": {"source_id": "PMC1"}},
                {"name": " Kdo ", "rag_provenance": {"source_id": "PMC2"}},
                {"name": "Kdo-lipid A", "rag_provenance": {"source_id": "PMC2"}},
            ]
        }
    }
    entities = conform_rag_additions_for_merge(rag_payload)["additions"]["entities"]
    assert [c["name"] for c in entities["compounds"]] == ["Kdo", "Kdo-lipid A"]


def test_conform_never_mutates_its_inputs():
    base = {"entities": {"compounds": [{"name": "Kdo"}]}}
    rag_payload = {
        "entities": {
            "compounds": [
                {"name": "Kdo", "rag_provenance": {"source_id": "seed_paper"}},
                {"name": "UMP", "rag_provenance": {"source_id": "PMC1"}},
            ]
        }
    }
    conform_rag_additions_for_merge(rag_payload, base)

    assert len(rag_payload["entities"]["compounds"]) == 2
    assert base == {"entities": {"compounds": [{"name": "Kdo"}]}}


def test_conform_tolerates_a_junk_base():
    rag_payload = {"entities": {"compounds": [{"name": "UMP"}]}}
    for junk in (None, "", 0, [], {"entities": None}, {"entities": {"compounds": "nope"}}):
        entities = conform_rag_additions_for_merge(rag_payload, junk)["additions"]["entities"]
        assert [c["name"] for c in entities["compounds"]] == ["UMP"], junk


def test_reactions_are_unaffected_by_the_entity_guard():
    """The guard touches ``additions.entities`` only — reaction merging is
    ``_merge_reactions``' job and keeps working as before."""
    base = {"entities": {"compounds": [{"name": "Kdo"}]}, "processes": {"reactions": []}}
    rag_payload = {
        "entities": {"compounds": [{"name": "Kdo"}]},
        "processes": {
            "reactions": [
                {
                    "name": "R1",
                    "inputs": ["Kdo"],
                    "outputs": ["Kdo-lipid A"],
                    "evidence": [{"text": "Kdo is transferred"}],
                }
            ]
        },
    }
    additions = conform_rag_additions_for_merge(rag_payload, base)["additions"]

    assert "entities" not in additions  # every entity row was already registered
    assert additions["processes"]["reactions"][0]["inputs"] == ["Kdo"]
    assert additions["processes"]["reactions"][0]["evidence"] == "Kdo is transferred"


# ---------------------------------------------------------------------------
# 6 — the wiring. A guard the only caller does not use is not a fix.
# ---------------------------------------------------------------------------
def test_the_orchestrator_passes_the_merge_base_to_conform():
    """The S3 seam must hand conform the SAME object it hands ``merge_additions``.

    ``seed_payload`` defaults to ``None``, which is the pre-fix behaviour, so a
    caller that forgets it gets a green conform module and a merged payload with
    every duplicate still in it. That is not hypothetical: the guard landed on
    2026-07-28 with ``streamlit_app.py`` still calling
    ``conform_rag_additions_for_merge(rag_result.payload)``, and every number the
    fix claims -- 88 entity rows -> 66 on PMC12444477/strict, 279 -> 220 across the
    eight delivered ``merged_payload.json`` with a RAG side -- was unreachable in
    production until the second argument was added. Every unit test in this file
    passes the base explicitly, so none of them can catch that.

    Checked against the SOURCE rather than by running the orchestrator, because
    importing ``streamlit_app`` executes a Streamlit script. The assertion is
    stronger than "two arguments were passed": the name given to conform must be
    the same name given to ``merge_additions`` as its base. Passing the Stage-1
    seed there instead would type-check, run, and quietly let back every entity
    that Stage 2 registered and Stage 1 did not ('lipid IV_A precursor',
    'Kdo-lipid A precursor', 'tetra-acylated disaccharide intermediate' in that
    same run).
    """
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))

    def _calls(name: str) -> List[ast.Call]:
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]

    conform_calls = _calls("conform_rag_additions_for_merge")
    assert len(conform_calls) == 1, "the S3 seam is the only caller; found %d" % len(conform_calls)
    call = conform_calls[0]

    base_arg: Any = None
    if len(call.args) > 1:
        base_arg = call.args[1]
    else:
        base_arg = next((kw.value for kw in call.keywords if kw.arg == "seed_payload"), None)
    assert isinstance(base_arg, ast.Name), (
        "conform_rag_additions_for_merge must be given the merge base; without it "
        "the seed's own entities are re-imported as duplicate rows"
    )

    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and node.value is call
    )
    envelope_name = assignment.targets[0].id  # type: ignore[union-attr]

    merge_call = next(
        node
        for node in _calls("merge_additions")
        if len(node.args) > 1
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == envelope_name
    )
    assert isinstance(merge_call.args[0], ast.Name)
    assert merge_call.args[0].id == base_arg.id, (
        "conform was given %r but merge_additions merges into %r -- the guard can "
        "only remove a duplicate it can see in the actual base"
        % (base_arg.id, merge_call.args[0].id)
    )

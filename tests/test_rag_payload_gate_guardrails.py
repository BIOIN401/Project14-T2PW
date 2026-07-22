"""Guardrails keeping RAG-injected reactions exportable by the PWML gate.

Regression cover for the ``RAG-setup`` export failure: with RAG on, the
required-field gate rejected the run with ``reaction_missing_left_participants``
/ ``reaction_missing_right_participants`` / ``duplicate_reaction_enzyme_complex``
on reactions that only exist because RAG added them.

Three distinct defects produced those six errors, and each gets a test here:

1. The RAG reaction builders dropped a candidate only when **both** sides were
   empty, so a half-transcribed statement ("substrate mentioned, product never
   stated") was emitted as a one-sided reaction named ``"<substrate> -> ?"``.
2. The synthesized RAG payload replaced the merged payload *after*
   ``merge_additions`` had already run the post-merge hardening, so the RAG rows
   never saw ``_normalize_reaction_actors`` (actor dedup) or
   ``filter_unresolvable_reactions`` (empty-side removal) at all.
3. ``promote_catalysts`` deduped a promoted catalyst only against ``modifiers``,
   so promoting an actor already listed under ``enzymes`` produced the same
   protein complex twice — and emptied the reaction's input list on the way.

The last test is the coverage hole that let this ship: nothing previously fed a
RAG-shaped reaction through ``validate_required_pwml_contract``.
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
    fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **_: None)
            )

    class _FakeOpenAIError(Exception):
        pass

    fake_openai.OpenAI = _FakeOpenAI
    fake_openai.RateLimitError = _FakeOpenAIError
    fake_openai.APIError = _FakeOpenAIError
    fake_openai.APITimeoutError = _FakeOpenAIError
    fake_openai.AuthenticationError = _FakeOpenAIError
    fake_openai.BadRequestError = _FakeOpenAIError
    sys.modules["openai"] = fake_openai

import json  # noqa: E402

from t2pw.pipeline.pipeline import (  # noqa: E402
    _inject_name_based_modifiers,
    apply_post_merge_cleanup,
    merge_additions,
)
from t2pw.pipeline.process_normalizer import promote_catalysts  # noqa: E402
from t2pw.mapping.map_ids import (  # noqa: E402
    _rewrite_reaction_protein_enzymes_to_complexes,
)
from t2pw.pwml.ir import validate_required_pwml_contract  # noqa: E402
from t2pw.rag import extract as rag_extract  # noqa: E402
from t2pw.rag import synthesize as rag_synthesize  # noqa: E402
from t2pw.rag.acquire import CandidatePaper  # noqa: E402
from t2pw.rag.ingest import _split_sections, chunk_paper  # noqa: E402
from t2pw.rag.synthesize import _is_bibliography_text  # noqa: E402


class _MemoryCache:
    """Minimal cache stand-in for the ID-mapping helpers."""

    def __init__(self) -> None:
        self._store: dict = {}

    def get(self, key: str):  # noqa: ANN201
        return self._store.get(key)

    def set(self, key: str, value) -> None:  # noqa: ANN001
        self._store[key] = value


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _exportable_payload() -> dict:
    """A minimal payload that clears the required-field gate on its own."""
    return {
        # The orchestrator fills these in before invoking the gate.
        "metadata": {
            "name": "Test pathway",
            "pathway_name": "Test pathway",
            "subject": "Metabolic",
            "pathway_subject": "Metabolic",
            "description": "fixture",
        },
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [
                {
                    "name": "Glucose",
                    "pathbank_compound_id": 101,
                    "mapped_ids": {"chebi": "CHEBI:17234"},
                },
                {
                    "name": "Glucose 6-phosphate",
                    "pathbank_compound_id": 102,
                    "mapped_ids": {"chebi": "CHEBI:4170"},
                },
            ],
            "proteins": [
                {
                    "name": "Hexokinase",
                    "species": "Homo sapiens",
                    "pathbank_protein_id": 201,
                    "mapped_ids": {"uniprot": "P19367"},
                }
            ],
            "protein_complexes": [
                {
                    "name": "Hexokinase complex",
                    "species": "Homo sapiens",
                    "components": ["Hexokinase"],
                }
            ],
        },
        "biological_states": [
            {
                "name": "cytosol",
                "species": "Homo sapiens",
                "subcellular_location": "cytosol",
            }
        ],
        "processes": {
            "reactions": [
                {
                    "name": "Glucose phosphorylation",
                    "inputs": ["Glucose"],
                    "outputs": ["Glucose 6-phosphate"],
                    "biological_state": "cytosol",
                    "enzymes": [
                        {"entity": "Hexokinase complex", "entity_type": "protein_complex"}
                    ],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


def _error_codes(report: dict) -> set:
    return {error["code"] for error in report.get("errors", [])}


# --------------------------------------------------------------------------
# Defect 1 — one-sided reactions must never be built
# --------------------------------------------------------------------------
def test_prose_extraction_drops_reaction_missing_a_side() -> None:
    """A passage naming a substrate but no product yields no reaction.

    Previously this produced ``inputs=[X], outputs=[]``, which synthesis then
    named ``"X -> ?"`` — the shape of gate errors 1 and 2.
    """
    model_output = json.dumps(
        {
            "reactions": [
                {
                    "name": "",
                    "inputs": [{"name": "R-3-hydroxymyristoyl-ACP", "stoichiometry": None}],
                    "outputs": [],
                    "enzymes": ["LpxC"],
                    "reversible": False,
                },
                {
                    "name": "complete step",
                    "inputs": [{"name": "UDP-GlcNAc", "stoichiometry": None}],
                    "outputs": [{"name": "lipid X", "stoichiometry": None}],
                    "enzymes": ["LpxA"],
                    "reversible": False,
                },
            ]
        }
    )

    reactions = rag_extract.extract_reactions_from_text(
        "some passage", chat_fn=lambda _system, _user: model_output
    )

    assert [r["name"] for r in reactions] == ["complete step"]
    assert all(r["inputs"] and r["outputs"] for r in reactions)


def test_arrow_parser_drops_equation_missing_a_side() -> None:
    """An equation whose product side parses to nothing is not a reaction."""
    assert rag_synthesize._parse_reaction_line("UDP-N-acetylglucosamine ->") is None

    kept = rag_synthesize._parse_reaction_line("UDP-GlcNAc -> lipid X | enzyme: LpxA")
    assert kept is not None
    assert kept["inputs"] and kept["outputs"]


def test_extracted_reaction_builder_drops_one_sided_candidate() -> None:
    """``_reaction_from_extracted`` refuses a candidate missing either side."""
    prov = {"source_id": "PMC1", "source_uri": "https://example.org/1"}
    evidence = {"text": "...", "chunk_id": "c1"}
    paper = {"source_id": "PMC1"}

    one_sided = rag_synthesize._reaction_from_extracted(
        {"name": "", "inputs": [{"name": "UDP-GlcNAc"}], "outputs": [], "enzymes": []},
        prov,
        evidence,
        paper,
        0.9,
    )
    assert one_sided is None

    complete = rag_synthesize._reaction_from_extracted(
        {
            "name": "acylation",
            "inputs": [{"name": "UDP-GlcNAc"}],
            "outputs": [{"name": "lipid X"}],
            "enzymes": ["LpxA"],
        },
        prov,
        evidence,
        paper,
        0.9,
    )
    assert complete is not None
    assert complete.inputs and complete.outputs


# --------------------------------------------------------------------------
# Defect 2 — the adopted RAG payload must get the same post-merge hardening
# --------------------------------------------------------------------------
def test_post_merge_cleanup_removes_one_sided_reaction_and_dedupes_actors() -> None:
    """The hardening ``merge_additions`` applies also cleans a RAG payload.

    Mirrors the real failure: a one-sided reaction, and one reaction carrying the
    same protein complex twice under two different key-shapes (a locally promoted
    actor with ``provenance``/``confidence``, and a RAG actor with ``evidence``).
    """
    payload = _exportable_payload()
    payload["processes"]["reactions"].extend(
        [
            {
                "name": "R-3-hydroxymyristoyl-ACP -> ?",
                "inputs": ["Glucose"],
                "outputs": [],
                "biological_state": "cytosol",
                "rag_confidence": 0.7,
            },
            {
                "name": "duplicated actor reaction",
                "inputs": ["Glucose"],
                "outputs": ["Glucose 6-phosphate"],
                "biological_state": "cytosol",
                "enzymes": [
                    {
                        "entity": "Hexokinase complex",
                        "entity_type": "protein_complex",
                        "role": "catalyst",
                        "confidence": 1.0,
                        "provenance": "extracted",
                    },
                    {
                        "entity": "Hexokinase complex",
                        "entity_type": "protein_complex",
                        "role": "catalyst",
                        "evidence": "[{'text': 'from a retrieved paper'}]",
                    },
                ],
            },
        ]
    )

    cleaned, removed = apply_post_merge_cleanup(payload)

    names = [r["name"] for r in cleaned["processes"]["reactions"]]
    assert "R-3-hydroxymyristoyl-ACP -> ?" not in names
    assert removed == ["R-3-hydroxymyristoyl-ACP -> ?"]

    duplicated = next(
        r for r in cleaned["processes"]["reactions"] if r["name"] == "duplicated actor reaction"
    )
    actor_names = [
        row.get("protein_complex") or row.get("protein") or row.get("entity")
        for row in duplicated["enzymes"]
    ]
    assert actor_names == ["Hexokinase complex"]


def test_post_merge_cleanup_is_idempotent_on_a_clean_payload() -> None:
    """Hardening a already-valid payload changes nothing and removes nothing."""
    payload = _exportable_payload()
    cleaned, removed = apply_post_merge_cleanup(payload)

    assert removed == []
    assert [r["name"] for r in cleaned["processes"]["reactions"]] == [
        "Glucose phosphorylation"
    ]
    assert validate_required_pwml_contract(cleaned, strict_db=True)["ok"]


def test_cleanup_tolerates_rag_list_shaped_evidence() -> None:
    """Core cleanup must not assume ``evidence`` is a string.

    RAG stores a *list* of evidence records on the row under the same key core
    stages use for a sentence; reading it as text used to raise ``AttributeError``.
    """
    payload = _exportable_payload()
    payload["processes"]["reactions"][0]["evidence"] = [
        {"text": "Hexokinase phosphorylates glucose.", "source_id": "PMC1"},
        {"text": "A second retrieved passage.", "source_id": "PMC2"},
    ]

    cleaned, removed = apply_post_merge_cleanup(payload)

    assert removed == []
    assert validate_required_pwml_contract(cleaned, strict_db=True)["ok"]


def test_name_injection_ignores_rag_evidence_corpus() -> None:
    """The Stage-2 name heuristic must not mine a retrieved corpus.

    ``_inject_name_based_modifiers`` substring-matches every declared protein
    against a row's evidence. A RAG row's evidence is most of a paper, so honouring
    it attached every enzyme to every reaction — 99 spurious
    ``reaction_enzyme_must_be_protein_complex`` errors in the real payload.
    """
    payload = _exportable_payload()
    payload["entities"]["proteins"].append(
        {"name": "Phosphofructokinase", "species": "Homo sapiens"}
    )
    reaction = payload["processes"]["reactions"][0]
    reaction["modifiers"] = []
    reaction["evidence"] = [
        {"text": "Phosphofructokinase acts elsewhere in glycolysis.", "source_id": "PMC1"}
    ]

    _inject_name_based_modifiers(payload)

    injected = {
        str(row.get("entity") or "")
        for row in payload["processes"]["reactions"][0].get("modifiers", [])
    }
    assert "Phosphofructokinase" not in injected


def test_merge_additions_still_applies_the_name_heuristic() -> None:
    """Factoring the hardening out must not cost ``merge_additions`` its heuristic."""
    base = _exportable_payload()
    base["processes"]["reactions"][0]["modifiers"] = []
    base["processes"]["reactions"][0]["evidence"] = (
        "Glucose is phosphorylated by Hexokinase complex in the cytosol."
    )

    merged = merge_additions(base, {})

    actors = {
        str(row.get("protein_complex") or row.get("protein") or row.get("entity") or "")
        for row in merged["processes"]["reactions"][0].get("enzymes", [])
    }
    assert "Hexokinase complex" in actors


# --------------------------------------------------------------------------
# Defect 3 — catalyst promotion must not duplicate nor hollow out a reaction
# --------------------------------------------------------------------------
def test_promote_catalysts_does_not_duplicate_an_actor_already_in_enzymes() -> None:
    """Promotion dedupes against ``enzymes``, not just ``modifiers``.

    The real run promoted a protein already listed under ``enzymes`` into
    ``modifiers``, so the gate saw the complex twice (errors 5 and 6).
    """
    payload = _exportable_payload()
    reaction = payload["processes"]["reactions"][0]
    reaction["inputs"] = ["Glucose", "Hexokinase complex"]
    reaction["enzymes"] = [
        {
            "entity": "Hexokinase complex",
            "entity_type": "protein_complex",
            "role": "catalyst",
            "evidence": "[{'text': 'retrieved'}]",
        }
    ]
    reaction["modifiers"] = []

    promote_catalysts(payload)

    modifier_names = [
        row.get("entity") for row in payload["processes"]["reactions"][0]["modifiers"]
    ]
    assert modifier_names.count("Hexokinase complex") == 0, (
        "the complex is already an enzyme; promoting it again duplicates the actor"
    )


def test_promote_catalysts_never_empties_a_reactions_inputs() -> None:
    """Stripping every protein-like input must not leave the reaction hollow.

    Reaction 23 in the failing run reached the gate with ``inputs == []`` purely
    because promotion removed its only input.
    """
    payload = _exportable_payload()
    reaction = payload["processes"]["reactions"][0]
    reaction["inputs"] = ["Hexokinase complex"]
    reaction["modifiers"] = []

    promote_catalysts(payload)

    assert payload["processes"]["reactions"][0]["inputs"], (
        "promotion emptied the reaction's input list"
    )


# --------------------------------------------------------------------------
# Defect 4 — actors that only collide after PathBank complex renaming
# --------------------------------------------------------------------------
def test_actors_colliding_after_complex_rewrite_are_merged() -> None:
    """Two distinct actor names resolving to one complex must collapse.

    In the failing run a catalyst promoted out of the reaction's inputs and the
    enzyme the paper named were different strings until Stage 6 mapped both to
    ``3-hydroxyacyl-[acyl-carrier-protein] dehydratase``. Nothing deduped after
    the rename, so the gate saw the complex twice.
    """
    mapped = {
        "entities": {
            "proteins": [],
            "protein_complexes": [{"name": "FabZ complex", "components": ["FabZ"]}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "dehydration",
                    "enzymes": [
                        {
                            "entity": "FabZ complex",
                            "entity_type": "protein_complex",
                            "role": "catalyst",
                            "provenance": "extracted",
                        },
                        {
                            "entity": "fabz  complex",
                            "entity_type": "protein_complex",
                            "role": "catalyst",
                            "evidence": "retrieved passage",
                            "provenance": "inferred",
                        },
                    ],
                }
            ]
        },
    }

    report = _rewrite_reaction_protein_enzymes_to_complexes(
        mapped,
        db=None,
        cache=_MemoryCache(),  # type: ignore[arg-type]
        global_organism="Escherichia coli",
    )

    enzymes = mapped["processes"]["reactions"][0]["enzymes"]
    assert len(enzymes) == 1, enzymes
    # The surviving row keeps the observed provenance and the merged evidence.
    assert enzymes[0]["entity"] == "FabZ complex"
    assert enzymes[0]["provenance"] == "extracted"
    assert enzymes[0]["evidence"] == "retrieved passage"
    assert report["summary"]["reaction_enzyme_duplicate_actors_merged"] == 1


# --------------------------------------------------------------------------
# Defect 5 — the bibliography must never be mined for reactions
# --------------------------------------------------------------------------
def test_reference_list_is_labelled_and_not_chunked() -> None:
    """A tail reference list becomes its own section and is never emitted."""
    body = (
        "Introduction\nLipid A biosynthesis begins with UDP-GlcNAc. " + ("Body prose. " * 60)
        + "\nResults\nWe measured LPS levels. " + ("More prose. " * 60)
        + "\nReferences\n"
        + "53 Mohan S, Kelly TM, Anderson MS. 1994. An Escherichia coli gene (FabZ) "
        "encoding (3R)-hydroxymyristoyl acyl carrier protein dehydrase. J Biol Chem "
        "269: 32896. doi: 10.1016/S0021-9258(20)30075-2 7806516 PMC8053950\n"
        "54 Anderson MS, Raetz CR. 1987. Biosynthesis of lipid A precursors. "
        "J Biol Chem 262: 5159. doi: 10.1016/S0021-9258(18)61169-X 3549716 PMC2278089\n"
    )

    labels = [label for label, _span in _split_sections(body)]
    assert "references" in labels

    paper = CandidatePaper(
        id="PMC999002",
        source="europepmc",
        title="LPS synthesis",
        abstract="An abstract.",
        organism="Escherichia coli",
        full_text=body,
        source_uri="https://example.org/PMC999002",
        year="2026",
    )
    sections = {chunk.section for chunk in chunk_paper(paper)}
    assert "references" not in sections
    assert not any("Mohan S" in chunk.text for chunk in chunk_paper(paper))


def test_bibliography_density_check_separates_citations_from_prose() -> None:
    """The content backstop fires on a reference run, not on citing prose."""
    references = (
        "50 Hirvas L, Nurminen M, Vaara M. 1997. The lipid A biosynthesis deficiency "
        "of Escherichia coli. Microbiology 143: 73. doi: 10.1099/00221287-143-1-73 "
        "9025280 PMC1234567 51 De Lay NR, Cronan JE. 2008. Genetic interaction "
        "between the AcpT transferase and YejM. Genetics 178: 1327. "
        "doi: 10.1534/genetics.107.081836 18245839 PMC2278089 52 Mostafavi M, Wang L. "
        "2018. Interplay of fabZ and lpxC mutations. mSphere 3: e00508-18. "
        "doi: 10.1128/mSphere.00508-18 PMC6211225 30381354"
    )
    prose = (
        "The first step of LPS synthesis is the reversible acylation of "
        "UDP-N-acetylglucosamine by LpxA, whereas the second step, catalyzed by the "
        "deacetylase LpxC, is the committed and rate-limiting step of the pathway. "
        "Given the essential nature of LPS for outer membrane assembly, targeting "
        "LpxC has become a focal point for antimicrobial design, as described "
        "previously (doi: 10.1128/mbio.00567-26). Treatment reduces LPS levels."
    )

    assert _is_bibliography_text(references)
    assert not _is_bibliography_text(prose)


# --------------------------------------------------------------------------
# The coverage hole: a RAG-shaped payload through the real gate
# --------------------------------------------------------------------------
def test_rag_shaped_payload_clears_required_field_gate_after_cleanup() -> None:
    """End-to-end: the failing run's shapes are rejected, then pass once hardened."""
    payload = _exportable_payload()
    payload["processes"]["reactions"].extend(
        [
            {
                # errors 1/2: half-transcribed statement
                "name": "UDP-N-acetylglucosamine -> ?",
                "inputs": ["Glucose"],
                "outputs": [],
                "biological_state": "cytosol",
                "rag_confidence": 0.65,
                "source_papers": [{"source_id": "PMC13170283"}],
            },
            {
                # errors 3-6: hollow reaction mined from a bibliography entry,
                # carrying the same complex twice in two key-shapes
                "name": "(3R)-hydroxymyristoyl acyl carrier protein dehydration",
                "inputs": [],
                "outputs": [],
                "biological_state": "cytosol",
                "rag_confidence": 0.649,
                "enzymes": [
                    {
                        "entity": "Hexokinase complex",
                        "entity_type": "protein_complex",
                        "role": "catalyst",
                        "confidence": 1.0,
                        "provenance": "extracted",
                    },
                    {
                        "entity": "Hexokinase complex",
                        "entity_type": "protein_complex",
                        "role": "catalyst",
                        "evidence": "[{'text': 'reference list entry'}]",
                    },
                ],
            },
        ]
    )

    before = validate_required_pwml_contract(payload, strict_db=True)
    assert not before["ok"]
    assert {
        "reaction_missing_left_participants",
        "reaction_missing_right_participants",
        "duplicate_reaction_enzyme_complex",
    } <= _error_codes(before)

    cleaned, removed = apply_post_merge_cleanup(payload)
    after = validate_required_pwml_contract(cleaned, strict_db=True)

    assert after["ok"], after["errors"]
    assert sorted(removed) == [
        "(3R)-hydroxymyristoyl acyl carrier protein dehydration",
        "UDP-N-acetylglucosamine -> ?",
    ]

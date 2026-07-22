"""WP4 gap-retrieval tests — memory backend, stubbed (offline) embedder.

Self-contained and offline: no chromadb, no live network, no live LLM. The
guarded ``openai`` stub mirrors tests/test_audit_json_llm_payload.py because
importing ``t2pw.rag.retrieve`` pulls in ``t2pw.rag.ingest`` ->
``t2pw.rag.acquire`` -> ``t2pw.mapping`` -> ``t2pw.llm.client`` -> ``openai`` at
import time. Kept at the top so the module passes run alone.
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

from t2pw.rag.embed import Embedder  # noqa: E402
from t2pw.rag.retrieve import (  # noqa: E402
    EvidenceBundle,
    Gap,
    detect_gaps,
    format_retrieval_context,
    query_for_gap,
    retrieve_evidence,
)
from t2pw.rag.store import Chunk, MemoryVectorStore, Retrieved  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
def _dangling_payload() -> dict:
    # One reaction: caffeine -> theobromine + formaldehyde (enzyme NdmA).
    # theobromine is consumed by no further reaction -> the reaction dangles.
    return {
        "entities": {
            "compounds": [
                {"name": "caffeine"},
                {"name": "theobromine"},
                {"name": "formaldehyde"},
            ],
            "proteins": [{"name": "NdmA"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "caffeine N1-demethylation",
                    "inputs": ["caffeine"],
                    "outputs": ["theobromine", "formaldehyde"],
                    "enzymes": [{"protein": "NdmA"}],
                }
            ]
        },
    }


def _qa_report_dangling() -> dict:
    # The CLI ``qa_graph.py`` report shape: reaction #1 is degree <= 1.
    return {
        "dangling_nodes": [{"node": "reaction:#1", "degree": 1}],
        "missing_links_suspected": [],
        "orphan_components": [],
    }


def _offline_embedder(tmp_path: Path) -> Embedder:
    return Embedder(
        model="test-embed",
        dim=64,
        cache_path=tmp_path / "embeddings_cache.json",
        online=False,
    )


def _store_with_missing_step(tmp_path: Path) -> MemoryVectorStore:
    embedder = _offline_embedder(tmp_path)
    store = MemoryVectorStore(embed_fn=embedder.embed)
    missing_step = Chunk(
        id="chunk-ndmb",
        text=(
            "The enzyme NdmB catalyzes the N3-demethylation of theobromine, "
            "yielding 7-methylxanthine and formaldehyde, the next step after "
            "caffeine demethylation in bacterial caffeine catabolism."
        ),
        source_id="PMC888002",
        source_title="NdmB continues bacterial caffeine catabolism",
        source_type="paper",
        source_uri="https://europepmc.org/article/PMC/PMC888002",
        organism="Pseudomonas putida",
        section="results",
    )
    distractors = [
        Chunk(
            id=f"distract-{i}",
            text=text,
            source_id=f"D{i}",
            source_type="paper",
            source_uri=f"https://example.org/{i}",
            section="results",
        )
        for i, text in enumerate(
            [
                "Glycolysis converts glucose to pyruvate across ten steps.",
                "The citric acid cycle oxidizes acetyl-CoA in the mitochondrion.",
                "Photosystem II splits water during the light reactions.",
            ]
        )
    ]
    store.upsert([missing_step] + distractors)
    return store


# ---------------------------------------------------------------------------
# Acceptance: dangling reaction -> detect -> retrieve the missing step w/ prov.
# ---------------------------------------------------------------------------
def test_detect_then_retrieve_missing_step_with_provenance(tmp_path) -> None:
    payload = _dangling_payload()
    reports = {"qa_graph": _qa_report_dangling()}

    gaps = detect_gaps(payload, reports)
    dangling = [g for g in gaps if g.kind == "dangling_reaction"]
    assert dangling, "expected a dangling_reaction gap"
    gap = dangling[0]
    # Enriched from the payload (read-only) with participant + enzyme symbols.
    assert gap.label == "caffeine N1-demethylation"
    assert "theobromine" in gap.symbols
    assert "NdmA" in gap.symbols

    store = _store_with_missing_step(tmp_path)
    bundle = retrieve_evidence(gap, store, top_k=1)
    assert isinstance(bundle, EvidenceBundle)
    assert bundle.hits, "expected at least one evidence hit"
    top = bundle.hits[0].chunk
    assert top.source_id == "PMC888002"
    assert top.source_uri == "https://europepmc.org/article/PMC/PMC888002"
    assert "ndmb" in top.text.lower()


def test_detect_gaps_is_read_only(tmp_path) -> None:
    payload = _dangling_payload()
    reports = {"qa_graph": _qa_report_dangling()}
    payload_before = copy.deepcopy(payload)
    reports_before = copy.deepcopy(reports)

    detect_gaps(payload, reports)

    assert payload == payload_before  # payload untouched
    assert reports == reports_before  # reports untouched


# ---------------------------------------------------------------------------
# Classification across all five gap kinds / report shapes.
# ---------------------------------------------------------------------------
def test_detect_gaps_classifies_all_kinds() -> None:
    payload = _dangling_payload()
    reports = {
        "qa_graph": {
            "dangling_nodes": [{"node": "reaction:#1", "degree": 1}],
            "missing_links_suspected": [{"node": "compound:theobromine"}],
            "flags": {
                "missing_compartments": [
                    {"entity": "caffeine", "reason": "no subcellular location recorded"}
                ],
                "empty_reactions": [{"reaction": "orphan reaction"}],
            },
        },
        "gate": {
            "errors": [
                {
                    "path": "/entities/proteins/0",
                    "reason": "Protein 'NdmA' is missing a UniProt or DrugBank identifier.",
                }
            ]
        },
        "mapping": {
            "unmapped": [
                {"name": "mystery enzyme", "status": "unmapped", "type": "protein"}
            ]
        },
    }

    kinds = {g.kind for g in detect_gaps(payload, reports)}
    assert "dangling_reaction" in kinds
    assert "orphan_metabolite" in kinds
    assert "unmapped_enzyme" in kinds
    assert "missing_precursor" in kinds
    assert "missing_compartment" in kinds


def test_detect_gaps_dedupes_same_gap() -> None:
    payload = _dangling_payload()
    reports = {
        "qa_graph": {
            "dangling_nodes": [
                {"node": "reaction:#1", "degree": 1},
                {"node": "reaction:#1", "degree": 0},
            ]
        }
    }
    dangling = [g for g in detect_gaps(payload, reports) if g.kind == "dangling_reaction"]
    assert len(dangling) == 1


# ---------------------------------------------------------------------------
# query_for_gap — natural-language + exact symbols.
# ---------------------------------------------------------------------------
def test_query_for_gap_has_nl_and_exact_symbols() -> None:
    gap = Gap(
        kind="unmapped_enzyme",
        label="NdmA",
        symbols=["NdmA"],
    )
    query = query_for_gap(gap, seed_context="bacterial caffeine catabolism")
    assert "NdmA" in query  # exact symbol preserved for the lexical half
    assert "UniProt" in query  # natural-language intent
    assert "bacterial caffeine catabolism" in query  # seed context appended


# ---------------------------------------------------------------------------
# retrieve_evidence — top_k defaults to rag_config()["retrieve_top_k"].
# ---------------------------------------------------------------------------
def test_retrieve_evidence_default_top_k_from_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_RETRIEVE_TOP_K", "2")
    store = _store_with_missing_step(tmp_path)
    gap = Gap(kind="orphan_metabolite", label="theobromine", symbols=["theobromine"])
    bundle = retrieve_evidence(gap, store)  # no explicit top_k
    # 4 chunks in the store, config caps retrieval at 2.
    assert 1 <= len(bundle.hits) <= 2


# ---------------------------------------------------------------------------
# format_retrieval_context — mirrors build_retrieval_context + provenance.
# ---------------------------------------------------------------------------
def test_format_retrieval_context_mirrors_shape_and_carries_provenance() -> None:
    gap = Gap(kind="dangling_reaction", label="caffeine N1-demethylation")
    hit = Retrieved(
        chunk=Chunk(
            id="chunk-ndmb",
            text="NdmB demethylates theobromine to 7-methylxanthine.",
            source_id="PMC888002",
            source_title="NdmB paper",
            source_type="paper",
            source_uri="https://europepmc.org/article/PMC/PMC888002",
            section="results",
        ),
        score=0.83,
    )
    bundle = EvidenceBundle(gap=gap, query="q", hits=[hit])
    text = format_retrieval_context([bundle])

    # Mirrors the build_retrieval_context block labels.
    assert "Source:" in text
    assert "Reactions:" in text
    # Gap-tagged evidence header + mandatory additive provenance.
    assert "[Evidence 1]" in text
    assert "dangling_reaction" in text
    assert "source_uri=https://europepmc.org/article/PMC/PMC888002" in text
    assert "source_id=PMC888002" in text


def test_format_retrieval_context_empty_is_noop() -> None:
    assert format_retrieval_context([]) == ""
    empty_bundle = EvidenceBundle(gap=Gap("orphan_metabolite", "x"), query="q", hits=[])
    assert format_retrieval_context([empty_bundle]) == ""


def test_dangling_reaction_gap_captures_dict_participant_symbols() -> None:
    # Regression: real payloads carry reaction participants as dicts
    # ({"name": "caffeine"}), not bare strings. The dangling-reaction gap must
    # still pick up the exact substrate/product symbols so the lexical half of
    # retrieval can hit them — previously only str participants were read, so a
    # real reaction's compounds were silently dropped from the query.
    payload = {
        "processes": {
            "reactions": [
                {
                    "name": "caffeine N1-demethylation",
                    "inputs": [{"name": "caffeine"}],
                    "outputs": [{"name": "theobromine"}, {"name": "formaldehyde"}],
                    "enzymes": [{"entity": "NdmA", "entity_type": "protein"}],
                }
            ]
        }
    }
    gaps = detect_gaps(payload, {"qa_graph": _qa_report_dangling()})
    dangling = [g for g in gaps if g.kind == "dangling_reaction"]
    assert dangling
    symbols = {s.casefold() for s in dangling[0].symbols}
    assert {"caffeine", "theobromine", "formaldehyde", "ndma"}.issubset(symbols)


# ---------------------------------------------------------------------------
# Payload-derived connectivity gaps ("dangling ends") — no report required.
# ---------------------------------------------------------------------------
def _lipid_a_payload() -> dict:
    """Two linked reactions ending in an unconsumed Kdo2-lipid A (the real case)."""
    return {
        "processes": {
            "reactions": [
                {
                    "name": "lipid IVA acylation",
                    "inputs": ["lipid IVA", "ATP"],
                    "outputs": ["lipid A", "ADP"],
                    "enzymes": [{"protein": "LpxL"}],
                },
                {
                    "name": "Kdo transfer",
                    "inputs": ["lipid A", "CMP-Kdo"],
                    "outputs": ["Kdo2-lipid A", "H2O"],
                    "enzymes": [{"protein": "WaaA"}],
                },
            ]
        }
    }


def test_terminal_product_yields_connectivity_gap_without_any_report() -> None:
    # The pathway's last product is consumed by no reaction -> a gap naming it,
    # so RAG can search for the reaction that EXTENDS the pathway. No reports
    # are supplied at all (the production case the live app hits).
    gaps = detect_gaps(_lipid_a_payload(), {})
    labels = {g.label.casefold() for g in gaps}
    assert "kdo2-lipid a" in labels

    terminal = [g for g in gaps if g.label.casefold() == "kdo2-lipid a"]
    assert terminal, "expected a gap for the terminal product"
    gap = terminal[0]
    assert gap.kind == "orphan_metabolite"
    assert gap.source == "payload"
    assert "terminal product" in gap.detail
    # The generated query asks for the reaction that would connect it.
    query = query_for_gap(gap)
    assert "Kdo2-lipid A" in query
    assert "links into the pathway" in query

    # The reaction sitting on the open end is reachable as a dangling reaction.
    dangling = [g for g in gaps if g.kind == "dangling_reaction"]
    assert any(g.label == "Kdo transfer" for g in dangling)

    # ``lipid A`` is produced AND consumed -> internal, never a gap.
    assert "lipid a" not in labels


def test_connectivity_gaps_flag_unfed_substrates() -> None:
    gaps = detect_gaps(_lipid_a_payload(), None)
    unfed = [g for g in gaps if "unfed substrate" in g.detail]
    labels = {g.label.casefold() for g in unfed}
    assert "lipid iva" in labels  # consumed by reaction 1, produced by nothing
    assert "cmp-kdo" in labels


def test_connectivity_gaps_skip_ubiquitous_cofactors() -> None:
    # A pathway whose only open ends are cofactors yields no connectivity gap.
    payload = {
        "processes": {
            "reactions": [
                {
                    "name": "cofactor-only turnover",
                    "inputs": ["ATP", "H2O"],
                    "outputs": ["ADP", "Pi"],
                }
            ]
        }
    }
    gaps = detect_gaps(payload, {})
    assert gaps == []

    # And in a real pathway the cofactor by-products are not gaps either.
    labels = {g.label.casefold() for g in detect_gaps(_lipid_a_payload(), {})}
    for cofactor in ("atp", "adp", "h2o"):
        assert cofactor not in labels


def test_connectivity_gaps_handle_dict_participants() -> None:
    payload = {
        "processes": {
            "reactions": [
                {
                    "name": "Kdo transfer",
                    "inputs": [{"name": "lipid A", "stoichiometry": 1}],
                    "outputs": [{"name": "Kdo2-lipid A", "stoichiometry": 1}],
                }
            ]
        }
    }
    labels = {g.label for g in detect_gaps(payload, {})}
    assert "Kdo2-lipid A" in labels
    assert "lipid A" in labels


def test_connectivity_gaps_do_not_duplicate_report_gaps() -> None:
    # ``theobromine`` is both a report-flagged orphan and a payload terminal
    # product; it must be emitted once.
    reports = {
        "qa_graph": {
            "dangling_nodes": [{"node": "reaction:#1", "degree": 1}],
            "missing_links_suspected": [{"node": "compound:theobromine"}],
        }
    }
    gaps = detect_gaps(_dangling_payload(), reports)
    assert len([g for g in gaps if g.label.casefold() == "theobromine"]) == 1
    assert len([g for g in gaps if g.kind == "dangling_reaction"]) == 1


def test_connectivity_detection_is_read_only() -> None:
    payload = _lipid_a_payload()
    before = copy.deepcopy(payload)
    detect_gaps(payload, {})
    assert payload == before

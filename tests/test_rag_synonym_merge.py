"""Synonym-canonical merge tests — offline, deterministic, self-contained.

Proves the follow-up to the direction-aware ``conflict_key`` fix: an OPTIONAL,
injected ``synonym_resolver`` collapses cross-paper reaction duplicates that
differ only by a compound/enzyme SYNONYM, while the default (``resolver=None``)
path stays byte-identical to today.

No chromadb, no network, no live LLM. ``t2pw.rag.synthesize`` and
``t2pw.rag.synonyms`` import none of that stack; the guarded ``openai`` stub is
kept anyway (mirrors the sibling rag tests) so the module also passes run alone.
"""

from __future__ import annotations

import re
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

from t2pw.rag.synonyms import build_offline_synonym_resolver  # noqa: E402
from t2pw.rag.synthesize import (  # noqa: E402
    _build_entities,
    _Participant,
    _Reaction,
    _resolve_reactions,
)

CACHE = ROOT / "data" / "id_mapping_cache.json"


def _rxn(name, inputs, outputs, *, source_id, chunk_id, enzymes=None):
    return _Reaction(
        name=name,
        inputs=[_Participant(n) for n in inputs],
        outputs=[_Participant(n) for n in outputs],
        enzymes=list(enzymes or []),
        provenance=[{"source_id": source_id, "chunk_id": chunk_id}],
        scores=[],
    )


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", value.strip().casefold()))


def _dict_resolver(mapping):
    """In-memory grouping resolver: normalized-name -> token, else the norm itself."""

    def resolve(name: str) -> str:
        norm = _norm(name)
        return mapping.get(norm, norm)

    return resolve


# ---------------------------------------------------------------------------
# 1. Injected resolver merges synonym-only duplicates; opt-in (None keeps two).
# ---------------------------------------------------------------------------
def test_injected_resolver_merges_synonym_duplicate_reactions():
    """Two reactions identical except one names UDP-GlcNAc, the other the full
    synonym, MERGE to one row (provenance unioned) under an injected resolver."""
    resolver = _dict_resolver(
        {
            "udpglcnac": "syn::compound::udp_glcnac",
            "udpnacetylglucosamine": "syn::compound::udp_glcnac",
        }
    )
    r1 = _rxn(
        "acetyltransfer (paper A)",
        inputs=["UDP-GlcNAc", "acetyl-CoA"],
        outputs=["product"],
        source_id="PMID:1",
        chunk_id="cA",
    )
    r2 = _rxn(
        "acetyltransfer (paper B)",
        inputs=["UDP-N-acetylglucosamine", "acetyl-CoA"],
        outputs=["product"],
        source_id="PMID:2",
        chunk_id="cB",
    )

    resolved, conflicts = _resolve_reactions([r1, r2], resolver)

    assert len(resolved) == 1, "synonym-only duplicates must collapse to one reaction"
    assert conflicts == [], "identical signatures are a merge, not a conflict"
    # Provenance from BOTH source papers is unioned onto the survivor.
    assert {p["source_id"] for p in resolved[0].provenance} == {"PMID:1", "PMID:2"}
    # GROUPING-ONLY: the survivor keeps a REAL display name (never a rewritten
    # canonical token), and its participants keep their original spellings.
    assert resolved[0].name in {"acetyltransfer (paper A)", "acetyltransfer (paper B)"}
    assert any(
        p.name in {"UDP-GlcNAc", "UDP-N-acetylglucosamine"} for p in resolved[0].inputs
    )


def test_synonym_merge_is_opt_in_resolver_none_keeps_both():
    """The SAME pair stays as two reactions when no resolver is injected — proving
    the feature is opt-in and the default path is unchanged."""
    r1 = _rxn(
        "acetyltransfer (paper A)",
        inputs=["UDP-GlcNAc", "acetyl-CoA"],
        outputs=["product"],
        source_id="PMID:1",
        chunk_id="cA",
    )
    r2 = _rxn(
        "acetyltransfer (paper B)",
        inputs=["UDP-N-acetylglucosamine", "acetyl-CoA"],
        outputs=["product"],
        source_id="PMID:2",
        chunk_id="cB",
    )

    resolved, conflicts = _resolve_reactions([r1, r2], None)

    assert len(resolved) == 2, "without a resolver the synonyms do NOT collapse"
    assert conflicts == []


# ---------------------------------------------------------------------------
# 2. The REAL offline resolver over data/id_mapping_cache.json.
# ---------------------------------------------------------------------------
def test_offline_resolver_collapses_real_udp_synonyms():
    resolve = build_offline_synonym_resolver(str(CACHE))
    a = resolve("UDP-GlcNAc")
    b = resolve("UDP-N-acetylglucosamine")
    c = resolve("uridine diphosphate-N-acetyl-glucosamine (UDP-GlcNAc)")

    assert a == b, "verified synonyms (same ChEBI id) must share a grouping token"
    assert a == c, "the 'long name (ABBREV)' form must collapse to the same token"
    assert a.startswith("syn::"), "a resolved token is namespaced, not a bare name"


def test_offline_resolver_leaves_unmapped_placeholder_distinct():
    resolve = build_offline_synonym_resolver(str(CACHE))
    udp = resolve("UDP-GlcNAc")
    placeholder = resolve("LpxA product")

    assert placeholder != udp, "an unmapped placeholder must NOT merge into the UDP node"
    # An unmapped name groups by its own normalized form (today's effective behavior).
    assert placeholder == _norm("LpxA product")


def test_offline_resolver_degrades_gracefully_on_missing_file():
    """A missing/broken cache yields a normalize-only resolver — never raises."""
    resolve = build_offline_synonym_resolver("does/not/exist.json")
    assert resolve("UDP-GlcNAc") == _norm("UDP-GlcNAc")
    assert resolve("UDP-GlcNAc") != resolve("UDP-N-acetylglucosamine")


# ---------------------------------------------------------------------------
# 3. Determinism guard — resolver=None keys equal today's literal values.
# ---------------------------------------------------------------------------
def test_resolver_none_keys_are_bytewise_today():
    rxn = _rxn(
        "sample",
        inputs=["UDP-GlcNAc", "ATP"],
        outputs=["ADP"],
        source_id="PMID:9",
        chunk_id="c9",
    )

    # conflict_key(None): (sorted casefold inputs, sorted casefold outputs).
    assert rxn.conflict_key(None) == (("atp", "udp-glcnac"), ("adp",))
    assert rxn.conflict_key() == rxn.conflict_key(None)  # default arg == explicit None

    # signature(None): sorted (casefold name, stoichiometry) pairs + flags.
    assert rxn.signature(None) == (
        (("atp", None), ("udp-glcnac", None)),
        (("adp", None),),
        False,
        "",
    )
    assert rxn.signature() == rxn.signature(None)


# ---------------------------------------------------------------------------
# 4. Entity unification — synonym compounds list ONCE under a resolver.
# ---------------------------------------------------------------------------
def test_entities_unify_synonym_compounds_under_resolver():
    resolver = _dict_resolver(
        {
            "udpglcnac": "syn::compound::udp_glcnac",
            "udpnacetylglucosamine": "syn::compound::udp_glcnac",
        }
    )
    r1 = _rxn(
        "rxn A",
        inputs=["UDP-GlcNAc"],
        outputs=["out1"],
        source_id="PMID:1",
        chunk_id="cA",
    )
    r2 = _rxn(
        "rxn B",
        inputs=["UDP-N-acetylglucosamine"],
        outputs=["out2"],
        source_id="PMID:2",
        chunk_id="cB",
    )

    ent_syn, _ = _build_entities([r1, r2], resolver)
    ent_none, _ = _build_entities([r1, r2], None)

    syn_names = {c["name"] for c in ent_syn.get("compounds", [])}
    none_names = {c["name"] for c in ent_none.get("compounds", [])}

    # Under the resolver the two UDP synonyms collapse to ONE compound node (real
    # display name kept); without it both spellings list separately.
    udp_syn = syn_names & {"UDP-GlcNAc", "UDP-N-acetylglucosamine"}
    udp_none = none_names & {"UDP-GlcNAc", "UDP-N-acetylglucosamine"}
    assert len(udp_syn) == 1, "synonyms unify to a single entity node"
    assert len(udp_none) == 2, "without a resolver both spellings appear"

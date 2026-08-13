"""Adversarial regressions for the gap-admission gate.

Each block here corresponds to a way the first version of the gate could be
walked past, all of them found by replaying real artifacts rather than by
imagining attacks:

1. **Actor-based connectivity.** Two unrelated reactions that share an enzyme
   looked connected. On the real ``runs/2026-07-28_0919`` PMC12444477 payload —
   where a separate defect had sprayed every enzyme onto every reaction —
   thirteen *phospholipid* reactions read as filling the lipid A pathway's WaaA
   and LpxM gaps on the strength of a shared catalyst name.
2. **Bag-of-words evidence.** "the chunk contains A, B and Enz1 somewhere" was
   treated as evidence that Enz1 converts A to B. A chunk that says
   "Enz1 catalyzes A to B. Enz2 catalyzes X to Y." then supports ``A -> Y``.
3. **Substring name matching.** ``LpxA`` matched ``LpxAB``, ``NdmA`` matched
   ``NdmAB``, ``AMP`` matched ``cAMP``.
4. **Type-blind gap filling.** A reaction "filled" a missing-compartment gap by
   mentioning the compound, and an unmapped-enzyme gap by mentioning the protein.
5. **Unbounded chaining.** A chain could run arbitrarily far from the gap and
   could hop through CoA into unrelated chemistry.

Offline / deterministic: no chromadb, no network, no live LLM.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

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

from t2pw.rag.admission import (  # noqa: E402
    AdmissionPolicy,
    RagReactionCandidate,
    admit_candidates,
    locate_span,
    name_pattern,
    split_statements,
    states_name,
)
from t2pw.rag.retrieve import (  # noqa: E402
    EvidenceBundle,
    Gap,
    detect_gaps,
    query_for_gap,
)
from t2pw.rag.store import Chunk, Retrieved  # noqa: E402
from t2pw.rag.synthesize import synthesize_with_report  # noqa: E402


REQUESTED_PATHWAY = "caffeine degradation"
REQUESTED_ORGANISM = "Pseudomonas putida"


# ---------------------------------------------------------------------------
# Shared fixtures: a seed whose only open end is the metabolite B.
# ---------------------------------------------------------------------------
def _seed_with_gap_at_b() -> dict:
    """``B -> P``. B is consumed but produced by nothing: the gap is at B."""
    return {
        "entities": {
            "species": [{"name": "Pseudomonas putida"}],
            "compounds": [{"name": "B"}, {"name": "P"}],
            "proteins": [{"name": "SeedEnz"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "R0 seed step",
                    "inputs": ["B"],
                    "outputs": ["P"],
                    "enzymes": [{"protein": "SeedEnz"}],
                    "scope_membership": "core",
                }
            ]
        },
    }


def _seed_context() -> dict:
    return {
        "text": "caffeine degradation in Pseudomonas putida",
        "source": {
            "source_id": "PMID:SEED",
            "source_title": "seed paper",
            "source_type": "paper",
            "organism": "Pseudomonas putida",
        },
    }


def _gap_at_b(seed: dict = None) -> Gap:
    gaps = detect_gaps(
        seed if seed is not None else _seed_with_gap_at_b(),
        {},
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    return next(g for g in gaps if g.label == "B" and g.kind == "orphan_metabolite")


def _chunk(chunk_id: str, text: str, **kwargs) -> Chunk:
    fields = {
        "source_id": "PMID:B",
        "source_title": "downstream paper",
        "source_type": "paper",
        "source_uri": "https://example.org/PMID:B",
        "organism": "Pseudomonas putida",
        "section": "results",
    }
    fields.update(kwargs)
    return Chunk(id=chunk_id, text=text, **fields)


def _bundle(gap: Gap, chunk: Chunk, score: float = 0.9) -> EvidenceBundle:
    return EvidenceBundle(
        gap=gap, query=query_for_gap(gap), hits=[Retrieved(chunk=chunk, score=score)]
    )


def _synthesize(bundles, seed=None, **kwargs):
    params = {
        "requested_pathway": REQUESTED_PATHWAY,
        "requested_organism": REQUESTED_ORGANISM,
    }
    params.update(kwargs)
    return synthesize_with_report(
        seed if seed is not None else _seed_with_gap_at_b(),
        list(bundles),
        _seed_context(),
        **params,
    )


def _names(result) -> list:
    return [r.get("name") for r in result.payload["processes"]["reactions"]]


def _by_name(rows) -> dict:
    return {row["name"]: row for row in rows}


# ===========================================================================
# 1. Connectivity is CHEMICAL — catalysts are not graph nodes.
# ===========================================================================
def test_a_shared_catalyst_does_not_connect_two_unrelated_reactions() -> None:
    """The exact regression: ``A -> B / SharedE`` passes, ``X -> Y / SharedE`` does not.

    Both reactions are stated by the same paper, both are catalysed by the same
    enzyme, and both are equally well evidenced. Only the first touches the gap's
    metabolite. If a catalyst could carry connectivity, the second would ride in
    on the first's coat-tails — which is what happened to thirteen phospholipid
    reactions in the ``2026-07-28_0919`` replay.
    """
    gap = _gap_at_b()
    chunk = _chunk(
        "shared-catalyst",
        "\n".join(
            [
                "name: fills the gap | A -> B | enzyme: SharedE",
                "name: unrelated step | X -> Y | enzyme: SharedE",
            ]
        ),
    )
    result = _synthesize([_bundle(gap, chunk)])

    assert "fills the gap" in _names(result)
    assert "unrelated step" not in _names(result)

    rejected = _by_name(result.admission["rejected"])
    assert "unrelated step" in rejected
    assert rejected["unrelated step"]["reasons"][0].startswith(
        ("does_not_fill_named_gap", "adjacent_pathway_context_only")
    )

    # ...and the enzyme's own row did not smuggle X or Y into the payload either.
    entity_names = {
        row["name"]
        for bucket in result.payload["entities"].values()
        for row in bucket
    }
    assert not entity_names & {"X", "Y"}


def test_a_reaction_name_or_pathway_word_never_expands_the_frontier() -> None:
    """Only substrates and products are nodes — not names, not vocabulary.

    The second reaction here is *named* after the gap's metabolite and its
    evidence is full of the requested pathway's vocabulary. Neither is chemistry.
    """
    gap = _gap_at_b()
    chunk = _chunk(
        "name-bait",
        "\n".join(
            [
                "name: fills the gap | A -> B | enzyme: SharedE",
                "name: B handling during caffeine degradation | X -> Y | enzyme: OtherE",
            ]
        ),
    )
    result = _synthesize([_bundle(gap, chunk)])

    assert "fills the gap" in _names(result)
    assert "B handling during caffeine degradation" not in _names(result)


# ===========================================================================
# 2. Evidence is LOCAL and RELATIONAL, not bag-of-words.
# ===========================================================================
_TWO_SENTENCE_TEXT = "Enz1 catalyzes A to B. Enz2 catalyzes X to Y."


def _prose_extractor(claim: dict):
    """A stub prose extractor returning exactly one claim, as the real one would."""

    def _run(_text: str):
        return [claim]

    return _run


def _prose_result(claim: dict, *, text: str = _TWO_SENTENCE_TEXT):
    gap = _gap_at_b()
    chunk = _chunk("two-sentences", text)
    return _synthesize([_bundle(gap, chunk)], prose_extractor=_prose_extractor(claim))


def test_the_honest_single_sentence_claim_is_accepted() -> None:
    """``A -> B / Enz1`` is stated by one sentence, so it survives."""
    result = _prose_result(
        {
            "name": "A to B",
            "inputs": [{"name": "A", "stoichiometry": None}],
            "outputs": [{"name": "B", "stoichiometry": None}],
            "enzymes": ["Enz1"],
            "reversible": False,
            "quote": "Enz1 catalyzes A to B.",
        }
    )
    assert "A to B" in _names(result)
    row = _by_name(result.admission["accepted"])["A to B"]
    assert row["evidence"]["span"] == "Enz1 catalyzes A to B."
    # The parent chunk pointer is retained beside the span, not replaced by it.
    assert row["evidence"]["chunk_id"] == "two-sentences"


def test_a_claim_stitched_across_two_sentences_is_refused() -> None:
    """``A -> Y / Enz1`` takes its substrate from one sentence and its product
    from the next. No single statement says it, so there is no span to cite."""
    result = _prose_result(
        {
            "name": "A to Y",
            "inputs": [{"name": "A", "stoichiometry": None}],
            "outputs": [{"name": "Y", "stoichiometry": None}],
            "enzymes": ["Enz1"],
            "reversible": False,
            "quote": _TWO_SENTENCE_TEXT,  # the model offers the WHOLE passage
        }
    )
    assert "A to Y" not in _names(result)
    rejected = _by_name(result.admission["rejected"])["A to Y"]
    assert rejected["reasons"][0].startswith(
        ("no_local_evidence_span", "evidence_span_is_not_one_statement")
    )


def test_a_catalyst_borrowed_from_the_other_sentence_is_refused() -> None:
    """``A -> B / Enz2``: the reaction is real, the attribution is not."""
    result = _prose_result(
        {
            "name": "A to B misattributed",
            "inputs": [{"name": "A", "stoichiometry": None}],
            "outputs": [{"name": "B", "stoichiometry": None}],
            "enzymes": ["Enz2"],
            "reversible": False,
            "quote": "",
        }
    )
    assert "A to B misattributed" not in _names(result)
    rejected = _by_name(result.admission["rejected"])["A to B misattributed"]
    assert rejected["reasons"][0].startswith("no_local_evidence_span")


def test_co_occurrence_without_a_relation_is_not_a_reaction() -> None:
    """Two names in one sentence and no verb between them is not evidence."""
    gap = _gap_at_b()
    chunk = _chunk("listing", "Levels of A and B were measured in each sample.")
    result = _synthesize(
        [_bundle(gap, chunk)],
        prose_extractor=_prose_extractor(
            {
                "name": "invented from a listing",
                "inputs": [{"name": "A", "stoichiometry": None}],
                "outputs": [{"name": "B", "stoichiometry": None}],
                "enzymes": [],
                "reversible": False,
                "quote": "Levels of A and B were measured in each sample.",
            }
        ),
    )
    assert "invented from a listing" not in _names(result)
    rejected = _by_name(result.admission["rejected"])["invented from a listing"]
    assert any(r.startswith("evidence_states_no_reaction_relation") for r in rejected["reasons"])


def test_an_arrow_candidate_cites_its_own_parsed_line_not_the_whole_chunk() -> None:
    """Determinism: the span of an arrow parse is exactly the line it parsed."""
    gap = _gap_at_b()
    chunk = _chunk(
        "multi-line",
        "\n".join(
            [
                "Some prose about the pathway that mentions X and Y in passing.",
                "name: fills the gap | A -> B | enzyme: SharedE",
                "More prose naming Z.",
            ]
        ),
    )
    result = _synthesize([_bundle(gap, chunk)])
    row = _by_name(result.admission["accepted"])["fills the gap"]
    assert row["evidence"]["span"] == "name: fills the gap | A -> B | enzyme: SharedE"
    assert row["evidence"]["chunk_id"] == "multi-line"


def test_split_statements_keeps_species_names_and_abbreviations_whole() -> None:
    """A sentence splitter that breaks ``E. coli`` makes every organism invisible."""
    assert split_statements("E. coli grows fast. S. aureus does not.") == [
        "E. coli grows fast.",
        "S. aureus does not.",
    ]
    assert split_statements("See Fig. 3 for the scheme.") == [
        "See Fig. 3 for the scheme."
    ]
    assert len(split_statements("A -> B\nX -> Y")) == 2


def test_locate_span_refuses_a_quote_that_is_not_in_the_chunk() -> None:
    """A paraphrased or invented quote is not evidence, however plausible."""
    chunk_text = "Enz1 catalyzes A to B."
    assert locate_span(chunk_text, "Enz1 converts A into B.", ["A", "B", "Enz1"]) == (
        "Enz1 catalyzes A to B."  # falls back to the real sentence
    )
    assert locate_span(chunk_text, "Enz9 catalyzes Q to R.", ["Q", "R", "Enz9"]) == ""


# ===========================================================================
# 3. Name matching is BOUNDARY AWARE.
# ===========================================================================
def test_name_matching_does_not_match_a_longer_identifier() -> None:
    assert not states_name("LpxAB forms the acyltransferase complex.", "LpxA")
    assert not states_name("NdmAB was purified together.", "NdmA")
    assert not states_name("cAMP levels rose sharply.", "AMP")
    assert not states_name("AMPK was activated.", "AMP")


def test_name_matching_still_matches_the_whole_name() -> None:
    assert states_name("LpxA catalyses the first step.", "LpxA")
    assert states_name("The product is NdmA, not NdmB.", "NdmA")
    assert states_name("ATP is hydrolysed to AMP and PPi.", "AMP")
    # Trailing punctuation and possessives do not hide a name.
    assert states_name("LpxA, LpxC and LpxD act in order.", "LpxC")


def test_punctuation_equivalent_biochemical_names_still_match() -> None:
    """``lipid IV A`` / ``lipid IV_A`` / ``lipid-IV-A`` are one name."""
    for written in ("lipid IV_A accumulates.", "lipid IV A accumulates.", "lipid-IV-A accumulates."):
        assert states_name(written, "lipid IV A"), written
        assert states_name(written, "lipid IV_A"), written
    assert states_name("UDP-GlcNAc is the substrate.", "UDP GlcNAc")
    assert states_name("7-methylxanthine is released.", "7 methylxanthine")


def test_name_pattern_is_none_for_a_contentless_name() -> None:
    assert name_pattern("") is None
    assert name_pattern("   ---   ") is None


def test_a_catalyst_whose_span_only_names_a_longer_protein_is_refused() -> None:
    """Integration: the boundary rule reaches the real admission decision."""
    gap = _gap_at_b()
    chunk = _chunk("complex", "LpxAB converts A to B in a single step.")
    result = _synthesize(
        [_bundle(gap, chunk)],
        prose_extractor=_prose_extractor(
            {
                "name": "A to B by LpxA",
                "inputs": [{"name": "A", "stoichiometry": None}],
                "outputs": [{"name": "B", "stoichiometry": None}],
                "enzymes": ["LpxA"],
                "reversible": False,
                "quote": "LpxAB converts A to B in a single step.",
            }
        ),
    )
    assert "A to B by LpxA" not in _names(result)


# ===========================================================================
# 4. A gap has a TYPE, and a reaction cannot fill every kind of gap.
# ===========================================================================
def _typed_gap(kind: str, label: str, **kwargs) -> Gap:
    return Gap(
        kind=kind,
        label=label,
        symbols=[label],
        source="gate",
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
        **kwargs,
    )


def _typed_candidate(gap: Gap, span: str, **kwargs) -> RagReactionCandidate:
    fields = dict(
        gap_id=gap.gap_id,
        name="A to B",
        inputs=["A"],
        outputs=["B"],
        enzymes=[],
        evidence={"text": span, "chunk_id": "c1", "source_id": "PMID:B"},
        evidence_span=span,
        source_paper={"source_id": "PMID:B"},
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    fields.update(kwargs)
    return RagReactionCandidate(**fields)


def _typed_gap_result(gap, span, **candidate_kwargs):
    """Run the gate for a typed gap and return ``(accepted, report)``."""
    candidate = _typed_candidate(gap, span, **candidate_kwargs)
    return admit_candidates(
        [candidate], gaps=[gap], seed_payload=_seed_with_gap_at_b()
    )


def test_a_reaction_never_fills_a_typed_gap_however_good_its_evidence() -> None:
    """A reaction row expresses no identity and no location. It cannot be the fill.

    This is the separation the gate now makes: a reaction candidate offered to a
    typed gap is always refused, and whatever typed evidence its span carries is
    lifted out as a SEPARATE proposal. Before, an identifier appearing anywhere in
    the span admitted the reaction and the gap was reported closed while the
    protein still had no ``mapped_ids`` at all.
    """
    for kind, span in (
        ("missing_compartment", "A is converted to B in the periplasm."),
        ("unmapped_enzyme", "EnzX (UniProt P12345) converts A to B."),
    ):
        gap = _typed_gap(kind, "B" if kind == "missing_compartment" else "EnzX")
        accepted, report = _typed_gap_result(
            gap, span, enzymes=["EnzX"] if kind == "unmapped_enzyme" else []
        )
        assert accepted == [], kind
        assert report.rejected[0]["reasons"][0].startswith(
            "candidate_type_cannot_fill_gap"
        ), kind


def test_typed_evidence_becomes_a_structured_proposal_not_an_admission() -> None:
    """The identifier is parsed into a structured candidate, and stops there."""
    gap = _typed_gap("unmapped_enzyme", "EnzX")
    _accepted, report = _typed_gap_result(
        gap, "EnzX (UniProt P12345) converts A to B.", enzymes=["EnzX"]
    )

    assert len(report.resolutions) == 1
    proposal = report.resolutions[0]
    assert proposal["kind"] == "identifier"
    assert proposal["value"] == {"uniprot": "P12345"}
    assert proposal["target"] == "EnzX"
    assert proposal["status"] == "proposed"
    assert proposal["applied"] is False

    # The recommendation is recorded as a recommendation, not as an action.
    assert report.recommended_route == [
        {"gap_id": gap.gap_id, "label": "EnzX", "recommended_route": "identity_resolver"}
    ]


def test_a_malformed_accession_is_not_an_identifier_at_all() -> None:
    """"UniProt" next to a token is not an accession; the grammar decides."""
    gap = _typed_gap("unmapped_enzyme", "EnzX")
    _accepted, report = _typed_gap_result(
        gap, "EnzX (UniProt not-an-accession) converts A to B.", enzymes=["EnzX"]
    )
    assert report.resolutions == []


def test_a_compartment_the_schema_cannot_express_is_not_a_resolution() -> None:
    gap = _typed_gap("missing_compartment", "B")
    _accepted, report = _typed_gap_result(
        gap, "A is converted to B in the nucleoid-associated region."
    )
    assert report.resolutions == []


def test_a_missing_precursor_proposal_needs_a_genuinely_incomplete_reaction() -> None:
    """A complete reaction is not missing a precursor, whatever a report said.

    ``R0 seed step`` in the shared fixture has both sides populated, so nothing
    may be appended to it — that would be inventing chemistry, not completing it.
    The same evidence against a reaction that really has an empty side does
    produce a proposal, which is pinned end-to-end in
    tests/test_rag_typed_gap_resolution.py.
    """
    gap = _typed_gap("missing_precursor", "R0 seed step", target_symbols=["B", "P"])
    _accepted, report = _typed_gap_result(
        gap,
        "Q is converted to P by SeedEnz.",
        name="supplies",
        inputs=["Q"],
        outputs=["P"],
    )
    assert report.resolutions == []

    incomplete = {
        "entities": {
            "species": [{"name": "Pseudomonas putida"}],
            "compounds": [{"name": "P"}],
        },
        "processes": {
            "reactions": [{"name": "R0 seed step", "inputs": [], "outputs": ["P"]}]
        },
    }
    gap2 = _typed_gap(
        "missing_precursor",
        "R0 seed step",
        target_symbols=["P"],
        missing_side="inputs",
    )
    candidate = _typed_candidate(
        gap2,
        "Q is converted to P by SeedEnz.",
        name="supplies",
        inputs=["Q"],
        outputs=["P"],
    )
    _accepted2, report2 = admit_candidates(
        [candidate], gaps=[gap2], seed_payload=incomplete
    )
    assert len(report2.resolutions) == 1
    assert report2.resolutions[0]["kind"] == "precursor"
    assert report2.resolutions[0]["value"] == {
        "participants": ["Q"],
        "participant": "Q",
        "side": "inputs",
        "reaction": "R0 seed step",
    }


def test_a_connectivity_gap_is_still_fillable_by_a_plain_reaction() -> None:
    """The type rule must not make the normal case unfillable."""
    gap = _gap_at_b()
    chunk = _chunk("plain", "name: fills the gap | A -> B | enzyme: SharedE")
    result = _synthesize([_bundle(gap, chunk)])
    assert "fills the gap" in _names(result)


# ===========================================================================
# 5. Chained admission is BOUNDED and AUDITED.
# ===========================================================================
_CHAIN_LINES = [
    "name: hop0 | A -> B | enzyme: E1",
    "name: hop1 | A -> C | enzyme: E2",
    "name: hop2 | C -> D | enzyme: E3",
    "name: hop3 | D -> E | enzyme: E4",
]


def _chain_result(lines=None, **kwargs):
    gap = _gap_at_b()
    chunk = _chunk("chain", "\n".join(lines if lines is not None else _CHAIN_LINES))
    return _synthesize([_bundle(gap, chunk)], **kwargs)


def test_a_legitimate_chain_within_the_bound_is_admitted_and_audited() -> None:
    """``A -> B`` fills the gap; ``A -> C``, ``C -> D`` extend it by one hop each."""
    result = _chain_result()
    accepted = _by_name(result.admission["accepted"])

    assert set(accepted) == {"hop0", "hop1", "hop2"}
    assert "hop3" not in _names(result)

    gap_id = _gap_at_b().gap_id
    assert accepted["hop0"]["chain"] == {
        "hop": 0,
        "gap_id": gap_id,
        "shared_metabolite": "b",
    }
    # Every chained acceptance records hop distance, parent claim, the shared
    # metabolite and the gap.
    assert accepted["hop1"]["chain"]["hop"] == 1
    assert accepted["hop1"]["chain"]["shared_metabolite"] == "a"
    assert accepted["hop1"]["chain"]["parent_name"] == "hop0"
    assert accepted["hop1"]["chain"]["parent_claim"]
    assert accepted["hop1"]["chain"]["gap_id"] == gap_id
    assert accepted["hop2"]["chain"]["hop"] == 2
    assert accepted["hop2"]["chain"]["parent_name"] == "hop1"

    # Direct and chained precision are reported apart.
    assert result.admission["acceptance_breakdown"] == {
        "direct": 1,
        "chained": 2,
        "chained_by_hop": {"1": 1, "2": 1},
    }


def test_a_chain_beyond_the_configured_bound_is_refused_with_its_own_reason() -> None:
    result = _chain_result()
    rejected = _by_name(result.admission["rejected"])
    assert "hop3" in rejected
    assert rejected["hop3"]["reasons"][0].startswith("chain_distance_exceeds_limit")

    # Widening the bound admits exactly the next link, and nothing else changes.
    wider = _chain_result(admission_policy=AdmissionPolicy(max_chain_hops=3))
    assert set(_by_name(wider.admission["accepted"])) == {"hop0", "hop1", "hop2", "hop3"}
    assert wider.admission["accepted"][-1]["chain"]["hop"] == 3

    # Narrowing it to 0 admits only the direct fill.
    narrow = _chain_result(admission_policy=AdmissionPolicy(max_chain_hops=0))
    assert set(_by_name(narrow.admission["accepted"])) == {"hop0"}


def test_a_chain_cannot_escape_through_a_currency_metabolite() -> None:
    """CoA is released by half of metabolism; it must not be a bridge.

    Without the hub exclusion, the acetyl-CoA reaction below is one hop from the
    gap and drags in whatever else touches CoA — the frontier stops being a
    pathway and becomes the whole corpus.
    """
    result = _chain_result(
        [
            "name: hop0 | A -> B + CoA | enzyme: E1",
            "name: via CoA | CoA -> Zed | enzyme: E9",
        ]
    )
    assert "hop0" in _names(result)
    assert "via CoA" not in _names(result)
    rejected = _by_name(result.admission["rejected"])
    assert rejected["via CoA"]["reasons"][0].startswith(
        ("does_not_fill_named_gap", "adjacent_pathway_context_only")
    )


def test_chain_hop_distances_do_not_depend_on_candidate_order() -> None:
    """BFS levels, not iteration order: shuffling the evidence changes nothing."""
    forward = _chain_result(_CHAIN_LINES)
    reversed_ = _chain_result(list(reversed(_CHAIN_LINES)))

    def _hops(result):
        return {
            row["name"]: row["chain"]["hop"]
            for row in result.admission["accepted"]
        }

    assert _hops(forward) == _hops(reversed_) == {"hop0": 0, "hop1": 1, "hop2": 2}
    assert {r["name"] for r in reversed_.admission["rejected"]} == {"hop3"}


# ===========================================================================
# 6. Pathway / organism comparisons use OBSERVED metadata.
# ===========================================================================
def test_paper_observed_metadata_reaches_the_admission_decision() -> None:
    """A paper the eligibility screen read as human is refused for a P. putida run.

    The span itself names no organism, so the paper-level observation carried down
    the chunk is what decides — which is the whole reason it is carried.
    """
    gap = _gap_at_b()
    chunk = _chunk(
        "human-paper",
        "name: fills the gap | A -> B | enzyme: SharedE",
        organism="Homo sapiens",
        observed_organisms=["Homo sapiens"],
    )
    result = _synthesize([_bundle(gap, chunk)])

    assert "fills the gap" not in _names(result)
    row = result.admission["rejected"][0]
    assert row["observed_organisms"] == ["Homo sapiens"]
    assert row["organism_match"] == "mismatch"
    assert row["reasons"][0].startswith("organism_mismatch")


def test_an_organism_named_in_the_span_overrides_the_paper_level_reading() -> None:
    """A review is "about" two species; the sentence backing a reaction is not."""
    gap = _gap_at_b()
    chunk = _chunk(
        "review",
        "In Homo sapiens cells, SharedE converts A to B.",
        organism="Pseudomonas putida",
        observed_organisms=["Pseudomonas putida", "Homo sapiens"],
    )
    result = _synthesize(
        [_bundle(gap, chunk)],
        prose_extractor=_prose_extractor(
            {
                "name": "human step",
                "inputs": [{"name": "A", "stoichiometry": None}],
                "outputs": [{"name": "B", "stoichiometry": None}],
                "enzymes": ["SharedE"],
                "reversible": False,
                "quote": "In Homo sapiens cells, SharedE converts A to B.",
            }
        ),
    )
    assert "human step" not in _names(result)
    row = _by_name(result.admission["rejected"])["human step"]
    assert row["observed_organisms"] == ["Homo sapiens"]
    assert row["organism_match"] == "mismatch"


def test_a_same_genus_different_species_observation_is_not_an_exact_match() -> None:
    """``genus_level`` is related-taxon support, and is never reported as ``match``."""
    gap = _gap_at_b()
    chunk = _chunk(
        "sibling",
        "name: fills the gap | A -> B | enzyme: SharedE",
        organism="Pseudomonas aeruginosa",
        observed_organisms=["Pseudomonas aeruginosa"],
    )
    result = _synthesize([_bundle(gap, chunk)])

    row = _by_name(result.admission["accepted"])["fills the gap"]
    assert row["organism_match"] == "genus_level"

    # Under the strict policy that related taxon is refused outright.
    strict = _synthesize(
        [_bundle(gap, chunk)],
        admission_policy=AdmissionPolicy(organism_policy="strict"),
    )
    assert "fills the gap" not in _names(strict)
    assert strict.admission["rejected"][0]["reasons"][0].startswith("organism_mismatch")


def test_a_bare_genus_observation_is_not_an_exact_match_either() -> None:
    gap = _gap_at_b()
    chunk = _chunk(
        "genus-only",
        "name: fills the gap | A -> B | enzyme: SharedE",
        organism="Pseudomonas",
        observed_organisms=["Pseudomonas"],
    )
    result = _synthesize([_bundle(gap, chunk)])
    assert (
        _by_name(result.admission["accepted"])["fills the gap"]["organism_match"]
        == "genus_level"
    )


def test_a_requested_organism_is_never_promoted_into_the_observed_fields() -> None:
    """The stamping bug this whole requested/observed split exists to prevent."""
    gap = _gap_at_b()
    chunk = _chunk(
        "silent",
        "name: fills the gap | A -> B | enzyme: SharedE",
        organism="",
        observed_organisms=[],
    )
    result = _synthesize([_bundle(gap, chunk)])

    row = _by_name(result.admission["accepted"])["fills the gap"]
    assert row["observed_organisms"] == []
    assert row["organism_match"] == "unknown"  # not "match"
    assert row["requested_organism"] == REQUESTED_ORGANISM


def test_an_explicitly_different_pathway_observation_is_a_mismatch() -> None:
    """Positive evidence of another lexicon pathway refuses the candidate."""
    gap = _gap_at_b()
    chunk = _chunk(
        "other-pathway",
        "name: fills the gap | A -> B | enzyme: SharedE",
        observed_pathways=["lipid a biosynthesis"],
    )
    result = _synthesize(
        [_bundle(gap, chunk)], requested_pathway="menaquinone biosynthesis"
    )
    assert "fills the gap" not in _names(result)
    row = result.admission["rejected"][0]
    assert row["requested_pathway_match"] == "mismatch"
    assert row["reasons"][0].startswith("requested_pathway_mismatch")


def test_silence_about_the_pathway_stays_unknown_and_is_admitted() -> None:
    """A mechanistic sentence rarely names its pathway; that is not contradiction."""
    gap = _gap_at_b()
    chunk = _chunk("silent-pathway", "name: fills the gap | A -> B | enzyme: SharedE")
    result = _synthesize([_bundle(gap, chunk)], requested_pathway="menaquinone biosynthesis")

    row = _by_name(result.admission["accepted"])["fills the gap"]
    assert row["requested_pathway_match"] == "unknown"

    # ...unless the run demands a positive match.
    strict = _synthesize(
        [_bundle(gap, chunk)],
        requested_pathway="menaquinone biosynthesis",
        admission_policy=AdmissionPolicy(require_pathway_match=True),
    )
    assert "fills the gap" not in _names(strict)


# ===========================================================================
# 7. Gap attribution survives merging.
# ===========================================================================
def test_one_claim_admitted_for_two_gaps_keeps_both_attributions() -> None:
    """A passage is routinely top-k for several gaps; neither gap may be lost.

    The same canonical claim arrives once per gap, is admitted for each, and the
    two collapse into ONE payload row. If only the first ``gap_id`` survived, the
    second gap would be reported unfilled while the reaction that fills it sits
    in the delivered payload.
    """
    seed = {
        "entities": {
            "species": [{"name": "Pseudomonas putida"}],
            "compounds": [{"name": "B"}, {"name": "P"}, {"name": "A"}],
        },
        "processes": {
            "reactions": [
                {"name": "R0", "inputs": ["B"], "outputs": ["P"]},
                {"name": "R1", "inputs": ["Q"], "outputs": ["A"]},
            ]
        },
    }
    gaps = detect_gaps(
        seed, {}, requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    gap_b = next(g for g in gaps if g.label == "B" and g.kind == "orphan_metabolite")
    gap_a = next(g for g in gaps if g.label == "A" and g.kind == "orphan_metabolite")

    chunk = _chunk("shared", "name: bridges A and B | A -> B | enzyme: E1")
    result = _synthesize(
        [_bundle(gap_b, chunk), _bundle(gap_a, chunk, score=0.4)],
        seed=seed,
        gaps=gaps,
    )

    rows = [
        r
        for r in result.payload["processes"]["reactions"]
        if r["name"] == "bridges A and B"
    ]
    assert len(rows) == 1, "one claim, one row"
    prov = rows[0]["rag_provenance"]
    assert set(prov["gap_ids"]) == {gap_a.gap_id, gap_b.gap_id}
    assert prov["gap_id"] in prov["gap_ids"]  # primary is one of them

    # Neither gap is reported unfilled.
    unfilled = {row.get("gap_id") for row in result.unresolved_gaps}
    assert gap_a.gap_id not in unfilled
    assert gap_b.gap_id not in unfilled


def test_a_single_gap_row_keeps_its_old_provenance_shape() -> None:
    """Blast radius: ``gap_ids`` appears only when it adds something."""
    gap = _gap_at_b()
    chunk = _chunk("plain", "name: fills the gap | A -> B | enzyme: E1")
    result = _synthesize([_bundle(gap, chunk)])
    prov = next(
        r for r in result.payload["processes"]["reactions"] if r["name"] == "fills the gap"
    )["rag_provenance"]
    assert prov["gap_id"] == gap.gap_id
    assert "gap_ids" not in prov


# ===========================================================================
# 8. The span must state the RELATIONSHIP, with direction and roles.
# ===========================================================================
def _claim(name, inputs, outputs, enzymes, quote):
    return {
        "name": name,
        "inputs": [{"name": n, "stoichiometry": None} for n in inputs],
        "outputs": [{"name": n, "stoichiometry": None} for n in outputs],
        "enzymes": list(enzymes),
        "reversible": False,
        "quote": quote,
    }


def _relation_result(text, claim):
    gap = _gap_at_b()
    chunk = _chunk("relation", text)
    return _synthesize([_bundle(gap, chunk)], prose_extractor=_prose_extractor(claim))


def test_a_reversed_arrow_span_refutes_a_forward_claim() -> None:
    """(a) claim ``A -> B``, span ``B -> A``. Same names, opposite chemistry."""
    result = _relation_result(
        "B -> A", _claim("reversed arrow", ["A"], ["B"], [], "B -> A")
    )
    assert "reversed arrow" not in _names(result)
    reasons = _by_name(result.admission["rejected"])["reversed arrow"]["reasons"]
    assert any(r.startswith("evidence_direction_disagrees_with_claim") for r in reasons)


def test_a_reversible_arrow_span_does_admit_either_direction() -> None:
    """The exception the rule names: the span itself states reversibility."""
    result = _relation_result(
        "B <=> A", _claim("reversible", ["A"], ["B"], [], "B <=> A")
    )
    assert "reversible" in _names(result)


def test_a_reversed_prose_span_refutes_a_forward_claim() -> None:
    """(b) claim ``A -> B``, span "B is converted to A"."""
    result = _relation_result(
        "B is converted to A by SeedEnz.",
        _claim("reversed prose", ["A"], ["B"], [], "B is converted to A by SeedEnz."),
    )
    assert "reversed prose" not in _names(result)
    reasons = _by_name(result.admission["rejected"])["reversed prose"]["reasons"]
    assert any(r.startswith("evidence_direction_disagrees_with_claim") for r in reasons)


def test_co_occurrence_with_a_word_that_merely_contains_a_cue_is_refused() -> None:
    """(c) "form" inside "formaldehyde" is not a reaction assertion."""
    text = "A and B were measured with formaldehyde."
    result = _relation_result(text, _claim("measured", ["A"], ["B"], [], text))
    assert "measured" not in _names(result)
    reasons = _by_name(result.admission["rejected"])["measured"]["reasons"]
    assert any(r.startswith("evidence_states_no_reaction_relation") for r in reasons)

    # ...and the cue matcher itself agrees, at the level below.
    from t2pw.rag.admission import _RELATION_CUE_RE

    assert not _RELATION_CUE_RE.search("formaldehyde was quantified")
    assert not _RELATION_CUE_RE.search("the transferase was purified")
    assert _RELATION_CUE_RE.search("the acyl group is transferred")
    assert _RELATION_CUE_RE.search("B is formed from A")


def test_a_catalyst_from_a_neighbouring_clause_is_not_this_reactions_catalyst() -> None:
    """(d) "Enz1 inhibits X, while A is converted to B" does not attribute Enz1."""
    text = "Enz1 inhibits X, while A is converted to B."
    result = _relation_result(text, _claim("borrowed", ["A"], ["B"], ["Enz1"], text))
    assert "borrowed" not in _names(result)
    reasons = _by_name(result.admission["rejected"])["borrowed"]["reasons"]
    assert any(r.startswith("unsupported_catalyst_injection") for r in reasons)

    # The same sentence without the false attribution IS admitted: the reaction
    # is real, only the catalyst claim was not.
    honest = _relation_result(text, _claim("honest", ["A"], ["B"], [], text))
    assert "honest" in _names(honest)


def test_the_implemented_prose_patterns_are_all_accepted() -> None:
    """(e) each explicitly implemented role-assigning template still works."""
    cases = {
        "SeedEnz catalyzes the conversion of A to B.": ["SeedEnz"],
        "A is converted to B by SeedEnz.": ["SeedEnz"],
        "B is produced from A by SeedEnz.": ["SeedEnz"],
        "SeedEnz converts A to B.": ["SeedEnz"],
        "SeedEnz catalyzes the demethylation of A, producing B.": ["SeedEnz"],
        "SeedEnz phosphorylates A to form B.": ["SeedEnz"],
    }
    for text, enzymes in cases.items():
        result = _relation_result(text, _claim("pattern", ["A"], ["B"], enzymes, text))
        assert "pattern" in _names(result), text


def test_an_arrow_span_must_agree_on_participants_too() -> None:
    """A span that states a different product does not support the claim."""
    result = _relation_result(
        "A -> C", _claim("wrong product", ["A"], ["B"], [], "A -> C")
    )
    assert "wrong product" not in _names(result)


# ===========================================================================
# 9. Hub metabolites as the gap TARGET.
# ===========================================================================
def _hub_seed() -> dict:
    """``acetyl-CoA -> P``: the pathway's open end is a currency metabolite."""
    return {
        "entities": {
            "species": [{"name": "Pseudomonas putida"}],
            "compounds": [{"name": "acetyl-CoA"}, {"name": "P"}],
        },
        "processes": {
            "reactions": [
                {"name": "R0 seed step", "inputs": ["acetyl-CoA"], "outputs": ["P"]}
            ]
        },
    }


def test_a_hub_target_gap_admits_neither_of_two_unrelated_acetyl_coa_reactions() -> None:
    """Everything touches acetyl-CoA, so touching it proves nothing.

    Both reactions below are real, well-evidenced, and share the gap's metabolite.
    Neither shares anything else with the pathway, so neither may be admitted just
    because the dead end happens to be a currency metabolite.
    """
    seed = _hub_seed()
    gaps = detect_gaps(
        seed,
        {},
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    gap = next(g for g in gaps if g.label == "acetyl-CoA")
    chunk = _chunk(
        "hub",
        "\n".join(
            [
                "name: fatty acid step | acetyl-CoA -> malonyl-CoA | enzyme: AccA",
                "name: citrate step | acetyl-CoA -> citrate | enzyme: GltA",
            ]
        ),
    )
    result = _synthesize([_bundle(gap, chunk)], seed=seed, gaps=gaps)

    assert "fatty acid step" not in _names(result)
    assert "citrate step" not in _names(result)
    for row in result.admission["rejected"]:
        assert row["reasons"][0].startswith("hub_target_needs_additional_evidence")

    # The gap is left open for review rather than closed by whatever came first.
    unresolved = {g["gap_id"] for g in result.unresolved_gaps if g.get("gap_id")}
    assert gap.gap_id in unresolved


def test_a_non_hub_target_still_fills_directly() -> None:
    """theobromine and UMP are not hubs; the guard must not touch them."""
    gap = _gap_at_b()
    chunk = _chunk("plain", "name: fills the gap | A -> B | enzyme: SharedE")
    assert "fills the gap" in _names(_synthesize([_bundle(gap, chunk)]))


# ===========================================================================
# 10. Span-local pathway evidence takes precedence over the paper's.
# ===========================================================================
_MIXED_QUOTE = (
    "Linking caffeine degradation to cholesterol biosynthesis, SharedE "
    "converts A to B."
)


def test_a_span_naming_another_pathway_is_a_mismatch_despite_the_papers_topic() -> None:
    """The paper is about caffeine degradation; THIS sentence is not."""
    gap = _gap_at_b()
    quote = "In cholesterol biosynthesis, SharedE converts A to B."
    chunk = _chunk("mixed-paper", quote, observed_pathways=["caffeine degradation"])
    result = _synthesize(
        [_bundle(gap, chunk)],
        prose_extractor=_prose_extractor(
            _claim("off-pathway sentence", ["A"], ["B"], ["SharedE"], quote)
        ),
    )
    assert "off-pathway sentence" not in _names(result)
    row = _by_name(result.admission["rejected"])["off-pathway sentence"]
    assert row["requested_pathway_match"] == "mismatch"
    assert row["reasons"][0].startswith("requested_pathway_mismatch")


def test_a_span_naming_both_pathways_is_recorded_as_mixed_not_as_match() -> None:
    """Mixed local evidence is its own state, never silently rounded to match."""
    gap = _gap_at_b()
    chunk = _chunk("both", _MIXED_QUOTE)
    claim = _claim("both pathways", ["A"], ["B"], ["SharedE"], _MIXED_QUOTE)

    result = _synthesize([_bundle(gap, chunk)], prose_extractor=_prose_extractor(claim))
    row = _by_name(result.admission["accepted"])["both pathways"]
    assert row["requested_pathway_match"] == "mixed"

    strict = _synthesize(
        [_bundle(gap, chunk)],
        prose_extractor=_prose_extractor(claim),
        admission_policy=AdmissionPolicy(require_pathway_match=True),
    )
    assert "both pathways" not in _names(strict)



# ===========================================================================
# 11. The chain audit trail is stable under permutation.
# ===========================================================================
_TWO_PARENTS = [
    "name: parentA | A -> B | enzyme: E1",
    "name: parentB | A -> B + M | enzyme: E2",
    "name: parentC | A -> B + M | enzyme: E3",
    "name: child | M -> N | enzyme: E4",
]


def test_the_recorded_parent_is_stable_when_two_parents_share_the_metabolite() -> None:
    """Two same-hop reactions introduce M; the child's lineage must not depend on order.

    ``parentB`` and ``parentC`` are both admitted at hop 0 and both introduce
    ``M``, so both are legitimate parents for ``child``. Picking "whichever came
    first" makes the audit trail an artefact of the retrieval order; the parent is
    chosen by the smallest canonical claim key instead, which is a property of the
    claims themselves.
    """
    forward = _chain_result(_TWO_PARENTS)
    reversed_ = _chain_result(list(reversed(_TWO_PARENTS)))

    def _child(result):
        return _by_name(result.admission["accepted"])["child"]["chain"]

    assert _child(forward)["hop"] == 1
    assert _child(forward)["shared_metabolite"] == "m"
    assert _child(forward) == _child(reversed_)
    assert _child(forward)["parent_name"] in {"parentB", "parentC"}
    assert _child(forward)["parent_claim"]


def test_every_permutation_of_the_evidence_gives_the_same_lineage() -> None:
    """Not just one reversal: all six orderings of the three parents agree."""
    import itertools

    lineages = set()
    for order in itertools.permutations(_TWO_PARENTS[:3]):
        result = _chain_result(list(order) + [_TWO_PARENTS[3]])
        chain = _by_name(result.admission["accepted"])["child"]["chain"]
        lineages.add((chain["hop"], chain["shared_metabolite"], chain["parent_claim"]))
    assert len(lineages) == 1, lineages


# ===========================================================================
# 6. Multi-relation spans (C-061). One sentence, several stated reactions.
# ===========================================================================
#: PMC12421875's MenA/MenG sentence — two reactions, one statement, and the
#: shape that made a verbatim claim disagree with its own evidence.
_TWO_CLAUSE_SPAN = (
    "Subsequently, MenA joins DHNA and prenyl diphosphate to produce "
    "demethylmenaquinone (DMK), and MenG demethylates DMK to generate MK ( Fig. 1A )."
)


def _span_check(inputs, outputs, enzymes=(), span=_TWO_CLAUSE_SPAN):
    from t2pw.rag.admission import validate_evidence_span

    return validate_evidence_span(
        span, inputs=list(inputs), outputs=list(outputs), enzymes=list(enzymes)
    )


@pytest.mark.parametrize(
    "inputs,outputs,enzymes,why",
    [
        (
            ["DHNA", "prenyl diphosphate"],
            ["demethylmenaquinone (DMK)"],
            ["MenG"],
            "clause one's chemistry under clause two's enzyme",
        ),
        (
            ["DMK"],
            ["MK"],
            ["MenA"],
            "clause two's chemistry under clause one's enzyme",
        ),
        (
            ["DHNA", "prenyl diphosphate"],
            ["MK"],
            ["MenA"],
            "first substrates stitched to the last product",
        ),
        (
            ["demethylmenaquinone (DMK)"],
            ["DHNA", "prenyl diphosphate"],
            ["MenA"],
            "clause one, backwards",
        ),
        (["MK"], ["DMK"], ["MenG"], "clause two, backwards"),
        (
            ["DHNA"],
            ["demethylmenaquinone (DMK)"],
            ["MenA"],
            "clause one with a stated co-substrate dropped",
        ),
        (
            # ATP here would be ADMITTED, and correctly: currency is exempt in
            # the "candidate has extra" direction, because a transcriber
            # legitimately writes the implicit cofactor of a step described in
            # words. Chorismate is not currency, and the span never states it.
            ["DHNA", "prenyl diphosphate", "chorismate"],
            ["demethylmenaquinone (DMK)"],
            ["MenA"],
            "a non-currency substrate the span never states",
        ),
    ],
)
def test_a_sentence_stating_two_reactions_supports_neither_a_third(
    inputs, outputs, enzymes, why
) -> None:
    """Reading BOTH relations must not make the pair support their mixtures.

    This is the merge-gate-6 half of C-061. Widening the parser widens what the
    span is read as SAYING; the agreement predicate is untouched, so each stated
    relation is still judged on substrates, products, direction and catalyst at
    once, and a claim that matches none of them outright is refused.
    """
    verdict = _span_check(inputs, outputs, enzymes)
    assert not verdict.ok, (why, verdict.relation)
    assert verdict.reasons, why


def test_the_two_reactions_the_sentence_does_state_are_supported() -> None:
    """The other half: the refusals above are not a gate that refuses everything."""
    assert _span_check(
        ["DHNA", "prenyl diphosphate"], ["demethylmenaquinone (DMK)"], ["MenA"]
    ).ok
    assert _span_check(["DMK"], ["MK"], ["MenG"]).ok


def test_a_claim_stitched_across_two_sentences_is_still_refused() -> None:
    """The pre-existing guarantee, re-checked against the wider parser.

    ``split_statements`` refuses a multi-statement span before any parsing
    happens, and enumerating every match position does not reach that check.
    """
    verdict = _span_check(
        ["A"],
        ["Y"],
        ["Enz1"],
        span="Enz1 catalyzes A to B. Enz2 catalyzes X to Y.",
    )
    assert not verdict.ok
    assert any("evidence_span_is_not_one_statement" in r for r in verdict.reasons)


# ===========================================================================
# 7. Participant-dropping restarts (C-061a). REV-061's hole in section 6.
# ===========================================================================
# Trying every template at every word start also re-matches THE SAME reaction
# with a stated participant deleted: ``is_converted_to`` and ``is_produced_from``
# open on an unanchored participant group, so a restart past a coordinator reads
# "Isochorismate and 2-oxoglutarate are converted to SEPHCHC" as
# ``2-oxoglutarate -> SEPHCHC``. That contradicts the invariant
# ``validate_evidence_span`` states in its own docstring — currency is "never
# exempt in the 'span has it, candidate dropped it' direction, because dropping a
# real substrate changes the chemistry" — and it is merge rule 6: a biological
# gate weakened in the direction that increases PWML output.
#
# ``\b\w`` also restarts INSIDE a compound, offering ``oxoglutarate`` from
# ``2-oxoglutarate`` and ``3-oxidosqualene`` from ``2,3-oxidosqualene`` — the very
# defect ``_SPECIES_SPLIT_RE``'s comment records, reopened by another route.
_MEND = "Isochorismate and 2-oxoglutarate are converted to SEPHCHC by MenD."
_THREE = "A, B and C are converted to D by E."
_SDH = "Fumarate and NADH are produced from succinate by Sdh."
_HK = "Glucose and ATP are converted to glucose-6-phosphate by HK."
_LOCANT = "2,3-oxidosqualene is converted to lanosterol by Erg7."


@pytest.mark.parametrize(
    "span,inputs,outputs,enzymes,why",
    [
        (_MEND, ["2-oxoglutarate"], ["SEPHCHC"], ["MenD"], "A1 first substrate deleted"),
        (_MEND, ["Isochorismate"], ["SEPHCHC"], ["MenD"], "A5 last substrate deleted"),
        (_THREE, ["B", "C"], ["D"], ["E"], "A1 a later substrate deleted"),
        (_THREE, ["C"], ["D"], ["E"], "A1 two substrates deleted"),
        (_SDH, ["succinate"], ["NADH"], ["Sdh"], "A1 a stated co-product deleted"),
        (_HK, ["ATP"], ["glucose-6-phosphate"], ["HK"], "A1 empty non-currency lhs"),
        (_MEND, ["oxoglutarate"], ["SEPHCHC"], ["MenD"], "A2 hyphen locant fragment"),
        (_LOCANT, ["3-oxidosqualene"], ["lanosterol"], ["Erg7"], "A2 comma locant"),
        (_LOCANT, ["oxidosqualene"], ["lanosterol"], ["Erg7"], "A2 comma locant, both cuts"),
    ],
)
def test_a_reading_the_span_does_not_state_is_not_admitted(
    span, inputs, outputs, enzymes, why
) -> None:
    """A restart may not delete a participant the sentence states.

    ``_HK`` is the sharp one: ATP is currency, so the surviving reading's
    non-currency substrate set is ``{glucose}`` and the claim's is EMPTY. Before
    the correction the drop-reading made those two empty sets equal and the claim
    was admitted against a reaction with no substrate at all.
    """
    verdict = _span_check(inputs, outputs, enzymes, span=span)
    assert not verdict.ok, (why, verdict.relation)
    assert verdict.reasons, why


@pytest.mark.parametrize(
    "span,inputs,outputs,enzymes",
    [
        (_MEND, ["Isochorismate", "2-oxoglutarate"], ["SEPHCHC"], ["MenD"]),
        (_THREE, ["A", "B", "C"], ["D"], ["E"]),
        (_SDH, ["succinate"], ["Fumarate", "NADH"], ["Sdh"]),
        (_HK, ["Glucose", "ATP"], ["glucose-6-phosphate"], ["HK"]),
        (_LOCANT, ["2,3-oxidosqualene"], ["lanosterol"], ["Erg7"]),
    ],
)
def test_the_reading_each_span_does_state_survives_the_filter(
    span, inputs, outputs, enzymes
) -> None:
    """The filter discriminates by SUBSET, and the reading it must keep is the
    superset — the same shape as C-061's MenA reading. Getting the direction
    backwards would delete exactly what the pair of cards exists to recover."""
    assert _span_check(inputs, outputs, enzymes, span=span).ok


def test_dropping_the_first_and_the_last_participant_behave_identically() -> None:
    """A5. The asymmetry was the proof it was an artifact, not a reading."""
    first = _span_check(["2-oxoglutarate"], ["SEPHCHC"], ["MenD"], span=_MEND)
    last = _span_check(["Isochorismate"], ["SEPHCHC"], ["MenD"], span=_MEND)
    assert (first.ok, last.ok) == (False, False)
    assert first.reasons and last.reasons


@pytest.mark.parametrize("span", [_MEND, _THREE, _SDH, _HK, _LOCANT, _TWO_CLAUSE_SPAN])
def test_the_filter_only_ever_removes_and_never_empties(span) -> None:
    """A6 and A8, structurally.

    Survivors are the SAME objects in the SAME order — no branch constructs,
    rewrites or reorders a relation — and a non-empty list stays non-empty,
    because both drop conditions require a strictly larger participant count so
    the largest reading has no possible eliminator. A span that did somehow lose
    every reading would still fail CLOSED: ``validate_evidence_span`` refuses an
    empty relation list rather than falling through to a default.
    """
    from t2pw.rag import admission

    body = admission._text(span)
    raw = [
        relation
        for name, pattern in admission._ALL_PROSE_PATTERNS
        for match in admission._pattern_matches(pattern, body, first_only=False)
        for relation in [admission._relation_from_match(name, match, body)]
        if relation is not None
    ]
    kept = admission._drop_participant_subsets(raw)
    assert raw, span
    assert kept, "the filter emptied a non-empty reading list"
    assert [id(r) for r in kept] == [id(r) for r in raw if any(r is k for k in kept)]


def test_the_single_relation_shim_is_untouched_by_the_restart_exclusion() -> None:
    """``parse_span_relation`` reproduces the historical leftmost match, and the
    exclusion never removes offset 0, so the shim's answer cannot move."""
    from t2pw.rag.admission import parse_span_relation

    relation = parse_span_relation(_MEND)
    assert relation is not None
    assert relation.inputs == ["Isochorismate", "2-oxoglutarate"]
    assert relation.outputs == ["SEPHCHC"]

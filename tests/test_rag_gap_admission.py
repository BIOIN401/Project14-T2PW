"""Gap-bound admission: a retrieved reaction enters the pathway only by earning it.

What these tests pin
--------------------
Before the admission gate, :mod:`t2pw.rag.synthesize` transcribed every reaction
stated by every retrieved passage and merged them all. Because a passage is
retrieved per gap but states whatever the paper says, that was functionally
"broad pathway retrieval, then merge everything": run ``2026-07-28_0919`` leg
PMC12444477/strict delivered 27 reactions of which #10-26 were phospholipid
biosynthesis imported from PMC12898747 — real biology from a real paper, filling
no gap of the lipid A pathway that was actually requested, and indistinguishable
from the requested pathway's own chemistry once merged.

Every test below therefore drives the REAL
``synthesize_with_report`` / ``admit_candidates`` path with a real
``EvidenceBundle``. None of them constructs an admission verdict by hand — that
is precisely the mistake ``tests/test_scope_membership_filter_end_to_end.py``
documents for the out-of-scope filter, whose two predecessors passed for the
entire time the feature was inert because they hand-built the payload the real
pipeline never produced.

Offline / deterministic: no chromadb, no network, no live LLM. The guarded
``openai`` stub mirrors tests/test_rag_synthesize.py because importing
``t2pw.rag.retrieve`` pulls ``ingest -> acquire -> mapping -> llm.client ->
openai`` at import time.
"""

from __future__ import annotations

import json
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

from t2pw.pipeline.pipeline import (  # noqa: E402
    clean_inference_output,
    filter_out_of_scope_reactions,
    merge_additions,
)
from t2pw.pipeline.process_normalizer import normalize_process_payload  # noqa: E402
from t2pw.pipeline.stage_contracts import validate_post_extraction  # noqa: E402
from t2pw.rag.admission import (  # noqa: E402
    ADMISSION_RULES,
    AdmissionPolicy,
    RagReactionCandidate,
    SCOPE_ADMITTED,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    admit_candidates,
)
from t2pw.rag.conform import conform_rag_additions_for_merge  # noqa: E402
from t2pw.rag.retrieve import (  # noqa: E402
    EvidenceBundle,
    Gap,
    GapContractError,
    detect_gaps,
    make_gap_id,
    query_for_gap,
)
from t2pw.rag.store import Chunk, Retrieved  # noqa: E402
from t2pw.rag.synthesize import synthesize_with_report  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — one seed pathway with one real dangling end, and papers about it.
# ---------------------------------------------------------------------------
REQUESTED_PATHWAY = "caffeine degradation"
REQUESTED_ORGANISM = "Pseudomonas putida"


def _seed_payload() -> dict:
    """Caffeine -> theobromine. Theobromine is produced and consumed by nothing."""
    return {
        "entities": {
            "species": [{"name": "Pseudomonas putida"}],
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
                    "name": "R1 caffeine N3-demethylation",
                    "inputs": ["caffeine", "O2"],
                    "outputs": ["theobromine", "formaldehyde"],
                    "enzymes": [{"protein": "NdmA"}],
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
            "source_title": "Caffeine degradation (seed paper)",
            "source_type": "paper",
            "source_uri": "https://example.org/seed",
            "organism": "Pseudomonas putida",
        },
    }


def _theobromine_gap(**overrides) -> Gap:
    """The real dangling end of the seed, as ``detect_gaps`` produces it."""
    gaps = detect_gaps(
        _seed_payload(),
        {},
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    gap = next(g for g in gaps if g.label.casefold() == "theobromine")
    for key, value in overrides.items():
        setattr(gap, key, value)
    return gap


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_id: str,
    organism: str = "Pseudomonas putida",
    title: str = "Downstream caffeine catabolism",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        source_id=source_id,
        source_title=title,
        source_type="paper",
        source_uri=f"https://example.org/{source_id}",
        organism=organism,
        section="results",
    )


#: The exact gap-filling step: it CONSUMES theobromine, closing the seed's end.
_FILLING_EQUATION = (
    "name: R2 theobromine N7-demethylation | "
    "theobromine + O2 -> 7-methylxanthine + formaldehyde | enzyme: NdmB"
)

#: Correct biology from the same paper about a DIFFERENT pathway. It touches
#: nothing in the caffeine route: the "merely adjacent pathway context" case.
_NON_GAP_EQUATION = (
    "name: glucose phosphorylation | glucose + ATP -> glucose-6-phosphate + ADP "
    "| enzyme: hexokinase"
)


def _bundle(*lines: str, chunk_id: str = "chunkB1", source_id: str = "PMID:B",
            organism: str = "Pseudomonas putida", gap: Gap = None) -> EvidenceBundle:
    """A bundle for ``gap``, carrying the query the real retriever would have run.

    A gap with a blanked ``gap_id`` cannot produce a query at all (that is the
    point of :class:`GapContractError`), so such a bundle records the retrieval
    that WOULD have happened and then drops the id — the only way to reach the
    downstream gate with an unattributed candidate.
    """
    gap = gap if gap is not None else _theobromine_gap()
    chunk = _chunk(chunk_id, "\n".join(lines), source_id=source_id, organism=organism)
    if gap.gap_id:
        query = query_for_gap(gap)
    else:
        query = query_for_gap(_theobromine_gap())
    return EvidenceBundle(gap=gap, query=query, hits=[Retrieved(chunk=chunk, score=0.9)])


def _synthesize(bundles, **kwargs):
    params = {
        "requested_pathway": REQUESTED_PATHWAY,
        "requested_organism": REQUESTED_ORGANISM,
    }
    params.update(kwargs)
    return synthesize_with_report(
        _seed_payload(), list(bundles), _seed_context(), **params
    )


def _reaction_names(payload: dict) -> list:
    return [r.get("name") for r in payload["processes"]["reactions"]]


def _participant_names(participants) -> list:
    return [p if isinstance(p, str) else p.get("name") for p in participants]


# ---------------------------------------------------------------------------
# The gap contract.
# ---------------------------------------------------------------------------
def test_every_detected_gap_carries_a_stable_id_and_a_structured_description() -> None:
    """Requirement 1: a gap is a named request, not "something is missing near X"."""
    gaps = detect_gaps(
        _seed_payload(),
        {},
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    assert gaps

    for gap in gaps:
        assert gap.gap_id, f"{gap.kind}/{gap.label} has no gap_id"
        assert gap.missing_relationship
        assert gap.expected_type
        assert gap.reason
        assert gap.requested_pathway == REQUESTED_PATHWAY
        assert gap.requested_organism == REQUESTED_ORGANISM

    # Ids are unique within a detection pass...
    ids = [gap.gap_id for gap in gaps]
    assert len(ids) == len(set(ids))
    # ...and STABLE: the same pathway detected again yields the same ids, so an
    # admission report can be diffed across runs.
    again = detect_gaps(_seed_payload(), {}, requested_pathway=REQUESTED_PATHWAY)
    assert [gap.gap_id for gap in again] == ids

    theobromine = next(g for g in gaps if g.label.casefold() == "theobromine")
    assert theobromine.gap_id == make_gap_id(theobromine.kind, "theobromine")
    assert theobromine.expected_type == "reaction"
    # The adjacent EXISTING entities are the anchor an admitted candidate must
    # attach to — here, the rest of the reaction that produced theobromine.
    assert {"caffeine", "NdmA"} <= set(theobromine.adjacent_entities)


def test_every_retrieval_query_names_its_gap_and_a_broad_query_is_refused() -> None:
    """Requirement 2: no query without a gap_id, so no unattributable retrieval."""
    gap = _theobromine_gap()
    query = query_for_gap(gap, seed_context="caffeine degradation")

    assert f"Gap-ID: {gap.gap_id}" in query
    assert "theobromine" in query  # the exact symbol still steers the lexical half
    assert "caffeine degradation" in query  # seed context still appended

    # A gap whose id has been stripped is exactly the "retrieve the whole pathway
    # and merge whatever comes back" shape, and it cannot be built.
    gap.gap_id = ""
    try:
        query_for_gap(gap)
    except GapContractError as exc:
        assert "gap_id" in str(exc)
    else:  # pragma: no cover - the raise above is the assertion
        raise AssertionError("a gap with no gap_id must not produce a query")


def test_a_gaps_fill_target_is_its_metabolites_not_its_catalysts() -> None:
    """Sharing a CATALYST with the gap's reaction is not filling the gap.

    Found by the bounded replay of ``runs/2026-07-28_0919`` PMC12444477: a
    separate historical defect had sprayed every enzyme onto every reaction, so
    thirteen imported *phospholipid* reactions listed ``WaaA`` or ``LpxM`` among
    their catalysts. While the fill target was ``Gap.symbols`` (which carries the
    neighbouring reaction's name and catalysts as RETRIEVAL aids) all thirteen
    read as "fills the WaaA/LpxM reaction gap **directly**" — the exact
    wrong-pathway import the gate exists to stop. Every ``missing_relationship``
    says "shares a substrate or product", and the target now matches that wording.
    """
    gaps = detect_gaps(_seed_payload(), {}, requested_pathway=REQUESTED_PATHWAY)
    reaction_gap = next(g for g in gaps if g.kind == "dangling_reaction")

    # The retrieval symbols still carry the catalyst — a lexical retriever needs it.
    assert "NdmA" in reaction_gap.symbols
    # The FILL target does not.
    assert "NdmA" not in reaction_gap.target_names()
    assert {"caffeine", "theobromine"} <= set(reaction_gap.target_names())

    # An unrelated reaction that shares only the catalyst is refused.
    sharing_only_the_catalyst = RagReactionCandidate(
        gap_id=reaction_gap.gap_id,
        name="unrelated step",
        inputs=["compound-Q"],
        outputs=["compound-R"],
        enzymes=["NdmA"],
        evidence={
            "text": "NdmA also converts compound-Q to compound-R.",
            "chunk_id": "c1",
            "source_id": "PMID:X",
        },
        evidence_span="NdmA also converts compound-Q to compound-R.",
        source_paper={"source_id": "PMID:X"},
    )
    accepted, report = admit_candidates(
        [sharing_only_the_catalyst], gaps=gaps, seed_payload=_seed_payload()
    )
    assert accepted == []
    # ``does_not_fill_named_gap``, not ``adjacent_pathway_context_only``: sharing
    # NdmA does not even make it adjacent, because a catalyst is not a node.
    assert report.rejected[0]["reasons"][0].startswith("does_not_fill_named_gap")


def test_the_admission_rules_are_declared_as_data() -> None:
    """The gate's contract is printable and assertable, not buried in control flow."""
    names = [name for name, _ in ADMISSION_RULES]
    assert names == [
        "named_gap",
        "valid_entity_types",
        "local_evidence_span",
        "evidence_supports_participants",
        "relation_agrees",
        "no_unsupported_catalyst",
        "organism_compatible",
        "requested_pathway",
        "gap_type_compatible",
        "connects_to_pathway",
        "fills_its_named_gap",
        # C-059 appended two, and only appended: the eleven above are unchanged in
        # name and in order. A "gap" an existing reaction already covers is not a
        # gap, and one canonical claim from one span is admitted once however many
        # adjacent gaps retrieved it.
        "gap_not_already_covered",
        "not_already_admitted_for_another_gap",
    ]


# ---------------------------------------------------------------------------
# Admission — the required cases.
# ---------------------------------------------------------------------------
def test_an_exact_gap_filling_reaction_is_accepted() -> None:
    """The whole point: a reaction that closes the named gap gets in."""
    result = _synthesize([_bundle(_FILLING_EQUATION)])

    assert "R2 theobromine N7-demethylation" in _reaction_names(result.payload)

    admission = result.admission
    assert admission["counts"] == {"accepted": 1, "rejected": 0, "considered": 1}

    row = admission["accepted"][0]
    gap = _theobromine_gap()
    # Requirement 3: every field the decision rests on is on the candidate.
    assert row["gap_id"] == gap.gap_id
    assert row["source_paper"]["source_id"] == "PMID:B"
    assert row["evidence"]["chunk_id"] == "chunkB1"
    assert row["evidence"]["source_uri"] == "https://example.org/PMID:B"
    assert "theobromine" in row["evidence"]["span"]
    # The span narrows the claim to ONE parsed line; the parent chunk pointer is
    # retained beside it, never replaced.
    assert row["evidence"]["span"] == _FILLING_EQUATION
    assert row["organism"] == "Pseudomonas putida"
    assert row["requested_organism"] == REQUESTED_ORGANISM
    assert row["organism_match"] == "match"
    assert row["requested_pathway"] == REQUESTED_PATHWAY
    assert row["requested_pathway_match"] in {"match", "unknown"}
    assert row["scope_membership"] == SCOPE_ADMITTED
    assert "theobromine" in row["connected_entities"]
    assert row["confidence"] > 0
    assert row["status"] == STATUS_ACCEPTED
    assert row["reasons"] and "fills_named_gap" in row["reasons"][0]

    # The gap is no longer reported unfilled.
    assert gap.gap_id not in {g.get("gap_id") for g in result.unresolved_gaps}


def test_a_related_but_non_gap_reaction_is_rejected() -> None:
    """Correct biology, from the same paper, filling no gap: it stays out.

    This is the PMC12898747 shape that put 17 phospholipid-biosynthesis reactions
    into a lipid A payload. The reaction is perfectly real and perfectly cited —
    it simply is not what was searched for.
    """
    result = _synthesize([_bundle(_FILLING_EQUATION, _NON_GAP_EQUATION)])

    names = _reaction_names(result.payload)
    assert "R2 theobromine N7-demethylation" in names
    assert "glucose phosphorylation" not in names

    # ...and nothing it named leaked in through the entity registry either.
    entity_names = {
        row["name"]
        for bucket in result.payload["entities"].values()
        for row in bucket
    }
    assert not entity_names & {"glucose", "glucose-6-phosphate", "hexokinase"}

    rejected = result.admission["rejected"]
    assert [row["name"] for row in rejected] == ["glucose phosphorylation"]
    assert rejected[0]["status"] == STATUS_REJECTED
    reason = rejected[0]["reasons"][0]
    assert reason.startswith(
        ("does_not_fill_named_gap", "adjacent_pathway_context_only",
         "not_connected_to_pathway_or_anchor")
    )
    assert _theobromine_gap().gap_id in reason


def test_a_wrong_organism_reaction_is_rejected() -> None:
    """An explicit organism mismatch refuses the candidate under any policy but 'off'."""
    bundle = _bundle(
        _FILLING_EQUATION, chunk_id="chunkH", source_id="PMID:H", organism="Homo sapiens"
    )
    result = _synthesize([bundle])

    assert "R2 theobromine N7-demethylation" not in _reaction_names(result.payload)
    rejected = result.admission["rejected"]
    assert len(rejected) == 1
    assert rejected[0]["organism"] == "Homo sapiens"
    assert rejected[0]["organism_match"] == "mismatch"
    assert rejected[0]["reasons"][0].startswith("organism_mismatch")

    # A related taxon is NOT a mismatch: the gate reuses the eligibility
    # taxonomy, where a same-genus species is 'genus_level' evidence.
    related = _bundle(
        _FILLING_EQUATION,
        chunk_id="chunkR",
        source_id="PMID:R",
        organism="Pseudomonas aeruginosa",
    )
    assert "R2 theobromine N7-demethylation" in _reaction_names(
        _synthesize([related]).payload
    )

    # An unnamed organism is admitted by the default policy and refused by
    # 'strict' — that is what "explicitly unknown under configured policy" means.
    silent = _bundle(
        _FILLING_EQUATION, chunk_id="chunkU", source_id="PMID:U", organism=""
    )
    assert "R2 theobromine N7-demethylation" in _reaction_names(
        _synthesize([silent]).payload
    )
    strict = _synthesize(
        [silent], admission_policy=AdmissionPolicy(organism_policy="strict")
    )
    assert "R2 theobromine N7-demethylation" not in _reaction_names(strict.payload)
    assert strict.admission["rejected"][0]["reasons"][0].startswith(
        "organism_unknown_refused"
    )


def test_a_candidate_with_no_gap_id_is_rejected() -> None:
    """Requirement: a reaction that names no gap cannot be admitted as filling one."""
    # Path 1 — through synthesis: a bundle whose gap has had its id stripped.
    result = _synthesize([_bundle(_FILLING_EQUATION, gap=_theobromine_gap(gap_id=""))])
    assert result.payload["processes"]["reactions"] == [
        r for r in result.payload["processes"]["reactions"] if r["name"].startswith("R1")
    ]
    assert "R2 theobromine N7-demethylation" not in _reaction_names(result.payload)
    assert result.admission["rejected"][0]["reasons"][0].startswith("no_gap_id")

    # Path 2 — straight at the gate, including a gap_id that was never detected.
    gap = _theobromine_gap()
    orphan = RagReactionCandidate(
        gap_id="gap-orphan_metabolite-deadbeef",
        name="R2",
        inputs=["theobromine"],
        outputs=["7-methylxanthine"],
        evidence={"text": "theobromine -> 7-methylxanthine", "chunk_id": "c"},
        evidence_span="theobromine -> 7-methylxanthine",
    )
    accepted, report = admit_candidates([orphan], gaps=[gap], seed_payload=_seed_payload())
    assert accepted == []
    assert report.rejected[0]["reasons"][0].startswith("unknown_gap_id")


def test_scope_membership_survives_synthesis_conform_cleaning_and_normalization() -> None:
    """The label the gate wrote must reach the merged, normalized payload intact.

    ``pipeline._carry_scope_membership`` records its own scope limit as "a RAG row
    cannot carry a scope label even in principle -- ``_ALLOWED_ROW_KEYS`` in
    ``rag/synthesize.py``". This walks the real boundary chain and pins that it
    now can: synthesize -> conform -> clean_inference_output -> merge_additions ->
    normalize_process_payload.
    """
    result = _synthesize([_bundle(_FILLING_EQUATION)])
    rag_row = next(
        r
        for r in result.payload["processes"]["reactions"]
        if r["name"] == "R2 theobromine N7-demethylation"
    )
    # Boundary 1 — synthesis.
    assert rag_row["scope_membership"] == SCOPE_ADMITTED
    assert rag_row["rag_provenance"]["gap_id"] == _theobromine_gap().gap_id

    # Boundary 2 — conformance to the Stage-2 additions envelope.
    envelope = conform_rag_additions_for_merge(result.payload, _seed_payload())
    conformed = next(
        r
        for r in envelope["additions"]["processes"]["reactions"]
        if r["name"] == "R2 theobromine N7-demethylation"
    )
    assert conformed["scope_membership"] == SCOPE_ADMITTED

    # Boundary 3 — the core cleaner's key-whitelist rebuild.
    cleaned = clean_inference_output(envelope)
    cleaned_row = next(
        r
        for r in cleaned["additions"]["processes"]["reactions"]
        if r["name"] == "R2 theobromine N7-demethylation"
    )
    assert cleaned_row["scope_membership"] == SCOPE_ADMITTED

    # Boundary 4 — the merge into the seed, then normalization.
    merged = merge_additions(_seed_payload(), envelope)
    merged_row = next(
        r
        for r in merged["processes"]["reactions"]
        if r.get("name") == "R2 theobromine N7-demethylation"
    )
    assert merged_row["scope_membership"] == SCOPE_ADMITTED

    normalized, _norm_report = normalize_process_payload(merged)
    normalized_row = next(
        r
        for r in normalized["processes"]["reactions"]
        if r.get("name") == "R2 theobromine N7-demethylation"
    )
    assert normalized_row["scope_membership"] == SCOPE_ADMITTED

    # And the label is one the core out-of-scope filter reads as IN scope, so the
    # admitted reaction is not deleted by the very filter it can now reach.
    filtered, removed = filter_out_of_scope_reactions(normalized)
    assert removed == []
    assert "R2 theobromine N7-demethylation" in _reaction_names(filtered)


def test_a_rejected_candidate_cannot_reappear_through_stitching() -> None:
    """Requirement 6: synthesis must not launder a refusal into an accepted row.

    The rejected reaction here shares a metabolite with an ACCEPTED one
    (7-methylxanthine), which is exactly the shape that would let stitch detection
    or conflict merging pull it back in: it is one hop from the admitted chain but
    it belongs to a gap that was never detected, and its own gap_id is absent.
    """
    laundered = (
        "name: 7-methylxanthine export | 7-methylxanthine -> 7-methylxanthine-out "
        "| enzyme: SomeTransporter"
    )
    result = _synthesize(
        [
            _bundle(_FILLING_EQUATION),
            _bundle(
                laundered,
                chunk_id="chunkC",
                source_id="PMID:C",
                gap=_theobromine_gap(gap_id=""),
            ),
        ]
    )

    names = _reaction_names(result.payload)
    assert "R2 theobromine N7-demethylation" in names
    assert "7-methylxanthine export" not in names

    # It is absent from every downstream product of synthesis, not just the
    # reaction list: no entity row, no stitch, no provenance donation.
    entity_names = {
        row["name"] for bucket in result.payload["entities"].values() for row in bucket
    }
    assert "7-methylxanthine-out" not in entity_names
    assert "SomeTransporter" not in entity_names
    stitched = json.dumps(result.stitched)
    assert "7-methylxanthine export" not in stitched
    assert "PMID:C" not in json.dumps(result.payload)

    # ...and it is PRESERVED in the report rather than silently vanishing.
    assert "7-methylxanthine export" in {r["name"] for r in result.admission["rejected"]}


def test_duplicate_evidence_does_not_amplify() -> None:
    """The same passage retrieved for several gaps must not multiply anything.

    This is the run 2026-07-28_0919 failure in miniature: one chunk was top-k for
    29 gaps and its 4,812-character passage was billed 29 times onto one reaction.
    Two gaps retrieve the SAME chunk here, so the same claim arrives twice.
    """
    gap_a = _theobromine_gap()
    gaps = detect_gaps(
        _seed_payload(), {}, requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    gap_b = next(g for g in gaps if g.label.casefold() == "r1 caffeine n3-demethylation")

    chunk = _chunk("chunkDUP", _FILLING_EQUATION, source_id="PMID:B")
    bundles = [
        EvidenceBundle(gap=gap_a, query=query_for_gap(gap_a),
                       hits=[Retrieved(chunk=chunk, score=0.9)]),
        EvidenceBundle(gap=gap_b, query=query_for_gap(gap_b),
                       hits=[Retrieved(chunk=chunk, score=0.4)]),
    ]
    result = _synthesize(bundles, gaps=gaps)

    rows = [
        r
        for r in result.payload["processes"]["reactions"]
        if r["name"] == "R2 theobromine N7-demethylation"
    ]
    assert len(rows) == 1, "one claim, one row"
    row = rows[0]
    # One passage -> one evidence record, one source paper, one source ref.
    assert len(row["evidence"]) == 1
    assert len(row["source_papers"]) == 1
    assert row["source_refs"] == ["PMID:B"]
    # ...and the enzyme actor rows carry the same single passage, not one per gap.
    assert len(row["enzymes"][0]["evidence"]) == 1

    # The conform boundary's join is the last place a repeat is still separable.
    envelope = conform_rag_additions_for_merge(result.payload, _seed_payload())
    conformed = next(
        r
        for r in envelope["additions"]["processes"]["reactions"]
        if r["name"] == "R2 theobromine N7-demethylation"
    )
    assert conformed["evidence"].count("theobromine + O2") == 1


def test_seed_and_rag_claims_stay_distinguishable() -> None:
    """A reader of the merged payload must be able to tell the two apart."""
    result = _synthesize([_bundle(_FILLING_EQUATION)])
    rows = {r["name"]: r for r in result.payload["processes"]["reactions"]}

    seed = rows["R1 caffeine N3-demethylation"]
    rag = rows["R2 theobromine N7-demethylation"]

    # Provenance points at different papers...
    assert seed["source_refs"] == ["PMID:SEED"]
    assert rag["source_refs"] == ["PMID:B"]
    # ...and only the RAG row claims to fill a gap. A seed reaction fills none:
    # it IS the pathway being extended.
    assert "gap_id" not in seed["rag_provenance"]
    assert rag["rag_provenance"]["gap_id"] == _theobromine_gap().gap_id
    # The seed keeps its OWN Stage-1 scope label, not the gate's verdict.
    assert seed["scope_membership"] == "core"
    assert rag["scope_membership"] == SCOPE_ADMITTED
    # The admission report accounts for the RAG claim only — the seed reaction is
    # never a candidate, so it can never be rejected out of its own pathway.
    assert result.admission["counts"]["considered"] == 1
    assert {r["name"] for r in result.admission["accepted"]} == {
        "R2 theobromine N7-demethylation"
    }


def test_an_accepted_rag_reaction_keeps_referential_integrity() -> None:
    """Every participant and catalyst it names is registered, and it connects."""
    result = _synthesize([_bundle(_FILLING_EQUATION)])
    payload = result.payload

    assert validate_post_extraction(payload)["ok"] is True

    registered = {
        row["name"].casefold()
        for bucket in ("compounds", "proteins", "protein_complexes")
        for row in payload["entities"].get(bucket, [])
    }
    rag = next(
        r
        for r in payload["processes"]["reactions"]
        if r["name"] == "R2 theobromine N7-demethylation"
    )
    referenced = _participant_names(rag["inputs"]) + _participant_names(rag["outputs"])
    referenced += [actor["entity"] for actor in rag["enzymes"]]
    for name in referenced:
        assert name.casefold() in registered, f"{name} is referenced but not registered"

    # It really does close the seed's dangling end: the seed's terminal product
    # is this reaction's substrate, and the two come from different papers.
    seed = next(
        r
        for r in payload["processes"]["reactions"]
        if r["name"] == "R1 caffeine N3-demethylation"
    )
    assert "theobromine" in _participant_names(seed["outputs"])
    assert "theobromine" in _participant_names(rag["inputs"])
    assert {s["metabolite"].casefold() for s in result.stitched} >= {"theobromine"}

    # Every element still carries a resolvable source pointer.
    for row in payload["processes"]["reactions"]:
        assert row["source_refs"]


# ---------------------------------------------------------------------------
# Bounds — the report is a diagnostic, not a second payload.
# ---------------------------------------------------------------------------
def test_the_admission_report_is_bounded_and_counts_what_it_drops() -> None:
    """A 4.70 MB payload of repeated passages is this subsystem's documented past."""
    long_passage = _FILLING_EQUATION + "\n" + ("filler sentence. " * 2000)
    result = _synthesize(
        [_bundle(long_passage)],
        admission_policy=AdmissionPolicy(
            max_report_entries=0, evidence_snippet_chars=64
        ),
    )

    report = result.admission
    # Counts are complete even when the kept rows are not.
    assert report["counts"]["accepted"] == 1
    assert report["accepted"] == []
    assert report["truncated"]["accepted"] == 1

    # With a normal cap the quote is a pointer-sized excerpt, not the passage.
    sized = _synthesize(
        [_bundle(long_passage)],
        admission_policy=AdmissionPolicy(evidence_snippet_chars=64),
    )
    quote = sized.admission["accepted"][0]["evidence"]["span"]
    assert len(quote) <= 65  # 64 chars + the ellipsis
    # The bound guards the parts that grow with the payload: rows, spans, gap
    # attributions. ``identity_outcomes`` is a fixed ~380-byte block of integers
    # (five proposal counts, five target counts, three gap counts) that is the
    # same size for one candidate and for ten thousand, so it is inside the bound
    # rather than exempt from it.
    assert len(json.dumps(sized.admission)) < 4600
    assert len(json.dumps(sized.admission["identity_outcomes"])) < 500


def test_a_gap_whose_evidence_was_all_refused_is_reported_unfilled() -> None:
    """"Retrieved something" is not "filled": the two failure modes are named apart."""
    result = _synthesize([_bundle(_NON_GAP_EQUATION)])

    gap = _theobromine_gap()
    unfilled = {row["gap_id"]: row for row in result.unresolved_gaps if row.get("gap_id")}
    assert gap.gap_id in unfilled
    assert "nothing admissible resolved this gap" in unfilled[gap.gap_id]["reason"]

    # A gap with no evidence at all reports the OTHER reason, so a reader knows
    # whether to widen acquisition or read the admission report.
    empty = EvidenceBundle(gap=gap, query=query_for_gap(gap), hits=[])
    dry = _synthesize([empty])
    assert "no supporting evidence retrieved" in next(
        row for row in dry.unresolved_gaps if row.get("gap_id") == gap.gap_id
    )["reason"]


def test_an_unsupported_catalyst_cannot_be_injected() -> None:
    """A catalyst the cited passage never names is not admitted with the reaction."""
    gap = _theobromine_gap()
    candidate = RagReactionCandidate(
        gap_id=gap.gap_id,
        name="R2",
        inputs=["theobromine"],
        outputs=["7-methylxanthine"],
        enzymes=["NdmZ"],  # nowhere in the passage below
        evidence={
            "text": "Theobromine is converted to 7-methylxanthine.",
            "chunk_id": "c1",
            "source_id": "PMID:B",
        },
        evidence_span="Theobromine is converted to 7-methylxanthine.",
        source_paper={"source_id": "PMID:B"},
    )
    accepted, report = admit_candidates(
        [candidate], gaps=[gap], seed_payload=_seed_payload()
    )
    assert accepted == []
    assert report.rejected[0]["reasons"][0].startswith("unsupported_catalyst_injection")

    # The same claim WITHOUT the invented catalyst is admitted.
    candidate.enzymes = []
    accepted, _ = admit_candidates(
        [candidate], gaps=[gap], seed_payload=_seed_payload()
    )
    assert [c.name for c in accepted] == ["R2"]


def test_a_participant_the_passage_never_states_is_refused() -> None:
    """"Evidence directly supports substrates and products", enforced literally."""
    gap = _theobromine_gap()
    candidate = RagReactionCandidate(
        gap_id=gap.gap_id,
        name="R2",
        inputs=["theobromine"],
        outputs=["7-methylxanthine", "xanthosine-5-phosphate"],
        evidence={
            "text": "Theobromine is converted to 7-methylxanthine.",
            "chunk_id": "c1",
            "source_id": "PMID:B",
        },
        evidence_span="Theobromine is converted to 7-methylxanthine.",
        source_paper={"source_id": "PMID:B"},
    )
    accepted, report = admit_candidates(
        [candidate], gaps=[gap], seed_payload=_seed_payload()
    )
    assert accepted == []
    reason = report.rejected[0]["reasons"][0]
    # The span states theobromine -> 7-methylxanthine; the claim adds a product
    # the sentence never mentions, so the parsed relation and the claim disagree.
    assert reason.startswith("evidence_relation_disagrees_with_claim")
    assert "xanthosine-5-phosphate" in reason


def test_a_downstream_chain_from_the_same_passage_is_admitted_as_one_extension() -> None:
    """A gap fill is often a CHAIN, and the later links only touch earlier ones.

    R2 closes the theobromine end; R3 consumes R2's product and R4 consumes R3's.
    Neither R3 nor R4 mentions theobromine, so a per-candidate rule would refuse
    them and deliver a pathway that stops one step after the gap. The per-gap
    fixpoint admits the chain — and records HOW each link qualified, so a chained
    admission is never mistaken for a direct one.
    """
    result = _synthesize(
        [
            _bundle(
                _FILLING_EQUATION,
                "name: R3 7-methylxanthine oxidation | "
                "7-methylxanthine + O2 -> 7-methyluric acid | enzyme: NdmC",
                "name: R4 ring opening | "
                "7-methyluric acid -> methyl-allantoin | enzyme: NdmD",
                _NON_GAP_EQUATION,
            )
        ]
    )

    names = _reaction_names(result.payload)
    assert {"R2 theobromine N7-demethylation", "R3 7-methylxanthine oxidation",
            "R4 ring opening"} <= set(names)
    assert "glucose phosphorylation" not in names

    by_name = {row["name"]: row for row in result.admission["accepted"]}
    assert by_name["R2 theobromine N7-demethylation"]["reasons"][0].startswith(
        "fills_named_gap_directly"
    )
    assert by_name["R3 7-methylxanthine oxidation"]["reasons"][0].startswith(
        "fills_named_gap_via_admitted_chain"
    )
    assert by_name["R4 ring opening"]["reasons"][0].startswith(
        "fills_named_gap_via_admitted_chain"
    )

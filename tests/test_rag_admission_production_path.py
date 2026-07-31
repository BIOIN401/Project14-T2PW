"""Eligibility + gap admission through the REAL orchestration path, end to end.

Everything else about admission is tested at the ``synthesize_with_report`` /
``admit_candidates`` seam. That is the right place for rule semantics, but it
cannot catch a wiring defect, and the wiring is where this subsystem has failed
before: the observed pathway/organism metadata the gate compares against is
produced by the **eligibility screen** on a raw candidate paper, has to survive
``select`` and ``ingest`` into a *chunk*, and only then reaches the gate. If the
screen is never called — which it was not, until this file existed — every paper
arrives with empty observations, i.e. "unknown", i.e. the permissive verdict, and
the organism rule silently stops rejecting anything while every unit test passes.

So this file drives ``maybe_run_rag``'s actual chain:

    acquire -> ELIGIBILITY SCREEN -> select -> (full text) -> ingest
            -> retrieve -> synthesize -> admission

The papers handed to it are **raw**: ``observed_pathways=[]``,
``observed_organisms=[]``, ``organism_match="unknown"``, exactly as
``acquire.search_candidates`` returns them (``_raw_paper`` asserts it). Nothing
here hand-populates an observation and nothing stubs ``screen_candidate`` /
``apply_decision`` — if an assertion below sees an observed organism, the real
screen put it there.

Only the two network calls and the two LLM calls are replaced. Screening,
selection, chunking, embedding, retrieval, transcription and admission all run
for real, offline: the memory vector backend and a blank embedding model force
the deterministic lexical embedder, so no endpoint is contacted.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

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

if "streamlit" not in sys.modules:
    _st = MagicMock(name="streamlit")
    _st.columns.side_effect = lambda spec=1, **__: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    _st.tabs.side_effect = lambda labels, **__: [MagicMock() for _ in labels]
    _st.form_submit_button.return_value = False
    _st.checkbox.return_value = False
    _st.session_state = MagicMock()
    _st.session_state.get.return_value = None
    _st.session_state.__contains__ = lambda _s, _k: False
    sys.modules["streamlit"] = _st

import t2pw.app.streamlit_app as app  # noqa: E402
from t2pw.rag import acquire, extract, select  # noqa: E402
from t2pw.pipeline.entity_identity import has_protein_external_identity  # noqa: E402
from t2pw.rag.acquire import CandidatePaper  # noqa: E402


REQUESTED_PATHWAY = "caffeine degradation"
REQUESTED_ORGANISM = "Pseudomonas putida"

#: The gap-filling step as a real paper states it: prose, no arrow. Chunking
#: joins a passage's lines into one window (``ingest._iter_windows``), so the
#: arrow parser sees a run-on line and transcribes nothing usable — which is why
#: real papers yield ~zero arrow candidates and the production transcription path
#: is the prose extractor.
_GAP_SENTENCE = (
    "NdmB catalyzes the N7-demethylation of theobromine to 7-methylxanthine "
    "and formaldehyde."
)
#: The same reaction, asserted about human cells instead.
_HUMAN_GAP_SENTENCE = (
    "In Homo sapiens cells, NdmB catalyzes the N7-demethylation of theobromine "
    "to 7-methylxanthine and formaldehyde."
)
#: Correct biology from the same paper, about a different pathway.
_OFF_PATHWAY_SENTENCE = "Hexokinase phosphorylates glucose to glucose-6-phosphate."

_MECHANISTIC_ABSTRACT = (
    "The {pathway} pathway of {organism} proceeds by sequential N-demethylation. "
    "We purified the enzyme NdmB and showed that it catalyzes the "
    "N7-demethylation of theobromine to 7-methylxanthine and formaldehyde. "
    "Kinetic analysis confirmed the reaction products and the substrate "
    "specificity of the enzyme."
)


def _full_text(*sentences: str) -> str:
    return "\n".join(
        [
            "Introduction",
            "Bacterial caffeine catabolism proceeds by sequential N-demethylation.",
            "Results",
            *sentences,
            "The reaction was confirmed by mass spectrometry of the products.",
            "Discussion",
            "These data extend the known route beyond theobromine.",
        ]
    )


def _seed_payload() -> dict:
    """Caffeine -> theobromine. Theobromine is the pathway's open end."""
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
                    "evidence": "NdmA removes the N3 methyl group of caffeine.",
                    "scope_membership": "core",
                }
            ]
        },
    }


def _pathway_context() -> dict:
    return {
        "pathway_name": REQUESTED_PATHWAY,
        "likely_organism": REQUESTED_ORGANISM,
        "key_compounds": ["caffeine", "theobromine", "7-methylxanthine"],
        "key_proteins": ["NdmA", "NdmB"],
        "gap_terms": ["theobromine"],
        "scope_clarity_score": 0.2,
    }


def _raw_paper(
    ident: str, *, title: str, abstract: str, body: tuple = (_GAP_SENTENCE,)
) -> CandidatePaper:
    """A candidate exactly as acquisition returns it: NOTHING observed yet.

    The observed fields are left at their empty defaults on purpose, and asserted
    empty here. Filling them in the fixture would test the fixture rather than the
    pipeline — the whole point of this file is that the real screen fills them.
    """
    paper = CandidatePaper(
        id=ident,
        source="europepmc",
        title=title,
        abstract=abstract,
        source_uri=f"https://example.org/{ident}",
        year="2025",
        full_text=_full_text(*body),
    )
    assert paper.observed_pathways == []
    assert paper.observed_organisms == []
    assert paper.organism_match == "unknown"
    assert paper.requested_pathway == ""
    assert paper.requested_organism == ""
    return paper


def _putida_paper(ident: str = "PMC_PUTIDA", body: tuple = (_GAP_SENTENCE,)):
    return _raw_paper(
        ident,
        title=(
            f"Downstream steps of caffeine degradation in Pseudomonas putida ({ident})"
        ),
        abstract=_MECHANISTIC_ABSTRACT.format(
            pathway="caffeine degradation", organism="Pseudomonas putida"
        ),
        body=body,
    )


def _human_paper(ident: str = "PMC_HUMAN"):
    return _raw_paper(
        ident,
        title=f"Downstream steps of caffeine degradation in Homo sapiens ({ident})",
        abstract=_MECHANISTIC_ABSTRACT.format(
            pathway="caffeine degradation", organism="Homo sapiens"
        ),
        body=(_HUMAN_GAP_SENTENCE,),
    )


def _junk_paper(ident: str = "PMC_JUNK"):
    """A clinical case report: no mechanism, no pathway, wrong organism."""
    return _raw_paper(
        ident,
        title=f"A case report of severe sepsis in a 62-year-old patient ({ident})",
        abstract=(
            "We describe a patient who presented with fever and hypotension. "
            "Blood cultures grew a Gram-negative organism. The patient was "
            "treated with meropenem and recovered after fourteen days in "
            "intensive care."
        ),
        body=("The patient recovered.",),
    )


def _validated_in_human_paper(ident: str = "PMC_MIXED"):
    """Screens as P. putida (its abstract is), but the RESULTS sentence is human.

    This is how a paper reaches the admission gate with a span-local organism the
    paper-level screen never saw — and it is the case the gate's own organism rule
    exists for, as opposed to the screen's.
    """
    return _raw_paper(
        ident,
        title=(
            f"Caffeine degradation in Pseudomonas putida, validated in cell lines "
            f"({ident})"
        ),
        abstract=_MECHANISTIC_ABSTRACT.format(
            pathway="caffeine degradation", organism="Pseudomonas putida"
        ),
        body=(_HUMAN_GAP_SENTENCE,),
    )


@pytest.fixture
def rag_env(monkeypatch, tmp_path):
    """Turn RAG on, offline: memory backend, blank embedding model, no network."""
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "memory")
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    # Blank model -> Embedder._embed_via_api returns immediately and the
    # deterministic lexical fallback is used. No endpoint is contacted.
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "")
    monkeypatch.setenv("RAG_EMBEDDING_BASE_URL", "")
    # The production transcription path. The LLM behind it is replaced below by
    # a deterministic stub; the quote, its verbatim validation against the chunk
    # and the span rules all run for real.
    monkeypatch.setenv("RAG_EXTRACT_REACTIONS", "true")
    monkeypatch.setenv("RAG_ELIGIBILITY_ENABLED", "true")
    monkeypatch.setenv("RAG_RETRIEVE_TOP_K", "4")
    monkeypatch.setenv("RAG_SELECT_MAX_PAPERS", "4")
    monkeypatch.setenv("RAG_ACQUIRE_MAX_PAPERS", "4")
    return tmp_path


def _offline_preprocess(text, **_kwargs):
    """Stand in for Stage 0's LLM call inside ``select`` (one per candidate)."""
    body = str(text or "")
    organism = "Homo sapiens" if "Homo sapiens" in body else REQUESTED_ORGANISM
    return {
        "pathway_name": REQUESTED_PATHWAY,
        "likely_organism": organism,
        "key_compounds": ["theobromine", "7-methylxanthine"],
        "key_proteins": ["NdmB"],
        "document_type": "primary_research",
        "scope_clarity_score": 0.8,
    }


def _claims_for(text: str):
    """What a well-behaved extractor reports for a passage, quoting it verbatim."""
    out = []
    if "theobromine to 7-methylxanthine" in text:
        quote = _HUMAN_GAP_SENTENCE if "Homo sapiens" in text else _GAP_SENTENCE
        out.append(
            {
                "name": "R2 theobromine N7-demethylation",
                "inputs": [{"name": "theobromine", "stoichiometry": None}],
                "outputs": [
                    {"name": "7-methylxanthine", "stoichiometry": None},
                    {"name": "formaldehyde", "stoichiometry": None},
                ],
                "enzymes": ["NdmB"],
                "reversible": False,
                "quote": quote,
            }
        )
    if "Hexokinase" in text:
        out.append(
            {
                "name": "glucose phosphorylation",
                "inputs": [{"name": "glucose", "stoichiometry": None}],
                "outputs": [{"name": "glucose-6-phosphate", "stoichiometry": None}],
                "enzymes": ["hexokinase"],
                "reversible": False,
                "quote": _OFF_PATHWAY_SENTENCE,
            }
        )
    return out


def _stub_prose_extractor(_chat_fn=None):
    """A deterministic, WELL-BEHAVED stand-in for the LLM prose extractor.

    It reports only what the passage in front of it says, quoting the exact
    sentence. The gate's handling of a badly-behaved extractor (paraphrased quote,
    quote stitched from two sentences, borrowed catalyst, reversed direction) is
    covered in tests/test_rag_admission_adversarial.py.
    """

    def _run(text: str):
        return _claims_for(str(text or ""))

    return _run


def _run(monkeypatch, papers, reports=None):
    """Drive the real chain; record which papers had full text fetched."""
    fetched: list = []
    monkeypatch.setattr(acquire, "search_candidates", lambda *_a, **_k: list(papers))

    def _fetch(paper, **_k):
        fetched.append(paper.id)
        return paper.full_text

    monkeypatch.setattr(acquire, "fetch_full_text", _fetch)
    monkeypatch.setattr(select, "preprocess", _offline_preprocess)
    monkeypatch.setattr(extract, "make_prose_extractor", _stub_prose_extractor)
    result = app.maybe_run_rag(
        pathway_context=_pathway_context(),
        user_flag=True,
        seed_payload=_seed_payload(),
        reports=reports if reports is not None else {},
        seed_text="Caffeine is demethylated to theobromine by NdmA.",
    )
    return result, fetched


def _rows(result, bucket):
    """Admission rows keyed by ``(source_id, name)``.

    Keyed by SOURCE as well as name because several papers — and the indexed seed
    itself — state the same reaction, so a name-only key silently keeps whichever
    row came last.
    """
    admission = getattr(result.synthesis, "admission", {}) or {}
    return {
        (row["evidence"]["source_id"], row["name"]): row
        for row in admission.get(bucket, [])
    }


# ---------------------------------------------------------------------------
# The screen runs, and its verdict is what reaches the gate.
# ---------------------------------------------------------------------------
def test_the_eligibility_screen_actually_runs_on_raw_acquired_papers(
    rag_env, monkeypatch
) -> None:
    """The wiring itself: screening happened, and it wrote the observations."""
    paper = _putida_paper()
    result, _fetched = _run(monkeypatch, [paper])

    assert result is not None and not result.error, getattr(result, "error", "")
    screening = result.diagnostics["eligibility"]
    assert screening["enabled"] is True
    assert screening["screened"] == 1
    assert screening["kept"] == 1
    assert screening["dropped"] == 0
    assert screening["by_outcome"] == {"eligible": 1}

    # The screen wrote the REQUEST and the OBSERVATIONS into their separate slots
    # on the candidate object it was given. The fixture supplied neither.
    assert paper.requested_organism == REQUESTED_ORGANISM
    assert paper.requested_pathway == REQUESTED_PATHWAY
    assert paper.observed_organisms == [REQUESTED_ORGANISM]
    assert paper.observed_pathways == [REQUESTED_PATHWAY]
    assert paper.organism_match == "match"


def test_a_matching_paper_reaches_admission_with_the_screens_observations(
    rag_env, monkeypatch
) -> None:
    result, _fetched = _run(monkeypatch, [_putida_paper()])

    assert result is not None and not result.error, getattr(result, "error", "")
    accepted = _rows(result, "accepted")
    key = ("PMC_PUTIDA", "R2 theobromine N7-demethylation")
    assert key in accepted, result.synthesis.admission["reason_counts"]
    row = accepted[key]

    # These values exist ONLY because the real screen produced them and ingest
    # carried them onto the chunk — the fixture's paper had empty lists.
    assert row["observed_organisms"] == [REQUESTED_ORGANISM]
    assert row["organism_match"] == "match"
    assert row["requested_organism"] == REQUESTED_ORGANISM
    assert row["evidence"]["span"] == _GAP_SENTENCE
    assert row["evidence"]["chunk_id"]
    assert row["chain"]["hop"] == 0

    delivered = {r["name"]: r for r in result.payload["processes"]["reactions"]}
    assert delivered["R2 theobromine N7-demethylation"]["scope_membership"] == "core"


def test_a_different_species_paper_is_rejected_by_the_screen_before_any_fetch(
    rag_env, monkeypatch
) -> None:
    """Rejected papers cost nothing: no full text, no chunks, no retrieval."""
    result, fetched = _run(monkeypatch, [_human_paper()])

    assert result is not None and not result.error, getattr(result, "error", "")
    screening = result.diagnostics["eligibility"]
    assert screening["dropped"] == 1
    assert screening["kept"] == 0
    assert "ineligible_organism" in screening["by_outcome"]
    assert fetched == [], "a rejected paper must not have its full text fetched"
    assert result.diagnostics["papers_selected"] == 0
    assert "R2 theobromine N7-demethylation" not in [
        r.get("name") for r in result.payload["processes"]["reactions"]
    ]


def test_a_junk_paper_is_never_ingested(rag_env, monkeypatch) -> None:
    """A clinical case report is dropped by the screen, so it is never indexed."""
    result, fetched = _run(monkeypatch, [_putida_paper(), _junk_paper()])

    assert result is not None and not result.error, getattr(result, "error", "")
    screening = result.diagnostics["eligibility"]
    assert screening["screened"] == 2
    assert screening["kept"] == 1
    assert screening["dropped"] == 1
    assert "PMC_JUNK" not in fetched
    assert "PMC_PUTIDA" in fetched

    # Nothing from the junk paper reached admission or the payload.
    admission = result.synthesis.admission
    sources = {row["evidence"]["source_id"] for row in admission["accepted"]}
    sources |= {row["evidence"]["source_id"] for row in admission["rejected"]}
    assert "PMC_JUNK" not in sources


def test_the_admission_organism_rule_still_fires_on_a_paper_the_screen_passed(
    rag_env, monkeypatch
) -> None:
    """Screen and gate ask different questions, at different resolutions.

    This paper's ABSTRACT is about P. putida, so the paper-level screen keeps it.
    Its results SENTENCE asserts the reaction in human cells, and that sentence is
    what the claim rests on — so the gate refuses it. Neither check is redundant.
    """
    result, fetched = _run(monkeypatch, [_validated_in_human_paper()])

    assert result is not None and not result.error, getattr(result, "error", "")
    assert result.diagnostics["eligibility"]["kept"] == 1
    assert "PMC_MIXED" in fetched

    rejected = _rows(result, "rejected")
    key = ("PMC_MIXED", "R2 theobromine N7-demethylation")
    assert key in rejected
    row = rejected[key]
    assert row["observed_organisms"] == ["Homo sapiens"]
    assert row["organism_match"] == "mismatch"
    assert row["reasons"][0].startswith("organism_mismatch")

    assert "R2 theobromine N7-demethylation" not in [
        r.get("name") for r in result.payload["processes"]["reactions"]
    ]
    assert result.diagnostics["candidates_accepted"] == 0
    assert "organism_mismatch" in result.diagnostics["admission_reason_counts"]


def test_an_off_pathway_reaction_from_an_eligible_paper_is_still_refused(
    rag_env, monkeypatch
) -> None:
    """Right organism, eligible paper, wrong chemistry: connectivity decides."""
    result, _fetched = _run(
        monkeypatch, [_putida_paper(body=(_GAP_SENTENCE, _OFF_PATHWAY_SENTENCE))]
    )

    assert result is not None and not result.error, getattr(result, "error", "")
    delivered = [r.get("name") for r in result.payload["processes"]["reactions"]]
    assert "R2 theobromine N7-demethylation" in delivered
    assert "glucose phosphorylation" not in delivered

    rejected = _rows(result, "rejected")
    key = ("PMC_PUTIDA", "glucose phosphorylation")
    assert key in rejected
    assert rejected[key]["reasons"][0].startswith(
        ("does_not_fill_named_gap", "adjacent_pathway_context_only")
    )

    # Acceptance is reported as a breakdown, not as a "precision" never computed.
    # ``direct`` counts CANDIDATES, and one claim retrieved for several gaps is
    # several candidates that later merge into one row — which is why the payload
    # has one reaction and this number is larger.
    breakdown = result.synthesis.admission["acceptance_breakdown"]
    assert breakdown["chained"] == 0
    assert breakdown["chained_by_hop"] == {}
    assert breakdown["direct"] >= 1
    assert (
        len([r for r in delivered if r == "R2 theobromine N7-demethylation"]) == 1
    )


def test_the_gap_is_reported_closed_only_because_the_payload_closes_it(
    rag_env, monkeypatch
) -> None:
    """End-to-end: theobromine stops being a dead end in the delivered payload."""
    result, _fetched = _run(monkeypatch, [_putida_paper()])

    reactions = result.payload["processes"]["reactions"]
    produced = {
        p if isinstance(p, str) else p.get("name")
        for r in reactions
        for p in r.get("outputs", [])
    }
    consumed = {
        p if isinstance(p, str) else p.get("name")
        for r in reactions
        for p in r.get("inputs", [])
    }
    assert "theobromine" in produced and "theobromine" in consumed

    unresolved = {row.get("label") for row in result.synthesis.unresolved_gaps}
    assert "theobromine" not in unresolved


# ---------------------------------------------------------------------------
# Protein identity: fail-closed in production, verified only on real evidence.
# ---------------------------------------------------------------------------
_IDENTITY_SENTENCE = (
    "NdmA (UniProt P12345) removes the N3 methyl group of caffeine."
)


def _unmapped_enzyme_gap():
    from t2pw.rag.retrieve import Gap

    return Gap(
        kind="unmapped_enzyme",
        label="NdmA",
        symbols=["NdmA"],
        source="gate",
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )


def test_production_identity_is_fail_closed_and_says_so(rag_env, monkeypatch) -> None:
    """The wired default verifies nothing, and reports why.

    ``maybe_run_rag`` wires ``identity.fail_closed_verifier`` because the RAG
    chain produces passages, not resolver candidates, and the real ladder judges a
    shipped accession against resolver evidence. So an unmapped-enzyme gap is
    never closed by a run of this chain — it keeps the PathBank Unknown-protein
    fallback — and ``identity_attempts`` records that the ladder was asked rather
    than skipped.
    """
    paper = _putida_paper(body=(_IDENTITY_SENTENCE,))
    # A Stage-3 gate error is what raises an unmapped-enzyme gap in a real run;
    # without one there is no identity question for the chain to answer.
    result, _fetched = _run(
        monkeypatch,
        [paper],
        reports={
            "gate": {
                "errors": [
                    {"reason": "Protein 'NdmA' is missing a UniProt identifier"}
                ]
            }
        },
    )

    assert result is not None and not result.error, getattr(result, "error", "")
    assert any(g.kind == "unmapped_enzyme" for g in result.gaps)
    attempts = result.diagnostics["identity_attempts"]
    assert attempts, "the identity ladder must be asked, not skipped"
    assert all(a["verified"] is False for a in attempts)
    assert {a["reason"] for a in attempts} == {"no_local_resolver_evidence"}

    # No protein in the delivered payload gained an external identity.
    for row in result.payload["entities"].get("proteins", []):
        assert not has_protein_external_identity(row), row

    assert result.diagnostics["typed_resolutions_proposed"] >= 1
    assert result.diagnostics["typed_resolutions_applied"] == 0

    # The identity outcomes are reported in three separate UNITS, and production
    # resolves exactly zero real identities.
    outcomes = result.diagnostics["identity_outcomes"]
    assert outcomes["proposals"]["verified"] == 0
    assert outcomes["proposals"]["annotation_only"] == 0
    assert outcomes["proposals"]["conflicting"] == 0
    assert outcomes["proposals"]["unverified_no_resolver"] >= 1
    assert outcomes["targets"]["verified"] == 0

    # Gaps are counted apart from proposals, and named for what the payload
    # actually contains: this stage inserts no Unknown sentinel, so these gaps are
    # awaiting one, not already carried by one.
    assert outcomes["unresolved_identity_gaps"] >= 1
    assert outcomes["unresolved_for_unknown_fallback"] == outcomes[
        "unresolved_identity_gaps"
    ]
    assert outcomes["unknown_fallback_present"] == 0


def test_a_wired_resolver_verifies_and_the_payload_gains_mapped_ids() -> None:
    """The other half: with real candidate evidence, the ladder confirms.

    Driven at the synthesis seam rather than through ``maybe_run_rag``, because
    wiring a resolver into the orchestrator is exactly what is NOT true today —
    asserting it through the app would claim a production capability that does not
    exist. What is asserted here is that the adapter and the ladder work, so the
    day a resolver is available the seam is ready.
    """
    from t2pw.rag.identity import build_identity_verifier
    from t2pw.rag.retrieve import EvidenceBundle, Retrieved, query_for_gap
    from t2pw.rag.store import Chunk
    from t2pw.rag.synthesize import synthesize_with_report

    gap = _unmapped_enzyme_gap()
    chunk = Chunk(
        id="c1",
        text=_IDENTITY_SENTENCE,
        source_id="PMC_PUTIDA",
        source_title="Downstream caffeine catabolism",
        source_type="paper",
        source_uri="https://example.org/PMC_PUTIDA",
        organism=REQUESTED_ORGANISM,
        section="results",
        observed_organisms=[REQUESTED_ORGANISM],
        observed_pathways=[REQUESTED_PATHWAY],
    )
    # The protein has to be an entity of the SYNTHESIZED payload for an identity
    # to be written onto it, and synthesis rebuilds entities from the reactions it
    # resolved -- so the target is the seed reaction's own catalyst.
    seed = _seed_payload()

    candidates = [
        {
            "uniprot": "P12345",
            "name": "NdmA",
            "protein_name": "NdmA",
            "gene": "NdmA",
            "organism": REQUESTED_ORGANISM,
            "species": REQUESTED_ORGANISM,
            "score": 0.95,
            "source": "uniprot",
            "entity_type": "protein",
        }
    ]
    attempts: list = []
    result = synthesize_with_report(
        seed,
        [
            EvidenceBundle(
                gap=gap,
                query=query_for_gap(gap),
                hits=[Retrieved(chunk=chunk, score=0.9)],
            )
        ],
        {"source": {"source_id": "PMID:SEED", "source_type": "paper"}},
        gaps=[gap],
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
        identity_verifier=build_identity_verifier(
            lambda _name: candidates, organism=REQUESTED_ORGANISM, record=attempts
        ),
    )

    assert attempts and attempts[0]["verified"] is True, attempts
    proteins = {r["name"]: r for r in result.payload["entities"]["proteins"]}
    assert proteins["NdmA"]["mapped_ids"] == {"uniprot": "P12345"}
    assert has_protein_external_identity(proteins["NdmA"])
    assert gap.gap_id not in {
        row.get("gap_id") for row in result.unresolved_gaps
    }

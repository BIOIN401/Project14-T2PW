"""C-035 — lineage emission at the RAG synthesis and admission stages.

Offline, deterministic, self-contained. The guarded ``openai`` stub and the
``sys.path`` preamble mirror ``tests/test_rag_synthesize.py``: importing
``t2pw.rag.retrieve`` (used here only to build the ``EvidenceBundle`` / ``Gap``
fixtures) pulls ``t2pw.rag.ingest`` -> ``acquire`` -> ``mapping`` -> ``llm.client``
-> ``openai`` at import time, and there is no ``conftest.py`` in this repo.

Everything NEW to this card (``admission_lineage_entry``) is imported INSIDE the
test that uses it, never at module scope. That is deliberate: the G9 base proof
runs this file against the base commit, and a module-level import of a symbol the
base does not have would turn every test in the file into a collection-time
``ImportError`` — which is not behavioural evidence of anything, and would also
stop the preservation test below from doing its job (it has to PASS on base).
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

from t2pw.pipeline.lineage import LINEAGE_KEY, read as read_lineage  # noqa: E402
from t2pw.rag.retrieve import EvidenceBundle, Gap  # noqa: E402
from t2pw.rag.store import Chunk, Retrieved  # noqa: E402
from t2pw.rag.synthesize import synthesize_with_report  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — paper A (the seed) and paper B (retrieved).
# ---------------------------------------------------------------------------
#: A record some EARLIER stage wrote. ``inferred`` is not a SOURCED origin and
#: ``derived`` is not direct/indirect support, so it validly carries no source.
SEED_ENTRY = {
    "stage": "gap_resolution",
    "origin": "inferred",
    "support": "derived",
    "paper_explicit": "not_evaluated",
    "reason": "added by the round-1 gap resolver",
    "review_required": False,
    "uncertainty": "",
    "sources": [],
}


def _seed_payload(*, lineage_on_r3: bool = False) -> dict:
    r3: dict = {
        "name": "R3 to theobromine",
        "inputs": ["7-methylxanthine"],
        "outputs": ["theobromine"],
    }
    if lineage_on_r3:
        r3[LINEAGE_KEY] = [dict(SEED_ENTRY)]
    return {
        "entities": {
            "species": [{"name": "Pseudomonas putida"}],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [
                {"name": "caffeine"},
                {"name": "paraxanthine"},
                {"name": "theobromine"},
            ],
            "proteins": [{"name": "NdmA"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "R1 caffeine demethylation",
                    "inputs": ["caffeine"],
                    "outputs": ["paraxanthine", "formaldehyde"],
                    "enzymes": [{"protein": "NdmA"}],
                },
                {
                    "name": "R2 paraxanthine demethylation",
                    "inputs": ["paraxanthine"],
                    "outputs": ["7-methylxanthine", "formaldehyde"],
                },
                r3,
            ]
        },
    }


def _seed_context() -> dict:
    return {
        "text": "caffeine degradation pathway in Pseudomonas putida",
        "source": {
            "source_id": "PMID:0001",
            "source_title": "Caffeine degradation (paper A)",
            "source_type": "paper",
            "source_uri": "https://example.org/paperA",
            "organism": "Pseudomonas putida",
        },
    }


def _chunk_paper_b() -> Chunk:
    text = "\n".join(
        [
            "name: R4 theobromine demethylation | "
            "theobromine + O2 -> 3-methylxanthine + formaldehyde | enzyme: NdmB",
            "name: R5 3-methylxanthine oxidation | "
            "3-methylxanthine + O2 -> 3-methyluric acid | enzyme: NdmC",
            "name: R6 ring opening | "
            "3-methyluric acid -> methyl-allantoin | enzyme: NdmD",
        ]
    )
    return Chunk(
        id="chunkB1",
        text=text,
        source_id="PMID:0002",
        source_title="Downstream caffeine catabolism (paper B)",
        source_type="paper",
        source_uri="https://example.org/paperB",
        organism="Pseudomonas putida",
        section="results",
    )


def _bundle(label: str = "theobromine") -> EvidenceBundle:
    gap = Gap(
        kind="dangling_reaction",
        label=label,
        detail=f"{label} is consumed by no reaction",
        symbols=[label],
        source="qa_graph",
    )
    return EvidenceBundle(
        gap=gap,
        query=f"find the reaction that consumes {label}",
        hits=[Retrieved(chunk=_chunk_paper_b(), score=0.91)],
    )


def _run(**kwargs):
    return synthesize_with_report(
        _seed_payload(**kwargs), [_bundle()], _seed_context()
    )


def _rxn(payload: dict, name: str) -> dict:
    return next(r for r in payload["processes"]["reactions"] if r["name"] == name)


def _entries(row: dict, stage: str = "") -> list:
    entries = list(read_lineage(row).canonical())
    return [e for e in entries if not stage or e.stage == stage]


def _strip(value):
    """``value`` with every ``LINEAGE_KEY`` removed, recursively."""
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k != LINEAGE_KEY}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# 1 — the admission stage's record. NEW ACCEPTANCE.
# ---------------------------------------------------------------------------
def test_admitted_rag_reaction_records_the_gap_it_was_admitted_against():
    """NEW ACCEPTANCE — no pre-existing behaviour is claimed.

    R-004: the verdict is keyed on ``(gap_id, claim_identity())`` while at least
    one consumer compares on a key that omits ``gap_id``, yielding false
    "reintroduction" findings. The gap has to be ON THE ROW for that dispute to
    be decidable, which is what this asserts.
    """
    r4 = _rxn(_run().payload, "R4 theobromine demethylation")

    admission = _entries(r4, "rag_admission")
    assert len(admission) == 1, admission
    entry = admission[0]
    assert entry.origin == "rag_literature"
    assert entry.support == "direct"
    # Never ``not_explicit``: this gate compares against the REQUESTED pathway and
    # organism and never asks what the supplied paper said, so it may not answer.
    assert entry.paper_explicit == "not_evaluated"
    # The gate ADMITTED it; a lineage that demanded review would contradict the
    # verdict it is recording.
    assert entry.review_required is False
    # THE point of the record: which gap this claim was admitted against.
    assert _bundle().gap.gap_id in entry.reason
    assert "fills_named_gap" in entry.reason
    assert r4["rag_provenance"]["gap_id"] == _bundle().gap.gap_id
    # Backed by a named record, not an anonymous assertion.
    assert [s.source_id for s in entry.sources] == ["PMID:0002"]
    assert entry.sources[0].locator == "chunkB1"


# ---------------------------------------------------------------------------
# 2 — the retrieval stage's record. NEW ACCEPTANCE.
# ---------------------------------------------------------------------------
def test_rag_reaction_records_a_retrieval_entry_naming_its_source_paper():
    """NEW ACCEPTANCE. R-003: a RAG import with no carrier is indistinguishable
    from paper-explicit content. After this, it says which stage imported it."""
    r4 = _rxn(_run().payload, "R4 theobromine demethylation")

    retrieval = _entries(r4, "rag_retrieval")
    assert len(retrieval) == 1, retrieval
    entry = retrieval[0]
    assert entry.origin == "rag_literature"
    assert entry.paper_explicit == "not_evaluated"
    assert _bundle().gap.gap_id in entry.reason
    assert [s.source_id for s in entry.sources] == ["PMID:0002"]
    # Both stages are on the row, and in pipeline order.
    assert [e.stage for e in _entries(r4)] == ["rag_retrieval", "rag_admission"]


# ---------------------------------------------------------------------------
# 3 — what this stage must NOT claim. NEW ACCEPTANCE.
# ---------------------------------------------------------------------------
def test_seed_reaction_gets_no_lineage_authored_by_this_stage():
    """NEW ACCEPTANCE.

    Synthesis did not introduce a seed reaction and cannot tell how it arose — in
    research mode the "seed" of round N+1 is round N's payload, so a row here may
    already be a RAG import or a gap fill. Stamping ``paper_stated`` would be
    inventing an origin, so nothing is written at all.
    """
    payload = _run().payload
    r1 = _rxn(payload, "R1 caffeine demethylation")

    assert LINEAGE_KEY not in r1
    # ...and the seed's contextual scaffolding is untouched, byte for byte.
    assert payload["entities"]["species"] == [{"name": "Pseudomonas putida"}]
    assert payload["entities"]["subcellular_locations"] == [{"name": "cytosol"}]


def test_entity_already_in_the_seed_is_not_relabelled_as_a_rag_import():
    """NEW ACCEPTANCE — the honesty guard on entity rows.

    ``_build_entities`` runs over seed and RAG reactions alike. ``theobromine`` is
    the seed pathway's own compound that paper B merely also mentions; claiming
    RAG literature introduced it would be inventing an origin. ``methyl-allantoin``
    and ``NdmB`` exist only because of the import, so they are attributed.
    """
    payload = _run().payload
    compounds = {c["name"]: c for c in payload["entities"]["compounds"]}
    proteins = {p["name"]: p for p in payload["entities"]["proteins"]}

    assert LINEAGE_KEY not in compounds["theobromine"]
    assert LINEAGE_KEY not in compounds["caffeine"]
    assert LINEAGE_KEY not in proteins["NdmA"]

    rag_only = _entries(compounds["methyl-allantoin"], "rag_retrieval")
    assert len(rag_only) == 1
    # ``derived``, not ``direct``: the passage states a REACTION; the entity row is
    # the deterministic projection of it onto participants.
    assert rag_only[0].support == "derived"
    assert [s.source_id for s in rag_only[0].sources] == ["PMID:0002"]
    assert _entries(proteins["NdmB"], "rag_retrieval")


# ---------------------------------------------------------------------------
# 4 — the append-only hole this closes. G9 REGRESSION.
# ---------------------------------------------------------------------------
def test_inbound_seed_lineage_survives_synthesis():
    """G9 REGRESSION — fails behaviourally on base ``b5bbf08``.

    ``_reaction_row`` builds every row from scratch off ``_Reaction``, which had no
    lineage field, so an attribution an earlier stage had written onto a seed row
    was silently erased at every synthesis. On base the key is simply absent from
    the emitted row (an ``assert``/``KeyError`` failure, not an import error —
    ``t2pw.pipeline.lineage`` exists at base and this test imports only symbols
    base already has). A lineage is append-only; erasing one makes "which stage
    introduced this?" permanently unanswerable.
    """
    r3 = _rxn(_run(lineage_on_r3=True).payload, "R3 to theobromine")

    assert LINEAGE_KEY in r3, "the inbound attribution was erased by synthesis"
    stages = [(e.stage, e.origin) for e in _entries(r3)]
    assert ("gap_resolution", "inferred") in stages
    assert r3[LINEAGE_KEY][0]["reason"] == "added by the round-1 gap resolver"


def test_malformed_inbound_lineage_is_dropped_not_raised():
    """NEW ACCEPTANCE — fail-safe, not fail-open-with-a-crash.

    Synthesis is not the lineage validator (``rag/graph_delta.py`` is, and it
    REPORTS a malformed lineage as a violation). A payload that synthesized before
    must not start raising because some earlier stage wrote a bad record — that
    would be this instrument changing what the pipeline produces.
    """
    seed = _seed_payload()
    seed["processes"]["reactions"][2][LINEAGE_KEY] = "not-a-list"

    payload = synthesize_with_report(seed, [_bundle()], _seed_context()).payload

    r3 = _rxn(payload, "R3 to theobromine")
    assert LINEAGE_KEY not in r3
    assert len(payload["processes"]["reactions"]) == 6


# ---------------------------------------------------------------------------
# 5 — recording provenance changed nothing. PRESERVATION (passes on base AND tip).
# ---------------------------------------------------------------------------
def test_synthesis_decisions_and_outputs_are_unchanged_by_lineage():
    """PRESERVATION — passes on base ``b5bbf08`` and at the tip.

    Every value asserted here is a DECISION or an OUTPUT of the instrumented
    stage: which reactions were admitted, what chemistry each carries, which gap
    each was admitted for, the scope label the gate wrote, and the accepted /
    rejected counts. Stripping ``LINEAGE_KEY`` is a no-op on base and removes only
    the new key at the tip, so agreement here is evidence the instrument is inert.
    """
    result = _run()
    payload = _strip(result.payload)

    assert sorted(r["name"] for r in payload["processes"]["reactions"]) == [
        "R1 caffeine demethylation",
        "R2 paraxanthine demethylation",
        "R3 to theobromine",
        "R4 theobromine demethylation",
        "R5 3-methylxanthine oxidation",
        "R6 ring opening",
    ]
    r4 = _rxn(payload, "R4 theobromine demethylation")
    assert r4["inputs"] == ["theobromine", "O2"]
    assert r4["outputs"] == ["3-methylxanthine", "formaldehyde"]
    assert [e["entity"] for e in r4["enzymes"]] == ["NdmB"]
    assert r4["source_refs"] == ["PMID:0002"]
    assert r4["scope_membership"] == "core"
    assert r4["rag_provenance"]["gap_id"] == _bundle().gap.gap_id
    # The gate's own bookkeeping is untouched.
    assert result.admission["counts"] == {
        "accepted": 3,
        "rejected": 0,
        "considered": 3,
    }
    assert result.contract_report["ok"] is True
    # No lineage anywhere it was not earned: scaffolding is copied, not authored.
    for bucket in ("species", "subcellular_locations"):
        for row in result.payload["entities"][bucket]:
            assert LINEAGE_KEY not in row
    # Enzyme ACTOR rows are sub-rows of a reaction, not independently attributable
    # elements; the protein's own entity row is where its attribution lives.
    assert LINEAGE_KEY not in _rxn(result.payload, "R4 theobromine demethylation")[
        "enzymes"
    ][0]


# ---------------------------------------------------------------------------
# 6 — unit-level honesty rules on the gate's builder. NEW ACCEPTANCE.
# ---------------------------------------------------------------------------
def _candidate(**kwargs):
    from t2pw.rag.admission import RagReactionCandidate

    base = dict(
        gap_id="g1",
        name="A -> B",
        inputs=["A"],
        outputs=["B"],
        evidence_span="A is converted to B.",
        source_paper={"source_id": "PMID:9", "source_type": "paper"},
        evidence={"chunk_id": "c1", "source_id": "PMID:9"},
        status="accepted",
        reasons=["fills_named_gap_directly: via A"],
    )
    base.update(kwargs)
    return RagReactionCandidate(**base)


def test_rejected_candidate_gets_no_admission_entry():
    """NEW ACCEPTANCE — only an ADMITTED claim is attributed to the gate."""
    from t2pw.rag.admission import admission_lineage_entry

    assert admission_lineage_entry(_candidate(status="rejected")) is None
    assert admission_lineage_entry(_candidate(status="")) is None
    assert admission_lineage_entry(_candidate()) is not None


def test_unciteable_admitted_claim_gets_no_entry_rather_than_an_anonymous_one():
    """NEW ACCEPTANCE — a row asserting provenance it does not have is worse than
    a row with none. ``rag_literature`` must identify its supporting record."""
    from t2pw.rag.admission import admission_lineage_entry

    nameless = _candidate(source_paper={}, evidence={"chunk_id": "c1"})
    assert admission_lineage_entry(nameless) is None

    # A URI alone is still a citable pointer, so that one IS recorded.
    by_uri = _candidate(
        source_paper={}, evidence={"chunk_id": "c1", "source_uri": "https://x/y"}
    )
    entry = admission_lineage_entry(by_uri)
    assert entry is not None and entry.sources[0].uri == "https://x/y"


def test_gate_observations_ride_in_uncertainty_and_never_flip_review_required():
    """NEW ACCEPTANCE.

    A non-exact organism match and a chained (hop > 0) admission are real nuance
    the gate observed, and they are written down — but in ``uncertainty``, which
    gates nothing. Promoting them to ``review_required`` would be this record
    contradicting the verdict it exists to describe.
    """
    from t2pw.rag.admission import admission_lineage_entry

    entry = admission_lineage_entry(
        _candidate(
            organism_match="unknown",
            requested_organism="Escherichia coli",
            chain={"hop": 2, "gap_id": "g1"},
        )
    )
    assert entry is not None
    assert entry.review_required is False
    assert "organism match is 'unknown'" in entry.uncertainty
    assert "chain hop 2" in entry.uncertainty


def test_merged_claims_keep_every_admission_attribution():
    """NEW ACCEPTANCE — the union rule in ``_merge_into``.

    One canonical claim retrieved for two gaps merges into ONE row that genuinely
    was admitted for both. ``gap_ids`` already unions for exactly this reason;
    lineage must too, or the second admission becomes unattributable.
    """
    from t2pw.rag.synthesize import _Reaction, _merge_into

    a = _Reaction(name="x", inputs=[], outputs=[], lineage=[{"n": 1}, {"n": 2}])
    b = _Reaction(name="x", inputs=[], outputs=[], lineage=[{"n": 2}, {"n": 3}])

    _merge_into(a, b)

    assert a.lineage == [{"n": 1}, {"n": 2}, {"n": 3}]

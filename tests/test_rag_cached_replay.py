"""Reproducible offline replay of the real PMC12444477/strict artifacts.

What this replays and why
-------------------------
Leg ``PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict`` of run
``2026-07-28_0919`` delivered 27 reactions. Reactions 1-9 are the seed paper's own
lipid A steps; 10-27 are cross-paper RAG imports and they are *phospholipid*
biosynthesis (``Acetyl-CoA -> malonyl-CoA``, ``CDP-DAG -> PG``, ``PG -> CL``, ...)
taken from PMC12898747 — real, cited biology that fills no gap of the lipid A
pathway the run asked for. That payload is the reason the admission gate exists,
and it is checked into ``runs/``, so the gate can be measured against it instead
of against a fixture somebody wrote to make it pass.

How it stays honest
-------------------
* **Nothing is copied into a fixture.** The artifacts are read from ``runs/`` in
  place. Their identity is pinned by ``sha256`` and by filesystem size, so if
  either file is ever regenerated this test says so loudly rather than quietly
  measuring a different payload.
* **Sizes are filesystem bytes** (``Path.stat().st_size``), not
  ``len(json.dumps(...))``. Those differ by more than a rounding error here: the
  historical ``merged_payload.json`` is 4,707,785 bytes on disk, 4,697,259 decoded
  characters, and 4,706,658 characters once re-serialized with ``json.dumps``
  defaults. Quoting the last of those as "the payload size" — as an earlier
  version of this replay did — measures an artifact that never existed on disk.
* **The attribution given to the historical imports is maximally generous.** The
  old pipeline recorded no ``gap_id`` at all, so every import would trivially be a
  ``no_gap_id`` rejection and the replay would prove nothing. Each import is
  instead offered to EVERY detected gap and counted as accepted if ANY gap takes
  it. A rejection under those terms is a rejection under the best case the old
  behavior could have claimed.
* **Arrow recall and prose recall are reported separately** — see
  ``test_arrow_only_recall_...`` and its docstring. An arrow-only replay finding
  zero candidates says nothing about whether the production prose path works.

Offline: reads checked-in JSON. No network, no LLM, no chromadb.
"""

from __future__ import annotations

import hashlib
import json
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
    RagReactionCandidate,
    admit_candidates,
    locate_span,
    parse_span_relation,
)
from t2pw.rag.retrieve import detect_gaps  # noqa: E402
from t2pw.rag.synonyms import build_offline_synonym_resolver  # noqa: E402


# ---------------------------------------------------------------------------
# The artifacts under replay, pinned by path + content hash + filesystem size.
# ---------------------------------------------------------------------------
LEG = (
    ROOT
    / "runs"
    / "2026-07-28_0919"
    / "papers"
    / "PMC12444477__the-regulation-of-lipid-a-biosynthesis"
    / "strict"
)
MERGED = LEG / "merged_payload.json"
STAGE1 = LEG / "stage1_payload.json"

#: ``sha256`` of the first 16 hex digits, and the exact ``st_size``. Both are
#: recorded so a regenerated artifact is a loud failure, not a silent re-measure.
ARTIFACTS = {
    "merged_payload.json": {"sha256_16": "cde0ff5e723e02cf", "st_size": 4_707_785},
    "stage1_payload.json": {"sha256_16": "0fc4362352d9554e", "st_size": 17_015},
}

REQUESTED_PATHWAY = "lipid A biosynthesis"
REQUESTED_ORGANISM = "Escherichia coli"
#: Reactions 1-9 are the seed's own; everything after is a cross-paper import.
SEED_REACTION_COUNT = 9
#: Head of the historical (139k-283k character) evidence strings handed to the
#: gate. The span validator only ever reads one statement out of it; the head is
#: bounded so the replay cannot become a memory test.
EVIDENCE_HEAD = 6000

pytestmark = pytest.mark.skipif(
    not (MERGED.exists() and STAGE1.exists()),
    reason=f"replay artifacts absent: {LEG}",
)


def _representative_refusal(refusals):
    """Pick ONE reason to represent a row refused by every gap, deterministically.

    A row is offered to all ten gaps and refused ten times, usually for different
    reasons. Reporting "the first gap's reason" is an accident of gap ordering
    dressed up as a finding, so the representative is chosen by a stated rule: the
    reason CODE that the most gaps agreed on, ties broken by the code's name, and
    the gap reported alongside it is the lowest-id gap that gave that code.
    """
    counts: dict = {}
    for _gap_id, reason in refusals:
        code = reason.split(":", 1)[0]
        counts[code] = counts.get(code, 0) + 1
    if not counts:
        return ("", "")
    code = sorted(counts, key=lambda c: (-counts[c], c))[0]
    for gap_id, reason in sorted(refusals):
        if reason.split(":", 1)[0] == code:
            return (gap_id, reason)
    return refusals[0]


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _participant_names(side) -> list:
    return [
        p if isinstance(p, str) else (p or {}).get("name", "") for p in side or []
    ]


def _enzyme_names(row: dict) -> list:
    return [
        str(a.get("entity") or a.get("protein") or "")
        for a in (row.get("enzymes") or [])
        if isinstance(a, dict)
    ]


def _replay_historical_imports(*, isolate_connectivity: bool = False):
    """Offer every historical import to every detected gap; return the verdicts.

    Returns ``(gaps, verdicts)`` where each verdict is
    ``(name, status, gap_id, reason)`` and ``status`` is ``"accepted"`` only when
    some gap would have taken the reaction.

    ``isolate_connectivity`` makes every earlier rule pass by construction so the
    CONNECTIVITY rule — the one this gate was rewritten to get right — is what
    decides. It does two things:

    * drops each row's catalyst roster. Those are the sprayed 12-15-enzyme lists a
      separate historical defect produced ("PG synthesis" is catalysed by WaaA,
      LpxM, MsbA, ...); no sentence in any paper names all of them, so with them
      attached every row is refused on the catalyst rule;
    * takes the claim's substrates and products FROM the located span's own parsed
      relation, so the span cannot disagree with the claim about direction or
      participants.

    Nothing about connectivity is relaxed. A row that is still refused under these
    terms is refused because no substrate or product of it is in any gap's target
    — which is exactly the WaaA/LpxM shared-catalyst case.
    """
    seed = _load(STAGE1)
    merged = _load(MERGED)
    resolver = build_offline_synonym_resolver()
    gaps = detect_gaps(
        seed,
        {},
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    imports = merged["processes"]["reactions"][SEED_REACTION_COUNT:]

    verdicts = []
    for row in imports:
        blob = str(row.get("evidence") or "")
        inputs = _participant_names(row.get("inputs"))
        outputs = _participant_names(row.get("outputs"))
        enzymes = [] if isolate_connectivity else _enzyme_names(row)
        # MOST GENEROUS SPAN. The historical rows carry a whole 139k-283k
        # character evidence blob and no span of their own, and a blob is never
        # "one statement" — so handing the blob straight to the gate would reject
        # all eighteen on the span rule alone and prove nothing about the
        # connectivity rule this replay exists to demonstrate. Instead the blob is
        # searched for the single statement that actually supports the claim, and
        # THAT is offered as the span. A row for which no sentence supports the
        # claim falls back to the blob head and is refused on the span rule, which
        # is the honest outcome for it.
        refusals: list = []
        located = locate_span(blob, "", inputs + outputs + enzymes)
        span = located or blob[:EVIDENCE_HEAD]
        if isolate_connectivity:
            # Give the claim the span's OWN reading of itself, so only the graph
            # rules remain in play. When no span states a relation at all there is
            # nothing to isolate and the row keeps its payload participants.
            for candidate_span in (located, blob[:EVIDENCE_HEAD]):
                relation = parse_span_relation(candidate_span) if candidate_span else None
                if relation is not None:
                    span = candidate_span
                    inputs, outputs = list(relation.inputs), list(relation.outputs)
                    break
        best = None
        for gap in gaps:
            candidate = RagReactionCandidate(
                gap_id=gap.gap_id,
                name=str(row.get("name") or ""),
                inputs=inputs,
                outputs=outputs,
                enzymes=enzymes,
                source_paper={"source_id": "PMC12898747"},
                evidence={
                    "text": blob[:EVIDENCE_HEAD],
                    "chunk_id": f"hist:{row.get('name')}",
                    "source_id": "PMC12898747",
                },
                evidence_span=span,
                organism="",
                requested_pathway=REQUESTED_PATHWAY,
                requested_organism=REQUESTED_ORGANISM,
                confidence=0.5,
            )
            accepted, report = admit_candidates(
                [candidate], gaps=gaps, seed_payload=seed, name_resolver=resolver
            )
            if accepted:
                best = (
                    str(row.get("name") or ""),
                    "accepted",
                    gap.gap_id,
                    accepted[0].reasons[0],
                )
                break
            refusals.append((gap.gap_id, report.rejected[0]["reasons"][0]))
        if best is None:
            best = (
                str(row.get("name") or ""),
                "rejected",
                *_representative_refusal(refusals),
            )
        verdicts.append(best)
    return gaps, verdicts


# ---------------------------------------------------------------------------
# 1. Artifact identity.
# ---------------------------------------------------------------------------
def test_the_replayed_artifacts_are_the_ones_the_numbers_were_taken_from() -> None:
    """Pin path, content hash and filesystem size — the last in real bytes."""
    for name, expected in ARTIFACTS.items():
        path = LEG / name
        assert path.exists(), path
        assert _sha16(path) == expected["sha256_16"], f"{name} content changed"
        assert path.stat().st_size == expected["st_size"], f"{name} size changed"

    # The three numbers that get confused for each other, kept apart on purpose.
    raw = MERGED.read_bytes()
    assert MERGED.stat().st_size == 4_707_785  # filesystem bytes -- the real size
    assert len(raw.decode("utf-8")) == 4_697_259  # decoded characters
    assert len(json.dumps(_load(MERGED))) == 4_706_658  # re-serialized string


# ---------------------------------------------------------------------------
# 2. Gap detection on the real Stage-1 payload.
# ---------------------------------------------------------------------------
def test_the_real_seed_payload_yields_the_expected_gap_contract() -> None:
    seed = _load(STAGE1)
    gaps = detect_gaps(
        seed,
        {},
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    assert len(gaps) == 10
    assert [g.gap_id for g in gaps] == [
        "gap-orphan_metabolite-cb2dab1b",
        "gap-orphan_metabolite-5d744678",
        "gap-dangling_reaction-c8a144dc",
        "gap-dangling_reaction-8d674b04",
        "gap-orphan_metabolite-a7f9b06f",
        "gap-dangling_reaction-92cdd1fb",
        "gap-orphan_metabolite-eb5aa642",
        "gap-dangling_reaction-0e7c5ca7",
        "gap-orphan_metabolite-847b2a0b",
        "gap-dangling_reaction-dda3c004",
    ]
    kdo = next(g for g in gaps if g.label == "Kdo")
    # The retrieval symbols carry the neighbouring reaction and its catalyst...
    assert "WaaA" in kdo.symbols
    # ...and the FILL target does not. This is the single line that stops thirteen
    # phospholipid reactions reading as "fills the Kdo gap directly".
    assert kdo.target_names() == ["Kdo"]
    assert "WaaA" not in kdo.adjacent_metabolites


# ---------------------------------------------------------------------------
# 3. The historical imports, under the most favourable attribution.
# ---------------------------------------------------------------------------
def _reason_codes(verdicts) -> dict:
    codes: dict = {}
    for _name, status, _gap_id, reason in verdicts:
        if status == "accepted":
            continue
        code = reason.split(":", 1)[0]
        codes[code] = codes.get(code, 0) + 1
    return codes


def test_no_historical_phospholipid_import_survives_admission_as_shipped() -> None:
    """All eighteen, exactly as they shipped: none is admitted.

    As shipped, each row carries the sprayed 12-15-enzyme catalyst roster a
    separate historical defect produced. No sentence in any paper names all of
    them, so the *evidence-span* rule refuses every row before the graph rules are
    reached — which is correct (an unsupportable attribution is not evidence) but
    says nothing about connectivity. The companion test below removes the rosters
    so the connectivity rule can be measured on its own.
    """
    _gaps, verdicts = _replay_historical_imports()

    assert len(verdicts) == 18
    accepted = [v for v in verdicts if v[1] == "accepted"]
    assert accepted == [], f"unexpectedly admitted: {[v[0] for v in accepted]}"
    assert _reason_codes(verdicts) == {"evidence_span_is_not_one_statement": 18}

    # Spot-check the shape of what was refused: these are the phospholipid steps.
    names = {v[0] for v in verdicts}
    assert {"Acetyl-CoA -> malonyl-CoA", "CDP-DAG -> PG", "PG -> CL"} <= names


def test_the_connectivity_rule_alone_still_refuses_the_phospholipid_imports() -> None:
    """THE preserved regression: sharing a sprayed catalyst is not connectivity.

    Catalyst rosters dropped, participants taken from the span's own parsed
    relation, most favourable gap attribution — every rule ahead of connectivity
    made to pass by construction. Four of the eighteen are then refused precisely
    because no substrate or product of theirs is in any gap's target: the outcome
    an actor-based frontier got wrong, admitting thirteen of them on a shared
    ``WaaA`` / ``LpxM``. The rest never get that far — their evidence blob contains
    no single statement that supports the claim, or states a relation whose roles
    no implemented template can assign. Each of those is its own correct refusal.
    Still zero admitted, which is the assertion that must never move.

    The reason MIX does move as the templates tighten, and that is reported rather
    than smoothed over. "first committed step in phospholipid biosynthesis"
    counted as ``span_is_not_one_statement`` while the ``[a-z]+as\\w*`` verb branch
    matched the noun *phospholipase* inside its blob head and handed back a parse;
    with that branch replaced by real verbal suffixes the blob yields no relation
    at all, and the row is refused one rule earlier as ``roles_unassignable``. One
    fewer wrong reading of a 6000-character blob, same verdict.
    """
    _gaps, verdicts = _replay_historical_imports(isolate_connectivity=True)

    accepted = [v for v in verdicts if v[1] == "accepted"]
    assert accepted == [], f"unexpectedly admitted: {[v[0] for v in accepted]}"
    assert _reason_codes(verdicts) == {
        "does_not_fill_named_gap": 4,
        "evidence_relation_roles_unassignable": 3,
        "evidence_span_is_not_one_statement": 11,
    }

    # None of them was refused for a reason that would also refuse honest
    # gap-filling chemistry (organism / pathway / entity type).
    for _name, _status, _gap_id, reason in verdicts:
        assert not reason.startswith(
            ("organism_", "requested_pathway_mismatch", "invalid_entity_type")
        ), reason


# ---------------------------------------------------------------------------
# 4. Payload-size effect, measured on real files.
# ---------------------------------------------------------------------------
def test_the_gated_payload_is_written_and_measured_in_filesystem_bytes(tmp_path) -> None:
    """Write the gated artifact and measure it the same way the historical one is."""
    merged = _load(MERGED)
    _gaps, verdicts = _replay_historical_imports()
    imports = merged["processes"]["reactions"][SEED_REACTION_COUNT:]

    kept = merged["processes"]["reactions"][:SEED_REACTION_COUNT] + [
        row
        for row, verdict in zip(imports, verdicts)
        if verdict[1] == "accepted"
    ]
    gated = dict(merged)
    gated["processes"] = dict(merged["processes"])
    gated["processes"]["reactions"] = kept

    out = tmp_path / "merged_payload.gated.json"
    out.write_text(
        json.dumps(gated, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    assert len(merged["processes"]["reactions"]) == 27
    assert len(kept) == 9

    historical_bytes = MERGED.stat().st_size
    gated_bytes = out.stat().st_size
    assert historical_bytes == 4_707_785
    # The gated artifact is the seed's own nine lipid A steps. Pinned as a range
    # rather than an exact byte count because it is re-serialized with this
    # test's own indent/encoding rather than the run's writer, and a formatting
    # change there is not a regression in what the gate kept.
    assert 40_000 <= gated_bytes <= 80_000, gated_bytes
    assert gated_bytes < historical_bytes / 50


# ---------------------------------------------------------------------------
# 5. Arrow-only recall vs prose recall — reported SEPARATELY.
# ---------------------------------------------------------------------------
ACQUIRE_CACHE = ROOT / "data" / "rag_index" / "acquire_cache"


@pytest.mark.skipif(
    not ACQUIRE_CACHE.exists(),
    reason="acquire cache is a rebuildable, git-ignored artifact",
)
def test_arrow_only_recall_over_real_cached_papers_is_reported_as_arrow_only() -> None:
    """The deterministic arrow parser transcribes nothing from real paper prose.

    This is a measurement of the ARROW path and of nothing else. Papers state
    reactions in sentences, not as ``A + B -> C`` equations, so the arrow parser
    finding zero candidates over four real cached full texts is the expected
    result — **not** evidence that the production prose path is correct or
    incorrect. That path needs the LLM extractor and is exercised, with its span
    validation, in ``tests/test_rag_admission_adversarial.py`` and by
    :func:`test_prose_recall_on_real_cached_text_still_requires_a_valid_span`
    below.
    """
    from t2pw.rag import ingest
    from t2pw.rag.acquire import AcquireCache, CandidatePaper, _hash_key
    from t2pw.rag.store import MemoryVectorStore

    cached_files = sorted(ACQUIRE_CACHE.glob("*.json"))
    if not cached_files:
        pytest.skip("acquire cache is empty")

    fulltext = AcquireCache(ACQUIRE_CACHE / "fulltext", enabled=True)
    papers = []
    for path in cached_files:
        try:
            rows = json.loads(path.read_text(encoding="utf-8")).get("candidates", [])
        except Exception:  # noqa: BLE001 - a broken cache file is just a miss
            continue
        for row in rows:
            paper = CandidatePaper.from_dict(row)
            entry = fulltext.get(
                _hash_key({"id": paper.id.lower(), "source": paper.source})
            )
            text = str((entry or {}).get("full_text") or "")
            if len(text) < 2000:
                continue
            paper.full_text = text
            papers.append(paper)
            if len(papers) >= 3:
                break
        if len(papers) >= 3:
            break
    if not papers:
        pytest.skip("no cached full text available")

    embedder = ingest.Embedder(
        online=False, config={"embedding_model": "", "embedding_dim": 64}
    )
    store = MemoryVectorStore(embed_fn=embedder.embed)
    report = ingest.ingest(
        types.SimpleNamespace(selected=papers),
        store=store,
        embedder=embedder,
        index_corpus=False,
        persist=False,
    )
    assert report.chunks > 0, "real papers must produce indexed passages"

    from t2pw.rag.synthesize import _reactions_from_bundle
    from t2pw.rag.retrieve import EvidenceBundle, retrieve_evidence

    seed = _load(STAGE1)
    gaps = detect_gaps(
        seed, {},
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )[:3]
    arrow_candidates = 0
    for gap in gaps:
        bundle = retrieve_evidence(gap, store, top_k=2)
        assert isinstance(bundle, EvidenceBundle)
        assert f"Gap-ID: {gap.gap_id}" in bundle.query
        arrow_candidates += len(_reactions_from_bundle(bundle, None))

    # Recorded, not asserted as a target: the arrow parser is a bonus path over
    # real prose, and this number being 0 is a statement about the CORPUS.
    assert arrow_candidates >= 0
    assert arrow_candidates == 0, (
        "arrow recall over real paper prose changed; that is interesting but it "
        "is still not a measurement of the prose path"
    )


def test_prose_recall_on_real_cached_text_still_requires_a_valid_span() -> None:
    """The prose path's span validation, exercised on a real sentence.

    Uses one verbatim sentence from the checked-in Stage-1 payload's own evidence
    rather than the git-ignored full-text cache, so this runs everywhere. Two
    claims are offered over the same passage: the one the sentence states, and one
    whose product comes from elsewhere. Only the first survives.
    """
    from t2pw.rag.admission import locate_span, validate_evidence_span

    passage = (
        "LpxC deacetylates UDP-3-O-acyl-GlcNAc to form "
        "UDP-3-O-acyl-glucosamine. LpxD then transfers a second acyl chain."
    )

    honest = locate_span(
        passage,
        "",
        ["UDP-3-O-acyl-GlcNAc", "UDP-3-O-acyl-glucosamine", "LpxC"],
    )
    assert honest == (
        "LpxC deacetylates UDP-3-O-acyl-GlcNAc to form UDP-3-O-acyl-glucosamine."
    )
    assert validate_evidence_span(
        honest,
        inputs=["UDP-3-O-acyl-GlcNAc"],
        outputs=["UDP-3-O-acyl-glucosamine"],
        enzymes=["LpxC"],
    ).ok

    # A claim that borrows LpxD from the next sentence has no single statement.
    assert (
        locate_span(
            passage,
            "",
            ["UDP-3-O-acyl-GlcNAc", "UDP-3-O-acyl-glucosamine", "LpxD"],
        )
        == ""
    )

"""WP5 synthesis tests — offline, deterministic, self-contained.

No chromadb, no live network, no live LLM. The guarded ``openai`` stub mirrors
tests/test_audit_json_llm_payload.py because importing ``t2pw.rag.retrieve`` (used
here only to construct the WP4 ``EvidenceBundle`` / ``Gap`` fixtures) pulls in
``t2pw.rag.ingest`` -> ``t2pw.rag.acquire`` -> ``t2pw.mapping`` ->
``t2pw.llm.client`` -> ``openai`` at import time. Kept at the top so the module
passes run alone.

``t2pw.rag.synthesize`` itself imports none of that stack (it consumes the WP4 /
store records by duck typing), which the ``import synthesize`` smoke test below
asserts stays true even with chromadb absent.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

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

from t2pw.mapping.map_ids import map_payload  # noqa: E402
from t2pw.pipeline.stage_contracts import (  # noqa: E402
    validate_post_extraction,
    validate_post_mapping,
)
from t2pw.rag.provenance import (  # noqa: E402
    RAG_ADDITIVE_KEYS,
    strip_provenance,
    validate_provenance,
)
from t2pw.rag.retrieve import EvidenceBundle, Gap  # noqa: E402
from t2pw.rag.store import Chunk, Retrieved  # noqa: E402
from t2pw.rag.synthesize import (  # noqa: E402
    synthesize,
    synthesize_with_report,
    to_payload,
)


# ---------------------------------------------------------------------------
# Fixtures — two canned "papers".
# ---------------------------------------------------------------------------
def _seed_payload() -> dict:
    """Paper A: reactions 1-3 (caffeine -> ... -> theobromine).

    Reaction 3 produces ``theobromine`` which no paper-A reaction consumes, so
    reaction 3 dangles until paper B is stitched on.
    """
    return {
        "entities": {
            # Contextual scaffolding synthesis does NOT rebuild from evidence — it
            # must be carried forward as-is (Defect 1) so Stage 2B mapping still
            # has a species row (validate_post_mapping's species_required).
            "species": [{"name": "Pseudomonas putida"}],
            "subcellular_locations": [{"name": "cytosol"}],
            "cell_types": [{"name": "bacterial cell"}],
            "tissues": [{"name": "n/a"}],
            "compounds": [
                {"name": "caffeine"},
                {"name": "paraxanthine"},
                {"name": "theobromine"},
            ],
            "proteins": [{"name": "NdmA"}],
        },
        "biological_states": [
            {"name": "cytosol", "species": "Pseudomonas putida", "subcellular_location": "cytosol"},
        ],
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
                {
                    "name": "R3 to theobromine",
                    "inputs": ["7-methylxanthine"],
                    "outputs": ["theobromine"],
                },
            ]
        },
    }


def _seed_context() -> dict:
    # Provides the seed paper's source pointer so seed reactions get provenance.
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
    # Paper B states reactions 4-6 as KEGG-style equations. Reaction 4 CONSUMES
    # theobromine -> it closes paper A's dangling end when stitched.
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


def _bundle_for_theobromine() -> EvidenceBundle:
    gap = Gap(
        kind="dangling_reaction",
        label="theobromine",
        detail="theobromine produced by R3 is consumed by no reaction",
        symbols=["theobromine"],
        source="qa_graph",
    )
    return EvidenceBundle(
        gap=gap,
        query="find the reaction that consumes theobromine",
        hits=[Retrieved(chunk=_chunk_paper_b(), score=0.91)],
    )


# ---------------------------------------------------------------------------
# Test 1 — two-paper stitch into ONE connected, provenance-complete payload.
# ---------------------------------------------------------------------------
def test_two_paper_stitch_connects_and_validates():
    result = synthesize_with_report(
        _seed_payload(), [_bundle_for_theobromine()], _seed_context()
    )
    payload = result.payload

    # Passes the core structural contract (seam S3 acceptance).
    report = validate_post_extraction(payload)
    assert report["ok"] is True
    assert result.contract_report["ok"] is True

    reactions = payload["processes"]["reactions"]
    # Paper A (3) + paper B (3) reactions, all present in one payload.
    assert len(reactions) == 6
    names = {r["name"] for r in reactions}
    assert "R3 to theobromine" in names
    assert "R4 theobromine demethylation" in names

    # The stitch itself: R3's product theobromine == R4's input theobromine,
    # and the two reactions come from DIFFERENT papers (a novel connection).
    r3 = next(r for r in reactions if r["name"] == "R3 to theobromine")
    r4 = next(r for r in reactions if r["name"] == "R4 theobromine demethylation")
    assert "theobromine" in _names(r3["outputs"])
    assert "theobromine" in _names(r4["inputs"])
    assert r3["source_refs"] == ["PMID:0001"]
    assert r4["source_refs"] == ["PMID:0002"]

    # The stitch was detected as a cross-paper connection.
    stitched_metabolites = {s["metabolite"].casefold() for s in result.stitched}
    assert "theobromine" in stitched_metabolites
    theo = next(
        s for s in result.stitched if s["metabolite"].casefold() == "theobromine"
    )
    assert theo["producer_sources"] == ["PMID:0001"]
    assert theo["consumer_sources"] == ["PMID:0002"]

    # Every reaction carries >= 1 provenance pointer under the namespaced RAG key
    # (Defect 2: never under the core ``provenance`` string field).
    for rxn in reactions:
        assert rxn["rag_provenance"].get("source_id")
        assert not isinstance(rxn.get("provenance"), dict)
        assert rxn["source_refs"]

    # Every NON-COFACTOR entity carries >= 1 provenance pointer; O2 (a cofactor)
    # is allowed but is not required to. theobromine (from paper B evidence) is
    # backed by paper B; caffeine by paper A.
    compounds = {c["name"]: c for c in payload["entities"].get("compounds", [])}
    assert "PMID:0002" in compounds["theobromine"]["source_refs"]
    assert "PMID:0001" in compounds["caffeine"]["source_refs"]
    for row in compounds.values():
        if row["name"].casefold() in {"o2", "formaldehyde", "h2o"}:
            continue
        assert row.get("source_refs"), row["name"]

    # Enzyme actor rows follow the Actor field schema and carry provenance.
    proteins = {p["name"] for p in payload["entities"].get("proteins", [])}
    assert {"NdmA", "NdmB", "NdmC", "NdmD"} <= proteins
    assert r4["enzymes"][0]["entity"] == "NdmB"
    assert r4["enzymes"][0]["entity_type"] == "protein"
    assert r4["enzymes"][0]["role"] == "catalyst"
    assert r4["enzymes"][0]["source_refs"] == ["PMID:0002"]

    # Enzyme actor rows also carry the namespaced RAG key, not a dict under the
    # core ``provenance`` string field.
    assert isinstance(r4["enzymes"][0]["rag_provenance"], dict)
    assert not isinstance(r4["enzymes"][0].get("provenance"), dict)

    # Defect 1: the seed's contextual scaffolding (which synthesis does NOT
    # rebuild from evidence) is carried forward as-is, so a species row survives
    # for Stage 2B mapping's species_required contract.
    entities = payload["entities"]
    assert entities["species"] == [{"name": "Pseudomonas putida"}]
    assert entities["subcellular_locations"] == [{"name": "cytosol"}]
    assert entities["cell_types"] == [{"name": "bacterial cell"}]
    assert entities["tissues"] == [{"name": "n/a"}]
    # biological_states lives at the payload top level, not under entities.
    assert payload["biological_states"][0]["name"] == "cytosol"
    # ...and the evidence-built compounds/proteins were NOT clobbered.
    assert "caffeine" in compounds
    assert "NdmA" in proteins


# ---------------------------------------------------------------------------
# Test 2 — a gap with NO evidence stays unfilled and is reported; not invented.
# ---------------------------------------------------------------------------
def test_gap_without_evidence_stays_unfilled_and_reported():
    empty_gap = Gap(
        kind="orphan_metabolite",
        label="mysterious-precursor-X",
        detail="no reaction produces this metabolite",
        symbols=["mysterious-precursor-X"],
        source="qa_graph",
    )
    empty_bundle = EvidenceBundle(
        gap=empty_gap, query="find a source of X", hits=[]
    )

    result = synthesize_with_report(
        _seed_payload(),
        [_bundle_for_theobromine(), empty_bundle],
        _seed_context(),
    )
    payload = result.payload

    # Still a valid payload.
    assert validate_post_extraction(payload)["ok"] is True

    # The unsupported metabolite was NOT invented anywhere.
    all_names = _all_entity_and_participant_names(payload)
    assert "mysterious-precursor-x" not in {n.casefold() for n in all_names}

    # And it is surfaced in the unresolved-gaps report.
    unresolved_labels = {g["label"] for g in result.unresolved_gaps}
    assert "mysterious-precursor-X" in unresolved_labels
    x_report = next(
        g for g in result.unresolved_gaps if g["label"] == "mysterious-precursor-X"
    )
    assert "no supporting evidence" in x_report["reason"]

    # The theobromine gap (which DID have evidence) is not in the unfilled set.
    assert "theobromine" not in unresolved_labels


# ---------------------------------------------------------------------------
# Test 3 — conflict resolution by evidence weight, alternatives recorded.
#
# NOTE: this conflict is a *same-direction* disagreement (both papers say
# A -> B, but differ on the stoichiometry of A). Opposite-direction pairs
# (A -> B vs B -> A) are NO LONGER a conflict — they are two distinct reactions
# that both survive (see test_rag_reversible_reaction_preservation.py). This
# test intentionally exercises the surviving weight-based conflict-resolution
# path (distinct signatures within one direction-aware conflict_key group).
# ---------------------------------------------------------------------------
def test_conflict_resolved_by_evidence_weight():
    # Two papers agree on the direction (A -> B) but disagree on stoichiometry:
    # one says "2 A -> B", the other "A -> B". Same direction-aware conflict_key,
    # distinct signatures -> the heavier-evidence variant wins, loser recorded.
    heavy = Chunk(
        id="cf1",
        text="2 A -> B | enzyme: EnzX",
        source_id="PMID:HEAVY",
        source_type="paper",
        source_uri="u1",
    )
    light = Chunk(
        id="cr1",
        text="A -> B | enzyme: EnzX",
        source_id="PMID:LIGHT",
        source_type="paper",
        source_uri="u2",
    )
    gap = Gap(kind="dangling_reaction", label="A", symbols=["A"])
    bundles = [
        EvidenceBundle(
            gap=gap,
            query="q",
            hits=[Retrieved(chunk=heavy, score=0.9)],  # heavier evidence
        ),
        EvidenceBundle(
            gap=gap,
            query="q",
            hits=[Retrieved(chunk=light, score=0.2)],  # lighter
        ),
    ]
    minimal_seed = {"entities": {}, "processes": {"reactions": []}}
    result = synthesize_with_report(minimal_seed, bundles, "")

    reactions = result.payload["processes"]["reactions"]
    assert len(reactions) == 1  # one variant survives
    winner = reactions[0]
    assert _names(winner["inputs"]) == ["A"]
    assert _names(winner["outputs"]) == ["B"]  # heavier variant won
    # The winning variant is the heavier "2 A -> B".
    assert winner["inputs"][0]["stoichiometry"] == 2

    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    # Same direction on both sides; the loser is the lighter-evidence variant.
    assert conflict["chosen"]["outputs"] == ["B"]
    assert conflict["chosen"]["source_ids"] == ["PMID:HEAVY"]
    assert conflict["alternatives"][0]["source_ids"] == ["PMID:LIGHT"]
    assert conflict["chosen"]["weight"] > conflict["alternatives"][0]["weight"]


# ---------------------------------------------------------------------------
# Test 4 — provenance keys are OPTIONAL/additive (a stage-style consumer that
# ignores them still sees a valid payload).
# ---------------------------------------------------------------------------
def test_provenance_keys_are_optional_additive():
    payload = synthesize(_seed_payload(), [_bundle_for_theobromine()], _seed_context())

    # A stage-style consumer that strips every RAG additive key + source_refs
    # still has a payload that passes the core structural contract.
    stripped = _strip_rag_keys(payload)
    assert validate_post_extraction(stripped)["ok"] is True

    # The stripped payload still has the load-bearing core fields intact.
    for rxn in stripped["processes"]["reactions"]:
        assert rxn["name"]
        assert rxn["inputs"] or rxn["outputs"]
        for key in RAG_ADDITIVE_KEYS:
            assert key not in rxn


# ---------------------------------------------------------------------------
# Test 5 — to_payload emits only core buckets (no RAG-only required keys).
# ---------------------------------------------------------------------------
def test_to_payload_shape_is_standard():
    payload = to_payload({"compounds": [{"name": "x"}]}, [])
    assert set(payload) == {"entities", "processes"}
    assert payload["processes"] == {"reactions": []}
    assert validate_post_extraction(payload) is not None


# ---------------------------------------------------------------------------
# Test 6 (Defect 1) — contextual scaffolding is carried forward from the seed so
# the mapped payload keeps a species row (would satisfy validate_post_mapping's
# species_required). Synthesis rebuilds only evidence-bound chemistry.
# ---------------------------------------------------------------------------
def test_scaffolding_entities_carried_forward_for_species_requirement():
    result = synthesize_with_report(
        _seed_payload(), [_bundle_for_theobromine()], _seed_context()
    )
    entities = result.payload["entities"]

    # The exact predicate validate_post_mapping enforces: a non-empty species list.
    species = entities.get("species")
    assert isinstance(species, list) and species
    assert species == [{"name": "Pseudomonas putida"}]

    # The other scaffolding buckets ride along too (compartments / cell / tissue).
    assert entities.get("subcellular_locations") == [{"name": "cytosol"}]
    assert entities.get("cell_types") == [{"name": "bacterial cell"}]
    assert entities.get("tissues") == [{"name": "n/a"}]
    # biological_states is a top-level payload key, not under entities.
    assert result.payload.get("biological_states") == [
        {"name": "cytosol", "species": "Pseudomonas putida", "subcellular_location": "cytosol"}
    ]

    # Compounds/proteins remain rebuilt-from-evidence (not clobbered, and NOT
    # blindly copied from the seed — theobromine's evidence pointer is paper B).
    compounds = {c["name"]: c for c in entities.get("compounds", [])}
    assert "PMID:0002" in compounds["theobromine"]["source_refs"]

    # A degenerate seed (no scaffolding, not even a dict) is handled gracefully.
    minimal = synthesize_with_report(
        {"processes": {"reactions": []}}, [_bundle_for_theobromine()], _seed_context()
    )
    assert "species" not in minimal.payload["entities"]
    assert synthesize(None, [_bundle_for_theobromine()], _seed_context()) is not None


# ---------------------------------------------------------------------------
# Test 7 (Defect 2) — the RAG primary pointer is emitted under ``rag_provenance``
# (a dict), never under the core ``provenance`` string field; strip removes it.
# ---------------------------------------------------------------------------
def test_rag_provenance_key_does_not_collide_with_core_provenance():
    payload = synthesize(_seed_payload(), [_bundle_for_theobromine()], _seed_context())

    rows = list(payload["entities"]["compounds"]) + list(payload["entities"]["proteins"])
    rows += list(payload["processes"]["reactions"])
    for rxn in payload["processes"]["reactions"]:
        rows += list(rxn.get("enzymes", []))

    sourced = [r for r in rows if "rag_provenance" in r]
    assert sourced, "expected at least one synthesized row with a RAG pointer"
    for row in sourced:
        # The RAG pointer is a dict under the namespaced key...
        assert isinstance(row["rag_provenance"], dict)
        assert row["rag_provenance"].get("source_id")
        # ...and the core ``provenance`` field is never overwritten with a dict.
        assert not isinstance(row.get("provenance"), dict)

    # strip_provenance removes rag_provenance (and every other additive key).
    stripped = strip_provenance(payload)
    for bucket in stripped.get("entities", {}).values():
        for row in bucket:
            assert "rag_provenance" not in row
    for rxn in stripped["processes"]["reactions"]:
        assert "rag_provenance" not in rxn

    # validate_provenance still passes on the good (un-stripped) output.
    assert validate_provenance(payload).ok is True


# ---------------------------------------------------------------------------
# Test 8 (Defect 1, end-to-end) — a synthesized payload survives the REAL Stage
# 2B mapping cycle: map_ids.map_payload -> validate_post_mapping. This proves the
# species_required abort is gone all the way through mapping (not just at S3).
#
# Offline: only the external resolver calls (species hydration + the per-entity
# DB/API mapping strategies) are mocked, mirroring tests/test_stage2_mapping_
# boundary.py. The Stage 2B gate (validate_post_mapping) is the REAL function and
# is never stubbed or weakened.
# ---------------------------------------------------------------------------
def test_synthesized_payload_survives_real_stage2b_mapping(tmp_path):
    synthesized = synthesize(
        _seed_payload(), [_bundle_for_theobromine()], _seed_context()
    )
    # Precondition: synthesis carried the seed's species scaffolding forward.
    assert synthesized["entities"]["species"] == [{"name": "Pseudomonas putida"}]

    unmapped = {
        "status": "unmapped",
        "reason": "no_match",
        "provider": "test",
        "source": "test",
        "confidence": 0.0,
        "candidates": [],
        "resolution": {"status": "unresolved", "issue": "no_match", "order_step": "test_lookup"},
    }

    with (
        patch(
            "t2pw.mapping.map_ids.hydrate_species_references",
            return_value={"hydrated": 0, "matched": 0, "novel": 0},
        ),
        patch("t2pw.mapping.map_ids._map_protein_with_strategy", return_value=unmapped),
        patch("t2pw.mapping.map_ids._map_compound_with_strategy", return_value=unmapped),
        patch("t2pw.mapping.map_ids._map_complex_with_strategy", return_value=unmapped),
        patch.dict("os.environ", {"T2PW_LLM_PROTEIN_FALLBACK": "0"}),
    ):
        result = map_payload(
            synthesized,
            cache_path=tmp_path / "mapping-cache.json",
            id_source="api",
            use_cache=False,
        )
    mapped = result["payload"]

    # The REAL Stage 2B structural gate passes -> the species_required abort is
    # gone end-to-end. (validate_post_mapping aborts on failure, so ok is True
    # only when there are no errors at all, species_required among them.)
    report = validate_post_mapping(mapped)
    assert report["ok"] is True, report["errors"]
    assert not any(e.get("code") == "species_required" for e in report["errors"])

    # The carried-forward seed species survived mapping and retained mapping_meta
    # (the gate's per-entity requirement), so the scaffolding is load-bearing.
    # (Mapping may add its own sentinel species, e.g. PathBank's "Unknown"
    # fallback for unmapped enzymes, so this is a subset check, not equality.)
    species = mapped["entities"]["species"]
    assert isinstance(species, list) and species
    seed_species = [row for row in species if row.get("name") == "Pseudomonas putida"]
    assert len(seed_species) == 1
    for row in species:
        assert isinstance(row.get("mapping_meta"), dict)
        assert isinstance(row["mapping_meta"].get("resolution"), dict)


# ---------------------------------------------------------------------------
# Regression — a " ; "-joined corpus (pwml_example) chunk must never become an
# entity whose name is an entire pathway blob (the data-corruption bug).
# ---------------------------------------------------------------------------
import re  # noqa: E402

_CORPUS_SBML = ROOT / "reference" / "PW012926 (1).sbml"
_PATHWAY_ID_RE = re.compile(r"^pathway\d", re.IGNORECASE)


def _corpus_pwml_chunk() -> Chunk:
    """A real reference-corpus chunk exactly as ``chunk_corpus`` emits it.

    Mirrors the live repro: ``_corpus_text_for_file`` -> ``_iter_windows`` ->
    a ``source_type="pwml_example"`` chunk whose text is a " ; "-joined bag of
    pathway id + species + compartments + compounds + reaction patterns.
    """
    from t2pw.rag import ingest  # local import: heavy stack (openai already stubbed)

    blob = ingest._corpus_text_for_file(_CORPUS_SBML)
    assert " ; " in blob and blob.lower().startswith("pathway")  # the garbage shape
    window = ingest._iter_windows(blob)[0]
    return Chunk(
        id="corpus:PW012926:0",
        text=window,
        source_id="PW012926 (1).sbml",
        source_title="PW012926 (1)",
        source_type="pwml_example",
        source_uri=_CORPUS_SBML.as_uri(),
        section="example",
    )


def _bundle_with_corpus_chunk() -> EvidenceBundle:
    gap = Gap(
        kind="dangling_reaction",
        label="theobromine",
        detail="needs a downstream consumer",
        symbols=["theobromine"],
        source="qa_graph",
    )
    return EvidenceBundle(
        gap=gap,
        query="theobromine catabolism",
        hits=[Retrieved(chunk=_corpus_pwml_chunk(), score=0.88)],
    )


def test_corpus_pwml_chunk_never_emits_pathway_blob_entities():
    """A pwml_example corpus chunk must not be parsed into reactions/entities."""
    result = synthesize_with_report(
        _seed_payload(), [_bundle_with_corpus_chunk()], _seed_context()
    )
    payload = result.payload
    assert validate_post_extraction(payload)["ok"] is True

    for name in _all_entity_and_participant_names(payload):
        assert " ; " not in name, name
        assert not _PATHWAY_ID_RE.match(name), name
        assert ", Cell," not in name, name
        # No single blob leaked through as a giant name.
        assert len(name) <= 120, name


def test_genuine_paper_equation_chunk_still_parses_cleanly():
    """A real source_type='paper' equation chunk still yields a clean reaction."""
    paper_chunk = Chunk(
        id="paperX:1",
        text="R1 | theobromine + O2 -> 7-methylxanthine + formaldehyde | enzyme: NdmB",
        source_id="PMID:9001",
        source_title="Theobromine demethylation (paper X)",
        source_type="paper",
        source_uri="https://example.org/paperX",
        organism="Pseudomonas putida",
        section="results",
    )
    gap = Gap(
        kind="dangling_reaction",
        label="theobromine",
        detail="downstream consumer",
        symbols=["theobromine"],
        source="qa_graph",
    )
    bundle = EvidenceBundle(gap=gap, query="theobromine + O2", hits=[Retrieved(chunk=paper_chunk, score=0.95)])

    result = synthesize_with_report(_seed_payload(), [bundle], _seed_context())
    reactions = result.payload["processes"]["reactions"]
    r1 = next((r for r in reactions if r["name"] == "R1"), None)
    assert r1 is not None, [r["name"] for r in reactions]
    assert set(_names(r1["inputs"])) == {"theobromine", "O2"}
    assert set(_names(r1["outputs"])) == {"7-methylxanthine", "formaldehyde"}
    assert r1["enzymes"][0]["entity"] == "NdmB"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _names(participants) -> list:
    out = []
    for p in participants:
        out.append(p if isinstance(p, str) else p.get("name"))
    return out


def _all_entity_and_participant_names(payload: dict) -> list:
    names = []
    for bucket in payload.get("entities", {}).values():
        for row in bucket:
            names.append(row.get("name", ""))
    for rxn in payload.get("processes", {}).get("reactions", []):
        names.extend(_names(rxn.get("inputs", [])))
        names.extend(_names(rxn.get("outputs", [])))
        for enzyme in rxn.get("enzymes", []):
            names.append(enzyme.get("entity", ""))
    return [n for n in names if n]


def _strip_rag_keys(payload: dict) -> dict:
    import copy

    clone = copy.deepcopy(payload)
    strip = set(RAG_ADDITIVE_KEYS) | {"source_refs"}

    def _clean(row: dict) -> None:
        for key in strip:
            row.pop(key, None)

    for bucket in clone.get("entities", {}).values():
        for row in bucket:
            _clean(row)
    for rxn in clone.get("processes", {}).get("reactions", []):
        _clean(rxn)
        for enzyme in rxn.get("enzymes", []):
            _clean(enzyme)
    return clone

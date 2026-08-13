"""NEW-CAPABILITY ACCEPTANCE TESTS for C-033 / D-003.

**Every test here is a new acceptance test, not a G9 regression.**
``src/t2pw/mapping/uniprot_evidence.py`` does not exist at the dispatch base, so
there is no base behaviour for these to fail against and none is manufactured.
The one genuine regression proof -- the pinned ``Fur`` -> P0A9A9 strip -- is in
``tests/test_identity_hydration_regression.py``.

Pinned here, clause by clause: the D-003 flow order; the failure semantics
(``unavailable`` / ``not_evaluated`` distinct from ``rejected``, and a dead
lookup never producing ``rejected``); the preservation rule (neither promoted
nor erased); the boundaries (exporters perform no lookups, saved canonical JSON
re-exports with no UniProt in reach); and the invariants -- hydration supplies
evidence to the rungs, it does not relax them.

**No test here touches the network.** Every fetch is a fake client; every cache
is a tmp file read by the real ``EnrichmentCache``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping.enrich_entities import EnrichmentCache  # noqa: E402
from t2pw.mapping.map_ids import (  # noqa: E402
    _enforce_shipped_identity_names,
    verify_real_protein_identity,
)
from t2pw.mapping.uniprot_evidence import (  # noqa: E402
    ENV_CACHE_PATH,
    ENV_FLAG,
    EVIDENCE_OK,
    EXACT_ACCESSION_MATCH_SCORE,
    UniProtIdentityEvidenceSource,
    candidate_from_uniprot_record,
    identity_evidence_for,
    identity_evidence_source,
    pathbank_evidence_is_adequate,
    uniprot_cache_key,
)
from t2pw.pipeline.entity_identity import (  # noqa: E402
    VERIFICATION_NOT_EVALUATED,
    VERIFICATION_REJECTED,
    VERIFICATION_UNAVAILABLE,
    VERIFICATION_VERIFIED,
    unverified_identity_claim,
)
from tests.test_identity_hydration_regression import CACHED_P0A9A9, cached_source  # noqa: E402

ECOLI = "Escherichia coli"

#: A human protein: the 'mcr genes' -> P08235 shape of run 2026-07-28_0919.
CACHED_P08235: Dict[str, Any] = {
    "status": "ok",
    "parsed": {
        "status": "ok",
        "uniprot_id": "P08235",
        "recommended_name": "Mineralocorticoid receptor",
        "gene_names": ["NR3C2", "MCR", "MLR"],
        "organism": "Homo sapiens",
        "taxonomy_id": "9606",
        "provenance": {"source": "UniProt", "query": "P08235"},
    },
}

#: A real E. coli protein that is simply not the one the entity names. From the
#: pinned run's own rejected pool, and deliberately NOT the accession
#: ``fur_ladder`` puts in the candidate list, so hydration is what supplies it.
CACHED_P0A6B7: Dict[str, Any] = {
    "status": "ok",
    "parsed": {
        "status": "ok",
        "uniprot_id": "P0A6B7",
        "recommended_name": "Cysteine desulfurase",
        "gene_names": ["iscS"],
        "organism": "Escherichia coli (strain K12)",
        "taxonomy_id": "83333",
        "provenance": {"source": "UniProt", "query": "P0A6B7"},
    },
}

RAW_P0A9A9: Dict[str, Any] = {
    "primaryAccession": "P0A9A9",
    "proteinDescription": {"recommendedName": {"fullName": {"value": "Ferric uptake regulation protein"}}},
    "genes": [{"geneName": {"value": "fur"}}],
    "organism": {"scientificName": "Escherichia coli (strain K12)", "taxonId": 83333},
    "sequence": {"length": 148},
}

PATHBANK_FUR_ROW = {
    "pathbank_protein_id": 4242, "name": "Ferric uptake regulation protein", "gene_name": "fur",
    "uniprot": "P0A9A9", "species_id": 3, "organism": ECOLI, "taxonomy_id": "562", "score": 1.08,
}

POOL_ROW = {
    "pathbank_protein_id": 5917, "name": "Succinate dehydrogenase iron-sulfur subunit",
    "gene_name": "sdhB", "uniprot": "P07014", "species_id": 3, "score": 0.65,
}


@pytest.fixture(autouse=True)
def _no_ambient_evidence_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Immunise this whole file against ambient environment state.

    ``identity_evidence_for`` falls through to
    ``default_identity_evidence_source()`` when nothing is installed, and that
    reads ``T2PW_IDENTITY_EVIDENCE``. Set to ``cache`` it changes what a test
    observes; set to ``network`` it builds a real ``HttpClient``, issues a live
    GET, and then ``cache.save()``s into ``T2PW_ENRICHMENT_CACHE`` -- whose
    default is the 39 MB tracked ``data/enrichment_cache.json``. Every source in
    this file is explicit, so the ambient one is removed for all of them.
    """

    monkeypatch.delenv(ENV_FLAG, raising=False)
    monkeypatch.delenv(ENV_CACHE_PATH, raising=False)


# ── doubles. None of them can reach a network. ─────────────────────────────


class _Response:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        self._payload, self.status_code = payload, status_code

    def json(self) -> Dict[str, Any]:
        return self._payload


class FakeClient:
    """An ``HttpClient`` stand-in. Records its calls; never opens a socket."""

    def __init__(self, payload: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None) -> None:
        self.payload, self.error = payload, error
        self.urls: List[str] = []

    def get(self, url: str, **_: Any) -> _Response:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return _Response(self.payload or {})


class FakeDb:
    """A ``PathBankDbResolver`` stand-in for ``map_protein_by_ids`` only."""

    def __init__(self, result: Any = None, error: Optional[Exception] = None) -> None:
        self.result, self.error = result, error
        self.calls: List[Dict[str, Any]] = []

    def map_protein_by_ids(self, ids: Dict[str, str], *, species: Any = None) -> Any:
        self.calls.append({"ids": dict(ids), "species": species})
        if self.error is not None:
            raise self.error
        return self.result


def fur_ladder(evidence_source: Any, accession: str = "P0A9A9") -> Dict[str, Any]:
    """The ladder for entity 'Fur' against a pool that cannot describe it."""

    return verify_real_protein_identity(
        "Fur", candidates=[dict(POOL_ROW)], mapped_ids={"uniprot": accession},
        organism=ECOLI, source="db", evidence_source=evidence_source,
    )


# ── D-003 flow order ───────────────────────────────────────────────────────


def test_pathbank_by_exact_identifier_is_consulted_before_uniprot(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. Step 2 answers, so step 3 is never reached."""

    db = FakeDb({"status": "mapped", "candidates": [dict(PATHBANK_FUR_ROW)]})
    client = FakeClient(RAW_P0A9A9)
    source = UniProtIdentityEvidenceSource(
        db=db, cache=EnrichmentCache(tmp_path / "cache.json"), client=client, allow_network=True
    )

    record = source.evidence_for("P0A9A9", organism=ECOLI)

    assert record["status"] == EVIDENCE_OK
    assert record["lookup"] == "pathbank_exact_identifier"
    assert record["candidate"]["evidence_source"] == "PathBankDB"
    assert db.calls == [{"ids": {"uniprot": "P0A9A9"}, "species": ECOLI}]
    assert client.urls == [], "UniProt must not be called when PathBank answered"


def test_uniprot_is_consulted_when_the_pathbank_row_cannot_answer_the_rungs(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. 'Inadequate' is defined against the rungs, not taste.

    A PathBank row carrying only a bare ``species_id`` foreign key is the gap
    ``tests/test_db_candidate_species_evidence.py`` documents: rung 3 cannot read
    it, so the evidence is inadequate and the flow goes on.
    """

    bare_row = {"pathbank_protein_id": 4242, "uniprot": "P0A9A9", "name": "Fur", "species_id": 3}
    assert pathbank_evidence_is_adequate(bare_row) is False
    assert pathbank_evidence_is_adequate(PATHBANK_FUR_ROW) is True

    source = cached_source(tmp_path, {"uniprot:P0A9A9": CACHED_P0A9A9})
    source.db = FakeDb({"status": "mapped", "candidates": [bare_row]})

    record = source.evidence_for("P0A9A9", organism=ECOLI)

    assert record["status"] == EVIDENCE_OK
    assert record["lookup"] == "uniprot_exact_accession"
    assert record["cached"] is True


def test_a_live_fetch_is_cached_and_persisted_with_its_response_provenance(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. D-003 step 4, over the relocated fetch and cache."""

    cache_file = tmp_path / "enrichment_cache.json"
    client = FakeClient(RAW_P0A9A9)
    source = UniProtIdentityEvidenceSource(
        cache=EnrichmentCache(cache_file), client=client, allow_network=True
    )

    record = source.evidence_for("P0A9A9", organism=ECOLI)

    assert record["status"] == EVIDENCE_OK and record["cached"] is False
    assert client.urls == ["https://rest.uniprot.org/uniprotkb/P0A9A9.json"]
    assert record["provenance"]["source"] == "UniProt"
    assert record["provenance"]["source_url"].endswith("P0A9A9.json")

    # Persisted under the key enrich_entities reads (`:1967`), so one fetch
    # serves both this ladder and the later enrichment pass.
    persisted = json.loads(cache_file.read_text(encoding="utf-8"))
    assert uniprot_cache_key("p0a9a9") == "uniprot:P0A9A9"
    assert persisted["proteins"]["uniprot:P0A9A9"]["parsed"]["gene_names"] == ["fur"]

    # A second ask is answered from the cache, not from a second request.
    assert source.evidence_for("P0A9A9", organism=ECOLI)["cached"] is True
    assert len(client.urls) == 1


# ── failure semantics: the states stay distinct ────────────────────────────


def test_a_network_failure_is_unavailable_and_never_a_rejection(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. Invariant: a lookup failure is not a biological verdict."""

    source = UniProtIdentityEvidenceSource(
        cache=EnrichmentCache(tmp_path / "cache.json"),
        client=FakeClient(error=OSError("connection reset")), allow_network=True,
    )

    record = source.evidence_for("P0A9A9", organism=ECOLI)

    assert record["status"] == VERIFICATION_UNAVAILABLE != VERIFICATION_REJECTED
    # The relocated fetch handles its own transport errors and reports them as
    # ``request_failed`` (`enrich_entities.py:528`); that shape is carried
    # through unchanged rather than re-derived here.
    assert record["detail"] == "request_failed:connection reset"

    verdict = fur_ladder(source)
    assert verdict["verified"] is False
    assert verdict["verification_status"] == VERIFICATION_UNAVAILABLE
    assert verdict["reason"] == "identity_evidence_missing"


def test_a_dead_database_is_unavailable_not_a_rejection(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. The same rule on the PathBank leg."""

    source = UniProtIdentityEvidenceSource(
        db=FakeDb(error=RuntimeError("no route to host")),
        cache=EnrichmentCache(tmp_path / "cache.json"), allow_network=False,
    )

    record = source.evidence_for("P0A9A9", organism=ECOLI)

    assert record["status"] == VERIFICATION_UNAVAILABLE
    assert record["detail"].startswith("pathbank_lookup_failed")


def test_with_no_source_nothing_is_looked_up_and_nothing_is_refuted() -> None:
    """NEW ACCEPTANCE. The default state, and the third of the three."""

    record = identity_evidence_for("P0A9A9", organism=ECOLI)

    assert record["status"] == VERIFICATION_NOT_EVALUATED
    assert record["detail"] == "no_identity_evidence_source"
    assert fur_ladder(None)["verification_status"] == VERIFICATION_NOT_EVALUATED


def test_the_four_verification_states_are_four_distinct_strings() -> None:
    """NEW ACCEPTANCE. ``not_evaluated`` may never collapse into a refutation."""

    states = [
        VERIFICATION_VERIFIED, VERIFICATION_REJECTED,
        VERIFICATION_UNAVAILABLE, VERIFICATION_NOT_EVALUATED,
    ]
    assert len(set(states)) == 4


def test_a_source_that_reports_a_refutation_is_clamped_not_believed() -> None:
    """NEW ACCEPTANCE. Only a rung may say ``rejected``; an injected source may not.

    Passed through, a foreign ``rejected`` would become the ladder's
    ``verification_status`` and suppress the preserved claim -- an absence of
    evidence rewritten as a biological verdict, which is invariant 8.
    """

    class RogueSource:
        def evidence_for(self, accession: str, *, organism: str = "") -> Dict[str, Any]:
            return {"status": VERIFICATION_REJECTED, "accession": accession,
                    "candidate": {"uniprot": accession, "protein_name": "Anything"}}

    record = identity_evidence_for("P0A9A9", organism=ECOLI, source=RogueSource())
    assert record["status"] == VERIFICATION_NOT_EVALUATED
    assert record["detail"] == "non_conforming_source_status:rejected"
    assert fur_ladder(RogueSource())["verification_status"] == VERIFICATION_NOT_EVALUATED

    # The harm the clamp prevents: the claim is preserved, not suppressed.
    row: Dict[str, Any] = {"name": "Fur", "mapped_ids": {"uniprot": "P0A9A9"}, "mapping_meta": {}}
    _enforce_shipped_identity_names(
        row, {"status": "unmapped", "candidates": []},
        kind="protein", entity_name="Fur", organism=ECOLI, evidence_source=RogueSource(),
    )
    assert unverified_identity_claim(row)["verification_status"] == VERIFICATION_NOT_EVALUATED


def test_an_unverifiable_accession_is_preserved_as_a_claim_and_not_promoted(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. D-003: do not promote it, do not erase it."""

    row: Dict[str, Any] = {"name": "Fur", "mapped_ids": {"uniprot": "P0A9A9"}, "mapping_meta": {}}
    source = UniProtIdentityEvidenceSource(
        cache=EnrichmentCache(tmp_path / "cache.json"),
        client=FakeClient(error=OSError("connection reset")), allow_network=True,
    )

    report = _enforce_shipped_identity_names(
        row, {"status": "unmapped", "candidates": []},
        kind="protein", entity_name="Fur", organism=ECOLI, evidence_source=source,
    )

    # Not promoted: the export identity degrades exactly as it did before.
    assert row["mapped_ids"] == {}
    assert row["mapping_meta"]["identity_status"] == "unresolved"
    assert report["rejected"] is True
    # Not erased: the accession survives as an explicit claim, filed under the
    # state that produced it rather than as a refutation nobody made.
    claim = unverified_identity_claim(row)
    assert claim["identifiers"] == {"uniprot": "P0A9A9"}
    assert claim["verification_status"] == VERIFICATION_UNAVAILABLE


def test_a_refuted_accession_is_not_recorded_as_an_unverified_claim(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. The distinction has to cut both ways to mean anything."""

    row: Dict[str, Any] = {"name": "mcr genes", "mapped_ids": {"uniprot": "P08235"}, "mapping_meta": {}}

    report = _enforce_shipped_identity_names(
        row, {"status": "unmapped", "candidates": []},
        kind="protein", entity_name="mcr genes", organism=ECOLI,
        evidence_source=cached_source(tmp_path, {"uniprot:P08235": CACHED_P08235}),
    )

    assert report["verification_status"] == VERIFICATION_REJECTED
    assert row["mapped_ids"] == {}
    assert unverified_identity_claim(row) == {}


# ── the rungs are unchanged: evidence, not leniency ────────────────────────


def test_hydration_does_not_admit_an_accession_from_another_organism(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. P08235 is real, well formed, and human. Rung 3 refuses it."""

    verdict = verify_real_protein_identity(
        "mcr genes", candidates=[], mapped_ids={"uniprot": "P08235"}, organism=ECOLI,
        evidence_source=cached_source(tmp_path, {"uniprot:P08235": CACHED_P08235}),
    )

    assert verdict["verified"] is False
    assert verdict["checks"]["species"] == "mismatch"
    assert verdict["verification_status"] == VERIFICATION_REJECTED


def test_hydration_does_not_admit_an_accession_that_names_another_molecule(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. Right organism, wrong protein. Rung 4 refuses it."""

    verdict = fur_ladder(cached_source(tmp_path, {"uniprot:P0A6B7": CACHED_P0A6B7}), accession="P0A6B7")

    assert verdict["verified"] is False
    assert verdict["checks"]["species"] == "ok"
    assert verdict["checks"]["name"] == "reject"
    assert verdict["verification_status"] == VERIFICATION_REJECTED


def test_a_well_formed_accession_that_nothing_describes_is_still_refused(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. Invariant: format validity is never grounds for acceptance."""

    verdict = fur_ladder(cached_source(tmp_path, {}), accession="P99999")

    assert verdict["verified"] is False
    assert verdict["checks"]["candidate_evidence"] == "no_candidate_describes_the_shipped_identifier"
    assert verdict["verification_status"] != VERIFICATION_REJECTED


def test_a_uniprot_record_that_names_nothing_yields_no_candidate() -> None:
    """NEW ACCEPTANCE. A candidate no rung could judge is not evidence."""

    assert candidate_from_uniprot_record({"uniprot_id": "P0A9A9"}) == {}
    assert candidate_from_uniprot_record({"recommended_name": "Fur"}) == {}
    built = candidate_from_uniprot_record(CACHED_P0A9A9["parsed"])
    assert built["score"] == EXACT_ACCESSION_MATCH_SCORE
    assert built["gene_names"] == ["fur"]
    # Never stamped: it would turn a strain-rank answer (83333) into a mismatch
    # against a species-rank request (562).
    assert "requested_taxonomy_id" not in built


def test_the_hydrated_candidate_is_a_copy_and_the_cached_record_is_not_mutated(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. 'Immutable' held by copying, because the ladder needs a dict."""

    source = cached_source(tmp_path, {"uniprot:P0A9A9": CACHED_P0A9A9})
    first = source.evidence_for("P0A9A9", organism=ECOLI)
    first["candidate"]["organism"] = "Vibrio cholerae"
    first["candidate"]["score"] = -99

    second = source.evidence_for("P0A9A9", organism=ECOLI)

    assert second["candidate"]["organism"] == "Escherichia coli (strain K12)"
    assert second["candidate"]["score"] == EXACT_ACCESSION_MATCH_SCORE
    assert CACHED_P0A9A9["parsed"]["organism"] == "Escherichia coli (strain K12)"


# ── boundaries: pre-freeze only ────────────────────────────────────────────


def test_no_exporter_performs_an_identity_evidence_lookup() -> None:
    """NEW ACCEPTANCE. D-003: exporters perform no network or database lookups.

    Checked structurally over the source tree rather than by importing, so a
    module reaching the evidence source by any spelling is caught.
    """

    offenders = [
        str(path.relative_to(SRC))
        for tree in (SRC / "t2pw" / "pwml", SRC / "t2pw" / "sbml")
        for path in tree.rglob("*.py")
        if "uniprot_evidence" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []

    importers = sorted(
        str(path.relative_to(SRC))
        for path in (SRC / "t2pw").rglob("*.py")
        if path.name != "uniprot_evidence.py" and "uniprot_evidence" in path.read_text(encoding="utf-8")
    )
    assert importers == [str(Path("t2pw") / "mapping" / "map_ids.py")]


def test_saved_canonical_json_re_exports_with_no_uniprot_in_reach(tmp_path: Path) -> None:
    """NEW ACCEPTANCE. D-003: UniProt must not be required to re-export saved JSON.

    Proved with a source whose client raises if touched: installed ambiently and
    then not reached, because network is not permitted and no db is wired.
    """

    exploding = UniProtIdentityEvidenceSource(
        cache=EnrichmentCache(tmp_path / "cache.json"),
        client=FakeClient(error=AssertionError("a re-export must never fetch")),
        allow_network=False,
    )

    with identity_evidence_source(exploding):
        verdict = fur_ladder(None)

    assert verdict["verified"] is False
    assert verdict["verification_status"] == VERIFICATION_NOT_EVALUATED
    assert exploding.client.urls == []
    assert verdict["identity_evidence"]["detail"] == "not_cached_and_network_not_permitted"

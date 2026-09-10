"""C-124 -- creating a species the local PathBank table does not have.

The measured defect
-------------------
``runs_smoke/2026-09-10_1135/papers/PMC13488460`` (mevalonate pathway) produced a
sound one-reaction core and then lost the whole export at the PWML required-field
gate::

    species_missing_taxonomy / species_missing_classification
    "PWML export failed: PWML required-field gate failed."

The organism is absent from the local PathBank species table, so its row reached
the gate with no taxonomy id and no Prokaryote|Eukaryote classification -- the two
facts PathWhiz needs to CREATE a species it has never seen. The archived payload
also shows what the same leg *did* resolve: a second species row, matched in the
local database under the reclassified spelling of the same organism, carrying that
spelling's PathBank id and naming the unresolved row's spelling as its own
``common_name``. NCBI answered for that spelling seconds later in the same leg.

So the fallback this card builds is: local lookup MISS -> ask the authority about
the name the local DATABASE itself asserts is the same organism -> take the
numeric taxonomy id and the classification NCBI answers with -> the species can be
created and the export continues. NCBI is the only source of an id anywhere in
this path, and every step is fail-closed.

Fixture
-------
``tests/fixtures/c124/PMC13488460_strict_2026-09-10.json`` is that leg's own
``strict/final_mapped.json``, verbatim except that ``source_text_index`` (33 KB of
paper offsets, which no gate here reads) was dropped. Nothing in this file
hard-codes the organism, its spellings or its real taxon id: every name is read
back out of the fixture and every taxonomy id a stub returns is synthetic.

No network. The NCBI client is stubbed through the existing ``client`` parameter.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Only symbols that exist at the base SHA are imported at module level, so the
# G9 base-failure proof below COLLECTS at 4d417e4f and fails on VALUES there.
# The symbols this card adds are reached through the module object inside the
# tests that need them, which are labelled NEW-CAPABILITY ACCEPTANCE.
import t2pw.mapping.map_ids as map_ids
from t2pw.mapping.map_ids import (
    MappingCache,
    _ncbi_esearch_taxid,
    _ncbi_taxonomy_lookup,
    _normalize_name,
    backfill_species_taxonomy,
)
from t2pw.pwml.ir import validate_required_pwml_contract

FIXTURE = Path(__file__).parent / "fixtures" / "c124" / "PMC13488460_strict_2026-09-10.json"


@pytest.fixture(autouse=True)
def _no_ncbi_rate_limit_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the E-utils courtesy sleep for the stubbed client.

    ``_ncbi_throttle`` only calls ``time.sleep`` so a burst of real requests
    cannot trip NCBI's 3/sec limit. No stub here makes a request, so the sleep is
    pure wall clock -- 21 s across this file. Nothing else about the lookup path
    is patched: every decision under test runs the production code.
    """
    monkeypatch.setattr("t2pw.mapping.map_ids._ncbi_throttle", lambda: None)

#: The two gate codes this card exists to stop losing an export to.
SPECIES_GATE_CODES = {"species_missing_taxonomy", "species_missing_classification"}

#: Synthetic taxonomy ids. A row carrying one of these can only have got it from
#: the stub below -- never from the fixture, never from a donor row.
STUB_TAXID = "990001"
ALT_TAXID = "990002"

#: What the stub's taxonomy record says. ``Bacteria`` is what
#: ``_classification_from_taxonomy`` maps to ``Prokaryote``; the expectation is a
#: property of the stub's own XML, not of any particular organism.
STUB_DIVISION = "Bacteria"
STUB_LINEAGE = "cellular organisms; Bacteria"
STUB_CLASSIFICATION = "Prokaryote"


# ---------------------------------------------------------------------------
# Fixture readers. Every organism name used below comes out of these.
# ---------------------------------------------------------------------------


def _payload() -> Dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _species_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [row for row in payload["entities"]["species"] if isinstance(row, dict)]


def _unresolved_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The row that blocked the export: no taxonomy id, no classification."""
    rows = [
        row
        for row in _species_rows(payload)
        if not str(row.get("taxonomy_id") or "").strip()
        and not str(row.get("classification") or "").strip()
    ]
    assert len(rows) == 1, "fixture must carry exactly one blocking species row"
    return rows[0]


def _donor_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The local DATABASE row that names the blocking row's spelling as its own."""
    target = _normalize_name(str(_unresolved_row(payload).get("name") or ""))
    rows = []
    for row in _species_rows(payload):
        if not row.get("pathbank_species_id"):
            continue
        spellings = [row.get("common_name"), row.get("raw_name"), *(row.get("aliases") or [])]
        if target in {_normalize_name(str(value or "")) for value in spellings if value}:
            rows.append(row)
    assert len(rows) == 1, "fixture must carry exactly one database row asserting that spelling"
    return rows[0]


def _pathway_header(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The pathway name/subject the production gate saw and the canonical payload
    does not carry. Without it the gate reports ``pathway_missing_name`` /
    ``pathway_missing_subject`` beside the species codes; with it the fixture
    reproduces that leg's committed ``pwml_required_field_gate_report.json``
    exactly -- error_codes == the two species codes, nothing else."""
    payload["metadata"] = {"pathway_name": "C-124 fixture pathway", "pathway_subject": "Metabolic"}
    return payload


def _gate_error_codes(payload: Dict[str, Any]) -> List[str]:
    report = validate_required_pwml_contract(copy.deepcopy(payload), strict_db=True)
    return [str(issue.get("code")) for issue in report["errors"]]


# ---------------------------------------------------------------------------
# The stub NCBI client. Injected through the existing ``client`` parameter of
# backfill_species_taxonomy -- no boundary is widened for it.
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, status_code: int, payload: Optional[Dict[str, Any]] = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Dict[str, Any]:
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class _StubTaxonomyClient:
    """Answers for the names it was told to answer for, and for nothing else.

    ``answers`` maps a normalized organism name to the taxid the stub hands back.
    Both esearch shapes the production ladder tries ("<name>[Scientific Name]"
    and the bare name) resolve to the same normalized key, so the stub reproduces
    the ladder without repeating it.
    """

    def __init__(self, answers: Optional[Dict[str, str]] = None, *, mode: str = "ok") -> None:
        self.answers = {_normalize_name(key): value for key, value in (answers or {}).items()}
        self.mode = mode
        self.terms: List[str] = []
        self.fetched: List[str] = []

    def _term_key(self, term: str) -> str:
        return _normalize_name(term.replace("[Scientific Name]", ""))

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, **_: Any) -> _StubResponse:
        params = params or {}
        if self.mode == "raise":
            raise RuntimeError("HTTP request failed after retries")
        if self.mode == "http_error":
            return _StubResponse(429)
        if url.endswith("esearch.fcgi"):
            term = str(params.get("term") or "")
            self.terms.append(term)
            taxid = self.answers.get(self._term_key(term), "")
            return _StubResponse(200, {"esearchresult": {"idlist": [taxid] if taxid else []}})
        if url.endswith("efetch.fcgi"):
            taxid = str(params.get("id") or "")
            self.fetched.append(taxid)
            xml = (
                "<TaxaSet><Taxon>"
                f"<TaxId>{taxid}</TaxId>"
                f"<Division>{STUB_DIVISION}</Division>"
                f"<Lineage>{STUB_LINEAGE}</Lineage>"
                "</Taxon></TaxaSet>"
            )
            return _StubResponse(200, None, xml)
        return _StubResponse(404)


def _stub_for_donor(payload: Dict[str, Any], taxid: str = STUB_TAXID) -> _StubTaxonomyClient:
    """A stub that answers ONLY for the database row's scientific name -- exactly
    what the production leg measured: the paper's own spelling resolved against
    nothing, the database spelling resolved."""
    return _StubTaxonomyClient({str(_donor_row(payload).get("name") or ""): taxid})


# ===========================================================================
# A -- THE REGRESSION. G9 BASE-FAILURE PROOF.
# ===========================================================================


def test_g9_a_blocking_species_is_created_from_the_db_asserted_synonym() -> None:
    """G9 BASE-FAILURE PROOF (A).

    Runs the production entry point with the production signature -- no argument
    added by this card -- so it executes at 4d417e4f too. At the base SHA the
    blocking row comes back with no ``taxonomy_id`` at all and the gate still
    reports both species codes: the assertions below fail ON VALUES, not on a
    missing symbol. At the tip the row carries the id NCBI answered with and the
    gate passes.
    """
    payload = _pathway_header(_payload())
    blocking = _unresolved_row(payload)
    organism = str(blocking["name"])
    donor = _donor_row(payload)

    # The committed failure, reproduced before anything is resolved.
    assert SPECIES_GATE_CODES.issubset(set(_gate_error_codes(payload)))
    assert set(_gate_error_codes(payload)) == SPECIES_GATE_CODES

    client = _stub_for_donor(payload)
    report = backfill_species_taxonomy(payload, client=client, enable_ncbi=True)

    row = _unresolved_row_by_name(payload, organism)
    assert row.get("taxonomy_id") == STUB_TAXID
    assert str(row.get("taxonomy_id") or "").isdigit()
    assert row.get("classification") == STUB_CLASSIFICATION
    # Display text is never rewritten -- the synonym is for lookup only.
    assert row["name"] == organism
    # The id came from the authority, not from the donor row sitting beside it.
    assert row["taxonomy_id"] != str(donor.get("taxonomy_id") or "")
    # Nor did it route through the PathBank ``Unknown`` sentinel: this row is a
    # creation, so it still carries no PathBank species id of its own.
    assert not any(row.get(key) for key in ("pathbank_species_id", "species_id", "pw_species_id"))

    meta = row["mapping_meta"]["taxonomy_backfill"]
    assert meta["source"] == "ncbi_db_synonym"
    assert meta["resolved_as"] == str(donor["name"])
    assert meta["donors"] == [str(donor["name"])]
    assert report["species_creation"]["resolved"] == 1
    assert organism not in report["unresolved"]

    # The existing pass ran FIRST, on the organism's own name, and failed there.
    assert client.terms[0] == f"{organism}[Scientific Name]"

    # And the export is no longer lost.
    assert _gate_error_codes(payload) == []
    assert validate_required_pwml_contract(copy.deepcopy(payload), strict_db=True)["ok"] is True


def _unresolved_row_by_name(payload: Dict[str, Any], organism: str) -> Dict[str, Any]:
    rows = [row for row in _species_rows(payload) if str(row.get("name") or "") == organism]
    assert len(rows) == 1
    return rows[0]


def _as_species_hydration_left_it(payload: Dict[str, Any]) -> str:
    """Rewind the database row to the shape ``backfill_species_taxonomy``
    actually sees in production, and return the strain-qualified name.

    ``final_mapped.json`` is post-freeze, so its species row carries the SHORT
    canonical name -- the pre-freeze species stage renamed it, keeping the
    database's own spelling in ``raw_name``. At mapping time, which is where
    this card's code runs, the row still holds that raw spelling and has no
    aliases yet. Reconstructed from the fixture's own recorded ``raw_name``.
    """
    donor = _donor_row(payload)
    raw_name = str(donor["raw_name"])
    donor["name"] = raw_name
    donor.pop("aliases", None)
    donor.pop("species_canonicalization", None)
    return raw_name


def test_g9_a2_the_database_row_is_reduced_to_species_rank_before_asking() -> None:
    """G9 BASE-FAILURE PROOF (A2) -- the shape production really has.

    The database row reaches this stage under its own strain-qualified name, so
    the question put to NCBI is the species-rank binomial that name reduces to,
    never the strain. The stub answers ONLY for that binomial: if the code asked
    for the strain name it would get nothing and the row would stay unresolved,
    which is what happens at 4d417e4f.
    """
    payload = _pathway_header(_payload())
    organism = str(_unresolved_row(payload)["name"])
    raw_name = _as_species_hydration_left_it(payload)
    binomial = map_ids._binomial_from_organism(raw_name)
    assert binomial != raw_name  # the fixture really is strain-qualified

    client = _StubTaxonomyClient({binomial: STUB_TAXID})
    report = backfill_species_taxonomy(payload, client=client, enable_ncbi=True)

    row = _unresolved_row_by_name(payload, organism)
    assert row.get("taxonomy_id") == STUB_TAXID
    assert row.get("classification") == STUB_CLASSIFICATION
    meta = row["mapping_meta"]["taxonomy_backfill"]
    assert meta["resolved_as"] == binomial
    assert meta["donors"] == [raw_name]
    assert report["species_creation"]["resolved"] == 1
    assert _gate_error_codes(payload) == []
    # The strain-qualified name was never the question.
    assert not any(term.startswith(raw_name) for term in client.terms)


# ===========================================================================
# B -- FAIL CLOSED: the authority says nothing.
# ===========================================================================


def test_g9_b_no_match_leaves_the_row_unresolved_and_the_gate_refusing() -> None:
    """NEW-CAPABILITY ACCEPTANCE (B) / REGRESSION GUARD.

    NCBI answers for nothing. Nothing is invented, nothing is borrowed from the
    database row beside it, and the gate refuses exactly as it does today. Holds
    at base and must keep holding.
    """
    payload = _pathway_header(_payload())
    organism = str(_unresolved_row(payload)["name"])
    client = _StubTaxonomyClient({})

    report = backfill_species_taxonomy(payload, client=client, enable_ncbi=True)

    row = _unresolved_row_by_name(payload, organism)
    assert not str(row.get("taxonomy_id") or "").strip()
    assert not str(row.get("classification") or "").strip()
    assert "taxonomy_backfill" not in row.get("mapping_meta", {})
    assert "species_creation" not in report
    assert organism in report["unresolved"]
    assert set(_gate_error_codes(payload)) == SPECIES_GATE_CODES


# ===========================================================================
# C -- FAIL CLOSED: the authority could not be reached. Nothing is cached.
# ===========================================================================


@pytest.mark.parametrize("mode", ["raise", "http_error"])
def test_c_transport_failure_resolves_nothing_and_caches_nothing(mode: str, tmp_path: Path) -> None:
    """NEW-CAPABILITY ACCEPTANCE (C).

    A dead network and a 429 are not evidence that a species does not exist
    (PRODUCT_CONTRACT S8), so neither may be memoized: one outage must not become
    a permanent silent "no such species". The row stays unresolved and the gate
    still refuses.
    """
    payload = _pathway_header(_payload())
    organism = str(_unresolved_row(payload)["name"])
    cache = MappingCache(tmp_path / "id_mapping_cache.json", enabled=True)
    client = _StubTaxonomyClient(
        {str(_donor_row(payload).get("name") or ""): STUB_TAXID}, mode=mode
    )

    report = backfill_species_taxonomy(payload, client=client, enable_ncbi=True, cache=cache)

    row = _unresolved_row_by_name(payload, organism)
    assert not str(row.get("taxonomy_id") or "").strip()
    assert not str(row.get("classification") or "").strip()
    assert "species_creation" not in report
    assert cache.data.get(map_ids._SPECIES_TAXONOMY_CACHE_SECTION, {}) == {}
    assert not (tmp_path / "id_mapping_cache.json").exists()
    assert set(_gate_error_codes(payload)) == SPECIES_GATE_CODES


def test_c2_a_resolved_answer_is_cached_under_the_organism_name(tmp_path: Path) -> None:
    """NEW-CAPABILITY ACCEPTANCE (C2).

    The counterpart of the rule above: an answer the authority actually gave IS
    memoized, in the EXISTING MappingCache, under a key carrying the organism
    name it was asked about.
    """
    payload = _pathway_header(_payload())
    donor_name = str(_donor_row(payload)["name"])
    cache = MappingCache(tmp_path / "id_mapping_cache.json", enabled=True)

    backfill_species_taxonomy(payload, client=_stub_for_donor(payload), enable_ncbi=True, cache=cache)

    section = cache.data.get(map_ids._SPECIES_TAXONOMY_CACHE_SECTION, {})
    assert len(section) == 1
    key, entry = next(iter(section.items()))
    assert _normalize_name(donor_name) in key
    assert entry["taxonomy_id"] == STUB_TAXID
    assert entry["classification"] == STUB_CLASSIFICATION
    assert entry["source"] == "ncbi"

    # A second run served from the memo asks the network nothing further and
    # still resolves to the same id.
    replay = _pathway_header(_payload())
    silent = _StubTaxonomyClient({})
    backfill_species_taxonomy(replay, client=silent, enable_ncbi=True, cache=cache)
    organism = str(_unresolved_row(_payload())["name"])
    assert _unresolved_row_by_name(replay, organism).get("taxonomy_id") == STUB_TAXID


# ===========================================================================
# D -- NO FABRICATION.
# ===========================================================================


def test_d1_no_taxonomy_id_appears_anywhere_without_an_ncbi_answer() -> None:
    """NEW-CAPABILITY ACCEPTANCE (D). With the authority silent, the set of
    taxonomy ids in the payload is exactly the set it started with."""
    payload = _payload()
    before = sorted(str(row.get("taxonomy_id") or "") for row in _species_rows(payload))
    backfill_species_taxonomy(payload, client=_StubTaxonomyClient({}), enable_ncbi=True)
    after = sorted(str(row.get("taxonomy_id") or "") for row in _species_rows(payload))
    assert after == before


def test_d2_the_created_id_is_verbatim_what_the_authority_returned() -> None:
    """NEW-CAPABILITY ACCEPTANCE (D). The id is carried through, not derived: a
    different stub answer produces a different row value and nothing else."""
    payload = _pathway_header(_payload())
    organism = str(_unresolved_row(payload)["name"])
    backfill_species_taxonomy(
        payload, client=_stub_for_donor(payload, ALT_TAXID), enable_ncbi=True
    )
    assert _unresolved_row_by_name(payload, organism).get("taxonomy_id") == ALT_TAXID


def test_d3_offline_the_pass_creates_nothing() -> None:
    """REGRESSION GUARD (D). With NCBI disabled, or with no client at all, there
    is no offline route to a taxonomy id for an unmatched organism -- the donor's
    own id is never copied onto it."""
    for kwargs in ({"client": None, "enable_ncbi": False}, {"client": None, "enable_ncbi": True}):
        payload = _pathway_header(_payload())
        organism = str(_unresolved_row(payload)["name"])
        report = backfill_species_taxonomy(payload, **kwargs)  # type: ignore[arg-type]
        row = _unresolved_row_by_name(payload, organism)
        assert not str(row.get("taxonomy_id") or "").strip()
        assert "species_creation" not in report
        assert set(_gate_error_codes(payload)) == SPECIES_GATE_CODES


def test_d4_a_disagreeing_or_unqualified_assertion_is_refused() -> None:
    """NEW-CAPABILITY ACCEPTANCE (D). The eligibility rules, exercised directly
    on the synonym resolver so each refusal is visible on its own. This one
    names a symbol this card adds, so it cannot run at the base SHA -- the
    base-failure proofs are A and A2, which do not.

    Every case is built from the fixture's own two names.
    """
    payload = _payload()
    blocking = str(_unresolved_row(payload)["name"])
    donor = _donor_row(payload)
    donor_name = str(donor["name"])

    def rows(*entries: Dict[str, Any]) -> List[Any]:
        return list(entries)

    good = {"name": donor_name, "pathbank_species_id": donor.get("pathbank_species_id"),
            "common_name": blocking}
    assert map_ids._db_asserted_species_synonym(rows(good), blocking)["term"] == donor_name

    # (a) the asserting row is not a database record -> it asserts nothing.
    not_a_record = {"name": donor_name, "common_name": blocking}
    assert map_ids._db_asserted_species_synonym(rows(not_a_record), blocking) == {}

    # (b) the asserting row's own name is rank-qualified -- which is how the
    #     database's own name reaches this stage -- so it is REDUCED to its
    #     species-rank binomial and the question is asked at that rank.
    strain = dict(good, name=f"{donor_name} str. XYZ")
    assert map_ids._db_asserted_species_synonym(rows(strain), blocking)["term"] == donor_name

    # (c) the blocking name is itself rank-qualified -> a species-rank answer is
    #     not the taxon that name states, so nothing is resolved.
    assert map_ids._db_asserted_species_synonym(rows(good), f"{blocking} str. XYZ") == {}

    # (d) two database records claim the same spelling for two different
    #     organisms -> we may not pick, so nothing is resolved.
    blocking_epithet = blocking.split(" ")[1]
    other = dict(good, name=f"Zymomonas {blocking_epithet}", pathbank_species_id=999999)
    assert map_ids._db_asserted_species_synonym(rows(good, other), blocking) == {}

    # (e) nobody asserts it.
    assert map_ids._db_asserted_species_synonym(rows(dict(good, common_name="")), blocking) == {}

    # (f) not a genus-level synonym: a different specific epithet under any
    #     genus is a different organism, whatever the common_name says.
    donor_genus = donor_name.split(" ")[0]
    assert map_ids._db_asserted_species_synonym(
        rows(dict(good, name=f"{donor_genus} zzzzzz")), blocking
    ) == {}

    # (g) the same genus and the same epithet is the blocking name itself, not a
    #     synonym of it.
    assert map_ids._db_asserted_species_synonym(rows(dict(good, name=blocking)), blocking) == {}

    # (h) a donor with no binomial at all resolves nothing.
    assert map_ids._db_asserted_species_synonym(rows(dict(good, name="Bacteria")), blocking) == {}


def test_d5_a_partly_resolved_row_is_never_completed_from_another_name() -> None:
    """REGRESSION GUARD (D). A row that already claims a taxon keeps its own
    claim: pairing that id with a classification fetched for a different name
    would state a combination no source ever stated."""
    payload = _pathway_header(_payload())
    organism = str(_unresolved_row(payload)["name"])
    client = _stub_for_donor(payload)
    _unresolved_row_by_name(payload, organism)["taxonomy_id"] = "424242"

    report = backfill_species_taxonomy(payload, client=client, enable_ncbi=True)

    row = _unresolved_row_by_name(payload, organism)
    assert row.get("taxonomy_id") == "424242"
    assert not str(row.get("classification") or "").strip()
    assert "species_creation" not in report
    assert "species_missing_classification" in _gate_error_codes(payload)


# ===========================================================================
# E -- THE GATE IS NOT WEAKENED.
# ===========================================================================


def test_e1_gate_still_refuses_a_species_genuinely_lacking_taxonomy() -> None:
    """REGRESSION GUARD (E). No backfill runs at all here: the gate's own
    refusal, on the committed payload, is unchanged."""
    payload = _pathway_header(_payload())
    assert set(_gate_error_codes(payload)) == SPECIES_GATE_CODES
    assert validate_required_pwml_contract(copy.deepcopy(payload), strict_db=True)["ok"] is False


def test_e2_gate_still_refuses_when_no_database_row_asserts_the_organism() -> None:
    """REGRESSION GUARD (E). Remove the database row that carries the synonym and
    the fallback has nothing authoritative to ask about: the organism stays
    unresolved and the export is refused, which is the correct outcome."""
    payload = _pathway_header(_payload())
    blocking = _unresolved_row(payload)
    organism = str(blocking["name"])
    donor_name = str(_donor_row(payload)["name"])
    payload["entities"]["species"] = [
        row for row in _species_rows(payload) if str(row.get("name") or "") != donor_name
    ]

    report = backfill_species_taxonomy(
        payload, client=_StubTaxonomyClient({donor_name: STUB_TAXID}), enable_ncbi=True
    )

    assert not str(_unresolved_row_by_name(payload, organism).get("taxonomy_id") or "").strip()
    assert "species_creation" not in report
    assert set(_gate_error_codes(payload)) == SPECIES_GATE_CODES


def test_e3_a_row_with_partial_taxonomy_still_fails_the_gate() -> None:
    """REGRESSION GUARD (E). Half the facts is still a refusal -- an id without a
    classification, and a classification without an id."""
    payload = _pathway_header(_payload())
    organism = str(_unresolved_row(payload)["name"])

    only_id = copy.deepcopy(payload)
    _unresolved_row_by_name(only_id, organism)["taxonomy_id"] = STUB_TAXID
    assert _gate_error_codes(only_id) == ["species_missing_classification"]

    only_class = copy.deepcopy(payload)
    _unresolved_row_by_name(only_class, organism)["classification"] = STUB_CLASSIFICATION
    assert _gate_error_codes(only_class) == ["species_missing_taxonomy"]

    bad_id = copy.deepcopy(payload)
    row = _unresolved_row_by_name(bad_id, organism)
    row["taxonomy_id"] = "not-a-number"
    row["classification"] = STUB_CLASSIFICATION
    assert _gate_error_codes(bad_id) == ["species_missing_taxonomy"]


# ===========================================================================
# F -- the refactored NCBI helpers behave exactly as before.
# ===========================================================================


def test_f_existing_ncbi_helpers_keep_their_contract() -> None:
    """REGRESSION GUARD (F). ``_ncbi_taxonomy_lookup`` and ``_ncbi_esearch_taxid``
    keep the shapes their existing callers rely on: a resolved record, ``{}`` on
    no match, ``{}`` on transport failure. Holds at base and must keep holding."""
    payload = _payload()
    donor_name = str(_donor_row(payload)["name"])

    resolved = _ncbi_taxonomy_lookup(_StubTaxonomyClient({donor_name: STUB_TAXID}), donor_name)
    assert resolved == {"taxonomy_id": STUB_TAXID, "classification": STUB_CLASSIFICATION}
    assert _ncbi_taxonomy_lookup(_StubTaxonomyClient({}), donor_name) == {}
    assert _ncbi_taxonomy_lookup(_StubTaxonomyClient({}, mode="raise"), donor_name) == {}
    assert _ncbi_taxonomy_lookup(_StubTaxonomyClient({}, mode="http_error"), donor_name) == {}
    assert _ncbi_taxonomy_lookup(_StubTaxonomyClient({donor_name: STUB_TAXID}), "") == {}

    client = _StubTaxonomyClient({donor_name: STUB_TAXID})
    assert _ncbi_esearch_taxid(client, f"{donor_name}[Scientific Name]") == STUB_TAXID
    assert _ncbi_esearch_taxid(_StubTaxonomyClient({}), donor_name) == ""
    assert _ncbi_esearch_taxid(_StubTaxonomyClient({}, mode="raise"), donor_name) == ""

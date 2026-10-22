"""ORCH-762: protein identity by taxonomy, never across a subspecies (FX1-FX6, D-106, D-108).

Every replay below drives a committed leg payload through the app's own mapping sequence
(Stage-2 ``map_payload`` -> ``normalize_process_payload`` -> Stage-6 ``map_payload``) with
the outside world replayed: UniProt/UniParc/UniProt-taxonomy answers recorded ONCE live
(``tests/fixtures/orch762/uniprot_responses.json``), every other host mocked, no PathBank
database, no model. See ``tests/orch762_harness.py`` and ``tests/fixtures/orch762/README.md``.

**G9 -- CORRECTION of observable behaviour** (``test_g9_*``). On base ``493c1a10`` each
fails on a behaviour, never on a missing symbol (they import nothing this lane adds; keys
are spelled out):

* KF147 Men enzymes found no record (free-text query -> E. coli/human -> species mismatch);
  they now resolve to the same-subspecies (IL1403) record (FX1, FX2).
* MenH tied with the MenG methyltransferase P49016 through the shared synonym ``menH``; it
  now resolves to its own record Q9CHK1 (FX4).
* MenA shipped A2RHR6 (subsp. *cremoris* MG1363) as VERIFIED when the organism read
  "Lactococcus lactis" -- an identity across the subspecies boundary D-108 forbids; it no
  longer does, at Stage 2, Stage 6 or the Phase-2 fallback.
* EntB lost its grounded P0ADI4 to the inferred compound ``holo-EntB`` (FX6).
* A cached free-text answer (the leg's own cache rows) replayed over a fresh lookup (FX1 key).
* PathBank's species tier stopped at ``species_not_found`` for "(strain KF147)" (FX3).

**NEW ACCEPTANCE TESTS** (``test_new_*``, no base failure claimed): the exact-sequence route
through UniParc (FX5), the borrowed-identity flag with its source strain (D-108 § 3), its
absence from every LLM-bound input (D-104 amendment 2), and the bound itself, tried from
every door.

**PRESERVATION** (``test_preservation_*``): pass at base and at tip.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT / "src", ROOT / "tests"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import orch762_harness as H  # noqa: E402
from t2pw.mapping import map_ids  # noqa: E402

BORROWED_KEY = "borrowed_protein_identities"  # spelled out (G9: no import of a new symbol)
IL1403 = "Lactococcus lactis subsp. lactis (strain IL1403)"
#: FX1 (D-108: the same-subspecies record of the same gene), ORCH-761 § 3.2 / § 7.
SAME_SUBSPECIES = {"MenF": "Q9CHJ9", "MenD": "Q9CHK0", "MenC": "Q9CHK4", "MenE": "Q9CHK3", "MenB": "Q9CHK2", "MenG": "P49016"}
#: Every UniProtKB record of the cremoris side (MG1363 / SK11) any lookup ever returned here.
CREMORIS = {"A2RHR6", "A2RM72", "A2RM73", "A2RM74", "A2RM75", "A2RM76", "A2RM77", "A2RJA5", "Q030P1"}


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Tuple[Dict[str, Any], Any, H.ReplayHttp]]:
    out: Dict[str, Tuple[Dict[str, Any], Any, H.ReplayHttp]] = {}
    for name, scenario in H.SCENARIOS.items():
        http = H.ReplayHttp()
        result, doors = scenario(http, tmp_path_factory.mktemp(name))
        out[name] = (result, doors, http)
    return out


def _payload(runs: Dict[str, Any], name: str, stage: str) -> Dict[str, Any]:
    result, _, http = runs[name]
    assert http.unrecorded == [], f"{name}: a UniProt query with no recorded answer: {http.unrecorded[:3]}"
    return result[stage]["payload"]


def _rows(payload: Dict[str, Any], bucket: str = "proteins") -> Dict[str, Dict[str, Any]]:
    return {str(row.get("name")): row for row in payload["entities"].get(bucket) or [] if isinstance(row, dict)}


def _verified(row: Dict[str, Any]) -> bool:
    return (row.get("mapping_meta") or {}).get("identity_status") == "verified"


# ── G9 ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("stage", ["stage2", "stage6"])
def test_g9_kf147_men_enzymes_resolve_to_their_same_subspecies_records(runs: Dict[str, Any], stage: str) -> None:
    rows = _rows(_payload(runs, "kf147", stage))
    shipped = {name: str((rows.get(name, {}).get("mapped_ids") or {}).get("uniprot") or "") for name in SAME_SUBSPECIES}
    assert shipped == SAME_SUBSPECIES
    assert all(_verified(rows[name]) for name in SAME_SUBSPECIES)


@pytest.mark.parametrize("scenario", ["kf147", "lactis_species"])
def test_g9_menh_resolves_to_its_own_record_not_the_menG_methyltransferase(runs: Dict[str, Any], scenario: str) -> None:
    for stage in ("stage2", "stage6"):
        menh = _rows(_payload(runs, scenario, stage)).get("MenH", {})
        assert (menh.get("mapped_ids") or {}).get("uniprot") == "Q9CHK1"
        assert _verified(menh)


@pytest.mark.parametrize("stage", ["stage2", "stage6"])
def test_g9_mena_no_longer_takes_the_cremoris_record(runs: Dict[str, Any], stage: str) -> None:
    payload = _payload(runs, "lactis_species", stage)
    mena = _rows(payload).get("MenA") or {}
    assert (mena.get("mapped_ids") or {}).get("uniprot") != "A2RHR6"
    shipped = {
        str((row.get("mapped_ids") or {}).get("uniprot") or "")
        for bucket in ("proteins", "compounds")
        for row in payload["entities"].get(bucket) or []
        if isinstance(row, dict)
    }
    assert not (shipped & CREMORIS), "an identity from across the subspecies boundary shipped"


def test_g9_entb_keeps_its_grounded_accession_on_the_orch755_shape(runs: Dict[str, Any]) -> None:
    payload = _payload(runs, "enterobactin_holo", "stage6")
    entb = _rows(payload)["EntB"]
    assert (entb.get("mapped_ids") or {}).get("uniprot") == "P0ADI4"
    assert "uniprot" not in ((entb.get("mapping_meta") or {}).get("rejected_mapped_ids") or {})
    holo = _rows(payload, "compounds")["holo-EntB"]
    claimed = dict(holo.get("mapped_ids") or {}, **((holo.get("mapping_meta") or {}).get("rejected_mapped_ids") or {}))
    assert "uniprot" not in claimed, "the inferred compound still claims a protein accession"


def test_g9_a_cached_free_text_answer_does_not_replay(tmp_path: Path) -> None:
    """The leg's own cache rows (E. coli/human best efforts, PathBank species_not_found)."""

    seeded = H.load_fixture("kf147_leg_cache_entries.json")
    (tmp_path / "id_mapping_cache.json").write_text(json.dumps(seeded), encoding="utf-8")
    http = H.ReplayHttp()
    with H.Doors(http):
        payload = H.stage2(H.load_fixture("menaquinone_kf147_merged.json"), tmp_path)["payload"]
    assert http.unrecorded == []
    assert H.shipped_uniprot(payload)["MenF"] == "Q9CHJ9"


class _SpeciesTable(map_ids.PathBankDbResolver):
    """A PathBank stand-in holding three species rows and no protein rows (ORCH-761 § 3.1)."""

    SPECIES = [
        {"id": 10733, "name": "Lactococcus lactis subsp. lactis (strain KF147)", "common_name": "", "taxonomy_id": 684738},
        {"id": 1030, "name": "Lactococcus lactis", "common_name": "", "taxonomy_id": 1358},
        {"id": 1206, "name": "Lactococcus lactis subsp. cremoris (strain MG1363)", "common_name": "", "taxonomy_id": 416870},
    ]

    def __init__(self) -> None:  # no connection, no driver import
        self._driver = object()
        self._conn = None
        self.last_error = ""
        self._species_meta = {}
        self._species_meta_lookups = set()
        self.queries: List[Tuple[str, Tuple[Any, ...]]] = []

    def _query(self, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        self.queries.append((sql, tuple(params)))
        if "FROM species" not in sql:
            return []

        def hit(row: Dict[str, Any]) -> bool:
            for value in params:
                text = str(value).casefold()
                if text.startswith("%") and text.endswith("%"):
                    if text[1:-1] and text[1:-1] in row["name"].casefold():
                        return True
                elif text in (row["name"].casefold(), str(row["taxonomy_id"])):
                    return True
            return False

        return [dict(row) for row in self.SPECIES if hit(row)]

    def close(self) -> None:
        return None


@pytest.mark.parametrize("with_taxonomy_id", [True, False], ids=["taxonomy_id", "strain_spelling"])
def test_g9_pathbank_species_tier_finds_the_strain(tmp_path: Path, with_taxonomy_id: bool) -> None:
    payload = H.load_fixture("menaquinone_kf147_merged.json")
    if not with_taxonomy_id:
        for row in payload["entities"]["species"]:
            row.pop("taxonomy_id", None)
    resolver = _SpeciesTable()
    with H.Doors(H.ReplayHttp()):
        from unittest.mock import patch

        with patch.object(map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: resolver)):
            mapped = map_ids.map_payload(
                deepcopy(payload), cache_path=tmp_path / "cache.json", id_source="db",
                allow_complex_wrapper_creation=False, allow_structural_cleanup=False,
            )["payload"]
    menf = _rows(mapped)["MenF"]["mapping_meta"]["resolution"]
    assert menf.get("issue") != "species_not_found", menf
    protein_sql = [params for sql, params in resolver.queries if "FROM proteins" in sql]
    assert protein_sql and all(10733 in params for params in protein_sql)


# ── NEW ACCEPTANCE TESTS ─────────────────────────────────────────────────────────────────


def test_new_exact_sequence_route_returns_the_identical_records_accession(runs: Dict[str, Any]) -> None:
    """FX5: KF147 MenB = UniParc UPI00000C68C6 = the active UniProtKB record Q9CHK2 (IL1403)."""

    menb = _rows(_payload(runs, "kf147", "stage2"))["MenB"]
    judged = menb["mapping_meta"]["identity_verdict"]["judged_candidate"]
    assert (judged["accession"], judged["identity_route"], judged["uniparc_id"]) == ("Q9CHK2", "exact_sequence_uniparc", "UPI00000C68C6")
    record = {row["name"]: row for row in _payload(runs, "kf147", "stage6")[BORROWED_KEY]}["MenB"]
    assert record["route"] == "exact_sequence_uniparc"


def test_new_exact_sequence_route_refuses_paralogs(runs: Dict[str, Any]) -> None:
    """KF147 carries three menA genes and two menF genes; no sequence is chosen among them."""

    payload = _payload(runs, "kf147", "stage2")
    assert "Q9CF18" not in H.shipped_uniprot(payload).values()  # menA_3's identical record
    menf = _rows(payload)["MenF"]["mapping_meta"]["identity_verdict"]["judged_candidate"]
    assert menf.get("identity_route") is None and menf["accession"] == "Q9CHJ9"


def _synthetic_http(uniprot: Dict[str, List[Dict[str, Any]]], uniparc: Dict[str, Any] = None, lineage: Dict[str, Any] = None) -> Any:
    """Answers from tables keyed by a substring of the query; everything else empty."""

    taxa = lineage or {
        "684738": {"taxonId": 684738, "scientificName": "Lactococcus lactis subsp. lactis (strain KF147)", "rank": "strain",
                   "lineage": [{"taxonId": 1360, "scientificName": "Lactococcus lactis subsp. lactis", "rank": "subspecies"},
                               {"taxonId": 1358, "scientificName": "Lactococcus lactis", "rank": "species"}]},
    }

    class Http(H.ReplayHttp):
        def get(self, url: str, params: Any = None, headers: Any = None) -> H.FakeResponse:
            key = H.request_key(url, params)
            self.calls.append(key)
            if "/taxonomy/" in url:
                body = taxa.get(url.rsplit("/", 1)[1])
                return H.FakeResponse(200, body) if body else H.FakeResponse(404, {})
            query = str((params or {}).get("query") or "")
            if "/uniparc/" in url:
                for needle, body in (uniparc or {}).items():
                    if needle in url or needle in query:
                        return H.FakeResponse(200, body)
                return H.FakeResponse(200, {"results": []})
            if "/uniprotkb/" in url:
                for needle, rows in uniprot.items():
                    if needle in query:
                        return H.FakeResponse(200, {"results": rows})
                return H.FakeResponse(200, {"results": []})
            self.mocked.append(key)
            return H.FakeResponse(404, {})

    return Http(responses={})


def _entry(acc: str, gene: str, organism: str, taxon: int, *, reviewed: bool = False, name: str = "isochorismate synthase") -> Dict[str, Any]:
    return {
        "primaryAccession": acc,
        "entryType": "UniProtKB reviewed (Swiss-Prot)" if reviewed else "UniProtKB unreviewed (TrEMBL)",
        "proteinDescription": {"recommendedName": {"fullName": {"value": name}}},
        "genes": [{"geneName": {"value": gene}}],
        "organism": {"scientificName": organism, "taxonId": taxon},
    }


MG1363 = "Lactococcus lactis subsp. cremoris (strain MG1363)"


def _one_protein(tmp_path: Path, http: Any, organism: str = "Lactococcus lactis subsp. lactis KF147", taxonomy_id: str = "684738") -> Dict[str, Any]:
    payload = {
        "entities": {
            "species": [{"name": organism, "taxonomy_id": taxonomy_id, "classification": "Prokaryote"}],
            "proteins": [{"name": "MenF", "organism": organism}],
            "compounds": [{"name": "chorismate"}, {"name": "isochorismate"}],
        },
        "processes": {"reactions": [{"name": "r", "inputs": ["chorismate"], "outputs": ["isochorismate"], "enzymes": [{"protein": "MenF"}]}]},
    }
    with H.Doors(http):
        return H.stage2(payload, tmp_path)["payload"]


def test_new_the_climb_never_passes_the_subspecies(tmp_path: Path) -> None:
    """A bound-tier answer holding a cremoris record (a lying server, a stale index) is refused."""

    http = _synthetic_http({"taxonomy_id:1360": [_entry("A2RM72", "menF", MG1363, 416870)]})
    payload = _one_protein(tmp_path, http)
    assert H.shipped_uniprot(payload)["MenF"] == ""
    verdict = _rows(payload)["MenF"]["mapping_meta"].get("identity_verdict") or {}
    assert verdict.get("reason") == "species_mismatch"


def test_new_a_scoped_lookup_opens_no_organism_free_door(runs: Dict[str, Any]) -> None:
    for scenario in ("kf147", "lactis_species"):
        result, doors, http = runs[scenario]
        queries = http.uniprot_queries()
        assert queries and all("organism_id%3A" in q or "taxonomy_id%3A" in q for q in queries), scenario
        assert doors.ai_calls == [], "the model alias tier is not consulted under a taxonomy scope"


def test_new_a_stale_candidate_list_is_not_promoted_under_a_scope(tmp_path: Path) -> None:
    """A row arriving with the free-text era's candidate list (A2RHR6 first, as the 2026-10-20
    leg wrote it) is not promoted back into a match: only candidates found under this scope are."""

    payload = H.load_fixture("menaquinone_lactis_species.json")
    mena = next(row for row in payload["entities"]["proteins"] if row["name"] == "MenA")
    mena["mapping_meta"] = {"candidates": [{
        "accession": "A2RHR6", "protein_name": "Prenyltransferase, UbiA family", "gene_names": ["menA"],
        "primary_gene_names": ["menA"], "organism": MG1363, "taxonomy_id": "416870", "reviewed": False,
        "score": 0.8, "matched_query": "MenA", "alias_source": "primary_name"}]}
    http = H.ReplayHttp()
    with H.Doors(http):
        mapped = H.stage2(payload, tmp_path)["payload"]
    assert http.unrecorded == []
    assert H.shipped_uniprot(mapped)["MenA"] == ""


def test_new_a_strain_with_no_known_subspecies_takes_no_subspecies_record(tmp_path: Path) -> None:
    """D-108 when the strain's lineage names no subspecies: the species is the bound, and a
    record filed under one of its subspecies may be across a boundary nobody can see."""

    lineage = {"999001": {"taxonId": 999001, "scientificName": "Lactococcus lactis (strain Z9)", "rank": "strain",
                          "lineage": [{"taxonId": 1358, "scientificName": "Lactococcus lactis", "rank": "species"}]}}
    http = _synthetic_http({"taxonomy_id:1358": [_entry("Q9CHJ9", "menF", IL1403, 272623)]}, lineage=lineage)
    payload = _one_protein(tmp_path, http, organism="Lactococcus lactis Z9", taxonomy_id="999001")
    assert H.shipped_uniprot(payload)["MenF"] == ""
    http = _synthetic_http({"taxonomy_id:1358": [_entry("Q0ZZZ1", "menF", "Lactococcus lactis (strain Y2)", 999002)]}, lineage=lineage)
    payload = _one_protein(tmp_path / "b", http, organism="Lactococcus lactis Z9", taxonomy_id="999001")
    assert H.shipped_uniprot(payload)["MenF"] == "Q0ZZZ1"


def test_new_exact_sequence_route_stays_inside_the_subspecies(tmp_path: Path) -> None:
    """The identical sequence's only active record is MG1363's: the bound clause returns nothing."""

    uniparc = {
        "gene:MenF": {"results": [{"uniParcId": "UPI0000TEST01"}]},
        "UPI0000TEST01": {"uniParcCrossReferences": [{"database": "UniProtKB/TrEMBL", "id": "A2RM72", "active": True,
                                                       "organism": {"taxonId": 416870}}]},
    }
    http = _synthetic_http({}, uniparc=uniparc)
    payload = _one_protein(tmp_path, http)
    assert H.shipped_uniprot(payload)["MenF"] == ""
    asked = [q for q in http.uniprot_queries() if "accession%3AA2RM72" in q]
    assert asked and all("taxonomy_id%3A1360" in q for q in asked)


def test_new_gene_synonym_tie_break_needs_a_different_reaction(tmp_path: Path) -> None:
    """FX4 fires only when the payload's other enzyme does something else."""

    rows = [_entry("P49016", "menG", IL1403, 272623, reviewed=True, name="Demethylmenaquinone methyltransferase"),
            _entry("Q9CHK1", "menX", IL1403, 272623, name="SHCHC synthase")]
    for row in rows:
        row["genes"][0]["synonyms"] = [{"value": "menH"}]
    payload = {
        "entities": {
            "species": [{"name": "Lactococcus lactis subsp. lactis KF147", "taxonomy_id": "684738", "classification": "Prokaryote"}],
            "proteins": [{"name": "MenH", "organism": "Lactococcus lactis subsp. lactis KF147"},
                         {"name": "MenG", "organism": "Lactococcus lactis subsp. lactis KF147"}],
            "compounds": [{"name": "SEPHCHC"}, {"name": "SHCHC"}],
        },
        # MenG is given MenH's own reaction: no different function, so no tie-break.
        "processes": {"reactions": [{"name": "r", "inputs": ["SEPHCHC"], "outputs": ["SHCHC"],
                                     "enzymes": [{"protein": "MenH"}, {"protein": "MenG"}]}]},
    }
    with H.Doors(_synthetic_http({'gene:"MenH") AND taxonomy_id:1360': rows})):
        mapped = H.stage2(payload, tmp_path)["payload"]
    assert H.shipped_uniprot(mapped)["MenH"] == ""


def test_new_borrowed_identities_are_flagged_with_their_source_strain(runs: Dict[str, Any]) -> None:
    result, _, _ = runs["kf147"]
    for stage in ("stage2", "stage6"):
        payload, report = result[stage]["payload"], result[stage]["report"]
        flags = {row["name"]: row["flag"] for row in payload[BORROWED_KEY]}
        expected = dict(SAME_SUBSPECIES, MenH="Q9CHK1")
        assert flags == {name: f"UniProt {acc}, from strain IL1403" for name, acc in expected.items()}
        assert report[BORROWED_KEY] == payload[BORROWED_KEY]
        for row in payload[BORROWED_KEY]:
            assert (row["source_taxonomy_id"], row["requested_taxonomy_id"], row["bound_taxonomy_id"]) == ("272623", "684738", "1360")
    # A species-level request is answered inside the organism it names: nothing is borrowed.
    assert BORROWED_KEY not in runs["lactis_species"][0]["stage6"]["payload"]
    assert BORROWED_KEY not in runs["enterobactin_holo"][0]["stage6"]["payload"]


def test_new_reaction_evidence_report_shows_the_flag(runs: Dict[str, Any]) -> None:
    from t2pw.app.reaction_evidence import build_report, render_text

    text = render_text(build_report({"final_mapped.json": _payload(runs, "kf147", "stage6")}))
    # The enzyme a reaction names is the Stage-6 wrapper; the flag is its component's.
    assert "MenB complex: unresolved; no identifier recorded; borrowed identity: MenB: UniProt Q9CHK2, from strain IL1403" in text
    assert "borrowed identity: MenH: UniProt Q9CHK1, from strain IL1403" in text


# -- the flag never reaches a model (D-104 amendment 2, D-108 § 3) --------------------------

_FLAG_TEXT = ("from strain IL1403", BORROWED_KEY)


def _llm_requests(payload: Dict[str, Any], report: Dict[str, Any], tmp_path: Path, wire: Any) -> Dict[str, List[str]]:
    """Every payload-reading LLM call site, driven with ``payload``; the serialized requests."""

    from t2pw.curation import audit_json_llm, completeness_audit, gap_resolver, interactive_curator, pathway_curator
    from t2pw.pipeline import localized_repair
    from t2pw.sbml.overwatch import run_sbml_overwatch

    def quiet(call: Any) -> None:
        try:
            call()
        except Exception:  # noqa: BLE001 - the request is what is under test
            pass

    drivers = {
        "audit": lambda: audit_json_llm.audit_payload(deepcopy(payload), use_llm=True),
        "completeness": lambda: completeness_audit.run_final_completeness_audit(
            source_text="MenB catalyzes OSB-CoA to DHNA-CoA.", locked_manifest=[{"reaction_name": "MenB reaction"}],
            final_pathway_json=deepcopy(payload)),
        "gap.species": lambda: gap_resolver.infer_entity_species(deepcopy(payload), entity_type="protein", entity_name="MenB"),
        "map_ids.species": lambda: map_ids._infer_species_with_gap_resolver(deepcopy(payload), entity_type="protein", name="MenB", use_llm=True),
        "interactive_curator": lambda: interactive_curator.run_interactive_curator_round(
            deepcopy(payload), b"", "check MenB", mapping_misses=interactive_curator.compact_mapping_misses(deepcopy(report))),
        "localized_repair.rows": lambda: localized_repair.repair_invalid_rows(
            deepcopy(payload), [{"pointer": "/processes/reactions/0", "code": "missing_inputs", "message": "x"}]),
    }

    def files_then(call: Any) -> Any:
        src = tmp_path / "mapped.json"
        src.write_text(json.dumps(payload), encoding="utf-8")
        (tmp_path / "pathway.sbml").write_text("<sbml/>", encoding="utf-8")
        (tmp_path / "validation.json").write_text("{}", encoding="utf-8")
        return call(src)

    drivers["overwatch"] = lambda: files_then(lambda src: run_sbml_overwatch(
        src, tmp_path / "pathway.sbml", tmp_path / "validation.json", tmp_path / "overwatch.json"))
    drivers["curator"] = lambda: files_then(lambda src: pathway_curator.run_pathway_curator(
        src, tmp_path / "curated.json", tmp_path / "curator_report.json", reaction_summary="R1: OSB-CoA -> DHNA-CoA",
        pathway_name="menaquinone", organism="Lactococcus lactis"))
    out: Dict[str, List[str]] = {}
    for name, drive in drivers.items():
        wire.calls.clear()
        wire.reply = "{}" if name != "curator" else "done"
        quiet(drive)
        assert wire.calls, f"{name}: issued no request"
        out[name] = [json.dumps({k: v for k, v in call.items() if k not in ("max_tokens", "timeout")}, sort_keys=True, default=repr)
                     for call in wire.calls]
    return out


def test_new_the_flag_never_reaches_an_llm_input(runs: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_orch757_llm_token_budgets import install_wire

    wire = install_wire(monkeypatch)
    result, _, _ = runs["kf147"]
    payload, report = result["stage6"]["payload"], result["stage6"]["report"]
    assert payload[BORROWED_KEY] and report[BORROWED_KEY]
    stripped = {k: v for k, v in payload.items() if k != BORROWED_KEY}
    stripped_report = {k: v for k, v in report.items() if k != BORROWED_KEY}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with_flag = _llm_requests(payload, report, tmp_path / "a", wire)
    without = _llm_requests(stripped, stripped_report, tmp_path / "b", wire)
    for name, requests in with_flag.items():
        assert not any(marker in request for request in requests for marker in _FLAG_TEXT), name
        assert requests == without[name], f"{name}: the flag changed an LLM input"


def test_new_the_mapping_report_walker_skips_the_flag(runs: Dict[str, Any]) -> None:
    from t2pw.curation.interactive_curator import compact_mapping_misses

    report = runs["kf147"][0]["stage6"]["report"]
    assert report[BORROWED_KEY], "nothing to skip: the report must carry the flag first"
    assert not any(marker in json.dumps(compact_mapping_misses(report)) for marker in _FLAG_TEXT)


# ── PRESERVATION ─────────────────────────────────────────────────────────────────────────


def test_preservation_enterobactin_grounded_identities_unchanged(runs: Dict[str, Any]) -> None:
    expected = {"EntC": "P0AEJ2", "EntA": "P15047", "EntE": "P10378", "EntF": "P11454", "TolC": "P02930", "TonB": "P02929"}
    for stage in ("stage2", "stage6"):
        shipped = H.shipped_uniprot(_payload(runs, "enterobactin_holo", stage))
        assert {name: shipped[name] for name in expected} == expected


def test_preservation_the_uniprot_tier_for_e_coli_judges_the_same_records(runs: Dict[str, Any]) -> None:
    """Without PathBank, E. coli Ent proteins tie across E. coli strains in UniProt at base;
    under the species scope (562) they tie the same way on the same K-12 record. (LDH is left
    out: its judged record moved, from a refused Thermus row to a refused E. coli row.)"""

    judged = {"EntC": "P0AEJ2", "EntB": "P0ADI4", "EntA": "P15047", "EntE": "P10378", "EntF": "P11454",
              "EntD": "P19925", "Fur": "P0A9A9", "TolC": "P02930", "TonB": "P02929", "MenD": "P17109"}
    rows = _rows(_payload(runs, "ecoli_uniprot", "stage2"))
    assert not any(H.shipped_uniprot(_payload(runs, "ecoli_uniprot", "stage2")).values())
    for name, accession in judged.items():
        verdict = rows[name]["mapping_meta"]["identity_verdict"]
        assert (verdict["reason"], verdict["judged_candidate"]["accession"]) == ("ambiguous_insufficient_margin", accession), name


@pytest.mark.parametrize("scenario,paper,count", [("kf147", "PMC12421875", 9), ("lactis_species", "PMC12421875", 9),
                                                  ("enterobactin_holo", "PMC12096016", 5)])
def test_preservation_gold_reactions_survive_the_replay(runs: Dict[str, Any], scenario: str, paper: str, count: int) -> None:
    from t2pw.bench.eval_v2 import reactions as R
    from t2pw.bench.eval_v2.gold import load_supplement

    case = load_supplement()[paper]
    for stage in ("stage2", "stage6"):
        score = R.score_leg(case, _payload(runs, scenario, stage), [], None, expected_species=case.species)
        matched = next(block["gold_reactions_matched"] for block in score.values() if isinstance(block, dict) and "gold_reactions_matched" in block)
        assert len(matched) == count, (stage, matched)


def test_preservation_no_taxonomy_id_keeps_the_free_text_lookup(tmp_path: Path) -> None:
    """Without a taxonomy id the lookup is the base lookup, query for query."""

    payload = H.load_fixture("menaquinone_kf147_merged.json")
    for row in payload["entities"]["species"]:
        row.pop("taxonomy_id", None)
    http = H.ReplayHttp()
    with H.Doors(http):
        mapped = H.stage2(payload, tmp_path)["payload"]
    assert http.unrecorded == []
    queries = http.uniprot_queries()
    assert queries and not any("organism_id%3A" in q or "taxonomy_id%3A" in q for q in queries)
    assert not any(H.shipped_uniprot(mapped).values())
    assert BORROWED_KEY not in mapped


def test_preservation_a_compound_row_that_is_also_an_enzyme_stays_on_the_protein_route(tmp_path: Path) -> None:
    payload = {
        "entities": {"species": [{"name": "Escherichia coli"}], "proteins": [],
                     "compounds": [{"name": "PhoX"}, {"name": "A"}, {"name": "B"}]},
        "element_locations": {"protein_locations": [{"protein": "PhoX", "biological_state": "cytoplasm"}]},
        "processes": {"reactions": [{"name": "r", "inputs": ["A"], "outputs": ["B"], "enzymes": [{"protein": "PhoX"}]}]},
    }
    with H.Doors(H.ReplayHttp(responses={})):
        mapped = H.stage2(payload, tmp_path)["payload"]
    phox = _rows(mapped, "compounds")["PhoX"]["mapping_meta"]
    assert (phox["route"], phox["route_reason"]) == ("protein", "known_protein_like")

"""C-078 / F-099: a name-keyed DB re-resolution may not restore a refused identity.

**G9 REGRESSION PROOF, not a new capability.** Every ``test_g9_*`` is a *content*
assertion about the identifiers that end up on the row -- ``mapped_ids`` and the
scalar columns -- and each fails behaviourally at base ``50420a1``, where the row
comes back renamed and carrying the very namespaces
``mapping_meta.rejected_mapped_ids`` records as refused. No test here asserts that
a symbol is absent.

Run it against a base checkout by pointing ``PYTHONPATH`` there. That is why this
module does **not** do the ``sys.path.insert(ROOT / "src")`` the older test
modules do, which would pin the import to this worktree and make a base run
silently measure the tip (``TEST_MATRIX`` § 0 rule 10).

Nothing here touches a database, a network or a model: a stub replaces the
resolver entirely, and the corpus test reads committed JSON only.

The fixtures are copied from the corpus, not invented. ``2,3-dihydro-2,3-
dihydroxybenzoate`` and PathBank row 41128 are lifted verbatim from
``runs_verify/2026-08-22_2147/papers/PMC12096016/research/final_mapped.json``;
``Fe3+`` / *ferric iron* and PathBank row 8133 from the same file.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

import t2pw.pwml.compound_resolution as cr
from t2pw.pwml.db_resolver import IDENTITY_REFUSED_STATUS
from t2pw.pwml.prefreeze_resolution import _PROVENANCE_FIELDS, resolve_compounds_prefreeze

#: This file lives in ``<repo>/tests``; the corpus lives in ``<repo>/runs_verify``.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The two verification runs C-078 § 2d names. T-105 and T-104.
CORPUS_RUNS = ("runs_verify/2026-08-22_2147", "runs_verify/2026-08-21_2239")

#: The exact seam row, verbatim from the corpus. Four namespaces refused by the
#: mapping stage's name gate; ``mapped_ids`` emptied and the PathBank scalar
#: cleared, which is what drops it into the name-keyed resolver.
REFUSED_ROW: Dict[str, Any] = {
    "name": "2,3-dihydro-2,3-dihydroxybenzoate",
    "class": "compound",
    "raw_name": "2,3-dihydro-2,3-dihydroxybenzoate",
    "mapped_ids": {},
    "mapping_meta": {
        "resolution": {"status": "novel", "issue": "implausible_name_match"},
        "name_gate": {"verdict": "reject", "reason": "no_shared_meaningful_token"},
        "rejected_mapped_ids": {
            "kegg": "C04171",
            "chebi": "CHEBI:15941",
            "pubchem": "23615184",
            "pathbank_compound_id": "40770",
        },
    },
}

#: PathBank compound 41128, verbatim from that row's own ``db_match.candidates``.
#: Carries ``chebi``, ``pubchem`` and an ``id`` -- three of the four refused
#: namespaces -- so admitting it restores them.
DDB_MATCH: Dict[str, Any] = {
    "status": "matched",
    "raw_name": "2,3-dihydro-2,3-dihydroxybenzoate",
    "chosen_rule": "exact_short_name_or_synonym",
    "confidence": 0.70,
    "candidates": [],
    "chosen": {
        "id": 41128,
        "name": "2,3-Dihydro-2,3-dihydroxybenzoic acid",
        "hmdb_id": None,
        "kegg_id": None,
        "pubchem_cid": "23615184",
        "chebi_id": "15941",
        "pwc_id": "PW_C041128",
        "short_name": "DDB",
    },
}

#: PathBank compound 8133, verbatim from the ``Fe3+`` row's ``db_match.chosen``.
FE3_MATCH: Dict[str, Any] = {
    "status": "matched",
    "raw_name": "ferric iron",
    "chosen_rule": "exact_short_name_or_synonym",
    "confidence": 0.70,
    "candidates": [],
    "chosen": {
        "id": 8133,
        "name": "Fe3+",
        "hmdb_id": "HMDB0012943",
        "kegg_id": "C14819",
        "pubchem_cid": "29936",
        "chebi_id": "29034",
        "cas": "20074-52-6",
        "biocyc_id": "CPD-10134",
        "chemspider_id": "27815",
        "pwc_id": "PW_C008133",
        "short_name": "Fe3+",
    },
}

#: ``glycolate`` -> ``Glycolic acid``: the measured extraction-name provenance
#: case ``prefreeze_resolution.py:496-507`` documents. No refusal record.
GLYCOLATE_MATCH: Dict[str, Any] = {
    "status": "matched",
    "raw_name": "glycolate",
    "chosen_rule": "exact_short_name_or_synonym",
    "confidence": 0.70,
    "candidates": [],
    "chosen": {
        "id": 1085,
        "name": "Glycolic acid",
        "hmdb_id": "HMDB0000115",
        "kegg_id": "C00160",
        "pubchem_cid": "757",
        "chebi_id": "17497",
        "pwc_id": "PW_C001085",
        "short_name": "Glycolate",
    },
}

#: Every column ``apply_compound_db_resolution`` stamps an identifier into.
STAMPED = (
    "pathwhiz_id", "pathbank_compound_id", "pw_compound_id", "db_id",
    "kegg_id", "chebi_id", "pubchem_cid", "hmdb_id", "pwc_id",
    "cas", "biocyc_id", "chemspider_id", "drugbank_id",
)

#: The four namespaces ``REFUSED_ROW`` refused.
REFUSED_NAMESPACES = ("kegg", "chebi", "pubchem", "pathbank_compound_id")


# --------------------------------------------------------------------------
# stubs -- no DB, no network
# --------------------------------------------------------------------------


class _StubResolver:
    """Replaces ``PathWhizCompoundResolver``; one canned verdict per row name."""

    def __init__(self, db: Any) -> None:
        self._by_name = db.by_name

    def resolve(self, row: Dict[str, Any]) -> Dict[str, Any]:
        name = str(row.get("name") or "").strip()
        match = self._by_name.get(name)
        if match is None:
            return {
                "status": "unmatched", "raw_name": name, "chosen": None,
                "candidates": [], "chosen_rule": "", "confidence": 0.0,
                "reason": "stub: no row for this name",
            }
        return copy.deepcopy(match)


class _StubDb:
    def __init__(self, by_name: Dict[str, Dict[str, Any]]) -> None:
        self.by_name = by_name

    def available(self) -> bool:
        return True

    @classmethod
    def from_env(cls) -> "_StubDb":  # pragma: no cover - production seam only
        return cls({})


@pytest.fixture(autouse=True)
def _stub_resolver(monkeypatch: Any) -> None:
    print(f"t2pw.pwml.compound_resolution -> {cr.__file__}")
    monkeypatch.setattr(cr, "PathWhizCompoundResolver", _StubResolver)


def _resolve(
    row: Dict[str, Any],
    by_name: Dict[str, Dict[str, Any]],
    *,
    name_index: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    report: Dict[str, Any] = {}
    resolved = cr._resolve_compound_rows(
        [copy.deepcopy(row)],
        db_resolver=_StubDb(by_name),
        strict_db=False,
        report=report,
        pointer_prefix="/entities/compounds",
        name_index=name_index,
    )
    assert len(resolved) == 1, "a refused row must never be dropped from the row set"
    return resolved[0], report


def _stamped_on(row: Dict[str, Any]) -> List[str]:
    return sorted(key for key in STAMPED if row.get(key) not in (None, ""))


def _refusals(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        issue
        for issue in (report.get("unresolved") or {}).get("db_identities", [])
        if isinstance(issue.get("admission"), dict) and not issue["admission"]["admitted"]
    ]


# --------------------------------------------------------------------------
# § 6.1 -- G9. Fails behaviourally at base 50420a1, passes at the tip.
# --------------------------------------------------------------------------


def test_g9_a_name_keyed_match_no_longer_restores_refused_identifiers() -> None:
    """AT BASE ``mapped_ids`` comes back {chebi, pubchem, pathbank_compound_id,
    pwc_id}; three of those are namespaces this row's own refusal record names.
    AT TIP it stays empty."""
    row, _ = _resolve(REFUSED_ROW, {"2,3-dihydro-2,3-dihydroxybenzoate": DDB_MATCH})
    restored = sorted(set(REFUSED_NAMESPACES) & set(row.get("mapped_ids") or {}))
    assert restored == [], (
        f"the identity gate refused {sorted(REFUSED_NAMESPACES)} and the DB "
        f"re-resolution restored {restored} into mapped_ids={row.get('mapped_ids')}"
    )
    assert row.get("mapped_ids") == {}


def test_g9_no_refused_namespace_reaches_a_scalar_column_either() -> None:
    """AT BASE: pathwhiz_id / pathbank_compound_id / pw_compound_id / chebi_id /
    pubchem_cid / pwc_id are all written. D-028 rule 3 -- never a partial apply."""
    row, _ = _resolve(REFUSED_ROW, {"2,3-dihydro-2,3-dihydroxybenzoate": DDB_MATCH})
    assert _stamped_on(row) == [], f"row gained {_stamped_on(row)}"
    assert "db_row" not in row, "a refused match must not seed a trusted db_row"


def test_g9_the_extracted_name_survives_the_refused_match() -> None:
    """AT BASE the row is renamed to '2,3-Dihydro-2,3-dihydroxybenzoic acid'."""
    row, _ = _resolve(REFUSED_ROW, {"2,3-dihydro-2,3-dihydroxybenzoate": DDB_MATCH})
    assert row["name"] == "2,3-dihydro-2,3-dihydroxybenzoate"


def test_g9_the_realized_corpus_case_fe3_no_longer_restores_its_refused_kegg() -> None:
    """The one row on the committed corpus where this actually completed.

    ``Fe3+`` (PMC12096016/research, extracted as *ferric iron*) refused
    ``kegg:C14819``; the committed ``final_mapped.json`` ships
    ``mapped_ids.kegg = C14819`` again, byte-equal to
    ``_mapped_ids_from_row(db_match.chosen)``. AT BASE this reproduces; AT TIP
    the kegg does not come back.
    """
    ferric = {
        "name": "ferric iron",
        "mapped_ids": {},
        "mapping_meta": {"rejected_mapped_ids": {"kegg": "C14819"}},
    }
    row, _ = _resolve(ferric, {"ferric iron": FE3_MATCH})
    assert "kegg" not in (row.get("mapped_ids") or {})
    assert row.get("kegg_id") in (None, "")
    assert row["name"] == "ferric iron"


# --------------------------------------------------------------------------
# § 6.2 -- non-vacuity. The guard is capable of being inert, and is.
# --------------------------------------------------------------------------


def test_a_row_with_no_refusal_record_still_resolves_and_gains_its_identifiers() -> None:
    """Without this the patch is indistinguishable from disabling DB resolution."""
    clean = {"name": "2,3-dihydro-2,3-dihydroxybenzoate", "mapped_ids": {}}
    row, report = _resolve(clean, {"2,3-dihydro-2,3-dihydroxybenzoate": DDB_MATCH})
    assert row["name"] == "2,3-Dihydro-2,3-dihydroxybenzoic acid"
    assert row["mapped_ids"] == {
        "chebi": "15941", "pubchem": "23615184",
        "pwc_id": "PW_C041128", "pathbank_compound_id": "41128",
    }
    assert row["pathbank_compound_id"] == 41128 and row["db_status"] == "matched"
    assert _refusals(report) == []


def test_an_empty_refusal_record_is_not_a_refusal() -> None:
    """``mapping_meta`` present, ``rejected_mapped_ids`` empty -- still admitted."""
    row, _ = _resolve(
        {
            "name": "2,3-dihydro-2,3-dihydroxybenzoate",
            "mapped_ids": {},
            "mapping_meta": {"rejected_mapped_ids": {}},
        },
        {"2,3-dihydro-2,3-dihydroxybenzoate": DDB_MATCH},
    )
    assert row["db_status"] == "matched" and row["mapped_ids"]


def test_a_refusal_the_match_would_not_restore_does_not_block_admission() -> None:
    """The predicate is namespace-precise, not "any refusal ends DB resolution".

    This row refused ``drugbank``. PathBank 41128 carries no drugbank, so
    admitting the match restores nothing that was refused, and it is admitted.
    """
    row, report = _resolve(
        {
            "name": "2,3-dihydro-2,3-dihydroxybenzoate",
            "mapped_ids": {},
            "mapping_meta": {"rejected_mapped_ids": {"drugbank": "DB01234"}},
        },
        {"2,3-dihydro-2,3-dihydroxybenzoate": DDB_MATCH},
    )
    assert row["db_status"] == "matched"
    assert row["mapped_ids"]["pathbank_compound_id"] == "41128"
    assert "drugbank" not in row["mapped_ids"]
    assert _refusals(report) == []


# --------------------------------------------------------------------------
# § 6.3 -- the legacy branch is untouched.
# --------------------------------------------------------------------------


def test_the_legacy_branch_is_untouched_by_the_guard() -> None:
    """A row carrying a PathBank scalar never reaches the name-keyed resolver.

    Refusal present, scalar still on the row: the legacy branch fires, records
    ``legacy_id_unverified``, and the guard never runs. This is the
    discrimination that proves the predicate is targeted rather than blunt.
    """
    legacy = {
        "name": "Fe3+",
        "pathbank_compound_id": 8133,
        "mapped_ids": {"kegg": "C14819", "pathbank_compound_id": "8133"},
        "mapping_meta": {"rejected_mapped_ids": {"kegg": "C14819"}},
    }
    row, report = _resolve(legacy, {"Fe3+": FE3_MATCH})
    assert row["db_status"] == "legacy_id_unverified"
    assert row["db_id"] == 8133 and row["chosen_rule"] == "legacy_pathwhiz_id_unverified"
    assert row["mapped_ids"] == {"kegg": "C14819", "pathbank_compound_id": "8133"}
    assert "db_match" not in row, "the legacy branch never consults the resolver"
    assert _refusals(report) == []
    assert report["db_resolution"]["compounds"][0]["reason"] == "payload_pathwhiz_id"


# --------------------------------------------------------------------------
# § 6.4 -- the refusal is record-only, never a drop (merge rule 7 / D-028 r4).
# --------------------------------------------------------------------------


def test_the_refusal_is_recorded_for_review_and_never_raises() -> None:
    row, report = _resolve(REFUSED_ROW, {"2,3-dihydro-2,3-dihydroxybenzoate": DDB_MATCH})

    # the row survives, with its extracted name and its refusal record intact
    assert row["name"] == "2,3-dihydro-2,3-dihydroxybenzoate"
    assert row["mapping_meta"]["rejected_mapped_ids"]["kegg"] == "C04171"
    assert row["db_status"] == IDENTITY_REFUSED_STATUS
    assert row["db_status"] != "matched", "a refused row is not a trusted identity"
    # the evidence is kept -- record-only means recorded, not discarded
    assert row["db_match"]["chosen"]["id"] == 41128

    refusals = _refusals(report)
    assert len(refusals) == 1, "the refusal was not filed for review"
    admission = refusals[0]["admission"]
    assert admission["admitted"] is False
    assert admission["reason"] == cr.REFUSED_IDENTITY_WOULD_BE_RESTORED
    assert admission["refused_namespaces"] == ["chebi", "pathbank_compound_id", "pubchem"]
    assert admission["admitted_by_match_rule"] == "exact_name_or_synonym_match"
    assert admission["rule"] == "exact_short_name_or_synonym"

    # the match itself is still logged as consulted
    assert report["db_resolution"]["compounds"][0]["status"] == "matched"


def test_the_veto_never_turns_a_refusal_into_an_admission() -> None:
    """Every D-028 refusal reason is preserved verbatim on a colliding row."""
    fuzzy = {
        "status": "matched", "raw_name": "OPDA", "chosen_rule": "fuzzy_name",
        "confidence": 0.65, "candidates": [],
        "chosen": {"id": 104723, "name": "Dinor-12-oxo-phytodienoate",
                   "kegg_id": "C01226", "chebi_id": "57411"},
    }
    row, report = _resolve(
        {"name": "OPDA", "mapped_ids": {},
         "mapping_meta": {"rejected_mapped_ids": {"kegg": "C01226"}}},
        {"OPDA": fuzzy},
    )
    admission = _refusals(report)[0]["admission"]
    assert admission["admitted"] is False
    assert admission["reason"] == "fuzzy_name_match_never_admitted"
    assert admission["refused_namespaces"] == ["kegg"]
    assert "admitted_by_match_rule" not in admission
    assert row["name"] == "OPDA" and _stamped_on(row) == []


def test_an_unmatched_result_on_a_refused_row_is_unchanged() -> None:
    row, report = _resolve(REFUSED_ROW, {})
    assert row["db_status"] == "unmatched"
    assert _refusals(report)[0]["admission"]["reason"] == "no_match_to_admit"
    assert "refused_namespaces" not in _refusals(report)[0]["admission"]


# --------------------------------------------------------------------------
# § 6.5 -- the extraction-name provenance carrier still holds.
# --------------------------------------------------------------------------


def test_provenance_fields_are_exactly_the_unconditional_apply_block() -> None:
    assert _PROVENANCE_FIELDS == (
        "raw_name", "db_status", "chosen_rule", "confidence", "db_match",
    )


def test_raw_name_still_records_the_extraction_name_not_the_canonical_one(
    monkeypatch: Any,
) -> None:
    """The measured ``glycolate`` -> ``Glycolic acid`` defect stays fixed.

    Runs the real pre-freeze entry point, whose fixed-point loop is what makes
    pass 2 unable to overwrite pass 1's account of the decision. The row carries
    no refusal, so this is also the guard proving inert on the rename path.
    """
    payload = {
        "entities": {"compounds": [{"name": "glycolate", "mapped_ids": {}}]},
        "processes": {"reactions": [
            {"name": "r1", "inputs": ["glycolate"], "outputs": ["glycolate"]},
        ]},
    }
    summary = resolve_compounds_prefreeze(
        payload,
        db_resolver=_StubDb({"glycolate": GLYCOLATE_MATCH}),
        strict_db=False,
        name_index=None,
    )
    assert summary["applied"] is True
    compound = payload["entities"]["compounds"][0]
    assert compound["name"] == "Glycolic acid"
    assert compound["raw_name"] == "glycolate", (
        "the canonical name overwrote the extraction name -- PRODUCT_CONTRACT § 3"
    )
    assert compound["db_status"] == "matched"
    assert compound["mapped_ids"]["kegg"] == "C00160"


# --------------------------------------------------------------------------
# § 6.6 -- collateral, on the committed corpus.
# --------------------------------------------------------------------------


def _corpus_artifacts() -> List[Path]:
    out: List[Path] = []
    for run in CORPUS_RUNS:
        out.extend(sorted((REPO_ROOT / run).rglob("final_mapped.json")))
    return out


def _corpus_db_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every real PathBank row this artifact recorded, chosen or merely offered."""
    out: List[Dict[str, Any]] = []
    for row in payload.get("entities", {}).get("compounds", []) or []:
        if not isinstance(row, dict):
            continue
        match = row.get("db_match")
        if not isinstance(match, dict):
            continue
        if isinstance(match.get("chosen"), dict):
            out.append(match["chosen"])
        for candidate in match.get("candidates") or []:
            if isinstance(candidate, dict) and isinstance(candidate.get("db_row"), dict):
                out.append(candidate["db_row"])
    return out


def test_collateral_every_committed_compound_row_without_a_refusal_is_inert() -> None:
    """The guard's blast radius on the whole T-104 + T-105 corpus is one row.

    Two measurements. First, every compound row replayed against **its own**
    recorded DB match -- what actually happened. Second, every row that carries
    no refusal record replayed against **every** PathBank row recorded anywhere
    in the corpus, because the guard's only other input is
    ``mapping_meta.rejected_mapped_ids``: a row with none cannot be changed by
    any match, and the cross-product says so rather than assuming it.
    """
    artifacts = _corpus_artifacts()
    assert len(artifacts) == 22, f"corpus moved: {len(artifacts)} artifacts"

    compounds = 0
    with_meta = 0
    without_refusal = 0
    with_refusal = 0
    with_match = 0
    fired: List[Tuple[str, str, List[str]]] = []
    clean_rows: List[Dict[str, Any]] = []
    db_rows: List[Dict[str, Any]] = []

    for path in artifacts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        db_rows.extend(_corpus_db_rows(payload))
        for row in payload.get("entities", {}).get("compounds", []) or []:
            if not isinstance(row, dict):
                continue
            compounds += 1
            meta = row.get("mapping_meta")
            refused = isinstance(meta, dict) and bool(meta.get("rejected_mapped_ids"))
            if isinstance(meta, dict):
                with_meta += 1
                with_refusal += int(refused)
                without_refusal += int(not refused)
            if not refused:
                clean_rows.append(row)
            chosen = (row.get("db_match") or {}).get("chosen") if isinstance(row.get("db_match"), dict) else None
            if not isinstance(chosen, dict):
                continue
            with_match += 1
            restored = cr._restored_refused_namespaces(row, chosen)
            if not refused:
                assert restored == [], (
                    f"guard fired on a row with no refusal record: "
                    f"{row.get('name')} in {path}"
                )
            elif restored:
                fired.append((str(path.relative_to(REPO_ROOT)), str(row.get("name")), restored))

    cross = sum(
        len(cr._restored_refused_namespaces(row, db_row))
        for row in clean_rows
        for db_row in db_rows
    )

    print(
        f"[C-078 collateral] artifacts={len(artifacts)} compound_rows={compounds} "
        f"with_mapping_meta={with_meta} with_refusal={with_refusal} "
        f"without_refusal={without_refusal} with_recorded_db_match={with_match} "
        f"guard_fires={len(fired)} cross_product={len(clean_rows)}x{len(db_rows)} "
        f"cross_product_fires={cross}"
    )
    for leg, name, namespaces in fired:
        print(f"[C-078 collateral] FIRES {name} {namespaces} {leg}")

    assert compounds == 130
    assert with_meta == 130 and with_refusal == 8 and without_refusal == 122
    assert len(clean_rows) == 122 and len(db_rows) == 17
    assert cross == 0, "the guard is not inert on rows carrying no refusal record"
    assert [(name, ns) for _leg, name, ns in fired] == [("Fe3+", ["kegg"])], (
        "the guard's blast radius on the committed corpus moved"
    )


def test_collateral_the_five_seam_rows_were_never_actually_restamped() -> None:
    """C-078 § 2f says 5 rows are 'exercised'. Measured: 5 are EXPOSED, 0 landed.

    All five ``2,3-dihydro-2,3-dihydroxybenzoate`` rows got ``status=ambiguous``
    from the live resolver, so ``apply_compound_db_resolution`` took its
    not-matched early return and stamped nothing. The guard is inert on all five
    -- there is no ``chosen`` to veto -- and this patch does not change them.
    """
    seam: List[Dict[str, Any]] = []
    for path in _corpus_artifacts():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("entities", {}).get("compounds", []) or []:
            if not isinstance(row, dict):
                continue
            rejected = ((row.get("mapping_meta") or {}) if isinstance(row.get("mapping_meta"), dict) else {}).get(
                "rejected_mapped_ids"
            ) or {}
            if any(key in rejected for key in ("pathbank_compound_id", "pw_compound_id", "pathwhiz_id")):
                seam.append(row)
    assert len(seam) == 5
    assert {str(row.get("name")) for row in seam} == {"2,3-dihydro-2,3-dihydroxybenzoate"}
    assert {str(row.get("db_status")) for row in seam} == {"ambiguous"}
    for row in seam:
        assert row.get("mapped_ids") == {}
        assert cr._restored_refused_namespaces(row, (row.get("db_match") or {}).get("chosen") or {}) == []


# --------------------------------------------------------------------------
# § 3 -- the ordering premise, through a production entry point.
# --------------------------------------------------------------------------


def _cli_args(input_path: Path, out_dir: Path) -> SimpleNamespace:
    import t2pw

    return SimpleNamespace(
        input_path=str(input_path),
        out_dir=str(out_dir),
        ref=str(Path(t2pw.__file__).resolve().parents[2] / "reference" / "PW000001.pwml"),
        name="C-078 ordering probe",
        subject="Metabolic",
        description="",
        width=3200,
        height=1400,
        background_color="#FFFFFF",
        non_strict_db=True,
    )


def _ordering_payload() -> Dict[str, Any]:
    """A ``final_mapped.json``-shaped payload carrying the real refused row."""
    return {
        "entities": {
            "species": [{"name": "Escherichia coli", "taxonomy_id": "562"}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [copy.deepcopy(REFUSED_ROW), {"name": "2,3-dihydroxybenzoate"}],
            "proteins": [{
                "name": "EntA", "species": "Escherichia coli",
                "uniprot": "P15047", "pathbank_protein_id": 201,
            }],
            "protein_complexes": [{
                "name": "EntA complex", "species": "Escherichia coli",
                "components": ["EntA"], "generated": True,
                "generation_reason": "single_protein_pathwhiz_wrapper",
            }],
        },
        "biological_states": [
            {"name": "cyto_state", "species": "Escherichia coli",
             "subcellular_location": "cytosol"},
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": "2,3-dihydro-2,3-dihydroxybenzoate", "biological_state": "cyto_state"},
                {"compound": "2,3-dihydroxybenzoate", "biological_state": "cyto_state"},
            ],
            "protein_locations": [{"protein": "EntA", "biological_state": "cyto_state"}],
        },
        "processes": {
            "reactions": [{
                "name": "2,3-diDHB dehydrogenation",
                "inputs": ["2,3-dihydro-2,3-dihydroxybenzoate"],
                "outputs": ["2,3-dihydroxybenzoate"],
                "biological_state": "cyto_state",
                "enzymes": [{"entity": "EntA complex", "entity_type": "protein_complex"}],
            }],
            "transports": [],
            "interactions": [],
        },
    }


def _compound_ir_record(ir: Dict[str, Any], *names: str) -> Optional[Dict[str, Any]]:
    for record in ir.get("entities", {}).get("compounds", []) or []:
        if str(record.get("name")) in names:
            return record
    return None


def test_g9_ordering_the_refusal_survives_the_production_cli_export(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """§ 3. The premise, proven behaviourally at a production entry point.

    ``writer.run_pwml_pipeline_export`` is the CLI export path
    (``README.md`` -> ``scripts/run_pwml.py`` -> here). It reads a
    ``final_mapped.json``-shaped payload -- i.e. the *output* of the mapping
    stage, refusal record and all -- runs ``run_prefreeze_resolution`` on it, and
    only then freezes it into the IR. Nothing between those two points re-imposes
    the refusal, which is exactly what § 3 asks to be shown rather than assumed.

    AT BASE the frozen compound comes back named
    ``2,3-Dihydro-2,3-dihydroxybenzoic acid`` carrying chebi / pubchem /
    pathbank_compound_id -- three refused namespaces, now canonical.
    AT TIP it keeps its extracted name and gains nothing.
    """
    from t2pw.pwml import writer as writer_mod

    monkeypatch.setattr(
        cr, "PathWhizCompoundResolver",
        lambda db: _StubResolver(_StubDb({"2,3-dihydro-2,3-dihydroxybenzoate": DDB_MATCH})),
    )
    import t2pw.mapping.map_ids as map_ids

    monkeypatch.setattr(
        map_ids, "PathBankDbResolver",
        type("_P", (), {"from_env": staticmethod(lambda: _StubDb({}))}),
    )

    payload_path = tmp_path / "final_mapped.json"
    payload_path.write_text(json.dumps(_ordering_payload()), encoding="utf-8")
    out_dir = tmp_path / "out"
    result = writer_mod.run_pwml_pipeline_export(_cli_args(payload_path, out_dir))
    print(f"[C-078 ordering] export ok={result.get('ok')} error={result.get('error', '')}")

    # the pre-freeze pass ran, on this payload, at this entry point
    prefreeze = json.loads((out_dir / "pwml_prefreeze_resolution_report.json").read_text(encoding="utf-8"))
    logged = prefreeze["compounds"]["resolution_report"]["db_resolution"]["compounds"]
    assert any(entry.get("status") == "matched" for entry in logged), (
        "the name-keyed resolver was never consulted -- the ordering premise fails"
    )

    # and the canonical graph it froze does not carry the refused identity
    ir = json.loads((out_dir / "final.pwml_ir.json").read_text(encoding="utf-8"))
    record = _compound_ir_record(
        ir, "2,3-dihydro-2,3-dihydroxybenzoate", "2,3-Dihydro-2,3-dihydroxybenzoic acid",
    )
    assert record is not None, "the compound was dropped from the canonical graph"
    restored = sorted(set(REFUSED_NAMESPACES) & set(record.get("mapped_ids") or {}))
    assert restored == [], (
        f"the frozen canonical graph carries refused namespaces {restored}: "
        f"name={record.get('name')!r} mapped_ids={record.get('mapped_ids')}"
    )
    assert _stamped_on(record) == [], f"frozen record gained {_stamped_on(record)}"
    assert record["name"] == "2,3-dihydro-2,3-dihydroxybenzoate"
    assert record.get("db_status") == IDENTITY_REFUSED_STATUS

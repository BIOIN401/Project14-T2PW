"""C-073 / F-096 -- an entity may not ship a real external accession it cannot support.

Run 2026-08-21_2239 emitted seven FALSE REAL IDENTIFIERS on legs that reported
``PASS``. This battery covers the two the identity layer can reach without
collateral, both measured over that run's 102 mapped rows (7 known-false, 95
legitimate):

* ``succinyl-CoA`` on PMC12180156/research -- the gold's designated HALLUCINATION
  TEST. Zero occurrences of "succinyl" in a 67,304-character source, shipped with
  ``hmdb``/``kegg``/``chebi``/``pubchem`` and four more. Neither its name, its
  ``short_name`` ``Suc-CoA`` nor either synonym is locatable in that paper.
* ``drugbank:DB00114`` claimed by BOTH ``ALAS2`` and ``Pyridoxal 5'-phosphate``
  on PMC12856317/research -- the only accession collision in the whole run.

THE ANTI-COLLATERAL ARM IS THE LOAD-BEARING ONE. A rule that strips a legitimate
accession is worse than the defect it fixes, so the six measured alias/format
misses each get their own arm here and every one of them must KEEP its
identifiers.

G9 NOTE. :func:`test_g9_unsupported_identity_is_not_shipped` is the base-failing
behavioural proof and is written to run unchanged on the base SHA: it touches no
symbol this card introduces, builds the source index from a literal shape, and
fails on an assertion about what ``map_payload`` ships -- not on an import. The
tests that exercise this card's new module import it inside their own bodies for
exactly that reason.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping import map_ids  # noqa: E402
from t2pw.mapping.map_ids import map_payload  # noqa: E402

#: The payload key and blob shape the Stage-2 merge writes. Spelled out as
#: literals so this file states the contract instead of importing it, which is
#: what lets the G9 proof run on the base SHA.
SOURCE_INDEX_KEY = "source_text_index"
INDEX_SCHEMA_VERSION = 1


def _fold(text: str) -> str:
    """The normalization the index stores: casefold, punctuation -> one space.

    An independent restatement of the production fold.
    :func:`test_the_test_fold_agrees_with_the_production_fold` pins the two
    together so this file cannot quietly diverge from the code it is judging.
    """
    lowered = str(text or "").casefold()
    folded = "".join(ch if ch.isalnum() else " " for ch in lowered)
    return " ".join(folded.split())


def source_index(text: str) -> Dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "length": len(text),
        "normalized": _fold(text),
    }


# ── the offline runner ───────────────────────────────────────────────────────


class _NoNetwork(RuntimeError):
    """Raised if anything under test reaches for the network."""


def run_offline(payload: Dict[str, Any], cache_path: Path) -> Dict[str, Any]:
    """``map_payload`` with every external door shut, as ``test_map_ids_mapping_lineage``
    does it: no resolver, no LLM, no NCBI, and an ``HttpClient`` that raises. What
    comes out is a function of the input alone."""
    env = {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0"}
    with patch.dict(os.environ, env), patch.object(
        map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)
    ), patch.object(
        map_ids, "_ai_protein_synonym_lookup", return_value=[]
    ), patch.object(
        map_ids.HttpClient, "get", side_effect=_NoNetwork("network call during an offline run")
    ):
        return map_payload(payload, cache_path=cache_path, id_source="db", use_cache=False)


# ── fixtures copied from run 2026-08-21_2239 ─────────────────────────────────

#: PMC12180156's own vocabulary, minus every trace of succinyl-CoA. "succinyl"
#: really does occur zero times in the 67,304-character original.
SOURCE_WITHOUT_SUCCINYL = (
    "Heme biosynthesis in erythroid precursors. Glycine is condensed by ALAS2 in the "
    "mitochondrial matrix, and ferrochelatase inserts ferrous iron into protoporphyrin "
    "IX. SFXN4 supplies the serine transport step required for one-carbon metabolism, "
    "and loss of SFXN4 impairs heme production."
)

#: PMC12856317 names it, so PMC12856317 keeps it. The rule is paper-relative.
SOURCE_WITH_SUCCINYL = SOURCE_WITHOUT_SUCCINYL + (
    " ALAS2 condenses succinyl-CoA with glycine to form 5-aminolevulinate."
)

SUCCINYL_IDS = {
    "hmdb": "HMDB0001022", "kegg": "C00091", "chebi": "15380", "pubchem": "439161",
    "cas": "604-98-8", "biocyc": "3-METHYLBENZYLSUCCINYL-COA", "chemspider": "388307",
    "pathbank_compound_id": "808",
}
HEME_IDS = {"hmdb": "HMDB0003178", "kegg": "C00032", "chebi": "17627", "pubchem": "444098"}
GLYCINE_IDS = {"hmdb": "HMDB0000123", "kegg": "C00037", "chebi": "15428", "pubchem": "750"}

FABRICATED_NAMESPACES = ("hmdb", "kegg", "chebi", "pubchem")


def succinyl_row() -> Dict[str, Any]:
    """The row PMC12180156/research actually shipped, names and all."""
    return {
        "name": "succinyl-CoA",
        "class": "compound",
        "short_name": "Suc-CoA",
        "synonyms": ["Succinyl-CoA", "Succinyl coenzyme A"],
        "mapped_ids": dict(SUCCINYL_IDS),
    }


def heme_payload(index: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """That leg's shape: the fabrication, two legitimate compounds that the paper
    really names, and a reaction that consumes all three."""
    payload: Dict[str, Any] = {
        "entities": {
            "compounds": [
                {"name": "heme", "class": "compound", "short_name": "Heme",
                 "synonyms": ["Heme", "Haem", "Protoheme"], "mapped_ids": dict(HEME_IDS)},
                {"name": "Glycine", "class": "compound", "short_name": "Gly",
                 "raw_name": "glycine", "synonyms": ["Glycine", "Aminoacetic acid", "Gly"],
                 "mapped_ids": dict(GLYCINE_IDS)},
                succinyl_row(),
            ],
            "proteins": [],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "ALAS2 condensation",
                    "inputs": ["Glycine", "succinyl-CoA"],
                    "outputs": ["heme"],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }
    if index is not None:
        payload[SOURCE_INDEX_KEY] = index
    return payload


def _row(result: Dict[str, Any], name: str, bucket: str = "compounds") -> Dict[str, Any]:
    rows = result["payload"]["entities"][bucket]
    matches = [r for r in rows if isinstance(r, dict) and r.get("name") == name]
    assert matches, f"{name} vanished from {bucket}: {[r.get('name') for r in rows]}"
    return matches[0]


# ── 1. G9 base-failing behavioural proof ─────────────────────────────────────


def test_g9_unsupported_identity_is_not_shipped(tmp_path: Path) -> None:
    """THE PROOF. Through the production entry point, with a source index that
    does not contain ``succinyl-CoA`` or any name that row offers, the row must
    not ship ``hmdb``/``kegg``/``chebi``/``pubchem``.

    On ``20e6b68`` ``map_payload`` has no access to the paper at all
    (``map_ids.py:7719-7722``: "Stage 2 resolves names against databases and never
    reads the paper"), so all four ship and this assertion fails. Nothing here
    names a symbol this card adds."""
    result = run_offline(
        heme_payload(source_index(SOURCE_WITHOUT_SUCCINYL)), tmp_path / "cache.json"
    )
    shipped = _row(result, "succinyl-CoA").get("mapped_ids") or {}

    assert [ns for ns in FABRICATED_NAMESPACES if shipped.get(ns)] == [], (
        "a row no name of which occurs in the source shipped real external "
        f"accessions: {shipped}"
    )


def test_g9_the_fabricated_accessions_are_withheld_not_deleted(tmp_path: Path) -> None:
    """Merge rule 7. The refusal is recorded, so the identifiers can be audited
    and the decision reversed; they are not silently dropped on the floor."""
    result = run_offline(
        heme_payload(source_index(SOURCE_WITHOUT_SUCCINYL)), tmp_path / "cache.json"
    )
    meta = _row(result, "succinyl-CoA").get("mapping_meta") or {}
    rejected = meta.get("rejected_mapped_ids") or {}

    assert set(FABRICATED_NAMESPACES) <= set(rejected)
    assert rejected["kegg"] == SUCCINYL_IDS["kegg"]


# ── 2. preservation: the six measured alias cases ────────────────────────────

ALIAS_CASES = [
    pytest.param(
        {"name": "2,3-dihydroxybenzoic acid", "synonyms": ["2,3-Dihydroxybenzoate"]},
        "EntB converts isochorismate to 2,3-dihydroxybenzoate during enterobactin assembly.",
        id="dihydroxybenzoic-acid-vs-benzoate",
    ),
    pytest.param(
        {"name": "L-serine", "synonyms": ["Serine"]},
        "Serine hydroxymethyltransferase consumes serine in the one-carbon cycle.",
        id="l-serine-vs-serine",
    ),
    pytest.param(
        {"name": "Adenosine triphosphate", "synonyms": ["ATP"], "short_name": "ATP"},
        "The adenylation domain performs an ATP-dependent activation of the substrate.",
        id="adenosine-triphosphate-vs-atp",
    ),
    pytest.param(
        {"name": "Adenosine monophosphate", "synonyms": ["AMP", "Adenylate"]},
        "Activation releases AMP and pyrophosphate from the enzyme active site.",
        id="adenosine-monophosphate-vs-amp",
    ),
    pytest.param(
        {"name": "ferric-enterobactin", "synonyms": []},
        "FepA imports ferric enterobactin across the outer membrane.",
        id="ferric-enterobactin-hyphen-vs-space",
    ),
    pytest.param(
        {"name": "Fe3", "synonyms": []},
        "Enterobactin chelates Fe3+ with very high affinity.",
        id="fe3-vs-fe3-plus",
    ),
]


@pytest.mark.parametrize("row_fields,source", ALIAS_CASES)
def test_preservation_measured_alias_cases_keep_their_accessions(
    row_fields: Dict[str, Any], source: str, tmp_path: Path
) -> None:
    """THE ANTI-COLLATERAL ARM. All six were measured as would-be collateral of
    the name-only rule; the synonym- and punctuation-aware rule keeps every one.
    Zero collateral on the 95 legitimate mapped rows is a hard requirement."""
    row = {"class": "compound", "mapped_ids": dict(HEME_IDS)}
    row.update(row_fields)
    payload = {
        "entities": {"compounds": [row], "proteins": [], "protein_complexes": []},
        "processes": {"reactions": []},
        SOURCE_INDEX_KEY: source_index(source),
    }
    result = run_offline(payload, tmp_path / "cache.json")
    shipped = _row(result, row["name"]).get("mapped_ids") or {}

    assert shipped.get("kegg") == HEME_IDS["kegg"], (
        f"{row['name']} lost a legitimate accession against a source that names it: {shipped}"
    )
    assert set(HEME_IDS) <= set(shipped)


# ── 3. paper-relativity ──────────────────────────────────────────────────────


def test_paper_relativity_the_same_name_is_withheld_here_and_kept_there(tmp_path: Path) -> None:
    """One name, two papers, two answers -- which is exactly right. PMC12180156
    never writes "succinyl"; PMC12856317 does, and there the row is legitimate."""
    absent = run_offline(
        heme_payload(source_index(SOURCE_WITHOUT_SUCCINYL)), tmp_path / "a.json"
    )
    present = run_offline(
        heme_payload(source_index(SOURCE_WITH_SUCCINYL)), tmp_path / "b.json"
    )

    assert not (_row(absent, "succinyl-CoA").get("mapped_ids") or {}).get("kegg")
    assert (_row(present, "succinyl-CoA").get("mapped_ids") or {}).get("kegg") == "C00091"
    # and the paper that names it disturbs nothing else
    assert (_row(present, "heme").get("mapped_ids") or {}).get("kegg") == HEME_IDS["kegg"]


# ── 4. duplicate accession claims ────────────────────────────────────────────


def _collision_payload(name_a: str, name_b: str, accession: str = "DB00114") -> Dict[str, Any]:
    return {
        "entities": {
            "compounds": [
                {"name": name_a, "class": "compound",
                 "mapped_ids": {"drugbank": accession, "kegg": "C00018"}},
                {"name": name_b, "class": "compound",
                 "mapped_ids": {"drugbank": accession, "kegg": "C00250"}},
            ],
            "proteins": [],
            "protein_complexes": [],
        },
        "processes": {"reactions": [{"name": "r", "inputs": [name_a], "outputs": [name_b]}]},
    }


def test_an_accession_claimed_by_two_differently_named_entities_does_not_ship(
    tmp_path: Path,
) -> None:
    """PMC12856317/research shipped ``drugbank:DB00114`` on BOTH ``ALAS2`` and
    ``Pyridoxal 5'-phosphate``. Two different molecules cannot both be DB00114.

    THE DETERMINISTIC RULE: neither claimant keeps it. Nothing on either row says
    it, rather than its rival, owns the accession, so choosing a winner would ship
    a coin toss as a fact. Withholding from all is order-independent and needs no
    tie-break nobody can justify. Every other identifier on both rows survives."""
    result = run_offline(
        _collision_payload("ALAS2", "Pyridoxal 5'-phosphate"), tmp_path / "cache.json"
    )
    alas2 = _row(result, "ALAS2").get("mapped_ids") or {}
    plp = _row(result, "Pyridoxal 5'-phosphate").get("mapped_ids") or {}

    assert not alas2.get("drugbank")
    assert not plp.get("drugbank")
    assert alas2.get("kegg") == "C00018"
    assert plp.get("kegg") == "C00250"


def test_the_collision_refusal_is_recorded_on_both_claimants(tmp_path: Path) -> None:
    from t2pw.mapping import identity_admission

    result = run_offline(
        _collision_payload("ALAS2", "Pyridoxal 5'-phosphate"), tmp_path / "cache.json"
    )
    for name in ("ALAS2", "Pyridoxal 5'-phosphate"):
        meta = _row(result, name).get("mapping_meta") or {}
        assert (meta.get("rejected_mapped_ids") or {}).get("drugbank") == "DB00114"
        record = meta.get(identity_admission.META_KEY) or {}
        assert identity_admission.RULE_ACCESSION_COLLISION in (record.get("rules") or [])

    section = result["report"][identity_admission.REPORT_KEY]
    assert section["counts"]["collisions"] == 1
    assert section["collisions"][0]["namespace"] == "drugbank"
    assert sorted(section["collisions"][0]["claimants"]) == ["ALAS2", "Pyridoxal 5'-phosphate"]


def test_two_rows_with_the_same_normalized_name_are_not_a_collision(tmp_path: Path) -> None:
    """A duplicate row is one entity written twice -- somebody else's finding, and
    not a reason to take a correct accession off either copy. Punctuation and case
    do not make two names different."""
    result = run_offline(
        _collision_payload("ferric-enterobactin", "Ferric Enterobactin"), tmp_path / "cache.json"
    )
    assert (_row(result, "ferric-enterobactin").get("mapped_ids") or {}).get("drugbank") == "DB00114"
    assert (_row(result, "Ferric Enterobactin").get("mapped_ids") or {}).get("drugbank") == "DB00114"


def test_one_entity_claiming_an_accession_from_two_buckets_is_not_a_collision() -> None:
    """ADVERSARIAL. The same entity resolved into two buckets is still ONE name,
    so the accession is uncontested and both copies keep it.

    Exercised against the two passes directly: routing a protein-shaped name
    through the compound loop is a different mechanism's business, and this
    assertion is about the cross-row index, not about which loop claimed the row.
    """
    payload = {
        "entities": {
            "compounds": [{"name": "ALAS2", "class": "compound",
                           "mapped_ids": {"drugbank": "DB00114"}}],
            "proteins": [{"name": "ALAS2", "mapped_ids": {"drugbank": "DB00114"}}],
            "protein_complexes": [],
        },
        "processes": {"reactions": []},
    }
    report = map_ids._admit_identities(payload, payload["entities"])

    assert report["counts"]["collisions"] == 0
    assert report["counts"]["identifiers_withheld"] == 0
    assert payload["entities"]["compounds"][0]["mapped_ids"]["drugbank"] == "DB00114"
    assert payload["entities"]["proteins"][0]["mapped_ids"]["drugbank"] == "DB00114"


def test_the_real_collision_shape_a_protein_and_a_compound() -> None:
    """The shape PMC12856317/research actually shipped: the compound ``Pyridoxal
    5'-phosphate`` and the protein ``ALAS2``, both claiming ``drugbank:DB00114``.
    Both lose it; both keep everything else and both survive as rows."""
    from t2pw.mapping import identity_admission

    payload = {
        "entities": {
            "compounds": [{"name": "Pyridoxal 5'-phosphate", "class": "compound",
                           "mapped_ids": {"drugbank": "DB00114", "kegg": "C00018"}}],
            "proteins": [{"name": "ALAS2",
                          "mapped_ids": {"drugbank": "DB00114", "uniprot": "P22557"}}],
            "protein_complexes": [],
        },
        "processes": {"reactions": []},
    }
    report = map_ids._admit_identities(payload, payload["entities"])
    compound = payload["entities"]["compounds"][0]
    protein = payload["entities"]["proteins"][0]

    assert report["counts"]["collisions"] == 1
    assert not compound["mapped_ids"].get("drugbank")
    assert not protein["mapped_ids"].get("drugbank")
    assert compound["mapped_ids"]["kegg"] == "C00018"
    assert protein["mapped_ids"]["uniprot"] == "P22557"
    assert compound["name"] == "Pyridoxal 5'-phosphate" and protein["name"] == "ALAS2"
    for row in (compound, protein):
        record = row["mapping_meta"][identity_admission.META_KEY]
        assert record["rules"] == [identity_admission.RULE_ACCESSION_COLLISION]


# ── 5. fail open ─────────────────────────────────────────────────────────────


def test_fail_open_no_source_index_withholds_nothing(tmp_path: Path) -> None:
    """PRODUCT_CONTRACT § 8: "not evaluated" is never "false". Without the paper
    the question cannot be asked, so the fabricated row ships exactly as it does
    today -- which is what keeps unit tests, ``interactive_curator`` and every
    legacy payload behaving identically."""
    from t2pw.mapping import identity_admission

    result = run_offline(heme_payload(), tmp_path / "cache.json")
    shipped = _row(result, "succinyl-CoA").get("mapped_ids") or {}

    assert shipped.get("kegg") == SUCCINYL_IDS["kegg"]
    meta = _row(result, "succinyl-CoA").get("mapping_meta") or {}
    assert identity_admission.META_KEY not in meta
    # The pass claims nothing at all rather than claiming support. A section that
    # said "supported" here would be a lie; silence is the honest record, and it
    # is also what keeps the § 7 lineage golden byte-identical.
    assert identity_admission.REPORT_KEY not in result["report"]


def test_fail_open_an_empty_source_index_reads_not_evaluated_never_supported(
    tmp_path: Path,
) -> None:
    """ADVERSARIAL: empty index vs absent index. An index that was OFFERED and is
    unusable is the case that must speak, and what it says is ``not_evaluated``."""
    from t2pw.mapping import identity_admission

    payload = heme_payload({"schema_version": INDEX_SCHEMA_VERSION, "length": 0, "normalized": ""})
    result = run_offline(payload, tmp_path / "cache.json")
    section = result["report"][identity_admission.REPORT_KEY]

    assert section["source_index"]["status"] == identity_admission.STATUS_NOT_EVALUATED
    assert section["source_index"]["reason"] == identity_admission.NOT_EVALUATED_EMPTY_INDEX
    assert section["source_index"]["offered"] is True
    assert section["counts"]["identifiers_withheld"] == 0
    assert (_row(result, "succinyl-CoA").get("mapped_ids") or {}).get("kegg") == "C00091"


def test_fail_open_an_index_of_an_unknown_schema_version_is_not_read(tmp_path: Path) -> None:
    """A shape this code does not understand is no evidence, not bad evidence."""
    payload = heme_payload(
        {"schema_version": INDEX_SCHEMA_VERSION + 99, "length": 10, "normalized": "heme"}
    )
    result = run_offline(payload, tmp_path / "cache.json")
    assert (_row(result, "succinyl-CoA").get("mapped_ids") or {}).get("kegg") == "C00091"


# ── 6. withhold is not delete ────────────────────────────────────────────────


def test_withheld_row_keeps_its_name_its_class_and_its_graph_role(tmp_path: Path) -> None:
    """Merge rule 7. The entity survives as an incomplete-but-correct row; only
    the claim it could not support comes off."""
    result = run_offline(
        heme_payload(source_index(SOURCE_WITHOUT_SUCCINYL)), tmp_path / "cache.json"
    )
    row = _row(result, "succinyl-CoA")

    assert row["name"] == "succinyl-CoA"
    assert row["class"] == "compound"
    assert row["synonyms"] == ["Succinyl-CoA", "Succinyl coenzyme A"]
    reactions = result["payload"]["processes"]["reactions"]
    assert len(reactions) == 1
    assert "succinyl-CoA" in reactions[0]["inputs"]


def test_the_refusal_is_recorded_in_the_mapping_report(tmp_path: Path) -> None:
    from t2pw.mapping import identity_admission

    result = run_offline(
        heme_payload(source_index(SOURCE_WITHOUT_SUCCINYL)), tmp_path / "cache.json"
    )
    section = result["report"][identity_admission.REPORT_KEY]

    assert section["source_index"]["status"] == "evaluated"
    assert section["counts"]["rows_withheld"] == 1
    assert section["counts"]["supported"] == 2  # heme and Glycine
    entry = section["withheld"][0]
    assert entry["name"] == "succinyl-CoA"
    assert entry["rule"] == identity_admission.RULE_NOT_SUPPORTED
    assert set(FABRICATED_NAMESPACES) <= set(entry["identifiers"])


def test_the_refusal_is_recorded_in_provenance_lineage(tmp_path: Path) -> None:
    """PRODUCT_CONTRACT § 3. A withheld identifier is a decision this stage made,
    so this stage has to be able to answer for it: ``excluded``, sourceless
    (nothing was retrieved that contradicts anything), and flagged for review."""
    from t2pw.mapping import identity_admission
    from t2pw.pipeline.lineage import read as read_lineage

    result = run_offline(
        heme_payload(source_index(SOURCE_WITHOUT_SUCCINYL)), tmp_path / "cache.json"
    )
    entries = [
        entry for entry in read_lineage(_row(result, "succinyl-CoA")).entries
        if entry.stage == "identifier_mapping"
    ]
    assert entries, "no identifier_mapping attribution on the withheld row"
    entry = entries[-1]

    assert entry.origin == "excluded"
    assert entry.support == "unsupported"
    assert entry.sources == ()
    assert entry.review_required is True
    assert identity_admission.RULE_NOT_SUPPORTED in entry.reason
    assert "rejected_mapped_ids" in entry.uncertainty

    # and a row the source DOES name is attributed exactly as before
    kept = [
        e for e in read_lineage(_row(result, "heme")).entries if e.stage == "identifier_mapping"
    ]
    assert kept and kept[-1].origin != "excluded"


# ── 7. adversarial arms ──────────────────────────────────────────────────────


def test_a_one_or_two_character_match_does_not_count_as_support(tmp_path: Path) -> None:
    """A two-letter hit is a coincidence, not a citation. The row offers a long
    name the source never writes and a two-character ``short_name`` the source
    does contain by accident; the short one must not rescue it."""
    payload = {
        "entities": {
            "compounds": [{
                "name": "Zorbaline glucoside",
                "short_name": "Zg",
                "class": "compound",
                "mapped_ids": dict(HEME_IDS),
            }],
            "proteins": [], "protein_complexes": [],
        },
        "processes": {"reactions": []},
        SOURCE_INDEX_KEY: source_index("The Zg fraction was resolved by size exclusion."),
    }
    result = run_offline(payload, tmp_path / "cache.json")
    shipped = _row(result, "Zorbaline glucoside").get("mapped_ids") or {}
    assert not shipped.get("kegg")


def test_a_row_offering_only_short_names_is_not_evaluated_and_keeps_its_ids(
    tmp_path: Path,
) -> None:
    """The other half of the same rule, and it points the other way: a name too
    short to look for is no evidence in EITHER direction, so the row is
    ``not_evaluated`` and nothing is taken from it."""
    from t2pw.mapping import identity_admission

    payload = {
        "entities": {
            "compounds": [{"name": "K+", "class": "compound", "mapped_ids": dict(HEME_IDS)}],
            "proteins": [], "protein_complexes": [],
        },
        "processes": {"reactions": []},
        SOURCE_INDEX_KEY: source_index("Potassium efflux was measured in intact cells."),
    }
    result = run_offline(payload, tmp_path / "cache.json")

    assert (_row(result, "K+").get("mapped_ids") or {}).get("kegg") == HEME_IDS["kegg"]
    section = result["report"][identity_admission.REPORT_KEY]
    assert section["counts"]["not_evaluated"] == 1
    assert section["counts"]["identifiers_withheld"] == 0


def test_a_placeholder_backed_row_is_left_alone() -> None:
    """PRODUCT_CONTRACT § 13. An Unknown-backed actor asserts no identity, so
    there is nothing to withhold and the gate must not reach it. Run against the
    two passes directly, because the protein ladder has its own opinion about
    placeholder rows and this assertion is about THIS gate."""
    payload = {
        "entities": {
            "compounds": [],
            "proteins": [{
                "name": "Unknown",
                "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 9659},
                "mapping_meta": {"identity_status": "placeholder"},
            }],
            "protein_complexes": [],
        },
        "processes": {"reactions": []},
        SOURCE_INDEX_KEY: source_index("This paper never names that actor."),
    }
    report = map_ids._admit_identities(payload, payload["entities"])

    assert report["counts"]["rows_examined"] == 0
    assert payload["entities"]["proteins"][0]["mapped_ids"]["pathbank_protein_id"] == 9659


def test_species_and_other_unmapped_buckets_are_outside_the_gate(tmp_path: Path) -> None:
    """The gate is confined to the buckets this mapper resolves identifiers for."""
    payload = {
        "entities": {
            "compounds": [],
            "proteins": [],
            "protein_complexes": [],
            "cell_types": [{"name": "hepatocyte", "mapped_ids": {"chebi": "17627"}}],
        },
        "processes": {"reactions": []},
        SOURCE_INDEX_KEY: source_index("No cell type is named anywhere in this abstract."),
    }
    result = run_offline(payload, tmp_path / "cache.json")
    cell = result["payload"]["entities"]["cell_types"][0]
    assert cell["mapped_ids"]["chebi"] == "17627"


def test_the_source_index_is_never_read_from_an_entity_row(tmp_path: Path) -> None:
    """A row cannot supply its own alibi. The index is a payload-level fact."""
    payload = heme_payload()
    payload["entities"]["compounds"][2][SOURCE_INDEX_KEY] = source_index(SOURCE_WITH_SUCCINYL)
    result = run_offline(payload, tmp_path / "cache.json")
    # no payload-level index -> fail open, unchanged behaviour
    assert (_row(result, "succinyl-CoA").get("mapped_ids") or {}).get("kegg") == "C00091"


def test_the_pass_is_idempotent_over_a_second_mapping_run(tmp_path: Path) -> None:
    """``map_payload`` runs more than once over the same payload with gap
    resolution in between. A second pass must reach the same verdict and must not
    duplicate the record."""
    first = run_offline(
        heme_payload(source_index(SOURCE_WITHOUT_SUCCINYL)), tmp_path / "a.json"
    )
    payload = copy.deepcopy(first["payload"])
    payload[SOURCE_INDEX_KEY] = source_index(SOURCE_WITHOUT_SUCCINYL)
    second = run_offline(payload, tmp_path / "b.json")

    row = _row(second, "succinyl-CoA")
    assert not (row.get("mapped_ids") or {}).get("kegg")
    assert row["name"] == "succinyl-CoA"


# ── the source-index carrier (§ 4a) ──────────────────────────────────────────


def test_screen_additions_writes_the_index_before_any_other_branch() -> None:
    """The Stage-2 merge is the last place that holds the paper. It writes the
    index onto the payload, changing no call signature anywhere."""
    from t2pw.mapping import identity_admission
    from t2pw.pipeline import entity_admission

    payload = {"entities": {"compounds": []}, "processes": {"reactions": []}}
    kept, ledger = entity_admission.screen_additions(payload, source_text=SOURCE_WITH_SUCCINYL)

    index = kept[identity_admission.SOURCE_INDEX_KEY]
    assert index["schema_version"] == identity_admission.SCHEMA_VERSION
    assert index["length"] == len(SOURCE_WITH_SUCCINYL)
    assert "succinyl coa" in index["normalized"]
    assert ledger["source_index"]["status"] == "written"


def test_screen_additions_writes_nothing_when_there_is_no_paper() -> None:
    """Today's production call sites pass no text. Their payload and their ledger
    must be byte-identical to what they were."""
    from t2pw.mapping import identity_admission
    from t2pw.pipeline import entity_admission

    payload = {"entities": {"compounds": []}, "processes": {"reactions": []}}
    kept, ledger = entity_admission.screen_additions(payload)

    assert identity_admission.SOURCE_INDEX_KEY not in kept
    assert "source_index" not in ledger


def test_seed_text_alone_never_writes_the_index() -> None:
    """C-060's A6 says this gate adds no top-level payload key. The index is the
    one addition C-073 charters, and it is written only for a caller that asks
    for it BY NAME -- never as a side effect of the evidence-span parameter."""
    from t2pw.mapping import identity_admission
    from t2pw.pipeline import entity_admission

    payload = {"entities": {"compounds": []}, "processes": {"reactions": []}}
    kept, _ledger = entity_admission.screen_additions(payload, seed_text=SOURCE_WITH_SUCCINYL)

    assert identity_admission.SOURCE_INDEX_KEY not in kept


def test_the_index_survives_the_rest_of_the_merge_and_reaches_map_payload(
    tmp_path: Path,
) -> None:
    """§ 4a end to end over everything ``merge_additions`` does AFTER the gate:
    an index written inside ``screen_additions`` survives
    ``apply_post_merge_cleanup``, travels on the payload the caller keeps, and is
    the thing ``map_payload`` acts on.

    That the key then reaches ``final_mapped.json`` is not asserted here but
    MEASURED on the committed corpus: ``entity_admission_report``, written at this
    same site, is present in every one of run 2026-08-21_2239's ten
    ``final_mapped.json`` files."""
    from t2pw.mapping import identity_admission
    from t2pw.pipeline import entity_admission
    from t2pw.pipeline.pipeline import apply_post_merge_cleanup

    merged, _ledger = entity_admission.screen_additions(
        heme_payload(), source_text=SOURCE_WITHOUT_SUCCINYL
    )
    merged, _removed = apply_post_merge_cleanup(merged)
    assert identity_admission.SOURCE_INDEX_KEY in merged

    result = run_offline(merged, tmp_path / "cache.json")
    assert not (_row(result, "succinyl-CoA").get("mapped_ids") or {}).get("kegg")


def test_the_test_fold_agrees_with_the_production_fold() -> None:
    """Pins this file's independent restatement of the normalization to the
    production one, so the fixtures cannot drift away from the code."""
    from t2pw.mapping import identity_admission

    for sample in (
        "Succinyl-CoA", "2,3-Dihydroxybenzoate", "Fe3+", "ferric-enterobactin",
        "δ-aminolevulinic acid", "Pyridoxal 5'-phosphate", "  padded  ", "",
    ):
        assert _fold(sample) == identity_admission.normalize_text(sample), sample


# ── 8. real-artifact replay ──────────────────────────────────────────────────

CORPUS = ROOT / "runs_verify" / "2026-08-21_2239" / "papers"


@pytest.mark.parametrize(
    "paper,leg,refused,kept",
    [
        (
            "PMC12180156", "research", ["succinyl-CoA"],
            ["heme", "Glycine", "ferrochelatase", "ALAS2", "SFXN4"],
        ),
    ],
)
def test_real_artifact_replay(paper: str, leg: str, refused: List[str], kept: List[str]) -> None:
    """The committed artifact, replayed against the committed source text.

    Runs the two passes directly rather than the whole mapper: the mapping loops
    reach PathBank and UniProt, and this assertion is about the refusal, not about
    re-resolving a run that already happened."""
    from t2pw.mapping import identity_admission

    mapped_path = CORPUS / paper / leg / "final_mapped.json"
    source_path = CORPUS / paper / "01_source_text.txt"
    if not mapped_path.exists() or not source_path.exists():
        pytest.skip(f"run directory absent: {mapped_path}")

    payload = json.loads(mapped_path.read_text(encoding="utf-8"))
    payload[identity_admission.SOURCE_INDEX_KEY] = identity_admission.build_source_index(
        source_path.read_text(encoding="utf-8", errors="replace")
    )
    entities = payload["entities"]
    before = {
        row.get("name"): dict(identity_admission.external_accessions(row))
        for _bucket, row in map_ids._identity_admission_rows(entities)
    }
    report = map_ids._admit_identities(payload, entities)
    after = {
        row.get("name"): dict(identity_admission.external_accessions(row))
        for _bucket, row in map_ids._identity_admission_rows(entities)
    }

    for name in refused:
        assert before[name], f"{name} shipped no accession to begin with"
        assert not after[name], f"{name} still ships {after[name]}"
    for name in kept:
        assert after[name] == before[name], f"{name} lost accessions: collateral"
    withheld = {entry["name"] for entry in report["withheld"]}
    assert withheld == set(refused), f"unexpected withholdings: {withheld}"

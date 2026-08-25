"""C-081 / F-096 -- a declared COFACTOR ROLE that no reaction uses may not ship
a real external accession.

T-106 (``runs_verify/2026-08-24_1428``) failed acceptance priority 1 with eight
false real identifiers. Two of them are ``Pyridoxal 5'-phosphate`` on the two
PMC12856317 legs, shipping ``drugbank:DB00114``, ``hmdb:HMDB0001491``,
``kegg:C00018``, ``chebi:18405`` and ``pubchem:1051``.

Neither pre-existing pass can reach them:

* **Pass A cannot.** The paper says "a PLP-dependent homodimer enzyme" and
  "covalently bound to the PLP cofactor" -- 22 occurrences of ``PLP``. The name
  IS in the source index, so ``identity_support`` returns ``supported``. This was
  measured on the stored artifacts before a line was written.
* **Pass B cannot.** On this run no other row claims those accessions, so there
  is no cross-kind collision.

What the payload does not contain is any REACTION using it. Stage 1 filed it
``class: "cofactor"`` -- a ROLE claim, not a kind claim -- and the only process
naming it is an ``interaction``: ``"pyridoxal 5-phosphate binds ALAS2"`` on the
research leg, ``"pyridoxal 5-phosphate cofactor of ALAS2"`` on strict. The gold
says the same thing in its own words: *"The ALAS2 cofactor. Never a substrate,
never a product, never a protein."*

THE ANTI-COLLATERAL ARM IS THE LOAD-BEARING ONE, exactly as it is in
``test_c073_identity_admission``. This is **not** a rule against cofactors: a
cofactor any reaction uses keeps every accession it has. ``ATP`` declares the
same role nine times across the committed corpus and is used by a reaction every
time. :func:`test_the_whole_committed_corpus_has_zero_collateral` replays the
predicate over all 91 committed ``final_mapped.json`` artifacts and pins the
count.

G9 NOTE. :func:`test_g9_an_unused_cofactor_role_does_not_ship_identifiers` is the
base-failing behavioural proof. It is written to run unchanged on base SHA
``e648287``: it names no symbol this card introduces, builds the payload from
literals, and fails on an assertion about what ``map_payload`` ships -- not on an
import. Tests that exercise this card's new predicate import it inside their own
bodies for exactly that reason.
"""

from __future__ import annotations

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

#: Spelled out as literals so this file states the contract instead of importing
#: it, which is what lets the G9 proof run on the base SHA.
SOURCE_INDEX_KEY = "source_text_index"
INDEX_SCHEMA_VERSION = 1

#: The five ``Pyridoxal 5'-phosphate`` shipped on both T-106 PMC12856317 legs,
#: copied from ``final_mapped.json`` ``/entities/compounds/4``.
PLP_IDS = {
    "drugbank": "DB00114", "hmdb": "HMDB0001491", "kegg": "C00018",
    "chebi": "18405", "pubchem": "1051",
}
#: ATP as PMC12096016 ships it -- the same declared role, used by a reaction.
ATP_IDS = {"hmdb": "HMDB0000538", "kegg": "C00002", "chebi": "15422", "pubchem": "5957"}
NADH_IDS = {"hmdb": "HMDB0001487", "kegg": "C00004", "chebi": "16908", "pubchem": "439153"}
GLYCINE_IDS = {"hmdb": "HMDB0000123", "kegg": "C00037", "chebi": "15428", "pubchem": "750"}
SREBF1_IDS = {"uniprot": "P36956"}

#: PMC12856317's own words, trimmed. ``PLP`` and the spelled-out name both occur,
#: so pass A reports ``supported`` and cannot be what refuses the row.
PLP_SOURCE = (
    "The enzyme that catalyzes the rate-limiting step for heme biosynthesis is "
    "aminolevulinic acid synthase (ALAS), a PLP-dependent homodimer enzyme that "
    "mediates the condensation of glycine and succinyl-CoA to produce aminolevulinic "
    "acid. These studies were performed using the catalytically active holoenzyme that "
    "is covalently bound to the pyridoxal 5-phosphate cofactor. In the absence of heme, "
    "increasing concentrations of PLP restored enzymatic activity."
)


def _fold(text: str) -> str:
    """The normalization the index stores: casefold, punctuation -> one space.

    An independent restatement of the production fold, as
    ``test_c073_identity_admission`` states it, so this file cannot quietly
    diverge from the code it is judging.
    """
    lowered = str(text or "").casefold()
    folded = "".join(ch if ch.isalnum() else " " for ch in lowered)
    return " ".join(folded.split())


def source_index(text: str) -> Dict[str, Any]:
    return {"schema_version": INDEX_SCHEMA_VERSION, "length": len(text),
            "normalized": _fold(text)}


# ── the offline runner ───────────────────────────────────────────────────────


class _NoNetwork(RuntimeError):
    """Raised if anything under test reaches for the network."""


def run_offline(payload: Dict[str, Any], cache_path: Path) -> Dict[str, Any]:
    """``map_payload`` with every external door shut. What comes out is a
    function of the input alone."""
    env = {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0"}
    with patch.dict(os.environ, env), patch.object(
        map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)
    ), patch.object(
        map_ids, "_ai_protein_synonym_lookup", return_value=[]
    ), patch.object(
        map_ids.HttpClient, "get", side_effect=_NoNetwork("network call during an offline run")
    ):
        return map_payload(payload, cache_path=cache_path, id_source="db", use_cache=False)


def _row(result: Dict[str, Any], name: str, bucket: str = "compounds") -> Dict[str, Any]:
    rows = result["payload"]["entities"][bucket]
    matches = [r for r in rows if isinstance(r, dict) and r.get("name") == name]
    assert matches, f"{name} vanished from {bucket}: {[r.get('name') for r in rows]}"
    return matches[0]


def plp_row(**overrides: Any) -> Dict[str, Any]:
    """The row both T-106 PMC12856317 legs shipped, class and synonyms and all."""
    row: Dict[str, Any] = {
        "name": "Pyridoxal 5'-phosphate",
        "raw_name": "pyridoxal 5-phosphate",
        "class": "cofactor",
        "short_name": "Pyr-5'P",
        "synonyms": ["Pyridoxal phosphate", "Pyridoxal 5'-phosphate",
                     "Pyridoxal 5-phosphate", "PLP"],
        "mapped_ids": dict(PLP_IDS),
    }
    row.update(overrides)
    return row


def alas2_payload(
    *,
    cofactor: Optional[Dict[str, Any]] = None,
    reaction_inputs: Optional[List[Any]] = None,
    interactions: Optional[List[Dict[str, Any]]] = None,
    index_text: str = PLP_SOURCE,
    extra_compounds: Optional[List[Dict[str, Any]]] = None,
    proteins: Optional[List[Dict[str, Any]]] = None,
    reactions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The T-106 PMC12856317 shape: one ALAS2 reaction that does NOT use the
    cofactor, plus the interaction that names it."""
    compounds: List[Dict[str, Any]] = [
        {"name": "Glycine", "class": "compound", "raw_name": "glycine",
         "synonyms": ["Glycine", "Gly"], "mapped_ids": dict(GLYCINE_IDS)},
    ]
    if cofactor is not None:
        compounds.append(cofactor)
    compounds.extend(extra_compounds or [])
    if reactions is None:
        reactions = [{
            "name": "ALAS2-catalyzed condensation of glycine and succinyl-CoA",
            "inputs": reaction_inputs if reaction_inputs is not None else ["Glycine"],
            "outputs": ["aminolevulinic acid"],
            "enzymes": [{"entity": "ALAS2", "entity_type": "protein", "role": "catalyst"}],
        }]
    payload: Dict[str, Any] = {
        "entities": {
            "compounds": compounds,
            "proteins": proteins if proteins is not None else [],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": reactions,
            "transports": [],
            "interactions": interactions if interactions is not None else [
                {"name": "pyridoxal 5-phosphate binds ALAS2",
                 "entity_1": "Pyridoxal 5'-phosphate", "entity_2": "ALAS2"}
            ],
        },
    }
    payload[SOURCE_INDEX_KEY] = source_index(index_text)
    return payload


# ── 1. the G9 base-failing behavioural proof ─────────────────────────────────


def test_g9_an_unused_cofactor_role_does_not_ship_identifiers(tmp_path: Path) -> None:
    """THE PROOF, through the production entry point.

    A row declaring ``class: "cofactor"`` whose name IS in the source index, that
    no reaction and no transport uses, and that only an ``interaction``
    references, must not ship ``drugbank``/``hmdb``/``kegg``/``chebi``/
    ``pubchem``.

    On base SHA ``e648287`` ``_admit_identities`` has two passes: pass A asks
    whether the name is locatable in the paper -- it is, 22 times -- and pass B
    asks whether another row of the other kind claims the same accession -- none
    does. So all five ship and this assertion fails. Nothing here names a symbol
    this card adds.
    """
    result = run_offline(alas2_payload(cofactor=plp_row()), tmp_path / "cache.json")
    shipped = _row(result, "Pyridoxal 5'-phosphate").get("mapped_ids") or {}

    assert [ns for ns in PLP_IDS if shipped.get(ns)] == [], (
        "a cofactor-role row that no reaction uses shipped real external "
        f"accessions: {shipped}"
    )


def test_g9_the_withheld_accessions_are_filed_not_deleted(tmp_path: Path) -> None:
    """Merge rule 7. The refusal is recorded, so it can be audited and reversed;
    the identifiers are not dropped on the floor."""
    result = run_offline(alas2_payload(cofactor=plp_row()), tmp_path / "cache.json")
    meta = _row(result, "Pyridoxal 5'-phosphate").get("mapping_meta") or {}
    rejected = meta.get("rejected_mapped_ids") or {}

    for namespace, value in PLP_IDS.items():
        assert rejected.get(namespace) == value, (
            f"{namespace}:{value} was neither shipped nor filed: {rejected}")


# ── 2. legitimate cofactors keep their identifiers ───────────────────────────


def test_a_cofactor_a_reaction_consumes_keeps_every_identifier(tmp_path: Path) -> None:
    """The anti-collateral arm. ATP declares the same role and is used by a
    reaction, so nothing is withheld. This is what makes the rule a rule about
    UNUSED roles rather than a rule against cofactors."""
    atp = {"name": "ATP", "class": "cofactor", "synonyms": ["Adenosine triphosphate"],
           "mapped_ids": dict(ATP_IDS)}
    payload = alas2_payload(
        cofactor=atp, reaction_inputs=["Glycine", "ATP"],
        index_text=PLP_SOURCE + " The adenylation step consumes ATP.",
        interactions=[],
    )
    shipped = _row(run_offline(payload, tmp_path / "cache.json"), "ATP").get("mapped_ids") or {}

    for namespace, value in ATP_IDS.items():
        assert shipped.get(namespace) == value, (
            f"a cofactor a reaction consumes lost {namespace}: {shipped}")


def test_a_cofactor_produced_by_a_reaction_keeps_every_identifier(tmp_path: Path) -> None:
    """NAD+/NADH with explicit paper support AND a reaction role. An OUTPUT is a
    role just as an input is."""
    nadh = {"name": "NADH", "class": "cofactor",
            "synonyms": ["Reduced nicotinamide adenine dinucleotide"],
            "mapped_ids": dict(NADH_IDS)}
    payload = alas2_payload(
        cofactor=nadh, reaction_inputs=["Glycine"],
        reactions=[{"name": "oxidation", "inputs": ["Glycine"], "outputs": ["NADH"]}],
        index_text=PLP_SOURCE + " The dehydrogenase reduces NADH in the matrix.",
        interactions=[],
    )
    shipped = _row(run_offline(payload, tmp_path / "cache.json"), "NADH").get("mapped_ids") or {}

    for namespace, value in NADH_IDS.items():
        assert shipped.get(namespace) == value, f"an output cofactor lost {namespace}: {shipped}"


def test_a_cofactor_named_as_an_enzyme_or_modifier_keeps_its_identifiers() -> None:
    """``enzymes`` and ``modifiers`` are participant roles too. Exercised through
    the pass directly so both actor shapes are covered in one place."""
    for field in ("enzymes", "modifiers"):
        payload = alas2_payload(
            cofactor=plp_row(), interactions=[],
            reactions=[{
                "name": "ALAS2 condensation", "inputs": ["Glycine"],
                "outputs": ["aminolevulinic acid"],
                field: [{"entity": "Pyridoxal 5'-phosphate", "role": "cofactor"}],
            }],
        )
        report = map_ids._admit_identities(payload, payload["entities"])
        assert report["counts"]["cofactor_rows_withheld"] == 0, (
            f"a cofactor named under {field!r} was refused: {report['withheld']}")


def test_a_cofactor_used_by_a_transport_keeps_its_identifiers() -> None:
    """``transports`` carry cargo and transporters, and both are roles."""
    payload = alas2_payload(
        cofactor=plp_row(), interactions=[],
        reactions=[],
    )
    payload["processes"]["transports"] = [
        {"name": "PLP import", "cargo": "Pyridoxal 5'-phosphate",
         "transporters": [{"entity": "SLC25A38"}]}
    ]
    report = map_ids._admit_identities(payload, payload["entities"])
    assert report["counts"]["cofactor_rows_withheld"] == 0, (
        f"a transported cofactor was refused: {report['withheld']}")


def test_a_reaction_naming_the_cofactor_by_an_alias_keeps_it() -> None:
    """Source-supported aliases. The reaction says ``PLP``; the row is named
    ``Pyridoxal 5'-phosphate`` and offers ``PLP`` as a synonym. Same molecule, so
    the role is the row's role."""
    payload = alas2_payload(
        cofactor=plp_row(), interactions=[],
        reactions=[{"name": "ALAS2 condensation", "inputs": ["Glycine", "PLP"],
                    "outputs": ["aminolevulinic acid"]}],
    )
    report = map_ids._admit_identities(payload, payload["entities"])
    assert report["counts"]["cofactor_rows_withheld"] == 0, (
        f"an alias spelling on the reaction side was not recognised: {report['withheld']}")


def test_a_reaction_naming_the_cofactor_by_its_raw_name_keeps_it() -> None:
    """The mapper renames rows to the database's canonical spelling -- T-106's row
    is ``Pyridoxal 5'-phosphate`` with ``raw_name`` ``pyridoxal 5-phosphate`` --
    while the reaction still carries the Stage-1 spelling. That must not read as
    an unused role."""
    payload = alas2_payload(
        cofactor=plp_row(), interactions=[],
        reactions=[{"name": "ALAS2 condensation",
                    "inputs": ["Glycine", "pyridoxal 5-phosphate"],
                    "outputs": ["aminolevulinic acid"]}],
    )
    report = map_ids._admit_identities(payload, payload["entities"])
    assert report["counts"]["cofactor_rows_withheld"] == 0, (
        f"the row's own raw_name on the reaction side was not recognised: {report['withheld']}")


# ── 3. the rule reaches nothing but declared cofactor roles ──────────────────


def test_a_regulatory_protein_is_never_touched() -> None:
    """SREBF1 is a transcription factor: a REGULATOR, extracted as
    ``class: "protein"``, used by no reaction in the T-106 PMC12782028 leg. This
    rule declines it, because a protein row declares no cofactor role. Regulators
    are not globally forbidden here, and this card does not reach them.

    Asserted at ``_admit_identities`` -- the seam this card owns -- rather than
    through ``map_payload``. Offline, with no PathBank resolver, the protein
    mapping loop rewrites ``mapped_ids`` for its own pre-existing reasons; that
    is not this pass's doing and asserting through it would measure the wrong
    thing. :func:`test_the_four_t106_regulator_and_gene_list_proteins_are_untouched`
    makes the same point on the real artifact, where the accessions are real.
    """
    srebf1 = {"name": "SREBF1", "class": "protein",
              "synonyms": ["Sterol regulatory element-binding protein 1", "SREBP1"],
              "mapped_ids": dict(SREBF1_IDS)}
    payload = alas2_payload(
        cofactor=None, proteins=[srebf1], interactions=[],
        index_text=PLP_SOURCE + " The enriched genes including SREBF1 and LBR.",
    )
    report = map_ids._admit_identities(payload, payload["entities"])

    assert report["counts"]["cofactor_rows_examined"] == 0, (
        f"a protein row entered the cofactor pass: {report}")
    assert report["withheld"] == [], f"a regulatory protein was refused: {report['withheld']}"
    assert srebf1["mapped_ids"].get("uniprot") == "P36956", (
        f"a regulatory protein row lost its accession: {srebf1['mapped_ids']}")


def test_the_four_t106_regulator_and_gene_list_proteins_are_untouched() -> None:
    """The other six of T-106's eight, on the real payload.

    ``SREBF1``/``SREBF2`` (``regulator_as_metabolite``) and ``LIPA``/``LBR``
    (``heading_or_prose``) are gold-forbidden and DO carry real UniProt
    accessions in ``runs_verify/2026-08-24_1428``. This card does not reach them
    -- see the report's "what this does NOT fix" -- and this test pins that so
    the boundary is a measured fact rather than a claim.
    """
    path = RUN_1428 / "papers" / "PMC12782028" / "research" / "final_mapped.json"
    if not path.exists():
        pytest.skip(f"T-106 artifact not present: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))

    report = map_ids._admit_identities(payload, payload["entities"])
    assert [e for e in report["withheld"]
            if e["rule"] == "cofactor_role_used_by_no_reaction"] == [], (
        "this card claimed rows it was measured not to reach")

    by_name = {r.get("name"): r for r in payload["entities"]["proteins"]}
    for name, accession in (("SREBF1", "P36956"), ("SREBF2", "Q12772"),
                            ("LIPA", "P38571"), ("LBR", "Q14739")):
        shipped = (by_name[name].get("mapped_ids") or {}).get("uniprot")
        assert shipped == accession, (
            f"{name} lost its accession to a pass that does not own it: {shipped}")


def test_an_ordinary_compound_used_by_no_reaction_is_never_touched() -> None:
    """A ``class: "compound"`` row is a KIND claim, not a role claim. Whether an
    unused compound may ship an accession is a different question and not this
    card's; the pass must abstain rather than answer it."""
    orphan = {"name": "Heme", "class": "compound", "mapped_ids": {"kegg": "C00032"}}
    payload = alas2_payload(cofactor=None, extra_compounds=[orphan], interactions=[])
    report = map_ids._admit_identities(payload, payload["entities"])
    assert report["counts"]["cofactor_rows_examined"] == 0, (
        f"a plain compound row entered the cofactor pass: {report}")
    assert orphan["mapped_ids"].get("kegg") == "C00032"


# ── 4. heading / prose adversarial ───────────────────────────────────────────


@pytest.mark.parametrize("field", ["name", "evidence", "description"])
def test_a_cofactor_named_only_in_process_PROSE_is_still_unused(field: str) -> None:
    """The T-106 strict leg's interaction is literally named ``"pyridoxal
    5-phosphate cofactor of ALAS2"``. A process NAME containing the molecule is
    prose about the process, not a role for the molecule, and reading every
    string under ``processes`` would let that sentence certify the accession."""
    payload = alas2_payload(
        cofactor=plp_row(), interactions=[],
        reactions=[{
            "name": "ALAS2 condensation", "inputs": ["Glycine"],
            "outputs": ["aminolevulinic acid"],
            field: "the pyridoxal 5-phosphate cofactor of ALAS2 is covalently bound",
        }],
    )
    report = map_ids._admit_identities(payload, payload["entities"])
    assert report["counts"]["cofactor_rows_withheld"] == 1, (
        f"prose in a process {field!r} was read as a participant role: {report}")


def test_an_interaction_endpoint_is_not_a_participant_role() -> None:
    """Both T-106 legs reference PLP from exactly one place: an ``interaction``
    endpoint. Binding an enzyme is not being a substrate or a product, which is
    what the gold's "never a substrate, never a product" says."""
    payload = alas2_payload(cofactor=plp_row())
    report = map_ids._admit_identities(payload, payload["entities"])
    withheld = [e for e in report["withheld"] if e["name"] == "Pyridoxal 5'-phosphate"]
    assert len(withheld) == 1, f"the interaction endpoint protected the row: {report}"
    assert withheld[0]["rule"] == "cofactor_role_used_by_no_reaction"


# ── 5. fail open ─────────────────────────────────────────────────────────────


def test_fail_open_a_payload_with_no_reactions_withholds_nothing(tmp_path: Path) -> None:
    """A payload holding no reaction and no transport cannot answer "does a
    reaction use this?". Every row would look unused, so the question is
    ``not_evaluated`` -- which PRODUCT_CONTRACT s.8 says is never ``false``."""
    payload = alas2_payload(cofactor=plp_row(), reactions=[], interactions=[])
    report = map_ids._admit_identities(payload, payload["entities"])

    assert report["counts"]["cofactor_rows_withheld"] == 0, (
        f"a payload with no reactions had identities stripped: {report['withheld']}")
    assert report["cofactor_participation"]["reactions_seen"] is False
    assert report["counts"]["cofactor_not_evaluated"] == 1


def test_fail_open_a_payload_with_no_processes_key_withholds_nothing() -> None:
    """The legacy shape, and the ``interactive_curator`` path."""
    payload = alas2_payload(cofactor=plp_row())
    payload.pop("processes")
    report = map_ids._admit_identities(payload, payload["entities"])
    assert report["counts"]["cofactor_rows_withheld"] == 0, report["withheld"]


def test_a_cofactor_row_shipping_no_accession_is_not_examined() -> None:
    """Nothing to withhold, so nothing to decide. The row keeps its name, its
    class and its graph role either way."""
    payload = alas2_payload(cofactor=plp_row(mapped_ids={}))
    report = map_ids._admit_identities(payload, payload["entities"])
    assert report["counts"]["cofactor_rows_examined"] == 0, report


# ── 6. the refusal is auditable, and the row survives ────────────────────────


def test_the_withheld_row_keeps_its_name_its_class_and_its_graph_role(
        tmp_path: Path) -> None:
    """Merge rule 7 again, from the other side: the entity survives. Only the
    identity claim is withheld."""
    result = run_offline(alas2_payload(cofactor=plp_row()), tmp_path / "cache.json")
    row = _row(result, "Pyridoxal 5'-phosphate")

    assert row["class"] == "cofactor"
    assert row["name"] == "Pyridoxal 5'-phosphate"
    interactions = result["payload"]["processes"]["interactions"]
    assert any(i.get("entity_1") == "Pyridoxal 5'-phosphate" for i in interactions), (
        "the row's graph role was removed along with its identity")


def test_the_refusal_names_its_rule_in_the_mapping_report() -> None:
    """Evidence provenance stays explicit: the report says what was refused, why,
    and how many names were looked for."""
    payload = alas2_payload(cofactor=plp_row())
    report = map_ids._admit_identities(payload, payload["entities"])
    entry = next(e for e in report["withheld"] if e["name"] == "Pyridoxal 5'-phosphate")

    assert entry["rule"] == "cofactor_role_used_by_no_reaction"
    assert entry["bucket"] == "compounds"
    assert entry["identifiers"] == PLP_IDS
    assert entry["names_evaluated"] >= 1
    assert report["counts"]["cofactor_rows_withheld"] == 1


def test_the_refusal_is_recorded_on_the_row_itself() -> None:
    """``mapping_meta.identity_admission`` carries the rule and the detail, so a
    reviewer reading one row can see why it is bare."""
    payload = alas2_payload(cofactor=plp_row())
    map_ids._admit_identities(payload, payload["entities"])
    row = payload["entities"]["compounds"][1]
    record = (row.get("mapping_meta") or {}).get("identity_admission") or {}

    assert "cofactor_role_used_by_no_reaction" in (record.get("rules") or [])
    assert record.get("withheld") == PLP_IDS
    detail = " ".join(d["detail"] for d in record.get("details") or [])
    assert "no reaction or transport" in detail


def test_the_pass_is_idempotent_over_a_second_run() -> None:
    """``map_payload`` runs more than once over the same payload during gap
    resolution. A second refusal must not double-count or erase the first."""
    payload = alas2_payload(cofactor=plp_row())
    first = map_ids._admit_identities(payload, payload["entities"])
    second = map_ids._admit_identities(payload, payload["entities"])

    assert first["counts"]["cofactor_rows_withheld"] == 1
    assert second["counts"]["cofactor_rows_withheld"] == 0, (
        "the second pass re-withheld an already-bare row")
    row = payload["entities"]["compounds"][1]
    assert (row["mapping_meta"]["rejected_mapped_ids"]) == PLP_IDS


# ── 7. the real T-106 artifacts ──────────────────────────────────────────────

RUN_1428 = ROOT / "runs_verify" / "2026-08-24_1428"


@pytest.mark.parametrize("mode", ["strict", "research"])
def test_the_two_t106_legs_are_refused_on_their_own_stored_payload(mode: str) -> None:
    """The stored artifact, not a reconstruction. Both PMC12856317 legs shipped
    ``Pyridoxal 5'-phosphate`` with five accessions; replaying the pass over the
    payload as committed must withhold exactly that row."""
    path = RUN_1428 / "papers" / "PMC12856317" / mode / "final_mapped.json"
    if not path.exists():
        pytest.skip(f"T-106 artifact not present: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))

    report = map_ids._admit_identities(payload, payload["entities"])
    names = [e["name"] for e in report["withheld"]
             if e["rule"] == "cofactor_role_used_by_no_reaction"]

    assert names == ["Pyridoxal 5'-phosphate"], (
        f"{mode}: expected exactly the PLP row to be refused, got {report['withheld']}")


def test_the_whole_committed_corpus_has_zero_collateral() -> None:
    """The measurement that licenses the rule, pinned.

    Replayed over every committed ``final_mapped.json``: the pass refuses only
    rows the pinned gold set forbids by name or by alias, and it refuses no row
    the gold allows. ``ATP`` declares the same role nine times and is used by a
    reaction every time.
    """
    from t2pw.bench.goldset import load_gold_set

    gold = load_gold_set(str(SRC / "t2pw" / "bench" / "gold" / "pinned_v1.json"))
    cases = {c.paper_id: c for c in gold.cases}
    #: Molecules the gold forbids under a DIFFERENT spelling in the same case --
    #: ``pyridoxal 5-phosphate``/``PLP``, ``thiamine diphosphate``/``ThDP``,
    #: ``5,10-methylene-THF``/``tetrahydrofolate``/``THF``. The gold matcher is
    #: exact-name; these are the same molecules it forbids, written with a
    #: parenthetical or without the locant.
    alias_of_a_forbidden_molecule = {
        "pyridoxal phosphate", "thiamine diphosphate (thdp)", "tetrahydrofolate (thf)",
    }

    artifacts = sorted(p for p in ROOT.rglob("final_mapped.json")
                       if ".git" not in p.parts and "worktrees" not in p.parts)
    if len(artifacts) < 20:
        pytest.skip(f"committed run corpus not present (found {len(artifacts)})")

    refused: List[Any] = []
    for path in artifacts:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entities = payload.get("entities")
        if not isinstance(entities, dict):
            continue
        report = map_ids._admit_identities(payload, entities)
        for entry in report["withheld"]:
            if entry["rule"] != "cofactor_role_used_by_no_reaction":
                continue
            case = next((c for pid, c in cases.items() if pid in str(path)), None)
            forbidden = case.forbidden_match(entry["name"]) if case else None
            refused.append((str(path.relative_to(ROOT)), entry["name"], bool(forbidden)))

    collateral = [r for r in refused
                  if not r[2] and r[1].casefold() not in alias_of_a_forbidden_molecule]
    assert collateral == [], f"the pass refused rows the gold allows: {collateral}"
    assert len(refused) >= 15, (
        f"the corpus replay found only {len(refused)} refusals; the measurement "
        f"that licensed this rule found 18")

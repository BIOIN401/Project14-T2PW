"""C-086 / F-116 — a component match must not promote one enzyme to a superset complex.

``_rewrite_reaction_protein_enzymes_to_complexes`` wraps a bare reaction enzyme in
a PathBank ``protein_complex`` because the PathWhiz importer refuses a bare
protein as a reaction enzyme. That rationale is legitimate and is not what these
tests attack. What they pin is the line the rewrite crossed on
``PMC12096016/strict``: it resolved **EntE** onto PathBank complex **3623**
(``EntB P0ADI4``, ``EntD P19925``, ``EntF P11454``, ``EntE P10378``) — a strict
superset — so the 2,3-DHB adenylation step (EC 6.2.1.71, EntE alone) shipped
three catalysts that do not perform it, and ``reactions[4]``
("EntF-catalyzed enterobactin synthesis") collapsed onto the *same* actor, making
two chemically distinct steps indistinguishable.

Every fixture below is measured, not invented. The enterobactin numbers are read
out of the committed artifact
``runs_verify/2026-08-24_1428/papers/PMC12096016/strict/final_mapped.json``
(``entities.protein_complexes[3]`` for complex 3623 and its four components;
``[0]``/``[1]``/``[2]`` for the one-component siblings 1143 / 1189 / 1190) and the
reaction shapes out of ``merged_payload.json`` beside it, which still shows the
five reactions carrying **bare proteins** as enzymes before mapping ran.

**G9 note.** Nothing here imports a symbol added by this card. Every assertion
runs through ``_rewrite_reaction_protein_enzymes_to_complexes``, which exists on
the base SHA, so the base failure is behavioural rather than an ImportError.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping.map_ids import (  # noqa: E402
    _rewrite_reaction_protein_enzymes_to_complexes,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
class _MemoryCache:
    def __init__(self) -> None:
        self.rows: Dict[tuple, Dict[str, Any]] = {}

    def get(self, namespace: str, key: str) -> Dict[str, Any] | None:
        return self.rows.get((namespace, key))

    def set(self, namespace: str, key: str, value: Dict[str, Any]) -> None:
        self.rows[(namespace, key)] = value


class _StubDb:
    """A PathBank stand-in that answers exactly what the real resolver answered.

    ``map_enzyme_protein_to_complex`` is the seam under test's only database
    call. It asks "which complexes list this protein as a *component*", which is
    why a superset can come back at all, and it is stubbed here with the rows the
    committed artifact proves the live database returned.
    """

    last_error = ""

    def __init__(self, by_protein: Dict[str, Dict[str, Any]]) -> None:
        self.by_protein = by_protein
        self.calls: List[str] = []

    def available(self) -> bool:
        return True

    def map_enzyme_protein_to_complex(
        self, protein_row: Dict[str, Any], species: str
    ) -> Dict[str, Any]:
        name = str(protein_row.get("name") or "")
        self.calls.append(name)
        answer = self.by_protein.get(name)
        assert answer is not None, f"unexpected enzyme lookup for {name!r}"
        # A fresh copy per call: the production cache stores what it is handed.
        return {
            **answer,
            "components": [dict(component) for component in answer["components"]],
        }


def _component(name: str, uniprot: str, pathbank_protein_id: int, gene: str) -> Dict[str, Any]:
    return {
        "name": name,
        "pathbank_protein_id": pathbank_protein_id,
        "stoichiometry": 1,
        "uniprot": uniprot,
        "mapped_ids": {"uniprot": uniprot, "pathbank_protein_id": str(pathbank_protein_id)},
        "gene_name": gene,
        "species_id": 3,
    }


#: ``entities.protein_complexes[3].components`` of the committed artifact.
_ENT_3623_COMPONENTS = [
    _component("EntB", "P0ADI4", 6224, "entB"),
    _component("EntD", "P19925", 6383, "entD"),
    _component("EntF", "P11454", 6312, "entF"),
    _component("EntE", "P10378", 6301, "entE"),
]


def _db_complex(
    name: str, complex_id: int, components: List[Dict[str, Any]], species_id: int = 3
) -> Dict[str, Any]:
    """The ``_complex_result_from_row`` shape a component match returns."""
    return {
        "status": "mapped",
        "provider": "PathBankDB",
        "source": "db",
        "name": name,
        "pathbank_complex_id": complex_id,
        "pathbank_protein_complex_id": complex_id,
        "species_id": species_id,
        "components": components,
        "confidence": 0.9,
        "chosen_rule": "enzyme_component_species",
        "candidates": [],
        "issues": [],
        "resolution": {"status": "matched", "order_step": "enzyme_component_species"},
    }


_ENTEROBACTIN_DB = {
    "EntC": _db_complex(
        "Isochorismate synthase", 1143, [_component("EntC", "P0AEJ2", 6238, "entC")]
    ),
    "EntB": _db_complex(
        "isochorismatase", 1189, [_component("EntB", "P0ADI4", 6224, "entB")]
    ),
    "EntA": _db_complex(
        "oxidoreductase (entA)", 1190, [_component("EntA", "P15047", 6341, "entA")]
    ),
    "EntE": _db_complex("enterobactin synthase", 3623, _ENT_3623_COMPONENTS),
    "EntF": _db_complex("enterobactin synthase", 3623, _ENT_3623_COMPONENTS),
}


def _protein(name: str, uniprot: str, pathbank_protein_id: int) -> Dict[str, Any]:
    return {
        "name": name,
        "species": "Escherichia coli",
        "species_id": 3,
        "pathbank_protein_id": pathbank_protein_id,
        "mapped_ids": {"uniprot": uniprot, "pathbank_protein_id": str(pathbank_protein_id)},
        "species_ref": {
            "name": "Escherichia coli",
            "pathbank_species_id": 3,
            "species_id": 3,
            "taxonomy_id": "562",
        },
    }


def _enterobactin_payload() -> Dict[str, Any]:
    """PMC12096016's five reactions as ``merged_payload.json`` records them."""
    return {
        "entities": {
            "proteins": [
                _protein("EntB", "P0ADI4", 6224),
                _protein("EntC", "P0AEJ2", 6238),
                _protein("EntA", "P15047", 6341),
                _protein("EntE", "P10378", 6301),
                _protein("EntF", "P11454", 6312),
                _protein("EntD", "P19925", 6383),
            ],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "EntC-catalyzed isochorismate formation",
                    "enzymes": [{"protein": "EntC", "provenance": "extracted", "confidence": 1.0}],
                },
                {
                    "name": "EntB isochorismatase reaction",
                    "enzymes": [{"protein": "EntB", "provenance": "extracted", "confidence": 1.0}],
                },
                {
                    "name": "EntA-catalyzed 2,3-diDHB dehydrogenation",
                    "enzymes": [{"protein": "EntA", "provenance": "extracted", "confidence": 1.0}],
                },
                {
                    "name": "EntE-catalyzed 2,3-DHB adenylation",
                    "enzymes": [{"protein": "EntE", "provenance": "extracted", "confidence": 1.0}],
                },
                {
                    "name": "EntF-catalyzed enterobactin synthesis",
                    "enzymes": [
                        {"protein": "EntF", "provenance": "extracted", "confidence": 1.0},
                        {"protein": "EntE", "provenance": "inferred", "confidence": 0.9},
                    ],
                },
            ]
        },
    }


def _run(mapped: Dict[str, Any], db: Any) -> Dict[str, Any]:
    return _rewrite_reaction_protein_enzymes_to_complexes(
        mapped,
        db=db,  # type: ignore[arg-type]
        cache=_MemoryCache(),  # type: ignore[arg-type]
        global_organism="Escherichia coli",
    )


def _actors(mapped: Dict[str, Any], reaction_index: int) -> List[str]:
    return [
        str(row.get("entity") or "")
        for row in mapped["processes"]["reactions"][reaction_index]["enzymes"]
    ]


def _complexes_by_name(mapped: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("name") or ""): row for row in mapped["entities"]["protein_complexes"]}


# --------------------------------------------------------------------------- #
# Fixture 1 — EntE must no longer be promoted onto superset complex 3623
# --------------------------------------------------------------------------- #
def test_ente_is_not_promoted_onto_the_superset_enterobactin_synthase_complex() -> None:
    """Base behaviour: ``reactions[3].enzymes[0].entity == 'enterobactin synthase'``.

    Catches the F-116 defect itself — the 2,3-DHB adenylation step acquiring
    EntB, EntD and EntF as catalysts because EntE is merely a *component* of the
    complex that carries them.
    """
    mapped = _enterobactin_payload()
    _run(mapped, _StubDb(_ENTEROBACTIN_DB))

    assert _actors(mapped, 3) == ["EntE complex"]

    complexes = _complexes_by_name(mapped)
    assert "enterobactin synthase" not in complexes, (
        "the superset complex must not be materialised for a single-subunit step"
    )
    wrapper = complexes["EntE complex"]
    assert [component["name"] for component in wrapper["components"]] == ["EntE"]
    assert wrapper["components"][0]["mapped_ids"]["uniprot"] == "P10378"
    # It is a generated PathWhiz wrapper, not a claim to be PathBank complex 3623.
    assert wrapper["generated"] is True
    assert wrapper["generation_reason"] == "single_protein_pathwhiz_wrapper"
    assert "pathbank_protein_complex_id" not in wrapper
    assert "pathbank_complex_id" not in wrapper


def test_refused_promotion_is_recorded_with_the_components_it_would_have_injected() -> None:
    """The audit trail the refusal owes a reviewer.

    Catches a silent fix: a refusal that changes the actor but leaves no record
    of which complex was refused or which catalysts it would have added. Also
    the **non-vacuity** proof for fixtures 1 and 2 — the new branch is shown to
    have executed once per superset actor, three actors across two reactions,
    each pinned to the actor slot it refused.
    """
    mapped = _enterobactin_payload()
    report = _run(mapped, _StubDb(_ENTEROBACTIN_DB))

    # Asserted FIRST and on keys the base SHA already emits, so this test's base
    # failure is a wrong VALUE, never a missing key. Measured on the base tree
    # (G11 C-086/05-base-actor-probe.json, same fixture):
    #   db_matched 5 · novel 0 · unresolved 1 · duplicate_actors_merged 1
    # Three of those five "confident database matches" were the defect, and the
    # merge of one was reactions[3] and reactions[4] collapsing onto one actor.
    summary = report["summary"]
    assert summary["reaction_protein_enzymes_rewritten_to_complexes"] == 6
    assert summary["reaction_enzyme_complexes_db_matched"] == 3
    assert summary["reaction_enzyme_complexes_novel"] == 2
    assert summary["reaction_enzyme_complexes_unresolved"] == 1
    assert "reaction_enzyme_duplicate_actors_merged" not in summary

    assert summary["reaction_enzyme_complex_superset_promotions_refused"] == 3

    refusals = [
        action
        for action in report["actions"]
        if action["type"] == "reaction_enzyme_complex_superset_promotion_refused"
    ]
    assert [action["json_pointer"] for action in refusals] == [
        "/processes/reactions/3/enzymes/0",
        "/processes/reactions/4/enzymes/0",
        "/processes/reactions/4/enzymes/1",
    ]
    assert [action["protein"] for action in refusals] == ["EntE", "EntF", "EntE"]
    assert {action["pathbank_protein_complex_id"] for action in refusals} == {3623}
    assert {action["protein_complex"] for action in refusals} == {"enterobactin synthase"}
    # EntE alone catalyses reactions[3]; EntB, EntD and EntF are the strangers.
    assert refusals[0]["uncovered_components"] == ["EntB", "EntD", "EntF"]
    # reactions[4] names EntF and EntE, so only EntB and EntD are unaccounted for
    # -- partial coverage is still a superset promotion and is still refused.
    assert refusals[1]["uncovered_components"] == ["EntB", "EntD"]
    assert refusals[2]["uncovered_components"] == ["EntB", "EntD"]


# --------------------------------------------------------------------------- #
# Fixture 2 — the two-steps-collapse case
# --------------------------------------------------------------------------- #
def test_two_chemically_distinct_steps_stay_distinguishable_by_actor() -> None:
    """Base behaviour: reactions 3 and 4 both read ``['enterobactin synthase']``.

    Catches the collapse the gold's "a named enzyme per step" requirement
    forbids: EC 6.2.1.71 adenylation and EntF-catalysed enterobactin synthesis
    being told apart by nothing.
    """
    mapped = _enterobactin_payload()
    _run(mapped, _StubDb(_ENTEROBACTIN_DB))

    adenylation = _actors(mapped, 3)
    synthesis = _actors(mapped, 4)
    assert adenylation == ["EntE complex"]
    assert synthesis == ["EntF complex", "EntE complex"]
    assert set(adenylation) != set(synthesis)
    # The enzyme the paper named for each step is the one that survives on it.
    assert synthesis[0] == "EntF complex"


# --------------------------------------------------------------------------- #
# Fixture 3 — the one-component controls, which must be untouched
# --------------------------------------------------------------------------- #
def test_one_component_sibling_wrappers_still_resolve_to_their_pathbank_complexes() -> None:
    """EntC->1143, EntB->1189, EntA->1190 are the control set.

    Catches an over-broad fix that refuses every component match and destroys
    legitimate component-to-complex resolution corpus-wide.

    **This test passes on the base SHA as well as at the tip, by design.** It is
    a control, not a proof: its whole value is that the correction leaves it
    exactly where it was, so it asserts nothing that only the corrected tree can
    satisfy.
    """
    mapped = _enterobactin_payload()
    _run(mapped, _StubDb(_ENTEROBACTIN_DB))

    assert _actors(mapped, 0) == ["Isochorismate synthase"]
    assert _actors(mapped, 1) == ["isochorismatase"]
    assert _actors(mapped, 2) == ["oxidoreductase (entA)"]

    complexes = _complexes_by_name(mapped)
    for complex_name, complex_id, component in (
        ("Isochorismate synthase", 1143, "EntC"),
        ("isochorismatase", 1189, "EntB"),
        ("oxidoreductase (entA)", 1190, "EntA"),
    ):
        row = complexes[complex_name]
        assert row["pathbank_protein_complex_id"] == complex_id
        # Still a real database match, not a generated wrapper standing in.
        assert row["mapping_meta"]["resolution"]["status"] == "matched"
        assert row["mapping_meta"]["chosen_rule"] == "enzyme_component_species"
        assert row.get("generated") is not True
        assert [entry["name"] for entry in row["components"]] == [component]


def test_a_component_named_only_by_its_database_name_still_counts_as_the_enzyme() -> None:
    """``hexokinase`` -> complex whose one component is ``Hexokinase-3`` (P52790).

    Catches a name-only coverage test. The component and the payload protein
    share no normalized name; only the UniProt accession says they are the same
    protein, and a fix that missed that would refuse a legitimate one-component
    wrapper — the exact shape
    ``test_glycolysis_reaction_enzyme_complex_components_reconcile_to_local_proteins``
    already pins.
    """
    mapped = {
        "entities": {
            "proteins": [
                {
                    "name": "hexokinase",
                    "species": "Homo sapiens",
                    "species_id": 1,
                    "pathbank_protein_id": 206,
                    "mapped_ids": {"uniprot": "P52790", "pathbank_protein_id": "206"},
                }
            ],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [{"name": "hexokinase reaction", "enzymes": [{"protein": "hexokinase"}]}]
        },
    }
    db = _StubDb(
        {
            "hexokinase": _db_complex(
                "hexokinase complex",
                431773,
                [
                    {
                        "name": "Hexokinase-3",
                        "uniprot": "P52790",
                        "pathbank_protein_id": 161288,
                        "stoichiometry": 1,
                    }
                ],
                species_id=1,
            )
        }
    )
    report = _rewrite_reaction_protein_enzymes_to_complexes(
        mapped, db=db, cache=_MemoryCache(), global_organism="Homo sapiens"  # type: ignore[arg-type]
    )

    assert _actors(mapped, 0) == ["hexokinase complex"]
    assert _complexes_by_name(mapped)["hexokinase complex"]["pathbank_protein_complex_id"] == 431773
    assert "reaction_enzyme_complex_superset_promotions_refused" not in report["summary"]


def test_a_database_complex_with_no_components_is_not_treated_as_a_superset() -> None:
    """Nothing is injected, so nothing is refused.

    Catches a fix that keys off "the complex is not one component" instead of
    "the complex carries components this reaction does not name", which would
    change the unrelated component-less-complex path (its own finding, not this
    card's) as a side effect.
    """
    mapped = {
        "entities": {
            "proteins": [
                {
                    "name": "NdmA",
                    "species": "Pseudomonas putida",
                    "species_id": 541,
                    "mapped_ids": {"uniprot": "A0A000"},
                }
            ],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [{"name": "caffeine demethylation", "enzymes": [{"protein": "NdmA"}]}]
        },
    }
    db = _StubDb({"NdmA": _db_complex("NdmA oxygenase", 4242, [], species_id=541)})
    report = _rewrite_reaction_protein_enzymes_to_complexes(
        mapped, db=db, cache=_MemoryCache(), global_organism="Pseudomonas putida"  # type: ignore[arg-type]
    )

    assert _actors(mapped, 0) == ["NdmA oxygenase"]
    assert "reaction_enzyme_complex_superset_promotions_refused" not in report["summary"]


# --------------------------------------------------------------------------- #
# Fixture 4 — an unrelated complex a reaction genuinely uses in full
# --------------------------------------------------------------------------- #
_MPC_COMPONENTS = [
    {
        "name": "MPC1",
        "uniprot": "Q9Y5U8",
        "pathbank_protein_id": 11,
        "stoichiometry": 1,
        "mapped_ids": {"uniprot": "Q9Y5U8", "pathbank_protein_id": "11"},
    },
    {
        "name": "MPC2",
        "uniprot": "O95563",
        "pathbank_protein_id": 12,
        "stoichiometry": 1,
        "mapped_ids": {"uniprot": "O95563", "pathbank_protein_id": "12"},
    },
]


def _mpc_payload(enzyme_names: List[str]) -> Dict[str, Any]:
    return {
        "entities": {
            "proteins": [
                {
                    "name": "MPC1",
                    "species": "Homo sapiens",
                    "species_id": 1,
                    "pathbank_protein_id": 11,
                    "mapped_ids": {"uniprot": "Q9Y5U8", "pathbank_protein_id": "11"},
                },
                {
                    "name": "MPC2",
                    "species": "Homo sapiens",
                    "species_id": 1,
                    "pathbank_protein_id": 12,
                    "mapped_ids": {"uniprot": "O95563", "pathbank_protein_id": "12"},
                },
            ],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "mitochondrial pyruvate import",
                    "enzymes": [{"protein": name} for name in enzyme_names],
                }
            ]
        },
    }


def _mpc_db() -> _StubDb:
    shared = _db_complex("MPC complex", 301, _MPC_COMPONENTS, species_id=1)
    return _StubDb({"MPC1": shared, "MPC2": shared})


def test_a_reaction_that_names_the_whole_assembly_keeps_its_multi_component_complex() -> None:
    """The preservation half of the ruling.

    Catches a fix that refuses every multi-component complex. Both subunits of
    the heterodimer are named by this reaction, so the complex is what performs
    the step and must survive — collapsing to a single actor, as it did before.
    """
    mapped = _mpc_payload(["MPC1", "MPC2"])
    report = _rewrite_reaction_protein_enzymes_to_complexes(
        mapped, db=_mpc_db(), cache=_MemoryCache(), global_organism="Homo sapiens"  # type: ignore[arg-type]
    )

    assert _actors(mapped, 0) == ["MPC complex"]
    complex_row = _complexes_by_name(mapped)["MPC complex"]
    assert complex_row["pathbank_protein_complex_id"] == 301
    assert [component["name"] for component in complex_row["components"]] == ["MPC1", "MPC2"]
    assert "reaction_enzyme_complex_superset_promotions_refused" not in report["summary"]


def test_the_same_complex_is_refused_when_the_reaction_names_only_one_subunit() -> None:
    """Same database, same complex, one variable changed: what the reaction names.

    This is the A/B that proves the decision is *"does this reaction account for
    the complex"* and not *"is this complex multi-component"*, and it is the
    non-vacuity proof for fixture 4 — the preceding test's pass is not the branch
    being unreachable.
    """
    mapped = _mpc_payload(["MPC1"])
    report = _rewrite_reaction_protein_enzymes_to_complexes(
        mapped, db=_mpc_db(), cache=_MemoryCache(), global_organism="Homo sapiens"  # type: ignore[arg-type]
    )

    assert _actors(mapped, 0) == ["MPC1 complex"]
    assert "MPC complex" not in _complexes_by_name(mapped)
    wrapper = _complexes_by_name(mapped)["MPC1 complex"]
    assert [component["name"] for component in wrapper["components"]] == ["MPC1"]
    assert report["summary"]["reaction_enzyme_complex_superset_promotions_refused"] == 1
    refusal = [
        action
        for action in report["actions"]
        if action["type"] == "reaction_enzyme_complex_superset_promotion_refused"
    ][0]
    assert refusal["uncovered_components"] == ["MPC2"]
    assert refusal["pathbank_protein_complex_id"] == 301


def test_one_enzyme_can_be_refused_in_one_reaction_and_kept_in_another() -> None:
    """The per-reaction memo, not a per-protein one.

    Catches a fix that caches the first reaction's verdict for a protein and
    applies it to every later reaction — which would make the answer depend on
    reaction order rather than on what each reaction names.
    """
    mapped = _mpc_payload(["MPC1"])
    mapped["processes"]["reactions"].append(
        {
            "name": "pyruvate import by the intact carrier",
            "enzymes": [{"protein": "MPC1"}, {"protein": "MPC2"}],
        }
    )
    report = _rewrite_reaction_protein_enzymes_to_complexes(
        mapped, db=_mpc_db(), cache=_MemoryCache(), global_organism="Homo sapiens"  # type: ignore[arg-type]
    )

    assert _actors(mapped, 0) == ["MPC1 complex"]
    assert _actors(mapped, 1) == ["MPC complex"]
    assert report["summary"]["reaction_enzyme_complex_superset_promotions_refused"] == 1
    names = _complexes_by_name(mapped)
    assert names["MPC1 complex"]["generated"] is True
    assert names["MPC complex"]["pathbank_protein_complex_id"] == 301

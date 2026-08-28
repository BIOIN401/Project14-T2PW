"""C-095 / F-133 — a generated one-protein wrapper must not inherit a superset complex.

F-133 is the remaining open path of F-116. C-086 closed the **component-match**
path inside ``_rewrite_reaction_protein_enzymes_to_complexes``; its tests pin that
path working and the live cohort confirmed it. One stage later the generic
complex-mapping loop in ``map_payload`` re-resolves **every** ``protein_complex``
row, the generated wrappers included, and
``PathBankDbResolver.map_protein_complex_row``'s ``resolved_component_species``
rule accepts a single component candidate as ``matched`` **without checking that
the candidate's membership matches the row's**. On ``PMC12096016/strict`` that
gave ``EntF complex`` and ``EntD complex`` — one-protein technical wrappers —
PathBank complex **3623** and all four of its components.

Every fixture below is measured, not invented. The PathBank rows are read out of
the committed artifact
``runs_verify/2026-08-27_1341/papers/PMC12096016/strict/final_mapped.json``:
complex 3623 ``enterobactin synthase`` with components ``EntB P0ADI4 6224``,
``EntD P19925 6383``, ``EntF P11454 6312``, ``EntE P10378 6301``; the
one-component siblings 1143 ``Isochorismate synthase`` [``EntC`` 6238], 1189
``isochorismatase`` [``EntB`` 6224], 1190 ``oxidoreductase (entA)`` [``EntA``
6341]; and the ten complexes whose names made ``EntE complex``'s name lookup
ambiguous (its ten recorded candidates, all matching ``%EntE%``).

**The resolver is real.** ``_SqlitePathBank`` subclasses the production
``PathBankDbResolver`` and replaces **only** ``_query`` — the MySQL cursor — with
sqlite over those rows. Every scoring, ordering and admission decision under test
is the shipped one, so ``resolved_component_species`` is genuinely reached rather
than asserted about. The harness reproduces the committed artifact row for row on
the base SHA, including ``EntE complex``'s abstention.

**§ 4 warning, obeyed.** ``EntE complex`` escaped 3623 by ABSTAINING on an
ambiguous name lookup at ``complex_name_species``; it never reaches
``resolved_component_species`` and would pass with this card's fix entirely
absent. It appears below only as a *preservation* case
(:func:`test_ente_wrapper_still_abstains_on_the_ambiguous_name_lookup`), and its
own assertion records that it stopped at ``complex_name_species``. The
non-vacuity proofs are ``EntF``/``EntD``, which reach
``resolved_component_species`` and are refused there.

**G9 note.** Nothing here imports a symbol this card adds. Every assertion runs
through ``map_payload``, which exists on the base SHA ``0128fa6``, so the base
failure is behavioural rather than an ``ImportError``.
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping import map_ids  # noqa: E402


# --------------------------------------------------------------------------- #
# The PathBank fixture — measured rows, real resolver
# --------------------------------------------------------------------------- #
#: ``species`` row 3, the one every candidate in the artifact carries.
SPECIES_ROWS: List[Tuple[int, str, str, str]] = [
    (3, "Escherichia coli", "E. coli", "562"),
]

#: ``proteins``: id, name, uniprot_id, gene_name, species_id.
PROTEIN_ROWS: List[Tuple[int, str, str, str, int]] = [
    (6224, "EntB", "P0ADI4", "entB", 3),
    (6238, "EntC", "P0AEJ2", "entC", 3),
    (6301, "EntE", "P10378", "entE", 3),
    (6312, "EntF", "P11454", "entF", 3),
    (6341, "EntA", "P15047", "entA", 3),
    (6383, "EntD", "P19925", "entD", 3),
]

#: ``protein_complexes``. 1143/1189/1190/3623 are the artifact's own rows; the
#: rest are ``EntE complex``'s other nine recorded candidates, present so the
#: § 4 abstention is reproduced rather than assumed.
COMPLEX_ROWS: List[Tuple[int, str, int]] = [
    (1143, "Isochorismate synthase", 3),
    (1189, "isochorismatase", 3),
    (1190, "oxidoreductase (entA)", 3),
    (3621, "enterobactin efflux transporter EntS", 3),
    (3622, "EntS-TolC Enterobactin Efflux Transport System", 3),
    (3623, "enterobactin synthase", 3),
    (3624, "ferric enterobactin outer membrane transport complex", 3),
    (3625, "ferric enterobactin ABC transporter", 3),
    (3815, "isopentenyl diphosphate isomerase", 3),
    (3929, "enterobactin B", 3),
    (3939, "Enterochelin esterase", 3),
    (422867, "Hemolytic enterotoxin HBL binding subunit HblA", 3),
    (422868, "Hemolytic enterotoxin HBL binding subunit HblB", 3),
]

#: ``protein_complex_proteins``. 3623 is the four-protein superset; the three
#: siblings really do have one component each — that split is what makes a narrow
#: fix available and it is measured, not argued.
MEMBERSHIP_ROWS: List[Tuple[int, int]] = [
    (1143, 6238),
    (1189, 6224),
    (1190, 6341),
    (3623, 6224),
    (3623, 6383),
    (3623, 6312),
    (3623, 6301),
]

SUPERSET_COMPONENTS = ["EntB", "EntD", "EntF", "EntE"]


def _sqlite_pathbank() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE species (
            id INTEGER PRIMARY KEY, name TEXT, common_name TEXT, taxonomy_id TEXT);
        CREATE TABLE proteins (
            id INTEGER PRIMARY KEY, name TEXT, uniprot_id TEXT, gene_name TEXT,
            species_id INTEGER, synonyms TEXT);
        CREATE TABLE protein_complexes (
            id INTEGER PRIMARY KEY, name TEXT, species_id INTEGER);
        CREATE TABLE protein_complex_proteins (
            protein_complex_id INTEGER, protein_id INTEGER);
        """
    )
    conn.executemany(
        "INSERT INTO species (id,name,common_name,taxonomy_id) VALUES (?,?,?,?)", SPECIES_ROWS
    )
    conn.executemany(
        "INSERT INTO proteins (id,name,uniprot_id,gene_name,species_id) VALUES (?,?,?,?,?)",
        PROTEIN_ROWS,
    )
    conn.executemany(
        "INSERT INTO protein_complexes (id,name,species_id) VALUES (?,?,?)", COMPLEX_ROWS
    )
    conn.executemany(
        "INSERT INTO protein_complex_proteins (protein_complex_id,protein_id) VALUES (?,?)",
        MEMBERSHIP_ROWS,
    )
    return conn


class _SqlitePathBank(map_ids.PathBankDbResolver):
    """The production resolver with sqlite behind ``_query``.

    Only the cursor is replaced. ``map_protein_complex_row``,
    ``map_protein_complex``, ``_find_complexes_by_component_protein_id``,
    ``_complex_result_from_row`` and the scoring inside them are the shipped code,
    so a test that says a row reached ``resolved_component_species`` is reporting
    the real rule firing.

    A query sqlite cannot answer is recorded rather than swallowed: production
    ``_query`` returns ``[]`` on any exception, which would silently turn a
    fixture mistake into "the database had nothing to say".
    """

    def __init__(self) -> None:
        super().__init__(host="fixture", port=0, user="fixture", password="", schema="pathbank")
        self._sqlite = _sqlite_pathbank()
        self.failed_queries: List[Tuple[str, str]] = []

    def available(self) -> bool:
        return True

    def _query(self, sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        try:
            cursor = self._sqlite.execute(sql.replace("%s", "?"), tuple(params))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:  # noqa: BLE001
            self.failed_queries.append((sql, repr(exc)))
            return []


class _NoNetwork(RuntimeError):
    """Raised if anything under test reaches for the network."""


def run_offline(
    payload: Dict[str, Any],
    cache_path: Path,
    *,
    module: ModuleType | None = None,
    allow_wrappers: bool = False,
) -> Dict[str, Any]:
    """``map_payload`` with every external door shut except the fixture database.

    The same convention ``test_c073_identity_admission.run_offline`` uses: no LLM,
    no NCBI, an ``HttpClient`` that raises. ``module`` selects which copy of
    ``map_ids`` runs, so the mutant below can be driven through the identical
    fixture.
    """

    target = module or map_ids
    db = _SqlitePathBank()
    env = {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0", "T2PW_OFFLINE_CURATOR": "1"}
    with patch.dict(os.environ, env), patch.object(
        target.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: db)
    ), patch.object(
        target, "_ai_protein_synonym_lookup", return_value=[]
    ), patch.object(
        target.HttpClient, "get", side_effect=_NoNetwork("network call during an offline run")
    ):
        out = target.map_payload(
            payload,
            cache_path=cache_path,
            id_source="db",
            use_cache=False,
            allow_complex_wrapper_creation=allow_wrappers,
        )
    assert db.failed_queries == [], f"the fixture database could not answer: {db.failed_queries}"
    return out


# --------------------------------------------------------------------------- #
# Payload fixtures
# --------------------------------------------------------------------------- #
def _protein(pathbank_id: int, name: str, uniprot: str) -> Dict[str, Any]:
    return {
        "name": name,
        "species": "Escherichia coli",
        "organism": "Escherichia coli",
        "pathbank_protein_id": pathbank_id,
        "mapped_ids": {"uniprot": uniprot, "pathbank_protein_id": str(pathbank_id)},
    }


def _component(pathbank_id: int, name: str, uniprot: str) -> Dict[str, Any]:
    return {
        "name": name,
        "stoichiometry": 1,
        "pathbank_protein_id": pathbank_id,
        "mapped_ids": {"uniprot": uniprot, "pathbank_protein_id": str(pathbank_id)},
    }


def _bare_component(name: str) -> Dict[str, Any]:
    """A declared component carrying a NAME and nothing else.

    This is the shape that makes the two seams disagree: with no accession the
    only key reconciliation can match on is the payload's alias list, and it then
    RENAMES the component. A component that carries a uniprot or a
    ``pathbank_protein_id`` is covered by those tokens whatever it is called, so
    it cannot reproduce the defect -- an earlier version of these two tests used
    one and passed at ``194d6cd``, proving nothing.
    """

    return {"name": name, "stoichiometry": 1}


def _wrapper_row(name: str, components: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A generated single-protein wrapper exactly as the artifact records one."""

    return {
        "name": name,
        "generated": True,
        "generation_reason": "single_protein_pathwhiz_wrapper",
        "species": "Escherichia coli",
        "components": components,
    }


def _declared_row(name: str, components: List[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    """A DECLARED complex row: no ``generated`` flag, no wrapper reason."""

    row: Dict[str, Any] = {
        "name": name,
        "species": "Escherichia coli",
        "components": components,
    }
    row.update(extra)
    return row


ALL_PROTEINS = [_protein(pid, name, uniprot) for pid, name, uniprot, _gene, _sp in PROTEIN_ROWS]


def _payload(complex_rows: List[Dict[str, Any]], reactions: List[Dict[str, Any]] | None = None):
    return {
        "metadata": {
            "pathway_name": "Enterobactin biosynthesis",
            "pathway_subject": "Metabolic",
            "organism": "Escherichia coli",
        },
        "entities": {
            "species": [
                {"name": "Escherichia coli", "taxonomy_id": "562", "pathbank_species_id": 3}
            ],
            "compounds": [],
            "proteins": [dict(row) for row in ALL_PROTEINS],
            "protein_complexes": complex_rows,
        },
        "processes": {"reactions": reactions or [], "transports": [], "interactions": []},
    }


def _row(result: Dict[str, Any], name: str) -> Dict[str, Any]:
    for row in result["payload"]["entities"]["protein_complexes"]:
        if isinstance(row, dict) and str(row.get("name") or "") == name:
            return row
    raise AssertionError(f"no protein_complex row named {name!r}")


def _names(row: Dict[str, Any]) -> List[str]:
    return [
        str(component.get("name") or "")
        for component in (row.get("components") or [])
        if isinstance(component, dict)
    ]


def _meta(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("mapping_meta") or {}


def _rule(row: Dict[str, Any]) -> str:
    return str(_meta(row).get("chosen_rule") or "")


def _order_step(row: Dict[str, Any]) -> str:
    resolution = _meta(row).get("resolution") or {}
    return str(resolution.get("order_step") or "")


# --------------------------------------------------------------------------- #
# 1 + 2. The two measured defects — both reach resolved_component_species
# --------------------------------------------------------------------------- #
def test_entf_wrapper_does_not_inherit_superset_complex_3623(tmp_path: Path) -> None:
    """§ 7.1 and the first half of the G9 proof.

    ``EntF complex`` is a generated wrapper for ONE protein. Complex 3623 is the
    only complex listing ``EntF`` (6312) as a component, so
    ``resolved_component_species`` takes ``len(component_candidates) == 1`` as
    licence and returns ``matched``. On ``0128fa6`` this row ships
    ``pathbank_complex_id`` and ``pathbank_protein_complex_id`` 3623 and the four
    components ``EntB, EntD, EntF, EntE`` — F-116's own sentence, one stage later.

    Catches: any fix that lets a wrapper keep a superset id, and any fix that
    drops one of the two ids while leaving the other.
    """

    result = run_offline(
        _payload([_wrapper_row("EntF complex", [_component(6312, "EntF", "P11454")])]),
        tmp_path / "cache.json",
    )
    row = _row(result, "EntF complex")

    # The case is not vacuous: the shipped rule really did fire and really did
    # produce 3623 as its single candidate.
    refused = _meta(row).get("refused_superset_complex") or {}
    assert refused.get("pathbank_protein_complex_id") == 3623, (
        "the row must record WHICH complex it refused, and it must be the one the "
        f"component rule offered: {refused}"
    )
    assert sorted(refused.get("uncovered_components") or []) == ["EntB", "EntD", "EntE"], (
        f"the refusal must name the catalysts that would have been injected: {refused}"
    )

    assert row.get("pathbank_complex_id") is None, row.get("pathbank_complex_id")
    assert row.get("pathbank_protein_complex_id") is None, row.get("pathbank_protein_complex_id")
    assert _meta(row).get("pathbank_complex_id") is None, _meta(row)
    assert _meta(row).get("pathbank_protein_complex_id") is None, _meta(row)
    assert _names(row) == ["EntF"], f"the wrapper must still stand for one protein: {_names(row)}"
    assert row.get("generated") is True and row.get("generation_reason") == (
        "single_protein_pathwhiz_wrapper"
    ), "the safe technical-wrapper representation is retained, not a new row shape"


def test_entd_wrapper_does_not_inherit_superset_complex_3623(tmp_path: Path) -> None:
    """§ 7.2. ``EntD`` (6383) — the second measured row, same shape, same single
    candidate 3623, and the same overwrite on ``0128fa6``.

    Catches a fix keyed on one protein id, one row index or one name.
    """

    result = run_offline(
        _payload([_wrapper_row("EntD complex", [_component(6383, "EntD", "P19925")])]),
        tmp_path / "cache.json",
    )
    row = _row(result, "EntD complex")

    refused = _meta(row).get("refused_superset_complex") or {}
    assert refused.get("pathbank_protein_complex_id") == 3623, refused
    assert sorted(refused.get("uncovered_components") or []) == ["EntB", "EntE", "EntF"], refused
    assert row.get("pathbank_complex_id") is None
    assert row.get("pathbank_protein_complex_id") is None
    assert _names(row) == ["EntD"], _names(row)


def test_both_measured_wrappers_stay_distinguishable_by_actor(tmp_path: Path) -> None:
    """F-116's biological consequence, restated for this seam: two chemically
    distinct steps must not become indistinguishable by actor.

    On ``0128fa6`` ``EntF complex`` and ``EntD complex`` both carry 3623 and the
    identical four-component list, so nothing in the payload tells the two steps
    apart. Catches a fix that refuses the id but still copies the components.
    """

    result = run_offline(
        _payload(
            [
                _wrapper_row("EntF complex", [_component(6312, "EntF", "P11454")]),
                _wrapper_row("EntD complex", [_component(6383, "EntD", "P19925")]),
            ]
        ),
        tmp_path / "cache.json",
    )
    entf = _row(result, "EntF complex")
    entd = _row(result, "EntD complex")

    assert _names(entf) != _names(entd), (
        f"the two wrappers collapsed onto the same membership: {_names(entf)}"
    )
    assert _names(entf) == ["EntF"] and _names(entd) == ["EntD"]
    assert {entf.get("pathbank_protein_complex_id"), entd.get("pathbank_protein_complex_id")} == {
        None
    }


# --------------------------------------------------------------------------- #
# 3. EntE — a PRESERVATION case, explicitly NOT the non-vacuity proof (§ 4)
# --------------------------------------------------------------------------- #
def test_ente_wrapper_still_abstains_on_the_ambiguous_name_lookup(tmp_path: Path) -> None:
    """§ 7.3, and § 4's warning made into an assertion.

    ``EntE complex`` never reaches ``resolved_component_species``: its name
    lookup matches ten complexes on ``%EntE%`` and returns ``ambiguous`` at
    ``complex_name_species``, which is an early return. This test asserts that
    abstention is UNCHANGED — and asserts the ``order_step`` explicitly, so the
    record shows this row could not have exercised the guard and is not offered
    as evidence that anything refused a superset.

    Catches a fix that makes the ambiguous name path silently resolve.
    """

    result = run_offline(
        _payload([_wrapper_row("EntE complex", [_component(6301, "EntE", "P10378")])]),
        tmp_path / "cache.json",
    )
    row = _row(result, "EntE complex")

    assert _order_step(row) == "complex_name_species", (
        "§ 4: this row must stop at the NAME lookup. If it now reaches "
        f"resolved_component_species the trap has been re-opened: {_meta(row)}"
    )
    assert (_meta(row).get("resolution") or {}).get("status") == "ambiguous"
    assert _rule(row) == ""
    assert row.get("pathbank_protein_complex_id") is None
    assert _names(row) == ["EntE"]
    assert not _meta(row).get("refused_superset_complex"), (
        "this row was never refused by this card — it abstained on its own, and "
        "recording a refusal here would misattribute the outcome"
    )


# --------------------------------------------------------------------------- #
# 4. The control set, measured BY VALUE
# --------------------------------------------------------------------------- #
def test_entc_entb_enta_still_resolve_to_1143_1189_1190_by_value(tmp_path: Path) -> None:
    """§ 7.4 and § 3. The preservation obligation, discharged as measurement.

    These three resolve through ``pathbank_protein_complex_id``, and their
    PathBank complexes genuinely have one component each. Asserted by value —
    the id, the rule, the component count and the component's name and accession
    — not by an argument from which rule was edited.

    Catches an over-broad fix that refuses on component count, on "the row is a
    complex", or on any rule-agnostic blanket.
    """

    rows = [
        _declared_row(
            "Isochorismate synthase",
            [_component(6238, "EntC", "P0AEJ2")],
            pathbank_protein_complex_id=1143,
        ),
        _declared_row(
            "isochorismatase",
            [_component(6224, "EntB", "P0ADI4")],
            pathbank_protein_complex_id=1189,
        ),
        _declared_row(
            "oxidoreductase (entA)",
            [_component(6341, "EntA", "P15047")],
            pathbank_protein_complex_id=1190,
        ),
    ]
    result = run_offline(_payload(rows), tmp_path / "cache.json")

    expected = {
        "Isochorismate synthase": (1143, "EntC", "P0AEJ2"),
        "isochorismatase": (1189, "EntB", "P0ADI4"),
        "oxidoreductase (entA)": (1190, "EntA", "P15047"),
    }
    for name, (complex_id, component_name, accession) in expected.items():
        row = _row(result, name)
        assert row.get("pathbank_complex_id") == complex_id, (name, row.get("pathbank_complex_id"))
        assert row.get("pathbank_protein_complex_id") == complex_id, (
            name,
            row.get("pathbank_protein_complex_id"),
        )
        assert _rule(row) == "pathbank_protein_complex_id", (name, _rule(row))
        assert _names(row) == [component_name], (name, _names(row))
        components = row.get("components") or []
        assert len(components) == 1, (name, components)
        assert (components[0].get("mapped_ids") or {}).get("uniprot") == accession, components[0]
        assert not _meta(row).get("refused_superset_complex"), (name, _meta(row))


def test_a_generated_wrapper_over_a_genuine_one_component_complex_still_maps(
    tmp_path: Path,
) -> None:
    """§ 3, tightened. The control set above is not ``generated``, so on its own
    it cannot show the guard spares a *wrapper* whose membership genuinely
    matches. This row is a generated wrapper that reaches
    ``resolved_component_species`` — ``EntC`` (6238) is a component of exactly one
    complex, 1143, which has exactly one component — and it must still map.

    Catches "refuse every generated wrapper that reaches the component rule",
    which would pass every other test in this file.
    """

    result = run_offline(
        _payload([_wrapper_row("EntC complex", [_component(6238, "EntC", "P0AEJ2")])]),
        tmp_path / "cache.json",
    )
    row = _row(result, "EntC complex")

    assert _rule(row) == "resolved_component_species", (
        f"the fixture must reach the rule under test, not abstain earlier: {_meta(row)}"
    )
    assert row.get("pathbank_complex_id") == 1143, row.get("pathbank_complex_id")
    assert row.get("pathbank_protein_complex_id") == 1143
    assert _names(row) == ["EntC"], _names(row)


# --------------------------------------------------------------------------- #
# 5 + 6. The over-broad-fix detectors
# --------------------------------------------------------------------------- #
def test_a_declared_row_naming_the_whole_assembly_still_maps_to_3623(tmp_path: Path) -> None:
    """§ 7.5. A complex row whose membership genuinely matches the multi-component
    database complex keeps it, with all four components.

    Catches a fix that refuses supersets by component count, or that treats every
    multi-component candidate as an injection.
    """

    result = run_offline(
        _payload(
            [
                _declared_row(
                    "enterobactin synthase",
                    [
                        _component(6224, "EntB", "P0ADI4"),
                        _component(6383, "EntD", "P19925"),
                        _component(6312, "EntF", "P11454"),
                        _component(6301, "EntE", "P10378"),
                    ],
                )
            ]
        ),
        tmp_path / "cache.json",
    )
    row = _row(result, "enterobactin synthase")

    assert row.get("pathbank_protein_complex_id") == 3623, row
    assert sorted(_names(row)) == sorted(SUPERSET_COMPONENTS), _names(row)


def test_a_declared_partial_row_is_still_enriched_to_the_full_complex(tmp_path: Path) -> None:
    """§ 5, the sentence a narrow fix has to honour: *a DECLARED complex row that
    lists partial components may still legitimately be enriched to the full
    database complex — that is correct grounding, not this defect.*

    Byte-for-byte the ``EntF``-shaped case, minus ``generated``/
    ``generation_reason``. It reaches ``resolved_component_species``, is a strict
    superset, and must STILL map to 3623 with four components, because a declared
    row is a biological claim about an assembly and a wrapper is not.

    Catches a fix that keys on component counts, on the component rule, or on
    "superset" alone rather than on the wrapper marker — the exact global
    disabling § 5 forbids.
    """

    result = run_offline(
        _payload([_declared_row("EntF partner complex", [_component(6312, "EntF", "P11454")])]),
        tmp_path / "cache.json",
    )
    row = _row(result, "EntF partner complex")

    assert _rule(row) == "resolved_component_species", _meta(row)
    assert row.get("pathbank_protein_complex_id") == 3623, row.get("pathbank_protein_complex_id")
    assert sorted(_names(row)) == sorted(SUPERSET_COMPONENTS), _names(row)
    assert not _meta(row).get("refused_superset_complex"), _meta(row)


# --------------------------------------------------------------------------- #
# 7. Wrapper generation with and without a database candidate
# --------------------------------------------------------------------------- #
def _reaction(name: str, enzyme: str) -> Dict[str, Any]:
    return {
        "name": name,
        "inputs": ["chorismate"],
        "outputs": ["enterobactin"],
        "enzymes": [{"entity": enzyme, "entity_type": "protein", "role": "catalyst"}],
    }


def test_wrapper_generation_survives_with_and_without_a_database_candidate(
    tmp_path: Path,
) -> None:
    """§ 7.7. Both wrapper-generation paths still produce a valid wrapper row.

    ``EntF`` HAS a database candidate (3623, via the component lookup); ``YbdZ``
    is absent from the fixture database entirely and has none. Run end to end
    with ``allow_complex_wrapper_creation=True``, so C-086's rewrite creates the
    wrappers and this card's loop then re-resolves them — the exact production
    ordering that produced F-133.

    Catches a fix that makes the wrapper-generation rationale unreachable, or that
    leaves a refused wrapper without the ``generated`` marker the PathWhiz
    importer path depends on.
    """

    payload = _payload(
        [],
        reactions=[
            _reaction("EntF-catalyzed enterobactin synthesis", "EntF"),
            _reaction("uncharacterised accessory step", "YbdZ"),
        ],
    )
    payload["entities"]["proteins"].append(
        {"name": "YbdZ", "species": "Escherichia coli", "organism": "Escherichia coli"}
    )
    result = run_offline(payload, tmp_path / "cache.json", allow_wrappers=True)

    complexes = {
        str(row.get("name") or ""): row
        for row in result["payload"]["entities"]["protein_complexes"]
        if isinstance(row, dict)
    }
    assert complexes, "wrapper creation produced no protein_complex rows at all"

    for name, row in complexes.items():
        assert map_ids.is_generated_complex_wrapper(row), (name, row.get("mapping_meta"))
        assert row.get("pathbank_protein_complex_id") != 3623, (
            f"{name} inherited the superset complex through the generation path: {row}"
        )
        assert len(row.get("components") or []) == 1, (name, _names(row))

    # Both reactions still carry an actor, and they are different actors.
    reactions = result["payload"]["processes"]["reactions"]
    actors = [
        str((reaction.get("enzymes") or [{}])[0].get("protein_complex") or "")
        or str((reaction.get("enzymes") or [{}])[0].get("entity") or "")
        for reaction in reactions
    ]
    assert all(actors), f"a reaction lost its enzyme: {actors}"
    assert len(set(actors)) == len(actors), f"two steps collapsed onto one actor: {actors}"


# --------------------------------------------------------------------------- #
# 8. The GUARD-THE-ROW proof: the third assignment branch
# --------------------------------------------------------------------------- #
#: The two edits that turn the row-level decision into the plausible-but-wrong
#: "guard the branch you first notice" fix. Both must match exactly once, which
#: is itself the assertion that the guard is still where this file thinks it is.
_MUTANT_DISABLE_ROW_SWAP = (
    "        if wrapper_identity_refusal:\n"
    "            result = _wrapper_identity_refused_result(\n"
)
_MUTANT_DISABLE_ROW_SWAP_TO = (
    "        if False:  # C-095 test mutant: row-level swap disabled\n"
    "            result = _wrapper_identity_refused_result(\n"
)
_MUTANT_GUARD_BRANCH_ONE = (
    '        if result.get("status") == "mapped":\n'
    "            # A confident DB match carries its own authoritative"
)
_MUTANT_GUARD_BRANCH_ONE_TO = (
    '        if result.get("status") == "mapped" and not wrapper_identity_refusal:\n'
    "            # A confident DB match carries its own authoritative"
)
#: The naive fix guards the id writes too — otherwise it would not even fix the
#: measured EntF row, and nobody would ship it. Guarding all three is what makes
#: it look complete.
_MUTANT_GUARD_ID_WRITES = (
    '        if result.get("status") == "mapped":\n'
    "            protein_complexes_mapped += 1"
)
_MUTANT_GUARD_ID_WRITES_TO = (
    '        if result.get("status") == "mapped" and not wrapper_identity_refusal:\n'
    "            protein_complexes_mapped += 1"
)


def _branch_only_mutant(tmp_path: Path) -> ModuleType:
    """A copy of ``map_ids`` whose guard sits on the ``status == "mapped"`` tests.

    This is the fix a reader writes after noticing ``complex_row["components"] =
    result["components"]`` at the top of the overwrite and the id writes below it,
    and nothing else. It is loaded, not described, so the claim "that fix is
    incomplete" is measured rather than argued.
    """

    source = (SRC / "t2pw" / "mapping" / "map_ids.py").read_text(encoding="utf-8")
    assert source.count(_MUTANT_DISABLE_ROW_SWAP) == 1, "the row-level guard moved"
    assert source.count(_MUTANT_GUARD_BRANCH_ONE) == 1, "the first assignment branch moved"
    assert source.count(_MUTANT_GUARD_ID_WRITES) == 1, "the id-write test moved"
    mutated = (
        source.replace(_MUTANT_DISABLE_ROW_SWAP, _MUTANT_DISABLE_ROW_SWAP_TO)
        .replace(_MUTANT_GUARD_BRANCH_ONE, _MUTANT_GUARD_BRANCH_ONE_TO)
        .replace(_MUTANT_GUARD_ID_WRITES, _MUTANT_GUARD_ID_WRITES_TO)
    )
    path = tmp_path / "map_ids_branch_only_mutant.py"
    path.write_text(mutated, encoding="utf-8")

    name = "t2pw_map_ids_c095_branch_only_mutant"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover - a load failure is a test failure, not a skip
        sys.modules.pop(name, None)
        raise
    return module


def _componentless_wrapper_payload() -> Dict[str, Any]:
    """A generated wrapper that reaches the overwrite with NO components.

    ``components: []`` is what routes the overwrite past branch 1 (once branch 1
    is guarded) and past branch 2 (``_safe_list(complex_row["components"])`` is
    falsy) into branch 3. The name matches PathBank complex 3623 exactly, so
    ``complex_name_species`` returns ``mapped`` carrying 3623's four hydrated
    components — a result with components and, once branch 1 is guarded, nowhere
    to go but branch 3.
    """

    return _payload([_wrapper_row("enterobactin synthase", [])])


def test_the_branch_only_fix_still_injects_the_superset_through_the_third_branch(
    tmp_path: Path,
) -> None:
    """§ 2's trap, measured rather than asserted.

    The mutant guards ``if result.get("status") == "mapped"`` and nothing else.
    It still refuses ``EntF`` — so it LOOKS complete — and then writes the same
    four superset components onto the component-less wrapper through
    ``elif _safe_list(result.get("components")):``, the third branch.

    This test FAILS if the mutant stops injecting, which is the only way to know
    the branch-3 fixture is not vacuous.
    """

    mutant = _branch_only_mutant(tmp_path)

    # It looks complete: the branch-1 case is refused by the mutant too.
    entf = _row(
        run_offline(
            _payload([_wrapper_row("EntF complex", [_component(6312, "EntF", "P11454")])]),
            tmp_path / "mutant-entf.json",
            module=mutant,
        ),
        "EntF complex",
    )
    assert entf.get("pathbank_protein_complex_id") is None, (
        "the branch-only mutant is supposed to fix the branch-1 case; if it does "
        "not, this test is comparing against the wrong thing"
    )

    # And it is not: branch 3 writes the superset anyway.
    row = _row(
        run_offline(
            _componentless_wrapper_payload(), tmp_path / "mutant-empty.json", module=mutant
        ),
        "enterobactin synthase",
    )
    assert sorted(_names(row)) == sorted(SUPERSET_COMPONENTS), (
        "the branch-3 fixture is vacuous: the branch-only mutant did not inject "
        f"the superset either, so it proves nothing. Got {_names(row)}"
    )


def test_the_row_level_guard_covers_the_third_assignment_branch(tmp_path: Path) -> None:
    """§ 2. The same payload the mutant fails on, through the shipped code.

    The decision is taken once, for the ROW, before the overwrite — so the object
    all three assignment branches and both id writes read is already the refusal.
    No branch can attach the superset, whichever one runs.

    Catches exactly the incomplete fix the test above demonstrates.
    """

    row = _row(
        run_offline(_componentless_wrapper_payload(), tmp_path / "cache.json"),
        "enterobactin synthase",
    )

    assert _names(row) == [], f"superset components reached a component-less wrapper: {_names(row)}"
    assert row.get("pathbank_complex_id") is None, row.get("pathbank_complex_id")
    assert row.get("pathbank_protein_complex_id") is None
    assert _meta(row).get("pathbank_complex_id") is None
    assert _meta(row).get("pathbank_protein_complex_id") is None
    refused = _meta(row).get("refused_superset_complex") or {}
    assert refused.get("pathbank_protein_complex_id") == 3623, refused


# --------------------------------------------------------------------------- #
# 9 + 10. REV-C095 F1/F2 — a result that confers no identity is not refused
# --------------------------------------------------------------------------- #
def _ente_under_a_synonym() -> Dict[str, Any]:
    """The payload row for EntE, canonically named and knowing ``EntE`` as an alias.

    Measured shape, not invented: PathBank protein 6301 is
    ``2,3-dihydroxybenzoate-AMP ligase`` and ``EntE`` is its gene-symbol synonym.
    ``_reconcile_components_against_local_proteins`` matches the alias and renames
    the component to this canonical name.
    """

    return {
        "name": "2,3-dihydroxybenzoate-AMP ligase",
        "synonyms": "EntE",
        "species": "Escherichia coli",
        "organism": "Escherichia coli",
        "pathbank_protein_id": 6301,
        "mapped_ids": {"uniprot": "P10378", "pathbank_protein_id": "6301"},
    }


def run_offline_without_db(payload: Dict[str, Any], cache_path: Path) -> Dict[str, Any]:
    """``map_payload`` with NO resolver at all — ``from_env`` returns ``None``.

    This is the state of every worktree in this sprint that has no ``.env``, so
    it is also the state most of the suite runs in. ``_map_complex_with_strategy``
    then returns ``db_unavailable`` carrying the ROW'S OWN components.
    """

    env = {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0", "T2PW_OFFLINE_CURATOR": "1"}
    with patch.dict(os.environ, env), patch.object(
        map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)
    ), patch.object(
        map_ids, "_ai_protein_synonym_lookup", return_value=[]
    ), patch.object(
        map_ids.HttpClient, "get", side_effect=_NoNetwork("network call during an offline run")
    ):
        return map_ids.map_payload(
            payload,
            cache_path=cache_path,
            id_source="db",
            use_cache=False,
            allow_complex_wrapper_creation=False,
        )


def test_db_unavailable_does_not_refuse_a_wrapper_against_itself(tmp_path: Path) -> None:
    """REV-C095 F1, proof A. With no resolver, ``db_unavailable`` echoes the row's
    OWN components back as the result's. Nothing is offered and nothing may be
    refused: the row must come out of the loop exactly as ``0128fa6`` leaves it.

    The payload protein is named by a SYNONYM of the declared component, so
    reconciliation renames the echoed component and the unfixed scope builder
    could no longer recognise the wrapper's own protein — which is how the
    refusal record came to name the wrapper's own catalyst as an injected
    stranger.

    Catches: evaluating the guard on a result that confers no identity, and a
    membership scope that does not survive reconciliation's renaming.
    """

    payload = _payload([_wrapper_row("EntE complex", [_bare_component("EntE")])])
    payload["entities"]["proteins"] = [_ente_under_a_synonym()]
    row = _row(run_offline_without_db(payload, tmp_path / "cache.json"), "EntE complex")

    assert not _meta(row).get("refused_superset_complex"), (
        "a result that confers no identity was refused, and the record names the "
        f"wrapper's own protein as an injected catalyst: {_meta(row).get('refused_superset_complex')}"
    )
    assert _rule(row) == "", _rule(row)
    assert _meta(row).get("resolution") == {
        "status": "unresolved",
        "issue": "db_unavailable",
    }, f"the db_unavailable resolution must be byte-identical to base: {_meta(row).get('resolution')}"
    assert row.get("pathbank_protein_complex_id") is None


def test_an_ambiguous_lookup_is_never_recorded_as_a_refused_complex(tmp_path: Path) -> None:
    """REV-C095 F1, proof B — on this file's own § 4 preservation case.

    ``EntE complex`` abstains at ``complex_name_species`` with ten candidates the
    resolver explicitly declined to choose between. Writing a refusal record for
    it named ``candidates[0]`` — ``ferric enterobactin outer membrane transport
    complex`` — as a complex the row had refused, asserting a match that never
    happened. An abstention is not a refusal and must leave no trace.

    Catches: the ``candidates[0]`` name fallback being reached from a
    non-``matched`` result, and any future guard that fires before an identity is
    on offer.
    """

    payload = _payload([_wrapper_row("EntE complex", [_bare_component("EntE")])])
    payload["entities"]["proteins"] = [
        dict(protein) for protein in ALL_PROTEINS if protein["name"] != "EntE"
    ] + [_ente_under_a_synonym()]
    row = _row(run_offline(payload, tmp_path / "cache.json"), "EntE complex")

    assert not _meta(row).get("refused_superset_complex"), (
        "an ambiguous abstention was recorded as a refused complex match: "
        f"{_meta(row).get('refused_superset_complex')}"
    )
    assert _order_step(row) == "complex_name_species", _meta(row)
    assert (_meta(row).get("resolution") or {}).get("status") == "ambiguous"
    assert _rule(row) == ""
    assert row.get("pathbank_protein_complex_id") is None
    assert len(row.get("components") or []) == 1


def test_a_source_named_complex_wrapper_is_not_refused(tmp_path: Path) -> None:
    """REV-C095 F2. ``process_normalizer.py:4999-5005`` emits a second wrapper
    kind, ``generation_reason: complex_named_source_entity_wrapper``, created
    BECAUSE THE SOURCE TEXT NAMED A COMPLEX -- a biological claim about an
    assembly, which is the declared row section 5 says must still be enriched.

    It is emitted with ``components: []``, the same shape that routes to the third
    assignment branch, so a guard keyed on ``generated`` alone would leave a
    source-named complex with neither components nor an id. Here the row is named
    exactly as PathBank complex 3623, matches by ``complex_name_species``, and
    must come out fully enriched.

    Catches: keying the guard on ``is_generated_complex_wrapper`` alone rather
    than on the charter's ``single_protein_pathwhiz_wrapper``.
    """

    row = _row(
        run_offline(
            _payload(
                [
                    {
                        "name": "enterobactin synthase",
                        "class": "protein_complex",
                        "generated": True,
                        "generation_reason": "complex_named_source_entity_wrapper",
                        "confidence": 0.8,
                        "provenance": "inferred",
                        "species": "Escherichia coli",
                        "components": [],
                    }
                ]
            ),
            tmp_path / "cache.json",
        ),
        "enterobactin synthase",
    )

    assert not _meta(row).get("refused_superset_complex"), (
        "a source-named complex wrapper was refused: "
        f"{_meta(row).get('refused_superset_complex')}"
    )
    assert row.get("pathbank_protein_complex_id") == 3623, row.get("pathbank_protein_complex_id")
    assert sorted(_names(row)) == sorted(SUPERSET_COMPONENTS), _names(row)

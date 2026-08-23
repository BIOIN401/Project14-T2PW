"""C-076 / F-102 -- a shared accession within one kind is identity, not conflict.

Why this file exists
--------------------
The product owner's ruling of 2026-08-23:

    The same UniProt accession may be shared by proven aliases of the same
    protein and by holo/apo states of the same underlying polypeptide. ``EntE``
    and ``enterobactin synthase`` are the same protein identity. Holo-EntB and
    apo-EntB may share the underlying UniProt accession while remaining distinct
    pathway states. Do not flag these as accession conflicts unless the entities
    are biologically unrelated or cross-kind.

``bench/semantic.py::_check_id_conflicts`` embodied the opposite rule: any
accession answering to two differently-*named* rows was a collision. That is the
exact predicate C-073's review rejected in the pipeline, for contradicting
**D-035 clause 3c** -- a matching stable external identifier is *proof* that two
differently-named rows are the same entity. The pipeline was corrected to a
cross-kind rule; the scorer was not, and ``no_real_id_or_name_conflict`` gates
(``release_status.SEMANTIC_GATING_CHECKS``), so the scorer went on demoting real
legs with a rule the sprint had already thrown out.

What is pinned here
-------------------
1. within-kind agreement no longer emits a finding -- on the two REAL pairs
   measured in ``runs_verify/2026-08-22_2147`` and on their synthetic minimals;
2. **cross-kind still fires** -- without this the card is a blanket disabling;
3. ``R196A`` -- a site-directed point mutant, i.e. a different polypeptide -- is
   still a false real identifier when it claims EntB's accession;
4. a modification state that invents an accession *nothing else claims* is still
   a false real identifier, so Arm B did not over-reach in the other direction;
5. the ``false_real`` coupling is unchanged: the collision branch appends a
   finding and never increments the counter;
6. the gold set still loads, and only the one PMC12096016 entry moved.

Offline and deterministic: no network, no LLM, no database.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench.goldset import (  # noqa: E402
    EXPORT_STRICT,
    ForbiddenIdentifier,
    GoldCase,
    GoldTerm,
    RELEVANCE_CORE,
    load_gold_set,
)
from t2pw.bench.semantic import (  # noqa: E402
    CHECK_ID_CONFLICT,
    ERR_FALSE_REAL_IDENTIFIERS,
    _check_id_conflicts,
    validate_semantic_coverage,
)


COLLISION = "accession_claimed_by_multiple_entities"
FALSE_REAL = "false_real_identifier"
UNMAPPED = "forbidden_identity_present_unmapped"

#: The two real pairs measured at T-105. Both claimants of both pairs sit in
#: ``entities.proteins`` -- verified against the committed artifacts below.
ENTB_ACCESSION = "P0ADI4"
ENTE_ACCESSION = "P10378"

#: The T-105 acceptance run whose findings this card corrects.
REPLAY_RUN = ROOT / "runs_verify" / "2026-08-22_2147"


# ---------------------------------------------------------------------------
# Helpers -- deliberately local, so this file does not inherit a fixture that
# another card may retune.
# ---------------------------------------------------------------------------
def _case(**overrides) -> GoldCase:
    base = dict(
        paper_id="PMCTEST",
        title="a test paper",
        requested_pathway="enterobactin biosynthesis",
        requested_organism="Escherichia coli",
        mechanistic_relevance=RELEVANCE_CORE,
        expected_pathway_anchors=(GoldTerm(name="enterobactin biosynthesis"),),
        min_connected_reactions=1,
        expected_export=EXPORT_STRICT,
    )
    base.update(overrides)
    return GoldCase(**base)


def _entities(*, compounds=(), proteins=(), protein_complexes=()) -> dict:
    return {
        "compounds": [dict(row) for row in compounds],
        "proteins": [dict(row) for row in proteins],
        "protein_complexes": [dict(row) for row in protein_complexes],
    }


def _kinds(check) -> list:
    return [finding.get("kind") for finding in check.findings]


#: The gold entry as C-076 leaves it: four modification states of one
#: polypeptide, and the point mutant split out on its own.
_STATES = ForbiddenIdentifier(
    name="apo-EntB",
    kind="modification_state",
    reason="Cofactor-loading states of ONE polypeptide.",
    aliases=("holo-EntB", "apo-Fur", "holo-Fur"),
)
_MUTANT = ForbiddenIdentifier(
    name="R196A",
    kind="strain_or_construct",
    reason="A site-directed point mutant: a different polypeptide sequence.",
)


# ---------------------------------------------------------------------------
# 1. Within one kind, a shared accession is agreement (D-035 clause 3c).
# ---------------------------------------------------------------------------
def test_entb_and_holo_entb_sharing_one_accession_is_not_a_conflict():
    # The exact PMC12096016/research shape: parent row, its holo state, one
    # accession, both in entities.proteins.
    case = _case(forbidden_identifiers=(_STATES, _MUTANT))
    check, false_real = _check_id_conflicts(
        case,
        _entities(proteins=[
            {"name": "EntB", "uniprot": ENTB_ACCESSION},
            {"name": "holo-EntB", "uniprot": ENTB_ACCESSION},
        ]),
    )
    assert COLLISION not in _kinds(check)
    assert FALSE_REAL not in _kinds(check)
    assert false_real == 0
    assert check.ok
    assert check.name == CHECK_ID_CONFLICT


def test_ente_and_enterobactin_synthase_sharing_one_accession_is_not_a_conflict():
    # The exact PMC12452463/strict shape. Neither name is forbidden at all --
    # this pair is pure alias agreement, with no gold entry in play.
    case = _case()
    check, false_real = _check_id_conflicts(
        case,
        _entities(proteins=[
            {"name": "EntE", "uniprot": ENTE_ACCESSION},
            {"name": "enterobactin synthase", "uniprot": ENTE_ACCESSION},
        ]),
    )
    assert check.findings == []
    assert check.ok
    assert false_real == 0


def test_two_compounds_sharing_one_accession_is_also_agreement():
    # The rule is about KIND, not about proteins specifically. Two spellings of
    # one metabolite resolving to one ChEBI id is the same clause-3c evidence.
    check, false_real = _check_id_conflicts(
        _case(),
        _entities(compounds=[
            {"name": "2,3-dihydroxybenzoate", "chebi": "16793"},
            {"name": "2,3-DHB", "chebi": "16793"},
        ]),
    )
    assert check.findings == []
    assert false_real == 0


def test_a_protein_and_a_protein_complex_are_the_same_kind():
    # entity_kind_class puts ``proteins`` and ``protein_complexes`` in one class,
    # so a complex named for its catalytic subunit is not a cross-kind claim.
    check, _ = _check_id_conflicts(
        _case(),
        _entities(
            proteins=[{"name": "EntE", "uniprot": ENTE_ACCESSION}],
            protein_complexes=[{"name": "enterobactin synthase", "uniprot": ENTE_ACCESSION}],
        ),
    )
    assert check.findings == []


# ---------------------------------------------------------------------------
# 2. Cross-kind still fires. Without this the card is a blanket disabling.
# ---------------------------------------------------------------------------
def test_cross_kind_accession_collision_is_still_a_conflict():
    # The one true defect C-073 fixed in the pipeline: one DrugBank id cannot
    # denote a protein and a metabolite at once. The scorer must agree with it.
    check, false_real = _check_id_conflicts(
        _case(),
        _entities(
            proteins=[{"name": "ALAS2", "drugbank": "DB00114"}],
            compounds=[{"name": "Pyridoxal 5'-phosphate", "drugbank": "DB00114"}],
        ),
    )
    assert not check.ok
    collisions = [f for f in check.findings if f.get("kind") == COLLISION]
    assert len(collisions) == 1
    assert collisions[0]["namespace"] == "drugbank"
    assert collisions[0]["identifier"] == "db00114"
    assert collisions[0]["entities"] == ["ALAS2", "Pyridoxal 5'-phosphate"]
    # 5. The false_real coupling is NOT changed by this card: the collision
    # branch appends a finding and increments nothing.
    assert false_real == 0


def test_cross_kind_needs_a_name_difference_too():
    # One entity routed into both ``compounds`` and ``proteins`` is one entity
    # written twice, not a protein masquerading as a metabolite. This mirrors
    # identity_admission.find_kind_conflicting_accessions, whose rule the scorer
    # is now the other half of.
    check, _ = _check_id_conflicts(
        _case(),
        _entities(
            proteins=[{"name": "EntB", "uniprot": ENTB_ACCESSION}],
            compounds=[{"name": "EntB", "uniprot": ENTB_ACCESSION}],
        ),
    )
    assert check.findings == []


def test_a_cross_kind_claim_still_fires_when_a_within_kind_pair_shares_the_same_id():
    # Mixed case: the same accession claimed by two proteins AND one compound.
    # The within-kind half must not launder the cross-kind half.
    check, _ = _check_id_conflicts(
        _case(),
        _entities(
            proteins=[
                {"name": "EntB", "uniprot": ENTB_ACCESSION},
                {"name": "holo-EntB", "uniprot": ENTB_ACCESSION},
            ],
            compounds=[{"name": "enterobactin", "uniprot": ENTB_ACCESSION}],
        ),
    )
    collisions = [f for f in check.findings if f.get("kind") == COLLISION]
    assert len(collisions) == 1
    assert collisions[0]["entities"] == ["EntB", "enterobactin", "holo-EntB"]


# ---------------------------------------------------------------------------
# 3. R196A stays forbidden -- the arm that proves Arm B did not over-reach.
# ---------------------------------------------------------------------------
def test_r196a_claiming_the_parent_accession_is_still_a_false_real_identifier():
    case = _case(forbidden_identifiers=(_STATES, _MUTANT))
    check, false_real = _check_id_conflicts(
        case,
        _entities(proteins=[
            {"name": "EntB", "uniprot": ENTB_ACCESSION},
            {"name": "R196A", "uniprot": ENTB_ACCESSION},
        ]),
    )
    assert not check.ok
    forged = [f for f in check.findings if f.get("kind") == FALSE_REAL]
    assert [f["name"] for f in forged] == ["R196A"]
    assert forged[0]["forbidden_kind"] == "strain_or_construct"
    assert false_real == 1


def test_holo_entb_is_exempt_where_r196a_is_not_in_the_same_payload():
    # Both rows claim the parent accession; only the mutant is condemned. This
    # is the distinction the whole card turns on.
    case = _case(forbidden_identifiers=(_STATES, _MUTANT))
    check, false_real = _check_id_conflicts(
        case,
        _entities(proteins=[
            {"name": "EntB", "uniprot": ENTB_ACCESSION},
            {"name": "holo-EntB", "uniprot": ENTB_ACCESSION},
            {"name": "R196A", "uniprot": ENTB_ACCESSION},
        ]),
    )
    assert [f["name"] for f in check.findings if f.get("kind") == FALSE_REAL] == ["R196A"]
    assert false_real == 1


# ---------------------------------------------------------------------------
# 4. A state that INVENTS an accession is still a fabricated identity.
# ---------------------------------------------------------------------------
def test_a_modification_state_with_its_own_invented_accession_is_still_forged():
    # Only SHARING the parent's accession became legitimate. A standalone
    # protein row carrying an accession nothing else claims is the case the gold
    # entry still exists to catch.
    case = _case(forbidden_identifiers=(_STATES, _MUTANT))
    check, false_real = _check_id_conflicts(
        case,
        _entities(proteins=[
            {"name": "EntB", "uniprot": ENTB_ACCESSION},
            {"name": "holo-EntB", "uniprot": "Q9XYZ9"},
        ]),
    )
    assert [f["name"] for f in check.findings if f.get("kind") == FALSE_REAL] == ["holo-EntB"]
    assert false_real == 1


def test_two_modification_states_cannot_vouch_for_each_other():
    # Both states carry an invented accession and no real parent row claims it.
    # A rule that only asked "is someone else claiming this?" would clear both.
    case = _case(forbidden_identifiers=(_STATES, _MUTANT))
    check, false_real = _check_id_conflicts(
        case,
        _entities(proteins=[
            {"name": "holo-Fur", "uniprot": "Q9XYZ9"},
            {"name": "apo-Fur", "uniprot": "Q9XYZ9"},
        ]),
    )
    assert sorted(f["name"] for f in check.findings if f.get("kind") == FALSE_REAL) == [
        "apo-Fur",
        "holo-Fur",
    ]
    assert false_real == 2


def test_a_state_sharing_only_one_of_two_accessions_is_still_forged():
    # Every accession the row carries must be accounted for. Half-sharing is a
    # fabricated identity on the unaccounted half.
    case = _case(forbidden_identifiers=(_STATES, _MUTANT))
    check, false_real = _check_id_conflicts(
        case,
        _entities(proteins=[
            {"name": "EntB", "uniprot": ENTB_ACCESSION},
            {"name": "holo-EntB", "uniprot": ENTB_ACCESSION, "kegg": "K02363"},
        ]),
    )
    assert [f["name"] for f in check.findings if f.get("kind") == FALSE_REAL] == ["holo-EntB"]
    assert false_real == 1


def test_an_unmapped_modification_state_still_reports_itself():
    # Unchanged by this card, on purpose: a forbidden name carrying NO accession
    # is still recorded, and still does not count toward false_real. This is the
    # PMC12096016 ``apo-EntB`` row exactly.
    case = _case(forbidden_identifiers=(_STATES, _MUTANT))
    check, false_real = _check_id_conflicts(
        case,
        _entities(proteins=[
            {"name": "EntB", "uniprot": ENTB_ACCESSION},
            {"name": "apo-EntB"},
        ]),
    )
    assert [f["kind"] for f in check.findings] == [UNMAPPED]
    assert check.findings[0]["forbidden_kind"] == "modification_state"
    assert false_real == 0


def test_a_state_whose_parent_is_a_compound_is_not_sharing_a_parent():
    # "Parent" is same-kind. A protein state cannot borrow legitimacy from a
    # metabolite that happens to carry the same string.
    case = _case(forbidden_identifiers=(_STATES, _MUTANT))
    check, false_real = _check_id_conflicts(
        case,
        _entities(
            proteins=[{"name": "holo-EntB", "chebi": "16793"}],
            compounds=[{"name": "2,3-dihydroxybenzoate", "chebi": "16793"}],
        ),
    )
    kinds = _kinds(check)
    assert FALSE_REAL in kinds
    assert COLLISION in kinds
    assert false_real == 1


# ---------------------------------------------------------------------------
# 5. false_real coupling, stated once as its own obligation.
# ---------------------------------------------------------------------------
def test_the_collision_branch_never_increments_false_real():
    # Three cross-kind collisions, zero forbidden identifiers, zero
    # placeholders: false_real must still be 0. This coupling predates C-076 and
    # the card must not move it as a side effect.
    check, false_real = _check_id_conflicts(
        _case(),
        _entities(
            proteins=[
                {"name": "ALAS2", "drugbank": "DB00114"},
                {"name": "MenD", "kegg": "C00001"},
                {"name": "EntF", "chebi": "16793"},
            ],
            compounds=[
                {"name": "Pyridoxal 5'-phosphate", "drugbank": "DB00114"},
                {"name": "water", "kegg": "C00001"},
                {"name": "2,3-dihydroxybenzoate", "chebi": "16793"},
            ],
        ),
    )
    assert len([f for f in check.findings if f.get("kind") == COLLISION]) == 3
    assert false_real == 0
    assert "0 false real identifier(s); 3 identity conflict finding(s) total" in check.summary


def test_scientific_error_count_follows_false_real_through_the_public_entry_point():
    # The priority-1 number the acceptance report prints is ERR_FALSE_REAL_
    # IDENTIFIERS, and it is fed by exactly the counter above.
    case = _case(forbidden_identifiers=(_STATES, _MUTANT))
    payload = {
        "entities": _entities(proteins=[
            {"name": "EntB", "uniprot": ENTB_ACCESSION},
            {"name": "holo-EntB", "uniprot": ENTB_ACCESSION},
        ]),
        "processes": {
            "reactions": [{
                "name": "enterobactin biosynthesis",
                "inputs": ["2,3-dihydroxybenzoate"],
                "outputs": ["enterobactin"],
                "evidence": "quoted",
            }],
            "transports": [],
            "interactions": [],
        },
    }
    for name in ("2,3-dihydroxybenzoate", "enterobactin"):
        payload["entities"]["compounds"].append({"name": name})
    report = validate_semantic_coverage(case, payload)
    assert report.checks[CHECK_ID_CONFLICT].ok
    assert report.scientific_errors[ERR_FALSE_REAL_IDENTIFIERS] == 0


# ---------------------------------------------------------------------------
# 6. Gold integrity -- only the one PMC12096016 entry moved.
# ---------------------------------------------------------------------------
def test_the_pinned_gold_still_loads():
    gold = load_gold_set()
    assert len(gold) == 10


def test_only_one_gold_entry_anywhere_carries_the_new_kind():
    gold = load_gold_set()
    tagged = [
        (case.paper_id, entry)
        for case in gold
        for entry in case.forbidden_identifiers
        if entry.kind == "modification_state"
    ]
    assert len(tagged) == 1
    paper_id, entry = tagged[0]
    assert paper_id == "PMC12096016"
    assert entry.name == "apo-EntB"
    assert entry.aliases == ("holo-EntB", "apo-Fur", "holo-Fur")
    assert "R196A" not in entry.aliases


def test_r196a_is_its_own_strain_or_construct_entry():
    case = {c.paper_id: c for c in load_gold_set()}["PMC12096016"]
    entry = case.forbidden_match("R196A")
    assert entry is not None
    assert entry.name == "R196A"
    assert entry.kind == "strain_or_construct"
    assert entry.aliases == ()


@pytest.mark.parametrize("name", ["apo-EntB", "holo-EntB", "apo-Fur", "holo-Fur"])
def test_every_modification_state_still_resolves_to_the_states_entry(name):
    case = {c.paper_id: c for c in load_gold_set()}["PMC12096016"]
    entry = case.forbidden_match(name)
    assert entry is not None
    assert entry.kind == "modification_state"


def test_the_touched_case_kept_every_other_forbidden_entry():
    # Byte-for-byte comparison lives in the branch diff; what a test can pin is
    # that the edit SPLIT one entry and removed nothing. Eleven names before,
    # the same eleven plus ``R196A`` after.
    case = {c.paper_id: c for c in load_gold_set()}["PMC12096016"]
    assert [entry.name for entry in case.forbidden_identifiers] == [
        "MenD",
        "lactate dehydrogenase",
        "DBH",
        "EntCBAE",
        "IC",
        "apo-EntB",
        "R196A",
        "NADH",
        "thiamine pyrophosphate",
        "2,5-DHB",
        "EntH",
        "pCA24N",
    ]


def test_no_other_case_lost_or_gained_a_forbidden_identifier():
    # 100 entries at base SHA 129d9b2, +1 for the split. Pinned as a literal so
    # a later card that deletes gold evidence has to move this number on the
    # record, exactly as merge rule 4 requires.
    gold = load_gold_set()
    total = sum(len(case.forbidden_identifiers) for case in gold)
    assert total == 101
    others = {
        case.paper_id: len(case.forbidden_identifiers)
        for case in gold
        if case.paper_id != "PMC12096016"
    }
    assert sum(others.values()) == 89


# ---------------------------------------------------------------------------
# 7. Real-artifact replay -- the two committed legs the ruling names.
# ---------------------------------------------------------------------------
def _replay(paper_id: str, mode: str):
    payload_path = REPLAY_RUN / "papers" / paper_id / mode / "final_mapped.json"
    if not payload_path.is_file():
        pytest.skip(f"{payload_path} is not committed in this tree")
    case = {c.paper_id: c for c in load_gold_set()}[paper_id]
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    return _check_id_conflicts(case, payload.get("entities") or {})


def test_replay_pmc12096016_research_loses_the_collision_and_the_false_real():
    # At base SHA 129d9b2 this leg scored, in payload order:
    #   NADH (false_real) | apo-EntB (unmapped) | holo-EntB (false_real)
    #   | uniprot:p0adi4 claimed by EntB + holo-EntB (collision)
    # false_real == 2. Both EntB rows sit in entities.proteins.
    check, false_real = _replay("PMC12096016", "research")
    assert COLLISION not in _kinds(check)
    assert [f["name"] for f in check.findings if f.get("kind") == FALSE_REAL] == ["NADH"]
    assert [f["name"] for f in check.findings if f.get("kind") == UNMAPPED] == ["apo-EntB"]
    assert false_real == 1
    # Still gating: the leg keeps two findings that this card does not touch.
    assert not check.ok


def test_replay_pmc12452463_strict_loses_its_collision():
    # At base: ``enterobactin synthase complex`` (unmapped) + uniprot:p10378
    # claimed by EntE + enterobactin synthase (collision). Both in proteins.
    check, false_real = _replay("PMC12452463", "strict")
    assert COLLISION not in _kinds(check)
    assert _kinds(check) == [UNMAPPED]
    assert false_real == 0
    assert not check.ok


def test_replay_pmc12096016_strict_is_untouched_by_this_card():
    # The control: the same paper's other leg has two false real identifiers,
    # NADH and NAD+, neither of which is an alias or a modification state.
    check, false_real = _replay("PMC12096016", "strict")
    assert [f["name"] for f in check.findings if f.get("kind") == FALSE_REAL] == ["NADH", "NAD+"]
    assert false_real == 2

"""C-082 / F-115 -- an ambiguous species rename is DECLINED, never a dead leg.

**G9 labelling, stated up front.** This card *corrects pre-existing observable
behaviour*. On ``runs_verify/2026-08-24_1428`` (T-106), leg
``PMC12444477/research`` ended ``status: error``, ``failure_kind: crash``, at
``post_pipeline`` with::

    Post-pipeline conversion failed: AMBIGUOUS_RENAME_TARGET: renaming
    'Escherichia coli K-12' to 'Escherichia coli' would merge it into
    ['Escherichia coli'], which another species row already answers to and keeps

Ten reactions, a connected core of ten and an 8/8 enzyme recall were discarded
with it: no ``release_status``, no preserved payload, no PWML. The guard was
right and stays; the **disposition** was the defect (permanent merge rule 7,
``PRODUCT_CONTRACT`` §1).

Every test below therefore FAILS on base ``e648287`` and passes at the tip, on
the behaviour rather than on a symbol: each one calls a function that exists at
both SHAs with the same signature, and at base every one of them raises
``PrefreezeResolutionError`` out of the call under test.

The one control that must NOT move is
:func:`test_the_compound_twin_still_refuses`: D-034 clause 1 ratified fail-closed
for compounds and D-035 clause 7 names a reference case that "must keep
refusing". It passes identically at both SHAs, which is what makes the rest of
this file a change of species disposition and not a weakened guard.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import pytest

from helpers_prefreeze import prefrozen
from t2pw.pwml.ir import SPECIES_CANONICALIZATION_FIELD, build_pwml_ir
from t2pw.pwml.prefreeze_resolution import (
    PrefreezeResolutionError,
    resolve_compounds_prefreeze,
    resolve_species_prefreeze,
    run_prefreeze_resolution,
)

#: The T-106 pair, verbatim. ``Escherichia coli K-12`` is taxonomy-identified and
#: strain-qualified, so rung 4 of the ladder (``_deterministic_species_name``)
#: collapses it onto the binomial -- offline, with no database and no name index,
#: which is what makes every assertion here valid in a bare worktree.
STRAIN = "Escherichia coli K-12"
STRAIN_TAXONOMY = "83333"
BINOMIAL = "Escherichia coli"
BINOMIAL_TAXONOMY = "562"


def _payload(*species: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metadata": {"pathway_name": "lipid A biosynthesis", "pathway_subject": "Metabolic"},
        "entities": {
            "species": [dict(row) for row in species],
            "compounds": [{"name": "UDP-GlcNAc"}, {"name": "UDP-3-O-acyl-GlcNAc"}],
            "proteins": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "LpxA acylation",
                    "inputs": ["UDP-GlcNAc"],
                    "outputs": ["UDP-3-O-acyl-GlcNAc"],
                }
            ]
        },
        "biological_states": [{"name": "cytosol", "species": STRAIN}],
    }


def _t106_payload() -> Dict[str, Any]:
    """The crash shape: a strain row that canonicalizes onto a row already there."""

    return _payload(
        {"name": STRAIN, "taxonomy_id": STRAIN_TAXONOMY},
        {"name": BINOMIAL, "taxonomy_id": BINOMIAL_TAXONOMY},
    )


def _species(payload: Dict[str, Any]) -> list:
    return payload["entities"]["species"]


def _names(payload: Dict[str, Any]) -> list:
    return [row["name"] for row in _species(payload)]


def _taxonomies(payload: Dict[str, Any]) -> list:
    return [row.get("taxonomy_id") for row in _species(payload)]


# ---------------------------------------------------------------------------
# 1. The leg completes, and the payload it completes with is the one it had.
# ---------------------------------------------------------------------------
def test_the_t106_crash_shape_completes_and_the_payload_is_preserved() -> None:
    """G9 correction. At base this call raises and the leg dies; here it returns.

    "Preserved" is asserted on the object, not on a code path: every species row
    is still present, under the name and the taxonomy id it arrived with, and the
    reaction that would have been lost with the leg is untouched. The stage still
    reports ``applied``, because it *did* run and did commit the rows -- what it
    declined is one rename, not the pass.
    """

    payload = _t106_payload()
    before = deepcopy(payload)

    summary = resolve_species_prefreeze(payload, name_index=None)

    assert summary["applied"] is True
    assert _names(payload) == [STRAIN, BINOMIAL]
    assert _taxonomies(payload) == [STRAIN_TAXONOMY, BINOMIAL_TAXONOMY]
    assert payload["processes"] == before["processes"]
    assert payload["entities"]["compounds"] == before["entities"]["compounds"]
    # The declined row is byte-identical to the one handed in, except for the
    # marker that records the declination. No half-applied rename: no raw_name,
    # no alias, no new spelling.
    declined_row = dict(_species(payload)[0])
    marker = declined_row.pop(SPECIES_CANONICALIZATION_FIELD)
    assert declined_row == before["entities"]["species"][0]
    assert marker["status"] == "rename_declined"


def test_the_ambiguity_is_recorded_where_the_contract_puts_it() -> None:
    """``PRODUCT_CONTRACT`` §3: the uncertainty has to be traceable, not implied.

    Three carriers, because three different readers need it. The row marker is
    the durable one -- it survives the freeze, the ``final_mapped.json`` round
    trip and ``deepcopy`` -- and it names the rename that did not happen, the
    name the row kept, and the row it would have merged into.
    """

    payload = _t106_payload()
    summary = resolve_species_prefreeze(payload, name_index=None)

    (record,) = summary["renames_declined"]
    assert record["code"] == "AMBIGUOUS_RENAME_TARGET"
    assert record["source"] == STRAIN
    assert record["kept_name"] == STRAIN
    assert record["declined_target"] == BINOMIAL
    assert record["collides_with"] == [BINOMIAL]
    assert "would merge it into" in record["message"]

    marker = _species(payload)[0][SPECIES_CANONICALIZATION_FIELD]
    assert marker["declined_rename"] == record

    # And the stage's own log does not claim a rename it declined.
    assert summary["rename_map"] == {}
    assert summary["renamed"] == 0
    assert summary["name_canonicalization"] == []


# ---------------------------------------------------------------------------
# 2. The refusal itself still holds -- the two rows are NOT merged.
# ---------------------------------------------------------------------------
def test_the_two_species_rows_are_not_merged_at_the_stage_or_in_the_exporter() -> None:
    """The half of the guard C-082 must not touch (merge rule 6).

    Merging *E. coli K-12* into *E. coli* fuses two organisms: the surviving row
    would carry one organism's name and the other's ``taxonomy_id``, which is a
    **wrong** identity, not a lossy one (``PRODUCT_CONTRACT`` §2, §5). So the
    assertion is carried all the way into the IR: two species records, two
    taxonomy ids, neither row dropped by ``_dedupe_named_rows`` and no
    ``DuplicateNamedRowError``, because after the declination the two rows do not
    share an exporter name key.
    """

    payload = _t106_payload()
    resolve_species_prefreeze(payload, name_index=None)

    assert len(_species(payload)) == 2
    assert len(set(_names(payload))) == 2, "the two organisms collapsed onto one name"

    ir, _report = build_pwml_ir(
        prefrozen(deepcopy(payload), name_index=None),
        pathway_name="lipid A biosynthesis",
        pathway_subject="Metabolic",
        strict_db=False,
        name_index=None,
    )
    assert [row["name"] for row in ir["species"]] == [STRAIN, BINOMIAL]
    assert sorted(
        str(row.get("taxonomy_id") or row.get("taxonomy-id") or "") for row in ir["species"]
    ) == sorted([STRAIN_TAXONOMY, BINOMIAL_TAXONOMY])


# ---------------------------------------------------------------------------
# 3. The run is not reported as a clean canonicalization.
# ---------------------------------------------------------------------------
def test_a_declined_rename_is_published_on_the_review_channel_not_as_a_pass() -> None:
    """D-035 clause 8: it may become a structured review result, never a success.

    ``report["review_required"]`` is not a new channel -- D-029's unreachable
    database already uses it, ``writer.run_pwml_pipeline_export`` returns it as
    ``prefreeze_review_required`` and both Streamlit seams publish it under that
    same name. Routing the declination there is what stops any consumer reading
    the canonicalization as complete: ``ok`` is a verdict, and it is False.
    """

    # The call FIRST, the module-local names after it. On base ``e648287`` this
    # line raises ``PrefreezeResolutionError`` and the test fails on the
    # behaviour, which is what merge gate G9 asks for -- an ImportError on a
    # symbol that does not exist yet would be symbol absence, and symbol absence
    # is not proof.
    report = run_prefreeze_resolution(_t106_payload(), strict_db=False, name_index=None)

    from t2pw.pwml.prefreeze_resolution import (  # noqa: PLC0415
        SPECIES_RENAME_DECLINED_REASON,
        _REVIEW_REQUIRED_REASONS,
    )

    assert report["ok"] is False
    assert report["failures"]["species"] == SPECIES_RENAME_DECLINED_REASON
    assert report["review_required"] == {"species": SPECIES_RENAME_DECLINED_REASON}
    # It is a REVIEW reason, not a defect-in-the-payload reason: every row the
    # payload arrived with is still there and still correct.
    assert SPECIES_RENAME_DECLINED_REASON in _REVIEW_REQUIRED_REASONS


def test_the_declination_never_reports_the_canonicalization_as_release_grade() -> None:
    """The leg must not come out of this stage claiming it is clean.

    Asserted negatively and exhaustively over the stage's OWN verdict surface,
    because that is the surface this card owns: there is no value of ``ok``,
    ``failures`` or ``review_required`` a downstream reader could take for a
    clean pass. The runtime ``release_status`` demotion is NOT constructible from
    this seam -- ``classify_release_status`` is reached only through
    ``strict_quarantine.quarantine_and_close`` and ``streamlit_app.run_pwml_export``,
    neither of which is this card's -- and that residual is reported rather than
    faked here.
    """

    report = run_prefreeze_resolution(_t106_payload(), strict_db=False, name_index=None)
    published = report["review_required"]  # the exact dict all three seams re-publish

    assert report["ok"] is not True
    assert published, "the declination has to be visible to the seam that publishes it"
    assert set(published) == {"species"}

    # And the control: a payload with nothing to decline publishes nothing, so a
    # consumer keying on this cannot be reading a constant.
    clean = run_prefreeze_resolution(
        _payload({"name": BINOMIAL, "taxonomy_id": BINOMIAL_TAXONOMY}),
        strict_db=False,
        name_index=None,
    )
    assert clean["review_required"] == {}
    assert "renames_declined" not in clean["species"]


# ---------------------------------------------------------------------------
# Controls: nothing else moved.
# ---------------------------------------------------------------------------
def test_the_compound_twin_still_refuses() -> None:
    """D-034 clause 1 / D-035 clause 7 -- compounds keep failing closed.

    Identical in shape to the species case and deliberately NOT changed: two
    genuinely distinct compound names canonicalizing onto one target still raise.
    Compounds are participants, so a declined compound rename has connectivity
    consequences a species one does not, and the reference case
    ``PMC13278307…/strict`` under ``C_canned`` is ruled to "must keep refusing".
    This passes identically on base and at tip.
    """

    payload = {
        "metadata": {"pathway_name": "P", "pathway_subject": "Metabolic"},
        "entities": {"compounds": [{"name": "PEtN-lipid A"}, {"name": "modified Lipid A"}]},
        "processes": {"reactions": []},
    }
    with pytest.raises(PrefreezeResolutionError) as excinfo:
        resolve_compounds_prefreeze(
            payload,
            strict_db=False,
            name_index=_TwoNamesOneTarget(),
        )
    assert excinfo.value.code == "AMBIGUOUS_RENAME_TARGET"


class _TwoNamesOneTarget:
    """A name index that canonicalizes two distinct compounds onto one name.

    Local to the control above so it cannot leak into another test's ladder. The
    species index protocol is answered too, and answered with nothing, because
    ``run_prefreeze_resolution`` hands one index to every canonicalizer.
    """

    def compound_canonical(self, **_kwargs: Any) -> Dict[str, Any]:
        return {"name": "lipid A"}

    def species_canonical(self, **_kwargs: Any) -> Dict[str, Any]:
        return {}


def test_an_unambiguous_strain_rename_still_applies() -> None:
    """The over-fire control. Declining is for a collision and nothing else.

    A strain row with no occupant to collide with is exactly the case the ladder
    exists for, and it must still be renamed, logged and marked. A card that
    turned every strain rename into a declination would leave every organism name
    non-canonical and collide on PathWhiz import -- the failure
    ``_deterministic_species_name`` was written to prevent.
    """

    payload = _payload({"name": STRAIN, "taxonomy_id": STRAIN_TAXONOMY})
    summary = resolve_species_prefreeze(payload, name_index=None)

    assert _names(payload) == [BINOMIAL]
    assert summary["rename_map"] == {STRAIN: BINOMIAL}
    assert "renames_declined" not in summary
    assert payload["entities"]["species"][0]["raw_name"] == STRAIN
    assert summary["name_canonicalization"] == [
        {
            "from": STRAIN,
            "to": BINOMIAL,
            "taxonomy_id": STRAIN_TAXONOMY,
            "source": "deterministic_strain_normalization",
        }
    ]


def test_a_reference_the_rename_would_break_still_raises() -> None:
    """``AMBIGUOUS_REFERENCE`` is a different code and a different defect.

    A name a *second entity* answers to cannot be redirected at all, so there is
    no "keep your own name" that resolves it -- the reference is ambiguous either
    way. F-115 does not reach this branch and C-082 does not touch it.
    """

    payload = _payload({"name": STRAIN, "taxonomy_id": STRAIN_TAXONOMY})
    payload["entities"]["proteins"].append({"name": STRAIN})
    before = deepcopy(payload)

    with pytest.raises(PrefreezeResolutionError) as excinfo:
        resolve_species_prefreeze(payload, name_index=None)
    assert excinfo.value.code == "AMBIGUOUS_REFERENCE"
    assert payload == before, "the payload must be untouched on a refusal"

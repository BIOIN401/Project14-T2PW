"""Research mode: FORMAT rules get skipped, biology/provenance only flags.

The contract these tests pin is the whole point of research mode:

* a **format-only** violation (missing external identity, a generated-wrapper
  rule) proceeds in research mode and still aborts in strict;
* a **biology** violation (a reaction with no inputs or outputs) is FLAGGED in
  research mode -- present in the report, never raised -- and still aborts in
  strict;
* a **structural guard** (the payload is not a payload) still aborts in BOTH,
  because there is no candidate pathway to review;
* strict mode is byte-for-byte what it was before the mode existed.

The last one is the regression firewall: every assertion about strict mode here
should hold identically against the pre-change code.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.entity_identity import route_entity_for_mapping  # noqa: E402
from t2pw.pipeline.export_mode import (  # noqa: E402
    PATHWHIZ,
    RESEARCH,
    coerce_mode,
    format_gaps,
    is_research,
    review_flags,
)
from t2pw.pipeline.stage_contracts import (  # noqa: E402
    StageContractError,
    run_stage_contract,
    validate_post_extraction,
    validate_post_remap,
)


def _biology_violation_payload() -> dict:
    """A reaction that claims nothing: no inputs and no outputs."""
    return {
        "entities": {
            "species": [{"name": "Homo sapiens"}],
            "proteins": [{"name": "Novel kinase X"}],
        },
        "processes": {
            "reactions": [{"name": "Unsupported step", "inputs": [], "outputs": []}],
        },
    }


def _format_violation_payload() -> dict:
    """A generated protein-complex wrapper whose component has no UniProt id.

    Every failing rule here exists only so the PathWhiz importer accepts the
    file -- a genuinely novel enzyme legitimately has no accession.
    """
    return {
        "entities": {
            "species": [{"name": "Homo sapiens"}],
            "proteins": [{"name": "Novel kinase X", "species": "Homo sapiens"}],
            "protein_complexes": [
                {
                    "name": "Novel kinase X complex",
                    "generated": True,
                    "components": [{"name": "Novel kinase X"}],
                }
            ],
        },
        "processes": {
            "reactions": [
                {"name": "Step", "inputs": ["ATP"], "outputs": ["ADP"]},
            ],
        },
    }


# --------------------------------------------------------------------------
# Biology violations: flagged in research, still fatal in strict.
# --------------------------------------------------------------------------


def test_biology_violation_aborts_in_strict_mode() -> None:
    with pytest.raises(StageContractError) as exc_info:
        validate_post_extraction(_biology_violation_payload())

    assert "reaction_missing_participants" in {
        issue["code"] for issue in exc_info.value.report["errors"]
    }


def test_biology_violation_is_flagged_not_aborted_in_research_mode() -> None:
    report = run_stage_contract(
        validate_post_extraction, _biology_violation_payload(), mode=RESEARCH
    )

    assert report["ok"] is True
    assert report["effect_on_failure"] == "annotate_only"
    assert report["research_review_required"] is True

    flagged = {issue["code"] for issue in review_flags(report)}
    assert "reaction_missing_participants" in flagged
    # Downgraded, not deleted: the finding keeps its identity and its pointer.
    flag = next(i for i in review_flags(report) if i["code"] == "reaction_missing_participants")
    assert flag["pointer"] == "/processes/reactions/0"
    assert flag["was_blocking"] is True
    assert flag["research_category"] == "BIOLOGY/PROVENANCE"


def test_biology_violation_is_not_counted_as_a_skipped_format_rule() -> None:
    report = run_stage_contract(
        validate_post_extraction, _biology_violation_payload(), mode=RESEARCH
    )

    assert format_gaps(report) == []
    assert report["research_summary"]["format_skipped"] == 0
    assert report["research_summary"]["review_flagged"] == 1


# --------------------------------------------------------------------------
# Format violations: skipped in research, still fatal in strict.
# --------------------------------------------------------------------------


def test_format_violation_aborts_in_strict_mode() -> None:
    with pytest.raises(StageContractError) as exc_info:
        validate_post_remap(_format_violation_payload())

    codes = {issue["code"] for issue in exc_info.value.report["errors"]}
    assert "generated_wrapper_component_missing_external_identity" in codes


def test_format_violation_proceeds_in_research_mode() -> None:
    report = run_stage_contract(
        validate_post_remap, _format_violation_payload(), mode=RESEARCH
    )

    assert report["ok"] is True
    skipped = {issue["code"] for issue in format_gaps(report)}
    assert "generated_wrapper_component_missing_external_identity" in skipped
    # A skipped FORMAT rule must never be silent: it is still recorded.
    assert report["research_summary"]["format_skipped"] >= 1
    assert all(i["research_category"] == "FORMAT" for i in format_gaps(report))


# --------------------------------------------------------------------------
# Structural guards abort in BOTH modes.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload, expected_code",
    [
        ("not a payload", "invalid_payload"),
        ({"processes": {}}, "entities_required"),
        ({"entities": {}}, "processes_required"),
    ],
)
def test_structural_guards_still_abort_in_research_mode(payload, expected_code) -> None:
    with pytest.raises(StageContractError) as exc_info:
        run_stage_contract(validate_post_extraction, payload, mode=RESEARCH)

    assert expected_code in {issue["code"] for issue in exc_info.value.report["errors"]}


# --------------------------------------------------------------------------
# Strict mode is unchanged, and neither mode mutates the payload.
# --------------------------------------------------------------------------


def test_strict_mode_is_a_plain_passthrough() -> None:
    payload = _format_violation_payload()
    # A payload that passes cleanly, so both calls take the success path.
    payload["entities"]["proteins"][0]["uniprot"] = "P00001"

    direct = validate_post_remap(deepcopy(payload))
    wrapped = run_stage_contract(validate_post_remap, deepcopy(payload), mode=PATHWHIZ)

    assert wrapped == direct
    # No research bookkeeping leaks into a strict report.
    assert "export_mode" not in wrapped
    assert "research_summary" not in wrapped


def test_default_mode_is_strict() -> None:
    with pytest.raises(StageContractError):
        run_stage_contract(validate_post_extraction, _biology_violation_payload())


@pytest.mark.parametrize("mode", [PATHWHIZ, RESEARCH])
def test_validators_never_mutate_the_payload(mode) -> None:
    payload = _biology_violation_payload()
    before = deepcopy(payload)
    try:
        run_stage_contract(validate_post_extraction, payload, mode=mode)
    except StageContractError:
        pass
    assert payload == before


# --------------------------------------------------------------------------
# Mode coercion: a typo must degrade to strict, never to relaxed.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("research", RESEARCH),
        ("Research (relaxed)", RESEARCH),
        ("relaxed", RESEARCH),
        ("pathwhiz", PATHWHIZ),
        ("strict PWML", PATHWHIZ),
        ("", PATHWHIZ),
        (None, PATHWHIZ),
        ("nonsense", PATHWHIZ),
    ],
)
def test_unrecognised_mode_falls_back_to_strict(value, expected) -> None:
    assert coerce_mode(value) == expected
    assert is_research(value) is (expected == RESEARCH)


# --------------------------------------------------------------------------
# Name leniency: a ':' must not re-route a biological name to the complex mapper.
# --------------------------------------------------------------------------


def test_colon_name_routes_to_complex_in_strict_mode() -> None:
    assert route_entity_for_mapping("receptor:kinase", "compound")["route"] == "complex"


def test_colon_name_is_left_alone_when_names_are_lenient() -> None:
    assert (
        route_entity_for_mapping("receptor:kinase", "compound", lenient_names=True)["route"]
        != "complex"
    )


def test_explicit_type_hint_still_wins_under_lenient_names() -> None:
    assert (
        route_entity_for_mapping("receptor:kinase", "protein_complex", lenient_names=True)["route"]
        == "complex"
    )


def test_biochemical_colon_name_is_unaffected_by_leniency() -> None:
    # "OPC-6:0-CoA" is a lipid, not a complex, in both modes.
    for lenient in (False, True):
        assert (
            route_entity_for_mapping("OPC-6:0-CoA", "compound", lenient_names=lenient)["route"]
            == "compound"
        )

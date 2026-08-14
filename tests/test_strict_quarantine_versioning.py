"""One decision per payload VERSION *under one set of RULES*, and research never mutates.

Three failures that a session-scoped decision cannot see:

* **A stale decision authorizing a payload it never judged.** The boundary runs
  before refinement review. The reviewer then edits the graph -- deleting the
  requested core is one click -- and grounding rewrites entity identifiers on the
  way into the exporter. A decision made before either is a statement about a
  graph that no longer exists, and reusing it exports material nothing admitted.
* **A strict quarantine applied to a research candidate.** Research mode exists
  for a novel pathway whose enzymes have no accessions yet, so every one of them
  is ``quarantined_unmapped_entity`` by construction. Applying that would delete
  exactly the rows the mode was built to keep and hand back a smaller strict
  graph in place of the candidate.
* **A decision reused across a change of rules.** The payload half of the check
  cannot see this one at all: a research decision returns the candidate untouched,
  so its resulting hash *is* the candidate's, and a strict export handed that same
  candidate matches on payload while having no strict decision behind it. The
  controls have to be compared too, which is what ``decision_input_hash`` is for.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.export_mode import PATHWHIZ, RESEARCH  # noqa: E402
from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    COVERAGE_REPORT_FILENAME,
    QUARANTINE_HISTORY_DIRNAME,
    QUARANTINE_REPORT_FILENAME,
    QUARANTINED_UNMAPPED_ENTITY,
    QUARANTINE_POLICY_VERSION,
    admitted_payload_hash,
    canonical_decision_inputs,
    clear_quarantine_artifacts,
    decision_identifier,
    decision_input_hash,
    decision_inputs_match,
    decision_matches,
    quarantine_and_close,
    quarantine_review_flags,
    write_quarantine_artifacts,
)

from test_strict_quarantine_contract_alignment import _base, _strict_failures  # noqa: E402


STAGE_ZERO_CONTEXT = {
    "key_compounds": ["L-glutamate", "glutathione"],
    "key_proteins": ["glutamate-cysteine ligase"],
}


def _controls(**overrides: Any) -> Dict[str, Any]:
    """The decision controls, defaulted to :func:`quarantine_and_close`'s own.

    A matcher call has to name the *same* controls the decision was taken under,
    which is the point: reuse is not a property of the report alone.
    """

    controls: Dict[str, Any] = {"mode": PATHWHIZ, "strict_db": True, "pathway_context": None}
    controls.update(overrides)
    return controls


# ── The hash itself ─────────────────────────────────────────────────────────


def test_the_hash_is_deterministic_and_key_order_independent() -> None:
    payload = _base()
    reordered = json.loads(json.dumps(payload))
    reordered["processes"] = reordered.pop("processes")
    reordered["entities"] = reordered.pop("entities")

    assert admitted_payload_hash(payload) == admitted_payload_hash(deepcopy(payload))
    assert admitted_payload_hash(payload) == admitted_payload_hash(reordered)


def test_any_content_change_changes_the_hash() -> None:
    payload = _base()
    before = admitted_payload_hash(payload)

    payload["processes"]["reactions"][0]["inputs"].append("glycine")

    assert admitted_payload_hash(payload) != before


def test_the_report_records_the_payload_it_judged_not_the_one_it_produced() -> None:
    payload = _base()
    payload["entities"]["compounds"].append({"name": "unreferenced ghost"})

    result = quarantine_and_close(payload, strict_db=True)
    report = result.quarantine_report

    # Closure removed the ghost, so the two hashes differ. Reuse is gated on the
    # RESULTING hash: the exporter holds what quarantine produced, and matching
    # that against what quarantine was given would fail on every run where
    # anything was actually removed.
    assert report["admitted_payload_hash"] == admitted_payload_hash(payload)
    assert report["resulting_payload_hash"] == admitted_payload_hash(result.payload)
    assert report["admitted_payload_hash"] != report["resulting_payload_hash"]
    assert decision_matches(report, result.payload, **_controls()) is True
    assert decision_matches(report, payload, **_controls()) is False


def test_a_report_with_no_hash_never_matches() -> None:
    """Reports written before these fields cannot prove what they judged."""

    assert decision_matches({"ok": True}, _base(), **_controls()) is False
    assert decision_matches({"resulting_payload_hash": ""}, _base(), **_controls()) is False
    assert decision_matches(None, _base(), **_controls()) is False
    # An admitted hash alone does not authorize anything: the payload it names
    # still contains the rows the decision quarantined.
    assert (
        decision_matches(
            {"admitted_payload_hash": admitted_payload_hash(_base())}, _base(), **_controls()
        )
        is False
    )
    # A payload hash that matches is still not enough without the input hash: a
    # pre-schema-3 report cannot say which rules produced it.
    result = quarantine_and_close(_base(), strict_db=True)
    legacy = {k: v for k, v in result.quarantine_report.items() if k != "decision_input_hash"}
    assert decision_matches(legacy, result.payload, **_controls()) is False


# ── The decision INPUT, not just the payload ────────────────────────────────
#
# Recording mode/strict_db/thresholds on the report is not a check; nothing
# consults a field nobody compares. These pin the comparison.


def _decided(payload: Dict[str, Any], **controls: Any):
    """Quarantine ``payload`` under ``controls``, spelled the same way twice."""

    filled = _controls(**controls)
    return quarantine_and_close(
        payload,
        mode=filled["mode"],
        strict_db=filled["strict_db"],
        pathway_context=filled["pathway_context"],
    )


def test_a_research_report_cannot_authorize_a_strict_export() -> None:
    """The worst reuse there is.

    A research decision quarantines nothing and returns the candidate unchanged,
    so its resulting hash is the candidate's. Hand that candidate to a strict
    export and the payload hash matches perfectly -- and every unmapped process
    the strict run exists to stop would ship, authorized by a report that never
    judged it under strict rules.
    """

    payload = _base()
    payload["entities"]["proteins"].append({"name": "unidentified enzyme", "species": "Homo sapiens"})
    payload["processes"]["reactions"][0]["modifiers"] = [{"entity": "unidentified enzyme"}]

    research = _decided(payload, mode=RESEARCH)

    # The payload half matches: research changed nothing.
    assert research.quarantine_report["resulting_payload_hash"] == admitted_payload_hash(
        research.payload
    )
    assert (
        decision_matches(research.quarantine_report, research.payload, **_controls(mode=RESEARCH))
        is True
    )
    # The decision half does not, and that is what refuses the reuse.
    assert (
        decision_matches(research.quarantine_report, research.payload, **_controls(mode=PATHWHIZ))
        is False
    )
    # And the strict verdict on the same payload really is different.
    strict = _decided(payload, mode=PATHWHIZ)
    assert strict.quarantine_report["counts"][QUARANTINED_UNMAPPED_ENTITY] == 1


def test_a_strict_report_cannot_authorize_a_research_export() -> None:
    """The reverse, which loses material rather than shipping it.

    A strict decision names a reduced graph. Reusing it for a research export
    would hand the reviewer that reduced graph and call it the untouched
    candidate -- research mode's one promise, broken by reuse.
    """

    boundary = _decided(_base(), mode=PATHWHIZ, pathway_context=STAGE_ZERO_CONTEXT)

    assert (
        decision_matches(
            boundary.quarantine_report,
            boundary.payload,
            **_controls(mode=PATHWHIZ, pathway_context=STAGE_ZERO_CONTEXT),
        )
        is True
    )
    assert (
        decision_matches(
            boundary.quarantine_report,
            boundary.payload,
            **_controls(mode=RESEARCH, pathway_context=STAGE_ZERO_CONTEXT),
        )
        is False
    )


def test_a_db_less_decision_cannot_authorize_a_strict_db_export() -> None:
    """``strict_db=False`` drops the protein-identity requirement entirely."""

    payload = _base()
    payload["entities"]["proteins"].append({"name": "unidentified enzyme", "species": "Homo sapiens"})
    payload["processes"]["reactions"][0]["modifiers"] = [{"entity": "unidentified enzyme"}]

    lax = _decided(payload, strict_db=False)

    assert lax.quarantine_report["counts"][QUARANTINED_UNMAPPED_ENTITY] == 0, (
        "strict_db=False must really be the laxer verdict, or this proves nothing"
    )
    assert decision_matches(lax.quarantine_report, lax.payload, **_controls(strict_db=False)) is True
    assert decision_matches(lax.quarantine_report, lax.payload, **_controls(strict_db=True)) is False


def test_a_changed_pathway_context_invalidates_reuse() -> None:
    """The requested core is a decision input, not decoration.

    Coverage is measured against it, so the same surviving graph passes under one
    request and fails under another. A decision taken for "glutathione
    biosynthesis" must not authorize an export of a pathway asked to be the TCA
    cycle.
    """

    boundary = _decided(_base(), pathway_context=STAGE_ZERO_CONTEXT)

    assert (
        decision_matches(
            boundary.quarantine_report, boundary.payload, **_controls(pathway_context=STAGE_ZERO_CONTEXT)
        )
        is True
    )
    other = {"key_compounds": ["citrate"], "key_proteins": ["aconitase"]}
    assert (
        decision_matches(
            boundary.quarantine_report, boundary.payload, **_controls(pathway_context=other)
        )
        is False
    )
    # Dropping it entirely is also a change: it switches the coverage regime.
    assert (
        decision_matches(boundary.quarantine_report, boundary.payload, **_controls())
        is False
    )


def test_an_irrelevant_context_field_does_not_invalidate_reuse() -> None:
    """The other half of "canonical": only what can change a verdict counts.

    ``pathway_context`` carries the paper title and organism guess alongside the
    core terms, and none of those reach a verdict. Hashing the raw context would
    void a good decision because a title moved, and every spurious invalidation is
    a re-run the reviewer waits through for nothing.
    """

    boundary = _decided(_base(), pathway_context=STAGE_ZERO_CONTEXT)
    same_core = {**STAGE_ZERO_CONTEXT, "pathway_name": "renamed", "likely_organism": "Mus musculus"}

    assert (
        decision_matches(
            boundary.quarantine_report, boundary.payload, **_controls(pathway_context=same_core)
        )
        is True
    )


def test_changed_thresholds_invalidate_reuse() -> None:
    """Every configurable knob that can flip the verdict is in the hash."""

    base_inputs = canonical_decision_inputs(**_controls(pathway_context=STAGE_ZERO_CONTEXT))
    reference = decision_input_hash(**_controls(pathway_context=STAGE_ZERO_CONTEXT))

    for field, value in (
        ("min_core_processes", 2),
        ("min_core_coverage", 0.9),
        ("confidence_floor", 0.5),
        ("max_iterations", 5),
        ("policy_version", "some-other-policy"),
    ):
        assert base_inputs[field] != value, field
        moved = decision_input_hash(**_controls(pathway_context=STAGE_ZERO_CONTEXT), **{field: value})
        assert moved != reference, f"{field} does not reach the decision hash"


def test_the_policy_version_is_recorded_and_hashed() -> None:
    result = quarantine_and_close(_base(), strict_db=True)

    assert result.quarantine_report["policy_version"] == QUARANTINE_POLICY_VERSION
    assert result.quarantine_report["decision_inputs"]["policy_version"] == QUARANTINE_POLICY_VERSION
    assert decision_inputs_match(result.quarantine_report, **_controls()) is True
    assert (
        decision_inputs_match(result.quarantine_report, **_controls(policy_version="next"))
        is False
    )


def test_an_identical_payload_under_identical_policy_still_reuses() -> None:
    """The fast path has to survive all of the above, or nothing ever reuses."""

    boundary = _decided(_base(), pathway_context=STAGE_ZERO_CONTEXT)
    controls = _controls(pathway_context=deepcopy(STAGE_ZERO_CONTEXT))

    assert decision_matches(boundary.quarantine_report, deepcopy(boundary.payload), **controls) is True


# ── Refinement moved the payload ────────────────────────────────────────────


def test_a_stale_decision_does_not_authorize_a_payload_whose_core_was_deleted() -> None:
    """The regression this whole mechanism exists for.

    The boundary admits a valid two-reaction glutathione pathway. Refinement then
    deletes both core reactions and leaves one unrelated survivor. The stored
    report still says ``ok`` -- it was true of the graph it judged -- and reusing
    it would export a pathway that is not the one requested.
    """

    admitted = _base()
    admitted["processes"]["reactions"].append(
        {
            "name": "citrate isomerisation",
            "inputs": ["citrate"],
            "outputs": ["isocitrate"],
            "enzymes": [],
            "biological_state": "cytosol",
        }
    )
    admitted["entities"]["compounds"].extend([{"name": "citrate"}, {"name": "isocitrate"}])
    admitted["element_locations"]["compound_locations"].extend(
        [
            {"compound": "citrate", "biological_state": "cytosol"},
            {"compound": "isocitrate", "biological_state": "cytosol"},
        ]
    )

    boundary = quarantine_and_close(
        admitted, strict_db=True, pathway_context=STAGE_ZERO_CONTEXT
    )
    assert boundary.ok is True

    # The reviewer deletes the requested core.
    refined = deepcopy(boundary.payload)
    refined["processes"]["reactions"] = [
        row for row in refined["processes"]["reactions"] if row["name"] == "citrate isomerisation"
    ]

    assert (
        decision_matches(
            boundary.quarantine_report, refined, **_controls(pathway_context=STAGE_ZERO_CONTEXT)
        )
        is False
    )

    rejudged = quarantine_and_close(
        refined, strict_db=True, pathway_context=STAGE_ZERO_CONTEXT
    )
    # MOVED BY C-041a (D-002, LOCKED). Base SHA: ``ok is False``. What this test
    # is about -- a stale decision does not authorize a payload whose core was
    # deleted -- is asserted by ``decision_matches`` above and is unchanged. The
    # re-judgement of the reduced payload now says review_required rather than
    # refusing outright, and still never says strict success.
    assert rejudged.ok is True
    assert rejudged.quarantine_report["release"]["status"] == "review_required"
    assert rejudged.quarantine_report["release"]["strict_acceptance_eligible"] is False
    assert any(
        reason.startswith("minimum_core:")
        for reason in rejudged.quarantine_report["review_reasons"]
    )


def test_a_refinement_that_adds_an_unexportable_process_loses_only_that_process() -> None:
    payload = _base()
    boundary = quarantine_and_close(payload, strict_db=True, pathway_context=STAGE_ZERO_CONTEXT)
    assert boundary.quarantine_report["counts"][QUARANTINED_UNMAPPED_ENTITY] == 0

    refined = deepcopy(boundary.payload)
    refined["entities"]["proteins"].append({"name": "OPR3", "species": "Homo sapiens"})
    refined["entities"]["compounds"].extend([{"name": "OPDA"}, {"name": "OPC-8:0"}])
    refined["element_locations"]["compound_locations"].extend(
        [
            {"compound": "OPDA", "biological_state": "cytosol"},
            {"compound": "OPC-8:0", "biological_state": "cytosol"},
        ]
    )
    refined["processes"]["reactions"].append(
        {
            "name": "peripheral OPDA reduction",
            "inputs": ["OPDA"],
            "outputs": ["OPC-8:0"],
            "modifiers": [{"entity": "OPR3"}],
            "biological_state": "cytosol",
        }
    )

    assert (
        decision_matches(
            boundary.quarantine_report, refined, **_controls(pathway_context=STAGE_ZERO_CONTEXT)
        )
        is False
    )
    assert _strict_failures(refined), "the added process must really break the gates"

    rejudged = quarantine_and_close(
        refined, strict_db=True, pathway_context=STAGE_ZERO_CONTEXT
    )

    assert rejudged.ok is True
    assert rejudged.states()["peripheral OPDA reduction"] == QUARANTINED_UNMAPPED_ENTITY
    assert [row["name"] for row in rejudged.payload["processes"]["reactions"]] == [
        "gamma-glutamylcysteine synthesis",
        "glutathione synthesis",
    ]
    assert _strict_failures(rejudged.payload) == []


def test_an_unchanged_payload_reuses_its_decision() -> None:
    """The fast path. Re-quarantining on every export would be the other bug.

    A clean payload comes out of closure identical to the way it went in, so the
    boundary's decision is about the payload the exporter holds and is reused --
    which is what makes "one decision per payload version" a rule rather than
    "re-run it every time and hope".
    """

    boundary = quarantine_and_close(_base(), strict_db=True, pathway_context=STAGE_ZERO_CONTEXT)

    assert (
        decision_matches(
            boundary.quarantine_report,
            boundary.payload,
            **_controls(pathway_context=STAGE_ZERO_CONTEXT),
        )
        is True
    )


def test_grounding_is_just_another_payload_mutation() -> None:
    """``apply_grounding`` runs inside the exporter, AFTER the boundary.

    It writes identifier fields onto entity rows, which is a content change like
    any other, so a decision made before it is a decision about a different
    payload version. This asserts the mutation really happens and that the hash
    sees it -- a grounding dictionary that matched nothing would make the test
    vacuous, so the fields are checked on the way past.
    """

    from t2pw.mapping.grounding import apply_grounding

    boundary = quarantine_and_close(_base(), strict_db=True, pathway_context=STAGE_ZERO_CONTEXT)
    assert (
        decision_matches(
            boundary.quarantine_report,
            boundary.payload,
            **_controls(pathway_context=STAGE_ZERO_CONTEXT),
        )
        is True
    )

    grounded, report = apply_grounding(
        deepcopy(boundary.payload),
        {"compounds": {"glutathione": {"hmdb": "HMDB0000125", "chebi": "CHEBI:16856"}}},
    )

    assert report["sections"]["compounds"]["fields_added"] == 2, "grounding must have changed something"
    grounded_row = next(
        row for row in grounded["entities"]["compounds"] if row["name"] == "glutathione"
    )
    assert grounded_row["hmdb"] == "HMDB0000125"

    assert admitted_payload_hash(grounded) != admitted_payload_hash(boundary.payload)
    assert (
        decision_matches(
            boundary.quarantine_report, grounded, **_controls(pathway_context=STAGE_ZERO_CONTEXT)
        )
        is False
    )


# ── History artifacts ───────────────────────────────────────────────────────


def test_a_superseded_decision_is_archived_rather_than_overwritten(tmp_path: Path) -> None:
    first = quarantine_and_close(_base(), strict_db=True, pathway_context=STAGE_ZERO_CONTEXT)
    write_quarantine_artifacts(first, tmp_path)

    second_payload = deepcopy(first.payload)
    second_payload["processes"]["reactions"].pop()
    second = quarantine_and_close(
        second_payload, strict_db=True, pathway_context=STAGE_ZERO_CONTEXT
    )
    write_quarantine_artifacts(second, tmp_path)

    current = json.loads((tmp_path / QUARANTINE_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert current["admitted_payload_hash"] == second.quarantine_report["admitted_payload_hash"]

    archive_key = decision_identifier(first.quarantine_report).replace(":", "_")
    archived = tmp_path / QUARANTINE_HISTORY_DIRNAME / archive_key / QUARANTINE_REPORT_FILENAME
    assert archived.exists(), "the first decision is the only record of why v1 was admitted"
    assert (
        json.loads(archived.read_text(encoding="utf-8"))["admitted_payload_hash"]
        == first.quarantine_report["admitted_payload_hash"]
    )
    # All four are archived, not just the report.
    assert (tmp_path / QUARANTINE_HISTORY_DIRNAME / archive_key / COVERAGE_REPORT_FILENAME).exists()


def test_strict_and_research_decisions_for_one_payload_do_not_collide(tmp_path: Path) -> None:
    """The reason history is keyed by the decision id and not by the payload.

    Both decisions here judge byte-identical input, so they share
    ``admitted_payload_hash`` exactly -- and they reach opposite conclusions. A
    payload-keyed archive would file the second on top of the first and lose the
    one that said "this would not export".
    """

    payload = _base()
    payload["entities"]["proteins"].append({"name": "unidentified enzyme", "species": "Homo sapiens"})
    payload["processes"]["reactions"][0]["modifiers"] = [{"entity": "unidentified enzyme"}]

    strict = quarantine_and_close(deepcopy(payload), strict_db=True, mode=PATHWHIZ)
    write_quarantine_artifacts(strict, tmp_path)
    research = quarantine_and_close(deepcopy(payload), strict_db=True, mode=RESEARCH)
    write_quarantine_artifacts(research, tmp_path)

    assert (
        strict.quarantine_report["admitted_payload_hash"]
        == research.quarantine_report["admitted_payload_hash"]
    ), "the collision this guards against requires identical input"
    assert strict.quarantine_report["decision_id"] != research.quarantine_report["decision_id"]

    archived = (
        tmp_path
        / QUARANTINE_HISTORY_DIRNAME
        / decision_identifier(strict.quarantine_report).replace(":", "_")
        / QUARANTINE_REPORT_FILENAME
    )
    assert archived.exists(), "the strict decision was overwritten by the research one"
    kept = json.loads(archived.read_text(encoding="utf-8"))
    assert kept["export_mode"] == PATHWHIZ
    assert kept["counts"][QUARANTINED_UNMAPPED_ENTITY] == 1
    current = json.loads((tmp_path / QUARANTINE_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert current["export_mode"] == RESEARCH


def test_rewriting_the_same_payload_version_archives_nothing(tmp_path: Path) -> None:
    """Only a *different* version supersedes; a rerun is not history."""

    result = quarantine_and_close(_base(), strict_db=True)
    write_quarantine_artifacts(result, tmp_path)
    write_quarantine_artifacts(result, tmp_path)

    assert not (tmp_path / QUARANTINE_HISTORY_DIRNAME).exists()


def test_clearing_removes_the_current_set_and_keeps_the_archive(tmp_path: Path) -> None:
    first = quarantine_and_close(_base(), strict_db=True)
    write_quarantine_artifacts(first, tmp_path)
    second_payload = deepcopy(first.payload)
    second_payload["processes"]["reactions"].pop()
    write_quarantine_artifacts(quarantine_and_close(second_payload, strict_db=True), tmp_path)

    removed = clear_quarantine_artifacts(tmp_path)

    assert len(removed) == 4
    assert not (tmp_path / QUARANTINE_REPORT_FILENAME).exists()
    assert (tmp_path / QUARANTINE_HISTORY_DIRNAME).exists()


def test_clearing_an_empty_directory_is_a_no_op(tmp_path: Path) -> None:
    assert clear_quarantine_artifacts(tmp_path) == []


# ── Research mode never mutates ─────────────────────────────────────────────


def _research_candidate() -> Dict[str, Any]:
    """A novel pathway: the enzyme is real, evidenced, and has no accession.

    This is the shape research mode exists for, and the shape a strict
    quarantine deletes.
    """

    payload = _base()
    payload["entities"]["proteins"].append(
        {
            "name": "novel 2026 oxidoreductase",
            "species": "Homo sapiens",
            "evidence": "Figure 3 of the source paper.",
        }
    )
    payload["entities"]["compounds"].append({"name": "novel metabolite"})
    payload["element_locations"]["compound_locations"].append(
        {"compound": "novel metabolite", "biological_state": "cytosol"}
    )
    payload["processes"]["reactions"].append(
        {
            "name": "novel oxidation",
            "inputs": ["glutathione"],
            "outputs": ["novel metabolite"],
            "modifiers": [{"entity": "novel 2026 oxidoreductase"}],
            "biological_state": "cytosol",
            "evidence": "Figure 3 of the source paper.",
        }
    )
    return payload


def test_research_mode_keeps_the_candidate_byte_for_byte() -> None:
    payload = _research_candidate()
    before = deepcopy(payload)

    result = quarantine_and_close(payload, strict_db=True, mode=RESEARCH)

    assert result.payload == before
    assert result.ok is True
    assert result.refusal_reasons == []
    assert result.removed_entity_report["counts"] == {
        "entities": 0,
        "locations": 0,
        "biological_states": 0,
    }


def test_research_mode_still_reports_what_a_strict_run_would_have_removed() -> None:
    """Fail-open must never mean fail-silent."""

    result = quarantine_and_close(_research_candidate(), strict_db=True, mode=RESEARCH)
    flags = quarantine_review_flags(result.quarantine_report)

    assert [flag["name"] for flag in flags] == ["novel oxidation"]
    assert flags[0]["state"] == QUARANTINED_UNMAPPED_ENTITY
    assert flags[0]["effect"] == "annotate_only"
    assert result.quarantine_report["research_mode"]["applied"] is False
    assert result.quarantine_report["export_mode"] == RESEARCH


def test_strict_mode_on_the_same_candidate_does_remove_it() -> None:
    """The contrast that makes the research assertion mean something."""

    strict = quarantine_and_close(_research_candidate(), strict_db=True, mode=PATHWHIZ)

    assert strict.states()["novel oxidation"] == QUARANTINED_UNMAPPED_ENTITY
    assert "novel oxidation" not in [
        row["name"] for row in strict.payload["processes"]["reactions"]
    ]
    assert quarantine_review_flags(strict.quarantine_report) == []


def test_research_mode_does_not_block_even_when_no_core_survives() -> None:
    """A research run has no minimum core: there is nothing to refuse."""

    payload = _research_candidate()
    for reaction in payload["processes"]["reactions"]:
        reaction["modifiers"] = [{"entity": "novel 2026 oxidoreductase"}]
    before = deepcopy(payload)

    result = quarantine_and_close(
        payload, strict_db=True, mode=RESEARCH, pathway_context=STAGE_ZERO_CONTEXT
    )

    assert result.ok is True
    assert result.payload == before
    assert result.quarantine_report["research_mode"]["would_have_refused"], (
        "the strict verdict is still recorded, it is just not acted on"
    )

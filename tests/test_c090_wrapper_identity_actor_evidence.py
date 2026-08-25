"""C-090 / F-117: a one-component wrapper is named by its component, and the gate sees it.

``entities.protein_complexes["Lanosterol 14-alpha demethylase"]`` carries
``pathbank_protein_complex_id 442`` and exactly ONE component, ``CYP51A1``. The paper
writes the gene symbol; the payload writes the PathBank canonical name. Under D-064 those
are the SAME protein identity, so the actor-evidence finding raised against
``runs_verify/2026-08-24_1428/papers/PMC12782028/strict`` -- an actor cited against the span
*"CYP51A1 catalyzes the conversion of lansterol to 4,4-dimethyl..."* -- was measurably
false, and the symbol that proves it was already on the record.

``test_a_*`` is a **G9 CORRECTION** of pre-existing observable behaviour and fails
behaviourally on the base SHA: at base ``_check_actor_evidence`` is passed only
``processes``, never sees the component list, and reports the finding. The rest are
labelled **NEW ACCEPTANCE** where they pin the new rule's shape, except ``test_b_*``, the
**ANTI-WIDENING** test, which is the whole point of the card: 3623 ``enterobactin
synthase`` has FOUR components (EntB, EntD, EntF, EntE), so F-116's superset must KEEP
FAILING this gate. If it stops firing, the rule was widened into the thing the check
exists to catch.

Every fixture below is a MEASURED shape lifted from the committed T-106 corpus
(``runs_verify/2026-08-24_1428``, ``evidence/g11/C-090/01-shape-probe.json``), not an
invention. The corpus classification (9 findings: A=2, B=3, C=4) is reproduced by
``test_g_*`` against the committed legs rather than re-derived as a study.

Offline and deterministic: no network, no LLM, no live paper leg. Only committed JSON.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench import semantic as _s  # noqa: E402
from t2pw.bench import semantic_production as sp  # noqa: E402
from t2pw.pipeline.release_status import (  # noqa: E402
    REASON_SEMANTIC_EVALUATION_FAILED,
    RELEASE_READY,
    REVIEW_REQUIRED,
    SEMANTIC_GATING_CHECKS,
    classify_release_status,
    semantic_verdict,
)

#: The T-106 corpus. `t106_verify_plan.txt` names this run directory; the shape probe
#: measured exactly NINE actor-evidence findings across its 20 legs.
CORPUS = ROOT / "runs_verify" / "2026-08-24_1428" / "papers"

#: The class-A leg and the class-B legs, by (paper, mode).
LEG_A = ("PMC12782028", "strict")
LEGS_B = (("PMC12096016", "strict"), ("PMC12452463", "strict"))


def _leg(paper: str, mode: str) -> Dict[str, Any]:
    path = CORPUS / paper / mode / "final_mapped.json"
    assert path.is_file(), f"committed corpus leg is missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Measured fixtures. Component lists, spans and actor rows are verbatim corpus shapes.
# ---------------------------------------------------------------------------
#: `entities.protein_complexes` row 442 of `PMC12782028/strict`. ONE component.
WRAPPER_442: Dict[str, Any] = {
    "name": "Lanosterol 14-alpha demethylase",
    "pathbank_complex_id": 442,
    "pathbank_protein_complex_id": 442,
    "components": [{
        "name": "CYP51A1",
        "pathbank_protein_id": 259,
        "stoichiometry": 1,
        "uniprot": "Q16850",
        "mapped_ids": {"uniprot": "Q16850", "pathbank_protein_id": "259"},
        "gene_name": "CYP51A1",
        "species_id": 1,
    }],
    "species_id": 1,
}

#: `entities.protein_complexes` row 3623 of `PMC12096016/strict`. FOUR components.
WRAPPER_3623: Dict[str, Any] = {
    "name": "enterobactin synthase",
    "pathbank_complex_id": 3623,
    "pathbank_protein_complex_id": 3623,
    "components": [
        {"name": "EntB", "pathbank_protein_id": 6224, "stoichiometry": 1,
         "uniprot": "P0ADI4", "gene_name": "entB", "species_id": 3},
        {"name": "EntD", "pathbank_protein_id": 6383, "stoichiometry": 1,
         "uniprot": "P19925", "gene_name": "entD", "species_id": 3},
        {"name": "EntF", "pathbank_protein_id": 6312, "stoichiometry": 1,
         "uniprot": "P11454", "gene_name": "entF", "species_id": 3},
        {"name": "EntE", "pathbank_protein_id": 6301, "stoichiometry": 1,
         "uniprot": "P10378", "gene_name": "entE", "species_id": 3},
    ],
    "species_id": 3,
}

#: The class-A span, verbatim from `/processes/reactions/1/enzymes/0` of `PMC12782028/strict`.
SPAN_A = "CYP51A1 catalyzes the conversion of lansterol to 4,4-dimethyl..."

#: The class-B span, verbatim from `/processes/reactions/2/enzymes/0` of `PMC12452463/strict`.
#: It NAMES EntE, one of 3623's four components -- which is exactly why a rule that read
#: multi-component wrappers would rescue F-116's superset.
SPAN_B = (
    "2,3-Dihydroxybenzoate-AMP ligase (EntE) activates DHB by adenylation, forming DHB-AMP"
)


def _reaction(name: str, actor_name: str, span: str) -> Dict[str, Any]:
    """One retained reaction naming one enzyme actor that cites one span."""

    return {
        "name": name,
        "enzymes": [{"entity": actor_name, "role": "catalyst", "evidence": span}],
    }


def _payload(complexes: List[Dict[str, Any]], reactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "entities": {"compounds": [], "proteins": [], "protein_complexes": deepcopy(complexes)},
        "processes": {"reactions": deepcopy(reactions)},
    }


def _check(payload: Dict[str, Any]) -> Any:
    """The actor-evidence verdict THROUGH THE PRODUCTION ENTRY POINT.

    Deliberately not a direct call to ``_check_actor_evidence``: the whole defect is that
    the caller withheld the entity table, so a test that reached past the caller would
    prove nothing about production and -- worse for G9 -- would fail at the base SHA with a
    ``TypeError`` on the new parameter instead of failing BEHAVIOURALLY. Through
    ``evaluate_production_semantics`` the same call is legal on both trees and the base
    tree answers with the finding.
    """

    report = sp.evaluate_production_semantics(payload, paper_id="C-090", mode="strict")
    return report.checks[_s.CHECK_ACTOR_EVIDENCE]


def _check_without_entities(payload: Dict[str, Any]) -> Any:
    """The BASE rule, reproduced exactly: the entity table withheld, as at 736c1a2."""

    return sp._check_actor_evidence(_s._processes(payload))


def _entities_named(result: Any) -> List[str]:
    return sorted(str(f.get("entity")) for f in result.findings)


# ---------------------------------------------------------------------------
# test_a -- G9 CORRECTION. Fails behaviourally on base 736c1a2.
# ---------------------------------------------------------------------------
def test_a_one_component_wrapper_named_by_its_component_stops_firing() -> None:
    """442 / CYP51A1: the wrapper's ONE component is named verbatim in the row's own span,
    so the actor is corroborated and NO finding is raised.

    At base the function is passed only ``processes``, cannot reach the component list, and
    reports ``'Lanosterol 14-alpha demethylase' ... does not name it``. That is the
    behavioural base failure this test exists to produce.
    """

    payload = _payload(
        [WRAPPER_442],
        [_reaction(
            "CYP51A1 reaction: lansterol -> 4,4-dimethylcholesta-8(9),14,24-trien-3b-ol",
            "Lanosterol 14-alpha demethylase", SPAN_A,
        )],
    )

    # NON-VACUITY, three ways: the row IS examined, the base lexical rule DOES reject it,
    # and the base call DOES raise the finding. Without all three a pass here would mean
    # only that nothing was looked at.
    assert sp._actor_named_in_span("Lanosterol 14-alpha demethylase", [SPAN_A]) is False
    base = _check_without_entities(payload)
    assert len(base.findings) == 1, base.summary
    assert _entities_named(base) == ["Lanosterol 14-alpha demethylase"]
    assert "1 examined" in base.summary

    result = _check(payload)
    assert result.findings == [], result.findings
    assert result.ok is True
    # The census is NOT moved by the rescue: the row was examined either way.
    assert result.summary.endswith("(1 examined, 0 carried no comparable name or span)")
    assert result.inapplicable_reason == ""


# ---------------------------------------------------------------------------
# test_b -- ANTI-WIDENING. The safety property. NEW ACCEPTANCE.
# ---------------------------------------------------------------------------
def test_b_multi_component_wrapper_keeps_firing_anti_widening() -> None:
    """3623 ``enterobactin synthase`` has FOUR components and its span names one of them.

    A rule that matched any component of any wrapper would rescue F-116's superset. This is
    the automatic-reject case, so it is asserted from both ends: the finding survives, AND
    the wrapper index refuses to give 3623 any symbols at all.
    """

    payload = _payload(
        [WRAPPER_3623],
        [_reaction("Activation of DHB to DHB-AMP", "enterobactin synthase", SPAN_B)],
    )

    # NON-VACUITY: the span really does name EntE on whole-token boundaries, so a widened
    # rule WOULD have rescued this row. The test is not passing because the shape is inert.
    assert sp._component_named_in_span(["ente"], [SPAN_B]) == "ente"
    assert len(WRAPPER_3623["components"]) == 4

    index = sp._sole_component_symbols(_s._entities(payload))
    assert index["enterobactin synthase"] == (), index

    result = _check(payload)
    assert _entities_named(result) == ["enterobactin synthase"], result.findings
    assert result.ok is False
    # Byte-identical to the base rule's verdict on the same payload.
    assert result.to_dict() == _check_without_entities(payload).to_dict()


# ---------------------------------------------------------------------------
# test_c -- NEW ACCEPTANCE.
# ---------------------------------------------------------------------------
def test_c_one_component_wrapper_whose_span_omits_the_component_keeps_firing() -> None:
    """One component, but the row's own span never names it. Still a finding.

    The rescue is EXACT IDENTITY against the cited span, not a licence for any actor that
    happens to be a wrapper.
    """

    span = "this step is carried out by a microsomal cytochrome P450 enzyme"
    payload = _payload(
        [WRAPPER_442],
        [_reaction("demethylation", "Lanosterol 14-alpha demethylase", span)],
    )

    # NON-VACUITY: the row reaches the finding branch (the base rule rejects it), the
    # wrapper IS recognised and DOES carry the symbol -- only the span fails.
    assert sp._actor_named_in_span("Lanosterol 14-alpha demethylase", [span]) is False
    index = sp._sole_component_symbols(_s._entities(payload))
    assert index["lanosterol 14 alpha demethylase"] == ("cyp51a1",), index
    assert "cyp51a1" not in _s.normalize_name(span)

    result = _check(payload)
    assert _entities_named(result) == ["Lanosterol 14-alpha demethylase"], result.findings
    assert result.ok is False


# ---------------------------------------------------------------------------
# test_d -- NEW ACCEPTANCE.
# ---------------------------------------------------------------------------
def test_d_bare_non_wrapper_actor_keeps_firing() -> None:
    """``EntE`` cited against a span that does not name it, with NO wrapper anywhere.

    The corpus class C, and F-079's original shape. An empty entity table must leave the
    check exactly as it was.
    """

    span = "Enterobactin is produced and secreted by a TolC-dependent process"
    payload = _payload([], [_reaction("enterobactin export", "EntE", span)])

    # NON-VACUITY: there is nothing to rescue with, and the base rule rejects the row.
    assert sp._sole_component_symbols(_s._entities(payload)) == {}
    assert sp._actor_named_in_span("EntE", [span]) is False

    result = _check(payload)
    assert _entities_named(result) == ["EntE"], result.findings
    assert result.to_dict() == _check_without_entities(payload).to_dict()


# ---------------------------------------------------------------------------
# test_e -- NEW ACCEPTANCE. The census and the anti-vacuity path.
# ---------------------------------------------------------------------------
def test_e_census_and_not_evaluated_path_are_unchanged() -> None:
    """``not_examined`` and the NOT-EVALUATED disposition are untouched by the wrapper arm.

    A wrapper actor that cites NO span is still NOT EXAMINED -- never rescued into a pass,
    which would let an actor evade the gate by deleting its evidence, and never failed.
    """

    # (1) A wrapper whose component would match, but the actor cites nothing at all.
    no_span = _payload(
        [WRAPPER_442],
        [{"name": "CYP51A1 reaction",
          "enzymes": [{"entity": "Lanosterol 14-alpha demethylase", "role": "catalyst"}]}],
    )
    result = _check(no_span)
    assert result.findings == []
    assert result.inapplicable_reason == sp.NO_ACTOR_SPANS
    assert result.summary.startswith("not evaluated:")
    assert result.to_dict() == _check_without_entities(no_span).to_dict()

    # (2) A mixed payload: one examined-and-rescued row beside one not-examined row.
    mixed = _payload(
        [WRAPPER_442],
        [
            _reaction("CYP51A1 reaction", "Lanosterol 14-alpha demethylase", SPAN_A),
            {"name": "unsourced", "enzymes": ["a bare string actor"]},
        ],
    )
    mixed_result = _check(mixed)
    base_mixed = _check_without_entities(mixed)
    assert mixed_result.summary.endswith("(1 examined, 1 carried no comparable name or span)")
    # NON-VACUITY: the census the tip reports is the census the BASE reported; only the
    # finding count moved.
    assert base_mixed.summary.endswith("(1 examined, 1 carried no comparable name or span)")
    assert len(base_mixed.findings) == 1 and mixed_result.findings == []


# ---------------------------------------------------------------------------
# test_f -- NEW ACCEPTANCE. The two narrowness guards that are not the component count.
# ---------------------------------------------------------------------------
def test_f_placeholder_and_ambiguous_wrappers_are_refused() -> None:
    """A ``name: "Unknown"`` component is not a symbol, and a name carried twice is not an
    identity. Both keep firing.
    """

    # (1) PathBank's placeholder component, measured on PMC12444477/strict's LpxL row.
    placeholder = {
        "name": "LpxL",
        "components": [{"name": "Unknown", "stoichiometry": 1, "pathbank_protein_id": 9659,
                        "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 9659}}],
    }
    span = "the acyltransferase step proceeds by an unknown mechanism"
    payload = _payload([placeholder], [_reaction("acylation", "LpxL", span)])
    # NON-VACUITY: the span really does carry the token "unknown", so only the guard
    # stops the rescue.
    assert "unknown" in _s.normalize_name(span).split(" ")
    assert sp._sole_component_symbols(_s._entities(payload))["lpxl"] == ()
    assert _entities_named(_check(payload)) == ["LpxL"]

    # (2) Two rows claiming the same name: two candidate identities, so neither is used.
    twin = deepcopy(WRAPPER_442)
    twin["components"] = [{"name": "SQLE", "gene_name": "SQLE"}]
    ambiguous = _payload(
        [WRAPPER_442, twin],
        [_reaction("CYP51A1 reaction", "Lanosterol 14-alpha demethylase", SPAN_A)],
    )
    # NON-VACUITY: with ONLY the genuine row the same payload IS rescued.
    assert _check(_payload([WRAPPER_442], _s._processes(ambiguous)["reactions"])).findings == []
    assert sp._sole_component_symbols(_s._entities(ambiguous)) == {}
    assert _entities_named(_check(ambiguous)) == ["Lanosterol 14-alpha demethylase"]


# ---------------------------------------------------------------------------
# test_g -- the MEASURED corpus classification, reproduced against committed legs.
# ---------------------------------------------------------------------------
def test_g_t106_corpus_classification_a2_b3_c4() -> None:
    """Nine actor-evidence findings exist across T-106. Exactly TWO move, and they are the
    CYP51A1 pair. The three ``enterobactin synthase`` findings -- F-116's superset -- and
    the four class-C findings all keep firing.
    """

    legs = sorted(CORPUS.glob("*/*/final_mapped.json"))
    assert len(legs) >= 10, legs

    base_pointers: List[str] = []
    tip_pointers: List[str] = []
    base_entities: List[str] = []
    for leg in legs:
        payload = json.loads(leg.read_text(encoding="utf-8"))
        rel = leg.relative_to(CORPUS).as_posix()
        for result, sink in ((_check_without_entities(payload), base_pointers),
                             (_check(payload), tip_pointers)):
            for finding in result.findings:
                sink.append(f"{rel}{finding['pointer']}")
        for finding in _check_without_entities(payload).findings:
            base_entities.append(str(finding.get("entity")))

    # The measured corpus total. NON-VACUITY for the whole test: if the corpus stopped
    # producing findings this is where it shows.
    assert len(base_pointers) == 9, sorted(base_pointers)
    assert sorted(base_entities) == sorted([
        "Lanosterol 14-alpha demethylase", "Lanosterol 14-alpha demethylase",   # A
        "enterobactin synthase", "enterobactin synthase", "enterobactin synthase",  # B
        "EntE", "EntE", "LSS", "LSS",                                            # C
    ])

    # A = 2, and they are the two CYP51A1 rows of PMC12782028/strict.
    moved = sorted(set(base_pointers) - set(tip_pointers))
    assert moved == [
        "PMC12782028/strict/final_mapped.json/processes/reactions/1/enzymes/0",
        "PMC12782028/strict/final_mapped.json/processes/reactions/1/modifiers/0",
    ], moved

    # B + C = 7 keep firing, and nothing NEW was raised.
    assert len(tip_pointers) == 7, sorted(tip_pointers)
    assert not set(tip_pointers) - set(base_pointers)

    # B = 3, named explicitly: the EntE superset stays demoted.
    survivors_b = [
        p for p in tip_pointers
        if p.startswith(("PMC12096016/strict", "PMC12452463/strict"))
    ]
    assert len(survivors_b) == 3, survivors_b


# ---------------------------------------------------------------------------
# test_h -- the recorded T-106 disposition of the class-A leg does not move.
# ---------------------------------------------------------------------------
def test_h_pmc12782028_strict_stays_blocked_on_core_coverage() -> None:
    """The leg whose findings flip must NOT become releasable.

    `PMC12782028/strict` is blocked on `requested_core_coverage_below_minimum:0.222<0.500`,
    which the technical chain applies BEFORE the semantic cap. Flipping its actor-evidence
    verdict from failed to passed removes one recorded reason and changes nothing else.
    """

    coverage = json.loads(
        (CORPUS / LEG_A[0] / LEG_A[1] / "coverage_summary.json").read_text(encoding="utf-8")
    )
    blocker = "requested_core_coverage_below_minimum:0.222<0.500"
    assert blocker in coverage["reasons"], coverage["reasons"]

    payload = _leg(*LEG_A)
    report = sp.evaluate_production_semantics(
        payload,
        requested_pathway="cholesterol biosynthesis",
        requested_organism="Homo sapiens",
        paper_id=LEG_A[0], mode=LEG_A[1],
    )
    evaluation, _reason, failed, evaluability = semantic_verdict(report)

    # NON-VACUITY: the check under test really is a GATING check on this leg, and at base
    # it really did name itself as a failure.
    assert _s.CHECK_ACTOR_EVIDENCE in SEMANTIC_GATING_CHECKS
    assert _check_without_entities(payload).ok is False
    assert _s.CHECK_ACTOR_EVIDENCE not in failed, failed

    status = classify_release_status(
        coverage, pipeline_executed=True, strict_gates_passed=True,
        semantic_evaluation=evaluation, semantic_failed_checks=failed,
        semantic_check_evaluability=evaluability,
    )
    assert status.status == REVIEW_REQUIRED, status.to_dict()
    assert status.status != RELEASE_READY
    assert blocker in status.reasons, status.reasons

    # And the disposition is the COVERAGE branch, not the semantic cap: classifying the
    # same leg with the semantics forced to a clean pass still lands on review_required
    # with the same blocker. So the flip could not have moved this leg whatever it did.
    forced = classify_release_status(
        coverage, pipeline_executed=True, strict_gates_passed=True,
        semantic_evaluation="passed", semantic_failed_checks=(),
    )
    assert forced.status == REVIEW_REQUIRED, forced.to_dict()
    assert blocker in forced.reasons
    assert REASON_SEMANTIC_EVALUATION_FAILED not in forced.reasons

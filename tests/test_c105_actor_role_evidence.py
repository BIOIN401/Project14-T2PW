"""C-105 -- an audit patch may not invent an actor role it has no evidence for.

The defect these tests pin is measured, not hypothetical:
``runs_verify/2026-08-28_1816/papers/PMC13231680/research/final_mapped.json``
``/processes/reactions/0`` was frozen naming one protein as both the reaction's
enzyme and its inhibitor, and the audit round that wrote it recorded why -- to
"resolve the structural inconsistency where an inhibitor is listed without a
target enzyme". PRODUCT_CONTRACT section 1 forbids inventing an enzyme, and a
schema complaint is not evidence.

**Every test here drives the public seam ``apply_patch_with_policy`` and asserts on
behaviour and on literal reason text -- no new symbol is imported.** That is
deliberate (G9): at base ``36f773c`` these tests RUN and FAIL on what the function
does, rather than erroring on a name that does not exist yet.

The fixture is generic on purpose. ``P``, ``A`` and ``B`` carry no paper, no
protein and no gold value: the rule under test is about the shape of a patch.

Sections 1-5 are the G9 behavioural proof and import no new symbol. Sections 6-8
were added by REV-105 correction round 1: section 6 is the preservation battery the
first draft lacked, and sections 7-8 import private symbols deliberately, so they
ERROR rather than fail at base -- they pin an invariant, they are not the base proof.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402


#: The greppable prefix the guard stamps on its rejections. Spelled literally
#: rather than imported so this file exercises behaviour at any SHA.
REASON_PREFIX = "unevidenced_actor_role"

#: A span that names P doing the thing the paper says it does -- being inhibited.
#: This is the span the real audit round argued from.
INHIBITOR_SPAN = "A significantly inhibited P activity in the assay"

#: The reaction's own evidence. It names the chemistry and NO actor at all, which
#: is what made the reaction look structurally incomplete in the first place.
REACTION_SPAN = "A is decomposed in the intestine, resulting in an antibacterial effect"

#: The rationale the audit stage actually gave, generalised off the protein name.
STRUCTURAL_RATIONALE = (
    "add P as an enzyme to the decomposition reaction to resolve the structural "
    "inconsistency where an inhibitor is listed without a target enzyme"
)

#: A real catalysis claim: P is named performing the catalytic role, on this
#: reaction's own chemistry.
CATALYSIS_SPAN = "P catalyses the conversion of A to B under physiological conditions"


def _payload() -> Dict[str, Any]:
    """One reaction whose only actor is an inhibitor, exactly as the card specifies."""

    return {
        "entities": {
            "compounds": [{"name": "A"}, {"name": "B"}],
            "proteins": [{"name": "P"}],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "A decomposition to B",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "enzymes": [],
                    "modifiers": [
                        {"entity": "P", "role": "inhibitor", "evidence": INHIBITOR_SPAN}
                    ],
                    "evidence": REACTION_SPAN,
                }
            ]
        },
    }


def _enzyme_add(evidence: Any) -> Dict[str, Any]:
    """The op shape the audit stage emitted: a bare-string enzyme at full confidence."""

    op: Dict[str, Any] = {
        "op": "add",
        "path": "/processes/reactions/0/enzymes/-",
        "value": "P",
        "confidence": 1.0,
    }
    if evidence is not None:
        op["evidence"] = evidence
    return op


def _reaction(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload["processes"]["reactions"][0]


# ---------------------------------------------------------------------------
# 1. The rejection case -- the measured defect. FAILS at base 36f773c.
# ---------------------------------------------------------------------------


def test_unevidenced_enzyme_add_is_rejected(tmp_path: Path) -> None:
    """A structural-consistency argument does not license inventing a catalyst.

    This is T-107's Priority-2 failure in miniature. At base the op clears on
    ``confidence >= 0.75`` alone and ``enzymes`` becomes ``["P"]``.
    """

    payload = _payload()
    before = copy.deepcopy(payload)
    rejected_log = tmp_path / "rejected_patch_log.json"

    result, report = apply_patch_with_policy(
        payload,
        [_enzyme_add(STRUCTURAL_RATIONALE)],
        stage="audit",
        rejected_log_path=rejected_log,
    )

    assert report["summary"]["accepted_count"] == 0, report
    assert report["summary"]["rejected_count"] == 1, report
    assert _reaction(result)["enzymes"] == [], _reaction(result)

    reason = report["rejected"][0]["reason"]
    assert reason.startswith(REASON_PREFIX), reason
    assert "catalysis" in reason, reason

    # Merge rule 7: the row survives, reviewable, exactly as it was. The guard
    # refuses the patch; it never deletes a reaction or strips the modifier.
    assert result["processes"] == before["processes"], result["processes"]
    assert result == before

    # Recorded, not silently dropped -- through the existing rejected-log channel.
    assert [entry["reason"] for entry in report["rejected_patch_log"]] == [reason]
    logged = json.loads(rejected_log.read_text(encoding="utf-8"))
    assert any(str(entry.get("reason", "")).startswith(REASON_PREFIX) for entry in logged), logged


# ---------------------------------------------------------------------------
# 2. The preservation case -- mandatory. A guard that rejects both is a defect.
# ---------------------------------------------------------------------------


def test_evidenced_enzyme_add_is_still_accepted() -> None:
    """The same patch, carrying a span that names P as the catalyst, is applied.

    PRODUCT_CONTRACT section 2 keeps depth recoverable through bounded repair;
    section 1 forbids a technically recoverable problem from producing no PWML.
    A guard that refused this would be a new defect, not a fix.
    """

    payload = _payload()

    result, report = apply_patch_with_policy(
        payload, [_enzyme_add(CATALYSIS_SPAN)], stage="audit"
    )

    assert report["summary"]["accepted_count"] == 1, report
    assert report["summary"]["rejected_count"] == 0, report
    assert report["rejected"] == []
    assert _reaction(result)["enzymes"] == ["P"]
    assert report["transaction"]["committed"] is True


# ---------------------------------------------------------------------------
# 3. Presence is not evidence. This is the case a reviewer will construct.
# ---------------------------------------------------------------------------


def test_actor_already_in_the_reaction_is_not_evidence_for_a_new_role(tmp_path: Path) -> None:
    """P is already an actor of this very reaction -- and that licenses nothing.

    The measured defect IS a protein legitimately present as an inhibitor being
    promoted to catalyst on the same reaction. A guard that read "the entity is
    already here" as corroboration would accept the exact patch this card exists
    to reject. The op below offers, as its evidence, the inhibitor row's own span
    -- which does name P, by name, on this reaction. It is still refused, because
    naming an actor is not naming it performing the role being added.
    """

    payload = _payload()
    assert _reaction(payload)["modifiers"][0]["entity"] == "P", "fixture precondition"

    result, report = apply_patch_with_policy(
        payload, [_enzyme_add(INHIBITOR_SPAN)], stage="audit"
    )

    assert report["summary"]["accepted_count"] == 0, report
    assert _reaction(result)["enzymes"] == []
    assert report["rejected"][0]["reason"].startswith(REASON_PREFIX)

    # And the same span DOES license the role it actually attests, so the refusal
    # above is about the role, not about the span being unusable.
    payload2 = _payload()
    payload2["processes"]["reactions"][0]["modifiers"] = []
    result2, report2 = apply_patch_with_policy(
        payload2,
        [
            {
                "op": "add",
                "path": "/processes/reactions/0/modifiers/-",
                "value": {"entity": "P", "role": "inhibitor", "evidence": INHIBITOR_SPAN},
                "confidence": 0.9,
            }
        ],
        stage="audit",
    )
    assert report2["summary"]["accepted_count"] == 1, report2
    assert _reaction(result2)["modifiers"][0]["role"] == "inhibitor"


# ---------------------------------------------------------------------------
# 4. Shape coverage -- the guard is scoped, and it fails closed.
# ---------------------------------------------------------------------------


def test_enzyme_add_with_no_evidence_field_at_all_is_rejected() -> None:
    payload = _payload()
    result, report = apply_patch_with_policy(payload, [_enzyme_add(None)], stage="audit")

    reason = report["rejected"][0]["reason"]
    assert reason.startswith(REASON_PREFIX), reason
    assert "no evidence span" in reason, reason
    assert _reaction(result)["enzymes"] == []


@pytest.mark.parametrize(
    "path",
    [
        "/processes/reactions/0/modifiers/-",
        "/processes/reactions/0/modifiers_or_enzymes/-",
        "/processes/reactions/0/catalysts/-",
    ],
)
def test_every_actor_container_on_a_reaction_is_guarded(path: str) -> None:
    payload = _payload()
    op = {
        "op": "add",
        "path": path,
        "value": {"entity": "P", "role": "catalyst"},
        "confidence": 1.0,
        "evidence": STRUCTURAL_RATIONALE,
    }
    _result, report = apply_patch_with_policy(payload, [op], stage="audit")
    assert report["summary"]["accepted_count"] == 0, (path, report)
    assert report["rejected"][0]["reason"].startswith(REASON_PREFIX)


def test_transport_actor_add_is_guarded_and_a_transport_span_licenses_it() -> None:
    payload = {
        "entities": {"compounds": [{"name": "A"}], "proteins": [{"name": "P"}]},
        "processes": {
            "transports": [
                {"name": "A import", "cargo": "A", "transporters": [], "evidence": REACTION_SPAN}
            ]
        },
    }
    op = {
        "op": "add",
        "path": "/processes/transports/0/transporters/-",
        "value": "P",
        "confidence": 1.0,
        "evidence": "P is required for the reaction to proceed",
    }
    _result, report = apply_patch_with_policy(copy.deepcopy(payload), [op], stage="audit")
    assert report["summary"]["accepted_count"] == 0, report
    assert report["rejected"][0]["reason"].startswith(REASON_PREFIX)

    licensed = dict(op, evidence="P transports A across the inner membrane")
    result2, report2 = apply_patch_with_policy(copy.deepcopy(payload), [licensed], stage="audit")
    assert report2["summary"]["accepted_count"] == 1, report2
    assert result2["processes"]["transports"][0]["transporters"] == ["P"]


# ---------------------------------------------------------------------------
# 5. The guard stays inside its boundary -- no collateral tightening.
# ---------------------------------------------------------------------------


def test_guard_does_not_touch_non_actor_paths_removes_or_clears() -> None:
    """Three ops the guard must leave exactly as it found them.

    An entity add is not an actor role; a remove is judged by the core-semantics
    guard that already exists; and emptying a role introduces no actor, so there
    is nothing to license.
    """

    payload = _payload()
    _result, report = apply_patch_with_policy(
        payload,
        [
            {
                "op": "add",
                "path": "/entities/compounds/-",
                "value": {"name": "C"},
                "confidence": 0.9,
                "evidence": "no actor named anywhere in this string",
            },
            {
                "op": "replace",
                "path": "/processes/reactions/0/enzymes",
                "value": [],
                "confidence": 0.9,
                "evidence": "no actor named anywhere in this string",
            },
        ],
        stage="audit",
    )
    reasons = [entry["reason"] for entry in report["rejected"]]
    assert not any(str(reason).startswith(REASON_PREFIX) for reason in reasons), reasons


def test_confidence_rejection_keeps_its_own_reason() -> None:
    """A patch below the confidence bar is still reported as below the bar.

    The guard runs last precisely so reports that already grep the existing reason
    strings keep reading what they always read.
    """

    payload = _payload()
    op = dict(_enzyme_add(STRUCTURAL_RATIONALE), confidence=0.10)
    _result, report = apply_patch_with_policy(payload, [op], stage="audit")
    assert report["rejected"][0]["reason"].startswith("Confidence "), report


# ---------------------------------------------------------------------------
# 6. REV-105 / B1. The preservation battery the first draft did not have.
#
# The original preservation control used the one-character name "P" beside the
# word "catalyses" -- the single shape a whole-name rule always handles, which is
# exactly why it passed while the class it stood for was broken. Every case below
# is a shape that whole-name matching refuses and the repository's calibrated rule
# licenses. Four are lifted from committed payloads or from
# bench.semantic_production._actor_named_in_span's own docstring.
# ---------------------------------------------------------------------------


def _enzyme_add_named(name: str, evidence: str) -> Dict[str, Any]:
    return {
        "op": "add",
        "path": "/processes/reactions/0/enzymes/-",
        "value": name,
        "confidence": 1.0,
        "evidence": evidence,
    }


#: (label, registry-spelled actor name, evidence span). Each MUST be accepted.
LICENSED_CASES = [
    # A wrapper name whose span uses the bare paper symbol. This is the example
    # _actor_named_in_span's docstring gives, and the rule it was calibrated on.
    ("wrapper name, symbol span", "ALAS2 complex",
     "ALAS2 mediates the condensation of glycine and succinyl-CoA"),
    ("wrapper name, symbol span 2", "MenD complex",
     "MenD catalyses the first irreversible step of menaquinone biosynthesis"),
    # A registry-spelled multi-token name whose span names it beside the symbol.
    ("multi-token registry name", "UDP-N-acetylglucosamine acyltransferase",
     "LpxA, the UDP-N-acetylglucosamine acyltransferase, catalyzes the reversible acylation"),
    ("DB name in parentheses", "oxidoreductase (entA)",
     "EntA (2,3-dihydro-2,3-dihydroxybenzoate dehydrogenase) oxidises the substrate"),
    # A hyphenated adjectival cue. Folding must map the separator to a space, or
    # "X-catalyzed" welds into one token and both name and cue vanish.
    ("hyphenated adjectival cue", "NDM-1",
     "NDM-1-catalyzed hydrolysis of the beta-lactam ring"),
    # Passive voice with a named agent -- the paper's usual construction.
    ("passive with agent", "NDM-1",
     "the beta-lactam ring is converted to the open form by NDM-1"),
    ("passive with agent 2", "Serine hydroxymethyltransferase, mitochondrial",
     "the reaction is catalyzed by serine hydroxymethyltransferase"),
    # Periphrastic constructions that carry no single catalysis verb.
    ("is the enzyme responsible for", "LpxC",
     "LpxC is the enzyme responsible for the deacetylation step"),
    ("breaks down", "PptA", "PptA breaks down the phosphonate backbone"),
    ("acts on", "EntB", "EntB acts on the aryl carrier domain"),
    # Both lifted from committed runs_verify payloads, and both were refused by the
    # first widening: the agent trails the product list by more than the original
    # 40-character bound, and "isochorismatase" is not an EC stem anyone enumerates.
    ("passive, agent after the products", "EntB",
     "isochorismate is converted to 2,3-dihydro-2,3-dihydroxybenzoate and pyruvate "
     "by EntB isochorismatase activity"),
    ("group transfer named as residues", "KDO transferase",
     "WaaA adds a number of Kdo residues"),
    # REV-105 finding 4. These three have fewer than three characters before
    # "ase", so the generic enzyme-noun rule cannot reach them. "lyase" was an
    # explicit stem at this card's first commit and licensing it is a REGRESSION
    # this round restores, not a new capability.
    ("short enzyme noun, lyase", "P", "P is the lyase for this step"),
    ("short enzyme noun, DNase", "EndA", "EndA is the DNase for this step"),
    ("short enzyme noun, RNase", "Rne", "Rne is the RNase for this step"),
]


@pytest.mark.parametrize(
    "label,name,evidence", LICENSED_CASES, ids=[case[0] for case in LICENSED_CASES]
)
def test_evidenced_actor_adds_survive_the_guard(label: str, name: str, evidence: str) -> None:
    payload = _payload()
    payload["entities"]["proteins"] = [{"name": name}]
    result, report = apply_patch_with_policy(
        payload, [_enzyme_add_named(name, evidence)], stage="audit"
    )
    assert report["summary"]["accepted_count"] == 1, (label, report["rejected"])
    assert _reaction(result)["enzymes"] == [name], label


#: Shapes that must STILL be refused after the widening. The first two are the
#: measured defect; the last two are the lexical traps the widening could have
#: opened.
REFUSED_CASES = [
    ("the structural-consistency rationale", "NDM-1", STRUCTURAL_RATIONALE.replace("P ", "NDM-1 ")),
    ("the inhibitor span it argued from", "NDM-1",
     "PSA significantly inhibited NDM-1 enzyme activity"),
    # F-079: EntE is a SUBSTRING of enterobactin. Whole-token boundaries only.
    ("substring of a compound name", "EntE",
     "Enterobactin is produced by a TolC-dependent process"),
    # "disease", "increase", "release" and "database" all end in "ase"; the
    # enzyme-noun cue is an allowlist of EC stems, not a bare -ase pattern.
    ("-ase words that are not enzymes", "LpxA",
     "LpxA was noted while the incidence of disease did not increase after release of the database"),
    # REV-105 finding 1. "mediat" must stay a catalysis cue for the ALAS2 case, and
    # that same cue makes an INHIBITION sentence read as catalysis unless the
    # contra-cue check refuses the window. Both of these were ACCEPTED before it.
    ("inhibition paraphrased as mediation", "NDM-1",
     "PSA-mediated inhibition of NDM-1 activity"),
    ("inhibition paraphrased, passive", "NDM-1",
     "the inhibition of NDM-1 is mediated by PSA"),
]


@pytest.mark.parametrize(
    "label,name,evidence", REFUSED_CASES, ids=[case[0] for case in REFUSED_CASES]
)
def test_unlicensed_actor_adds_are_still_refused(label: str, name: str, evidence: str) -> None:
    payload = _payload()
    payload["entities"]["proteins"] = [{"name": name}]
    result, report = apply_patch_with_policy(
        payload, [_enzyme_add_named(name, evidence)], stage="audit"
    )
    assert report["summary"]["accepted_count"] == 0, (label, report)
    assert report["rejected"][0]["reason"].startswith(REASON_PREFIX), label
    assert _reaction(result)["enzymes"] == [], label


def test_inhibition_cue_in_the_same_window_cannot_license_catalysis() -> None:
    """REV-105 finding 1: the paraphrase route back into the defect, closed.

    All three outcomes are asserted together because the fix is only correct as a
    set. ``mediat`` cannot simply be deleted -- the third case is
    _actor_named_in_span's own docstring example and a legitimate repair -- so the
    catalysis family instead refuses a window that also carries an inhibition cue.
    The first two spans say the protein is INHIBITED; before this check each of
    them licensed it as the reaction's CATALYST, which is the exact promotion this
    card exists to prevent, reachable by one rephrase of a rationale the audit
    stage regenerates every round.
    """

    payload_for = lambda name: dict(_payload(), entities={
        "compounds": [{"name": "A"}, {"name": "B"}],
        "proteins": [{"name": name}],
        "protein_complexes": [], "nucleic_acids": [],
    })

    for span in ("PSA-mediated inhibition of NDM-1 activity",
                 "the inhibition of NDM-1 is mediated by PSA"):
        _result, report = apply_patch_with_policy(
            payload_for("NDM-1"), [_enzyme_add_named("NDM-1", span)], stage="audit"
        )
        assert report["summary"]["accepted_count"] == 0, (span, report)
        assert report["rejected"][0]["reason"].startswith(REASON_PREFIX), span

    licensed = "ALAS2 mediates the condensation of glycine and succinyl-CoA"
    result, report = apply_patch_with_policy(
        payload_for("ALAS2 complex"),
        [_enzyme_add_named("ALAS2 complex", licensed)],
        stage="audit",
    )
    assert report["summary"]["accepted_count"] == 1, report["rejected"]
    assert _reaction(result)["enzymes"] == ["ALAS2 complex"]


# ---------------------------------------------------------------------------
# 7. The naming half must not drift from the rule it claims to reproduce.
# ---------------------------------------------------------------------------


def test_naming_rule_reproduces_the_calibrated_bench_rule() -> None:
    """The guard's naming half equals bench.semantic_production._actor_named_in_span.

    The guard cannot import that module -- t2pw.bench is the evaluation layer and
    t2pw.curation must not depend on it -- so the rule is duplicated. A test may
    import both, and this one pins them together: wherever the bench rule returns a
    verdict, the local token rule must return the same one. Without this the two
    copies drift and the docstring's claim of agreement rots, which is exactly what
    went wrong in the first draft of this guard.

    WHAT THIS PIN DOES NOT REACH, measured by REV-105: it binds
    ``_identifying_match_tokens`` and ``_match_fold``, the two helpers, and never
    calls ``_span_licenses_actor``. A regression introduced in the CONSUMER -- a
    substring comparison replacing the whole-token one inside the scan loop --
    would leave this test green. The F-079 refusal case
    ("EntE" inside "Enterobactin") is what catches that, not this. Do not read a
    green pin as proof the guard as a whole still matches the bench rule.
    """

    from t2pw.bench.semantic_production import _actor_named_in_span
    from t2pw.curation.apply_audit_patch import _identifying_match_tokens, _match_fold

    def names_locally(name: str, span: str) -> bool:
        wanted = set(_identifying_match_tokens(name))
        seen = {tok for tok in _match_fold(span).split(" ") if tok}
        return bool(wanted & seen)

    compared = 0
    for _label, name, span in LICENSED_CASES + REFUSED_CASES:
        verdict = _actor_named_in_span(name, [span])
        if verdict is None:
            continue
        compared += 1
        assert names_locally(name, span) is verdict, (name, span, verdict)
    assert compared >= 12, f"only {compared} cases carried a bench verdict"


def test_guard_is_stricter_than_naming_alone_and_that_is_the_point() -> None:
    """Naming is necessary, not sufficient -- the difference the defect turns on.

    The bench rule says the defect's spans DO name the protein, and it is right:
    they do. The guard still refuses them, because neither span names it performing
    the role being added. Asserting this keeps the two rules' relationship explicit
    rather than letting a future reader read "reproduces the bench rule" as "equals
    the bench rule".
    """

    from t2pw.bench.semantic_production import _actor_named_in_span

    span = "PSA significantly inhibited NDM-1 enzyme activity"
    assert _actor_named_in_span("NDM-1", [span]) is True

    payload = _payload()
    payload["entities"]["proteins"] = [{"name": "NDM-1"}]
    _result, report = apply_patch_with_policy(
        payload, [_enzyme_add_named("NDM-1", span)], stage="audit"
    )
    assert report["summary"]["accepted_count"] == 0, report
    assert "catalysis" in report["rejected"][0]["reason"]


# ---------------------------------------------------------------------------
# 8. C5 -- a rename is not an actor introduction and is not guarded here.
# ---------------------------------------------------------------------------


def test_renaming_an_existing_actor_row_is_not_guarded() -> None:
    """`replace .../modifiers/0/entity` swaps an identity inside an existing row.

    Out of scope by decision, not by oversight: pathway_curator's first documented
    job is repairing entity name mismatches, and guarding a rename here would block
    it. Recorded as residual route 3 in the module comment.
    """

    payload = _payload()
    op = {
        "op": "replace",
        "path": "/processes/reactions/0/modifiers/0/entity",
        "value": "P (corrected spelling)",
        "confidence": 0.9,
        "evidence": "the registry spells this protein differently",
    }
    _result, report = apply_patch_with_policy(payload, [op], stage="audit")
    reasons = [str(entry["reason"]) for entry in report["rejected"]]
    assert not any(reason.startswith(REASON_PREFIX) for reason in reasons), reasons

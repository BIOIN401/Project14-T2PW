"""C-107 -- the actor-evidence guard, calibrated in BOTH directions.

C-105 shipped the guard that stops an audit patch inventing an actor role. This
card corrects six measured miscalibrations in it. Two of them make the guard
refuse MORE (1a, 1f); four make it accept MORE (1b, 1c, 1d, 1e). Every test below
is a **correction of pre-existing observable behaviour**, not new capability, and
every one of the six carries a case that FAILS BEHAVIOURALLY at the C-106 base
``33a99e7`` and passes at this tip. Symbol absence is not the proof: sections 1-6
import nothing but the public seam ``apply_patch_with_policy``, so at base they
RUN and give the wrong answer rather than erroring on a name.

The base failures, measured (``evidence/c107_base_battery.log``):

===  ================================================================  ==========
1a   eleven near-synonyms admit the defect when a catalysis cue is in   11 of 11
     the window                                                         admitted
1b   the passive-with-agent cue licenses an actor who is not the agent  3 of 8
1c   a singular-only stoplist entry is bypassed by its own plural       8 of 14
1d   the transport family has no enzyme-family noun rule                2 of 7
1e   role ``cofactor`` refuses every cofactor span                      4 of 7
1f   ``mediat`` matches inside "intermediate"                           2 of 5
===  ================================================================  ==========

**THE PINNED SAFETY PROPERTY IS SECTION 0 AND IT COMES FIRST.** The F-146 patch
-- adding a protein as an ``enzyme`` on the rationale "to resolve the structural
inconsistency where an inhibitor is listed without a target enzyme" -- must stay
REJECTED. It is rejected at base and it must be rejected here; that pin is a
PRESERVATION control, not a base failure, and it is labelled as one. This card
widens the guard in four places and any of them re-admitting that patch would
make the card wrong however good the rest of it is.

The other direction is pinned just as hard. Section 1b's own preservation cases,
section 2's redox battery and section 4's real-enzyme cases exist because a guard
that refuses both directions is a new defect, not a fix -- which is exactly what
C-105's first draft was, at 12 of 29 legitimate cases and 258 of 692 corpus rows.

The fixture is generic. ``P``, ``Q``, ``A`` and ``B`` carry no paper and no
protein: what is under test is the shape of a span, not a paper's biology.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402

#: The greppable prefix the guard stamps on its rejections. Spelled literally
#: rather than imported, so this file exercises behaviour at any SHA.
REASON_PREFIX = "unevidenced_actor_role"


def _payload(name: str, container: str, bucket: str = "reactions") -> Dict[str, Any]:
    proc: Dict[str, Any] = {
        "name": "A to B",
        "inputs": ["A"],
        "outputs": ["B"],
        "evidence": "A is converted in the gut",
        container: [],
    }
    return {
        "entities": {
            "compounds": [{"name": "A"}, {"name": "B"}],
            "proteins": [{"name": name}],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "processes": {bucket: [proc]},
    }


def accepted(
    name: str,
    evidence: Optional[str],
    *,
    container: str = "enzymes",
    bucket: str = "reactions",
    role: Optional[str] = None,
) -> bool:
    """Drive ONE actor-role add through the real public seam. True == applied.

    Nothing private is imported and nothing is re-implemented: the verdict is the
    report the production entry point returns.
    """

    value: Any = name if role is None else {"entity": name, "role": role}
    payload = _payload(name, container, bucket)
    before = copy.deepcopy(payload)
    op: Dict[str, Any] = {
        "op": "add",
        "path": f"/processes/{bucket}/0/{container}/-",
        "value": value,
        "confidence": 1.0,
    }
    if evidence is not None:
        op["evidence"] = evidence
    result, report = apply_patch_with_policy(payload, [op], stage="audit")
    if report["summary"]["accepted_count"] == 1:
        return True
    # Merge rule 7: a refused patch leaves the payload exactly as it found it.
    assert result == before, result
    assert report["rejected"][0]["reason"].startswith(REASON_PREFIX), report["rejected"]
    return False


# ---------------------------------------------------------------------------
# 0. THE PINNED SAFETY PROPERTY -- a PRESERVATION control, green at base too.
# ---------------------------------------------------------------------------

#: The rationale the audit round actually gave, generalised off the protein name.
STRUCTURAL_RATIONALE = (
    "add P as an enzyme to the decomposition reaction to resolve the structural "
    "inconsistency where an inhibitor is listed without a target enzyme"
)


def test_the_structural_consistency_rationale_is_still_rejected() -> None:
    """F-146, pinned. Deliberately green at base -- it is what may not move.

    This card widens the guard in four places (1b, 1c, 1d, 1e). Every one of them
    is a route by which this patch could come back, so it is asserted here rather
    than only in C-105's file, and it was re-run after each of the six changes.
    """

    payload = _payload("P", "enzymes")
    payload["processes"]["reactions"][0]["modifiers"] = [
        {"entity": "P", "role": "inhibitor",
         "evidence": "A significantly inhibited P enzyme activity"}
    ]
    before = copy.deepcopy(payload)
    result, report = apply_patch_with_policy(
        payload,
        [{"op": "add", "path": "/processes/reactions/0/enzymes/-", "value": "P",
          "confidence": 1.0, "evidence": STRUCTURAL_RATIONALE}],
        stage="audit",
    )
    assert report["summary"]["accepted_count"] == 0, report
    assert result["processes"]["reactions"][0]["enzymes"] == []
    assert report["rejected"][0]["reason"].startswith(REASON_PREFIX)
    assert result == before


def test_the_inhibitor_span_it_argued_from_is_still_rejected() -> None:
    """The sentence the paper actually contains. Also green at base."""

    assert not accepted("P", "A significantly inhibited P enzyme activity")


# ---------------------------------------------------------------------------
# 1a. Inhibition near-synonyms. BASE FAILURE: 11 of 11 admitted.
# ---------------------------------------------------------------------------

#: The eleven REV-105 measured. Every one defeats the shipped contra-cue.
NEAR_SYNONYMS = [
    "blockade", "impairment", "disruption", "reduction", "loss", "silencing",
    "sequestration", "depletion", "ablation", "interference", "quenching",
]


#: The FOUR frames, not one. Correction round 1: the first version of this test
#: used frame A alone -- the frame C-107 section 1a quoted -- and a fix measured
#: against one frame closed one frame. REV-107 ran the other three and found 15
#: of 44 still open, all five of them stems that had gone in conditionally.
#: A route whose thesis is "one rephrase away" is not closed by a rule that is
#: itself one rephrase from defeat, so every synonym is now tested in all four.
_PAD_BEYOND_THE_OLD_GAP = "x" * 45

ATTENUATION_FRAMES = [
    ("A object present", "the {w} of P activity is mediated by Q"),
    ("B object absent", "the {w} of P is mediated by Q"),
    ("C object before the stem", "P activity showed {w} in the Q-mediated assay"),
    ("D object beyond 40 chars",
     "the {w} of P " + _PAD_BEYOND_THE_OLD_GAP + " enzymatic activity is mediated by Q"),
]


@pytest.mark.parametrize("word", NEAR_SYNONYMS)
@pytest.mark.parametrize(
    "frame,template", ATTENUATION_FRAMES, ids=[f[0] for f in ATTENUATION_FRAMES]
)
def test_each_inhibition_near_synonym_refuses_a_catalysis_promotion(
    frame: str, template: str, word: str
) -> None:
    """Each near-synonym, INDIVIDUALLY, in each of four frames.

    The shape matters and a word-level probe gets it wrong in both directions. In
    the bare frame "the <word> of P activity by Q" only "reduction" admits the
    defect, because the other ten carry no catalysis cue at all and the span is
    refused for having no cue rather than for having a contra. Put an actual
    catalysis cue in the window -- "is mediated by" -- and at the C-106 base all
    ELEVEN license the protein as the reaction's CATALYST off a sentence that
    says its activity was shut down. That is F-146 by paraphrase, and closing it
    is this card's `product_contract_violation`.

    What makes all four frames close together is that the contra is bound to THE
    ACTOR rather than to an adjacent activity noun: once the question is "is this
    protein what is being reduced", deleting the object, moving it in front of the
    stem, or pushing it past any character budget all stop mattering.
    """

    assert not accepted("P", template.format(w=word))


@pytest.mark.parametrize("word", NEAR_SYNONYMS)
def test_each_near_synonym_still_licenses_the_inhibitor_role_it_attests(word: str) -> None:
    """The same span DOES license what it actually says, for six of the eleven.

    Merge rule 7's direction: the six stems added to the inhibition family are
    there so an evidenced inhibitor row stops being refused, not only so a
    catalysis window is refused. The other five are matched only in the
    activity-directed phrase form, which is why this test asserts a floor rather
    than all eleven -- and the floor is what stops the "accept more" half of 1a
    being quietly dropped.
    """

    licensed = [
        w for w in NEAR_SYNONYMS
        if accepted("P", f"the {w} of P activity is mediated by Q",
                    container="modifiers", role="inhibitor")
    ]
    assert len(licensed) >= 6, licensed
    assert word in NEAR_SYNONYMS


#: Redox chemistry. "reduces|reducing|reduction of" is a CATALYSIS cue and must
#: stay one: deleting it to close the paraphrase above would break half of
#: enzymology to fix a rephrase. Every one of these is green at base and must
#: stay green -- this is the preservation half of 1a.
REDOX_SPANS = [
    "NADH-dependent reduction of the substrate by P",
    "P catalyses the reduction of the quinone to the quinol",
    "the reduction of A to B is carried out by P",
    "P reduces nitrite to nitric oxide",
    "reducing equivalents are transferred by P during the reduction of the disulfide",
    "P is the reductase for this step",
    "P catalyses the reduction of the disulfide bond of the substrate protein",
    "the two-electron reduction of the flavin is catalysed by P",
    # REV-107's B2 residual, correction round 1. The first version bound the
    # attenuation stem to an activity/level/expression noun ANYWHERE within 40
    # characters, and "level" and "function" then collided with real chemistry:
    # all three of these licensed at the C-106 base and were REFUSED at that tip.
    # Binding the stem to the ACTOR instead retires the collision, because a
    # substrate's level and an enzyme's function are not the actor's.
    "P catalyses the reduction of the substrate level in vitro",
    "P-dependent reduction of flavin is required for enzyme function",
]


@pytest.mark.parametrize("span", REDOX_SPANS)
def test_redox_catalysis_still_licenses(span: str) -> None:
    assert accepted("P", span), span


def test_redox_still_licenses_when_the_actor_is_not_the_thing_reduced() -> None:
    """The third B2-residual span, which needs an actor named in the span itself."""

    assert accepted(
        "ferrochelatase", "ferrochelatase reduces the cellular level of protoporphyrin"
    )
    assert accepted("ferrochelatase", "ferrochelatase reduces the substrate in this step")


def test_the_distinction_is_the_object_not_the_word() -> None:
    """One word, two objects, two verdicts -- stated as a single assertion.

    "reduction of the substrate" is chemistry; "reduction of P activity" is
    inhibition. The fix is a phrase rule, not a deletion, and this is the pair
    that says so.
    """

    assert accepted("P", "the reduction of the substrate is mediated by P")
    assert not accepted("P", "the reduction of P activity is mediated by Q")


# ---------------------------------------------------------------------------
# 1b. Passive with a named agent. BASE FAILURE: 3 of 8.
# ---------------------------------------------------------------------------

AGENT_IS_SOMEBODY_ELSE = [
    "A is converted to B by Q, and P was also detected in the assay",
    "A is produced by Q while P remained bound to the membrane",
    "B is formed by Q; P is unrelated to this step",
]

AGENT_IS_THE_ACTOR = [
    "A is converted to B by P in the intestine",
    "A is produced by P during the second step",
    "B is formed by the enzyme P",
    "A is converted to B by the purified recombinant P",
]


@pytest.mark.parametrize("span", AGENT_IS_SOMEBODY_ELSE)
def test_passive_with_agent_does_not_license_a_bystander(span: str) -> None:
    """The construction is evidence only when the agent after "by" IS the actor.

    At base the cue was "<passive verb> [^.]{0,80} by", which fired regardless of
    who followed "by" -- so any protein named anywhere in the 80-character window
    inherited somebody else's agency.
    """

    assert not accepted("P", span), span


@pytest.mark.parametrize("span", AGENT_IS_THE_ACTOR)
def test_passive_with_agent_still_licenses_the_real_agent(span: str) -> None:
    assert accepted("P", span), span


def test_passive_agent_survives_a_multi_token_registry_name() -> None:
    """Two shapes lifted from committed payloads, both of which must keep working.

    The first is the case a whole-name rule breaks; the second puts the agent
    more than 40 characters past the passive verb, behind a product list.
    """

    assert accepted(
        "Serine hydroxymethyltransferase, mitochondrial",
        "the reaction is catalyzed by serine hydroxymethyltransferase",
    )
    assert accepted(
        "EntB",
        "isochorismate is converted to 2,3-dihydro-2,3-dihydroxybenzoate and "
        "pyruvate by EntB isochorismatase activity",
    )


# ---------------------------------------------------------------------------
# 1c. The -ase stoplist. BASE FAILURE: 8 of 14.
# ---------------------------------------------------------------------------

PLURAL_BYPASSES = [
    "P appears in three purchases recorded in the supplement",
    "P appears beside two showcases in the exhibition",
    "P was photographed on the staircases of the institute",
    "P was left in one of the briefcases",
    "P pleases the reviewers of this manuscript",
]

NEWLY_LISTED_ENGLISH = [
    "P was found in the suitcases of the courier",
    "P was noted beside the grease on the bench",
    "P paraphrases the earlier report",
    "P creases the filter paper",
    "P appeases the reviewers",
]

REAL_ENZYME_NOUNS = [
    ("P", "P is the acyltransferase for this step"),
    ("P", "P and its paralogues are hydrolases of the same family"),
    ("P", "P belongs to the kinases described earlier"),
    ("P", "P is one of the phosphatases in this operon"),
    ("P", "P is the lyase for this step"),
    ("P", "P is the DNase for this step"),
    ("P", "P is the RNase for this step"),
]


@pytest.mark.parametrize("span", PLURAL_BYPASSES + NEWLY_LISTED_ENGLISH)
def test_ordinary_english_ase_words_do_not_license_an_enzyme(span: str) -> None:
    assert not accepted("P", span), span


@pytest.mark.parametrize("name,span", REAL_ENZYME_NOUNS)
def test_real_enzyme_family_nouns_still_license(name: str, span: str) -> None:
    """The other direction, and the one that matters more.

    C-105 round 1 refused 12 of 29 legitimate cases by inverting exactly this
    asymmetry. The stoplist stays a CLOSED list of English words; it did not
    become an allowlist of enzymes, and an unlisted enzyme noun still licenses.
    """

    assert accepted(name, span), span


def test_no_singular_only_stoplist_entry_can_be_bypassed_by_its_plural() -> None:
    """The GENERAL property, not the five measured instances.

    REV-105 named three plural bypasses and the Lead measured five. Listing five
    words would have fixed five words: the defect is that the exclusion matched
    the singular and then failed on the trailing "s", letting the plural through
    the generic ``[a-z]{3,}ases?`` shape. This walks EVERY entry that has no
    plural of its own listed and asserts none of them bypasses -- so the next
    entry someone adds cannot reintroduce it.

    It imports a private symbol on purpose. That makes it an INVARIANT pin, not
    part of the G9 base proof; the behavioural base failure for 1c is the
    parametrized case above.
    """

    from t2pw.curation.apply_audit_patch import _NON_ENZYME_ASE_WORDS

    listed = set(_NON_ENZYME_ASE_WORDS)
    singular_only = [
        w for w in sorted(listed)
        if not w.endswith("s") and w + "s" not in listed and w + "es" not in listed
    ]
    assert singular_only, "fixture precondition: the list still has singular-only entries"

    bypassed = [
        w + "s" for w in singular_only
        if accepted("P", f"P was mentioned near the word {w}s in the discussion")
    ]
    assert bypassed == [], bypassed


# ---------------------------------------------------------------------------
# 1d. The transport family. BASE FAILURE: 2 of 7.
# ---------------------------------------------------------------------------

TRANSPORTER_SPANS = [
    "P is the flippase for lipid A",
    "P is the translocase of the inner membrane",
    "P is the permease for this substrate",
    "P transports A across the inner membrane",
]


@pytest.mark.parametrize("span", TRANSPORTER_SPANS)
def test_a_transporter_named_by_its_family_noun_licenses(span: str) -> None:
    """"permease" was in the list as a one-off; the general rule was missing."""

    assert accepted("P", span, container="transporters", bucket="transports"), span


def test_a_non_transporter_span_still_refuses_a_transporter_row() -> None:
    """The preservation half: widening 1d may not license mere presence."""

    assert not accepted(
        "P", "P was detected in the membrane fraction",
        container="transporters", bucket="transports",
    )
    assert not accepted(
        "P", "A significantly inhibited P activity in the assay",
        container="transporters", bucket="transports",
    )


# ---------------------------------------------------------------------------
# 1e. Role "cofactor". BASE FAILURE: 4 of 7.
# ---------------------------------------------------------------------------

COFACTOR_SPANS = [
    "P is a required cofactor for the step",
    "the reaction requires P as a cofactor",
    "the enzyme is dependent on P for activity",
    "the conversion proceeds only in the presence of P",
    "P is the coenzyme of this reaction",
    "P is the prosthetic group of the enzyme",
]


@pytest.mark.parametrize("span", COFACTOR_SPANS)
def test_an_evidenced_cofactor_row_is_licensed(span: str) -> None:
    """At base ``cofactor`` fell to the "other" family and refused every one.

    The documented fallback is "strictly more permissive than the four families",
    and it still refused -- because it is built from those four patterns and not
    one of them contains a cofactor-predicating word. A permissive fallback with
    no vocabulary for the role has nothing to be permissive with.
    """

    assert accepted("P", span, container="modifiers", role="cofactor"), span


def test_licensing_the_cofactor_role_is_not_a_route_for_a_rationale() -> None:
    """The bare schema noun is NOT a cue, exactly as "enzyme" is not one.

    The catalysis family lists "is the enzyme" and refuses "as an enzyme", which
    is why the F-146 rationale is refused. The cofactor family is written the same
    way: "is the cofactor" and "cofactor for" are cues, "as a cofactor" is not, so
    a promoted rationale cannot license the role it is asking for.

    THIS IS NOT THE GATE THAT TYPES AN ENTITY. Licensing the ROLE lets an
    evidenced cofactor modifier row survive; whether the named thing is a protein
    at all is F-100's ``cofactor_as_protein`` question, decided in the identity
    and gold layers, and this guard never writes ``entities.proteins``.
    """

    assert not accepted(
        "P", "add P as a cofactor to resolve the structural inconsistency",
        container="modifiers", role="cofactor",
    )
    assert not accepted(
        "P", "P was purchased from a commercial supplier",
        container="modifiers", role="cofactor",
    )


#: Correction round 1. The first version of the cofactor family excluded the bare
#: schema noun and then admitted bare "requires", "required for", "depends on",
#: "dependent on" and "in the presence of" -- which is the vocabulary a structural
#: rationale is actually WRITTEN IN. These two went from refused at the C-106 base
#: to ACCEPTED at that tip, which is F-146 reopened in a fifth family and C-107
#: section 5's stop condition ("if you find yourself relaxing a check so that an
#: UNEVIDENCED patch passes, stop"). The dependence terms are now anchored to the
#: actor, so what is REQUIRED has to be the protein being added.
COFACTOR_RATIONALES = [
    "the reaction requires a cofactor, so P is added to resolve the structural inconsistency",
    "P depends on the payload structure being consistent",
    "the reaction is required to be structurally consistent, so P is added",
    "add P as a cofactor to resolve the structural inconsistency",
]


@pytest.mark.parametrize("span", COFACTOR_RATIONALES)
def test_a_structural_rationale_cannot_license_the_cofactor_role(span: str) -> None:
    assert not accepted("P", span, container="modifiers", role="cofactor"), span


#: The corollary, and it is wider than the cofactor role. _ANY_ROLE_CUE_RE is
#: rebuilt from every _ROLE_CUE_RES value, so anything added to the cofactor
#: family widens the "other" fallback for EVERY unmapped role. That is the
#: consequence of this card's own M6 mutation finding -- the vocabulary is what
#: leaks -- and the first version did not draw it: role "chaperone" with "P is
#: required for the reaction to proceed" was refused at base and accepted at that
#: tip. The dependence route now reaches the cofactor family only.
UNMAPPED_ROLE_SPANS = [
    ("chaperone", "P is required for the reaction to proceed"),
    ("chaperone", "the reaction requires a chaperone, so P is added"),
    ("scaffold", "the assembly is dependent on the payload being consistent, so P is added"),
    ("adaptor", "P is present in the complex"),
]


@pytest.mark.parametrize("role,span", UNMAPPED_ROLE_SPANS)
def test_the_other_fallback_did_not_widen_for_an_unmapped_role(role: str, span: str) -> None:
    assert not accepted("P", span, container="modifiers", role=role), (role, span)


def test_a_refused_cofactor_row_is_reported_against_the_cofactor_family() -> None:
    """The rejection names the family, and that makes the ROLE MAP load-bearing.

    Found by mutation, not by design. Deleting the ``cofactor`` entry from
    _ROLE_FAMILY_BY_ROLE left the whole suite GREEN (mutation M6, preserved in
    evidence/c107_mutation_attack.attempt1-m6-survivor.log), because
    _ANY_ROLE_CUE_RE is rebuilt from every _ROLE_CUE_RES value -- so once the
    cofactor VOCABULARY exists, the "other" fallback licenses the same spans and
    the map entry changes nothing a licensing test can see. What it does change
    is the reason string, which is what batch tooling greps: without the entry the
    row is refused as the "other" role rather than the "cofactor" role. Pinning
    that here is what makes M6 bite; M6b attacks the vocabulary itself.
    """

    payload = _payload("P", "modifiers")
    _result, report = apply_patch_with_policy(
        payload,
        [{"op": "add", "path": "/processes/reactions/0/modifiers/-",
          "value": {"entity": "P", "role": "cofactor"},
          "confidence": 1.0,
          "evidence": "P was purchased from a commercial supplier"}],
        stage="audit",
    )
    reason = report["rejected"][0]["reason"]
    assert reason.startswith(REASON_PREFIX), reason
    assert "the cofactor role" in reason, reason


# ---------------------------------------------------------------------------
# 1f. "mediat" inside "intermediate". BASE FAILURE: 2 of 5.
# ---------------------------------------------------------------------------

def test_mediat_inside_intermediate_does_not_license_catalysis() -> None:
    assert not accepted("P", "P is an intermediate carrier in this pathway")
    assert not accepted("P", "P accumulates as one of the intermediates")


def test_mediat_still_licenses_the_repair_it_exists_for() -> None:
    """The docstring example, and the two other shapes "mediat" must reach.

    The wrapper case uses a THREE-character symbol, not "P". C-105's original
    preservation control passed on a one-character name, which is the single
    shape a whole-name rule always handles, and that is why it passed while the
    class it stood for was broken. A one-character symbol also cannot exercise
    this case at all: _identifying_match_tokens drops tokens shorter than three
    characters, so "P complex" identifies on "complex" alone and no span that
    names the bare symbol can match it. The first draft of this test asserted
    exactly that and went red -- preserved as
    evidence/c107_tip_focused.attempt1-one-char-name-in-wrapper-fixture.log.
    """

    assert accepted("Pab complex", "Pab mediates the condensation of glycine and succinyl-CoA")
    assert accepted("P", "P-mediated hydrolysis of A yields B")
    assert accepted("P", "the step is mediated by P")


# ---------------------------------------------------------------------------
# 2. Boundary. The widenings did not spill into the guard's scope.
# ---------------------------------------------------------------------------

def test_a_patch_with_no_evidence_at_all_is_still_refused() -> None:
    assert not accepted("P", None)
    assert not accepted("P", None, container="transporters", bucket="transports")
    assert not accepted("P", None, container="modifiers", role="cofactor")


def test_a_span_naming_a_different_protein_is_still_refused() -> None:
    assert not accepted("P", "Q catalyses the conversion of A to B")
    assert not accepted("P", "Q is the flippase for lipid A",
                        container="transporters", bucket="transports")


# ---------------------------------------------------------------------------
# 3. CORRECTION ROUND 2 -- the attenuation words are WORDS, not stems.
#
# Written as stems with a trailing "[a-z]*", the contra added for 1a matched
# "reduc" inside "reductase", "block" inside "blocker", "silenc" inside
# "silencer" and "interfer" inside "interferon", and refused legitimate evidenced
# catalysis spans naming EC class 1 enzymes -- whose members are literally called
# reductase. That is finding 1f's defect class reintroduced in the OVER-REFUSAL
# direction, which is the direction C-105 round 1 was rejected for.
#
# Measured: 0 of 10 refused at the C-106 base, 8 of 10 at the round-1 tip.
# A LEFT ANCHOR ALONE IS NOT THE FIX and these cases say why: it repairs the two
# where the stem starts mid-word and leaves the four where the stem starts a
# genuine word, because "[a-z]*" still eats the rest of it.
# ---------------------------------------------------------------------------

ENZYME_NOUNS_CONTAINING_AN_ATTENUATION_WORD = [
    ("P4X", "the reductase P4X catalyses the conversion of A to B"),
    ("NfsB", "the nitroreductase NfsB catalyses the conversion of A to B"),
    ("YkgC", "the oxidoreductase YkgC catalyses the conversion of A to B"),
    ("P4X", "the blocker protein P4X catalyses the conversion of A to B"),
    ("P4X", "the silencer complex P4X catalyses the conversion of A to B"),
    ("IRF3", "interferon IRF3 catalyses the conversion of A to B"),
    ("P4X", "the disulfide reductase P4X catalyses the reduction of the substrate"),
    ("P4X", "P4X, a quinone oxidoreductase, catalyses the conversion of A to B"),
    ("P4X", "the ferredoxin-NADP reductase P4X reduces the flavin cofactor"),
    ("P4X", "the hydrolase P4X catalyses the conversion of A to B"),
]


@pytest.mark.parametrize(
    "name,span", ENZYME_NOUNS_CONTAINING_AN_ATTENUATION_WORD,
    ids=[s.split()[1] for _n, s in ENZYME_NOUNS_CONTAINING_AN_ATTENUATION_WORD],
)
def test_an_enzyme_noun_containing_an_attenuation_word_still_licenses(
    name: str, span: str
) -> None:
    assert accepted(name, span), span


#: The other direction, on the same words: an attenuation word that really IS one
#: must still refuse. Without these the repair above could be "delete the contra".
ATTENUATION_WORDS_STILL_REFUSE = [
    "the reduction of P4X activity is mediated by Q",
    "the reduction of P4X is mediated by Q",
    "P4X activity showed reduction in the Q-mediated assay",
    "the blockade of P4X activity is mediated by Q",
    "the silencing of P4X is mediated by Q",
    "the interference of P4X activity is mediated by Q",
    "the blocking of P4X is mediated by Q",
    "the loss of P4X is mediated by Q",
    "the quenching of P4X is mediated by Q",
    "the depletion of P4X is mediated by Q",
    "the disruption of P4X is mediated by Q",
]


@pytest.mark.parametrize("span", ATTENUATION_WORDS_STILL_REFUSE)
def test_a_real_attenuation_word_still_refuses_after_the_anchoring(span: str) -> None:
    assert not accepted("P4X", span), span


# ---------------------------------------------------------------------------
# 4. CORRECTION ROUND 2 -- the cofactor dependence route's family SCOPE.
#
# Correction round 1 moved the loose dependence terms out of the family
# vocabulary and into an actor-anchored route reaching the cofactor family only.
# Nothing pinned the SCOPE half of that: flipping the route's family test to
# `if True` put it back on every family with the entire suite GREEN, which makes
# blocking finding 2's regression reintroducible for free. That is an F-144
# vacuity in the fix for a blocking finding.
#
# Every span below is licensable ONLY by the dependence route: none of them
# carries a catalysis, activation, inhibition or transport cue.
# ---------------------------------------------------------------------------

DEPENDENCE_ONLY_SPANS = [
    ("enzymes", "reactions", None, "the enzyme is dependent on P for activity"),
    ("enzymes", "reactions", None, "the reaction requires P as a cofactor"),
    ("modifiers", "reactions", "catalyst",
     "the conversion proceeds only in the presence of P"),
    ("modifiers", "reactions", "inhibitor", "the reaction requires P as a cofactor"),
    ("modifiers", "reactions", "activator",
     "the enzyme is dependent on P for activity"),
    ("transporters", "transports", None,
     "the conversion proceeds only in the presence of P"),
    ("modifiers", "reactions", "chaperone", "the reaction requires P as a cofactor"),
    ("modifiers", "reactions", "scaffold",
     "the assembly is dependent on P for activity"),
]


@pytest.mark.parametrize(
    "container,bucket,role,span", DEPENDENCE_ONLY_SPANS,
    ids=[f"{c}-{r}" for c, _b, r, _s in DEPENDENCE_ONLY_SPANS],
)
def test_the_cofactor_dependence_route_is_scoped_to_the_cofactor_family(
    container: str, bucket: str, role: str, span: str
) -> None:
    """Catalysis, activation, inhibition, transport and "other" must all refuse."""

    assert not accepted(
        "P", span, container=container, bucket=bucket, role=role
    ), (container, role, span)


def test_the_cofactor_family_itself_still_licenses_the_same_spans() -> None:
    """The scope pin above must not be satisfiable by breaking the route.

    Asserted in the same file and against the same sentences, so a change that
    turns the route off entirely fails here instead of passing there.
    """

    for span in ("the enzyme is dependent on P for activity",
                 "the reaction requires P as a cofactor",
                 "the conversion proceeds only in the presence of P",
                 "the assembly is dependent on P for activity"):
        assert accepted("P", span, container="modifiers", role="cofactor"), span

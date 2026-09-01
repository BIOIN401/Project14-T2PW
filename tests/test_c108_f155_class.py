"""C-108 -- F-155, all five members, closed as ONE CLASS.

Five times in three consecutive cards on one file, a pattern matched INSIDE A
LONGER WORD or matched A SCHEMA NOUN, and something that is not evidence about
this actor performing this role was accepted as that evidence -- or something
that is evidence was refused. This file pins the class, not the five frames.

G9 LABELLING, PER MEMBER. Read this before reading the tests.

===  =================================================  ==========================
(a)  the transport family's bare schema noun            CORRECTION of pre-existing
     self-licenses                                       behaviour. BASE FAILURE.
(b)  ``[^.]`` is a length bound, not a sentence bound   **NEW ACCEPTANCE TEST.**
                                                         NO BASE FAILURE, AND NONE
                                                         IS FABRICATED.
(c)  an actor whose own NAME is an enzyme noun          CORRECTION of pre-existing
     licenses itself                                     behaviour. BASE FAILURE.
(d)  C-105's attenuation stems carry the identical      CORRECTION of pre-existing
     unanchored defect inside the CONTRA                 behaviour. BASE FAILURE.
(e)  three load-bearing anchors that no test covers     **NEW COVERAGE of existing
                                                         CORRECT behaviour.** NO
                                                         BASE FAILURE. The
                                                         MUTATION is the proof.
===  =================================================  ==========================

The base failures for (a), (c) and (d) are BEHAVIOURAL and were measured at base
``f67e00a`` through the real seam, never against a private regex. Symbol absence
is not the proof: sections A, C and D import nothing but
``apply_patch_with_policy``, so at base they RUN and give the wrong answer.

  ===  ==================================================  ==================
  (a)  ``add P as a transporter to resolve the             ACCEPTED at base
       structural inconsistency``                          REFUSED at tip
  (c)  ``LpxC hydrolase was quantified in the lysate``     ACCEPTED at base
                                                           REFUSED at tip
  (d)  ``the inhibitor protein P4X catalyses the           REFUSED at base
       conversion of A to B``                              ACCEPTED at tip
  ===  ==================================================  ==================

  measured: evidence/c108_base_frames.log  vs  evidence/c108_tip_frames_r3.log
            evidence/c108_base_battery.log vs  evidence/c108_tip_battery_r4.log
            (the battery's own C5 case moves 1 -> 0, which is (a)'s committed,
             pre-existing, behavioural base failure)

MEMBERS (b) AND (e) CARRY NO BASE FAILURE AND MUST NEVER BE GIVEN ONE. (b) makes
the code say what it already does and its corpus movement is zero in both
directions. (e) covers behaviour that is already correct; each of its three tests
FAILS when its anchor is removed and passes when it is restored, and that
mutation is the entire proof.

THE PINNED SAFETY PROPERTIES COME FIRST, in section 0. The F-146 patch stays
REJECTED, the inhibition CUE stays byte-identical so the ``other`` fallback for
every unmapped role does not move, and the contra keeps refusing every span that
attenuates the actor. This card refuses more in two places and accepts more in
one; any of them disturbing section 0 makes the card wrong however good the rest
of it is.

The fixture is generic. ``P``, ``P4X``, ``Q``, ``A`` and ``B`` carry no paper and
no protein: what is under test is the shape of a span.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

GUARD_PATH = SRC / "t2pw" / "curation" / "apply_audit_patch.py"
HARNESS_PATH = (REPO / "docs" / "pwml_recovery_sprint" / "evidence"
                / "c102_mutation_attack.py")

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402

#: The greppable prefix the guard stamps on its rejections. Spelled literally
#: rather than imported, so this file exercises BEHAVIOUR at any SHA.
REASON_PREFIX = "unevidenced_actor_role"


def _payload(name: str, container: str, bucket: str) -> Dict[str, Any]:
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
    seam: Any = None,
) -> bool:
    """Drive ONE actor-role add through the real public seam. True == applied.

    Nothing private is imported and nothing is re-implemented: the verdict is the
    report the production entry point returns. ``seam`` lets the mutation tests
    in section E drive a freshly-loaded copy of the module.
    """

    entry = seam or apply_patch_with_policy
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
    result, report = entry(payload, [op], stage="audit")
    if report["summary"]["accepted_count"] == 1:
        return True
    # Merge rule 7: a refused patch leaves the payload exactly as it found it.
    assert result == before, result
    assert report["rejected"][0]["reason"].startswith(REASON_PREFIX), report["rejected"]
    return False


def transports(name: str, evidence: Optional[str], **kw: Any) -> bool:
    return accepted(name, evidence, container="transporters", bucket="transports", **kw)


# ---------------------------------------------------------------------------
# 0. THE PINNED SAFETY PROPERTIES. PRESERVATION CONTROLS, GREEN AT BASE TOO.
# ---------------------------------------------------------------------------

#: The rationale the audit round actually gave, generalised off the protein name.
#: P1: it is rejected at base and it must be rejected here.
F146_RATIONALE = (
    "add P as an enzyme to the decomposition reaction to resolve the structural "
    "inconsistency where an inhibitor is listed without a target enzyme"
)


def test_p1_the_f146_patch_is_still_rejected() -> None:
    """P1. This card widens the guard in one place; re-admitting F-146 would end it."""

    assert not accepted("P", F146_RATIONALE)


def test_p1_the_inhibitor_span_f146_argued_from_is_still_rejected() -> None:
    assert not accepted("NDM-1", "PSA significantly inhibited NDM-1 enzyme activity")


def test_p2_the_inhibition_cue_is_untouched_so_the_other_fallback_cannot_move() -> None:
    """The (d) split anchors the CONTRA and leaves the CUE exactly as it was.

    **THIS ARM IS STRUCTURAL AND FAILS AT BASE ON SYMBOL ABSENCE.** It imports
    _C105_INHIBITION_STEMS_SRC and _C107_INHIBITION_WORDS_SRC, which this card
    creates, so at ``f67e00a`` it raises ImportError rather than giving a wrong
    answer. **That failure is NOT offered as a base failure and G9 explicitly
    refuses symbol absence as proof.** It is labelled here, at the arm, as well
    as in the file header -- REV-108 R-f. The behavioural pins for member (d)
    are the appositive and preservation blocks far below, which RUN at base.

    _ANY_ROLE_CUE_RE is rebuilt from every _ROLE_CUE_RES value, so a change to
    the inhibition CUE would widen or narrow the ``other`` fallback for every
    unmapped role -- C-107's own M6 mutation finding. This asserts the two
    objects are no longer the same object AND that the cue's source is unchanged,
    which is the only combination that closes (d) without MEMBER (d) moving the
    fallback.

    CORRECTION ROUND 1 -- REV-108 R-d. The claim is exactly that and no more.
    _ANY_ROLE_CUE_RE DOES move in this card, 2,584 -> 6,461 characters, and every
    one of those characters comes from MEMBER (a), which rewrote the catalysis
    and transport vocabularies. Whether it widens BEHAVIOURALLY is measured
    below, over the vocabulary (a) actually changed, rather than argued from a
    length.
    """

    from t2pw.curation.apply_audit_patch import (  # noqa: PLC0415
        _C105_INHIBITION_STEMS_SRC,
        _C107_INHIBITION_WORDS_SRC,
        _CATALYSIS_CONTRA_RE,
        _ROLE_CUE_RES,
    )

    assert _CATALYSIS_CONTRA_RE is not _ROLE_CUE_RES["inhibition"], (
        "the contra is still a bare alias of the cue; (d) cannot be closed while "
        "one object does two opposite jobs"
    )
    assert _ROLE_CUE_RES["inhibition"].pattern == (
        _C105_INHIBITION_STEMS_SRC + "|" + _C107_INHIBITION_WORDS_SRC
    ), "the inhibition CUE moved; the 'other' fallback moved with it"


#: C-107's own pin, restated here because (d) is the change most able to break
#: it. Every one of these is refused at base and must stay refused.
UNMAPPED_ROLE_SPANS = [
    ("chaperone", "P is required for the reaction to proceed"),
    ("chaperone", "the reaction requires a chaperone, so P is added"),
    ("scaffold", "the assembly is dependent on the payload being consistent, so P is added"),
    ("adaptor", "P is present in the complex"),
]


@pytest.mark.parametrize("role,span", UNMAPPED_ROLE_SPANS)
def test_p2_the_other_fallback_still_did_not_widen(role: str, span: str) -> None:
    assert not accepted("P", span, container="modifiers", role=role), (role, span)


# ---------------------------------------------------------------------------
# (a) THE SCHEMA NOUN SELF-LICENSES.  BASE FAILURE, MEASURED.
#
# `_ACTOR_ROLE_PATH_RE` protects containers called `transporters`, `cargo` and
# `cargo_complex`; `_ROLE_FAMILY_BY_ROLE` spells the role `transporter`; the
# catalysis containers are `enzymes`, `catalysts` and `modifiers_or_enzymes`.
# A rationale arguing from payload shape is written in exactly those words, and
# `transport` matched inside `transporter` while `catalys` matched inside
# `catalyst`. C-105 excluded `enzyme`, `enzymatic` and `activity` from catalysis
# for this precise reason and wrote down why; the reasoning is carried across
# here.
# ---------------------------------------------------------------------------

#: BASE FAILURE. ACCEPTED at f67e00a, every one -- and the first of them is the
#: battery's own C5 case, which prints C5=1 at base and C5=0 at this tip.
SCHEMA_RATIONALES_TRANSPORT = [
    "add P as a transporter to resolve the structural inconsistency",
    "add P as the transporter to resolve the structural inconsistency",
    "add P as transporters to resolve the structural inconsistency",
    "P should be added as a transporter to resolve the inconsistency",
    "adding P as a transporter resolves the structural inconsistency",
    "listing P among the transporters resolves the inconsistency",
    "the payload lists no transporter, so P is added",
    "P, a transporter, is added to resolve the structural inconsistency",
    "assign P the transporter role to resolve the inconsistency",
    "register P as a channel to resolve the structural inconsistency",
    "add P as a translocator to resolve the structural inconsistency",
    "add P as an importer to resolve the structural inconsistency",
    "add P as an exporter to resolve the structural inconsistency",
    "add P as a symporter to resolve the structural inconsistency",
    "add P as an antiporter to resolve the structural inconsistency",
    "add P as a uniporter to resolve the structural inconsistency",
    "add P as an extruder to resolve the structural inconsistency",
]


@pytest.mark.parametrize("span", SCHEMA_RATIONALES_TRANSPORT)
def test_a_a_transport_schema_noun_cannot_license_the_row_it_asks_for(span: str) -> None:
    """The rationale may not license the role it is requesting.

    THE FIX IS BOUND GRAMMATICALLY, NOT LEXICALLY, and that is what these
    seventeen phrasings test: nine of them do not contain the card's quoted
    wording at all. A fix bound to "add P as a transporter" would close one row
    here.
    """

    assert not transports("P", span), span


#: BASE FAILURE. The same defect in the CATALYSIS family, where C-105 believed it
#: had already excluded the schema noun: it wrote `is the catalyst`,
#: `catalyst responsible`, `catalyst for` and `catalyst of this` as periphrastic
#: re-admissions, which is only sensible if the bare noun was meant to be
#: excluded -- and those four alternatives had been dead code ever since, because
#: `catalys` matched `catalyst` first.
SCHEMA_RATIONALES_CATALYSIS = [
    "add P as a catalyst to resolve the structural inconsistency",
    "add P as the catalyst to resolve the structural inconsistency",
    "add P to the catalysts so the reaction has an actor",
    "P is proposed as a catalyst to resolve the inconsistency",
    "add P as a biocatalyst to resolve the structural inconsistency",
]


@pytest.mark.parametrize("span", SCHEMA_RATIONALES_CATALYSIS)
def test_a_a_catalysis_schema_noun_cannot_license_the_row_it_asks_for(span: str) -> None:
    assert not accepted("P", span), span


#: PRESERVATION, and the half that matters most: the VERB is a legitimate cue and
#: the schema noun returns in a PREDICATION. Deleting the stem would have broken
#: every one of these, which is why it is anchored instead of removed.
TRANSPORT_PREDICATIONS = [
    "P transports A across the inner membrane",
    "P transported A into the periplasm",
    "P is transporting A across the membrane",
    "A is transported across the inner membrane by P",
    "P mediates the transport of A across the inner membrane",
    "the transport of A is carried out by P",
    "P is the importer of A",
    "P imports A into the cell",
    "P exports A from the cytoplasm",
    "P translocates A across the bilayer",
    "the translocation of A is driven by P",
    "P is the translocase of the inner membrane",
    "P is the permease for this substrate",
    "P pumps protons across the membrane",
    "P channels calcium into the cytosol",
    "P shuttles A between the two compartments",
    "P secretes A into the medium",
    "the efflux of A is driven by P",
    "P drives the uptake of A",
    "P is a transporter",
    "P is the transporter for A in this step",
    "P is one of the transporters of the inner membrane",
    "P acts as a transporter of A across the inner membrane",
    "P functions as the channel for calcium entry",
    "P serves as a carrier for A in this step",
    "P was the transporter for A in the reconstituted system",
    "P is a symporter for A and sodium",
    "P is the channel through which A crosses the membrane",
    "P is the carrier protein for A in this step",
    "P is the efflux pump for A",
]


@pytest.mark.parametrize("span", TRANSPORT_PREDICATIONS)
def test_a_the_transport_verb_and_every_predication_still_license(span: str) -> None:
    assert transports("P", span), span


CATALYSIS_PREDICATIONS = [
    "P is a catalyst",
    "P is the catalyst for this step",
    "P acts as a catalyst in the conversion of A to B",
    "P is the enzyme responsible for the decomposition of A into B",
    "P catalyses the conversion of A to B under physiological conditions",
    "P-catalyzed conversion of A to B is the rate-limiting step",
    "the catalysis of A by P is the rate-limiting step",
]


@pytest.mark.parametrize("span", CATALYSIS_PREDICATIONS)
def test_a_the_catalysis_verb_and_every_predication_still_license(span: str) -> None:
    assert accepted("P", span), span


def test_a_the_transporter_row_is_not_licensed_by_mere_presence() -> None:
    """C-105's own preservation control, restated: widening (a)'s repair may not
    turn presence into evidence, and narrowing it may not turn it into evidence
    either."""

    assert not transports("P", "P was detected in the membrane fraction")
    assert not transports("P", "A significantly inhibited P activity in the assay")


# ---------------------------------------------------------------------------
# (b) ``[^.]`` IS A LENGTH BOUND.
#
# *** NEW ACCEPTANCE TEST. NEW STATEMENT ABOUT THE CODE, NOT A BEHAVIOUR
# CHANGE. NO BASE FAILURE IS CLAIMED AND NONE IS FABRICATED. ***
#
# Nothing below corrects an observable behaviour, so there is nothing for it to
# fail at ``f67e00a``. What it pins is the property the old spelling MISSTATED:
# `_match_fold` replaces every run of non-alphanumerics with a single space
# before any pattern runs, so no "." ever reaches the haystack and "[^.]"
# excluded nothing. `_match_fold` is shared and calibrated and is NOT touched,
# and no real sentence bound is introduced -- that would be a behaviour change in
# the over-refusal direction and is not chartered.
#
# The equivalence is measured as well as pinned:
# evidence/c108_fold_equivalence.py runs both spellings over every row of the
# 692-row corpus -- 2,076 pattern/row pairs, 0 disagreements -- and reports the
# folded alphabet over all 1,384 spans and actor names.
# ---------------------------------------------------------------------------

FOLD_CASES = [
    ("a.b", "a b"),
    ("a...b", "a b"),
    ("a\nb", "a b"),
    ("a\r\nb", "a b"),
    ("NDM-1-catalyzed hydrolysis", "ndm 1 catalyzed hydrolysis"),
    ("P is an enzyme. Q is not.", "p is an enzyme q is not"),
    ("étude", "tude"),
]


@pytest.mark.parametrize("raw,folded", FOLD_CASES)
def test_b_new_folding_strips_every_period_and_newline(raw: str, folded: str) -> None:
    """NEW. The premise the old spelling got wrong, stated as a test."""

    from t2pw.curation.apply_audit_patch import _match_fold  # noqa: PLC0415

    assert _match_fold(raw) == folded


def test_b_new_the_folded_alphabet_is_exactly_a_z_0_9_and_space() -> None:
    """NEW. Why "[^.]" excluded nothing: there is no "." for it to exclude."""

    from t2pw.curation.apply_audit_patch import _match_fold  # noqa: PLC0415

    corpus = [
        "P catalyses A. Then Q, at 37°C, inhibits it!",
        "UDP-3-O-[R-3-hydroxymyristoyl]-N-acetylglucosamine deacetylase",
        "MsbA: an ABC transporter paradigm\nsecond line\ttabbed",
        "β-hydroxyacyl-ACP — an intermediate",
    ]
    seen: set[str] = set()
    for text in corpus:
        seen |= set(_match_fold(text))
    assert seen <= set("abcdefghijklmnopqrstuvwxyz0123456789 "), sorted(seen)


def test_b_new_the_gap_construct_is_a_length_bound_only() -> None:
    """NEW. The construct bounds LENGTH; it does not bound a sentence.

    Both halves are asserted. A contra one "sentence" away still fires, because
    there are no sentences after folding -- and the same contra outside the
    length bound does not, because the LENGTH is what the construct enforces.
    """

    assert not accepted(
        "P", "P catalyses the conversion of A to B. The inhibitor of P was added."
    )
    assert accepted(
        "P",
        "P catalyses the conversion of A to B. " + ("x " * 60)
        + "The inhibitor of P was added.",
    )


def test_b_new_the_folded_character_class_is_the_spelling_that_says_so() -> None:
    """NEW. The constant exists and is what the affected patterns are built from.

    This is a STRUCTURAL assertion about a NEW symbol and it is deliberately NOT
    offered as a base failure: symbol absence is not proof of anything, which is
    exactly why (b) carries the behavioural pins above as well.
    """

    from t2pw.curation.apply_audit_patch import (  # noqa: PLC0415
        _FOLDED_CHAR_SRC,
        _ROLE_CUE_RES,
    )

    assert _FOLDED_CHAR_SRC == r"[a-z0-9 ]"
    assert "[^.]" not in _ROLE_CUE_RES["catalysis"].pattern
    assert _FOLDED_CHAR_SRC in _ROLE_CUE_RES["catalysis"].pattern


# ---------------------------------------------------------------------------
# (c) AN ACTOR WHOSE OWN NAME IS AN ENZYME NOUN LICENSES ITSELF.
#     BASE FAILURE, MEASURED.
#
# A NAME IS NOT A CLAIM, and the name is written by the same patch the guard is
# judging -- so the patch was supplying both halves of its own evidence.
# ---------------------------------------------------------------------------

#: BASE FAILURE. ACCEPTED at f67e00a, every one. Nine of the eleven do not use
#: the card's quoted wording.
NAME_ONLY_SPANS = [
    ("LpxC hydrolase", "LpxC hydrolase was quantified in the lysate"),
    ("LpxC synthase", "LpxC synthase was quantified in the lysate"),
    ("LpxC transferase", "LpxC transferase was detected in the membrane fraction"),
    ("LpxC hydrolase", "LpxC hydrolase was purchased from a commercial supplier"),
    ("MurA synthase", "MurA synthase levels were unchanged in the mutant"),
    ("P kinase", "P kinase was resolved on the gel"),
    ("LpxC hydrolase", "the hydrolase LpxC was quantified in the lysate"),
    ("LpxC hydrolase", "we quantified LpxC hydrolase in the lysate"),
    ("LpxC hydrolase", "the LpxC hydrolase band was excised from the gel"),
    ("LpxC hydrolase", "LpxC hydrolase, LpxD and LpxA were all detected"),
    ("LpxC hydrolase", "purified LpxC hydrolase was stored at minus 80"),
    ("ferrochelatase", "modulating ferrochelatase levels"),
    ("enterobactin synthase complex",
     "enterobactin synthase complex, which includes EntD, EntE, and EntF"),
]


@pytest.mark.parametrize("name,span", NAME_ONLY_SPANS)
def test_c_an_actors_own_name_is_not_evidence_about_the_actor(
    name: str, span: str
) -> None:
    assert not accepted(name, span), (name, span)


#: PRESERVATION. The same names, with a real claim beside them. Masking the
#: name's TOKENS instead of its contiguous RUNS would have broken these.
NAMED_AND_PREDICATED = [
    ("LpxC hydrolase", "LpxC hydrolase catalyses the conversion of A to B"),
    ("LpxC hydrolase", "LpxC hydrolase is a hydrolase"),
    ("LpxC hydrolase", "LpxC hydrolase hydrolyses A to give B"),
    ("LpxC hydrolase", "A is converted to B by LpxC hydrolase"),
    ("LpxC hydrolase", "the conversion is catalysed by the LpxC hydrolase"),
    ("LpxC hydrolase", "LpxC is the hydrolase for this step"),
    ("LpxC synthase", "LpxC synthase is the enzyme responsible for this step"),
    ("MurA synthase", "the conversion of A to B is catalysed by MurA synthase"),
    ("acyltransferase complex", "LpxA is the acyltransferase for this step"),
    ("DNA polymerase I", "DNA polymerase I catalyses the extension"),
    ("P hydrolase", "P is a hydrolase"),
    ("P hydrolase", "P is the hydrolase for this step"),
    ("serine hydroxymethyltransferase",
     "the reaction is catalyzed by serine hydroxymethyltransferase"),
    ("UDP-N-acetylglucosamine acyltransferase",
     "LpxA, the first enzyme in the pathway, catalyzes the reversible acylation "
     "of UDP-GlcNAc"),
    ("MenD complex", "MenD catalyses the first irreversible step"),
    ("ALAS2 complex", "ALAS2 mediates the condensation of glycine"),
]


@pytest.mark.parametrize("name,span", NAMED_AND_PREDICATED)
def test_c_a_predication_beside_the_name_still_licenses(name: str, span: str) -> None:
    assert accepted(name, span), (name, span)


def test_c_the_over_refusal_trap_the_card_names_explicitly() -> None:
    """The actor's registry name shares a token with the span's ONLY predicating
    phrase.

    Masking the name's TOKENS deletes "translocase" and kills the only cue in the
    span. Masking contiguous RUNS deletes "inner membrane" -- which is naming --
    and leaves "translocase", which is the predication. This is the difference
    between the two mechanisms, and it is the reason the second was chosen.
    """

    assert transports(
        "inner membrane translocase", "P is the translocase of the inner membrane"
    )


def test_c_the_agentive_by_is_a_predication_not_a_naming() -> None:
    """"by X" marks X as the AGENT, which is a claim about the role.

    Found by measurement, not by design: three corpus rows were refused before
    this exemption existed, and every one of them is a legitimate catalysis span.
    """

    assert accepted(
        "isochorismatase",
        "isochorismate is converted to 2,3-dihydro-2,3-dihydroxybenzoate and "
        "pyruvate by EntB isochorismatase activity",
    )


def test_c_the_cofactor_dependence_route_is_not_masked() -> None:
    """The dependence frame is ALREADY actor-anchored, so it reads the unmasked
    window.

    Masking it deletes the needle the frame is built around. Measured: 3 of the
    battery's 7 cofactor cases went red the first time this was got wrong
    (evidence/c108_tip_battery_r1.log), which is how it was caught.
    """

    for span in (
        "the reaction requires P as a cofactor",
        "the enzyme is dependent on P for activity",
        "the conversion proceeds only in the presence of P",
        "P is a required cofactor for the step",
    ):
        assert accepted("P", span, container="modifiers", role="cofactor"), span


# ---------------------------------------------------------------------------
# (d) C-105'S ATTENUATION STEMS CARRY THE IDENTICAL UNANCHORED DEFECT, INSIDE
#     THE CONTRA.  BASE FAILURE, MEASURED.
#
# MERGE RULE 6 BINDS. The cue is unchanged; only the contra is anchored, and the
# agent nouns return to it TARGET-DIRECTED. The preservation block below is
# larger than the correction block on purpose.
# ---------------------------------------------------------------------------

#: BASE FAILURE. REFUSED at f67e00a, every one. In all of them the actor IS the
#: attenuator -- an apposition -- and the span says in so many words that it
#: catalyses.
APPOSITIVE_AGENT_NOUNS = [
    "the repressor complex P4X catalyses the conversion of A to B",
    "the suppressor protein P4X catalyses the conversion of A to B",
    "the inhibitor protein P4X catalyses the conversion of A to B",
    "the inhibitor P4X catalyses the conversion of A to B",
    "P4X, the inhibitor, catalyses the conversion of A to B",
    "the potent inhibitor P4X catalyses the conversion of A to B",
    "inhibitor P4X catalyses the conversion of A to B",
    "the antagonist P4X catalyses the conversion of A to B",
    "the downregulator P4X catalyses the conversion of A to B",
    "the inactivator P4X catalyses the conversion of A to B",
    "the attenuator protein P4X catalyses the conversion of A to B",
    "the abolisher P4X catalyses the conversion of A to B",
    "the repressor P4X hydrolyses A to give B",
    "the suppressor P4X mediates the conversion of A to B",
    "the antagonist P4X converts A into B",
]


@pytest.mark.parametrize("span", APPOSITIVE_AGENT_NOUNS)
def test_d_an_appositive_agent_noun_is_not_evidence_the_actor_is_shut_down(
    span: str,
) -> None:
    assert accepted("P4X", span), span


#: PRESERVATION -- THE BIOLOGICAL GATE. Every one of these aims the attenuation
#: AT the actor and every one must stay REFUSED. Merge rule 6 forbids weakening
#: this to admit the block above, so the block above is closed by a grammatical
#: split and not by deletion.
ATTENUATION_AIMED_AT_THE_ACTOR = [
    "A significantly inhibited P activity in the assay",
    "the inhibitor of P was added while Q mediates the conversion",
    "inhibitors of P were added while Q mediates the conversion",
    "the inhibitor for P was added while Q mediates the conversion",
    "an inhibitor selective for P was added while Q mediates",
    "the inhibitor targeting P was added while Q mediates",
    "the inhibitor directed against P was added while Q mediates",
    "the suppressor acting on P was added while Q mediates",
    "the P inhibitor was added while Q mediates the conversion",
    "the P specific inhibitor was added while Q mediates",
    "the P activity inhibitor was added while Q mediates",
    "the repressor of the P gene was deleted while Q mediates",
    "P inhibitors were used while Q mediates the conversion",
    "an inhibitor of P blocks the mediated conversion",
    "the repression of P is mediated by Q",
    "the suppression of P activity is mediated by Q",
    "Q suppresses P and mediates the conversion of A to B",
    "Q represses P and mediates the conversion of A to B",
    "P was inactivated before the mediated conversion of A to B",
    "the downregulation of P is mediated by Q",
    "P is attenuated in the mutant, which mediates the conversion",
    "the antagonism of P is mediated by Q",
    "abolishing P activity is mediated by Q",
    "Q blocks P and mediates the conversion of A to B",
    "P has an inhibitory role and Q catalyses the conversion",
    "an inhibitory effect on P was seen while Q mediates the conversion",
    "the attenuation of P is mediated by Q",
    "the inhibition of P is mediated by Q",
]


@pytest.mark.parametrize("span", ATTENUATION_AIMED_AT_THE_ACTOR)
def test_d_the_contra_still_refuses_every_attenuation_aimed_at_the_actor(
    span: str,
) -> None:
    assert not accepted("P", span), span


def test_d_the_f146_inhibitor_span_is_the_same_question() -> None:
    """The span F-146 argued from is an attenuation aimed at the actor."""

    assert not accepted("NDM-1", "PSA-mediated inhibition of NDM-1 activity")
    assert not accepted("NDM-1", "the inhibition of NDM-1 is mediated by PSA")


#: PRESERVATION. The inhibition FAMILY must keep licensing an inhibitor row --
#: the other job the one shared object used to do.
INHIBITION_ROWS = [
    "P is an inhibitor of X",
    "P inhibits the reaction",
    "P is the repressor of the operon",
    "P suppresses the conversion of A to B",
    "P is a suppressor of this step",
    "P is an antagonist of the receptor",
    "holo-P binds to the promoters of three gene clusters, silencing gene expression",
]


@pytest.mark.parametrize("span", INHIBITION_ROWS)
def test_d_the_inhibition_family_still_licenses_the_row_it_attests(span: str) -> None:
    assert accepted("P", span, container="modifiers", role="inhibitor"), span


#: PRESERVATION. The anchoring must withhold the AGENT NOUN and nothing else --
#: which is why it is written "(?!ors?(?![a-z]))" and not a word anchor. A word
#: anchor would stop "inhibitory", "attenuated" and "repression" firing the
#: contra, and every one of them is inhibitory whatever its object.
STEM_INFLECTIONS_STILL_CONTRA = [
    "the inhibition of P is mediated by Q",
    "P inhibitory activity was measured while Q mediates the conversion",
    "the repression of P is mediated by Q",
    "the suppression of P is mediated by Q",
    "the antagonism of P is mediated by Q",
    "P was inactivated before the mediated conversion",
    "the downregulation of P is mediated by Q",
    "abolishing P was mediated by Q",
    "P was attenuated before the mediated conversion",
]


@pytest.mark.parametrize("span", STEM_INFLECTIONS_STILL_CONTRA)
def test_d_only_the_agent_noun_is_withheld_from_the_contra(span: str) -> None:
    assert not accepted("P", span), span


# ---------------------------------------------------------------------------
# (e) THREE LOAD-BEARING ANCHORS THAT NO TEST COVERS.
#
# *** NEW COVERAGE OF EXISTING CORRECT BEHAVIOUR. NO BASE FAILURE IS CLAIMED
# AND NONE IS FABRICATED. THE MUTATION IS THE PROOF. ***
#
# Behaviour is already correct in every case below and was correct at
# ``f67e00a``. What was missing is a test: these anchors are the fix for a
# BLOCKING finding (C-107 correction round 2), and an untested fix for a blocking
# finding can be deleted by a future refactor with the whole suite green -- the
# exact shape of the V9 gap C-107 round 2 was sent back to pin.
#
# Each test REMOVES its anchor, asserts the spans flip ACCEPT -> REFUSE, and
# restores. D-084: the restore replays SAVED BYTES and proves it by sha256 and
# CRLF count. `git checkout --` reverts more and a text-mode write reverts less;
# neither is used, and neither is `git status --porcelain`, which is exactly what
# a broken restore left clean.
# ---------------------------------------------------------------------------


def _harness():
    """Import C-106's mutation harness BY PATH, without running the attack."""

    spec = importlib.util.spec_from_file_location("c108_mutation_harness",
                                                  HARNESS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RELOADS = [0]


def _fresh_seam():
    """Load the guard module from disk again, under a private alias.

    A fresh alias each time, so a mutated load can never be served from
    ``sys.modules`` to a later test -- and so the ordinary import at the top of
    this file is never disturbed.
    """

    _RELOADS[0] += 1
    alias = f"c108_guard_reload_{_RELOADS[0]}"
    spec = importlib.util.spec_from_file_location(alias, GUARD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(alias, None)
    return mod.apply_patch_with_policy


#: The four ``-ablation`` / ``-interference`` words F-155 (e) measured. Every one
#: is an ordinary English word that merely CONTAINS an attenuation stem, and
#: every one licenses today.
ORDINARY_WORDS_CONTAINING_A_STEM = [
    "the photoablation of P was measured while Q mediates the conversion",
    "the counterinterference of P was noted while Q mediates the conversion",
    "the microablation of P was recorded while Q mediates the conversion",
    "nonimpairment of P was recorded while Q mediates the conversion",
]

#: "silencer" and "blocker" AFTER the actor, reached through F2. C-107's right
#: anchor is what stops the stem eating the rest of a genuine word.
AGENT_NOUNS_AFTER_THE_ACTOR = [
    "P activity was assayed and the silencer complex mediates the conversion of A to B",
    "P activity was assayed and the blocker complex mediates the conversion of A to B",
]

#: The three anchors, as byte-exact single-occurrence substitutions.
ANCHOR_MUTATIONS = [
    pytest.param(
        '_ATTENUATION_WORD_SRC = (\n    r"(?<![a-z])(?:"',
        '_ATTENUATION_WORD_SRC = (\n    r"(?:"',
        ORDINARY_WORDS_CONTAINING_A_STEM,
        id="left-anchor-on-_ATTENUATION_WORD_SRC",
    ),
    pytest.param(
        '|interfere|interferes|interfered|interfering|interference"\n    r")(?![a-z])"',
        '|interfere|interferes|interfered|interfering|interference"\n    r")"',
        AGENT_NOUNS_AFTER_THE_ACTOR,
        id="right-anchor-on-_ATTENUATION_WORD_SRC",
    ),
    pytest.param(
        '_C107_INHIBITION_WORDS_SRC = (\n    r"(?<![a-z])(?:blockades?',
        '_C107_INHIBITION_WORDS_SRC = (\n    r"(?:blockades?',
        ORDINARY_WORDS_CONTAINING_A_STEM,
        id="left-anchor-on-the-six-inhibition-additions",
    ),
]


@pytest.mark.parametrize("old,new,spans", ANCHOR_MUTATIONS)
def test_e_new_coverage_the_anchor_is_load_bearing(old, new, spans) -> None:
    """NEW COVERAGE. Remove the anchor and these legitimate spans flip to REFUSE.

    The assertion before the mutation is the non-vacuity control: if the spans do
    not license on the unmutated tree, the mutation proves nothing.
    """

    harness = _harness()
    assert harness.find_occurrences(GUARD_PATH, old) == 1, (
        "the anchor's source text moved; re-pin this mutation before reading on"
    )

    unmutated = _fresh_seam()
    for span in spans:
        assert accepted("P", span, seam=unmutated), (
            "non-vacuity: this span must license BEFORE the mutation", span
        )

    saved = harness.apply_mutation(GUARD_PATH, old, new)
    try:
        mutated = _fresh_seam()
        flipped = [span for span in spans if not accepted("P", span, seam=mutated)]
        assert flipped == spans, (
            "removing this anchor changed nothing; the anchor is untested for a "
            "reason other than oversight", flipped
        )
    finally:
        harness.restore_saved_bytes(GUARD_PATH, saved)

    restored = _fresh_seam()
    for span in spans:
        assert accepted("P", span, seam=restored), ("restore did not restore", span)


def test_e_new_coverage_the_restore_is_byte_exact_and_proves_it() -> None:
    """NEW COVERAGE. D-084's own property, exercised on THIS card's target file.

    A clean ``git status --porcelain`` is exactly what the broken restore
    produced while it was rewriting every line ending in the file, so the harness
    asserts sha256 and a CRLF count instead -- and this asserts that it does.
    """

    harness = _harness()
    before = GUARD_PATH.read_bytes()
    old = '_ATTENUATION_OBJECT_SRC = r"(?:activit|express|level|abundance|function)"'
    new = '_ATTENUATION_OBJECT_SRC = r"(?:activit|express|level|abundance|function)"  # x'
    assert harness.find_occurrences(GUARD_PATH, old) == 1

    saved = harness.apply_mutation(GUARD_PATH, old, new)
    try:
        assert GUARD_PATH.read_bytes() != before
    finally:
        harness.restore_saved_bytes(GUARD_PATH, saved)

    after = GUARD_PATH.read_bytes()
    assert harness.sha256_of(after) == harness.sha256_of(before)
    assert harness.crlf_count(after) == harness.crlf_count(before)


# ---------------------------------------------------------------------------
# CORRECTION ROUND 1 -- REV-108's BLOCKING FINDING.
#
# *** THESE ARE PRESERVATION CONTROLS AND THEY ARE GREEN AT BASE. NO BASE
# FAILURE IS CLAIMED FOR THEM AND NONE EXISTS. ***
#
# Every span below is REFUSED at ``f67e00a``. Round 0 of this card ADMITTED
# fourteen of them, and that was a weakened biological gate: in every one the
# actor is the thing being shut down, and the last four say so twice over -- the
# span goes on to state that the catalysis STOPPED. Round 1 restores the base
# verdict while KEEPING member (d), which is the whole difficulty: the same
# agent noun must refuse here and license in the appositive block above.
#
# WHY ROUND 0 GOT IT WRONG, because the mechanism matters more than the fix:
# round 0 gave the agent nouns back to the contra as a bounded closed list of
# TARGET-DIRECTED frames with ACCEPT as the default outside them. Handoff lesson
# 3 -- a bounded closed list flips polarity between a cue and a contra; in a cue
# it under-accepts and is safe, in a contra it under-REFUSES and is not. The
# default is now inverted: an agent noun REFUSES unless it stands in an
# apposition with this actor.
#
# Ten of the fourteen were found by this round attacking its own repair rather
# than fitting it to the four spans it was handed
# (evidence/c108_r1_blocking_repro.log).
# ---------------------------------------------------------------------------

ACTOR_IS_THE_TARGET = [
    # REV-108's four, verbatim
    "P4X is a target of the inhibitor and catalyses the conversion of A to B",
    "P4X was subject to inhibitors during the assay, yet catalyses A to B",
    "P4X, whose inhibitor was characterised, catalyses the conversion of A to B",
    "the repressor bound P4X and the catalysis of A to B stopped",
    # the same grammar, this round's own attack on its own repair
    "P4X is the target of a suppressor and catalyses the conversion of A to B",
    "P4X remained a target of the antagonist while catalysing A to B",
    "the inhibitor was raised against P4X, which catalyses A to B",
    "an inhibitor was co-crystallised with P4X, which catalyses A to B",
    "P4X, for which an inhibitor exists, catalyses the conversion of A to B",
    "the suppressor bound P4X and the catalysis of A to B stopped",
    "the antagonist blocked P4X and the conversion of A to B stopped",
    "the repressor acts on P4X, which catalyses the conversion of A to B",
    "inhibitors were screened against P4X, which catalyses A to B",
    "P4X sensitivity to the inhibitor was measured while it catalyses A to B",
]


@pytest.mark.parametrize("span", ACTOR_IS_THE_TARGET)
def test_r1_an_agent_noun_refuses_unless_it_is_in_apposition_with_the_actor(
    span: str,
) -> None:
    """PRESERVATION, green at base. Round 0 admitted these; round 1 does not."""

    assert not accepted("P4X", span), span


def test_r1_the_appositive_exemption_is_not_a_target_frame_list() -> None:
    """The polarity itself, asserted rather than described.

    An agent noun with NOTHING around it -- no target head, no compound, no
    apposition -- must still refuse. Round 0 accepted this, because it was not
    on the closed list of target frames. That is the defect in one line.
    """

    assert not accepted(
        "P4X", "an inhibitor was mentioned and P4X catalyses the conversion of A to B"
    )


def test_r1_every_agent_noun_in_the_window_is_checked_not_the_first() -> None:
    """An appositive agent noun does not license past a target-directed one."""

    assert not accepted(
        "P4X",
        "the inhibitor P4X catalyses the conversion of A to B, and the inhibitor "
        "of P4X abolished it",
    )


# ---------------------------------------------------------------------------
# CORRECTION ROUND 1 -- REV-108's SURVIVING MUTATIONS R2 and R3, and R-a.
#
# The reviewer mutated two of member (a)'s constructs and the suite stayed green:
# the predication modifier gap, and the transport verb inflections. A guard with
# no test is not evidence, and R-a is the same gap seen from the other side --
# the missing inflection and the missing test are one hole.
# ---------------------------------------------------------------------------

#: R-a. "channeled" is the US past tense and it was MISSING from round 0's verb
#: list, which had channelled/channeling/channelling only. That is a real
#: over-refusal this card introduced -- base ACCEPTS it through the bare
#: "channel" stem -- and it is fixed rather than registered.
TRANSPORT_VERB_INFLECTIONS = [
    "P channels calcium into the cytosol",
    "P channeled calcium into the cytosol",
    "P channelled calcium into the cytosol",
    "P is channeling calcium into the cytosol",
    "P is channelling calcium into the cytosol",
    "P pumps protons across the membrane",
    "P pumped protons across the membrane",
    "P is pumping protons across the membrane",
]


@pytest.mark.parametrize("span", TRANSPORT_VERB_INFLECTIONS)
def test_r1_the_transport_verb_inflections_are_load_bearing(span: str) -> None:
    """R3. Dropping "channel" and "pump" as bare nouns kept these as VERBS, and
    until now nothing said so."""

    assert transports("P", span), span


#: R2. The gap between the determiner and the agent noun. Without it a paper's
#: ordinary modifiers break the predication.
PREDICATION_GAP_SPANS = [
    "P is a high affinity transporter",
    "P is an inner membrane transporter",
    "P acts as an inner membrane channel",
    "P is the outer membrane transporter for A",
    "P was a well characterised sodium carrier",
]


@pytest.mark.parametrize("span", PREDICATION_GAP_SPANS)
def test_r1_the_predication_modifier_gap_is_load_bearing(span: str) -> None:
    assert transports("P", span), span


#: R-c. The one token the gap may not cross. "P is a substrate of the
#: transporter TonB" says P is the SUBSTRATE, and it licensed P as a transporter
#: at base and at round 0.
GENITIVE_NOT_A_PREDICATION = [
    "P is a substrate of the transporter TonB",
    "P is a product of the transporter TonB",
    "P is a substrate of the channel TonB",
]


@pytest.mark.parametrize("span", GENITIVE_NOT_A_PREDICATION)
def test_r1_the_predication_gap_may_not_cross_a_genitive(span: str) -> None:
    assert not transports("P", span), span


# ---------------------------------------------------------------------------
# CORRECTION ROUND 1 -- REV-108's R-d, measured rather than argued.
#
# Member (a) rewrote the catalysis and transport vocabularies, and
# _ANY_ROLE_CUE_RE is rebuilt from every one of them, so the ``other`` fallback
# for every unmapped role is rebuilt too. C-107's pin covers the spans ITS card
# moved; these cover the vocabulary THIS card moved.
# ---------------------------------------------------------------------------

FALLBACK_SPANS_OVER_THIS_CARDS_VOCABULARY = [
    ("chaperone", "add P as a transporter to resolve the structural inconsistency"),
    ("chaperone", "add P as a catalyst to resolve the structural inconsistency"),
    ("scaffold", "add P as a channel to resolve the structural inconsistency"),
    ("adaptor", "P is a substrate of the transporter TonB"),
    ("chaperone", "P was detected in the membrane fraction"),
]


@pytest.mark.parametrize("role,span", FALLBACK_SPANS_OVER_THIS_CARDS_VOCABULARY)
def test_r1_the_other_fallback_did_not_widen_over_this_cards_vocabulary(
    role: str, span: str
) -> None:
    """The fallback grew in CHARACTERS. This asks whether it grew in BEHAVIOUR."""

    assert not accepted("P", span, container="modifiers", role=role), (role, span)


def test_r1_the_fallback_still_licenses_what_it_licensed_before() -> None:
    """The other direction: narrowing the fallback would be a regression too."""

    assert accepted(
        "P", "P is a required cofactor for the step",
        container="modifiers", role="cofactor",
    )

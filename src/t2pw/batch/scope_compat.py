"""Is Stage 0's scope a legitimate SPECIALIZATION of what the batch requested?

Pure, deterministic, offline. No LLM, no lexicon lookup, no network, no state:
two strings in, one verdict out. It is separate from ``batch/driver.py`` because
it is a *rule*, and a rule that decides whether a paper's run proceeds has to be
readable and testable on its own -- ``tests/test_c125_scope_specialization.py``
drives the table below directly.

THE DEFECT THIS ANSWERS (C-125)
-------------------------------
``t2pw.rag.eligibility.apply_stage0_observation`` compares the requested pathway
with the pathway Stage 0 read and reports a conflict whenever the two names are
not the same lexicon entry. That is right for a *different* pathway and wrong for
a *narrower statement of the same one*. Measured in the C-122 final smoke, two of
twelve papers died at Stage 0 with zero reactions for exactly that reason::

    PMC13474940  requested 'fumonisin biosynthesis'
                 Stage 0 read 'fumonisin B1 biosynthesis'
    PMC13123502  requested 'steroidal saponin biosynthesis'
                 Stage 0 read 'steroidal saponin (polyphyllin) biosynthesis in
                               Paris polyphylla, focusing on UGT-mediated
                               3-O-glucosylation'

Both are the same pathway in the same paper, said at two levels of detail.
Neither paper was ever judged on its biology.

WHY NOT A SUBSTRING TEST -- the rule this module deliberately is NOT
-------------------------------------------------------------------
``"heme biosynthesis" in "heme biosynthesis inhibitor screening"`` is ``True``,
and admitting it would drive a biosynthesis extraction from an inhibitor-screening
paper. Containment of a *name* says nothing about whether the second phrase is the
same claim narrowed; it is satisfied by any phrase that merely quotes the first.
So this module compares normalized TOKEN STRUCTURE, split into two halves that are
judged by two different rules.

THE RULE
--------
``compare_scope(requested, observed)`` calls the Stage-0 scope a specialization of
the request when **both** hold:

1. **The process kind agrees exactly.** Each side's tokens are mapped through
   :func:`kind_of` to a canonical family (``biosynthesis``, ``degradation``,
   ``inhibition``, ``transport_uptake``, ``negation``, ``study_design``, ...) and
   the two *sets* must be equal. A request for X biosynthesis is not satisfied by
   X degradation however well the subject tokens match -- the process kind is a
   REQUIRED agreement, not a scored one -- and it is not satisfied by "X
   biosynthesis inhibitor screening" either, whose kind set is
   ``{biosynthesis, inhibition, study_design}``. Set equality rather than
   containment is what makes the forbidden substring case fail: Stage 0 may add
   detail to the subject, never a second process.
2. **Every meaningful subject token of the REQUEST survives in the Stage-0
   subject.** Stage 0 may ADD tokens -- ``B1``, ``polyphyllin``, the organism, a
   focus clause -- but may never DROP or REPLACE one the request made. That single
   direction is the whole safety property: it lets Stage 0 be more specific and
   never lets it be about something else. ``'steroidal glycoalkaloid
   biosynthesis'`` vs ``'potato solanidane glycoalkaloid biosynthesis'``
   (ORCH-734's PMC7910490) is rejected here and must stay rejected: the request's
   ``steroidal`` is gone, so Stage 0 is making a different, narrower claim about a
   different subject, not a narrower statement of this one.

RULE 1 IS THE WHOLE GUARD, AND IT IS OPEN-WORLD. READ THIS BEFORE TRUSTING IT.
------------------------------------------------------------------------------
**Rule 2 is satisfied automatically by any substring relation.** If the Stage-0
phrase contains the request verbatim, then every request token is present, and
rule 2 has nothing to say. So rule 1 -- and only rule 1 -- is what separates this
module from the naive substring acceptance the card forbids.

Rule 1 works off a CLOSED token list against an OPEN world. A process word that
:data:`PROCESS_KINDS` does not know is not neutral: :func:`subject_tokens` files
it as a SUBJECT token instead, subject additions are exactly what rule 2 permits,
and the pair is admitted. For that phrase the module degenerates to substring
acceptance. Round-1 review of C-125 reproduced twelve such seam flips --
``suppression of X biosynthesis``, ``... repression``, ``... silencing``,
``... knockdown``, ``... blockade``, ``... deficiency``, ``... drug screening``,
``antifungal agents that abolish X``, ``non-X``, ``loss of X``, ``review of X``,
and ``uptake`` vs ``efflux`` -- every one of them a phrase Stage 0 plausibly
writes. They are closed below, and closing them is a lexicon patch, not a proof.

**This module therefore makes no completeness claim.** The residual exposure is
stated as an executable test arm
(``test_new_capability_the_residual_open_world_exposure_is_real``) rather than as
a comment, so the next reader meets it as a fact rather than as reassurance. When
an admitted-but-wrong phrase turns up in a run, the fix is a new entry here plus
an arm in that table; nothing else in the seam has to move.

DIRECTION IS ONE-WAY, ON PURPOSE
--------------------------------
The request may be more general than Stage 0; the request may never be more
specific. Asking for ``'fumonisin B1 biosynthesis'`` and getting a paper about
fumonisins generally is a *different, unproven* claim -- the paper may never
establish B1 -- so it stays a conflict. That asymmetry falls out of rule 2 without
a special case: the request's ``b1`` has nowhere to go.

TRANSPORT IS SPLIT BY DIRECTION; EVERY OTHER FAMILY IS NOT
----------------------------------------------------------
``uptake`` and ``efflux`` are opposite claims about the same molecule, exactly as
``biosynthesis`` and ``degradation`` are, so they are separate families
(:data:`KIND_TRANSPORT_UPTAKE` / :data:`KIND_TRANSPORT_EFFLUX`) and a request for
one is never satisfied by the other. Undirected words (``transport``,
``transporter``, ``translocation``, ``trafficking``) form a third family that
matches neither, so ``'siderophore transport'`` vs ``'siderophore efflux'`` is
also refused. That is deliberate over-refusal: it costs a run that a human can
re-request precisely, and the alternative admits a direction flip.

TWO RECORDED PROPERTIES THAT ARE NOT DEFECTS TODAY
--------------------------------------------------
* :data:`_TOKEN_SPLIT` keeps ASCII only, so ``'β-carotene'`` and ``'α-carotene'``
  tokenize identically and this module cannot separate them. Round-1 review A/B'd
  it at the seam: no decision changes, because ``apply_stage0_observation`` raises
  no conflict for those pairs at the base SHA either, so the pair never reaches
  this rule. Latent, recorded, not repaired here.
* A request whose every token is a process word or a negation prefix (``'antibiotic
  biosynthesis'``) has an empty SUBJECT and is refused by
  :data:`REASON_NO_REQUEST_SUBJECT`. That is a refusal, i.e. the pre-C-125
  behaviour, so it can only cost a rescue -- never grant one.

WHAT THIS MODULE DOES NOT DO
----------------------------
* It never decides organism agreement. An organism conflict raised by
  ``apply_stage0_observation`` is left exactly as it was.
* It never rewrites the request, the observation, or a conflict message. Its only
  product is a verdict and, for the driver, a PARTITION of the conflict list.
* It fails CLOSED everywhere: an empty request, a request that states no process
  kind, an empty Stage-0 scope, or a conflict message whose wording it does not
  recognise all leave the conflict standing. A degenerate request never becomes
  compatible-with-everything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Mapping, Sequence, Tuple

# ── Canonical process families ─────────────────────────────────────────────
KIND_BIOSYNTHESIS = "biosynthesis"
KIND_DEGRADATION = "degradation"
KIND_INHIBITION = "inhibition"
KIND_TRANSPORT = "transport"
KIND_TRANSPORT_UPTAKE = "transport_uptake"
KIND_TRANSPORT_EFFLUX = "transport_efflux"
KIND_METABOLISM = "metabolism"
KIND_SIGNALING = "signaling"
KIND_REGULATION = "regulation"
KIND_DETOXIFICATION = "detoxification"
KIND_FERMENTATION = "fermentation"
KIND_NEGATION = "negation"
KIND_STUDY_DESIGN = "study_design"

#: Tokens that name a KIND OF PROCESS, or frame one, mapped to the canonical
#: family they belong to.
#:
#: THIS LIST IS THE GUARD. It is closed and the world is not, and an unlisted
#: process word does NOT fail safe: :func:`subject_tokens` files it as a subject
#: token, rule 2 permits subject additions, and the pair is admitted. See the
#: module docstring, "RULE 1 IS THE WHOLE GUARD, AND IT IS OPEN-WORLD". Adding a
#: family or a synonym here only ever makes the rule refuse MORE, so a new entry
#: cannot loosen the gate -- which is why extending it is the correct response to
#: an admitted-but-wrong phrase found in a run.
#:
#: Every entry is written in the singular form produced by :func:`_singular`.
PROCESS_KINDS: Mapping[str, str] = {
    # -- assembly of the molecule -------------------------------------------
    "biosynthesis": KIND_BIOSYNTHESIS,
    "biosynthetic": KIND_BIOSYNTHESIS,
    "biogenesis": KIND_BIOSYNTHESIS,
    "synthesis": KIND_BIOSYNTHESIS,
    "synthetic": KIND_BIOSYNTHESIS,
    "anabolism": KIND_BIOSYNTHESIS,
    "anabolic": KIND_BIOSYNTHESIS,
    "production": KIND_BIOSYNTHESIS,
    "formation": KIND_BIOSYNTHESIS,
    # -- taking the molecule apart ------------------------------------------
    "degradation": KIND_DEGRADATION,
    "degradative": KIND_DEGRADATION,
    "catabolism": KIND_DEGRADATION,
    "catabolic": KIND_DEGRADATION,
    "breakdown": KIND_DEGRADATION,
    "decomposition": KIND_DEGRADATION,
    "depolymerization": KIND_DEGRADATION,
    "depolymerisation": KIND_DEGRADATION,
    # -- acting AGAINST the process rather than running it -------------------
    # Everything that turns the pathway DOWN, off, or away, whether chemically,
    # genetically or descriptively. Round-1 review reproduced eight seam flips in
    # this family alone; each of those words is here for that reason.
    "inhibition": KIND_INHIBITION,
    "inhibitor": KIND_INHIBITION,
    "inhibitory": KIND_INHIBITION,
    "inhibiting": KIND_INHIBITION,
    "suppression": KIND_INHIBITION,
    "suppressor": KIND_INHIBITION,
    "suppressing": KIND_INHIBITION,
    "suppressed": KIND_INHIBITION,
    "repression": KIND_INHIBITION,
    "repressor": KIND_INHIBITION,
    "repressed": KIND_INHIBITION,
    "silencing": KIND_INHIBITION,
    "silenced": KIND_INHIBITION,
    "knockdown": KIND_INHIBITION,
    "knockout": KIND_INHIBITION,
    "blockade": KIND_INHIBITION,
    "blocker": KIND_INHIBITION,
    "blocking": KIND_INHIBITION,
    "blocked": KIND_INHIBITION,
    "antagonism": KIND_INHIBITION,
    "antagonist": KIND_INHIBITION,
    "abolish": KIND_INHIBITION,
    "abolished": KIND_INHIBITION,
    "abolition": KIND_INHIBITION,
    "attenuation": KIND_INHIBITION,
    "attenuated": KIND_INHIBITION,
    "disruption": KIND_INHIBITION,
    "disrupted": KIND_INHIBITION,
    "ablation": KIND_INHIBITION,
    "deficiency": KIND_INHIBITION,
    "deficient": KIND_INHIBITION,
    "impairment": KIND_INHIBITION,
    "impaired": KIND_INHIBITION,
    "loss": KIND_INHIBITION,
    # -- how the paper was DONE, rather than what the pathway is -------------
    # A screen for something and the thing itself are different scopes, and a
    # review of a pathway is a different document from a study of it. Both are
    # refused for the same reason: the request did not ask for them.
    "screening": KIND_STUDY_DESIGN,
    "screen": KIND_STUDY_DESIGN,
    "review": KIND_STUDY_DESIGN,
    "survey": KIND_STUDY_DESIGN,
    "meta": KIND_STUDY_DESIGN,
    # -- moving the molecule, SPLIT BY DIRECTION -----------------------------
    "uptake": KIND_TRANSPORT_UPTAKE,
    "import": KIND_TRANSPORT_UPTAKE,
    "influx": KIND_TRANSPORT_UPTAKE,
    "absorption": KIND_TRANSPORT_UPTAKE,
    "internalization": KIND_TRANSPORT_UPTAKE,
    "internalisation": KIND_TRANSPORT_UPTAKE,
    "acquisition": KIND_TRANSPORT_UPTAKE,
    "efflux": KIND_TRANSPORT_EFFLUX,
    "export": KIND_TRANSPORT_EFFLUX,
    "secretion": KIND_TRANSPORT_EFFLUX,
    "excretion": KIND_TRANSPORT_EFFLUX,
    "extrusion": KIND_TRANSPORT_EFFLUX,
    "transport": KIND_TRANSPORT,
    "transporter": KIND_TRANSPORT,
    "translocation": KIND_TRANSPORT,
    "trafficking": KIND_TRANSPORT,
    # -- broader / other families -------------------------------------------
    "metabolism": KIND_METABOLISM,
    "metabolic": KIND_METABOLISM,
    "signaling": KIND_SIGNALING,
    "signalling": KIND_SIGNALING,
    "transduction": KIND_SIGNALING,
    "regulation": KIND_REGULATION,
    "regulatory": KIND_REGULATION,
    "upregulation": KIND_REGULATION,
    "downregulation": KIND_REGULATION,
    "detoxification": KIND_DETOXIFICATION,
    "detoxication": KIND_DETOXIFICATION,
    "fermentation": KIND_FERMENTATION,
}

#: Prefixes that NEGATE or oppose whatever follows them. Kind-bearing rather than
#: subject-bearing: ``'non-fumonisin B1 biosynthesis'`` and ``'antifungal agents
#: that abolish ...'`` are not narrower statements of ``'fumonisin
#: biosynthesis'``, and treating ``non`` / ``antifungal`` as ordinary subject
#: tokens is precisely how both were admitted in round 1.
#:
#: They are symmetric: a request that itself says ``'antibiotic biosynthesis'``
#: carries the same family, so this can only refuse a pair the request did not
#: also frame that way.
NEGATION_PREFIXES: Tuple[str, ...] = ("anti", "non")

#: Grammatical glue and framing verbs. Dropped from the SUBJECT of either side
#: because they carry no identity.
#:
#: ``"a"`` is deliberately ABSENT: "lipid A biosynthesis" is a real pathway name
#: whose ``A`` is the subject, and dropping it would let a request for lipid A be
#: satisfied by a Stage-0 reading of lipids generally.
STOPWORDS: FrozenSet[str] = frozenset(
    {
        "an",
        "and",
        "as",
        "at",
        "between",
        "by",
        "during",
        "focused",
        "focusing",
        "for",
        "from",
        "in",
        "into",
        "involved",
        "involving",
        "mediated",
        "of",
        "on",
        "or",
        "related",
        "that",
        "the",
        "this",
        "through",
        "to",
        "under",
        "via",
        "which",
        "with",
        "within",
    }
)

#: Words for "a pathway" as such. Dropped from the subject so that
#: "fumonisin biosynthetic pathway" and "fumonisin biosynthesis" have the same
#: subject. Kept deliberately tiny -- anything that could name a molecule, a gene
#: or an organism stays in the subject.
STRUCTURAL_WORDS: FrozenSet[str] = frozenset(
    {"pathway", "route", "process", "mechanism"}
)

#: The exact prefix ``apply_stage0_observation`` gives every PATHWAY conflict it
#: raises (both of its message forms start with it). Matching on it is what keeps
#: this module from ever touching an ORGANISM conflict, and a future reword there
#: makes the match fail, which leaves the conflict standing -- fail-closed.
PATHWAY_CONFLICT_PREFIX = "Stage 0 read pathway '"

#: ASCII only, so every non-ASCII character is a separator. See the module
#: docstring: this cannot separate ``'β-carotene'`` from ``'α-carotene'``, which
#: is recorded and, measured at the seam, changes no decision today.
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _singular(token: str) -> str:
    """A crude, symmetric de-pluralisation. Applied to BOTH sides identically.

    Guarded so it cannot eat a real name: only tokens longer than four
    characters, and never one ending in a doubled or vowel-``s`` pair, which is
    what protects ``biosynthesis``, ``paris``, ``analysis``, ``loss`` and
    ``virus``.
    """
    if len(token) > 4 and token.endswith("s") and not token.endswith(
        ("ss", "us", "is", "as", "os")
    ):
        return token[:-1]
    return token


def tokenize(text: str) -> Tuple[str, ...]:
    """Normalized tokens of a scope phrase, in order, duplicates kept.

    Lowercased and split on every non-alphanumeric run, so parentheses, commas
    and hyphens are separators: ``"steroidal saponin (polyphyllin) biosynthesis"``
    and ``"3-O-glucosylation"`` tokenize the way a reader would expect, and
    ``"non-fumonisin"`` yields ``("non", "fumonisin")``. Digits survive as part of
    a token (``B1`` -> ``b1``).
    """
    return tuple(
        _singular(tok) for tok in _TOKEN_SPLIT.split(text.lower().strip()) if tok
    )


def kind_of(token: str) -> str:
    """The canonical process family ``token`` names, or ``""`` for a subject token.

    The single classifier: :func:`process_kinds` and :func:`subject_tokens` both
    go through it, so a token can never be counted as a process AND as a subject,
    and adding a family in one place changes both halves at once.
    """
    listed = PROCESS_KINDS.get(token)
    if listed:
        return listed
    if token in NEGATION_PREFIXES:
        return KIND_NEGATION
    # ``antifungal``, ``nonribosomal``: the prefix is fused to the word, so the
    # tokenizer cannot split it. Length-guarded so ``anti``/``non`` inside a short
    # token cannot fire on something unrelated.
    if len(token) > 4 and token.startswith(NEGATION_PREFIXES):
        return KIND_NEGATION
    return ""


def process_kinds(tokens: Iterable[str]) -> FrozenSet[str]:
    """The canonical process families named by ``tokens``."""
    return frozenset(kind for kind in (kind_of(tok) for tok in tokens) if kind)


def subject_tokens(tokens: Iterable[str]) -> FrozenSet[str]:
    """What the phrase is ABOUT: everything that is not process, glue or framing."""
    return frozenset(
        tok
        for tok in tokens
        if not kind_of(tok)
        and tok not in STOPWORDS
        and tok not in STRUCTURAL_WORDS
    )


#: Verdict reasons. Stable strings -- they are what a test and an operator read.
REASON_SPECIALIZES = "stage0_scope_specializes_request"
REASON_EMPTY_REQUEST = "requested_scope_is_empty"
REASON_EMPTY_OBSERVED = "stage0_scope_is_empty"
REASON_NO_REQUEST_KIND = "requested_scope_names_no_process_kind"
REASON_NO_REQUEST_SUBJECT = "requested_scope_names_no_subject"
REASON_KIND_MISMATCH = "process_kind_mismatch"
REASON_SUBJECT_DROPPED = "requested_subject_token_missing_from_stage0_scope"


@dataclass(frozen=True)
class ScopeComparison:
    """One verdict, with everything a reader needs to see why."""

    requested: str
    observed: str
    compatible: bool
    reason: str
    requested_kinds: FrozenSet[str] = frozenset()
    observed_kinds: FrozenSet[str] = frozenset()
    requested_subject: FrozenSet[str] = frozenset()
    observed_subject: FrozenSet[str] = frozenset()
    missing_subject: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "requested": self.requested,
            "observed": self.observed,
            "compatible": self.compatible,
            "reason": self.reason,
            "requested_kinds": sorted(self.requested_kinds),
            "observed_kinds": sorted(self.observed_kinds),
            "requested_subject": sorted(self.requested_subject),
            "observed_subject": sorted(self.observed_subject),
            "missing_subject": list(self.missing_subject),
        }


def compare_scope(requested: str, observed: str) -> ScopeComparison:
    """Is ``observed`` (Stage 0) a legitimate specialization of ``requested``?

    See the module docstring for the rule, for why it is not a substring test, and
    for the open-world limit of rule 1. Every early return is a REFUSAL: the only
    way to ``compatible=True`` is through both gates.
    """
    requested = str(requested or "")
    observed = str(observed or "")
    req_tokens = tokenize(requested)
    obs_tokens = tokenize(observed)

    if not req_tokens:
        # A blank request is not "compatible with everything"; it is a request
        # nobody made, and nothing can be a specialization of it.
        return ScopeComparison(requested, observed, False, REASON_EMPTY_REQUEST)
    if not obs_tokens:
        return ScopeComparison(requested, observed, False, REASON_EMPTY_OBSERVED)

    req_kinds = process_kinds(req_tokens)
    obs_kinds = process_kinds(obs_tokens)
    req_subject = subject_tokens(req_tokens)
    obs_subject = subject_tokens(obs_tokens)
    common = dict(
        requested_kinds=req_kinds,
        observed_kinds=obs_kinds,
        requested_subject=req_subject,
        observed_subject=obs_subject,
    )

    if not req_kinds:
        # "fumonisin" alone states no process. Admitting a Stage-0 reading against
        # it would be admitting on the subject alone, which is how a degradation
        # paper gets extracted as a biosynthesis.
        return ScopeComparison(
            requested, observed, False, REASON_NO_REQUEST_KIND, **common
        )
    if req_kinds != obs_kinds:
        return ScopeComparison(
            requested, observed, False, REASON_KIND_MISMATCH, **common
        )
    if not req_subject:
        return ScopeComparison(
            requested, observed, False, REASON_NO_REQUEST_SUBJECT, **common
        )

    missing = tuple(sorted(req_subject - obs_subject))
    if missing:
        return ScopeComparison(
            requested,
            observed,
            False,
            REASON_SUBJECT_DROPPED,
            missing_subject=missing,
            **common,
        )
    return ScopeComparison(requested, observed, True, REASON_SPECIALIZES, **common)


def specialization_note(requested: str, observed: str) -> str:
    """The operator-facing sentence for a conflict this rule withdrew."""
    return (
        f"Stage 0 read pathway '{observed}', which specializes the requested "
        f"'{requested}' (same process kind, no requested subject term dropped), "
        "so the run continued"
    )


def partition_specialization_conflicts(
    conflicts: Sequence[str],
    *,
    requested_pathway: str,
    observed_pathways: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Split ``conflicts`` into (still conflicting, withdrawn as specializations).

    Only a PATHWAY conflict naming an observed pathway that
    :func:`compare_scope` calls a specialization is withdrawn. Everything else --
    an organism conflict, a pathway conflict about a pathway that fails the rule,
    and any message whose wording this module does not recognise -- comes back in
    the first list untouched, in its original order and with its original text.
    """
    compatible = [
        pathway
        for pathway in observed_pathways
        if pathway and compare_scope(requested_pathway, pathway).compatible
    ]
    remaining: List[str] = []
    withdrawn: List[str] = []
    for conflict in conflicts:
        match = next(
            (
                pathway
                for pathway in compatible
                if conflict.startswith(f"{PATHWAY_CONFLICT_PREFIX}{pathway}'")
            ),
            "",
        )
        if match:
            withdrawn.append(specialization_note(requested_pathway, match))
        else:
            remaining.append(conflict)
    return remaining, withdrawn


__all__ = [
    "KIND_BIOSYNTHESIS",
    "KIND_DEGRADATION",
    "KIND_DETOXIFICATION",
    "KIND_FERMENTATION",
    "KIND_INHIBITION",
    "KIND_METABOLISM",
    "KIND_NEGATION",
    "KIND_REGULATION",
    "KIND_SIGNALING",
    "KIND_STUDY_DESIGN",
    "KIND_TRANSPORT",
    "KIND_TRANSPORT_EFFLUX",
    "KIND_TRANSPORT_UPTAKE",
    "NEGATION_PREFIXES",
    "PATHWAY_CONFLICT_PREFIX",
    "PROCESS_KINDS",
    "REASON_EMPTY_OBSERVED",
    "REASON_EMPTY_REQUEST",
    "REASON_KIND_MISMATCH",
    "REASON_NO_REQUEST_KIND",
    "REASON_NO_REQUEST_SUBJECT",
    "REASON_SPECIALIZES",
    "REASON_SUBJECT_DROPPED",
    "STOPWORDS",
    "STRUCTURAL_WORDS",
    "ScopeComparison",
    "compare_scope",
    "kind_of",
    "partition_specialization_conflicts",
    "process_kinds",
    "specialization_note",
    "subject_tokens",
    "tokenize",
]

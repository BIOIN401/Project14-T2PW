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
   :data:`PROCESS_KINDS` to a canonical family (``biosynthesis``, ``degradation``,
   ``inhibition``, ``transport``, ...) and the two *sets* must be equal. A request
   for X biosynthesis is not satisfied by X degradation however well the subject
   tokens match -- the process kind is a REQUIRED agreement, not a scored one --
   and it is not satisfied by "X biosynthesis inhibitor screening" either, whose
   kind set is ``{biosynthesis, inhibition}``. Set equality rather than
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

DIRECTION IS ONE-WAY, ON PURPOSE
--------------------------------
The request may be more general than Stage 0; the request may never be more
specific. Asking for ``'fumonisin B1 biosynthesis'`` and getting a paper about
fumonisins generally is a *different, unproven* claim -- the paper may never
establish B1 -- so it stays a conflict. That asymmetry falls out of rule 2 without
a special case: the request's ``b1`` has nowhere to go.

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

#: Tokens that name a KIND OF PROCESS, mapped to the canonical family they
#: belong to. Closed and explicit: a token that is not listed here is treated as
#: part of the SUBJECT, which is the conservative direction (an unlisted process
#: word on the Stage-0 side then reads as an extra subject token, which cannot
#: turn a reject into an accept -- rule 2 only ever checks the request's tokens
#: against the Stage-0 side).
#:
#: Every entry is written in the singular form produced by :func:`_singular`.
PROCESS_KINDS: Mapping[str, str] = {
    # -- assembly of the molecule -------------------------------------------
    "biosynthesis": "biosynthesis",
    "biosynthetic": "biosynthesis",
    "biogenesis": "biosynthesis",
    "synthesis": "biosynthesis",
    "synthetic": "biosynthesis",
    "anabolism": "biosynthesis",
    "anabolic": "biosynthesis",
    "production": "biosynthesis",
    "formation": "biosynthesis",
    # -- taking the molecule apart ------------------------------------------
    "degradation": "degradation",
    "degradative": "degradation",
    "catabolism": "degradation",
    "catabolic": "degradation",
    "breakdown": "degradation",
    "decomposition": "degradation",
    "depolymerization": "degradation",
    "depolymerisation": "degradation",
    # -- acting ON the process rather than running it ------------------------
    "inhibition": "inhibition",
    "inhibitor": "inhibition",
    "inhibitory": "inhibition",
    "inhibiting": "inhibition",
    # -- moving the molecule -------------------------------------------------
    "transport": "transport",
    "transporter": "transport",
    "translocation": "transport",
    "efflux": "transport",
    "uptake": "transport",
    "secretion": "transport",
    "trafficking": "transport",
    # -- broader / other families -------------------------------------------
    "metabolism": "metabolism",
    "metabolic": "metabolism",
    "signaling": "signaling",
    "signalling": "signaling",
    "transduction": "signaling",
    "regulation": "regulation",
    "regulatory": "regulation",
    "detoxification": "detoxification",
    "detoxication": "detoxification",
    "fermentation": "fermentation",
}

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

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _singular(token: str) -> str:
    """A crude, symmetric de-pluralisation. Applied to BOTH sides identically.

    Guarded so it cannot eat a real name: only tokens longer than four
    characters, and never one ending in a doubled or vowel-``s`` pair, which is
    what protects ``biosynthesis``, ``paris``, ``analysis`` and ``virus``.
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
    and ``"3-O-glucosylation"`` tokenize the way a reader would expect. Digits
    survive as part of a token (``B1`` -> ``b1``).
    """
    return tuple(
        _singular(tok) for tok in _TOKEN_SPLIT.split(text.lower().strip()) if tok
    )


def process_kinds(tokens: Iterable[str]) -> FrozenSet[str]:
    """The canonical process families named by ``tokens``."""
    return frozenset(
        PROCESS_KINDS[tok] for tok in tokens if tok in PROCESS_KINDS
    )


def subject_tokens(tokens: Iterable[str]) -> FrozenSet[str]:
    """What the phrase is ABOUT: everything that is not process, glue or framing."""
    return frozenset(
        tok
        for tok in tokens
        if tok not in PROCESS_KINDS
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

    See the module docstring for the rule and for why it is not a substring test.
    Every early return is a REFUSAL: the only way to ``compatible=True`` is
    through both gates.
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
    "partition_specialization_conflicts",
    "process_kinds",
    "specialization_note",
    "subject_tokens",
    "tokenize",
]

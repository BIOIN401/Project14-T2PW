"""Shared catalysis-cue predicate for the two name-based enzyme attachers.

Two independent passes attach an enzyme to a reaction purely because a protein
name appears in that reaction's text:

* ``process_normalizer.attach_enzymes_from_reaction_evidence`` (Stage 5), and
* ``pipeline._inject_name_based_modifiers`` (the Stage-2 post-merge injector).

They used to disagree completely. The normalizer required the actor name to sit
inside a catalysis-cue window *and* to be the only actor that qualified; the
pipeline injector accepted a bare substring hit anywhere in the row's evidence
and attached every actor that matched. Run 2026-07-28_0919 shows what the looser
copy costs: for PMC12444477 ("The regulation of lipid A biosynthesis") Stage 1
extracted 9 reactions with exactly 1 enzyme each, and the delivered strict
payload carried 27 reactions and 204 enzyme rows, 177 of which have evidence of
exactly 119 or 120 characters -- the fingerprint of the injector's
``revidence_text[:120]`` slice. Replaying the injector predicate over that
payload's own entity list reproduces all 204 attachments; replaying the
normalizer's cue predicate over the same rows yields 2.

The cue regex and the window scanner live here, in a leaf module with no
intra-package imports, so both call sites share one predicate instead of one
side reaching across a 4,500-line module for the other's private names.
"""

from __future__ import annotations

import re
from typing import Optional

# Widest evidence string the *name-based* attachers will read.
#
# Justification is measured, not guessed. In the PMC12444477 research payload of
# run 2026-07-28_0919 the 24 reactions' evidence lengths are
#   22, 39, 44, 53, 53, 73, 74, 75, 79, 102,
#   4636, 5113, 5113, 37946, 95170, 101739, 144419, 167474, 167474, 171962,
#   273702, 273702, 273702, 418122
# -- ten model-emitted sentences in [22..102], then a 45x empty gap, then RAG
# blob. The strict payload of the same paper has the same shape (nine values in
# [44..157], then 14324 and up); reaction #14 there is 139,576 characters made
# of one 4,812-character passage repeated 29 times, because ``rag/conform.py``
# flattens the retrieved evidence list into a single string before either
# attacher runs. Anything above this bound is therefore a retrieved corpus, and
# substring-matching protein names against a corpus attaches every enzyme to
# every reaction. 400 sits inside the empty gap with ~4x headroom over the
# longest genuine sentence observed and ~11x clearance under the smallest blob.
MAX_INJECTOR_EVIDENCE_CHARS = 400

# A protein name in a reaction sentence is not a claim that the protein
# catalyses it -- "LpxC is regulated by FtsH" names FtsH without making it the
# catalyst. These stems are the ones the Stage-5 attacher has always required
# within an 80-character window of the name.
ENZYME_EVIDENCE_CUE_RE = re.compile(
    r"(catalyz|catalys|catalytic|enzyme|enzymatic|mediated|dependent|activity|activat|promot|facilitat)",
    flags=re.IGNORECASE,
)


def collapse_whitespace(value: str) -> str:
    """Normalise runs of whitespace so a cue window is measured in real characters.

    Byte-for-byte the same rule as ``process_normalizer._canonical``; duplicated
    here only so this module keeps zero intra-package imports.
    """
    return re.sub(r"\s+", " ", (value or "").strip())


def cue_near_name(text: str, name: str, *, window: int = 80) -> Optional[str]:
    """Return the cue window around ``name`` in ``text``, or ``None``.

    Scans every occurrence of ``name`` and returns the first surrounding
    +/-``window``-character snippet that also contains a catalysis cue. Callers
    store that snippet as the attachment's evidence, which is what makes an
    injected row auditable: the reader sees the sentence that justified it
    rather than the first 120 characters of whatever the row happened to carry.
    """
    evidence = collapse_whitespace(text)
    actor_name = collapse_whitespace(name)
    if not evidence or not actor_name:
        return None
    for match in re.finditer(re.escape(actor_name), evidence, flags=re.IGNORECASE):
        start = max(0, match.start() - window)
        end = min(len(evidence), match.end() + window)
        snippet = evidence[start:end].strip()
        if ENZYME_EVIDENCE_CUE_RE.search(snippet):
            return snippet
    return None

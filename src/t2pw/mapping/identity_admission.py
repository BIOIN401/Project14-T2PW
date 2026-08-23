"""C-073 -- an entity may not ship a real external accession it cannot support.

F-096: run 2026-08-21_2239 emitted seven FALSE REAL IDENTIFIERS on legs the
pipeline reported ``PASS``. ``goldset.py``'s own design note calls that the worst
outcome the pipeline can produce, "because every structural gate passes and the
result is silently wrong", and acceptance priority 1 admits no non-zero count.

This module owns the two PURE PREDICATES the correction needs, and nothing else:

1. **Is this entity's identity supported by the paper it was extracted from?**
   Stage 2 resolves names against databases and deliberately never reads the
   paper (``map_ids.py:7719-7722``), so it cannot tell ``succinyl-CoA`` -- zero
   occurrences in the 67,304-character source of PMC12180156, the gold's
   designated HALLUCINATION TEST -- from the sixteen enzymes that paper really
   names. The paper is carried to it as a normalized SOURCE INDEX written at the
   Stage-2 merge; the predicate here just asks whether any name the row offers is
   locatable in that index.

2. **Does one accession answer to entities of INCOMPATIBLE KINDS?**
   ``drugbank:DB00114`` was claimed by both the protein ``ALAS2`` and the
   compound ``Pyridoxal 5'-phosphate`` on PMC12856317/research. One accession
   cannot denote both a protein and a metabolite, and F-096's gold classifies
   that row as ``cofactor_as_protein`` -- a TYPE error. That is the defect, and
   the predicate names it directly. The rule already existed twice in ``bench/``
   -- but only AFTER the strict quarantine, by which time the accession is
   already inside ``final_mapped.json``, which is a bench observation, not a gate.

   **A shared accession WITHIN one kind is not a collision at all.** D-035
   clause 3c is explicit that "at least one matching stable external identifier"
   is *proof that two differently-named rows are the same biological entity*.
   Reading that same fact as evidence that one of them is false would stand a
   LOCKED decision on its head. Measured over all committed ``final_mapped.json``
   artifacts, a name-difference predicate refuses 41 such pairs -- ``EntB`` /
   ``Isochorismatase (EntB)`` on ``uniprot:P0ADI4``, ``PEtN`` /
   ``Phosphoethanolamine`` across eight namespaces, ``LMRG_02730`` / ``MenI``
   (locus tag against gene symbol) -- every one of them one entity written twice,
   and every one of them legitimately owning the accession. The kind predicate
   refuses 1 and keeps all 41.

WHAT THIS MODULE MAY DO, AND THE DIRECTION OF ITS ERRORS
--------------------------------------------------------
It answers questions. It never edits a row: ``map_ids.map_payload`` owns the
withholding, so nothing here can drop an entity by accident. Withholding is
"take the accession off the row and file it under ``rejected_mapped_ids`` with a
reason" -- the entity survives with its name and its graph role intact (merge
rule 7).

Every ambiguity is resolved TOWARDS SUPPORT, because the collateral budget is
zero: a rule that strips a legitimate accession is worse than the defect it
fixes. Six alias/format misses were measured over the T-104 corpus
(``2,3-dihydroxybenzoic acid``/``...benzoate``, ``L-serine``/``Serine``,
``Adenosine triphosphate``/``ATP``, ``Adenosine monophosphate``/``AMP``,
``ferric-enterobactin``/``ferric enterobactin``, ``Fe3``/``Fe3+``) and all six
are kept, because the row's own ``synonyms``/``aliases``/``short_name`` are
consulted alongside its name and both sides are compared punctuation-blind.

It FAILS OPEN, always. No index, an empty index, or a row offering no name long
enough to look for is ``not_evaluated`` -- and PRODUCT_CONTRACT section 8's
``not_evaluated`` is never ``false``. The gate cannot fire on a payload it was
given no evidence about, which is what keeps unit tests, the
``interactive_curator`` path and every legacy payload behaving exactly as before.

C-075 -- WHAT ARMING IT IN PRODUCTION REQUIRED
----------------------------------------------
C-073 shipped this module dormant: no production caller supplied the paper, so
the answer was ``not_evaluated`` everywhere and the 1-catch / 0-collateral figure
was measured on ONE run (2026-08-21_2239, 102 eligible rows). Replayed over all
**70** committed ``final_mapped.json`` artifacts against their own
``01_source_text.txt`` -- 678 eligible rows -- the same predicate refuses **39**,
and only 3 of those are hallucinations. The product owner's ruling of 2026-08-23
names both of the missing routes by name, and each is implemented above as a
strictly ADDITIVE clause that can only KEEP an accession:

* "**or another permitted provenance route**" -> :func:`provenance_route`.
  34 of the 39 are rows the RAG path imported from a NAMED other document
  (``PlsB`` / ``uniprot:P0A7A7`` / ``rag_provenance.source_id = PMC12898747``).
  The seed paper is not their evidence base, so they are ``not_evaluated``.
* "**a proven alias**", "**legitimate aliases must retain valid mappings**" ->
  :meth:`SourceIndex.names_in_one_span`. 2 of the remaining 5 are alias/format
  misses whose variant sits on the PAPER's side rather than in the row's own
  ``synonyms``: ``pyridoxal phosphate`` against "pyridoxal 5'-phosphate", and
  ``CoA-SH`` against "succinyl-CoA".

What survives is 3: ``succinyl-CoA`` on two PMC12180156 legs and
``protoporphyrin IX`` on a third -- names with ZERO occurrences in a 67,553-
character paper, carrying no retrieval route. Collateral over 678 rows: 0.

Pure, offline, deterministic: no LLM, no network, no database, no clock, no I/O,
no mutation of anything the caller owns.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SCHEMA_VERSION",
    "SOURCE_INDEX_KEY",
    "REPORT_KEY",
    "META_KEY",
    "RULE_NOT_SUPPORTED",
    "RULE_ACCESSION_KIND_CONFLICT",
    "ENTITY_KIND_PROTEIN",
    "ENTITY_KIND_COMPOUND",
    "STATUS_SUPPORTED",
    "STATUS_UNSUPPORTED",
    "STATUS_NOT_EVALUATED",
    "NOT_EVALUATED_NO_INDEX",
    "NOT_EVALUATED_EMPTY_INDEX",
    "NOT_EVALUATED_NO_NAME",
    "NOT_EVALUATED_OTHER_PROVENANCE",
    "MIN_SUPPORT_CHARS",
    "EXTERNAL_ACCESSION_KEYS",
    "PROVENANCE_ROUTE_KEYS",
    "provenance_route",
    "SourceIndex",
    "normalize_text",
    "normalize_name_key",
    "build_source_index",
    "read_source_index",
    "candidate_names",
    "external_accessions",
    "identity_support",
    "accession_key",
    "entity_kind_class",
    "find_kind_conflicting_accessions",
    "conflict_matches",
]

#: Bumped only when the stored shape changes meaning. A reader that does not know
#: a version refuses to read it, which fails open rather than mis-parsing.
SCHEMA_VERSION = 1

#: Where the Stage-2 merge attaches the index, and where ``map_payload`` looks
#: for it. A top-level payload key, so it needs no new call signature anywhere.
SOURCE_INDEX_KEY = "source_text_index"

#: Where ``map_payload`` files this pass's decisions in the mapping report, and
#: on a refused row's ``mapping_meta``.
REPORT_KEY = "identity_admission"
META_KEY = "identity_admission"

RULE_NOT_SUPPORTED = "identity_not_supported_by_source"

#: One accession, two INCOMPATIBLE KINDS. Deliberately NOT "two different
#: names": D-035 clause 3c makes a shared identifier proof that two
#: differently-named rows are the SAME entity, so a name-difference rule would
#: read a locked decision backwards and strip 41 correct accessions to catch 1
#: false one. What cannot be true is that one accession denotes both a protein
#: and a metabolite.
RULE_ACCESSION_KIND_CONFLICT = "accession_claimed_across_incompatible_entity_kinds"

#: The two kind classes this predicate distinguishes, and the only distinction
#: it draws. ``protein_complexes`` is protein-ish: a complex of proteins sharing
#: a UniProt accession with one of its own components is D-035 3c agreement, not
#: a type error -- ``EntB`` / ``Isochorismatase (EntB)`` is exactly that shape.
ENTITY_KIND_PROTEIN = "protein"
ENTITY_KIND_COMPOUND = "compound"
_PROTEIN_BUCKETS: frozenset = frozenset({"proteins", "protein_complexes"})

STATUS_SUPPORTED = "supported"
STATUS_UNSUPPORTED = "unsupported"
STATUS_NOT_EVALUATED = "not_evaluated"

NOT_EVALUATED_NO_INDEX = "no_source_index"
NOT_EVALUATED_EMPTY_INDEX = "empty_source_index"
NOT_EVALUATED_NO_NAME = "no_evaluable_name"

#: C-075. The row was never claiming to come from the seed paper: it was imported
#: through the retrieval route and NAMES the document it came from, so the seed
#: paper is not its evidence base and this predicate has nothing to say about it.
NOT_EVALUATED_OTHER_PROVENANCE = "identity_from_another_admitted_source"

#: A one- or two-character match is a coincidence, not a citation: "Fe" occurs
#: inside "ferrochelatase" and every third English word contains "am". A
#: candidate shorter than this is not evidence in EITHER direction -- it cannot
#: support a row, and its absence cannot refuse one, which is why a row offering
#: only short names comes back ``not_evaluated`` instead of ``unsupported``.
MIN_SUPPORT_CHARS = 3

#: The external databases whose accessions are an identity CLAIM about the
#: molecule. Deliberately enumerated rather than "everything under
#: ``mapped_ids``": ``gene_name`` is a symbol the row already carried, not a
#: retrieved accession, and withholding it would take away the row's own name.
EXTERNAL_ACCESSION_KEYS: frozenset = frozenset({
    "uniprot", "drugbank", "chebi", "kegg", "hmdb", "pubchem", "cas", "biocyc",
    "chemspider", "pathbank_compound_id", "pathbank_protein_id",
    "pathbank_protein_complex_id", "pathbank_complex_id",
})

#: C-075 / the product owner's ruling of 2026-08-23: an identifier may survive on
#: "paper evidence, a proven alias, **or another permitted provenance route**".
#: These are the row-level marks of that other route -- the RAG retrieval path,
#: which imports an entity from a DIFFERENT, admitted document and records which
#: one. They are the source-NAMING half of ``pipeline._RAG_ROW_CARRIER_KEYS``
#: (``rag_provenance``, ``source_papers``, ``rag_confidence``).
#:
#: ``rag_confidence`` is excluded because it is a SCORE and names no source.
#: ``source_refs`` is excluded because it is overloaded: on a RAG row it holds
#: source ids, but on a Stage-1 row it holds evidence QUOTES. Measured, the
#: ``succinyl-CoA`` row of
#: ``runs/2026-08-02_2130/papers/PMC12180156/strict/final_mapped.json`` carries a
#: ``source_refs`` quote ("ALAS synthesizes the non-proteinogenic dALA from
#: succinyl-coenzyme A ...") that its own paper does not contain -- so reading
#: ``source_refs`` as a route would walk the hallucination out through the door
#: built for legitimate imports.
PROVENANCE_ROUTE_KEYS: Tuple[str, ...] = ("rag_provenance", "source_papers")

#: Placeholder spellings that name no record: ``map_ids``'
#: ``_is_real_protein_identifier`` list plus ``"0"``, because a PathBank scalar
#: of 0 is the absence of a row, not row zero. Duplicated rather than imported
#: because ``map_ids`` imports THIS module, and a predicate module that imports
#: its caller is a cycle waiting to happen.
_SENTINEL_IDENTIFIERS: frozenset = frozenset({"", "unknown", "n/a", "na", "none", "null", "-", "0"})

#: Single-valued name fields on an entity row, in the order § 4b names them.
_NAME_FIELDS: Tuple[str, ...] = ("name", "raw_name", "short_name")

#: List-valued name fields. ``synonyms`` is written by enrichment and is where
#: the alias forms live -- it is the whole reason the six measured alias misses
#: are not collateral.
_NAME_LIST_FIELDS: Tuple[str, ...] = ("aliases", "synonyms")

_WHITESPACE_RE = re.compile(r"\s+")


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def normalize_text(value: Any) -> str:
    """Casefold and squash punctuation, keeping word separation.

    Every run of non-alphanumeric characters becomes a single space, so
    ``ferric-enterobactin`` and ``ferric enterobactin`` fold together, ``Fe3+``
    folds to ``fe3``, and ``2,3-Dihydroxybenzoate`` folds to
    ``2 3 dihydroxybenzoate`` from either spelling. ``str.isalnum`` rather than a
    latin-only class, so ``δ-aminolevulinic acid`` keeps its delta on both sides
    of the comparison instead of one side silently losing a character.
    """
    text = str(value or "").casefold()
    if not text:
        return ""
    folded = "".join(ch if ch.isalnum() else " " for ch in text)
    return _WHITESPACE_RE.sub(" ", folded).strip()


def _is_insignificant(token: str) -> bool:
    """Whether a source token is material this module already refuses to treat as
    evidence: shorter than :data:`MIN_SUPPORT_CHARS`, or a bare number.

    Used only to decide what may sit BETWEEN two matched words of one name -- a
    locant (``5``), a stereodescriptor (``sn``, ``l``, ``d``), a Greek letter.
    """
    return len(token) < MIN_SUPPORT_CHARS or token.isdigit()


def normalize_name_key(value: Any) -> str:
    """Fold a name to a comparison key: normalized, then spaces removed.

    Used only to decide whether two rows are "differently named" for the
    collision rule, so ``ferric-enterobactin`` and ``ferric enterobactin`` are
    the same claimant, not two.
    """
    return normalize_text(value).replace(" ", "")


class SourceIndex:
    """A normalized copy of the seed paper, and the containment test over it.

    Two folds of the SAME stored string, both derived here rather than stored:
    the payload carries one string, and the space-free form costs one pass over
    it per mapping run. The space-free form exists so that a hyphenation the
    source and the row disagree about (``succinyl-CoA`` vs ``succinyl CoA``)
    cannot manufacture a refusal.
    """

    __slots__ = ("normalized", "squashed", "length", "_tokens", "_positions")

    def __init__(self, normalized: str, length: int = 0) -> None:
        self.normalized = normalized
        self.squashed = normalized.replace(" ", "")
        self.length = int(length)
        # Derived on first use and never stored on the payload: the token view
        # costs one pass over a string the payload already carries, so nothing
        # about the artifact's size changes.
        self._tokens: Optional[List[str]] = None
        self._positions: Optional[Dict[str, List[int]]] = None

    def __bool__(self) -> bool:
        return bool(self.normalized)

    def _token_view(self) -> Tuple[List[str], Dict[str, List[int]]]:
        if self._tokens is None:
            tokens = self.normalized.split()
            positions: Dict[str, List[int]] = {}
            for offset, token in enumerate(tokens):
                positions.setdefault(token, []).append(offset)
            self._tokens, self._positions = tokens, positions
        return self._tokens, (self._positions or {})

    def names_in_one_span(self, needle: str) -> bool:
        """Whether the candidate's SIGNIFICANT words occur here, in order, in one
        span, separated only by material that is not evidence in its own right.

        This is the "proven alias" half of the ruling, read off the SOURCE side
        instead of the row. The six alias cases C-073 measured were kept because
        the ROW carried the variant spelling in its own ``synonyms``; a row that
        carries none is not thereby unsupported, because the variation can sit on
        the paper's side just as easily. Measured on the committed corpus, two
        rows are exactly that shape and both are legitimate:

        * ``pyridoxal phosphate`` (``runs_verify/2026-08-04_1504/.../PMC12856317``)
          -- the paper writes "pyridoxal 5'-phosphate", which folds to
          ``pyridoxal 5 phosphate``. One locant sits between the two words, so
          plain substring containment misses a cofactor the paper names outright.
        * ``CoA-SH`` (``runs_verify/2026-08-04_1754/.../PMC12856317``) -- folds to
          ``coa sh``; the paper writes ``succinyl coa``. ``sh`` is two characters,
          which :data:`MIN_SUPPORT_CHARS` already rules is evidence in NEITHER
          direction, so what is left to look for is ``coa``, and it is there.

        Only INSIGNIFICANT tokens may sit between two matched words: shorter than
        :data:`MIN_SUPPORT_CHARS`, or a bare number. That is the same threshold
        the rest of this module uses to decide what counts as evidence, not a new
        one, and it is what stops two real words from opposite ends of a sentence
        being read as one name.

        Strictly ADDITIVE to :meth:`contains`: it is only ever consulted after
        substring containment has already failed, and it can only turn a refusal
        into a keep. No accession this predicate kept before can be taken away by
        it, so it cannot create collateral -- only remove it.
        """
        significant = [word for word in needle.split() if len(word) >= MIN_SUPPORT_CHARS]
        if not significant:
            return False
        tokens, positions = self._token_view()
        starts = positions.get(significant[0]) or []
        if not starts:
            return False
        if len(significant) == 1:
            return True
        total = len(tokens)
        for start in starts:
            cursor = start
            for word in significant[1:]:
                probe = cursor + 1
                while probe < total and tokens[probe] != word and _is_insignificant(tokens[probe]):
                    probe += 1
                if probe >= total or tokens[probe] != word:
                    cursor = -1
                    break
                cursor = probe
            if cursor >= 0:
                return True
        return False

    def contains(self, candidate: Any) -> bool:
        """Whether ``candidate`` is locatable in the source, punctuation-blind.

        Substring containment first, not token equality: a paper writes
        ``Serine`` in the middle of ``L-serine hydroxymethyltransferase`` and that
        still names serine. Then, only if that failed,
        :meth:`names_in_one_span`. The looser test is the safe direction here --
        every clause can only KEEP an accession, never take one away.
        """
        needle = normalize_text(candidate)
        if len(needle.replace(" ", "")) < MIN_SUPPORT_CHARS:
            return False
        if needle in self.normalized or needle.replace(" ", "") in self.squashed:
            return True
        return self.names_in_one_span(needle)


def build_source_index(text: Any) -> Optional[Dict[str, Any]]:
    """The compact, versioned index the Stage-2 merge attaches to the payload.

    ``None`` for an empty seed, so a leg with no source text carries no key at
    all and its payload stays byte-identical to today's.
    """
    normalized = normalize_text(text)
    if not normalized:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "length": len(str(text or "")),
        "normalized": normalized,
    }


def read_source_index(payload: Any) -> Optional[SourceIndex]:
    """Read the index off a payload, or ``None``.

    ``None`` for absent, malformed, empty, or a schema version this code does
    not know -- all four are "no evidence", and no evidence never refuses.
    """
    blob = _safe_dict(payload).get(SOURCE_INDEX_KEY)
    if not isinstance(blob, dict):
        return None
    if blob.get("schema_version") != SCHEMA_VERSION:
        return None
    normalized = blob.get("normalized")
    if not isinstance(normalized, str) or not normalized.strip():
        return None
    return SourceIndex(normalized, blob.get("length") or 0)


def candidate_names(row: Any) -> List[str]:
    """Every name this row offers as evidence of what it is, deduplicated.

    § 4b's list exactly: ``name``, ``raw_name``, ``short_name``, ``aliases``,
    ``synonyms``. Order is deterministic (fields first, in the order above, then
    list members in their stored order) so the "which name matched" record is
    reproducible.
    """
    out: List[str] = []
    seen: set = set()
    row = _safe_dict(row)
    values: List[Any] = [row.get(field) for field in _NAME_FIELDS]
    for field in _NAME_LIST_FIELDS:
        values.extend(_safe_list(row.get(field)))
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        key = normalize_name_key(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def provenance_route(row: Any) -> str:
    """Which permitted non-paper provenance route this row came in on, or ``""``.

    Names the key rather than returning a bool, so the answer is auditable.
    Total, and never raises: an unexpected shape under either key is simply not a
    route.
    """
    row = _safe_dict(row)
    for key in PROVENANCE_ROUTE_KEYS:
        value = row.get(key)
        if isinstance(value, dict) and value:
            return key
        if isinstance(value, list) and any(item for item in value):
            return key
    return ""


def _is_real_identifier(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.casefold() not in _SENTINEL_IDENTIFIERS


def external_accessions(row: Any) -> Dict[str, str]:
    """The real external accessions this row is about to ship, from ``mapped_ids``.

    ``mapped_ids`` only. The PathBank scalar columns on the row itself are the
    same values in another place; ``map_ids._strip_rejected_identifiers`` already
    takes those off in step with ``mapped_ids``, and reading them here as
    independent claims would double-count the row in the collision index.
    """
    ids = _safe_dict(_safe_dict(row).get("mapped_ids"))
    return {
        key: str(value).strip()
        for key, value in ids.items()
        if key in EXTERNAL_ACCESSION_KEYS and _is_real_identifier(value)
    }


def identity_support(row: Any, index: Optional[SourceIndex]) -> Dict[str, Any]:
    """Is this row's identity locatable in the source paper? Never raises.

    Returns ``{"status", "reason", "matched", "evaluated", "route"}`` where
    ``status`` is one of :data:`STATUS_SUPPORTED`, :data:`STATUS_UNSUPPORTED`,
    :data:`STATUS_NOT_EVALUATED`. ``matched`` names the candidate that was found,
    which is what makes a KEEP auditable and not merely silent; ``route`` names
    the permitted non-paper provenance that abstained, and is ``""`` otherwise.

    THE SEED PAPER IS NOT EVERY ROW'S EVIDENCE BASE (C-075). A row imported
    through the retrieval route states which document it came from, and the
    product owner's ruling admits "another permitted provenance route" alongside
    paper evidence and proven aliases. Asking THIS paper about such a row is
    asking the wrong document, and the honest answer to a question the evidence
    cannot address is ``not_evaluated``, never ``unsupported``
    (PRODUCT_CONTRACT § 8). Measured over all 70 committed ``final_mapped.json``
    artifacts replayed against their own ``01_source_text.txt``: 35 of 39
    refusals are rows of exactly this shape, each naming the paper it was
    imported from -- ``PlsB`` carrying ``uniprot:P0A7A7`` and
    ``rag_provenance.source_id = PMC12898747`` is the type case, and stripping it
    would be precisely the "legitimate mappings must be retained" half of the
    ruling being violated to enforce the other half.

    The names are looked for FIRST, so a row that IS named in the seed paper is
    reported ``supported`` with the matched name recorded, exactly as before; the
    route is consulted only where the answer would otherwise have been a refusal.
    """
    if index is None:
        return {"status": STATUS_NOT_EVALUATED, "reason": NOT_EVALUATED_NO_INDEX,
                "matched": "", "evaluated": 0, "route": ""}
    if not index:
        return {"status": STATUS_NOT_EVALUATED, "reason": NOT_EVALUATED_EMPTY_INDEX,
                "matched": "", "evaluated": 0, "route": ""}

    evaluated = 0
    for candidate in candidate_names(row):
        if len(normalize_name_key(candidate)) < MIN_SUPPORT_CHARS:
            # Too short to look for. Not counted as evaluated, so it can neither
            # support the row nor -- by its absence -- refuse it.
            continue
        evaluated += 1
        if index.contains(candidate):
            return {"status": STATUS_SUPPORTED, "reason": "", "matched": candidate,
                    "evaluated": evaluated, "route": ""}
    if not evaluated:
        return {"status": STATUS_NOT_EVALUATED, "reason": NOT_EVALUATED_NO_NAME,
                "matched": "", "evaluated": 0, "route": ""}
    route = provenance_route(row)
    if route:
        return {"status": STATUS_NOT_EVALUATED, "reason": NOT_EVALUATED_OTHER_PROVENANCE,
                "matched": "", "evaluated": evaluated, "route": route}
    return {"status": STATUS_UNSUPPORTED, "reason": RULE_NOT_SUPPORTED,
            "matched": "", "evaluated": evaluated, "route": ""}


def accession_key(namespace: Any, value: Any) -> str:
    """Fold an accession for identity comparison across rows.

    Case-insensitive, and a redundant ``"<namespace>:"`` prefix is dropped so
    ``chebi: CHEBI:15380`` and ``chebi: 15380`` are recognised as the same claim
    rather than two.
    """
    text = str(value or "").strip().casefold()
    prefix = f"{str(namespace or '').strip().casefold()}:"
    if prefix != ":" and text.startswith(prefix):
        text = text[len(prefix):].strip()
    return text


def entity_kind_class(bucket: Any) -> str:
    """Which of the two incompatible kinds a bucket belongs to.

    Total, and biased towards ``compound``: an unrecognised bucket is not
    protein-ish, and the caller only ever hands this the three buckets the mapper
    resolves identifiers for.
    """
    return ENTITY_KIND_PROTEIN if str(bucket or "") in _PROTEIN_BUCKETS else ENTITY_KIND_COMPOUND


def find_kind_conflicting_accessions(
    claims: Iterable[Tuple[Any, Any, Dict[str, str]]],
) -> Dict[Tuple[str, str], Tuple[Dict[str, str], ...]]:
    """Which ``(namespace, accession)`` pairs are claimed across incompatible kinds.

    ``claims`` is ``(kind_class, entity name, accessions)`` per row.

    A pair is refused ONLY when two of its claimants differ in **kind** AND in
    **normalized name**. Both conditions are load-bearing:

    * **Different kinds** is the incompatibility. One accession cannot denote a
      protein and a metabolite at once. Same-kind agreement is D-035 clause 3c
      evidence that the rows are the same entity and is left completely alone.
    * **Different names** keeps a routing artefact from reading as a type error.
      The same entity resolved into both ``compounds`` and ``proteins`` -- which
      ``route_entity_for_mapping`` can do -- is one entity written twice, not a
      protein masquerading as a metabolite.

    Returns the offending pairs mapped to their claimants (``kind`` and ``name``),
    sorted, so the caller's report is stable.
    """
    seen: Dict[Tuple[str, str], Dict[Tuple[str, str], str]] = {}
    for kind, name, accessions in claims:
        key_name = normalize_name_key(name)
        if not key_name:
            continue
        kind_class = str(kind or ENTITY_KIND_COMPOUND)
        for namespace, value in _safe_dict(accessions).items():
            folded = accession_key(namespace, value)
            if not folded:
                continue
            seen.setdefault((str(namespace), folded), {})[(kind_class, key_name)] = str(name or "")

    out: Dict[Tuple[str, str], Tuple[Dict[str, str], ...]] = {}
    for pair, claimants in sorted(seen.items()):
        keys = list(claimants)
        conflicted = any(
            left[0] != right[0] and left[1] != right[1]
            for index, left in enumerate(keys)
            for right in keys[index + 1:]
        )
        if conflicted:
            out[pair] = tuple(
                {"kind": kind_class, "name": claimants[(kind_class, key_name)]}
                for kind_class, key_name in sorted(keys)
            )
    return out


def conflict_matches(namespace: Any, value: Any, contested: Sequence[Tuple[str, str]]) -> bool:
    """Whether this row's identifier is one of the kind-conflicting pairs."""
    return (str(namespace), accession_key(namespace, value)) in set(contested)

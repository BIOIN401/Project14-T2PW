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

2. **Does one accession answer to two differently-named entities?**
   ``drugbank:DB00114`` was claimed by both ``ALAS2`` and ``Pyridoxal
   5'-phosphate`` on PMC12856317/research. Two different molecules cannot both be
   DB00114, so the claim is incompatible and may not ship. The rule already
   existed twice in ``bench/`` -- but only AFTER the strict quarantine, by which
   time the accession is already inside ``final_mapped.json``, which is a bench
   observation, not a gate.

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
    "RULE_ACCESSION_COLLISION",
    "STATUS_SUPPORTED",
    "STATUS_UNSUPPORTED",
    "STATUS_NOT_EVALUATED",
    "NOT_EVALUATED_NO_INDEX",
    "NOT_EVALUATED_EMPTY_INDEX",
    "NOT_EVALUATED_NO_NAME",
    "MIN_SUPPORT_CHARS",
    "EXTERNAL_ACCESSION_KEYS",
    "SourceIndex",
    "normalize_text",
    "normalize_name_key",
    "build_source_index",
    "read_source_index",
    "candidate_names",
    "external_accessions",
    "identity_support",
    "accession_key",
    "find_accession_collisions",
    "collision_matches",
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
RULE_ACCESSION_COLLISION = "accession_claimed_by_a_differently_named_entity"

STATUS_SUPPORTED = "supported"
STATUS_UNSUPPORTED = "unsupported"
STATUS_NOT_EVALUATED = "not_evaluated"

NOT_EVALUATED_NO_INDEX = "no_source_index"
NOT_EVALUATED_EMPTY_INDEX = "empty_source_index"
NOT_EVALUATED_NO_NAME = "no_evaluable_name"

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

    __slots__ = ("normalized", "squashed", "length")

    def __init__(self, normalized: str, length: int = 0) -> None:
        self.normalized = normalized
        self.squashed = normalized.replace(" ", "")
        self.length = int(length)

    def __bool__(self) -> bool:
        return bool(self.normalized)

    def contains(self, candidate: Any) -> bool:
        """Whether ``candidate`` is locatable in the source, punctuation-blind.

        Substring containment, not token equality: a paper writes ``Serine`` in
        the middle of ``L-serine hydroxymethyltransferase`` and that still names
        serine. The looser test is the safe direction here -- it can only KEEP an
        accession, never take one away.
        """
        needle = normalize_text(candidate)
        if len(needle.replace(" ", "")) < MIN_SUPPORT_CHARS:
            return False
        return needle in self.normalized or needle.replace(" ", "") in self.squashed


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

    Returns ``{"status", "reason", "matched", "evaluated"}`` where ``status`` is
    one of :data:`STATUS_SUPPORTED`, :data:`STATUS_UNSUPPORTED`,
    :data:`STATUS_NOT_EVALUATED`. ``matched`` names the candidate that was found,
    which is what makes a KEEP auditable and not merely silent.
    """
    if index is None:
        return {"status": STATUS_NOT_EVALUATED, "reason": NOT_EVALUATED_NO_INDEX,
                "matched": "", "evaluated": 0}
    if not index:
        return {"status": STATUS_NOT_EVALUATED, "reason": NOT_EVALUATED_EMPTY_INDEX,
                "matched": "", "evaluated": 0}

    evaluated = 0
    for candidate in candidate_names(row):
        if len(normalize_name_key(candidate)) < MIN_SUPPORT_CHARS:
            # Too short to look for. Not counted as evaluated, so it can neither
            # support the row nor -- by its absence -- refuse it.
            continue
        evaluated += 1
        if index.contains(candidate):
            return {"status": STATUS_SUPPORTED, "reason": "", "matched": candidate,
                    "evaluated": evaluated}
    if not evaluated:
        return {"status": STATUS_NOT_EVALUATED, "reason": NOT_EVALUATED_NO_NAME,
                "matched": "", "evaluated": 0}
    return {"status": STATUS_UNSUPPORTED, "reason": RULE_NOT_SUPPORTED,
            "matched": "", "evaluated": evaluated}


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


def find_accession_collisions(
    claims: Iterable[Tuple[Any, Dict[str, str]]],
) -> Dict[Tuple[str, str], Tuple[str, ...]]:
    """Which ``(namespace, accession)`` pairs answer to more than one name.

    ``claims`` is ``(entity name, accessions)`` per row. Two rows that normalize
    to the SAME name are one entity written twice -- a duplicate row, somebody
    else's finding -- and are never a collision. Returns the offending pairs
    mapped to their claimant names, sorted, so the caller's report is stable.
    """
    seen: Dict[Tuple[str, str], Dict[str, str]] = {}
    for name, accessions in claims:
        key_name = normalize_name_key(name)
        if not key_name:
            continue
        for namespace, value in _safe_dict(accessions).items():
            folded = accession_key(namespace, value)
            if not folded:
                continue
            seen.setdefault((str(namespace), folded), {})[key_name] = str(name or "")
    return {
        pair: tuple(sorted(names.values()))
        for pair, names in sorted(seen.items())
        if len(names) > 1
    }


def collision_matches(namespace: Any, value: Any, colliding: Sequence[Tuple[str, str]]) -> bool:
    """Whether this row's identifier is one of the contested pairs."""
    return (str(namespace), accession_key(namespace, value)) in set(colliding)

"""Strict name matching for evaluation v2 -- explicit, documented, no congener credit.

The pinned-v1 matcher (``bench.goldset.GoldTerm.match``) accepts CONTAINMENT in both
directions on whole-token boundaries. That is right for finding an anchor in a
payload vocabulary, and wrong for grading a reaction participant: ``lipid A`` is
credited by ``Kdo2-lipid A``, ``menaquinone`` by ``menaquinone-7`` and ``coenzyme A``
by ``acetyl coenzyme A`` -- three different molecules. Evaluation v2 therefore
matches by EQUALITY of a normalized key, and every normalization step is listed here:

1. ``bench.goldset.normalize_name`` (reused, not reimplemented): NFKC; unicode dashes
   to ``-``; a trailing ``(protein)``/``(enzyme)``/``(gene)``/``(complex)`` note
   dropped; Greek letters SPELLED OUT (``alpha``/``beta`` stay distinct, so
   alpha-ketoglutarate never meets beta-ketoglutarate); casefold; a trailing charge
   sign dropped (``NAD+`` == ``NAD``); every other punctuation run becomes one space.
   Digits and locants are KEPT, so ``PGE2`` != ``PGE1``, ``MK-7`` != ``MK-8`` and
   ``2,3-DHB`` != ``2,5-DHB``.
2. Protonation form: a word ending ``-ic acid`` becomes ``-ate``
   (``2,3-dihydroxybenzoic acid`` == ``2,3-dihydroxybenzoate``). Acid and conjugate
   base are the same metabolite at physiological pH; nothing else is folded.
3. ``pyrophosphate`` == ``diphosphate`` (the same P2O7 moiety under old and IUPAC
   names).

Deliberately NOT done: substring or token containment; stereo-descriptor or locant
stripping (``L-serine`` meets ``serine`` only where a gold alias says so, with the
justification written next to it); dropping chain-length suffixes. A synonym the rules
do not cover must be a declared gold alias.

Two reading rules for the CANDIDATE side only, both documented where used:

* a trailing parenthetical (``2,3-dihydroxybenzoic acid (2,3-DHB)``) may CORROBORATE
  the outer name, never override it: a name credits a term when the whole or the OUTER
  part matches, each by equality. When only the INSIDE matches while a non-empty outer
  name does not, the verdict is ``conflict`` (:func:`name_verdict`), never a match --
  ``heptanoyl pyrophosphate (HepPP)`` is a C7 acyl diphosphate, not heptaprenyl
  diphosphate, whatever its abbreviation says (ORCH-742c repair B2). The evaluator
  cannot tell a generic outer name (``oxidoreductase (entA)``) from a contradicting
  one (``isochorismate synthase (EntB)``), so both go to hand review;
* for ENZYMES only, a trailing word ``complex`` is dropped and a protein complex is
  also represented by its component names and gene names (from the payload's own
  ``components`` list), because the enzyme question is "which protein catalyses
  this", not "what is the container called".
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from t2pw.bench.goldset import GoldTerm, normalize_name

_IC_ACID = re.compile(r"ic acid\b")
_PAREN = re.compile(r"^(?P<outer>.*?)\s*\((?P<inner>[^()]*)\)\s*$")


def strict_key(value: Any) -> str:
    """The comparison key. Two names match in v2 iff their keys are equal."""

    text = str(value if value is not None else "")
    # Rule 2 runs on the casefolded text before punctuation collapses, so the
    # 'ic acid' word boundary is still visible.
    text = _IC_ACID.sub("ate", text.casefold()) if "ic acid" in text.casefold() else text
    key = normalize_name(text)
    return key.replace("pyrophosphate", "diphosphate")


def candidate_keys(value: Any) -> Set[str]:
    """Keys that may CREDIT one payload name: the whole, and the outer part of a
    trailing parenthetical. The inside is never a crediting key: see :func:`name_verdict`."""

    text = str(value if value is not None else "").strip()
    keys = {strict_key(text)}
    m = _PAREN.match(text)
    if m:
        keys.add(strict_key(m.group("outer")))
    keys.discard("")
    return keys


def inner_key(value: Any) -> str:
    """Key of a trailing parenthetical's inside when a non-empty outer name stands
    before it, else ``''``."""

    m = _PAREN.match(str(value if value is not None else "").strip())
    return strict_key(m.group("inner")) if m and strict_key(m.group("outer")) else ""


def name_verdict(candidate: Any, wanted: Set[str]) -> str:
    """``match`` | ``conflict`` | ``''`` for one payload name against a set of keys.

    ``conflict``: only the parenthetical's inside is in ``wanted`` and the outer name is
    not -- the abbreviation would credit a name the outer text contradicts."""

    if candidate_keys(candidate) & wanted:
        return "match"
    inner = inner_key(candidate)
    return "conflict" if inner and inner in wanted else ""


def term_keys(name: str, aliases: Iterable[str] = ()) -> Set[str]:
    keys = {strict_key(n) for n in (name, *aliases)}
    keys.discard("")
    return keys


def matches(candidate: Any, name: str, aliases: Iterable[str] = ()) -> bool:
    """Strict v2 match of one payload name against one gold term."""

    return bool(candidate_keys(candidate) & term_keys(name, aliases))


def lenient_matches(candidate: Any, name: str, aliases: Iterable[str] = ()) -> bool:
    """The pinned-v1 matcher on the same inputs, for the side-by-side column."""

    return GoldTerm(name=name, aliases=tuple(aliases)).match(candidate) != ""


def enzyme_keys(name: Any, complexes: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Set[str]:
    """Keys an enzyme reference in a reaction row can stand for.

    ``complexes`` maps a strict key of a ``protein_complexes`` row name to that row;
    its ``components`` (name and gene_name) are added. Enzyme-only rule, see module
    docstring.
    """

    keys = set(candidate_keys(name))
    for key in list(keys):
        if key.endswith(" complex"):
            keys.add(key[: -len(" complex")])
    for key in list(keys):
        row = (complexes or {}).get(key)
        if not row:
            continue
        for comp in row.get("components") or ():
            if isinstance(comp, dict):
                for field in ("name", "gene_name"):
                    if comp.get(field):
                        keys |= candidate_keys(comp[field])
    keys.discard("")
    return keys


def complex_index(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    out: Dict[str, Mapping[str, Any]] = {}
    for row in ents.get("protein_complexes") or ():
        if isinstance(row, dict) and row.get("name"):
            out[strict_key(row["name"])] = row
    return out


def participant_names(value: Any) -> List[str]:
    """Names from a reaction slot holding strings or participant dicts.

    Same key order as ``participant_schema.PARTICIPANT_SCHEMA_NAME_KEYS`` (the
    production reader's list), restated so this module imports no pipeline code.
    """

    keys = ("name", "entity", "compound", "protein", "protein_complex", "element",
            "element_collection", "nucleic_acid")
    out: List[str] = []
    for item in value if isinstance(value, list) else ([value] if isinstance(value, str) else []):
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            for k in keys:
                if item.get(k):
                    out.append(str(item[k]).strip())
                    break
    return out


def enzyme_names(row: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for slot in ("enzymes", "modifiers", "catalysts"):
        out.extend(participant_names(row.get(slot)))
    return list(dict.fromkeys(out))


def text_key(text: Any) -> str:
    """A passage folded with the SAME rules as a name, padded for token search."""

    return f" {strict_key(text)} "


def mentions(name: str, keyed_text: str, aliases: Iterable[str] = ()) -> bool:
    """Whether a passage (already passed through :func:`text_key`) names ``name`` or an
    alias on whole-token boundaries. For finding a name in PROSE -- a retrieved span,
    a paper -- never for grading a participant, which is :func:`matches`."""

    return any(k and f" {k} " in keyed_text for k in term_keys(name, aliases))


def any_match(candidates: Sequence[str], name: str, aliases: Iterable[str] = ()) -> Optional[str]:
    wanted = term_keys(name, aliases)
    for c in candidates:
        if candidate_keys(c) & wanted:
            return c
    return None


__all__ = [
    "any_match",
    "candidate_keys",
    "complex_index",
    "enzyme_keys",
    "enzyme_names",
    "inner_key",
    "lenient_matches",
    "matches",
    "name_verdict",
    "participant_names",
    "mentions",
    "strict_key",
    "term_keys",
    "text_key",
]

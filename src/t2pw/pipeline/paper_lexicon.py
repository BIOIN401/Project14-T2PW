"""The paper as the dictionary for Stage-1 ensembling (ORCH-754a; D-104 rules 1 and 2).

Pooled draws write one molecule several ways -- ``GSA``, ``Glu 1-semialdehyde (GSA)``,
``porphobilogen`` -- and ORCH-752 measured that this, not cofactors, is why three
chlorophyll draws pool to 41 reaction keys. The product owner's ruling: the only authority
on which spellings are one name is the paper itself.

Rule 1 -- abbreviations. Two names are one when the paper defines one as the abbreviation of
the other. Pairs are found with the Schwartz-Hearst algorithm (Schwartz & Hearst, PSB 2003),
restated here from its published reference implementation: the ``long form (SF)`` pattern,
the ``SF (long form)`` swap when the parenthetical has more than two words, the
shortest-long-form backward character match, and its validity limits. ONE extension, kept
narrow and labelled ``shared_locant`` in every record: the paper writes
``uroporphyrinogen (Urogen) III``, which defines ``Urogen III`` as ``uroporphyrinogen III``;
a roman numeral or single letter directly after the parenthesis extends both forms. An
abbreviation the paper defines with two different long forms is ambiguous and bridges
nothing.

Rule 2 -- typos snap to the paper's spelling. A name that does not occur in the paper, whose
unknown words are each within a small edit distance of exactly one paper word, snaps -- but
only when the snapped name then occurs in the paper as written. Safety, by construction: a
name under 6 characters never snaps; a name that occurs in the paper never snaps; only words
of 6+ characters are ever replaced, so short tokens that carry the chemistry (``1`` vs ``6``,
``alpha``/``beta``, ``D``/``L``, ``R``/``S``, ``cis``/``trans``, ``a``/``b``, roman numerals)
are never touched and every other character of the name, charge signs included, is kept; a
replaced word must keep its digits and Greek letters, its first two and last three
characters (``sulfate``/``sulfite``, ``maltose``/``maltase`` stay apart), its oxidation-state
and ``ethyl`` morphemes (``ferrocyanide``/``ferricyanide``, ``methyl``/``ethyl``), and differ
from it ONLY by inserted (or only by deleted) letters, at most 1 (2 from 10 characters up) --
never a substitution (ruling N4, 2026-10-20); and two or more candidates mean no snap. Every
snap is flagged ``typo_snapped`` by the ensemble, in its side file and text only. Known ceiling: a
one-letter edit inside a long word that changes the molecule in a way none of these rules
names would still snap -- but only onto a word the paper itself uses, from a name the paper
never uses.

Matching is by :func:`t2pw.pipeline.name_keys.strict_key`, the ensemble's key and
production's own copy of the evaluator's (that module says why it is a copy), so "occurs
in the paper" means "``mentions`` finds it on whole-token boundaries in the strict-keyed
text". Standard library only, besides that module (itself standard library only).
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from t2pw.pipeline.name_keys import strict_key

RULE_EXACT = "exact"
RULE_ABBREVIATION = "abbreviation"
RULE_TYPO = "typo"
RULE_DATABASE_ID = "database-id"

PATTERN_LONG_SHORT = "long (SF)"
PATTERN_SHORT_LONG = "SF (long)"
PATTERN_SHARED_LOCANT = "shared_locant"

_PAREN = re.compile(r"\s\(([^()]*)\)")
_TRAILING_PAREN = re.compile(r"^(?P<outer>.*?)\s*\((?P<inner>[^()]*)\)\s*$")
_LOCANT = re.compile(r"\s*([IVX]{1,4}|[A-Za-z])(?![\w-])")
_WORD = re.compile(r"\w+")
_GREEK = re.compile("[\\u0370-\\u03ff]|alpha|beta|gamma|delta|epsilon|omega", re.IGNORECASE)
_MIN_NAME = 6
#: One-letter edits that change WHICH molecule inside a word, where the prefix/suffix rule
#: cannot see them: an oxidation state (ferrocyanide/ferricyanide) or methyl -> ethyl.
_IDENTITY_MORPHEMES = re.compile(r"ferr[oi]|cupr[oi]|(?<!m)ethyl")
_DASHES = {**dict.fromkeys(range(0x2010, 0x2016), "-"), 0x2212: "-"}  # hyphens, dashes, minus


def spelling_key(value: object) -> str:
    """How a short form -- and, in the ensemble's rewrite map, every participant spelling --
    is matched: CASE-SENSITIVE, unlike every other name key here.
    ``PPI`` (protein-protein interaction, PMC12096016) is not ``PPi`` (pyrophosphate), and
    ``NO`` is not ``no``; only NFKC, dash style and whitespace are folded."""

    text = unicodedata.normalize("NFKC", str(value or "")).translate(_DASHES)
    return " ".join(text.split())


def _strict_key(value: object) -> str:
    return strict_key(value)


# ---------------------------------------------------------------------------- Schwartz-Hearst


def _best_long_form(short: str, long: str) -> Optional[str]:
    """``findBestLongForm``: match the short form's characters right to left; the first one
    must start a word. Returns the shortest long form ending at the end of ``long``."""

    s, l = len(short) - 1, len(long) - 1
    while s >= 0:
        ch = short[s].lower()
        if not ch.isalnum():
            s -= 1
            continue
        while (l >= 0 and long[l].lower() != ch) or (s == 0 and l > 0 and long[l - 1].isalnum()):
            l -= 1
        if l < 0:
            return None
        l -= 1
        s -= 1
    return long[long.rfind(" ", 0, l + 1) + 1:]


def _valid_pair(short: str, long: str) -> Optional[str]:
    """``extractAbbrPair``'s limits on a candidate pair; returns the long form or ``None``."""

    if len(short) < 2:
        return None
    best = _best_long_form(short, long)
    if best is None:
        return None
    long_words = len([t for t in re.split(r"[\s-]+", best) if t])
    short_chars = sum(1 for c in short if c.isalnum())
    if (len(best) < len(short) or f"{short} " in best or best.endswith(short)
            or long_words > short_chars * 2 or long_words > short_chars + 5 or short_chars > 10):
        return None
    return best


def abbreviation_pairs(text: str) -> List[Tuple[str, str, str]]:
    """Every ``(short form, long form, pattern)`` Schwartz-Hearst finds in ``text``."""

    out: List[Tuple[str, str, str]] = []
    for m in _PAREN.finditer(text):
        inner = m.group(1).split(", ")[0].split("; ")[0].strip()
        head = text[max(0, m.start() - 500):m.start()]  # ponytail: SH reads one sentence; 500 chars is longer than any clause it would keep
        cut = max(head.rfind(". "), head.rfind(", "), head.rfind("; "))
        long_cand = head[cut + 2:] if cut >= 0 else head
        if not inner or not long_cand.strip():
            continue
        pattern = PATTERN_LONG_SHORT
        short = inner
        if len(inner.split()) > 2 or len(inner) > len(long_cand):
            # SF (long form): the short form is the word before the parenthesis.
            short = head.rsplit(" ", 1)[-1] if " " in head else head
            long_cand, pattern = inner, PATTERN_SHORT_LONG
            if not any(c.isupper() for c in short):
                continue
        short = short.strip()
        if not short or not any(c.isalpha() for c in short) or not short[0].isalnum():
            continue
        long = _valid_pair(short, long_cand.strip())
        if long is None:
            continue
        if long.startswith("(") and ")" not in long:
            long = long[1:]  # SH keeps an unclosed outer parenthesis: "(2,5-dihydroxybenzoic acid (2,5-DHB)"
        # Written as the paper spells it, but with ASCII dashes: a U+2010 hyphen in a name is
        # a name a database lookup can miss, and strict_key already treats the two as one.
        long = spelling_key(long)
        out.append((short, long, pattern))
        tail = _LOCANT.match(text, m.end())
        if pattern == PATTERN_LONG_SHORT and tail:
            out.append((f"{short} {tail.group(1)}", f"{long} {tail.group(1)}", PATTERN_SHARED_LOCANT))
    return out


# ---------------------------------------------------------------------------- the lexicon


def _only_inserted_or_deleted(a: str, b: str) -> int:
    """How many letters turn one word into the other by INSERTION ONLY (or deletion only:
    the shorter word is a subsequence of the longer), or ``-1`` when that is impossible --
    any substitution, transposition or mixed change. Ruling N4: a snap never substitutes,
    so ``octanoate``/``octenoate``, ``sulfonate``/``sulfinate`` and ``heptanoate``/
    ``hexanoate`` can never meet."""

    short, long = sorted((a, b), key=len)
    if len(short) == len(long):
        return -1
    letters = iter(long)
    return len(long) - len(short) if all(ch in letters for ch in short) else -1


def _snappable(word: str, candidate: str) -> bool:
    """May ``word`` (not in the paper) be read as the paper's ``candidate``? See module doc."""

    if min(len(word), len(candidate)) < _MIN_NAME:
        return False
    if word[:2] != candidate[:2] or word[-3:] != candidate[-3:]:
        return False
    if re.findall(r"\d+", word) != re.findall(r"\d+", candidate):
        return False
    if _GREEK.findall(word) != _GREEK.findall(candidate):
        return False
    if _IDENTITY_MORPHEMES.findall(word) != _IDENTITY_MORPHEMES.findall(candidate):
        return False
    edits = _only_inserted_or_deleted(word, candidate)
    return 1 <= edits <= (1 if max(len(word), len(candidate)) < 10 else 2)


class PaperLexicon:
    """Rules 1 and 2 over one paper. Built once per ensemble; every method is pure."""

    def __init__(self, text: str) -> None:
        self.text = str(text or "")
        self._keyed = f" {_strict_key(self.text)} "
        words: Dict[str, str] = {}
        for w in _WORD.findall(self.text):
            words.setdefault(w.casefold(), w)
        self._words = words
        self._word_keys = list(words)
        found: Dict[str, Dict[str, Tuple[str, str, str]]] = {}
        for short, long, pattern in abbreviation_pairs(self.text):
            sk, lk = spelling_key(short), _strict_key(long)
            if sk and lk and sk != lk:
                found.setdefault(sk, {}).setdefault(lk, (short, long, pattern))
        #: case-sensitive short form -> (short, long, pattern), unambiguous pairs only
        self.pairs: Dict[str, Tuple[str, str, str]] = {
            sk: next(iter(forms.values())) for sk, forms in found.items() if len(forms) == 1}
        #: short-form key -> every long form the paper gave it, for the record
        self.ambiguous: Dict[str, List[str]] = {
            sk: sorted(f[1] for f in forms.values()) for sk, forms in found.items() if len(forms) > 1}

    def occurs(self, name: str) -> bool:
        key = _strict_key(name)
        return bool(key) and f" {key} " in self._keyed

    def snap(self, name: str) -> Optional[str]:
        """Rule 2: the paper's spelling of a misspelled ``name``, or ``None``."""

        if len(name.strip()) < _MIN_NAME or self.occurs(name):
            return None
        pieces: List[Tuple[Tuple[int, int], str]] = []
        for m in _WORD.finditer(name):
            word = m.group().casefold()
            if word in self._words:
                continue
            hits = [c for c in difflib.get_close_matches(word, self._word_keys, n=5, cutoff=0.8)
                    if _snappable(word, c)]
            if len(hits) != 1:
                return None
            pieces.append((m.span(), self._words[hits[0]]))
        if not pieces:
            return None
        out, last = [], 0
        for (start, end), word in pieces:
            out += [name[last:start], word]
            last = end
        snapped = "".join(out) + name[last:]
        return snapped if self.occurs(snapped) else None

    def long_form(self, name: str) -> Optional[Tuple[str, str]]:
        """Rule 1: ``(long form as the paper writes it, pattern)`` when the paper defines
        ``name`` -- or both halves of a ``X (Y)`` name -- as one abbreviation pair."""

        pair = self.pairs.get(spelling_key(name))
        if pair:
            return pair[1], pair[2]
        m = _TRAILING_PAREN.match(name.strip())
        if not m:
            return None
        outer, inner = m.group("outer"), m.group("inner")
        for short, other in ((inner, outer), (outer, inner)):
            pair = self.pairs.get(spelling_key(short))
            if pair and _strict_key(pair[1]) == _strict_key(other):
                return pair[1], pair[2]
        return None

    def canonical(self, name: str) -> Tuple[str, List[str], str]:
        """``(name to write, rules applied, abbreviation pattern)``: snap, then expand."""

        rules: List[str] = []
        out = name
        snapped = self.snap(name)
        if snapped is not None:
            out = snapped
            rules.append(RULE_TYPO)
        found = self.long_form(out)
        pattern = ""
        if found is not None and _strict_key(found[0]) != _strict_key(out):
            out, pattern = found
            rules.append(RULE_ABBREVIATION)
        return out, rules, pattern

    def summary(self) -> Dict[str, object]:
        return {
            "paper_chars": len(self.text),
            "abbreviation_pairs": len(self.pairs),
            "ambiguous_abbreviations": self.ambiguous,
        }


__all__ = [
    "PATTERN_LONG_SHORT", "PATTERN_SHARED_LOCANT", "PATTERN_SHORT_LONG", "PaperLexicon",
    "RULE_ABBREVIATION", "RULE_DATABASE_ID", "RULE_EXACT", "RULE_TYPO", "abbreviation_pairs", "spelling_key",
]


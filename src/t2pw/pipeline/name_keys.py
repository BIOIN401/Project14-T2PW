"""Strict name keys for production: ``strict_key``, ``text_key``, ``mentions``.

These are COPIES of the evaluation matcher's functions of the same names (and of the
v1 bench's ``canonical_text`` / ``normalize_name`` they are built on), kept here on
purpose (ORCH-754d). The evaluator measures what production does; if production called
the evaluator's normalizer, a bug in that normalizer would act on both sides at once and
the measurement could never see it. So ``t2pw.pipeline`` imports nothing from
``t2pw.bench``, and the evaluator imports nothing from here.

The cost is two copies that must agree. ``tests/test_name_keys_match_evaluation_matcher.py``
compares them on a broad probe set and is the alarm if they ever drift: a change to one
side is a change to the other, made deliberately, with that test kept green.

The rules (documented in full in the evaluation matcher's module docstring): NFKC; unicode
dashes to ``-``; a trailing ``(protein)``/``(enzyme)``/``(gene)``/``(complex)``/``(abbrev.)``
note dropped; Greek letters spelled out; casefold; a trailing charge sign dropped;
punctuation runs to one space; ``-ic acid`` -> ``-ate``; ``pyrophosphate`` ==
``diphosphate``. Digits and locants are kept. Standard library only.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, Set

# ---------------------------------------------------------------------------
# Copied verbatim from the v1 bench's normalization (canonical_text, normalize_name).
# ---------------------------------------------------------------------------
#: Greek letters are spelled out rather than stripped: "beta-hydroxyacyl-ACP" and
#: "β-hydroxyacyl-ACP" must compare equal, but "α-ketoglutarate" and
#: "β-ketoglutarate" must NOT -- stripping the letter would collapse two
#: different molecules onto one name.
_GREEK: Dict[str, str] = {
    "α": "alpha", "Α": "alpha",
    "β": "beta", "Β": "beta",
    "γ": "gamma", "Γ": "gamma",
    "δ": "delta", "Δ": "delta",
    "ε": "epsilon", "Ε": "epsilon",
    "κ": "kappa", "Κ": "kappa",
    "λ": "lambda", "Λ": "lambda",
    "μ": "mu", "Μ": "mu",
    "σ": "sigma", "Σ": "sigma",
    "ω": "omega", "Ω": "omega",
}

_DASHES = {"‐", "‑", "‒", "–", "—", "―", "−"}

#: Stripped before comparison. These are formatting, not identity: an entity
#: called "LpxC (deacetylase)" is the same protein as "LpxC".
_TRAILING_NOISE = re.compile(r"\s*\((?:protein|enzyme|gene|complex|abbrev\.?)\)\s*$", re.IGNORECASE)


def canonical_text(value: Any) -> str:
    """Collapse unicode punctuation and whitespace without changing identity."""

    text = str(value if value is not None else "")
    text = unicodedata.normalize("NFKC", text)
    out = []
    for char in text:
        if char in _DASHES:
            out.append("-")
        elif char == " ":
            out.append(" ")
        else:
            out.append(char)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def normalize_name(value: Any) -> str:
    """Fold an entity or term name into its comparable form.

    Greek letters become their Latin names, punctuation becomes a single space,
    and case is dropped. Digits are kept: ``MK-7`` and ``MK-4`` must not collide.
    """

    text = canonical_text(value)
    text = _TRAILING_NOISE.sub("", text)
    for greek, latin in _GREEK.items():
        text = text.replace(greek, latin)
    text = text.casefold()
    # Charge notation is identity-bearing for an ion but noise for a name match;
    # "NAD+" and "NAD" are the same entity for coverage purposes.
    text = re.sub(r"(?<=[a-z0-9])\s*[⁺⁻+]\s*$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Copied verbatim from the evaluation matcher (strict_key, term_keys, text_key, mentions).
# ---------------------------------------------------------------------------
_IC_ACID = re.compile(r"ic acid\b")


def strict_key(value: Any) -> str:
    """The comparison key. Two names match in v2 iff their keys are equal."""

    text = str(value if value is not None else "")
    # Rule 2 runs on the casefolded text before punctuation collapses, so the
    # 'ic acid' word boundary is still visible.
    text = _IC_ACID.sub("ate", text.casefold()) if "ic acid" in text.casefold() else text
    key = normalize_name(text)
    return key.replace("pyrophosphate", "diphosphate")


def term_keys(name: str, aliases: Iterable[str] = ()) -> Set[str]:
    keys = {strict_key(n) for n in (name, *aliases)}
    keys.discard("")
    return keys


def text_key(text: Any) -> str:
    """A passage folded with the SAME rules as a name, padded for token search."""

    return f" {strict_key(text)} "


def mentions(name: str, keyed_text: str, aliases: Iterable[str] = ()) -> bool:
    """Whether a passage (already passed through :func:`text_key`) names ``name`` or an
    alias on whole-token boundaries. For finding a name in PROSE -- a retrieved span,
    a paper -- never for grading a participant, which is the evaluation matcher's
    ``matches``."""

    return any(k and f" {k} " in keyed_text for k in term_keys(name, aliases))


__all__ = ["mentions", "strict_key", "text_key"]

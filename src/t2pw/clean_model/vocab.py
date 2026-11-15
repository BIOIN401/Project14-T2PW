"""Controlled interaction vocabulary with an explicit unsigned class.

The payload's ``relationship`` is free text (48 spellings over the 152 canonical runs). ``classify`` maps it
to ``(kind, sign)``. A sign is given only when the word states it: Activation/Inhibition are read from
``t2pw.pwml.interaction_types`` (read-only reuse), plus a few more words that state a sign outright
(``inactivates``, ``allosteric inhibitor of``). Never a guessed sign for regulates / modulates /
competition / binding (D-104 Amendment 1). Unknown words become ``other_unsigned``; the raw text is kept
by the caller.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from t2pw.pwml.interaction_types import pwml_interaction_type

KINDS = (
    "activation", "inhibition", "phosphorylation", "dephosphorylation", "binding", "association",
    "complex_formation", "ubiquitination", "degradation", "expression_regulation", "modification",
    "catalysis", "cofactor", "transport", "competition", "modulation", "other_unsigned",
)
SIGNS = ("positive", "negative", "unsigned")

# normalised word (casefold, _ and - read as spaces) -> (kind, sign)
_TABLE: Dict[str, Tuple[str, str]] = {}


def _add(kind: str, sign: str, *words: str) -> None:
    for w in words:
        _TABLE[w] = (kind, sign)


_add("inhibition", "negative", "inactivates", "inactivation", "allosteric inhibitor of", "inhibitor of")
_add("binding", "unsigned", "binding", "binds", "bind", "binds to", "bound to", "noncovalently bound to")
_add("association", "unsigned", "interacts with", "interacts", "interaction", "may interact with",
     "associates with", "co localizes with", "colocalizes with")
_add("complex_formation", "unsigned", "component of", "subunit of", "part of", "forms complex with",
     "complex formation")
_add("phosphorylation", "unsigned", "phosphorylates", "phosphorylation")
_add("dephosphorylation", "unsigned", "dephosphorylates", "dephosphorylation")
_add("ubiquitination", "unsigned", "ubiquitinates", "ubiquitination")
_add("degradation", "unsigned", "degrades", "degradation")
_add("expression_regulation", "unsigned", "regulates expression", "regulates transcription",
     "transcriptional regulation", "promoter of")
_add("modification", "unsigned", "modifies", "modification", "phosphopantetheinylates",
     "phosphopantetheinylation", "methylates", "acetylates", "glycosylates", "hydroxylates")
_add("catalysis", "unsigned", "catalyzes", "catalyses", "catalyzes hydrolysis of")
_add("cofactor", "unsigned", "cofactor of", "cofactor")
_add("transport", "unsigned", "transports", "aids transport of")
_add("competition", "unsigned", "competes with")
_add("modulation", "unsigned", "regulates", "regulation", "modulates", "modulation", "synergizes with",
     "affects", "influences")
_add("other_unsigned", "unsigned", "same as", "requires", "substrate of", "provides substrate for")  # not catalysis: the substrate does not catalyse


def normalise(raw: Any) -> str:
    return re.sub(r"[\s_\-]+", " ", str(raw or "")).casefold().strip(" \"'`.")


def classify(raw: Any) -> Tuple[str, str]:
    """``(kind, sign)`` for a relationship word; unmapped words are ``("other_unsigned", "unsigned")``."""
    typed = pwml_interaction_type(raw)  # Activation / Inhibition only when the sign is in the word
    if typed == "Activation":
        return "activation", "positive"
    if typed == "Inhibition":
        return "inhibition", "negative"
    key = normalise(raw)
    typed = pwml_interaction_type(key)  # hyphen/underscore spellings
    if typed:
        return ("activation", "positive") if typed == "Activation" else ("inhibition", "negative")
    return _TABLE.get(key, ("other_unsigned", "unsigned"))


def is_mapped(raw: Any) -> bool:
    return bool(pwml_interaction_type(raw)) or bool(pwml_interaction_type(normalise(raw))) or normalise(raw) in _TABLE

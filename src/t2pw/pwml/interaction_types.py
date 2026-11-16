"""Which interaction words PathWhiz can store, and how a row's word is read (ORCH-749, ORCH-754b).

Standard library only, so a presentation module (``app.reaction_evidence``) can check that an
IR row belongs to a payload row without importing the exporter. ``t2pw.pwml.ir`` re-exports
every name here; the table and the functions are the ones that lived in ``ir.py``, moved
verbatim.
"""

from __future__ import annotations

import re
from typing import Any, Dict


def _canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


#: The only interaction types PathWhiz stores (``interactions.interaction_type``
#: holds 11085 Activation and 4998 Inhibition rows and nothing else), keyed by
#: the spellings the pipeline really writes for them. ORCH-747 accepted the two
#: nouns alone, and no committed interaction is spelled that way -- they are verb
#: forms (``inhibits``, ``represses``, ``activates``, ...), so everything was
#: withheld. ORCH-749 adds the verb forms whose SIGN IS IN THE WORD: ``inhibits``
#: is Inhibition in verb form, not a guessed sign. A word that carries no sign
#: (``binds``, ``interacts``, ``modifies``, ``SAME_AS``, ``promoter_of``) is not
#: here and stays withheld: PathBank has no neutral interaction type, and reading
#: a sign into such a word would be the exporter repairing biology after the
#: freeze (merge rule 8). The lookup is an exact match on
#: :func:`_interaction_type_key` -- no stemming, no fuzzy matching -- so what
#: exports is always readable from this table. Census over the committed runs:
#: ``docs/pwml_recovery_sprint/evidence/orch749_interaction_spelling_census.json``.
PWML_INTERACTION_TYPES = {
    "activation": "Activation",
    "activates": "Activation",
    "stimulates": "Activation",
    "induces": "Activation",
    "upregulates": "Activation",
    "up-regulates": "Activation",
    "enhances": "Activation",
    "promotes": "Activation",
    "inhibition": "Inhibition",
    "inhibits": "Inhibition",
    "represses": "Inhibition",
    "repression": "Inhibition",
    "blocks": "Inhibition",
    "suppresses": "Inhibition",
    "downregulates": "Inhibition",
    "down-regulates": "Inhibition",
    "antagonizes": "Inhibition",
    # ``competes with`` / ``competes_with`` were here until ORCH-754b. Competition is
    # symmetric: the word names no direction, so drawing it as an Inhibition from
    # entity_1 to entity_2 was the exporter choosing one after the freeze (merge
    # rule 8). They are withheld as untyped now, with their word kept.
    "prevents": "Inhibition",
}


def _interaction_type_key(raw_type: Any) -> str:
    """The :data:`PWML_INTERACTION_TYPES` key for a relationship as written.

    Casefolded, whitespace collapsed, surrounding quotes and a trailing period
    dropped -- and nothing else. The lookup that follows is exact.
    """
    return re.sub(r"\s+", " ", str(raw_type or "")).casefold().strip(" \"'`“”‘’.")


def interaction_relationship_word(raw: Dict[str, Any]) -> str:
    """The relationship word an interaction row carries, read the way the exporter reads it."""
    return _canonical(raw.get("interaction_type") or raw.get("relationship") or raw.get("class") or "interaction")


def pwml_interaction_type(raw_type: Any) -> str:
    """``Activation`` / ``Inhibition`` when the sign is in the word, else ``""``.

    Signed types only. Which interactions reach the PWML is
    :func:`pwml_export_interaction_type` (L4, D-W26), which adds the unsigned types.
    """
    return PWML_INTERACTION_TYPES.get(_interaction_type_key(raw_type), "")


#: L4 (D-W26; PathWhiz ``t2pw/05-interaction-types``). The patched PathWhiz also stores five types
#: that carry NO sign: Binding (physical association, undirected), Regulation (directed influence,
#: sign not stated), Catalysis (left catalyses something about right), Modification (left covalently
#: modifies right) and Unspecified (the source says the two interact, not how). Exact match on
#: :func:`_interaction_type_key`, like :data:`PWML_INTERACTION_TYPES`; a word in neither table
#: (``component_of``, ``substrate_of``, ``transports``, ``degrades``, ``cofactor_of``, ...) stays
#: withheld from the PWML and is kept in the text record, our SBML and the web diagram. Kept apart
#: from the signed table on purpose: :func:`pwml_interaction_type` still answers "is the sign in the
#: word" for its readers (``clean_model.vocab``, the Stage-1 pooling key), which must never read a
#: sign into ``binds``.
PWML_UNSIGNED_INTERACTION_TYPES = {
    "binding": "Binding",
    "binds": "Binding",
    "binds to": "Binding",
    "binds_to": "Binding",
    "noncovalently_bound_to": "Binding",
    "regulates": "Regulation",
    "regulation": "Regulation",
    "modulates": "Regulation",
    "catalyzes": "Catalysis",
    "modifies": "Modification",
    "interaction": "Unspecified",
    "interacts with": "Unspecified",
    "interacts_with": "Unspecified",
    "interacts": "Unspecified",
    "associates with": "Unspecified",
    "competes_with": "Unspecified",
}

#: Edge ``visualization-template-id`` for both edges of an interaction, by type: PathWhiz's own
#: editor defaults (``interaction.js.coffee`` ``edgeTemplates`` on ``t2pw/05``).
PWML_INTERACTION_EDGE_TEMPLATES = {
    "Activation": 149,
    "Inhibition": 148,
    "Binding": 195,
    "Unspecified": 195,
    "Regulation": 219,
    "Catalysis": 5,
    "Modification": 218,
}


def pwml_export_interaction_type(raw_type: Any) -> str:
    """The PathWhiz ``interaction-type`` a relationship word is written as, else ``""``.

    Signed words first (:func:`pwml_interaction_type`), then the unsigned table. This is the
    one statement of which interactions reach the PWML; Stage 6 reads it to decide which
    protein endpoints get the single-protein wrapper.
    """
    key = _interaction_type_key(raw_type)
    return PWML_INTERACTION_TYPES.get(key) or PWML_UNSIGNED_INTERACTION_TYPES.get(key, "")


#: Withheld reason / review reason for a row that states no relationship word at all (L4 ruling).
RELATIONSHIP_TYPE_NOT_STATED = "relationship_type_not_stated"


def relationship_stated(raw: Dict[str, Any]) -> bool:
    """Does the row state a relationship word? :func:`interaction_relationship_word` falls back to
    ``interaction`` (Unspecified) when it does not; such a row is drawn only with paper evidence."""
    return any(_canonical(raw.get(f)) for f in ("interaction_type", "relationship", "class"))

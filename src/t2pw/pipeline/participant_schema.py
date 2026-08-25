"""C-089 -- ONE definition of what a participant KEY and a participant SLOT are.

Ruling 7: *"define or reuse one canonical participant-key source; have both
readers consume it; avoid duplicating another narrower tuple."* This module is
that source. It is a **stdlib-only leaf**: it imports nothing, not even from
``t2pw``, so ``mapping`` and ``bench`` can both read it without dragging pydantic
into ``identity_admission`` (whose purity is a stated design promise) or the
XML/SBML parser in ``pipeline.canonical`` into the benchmark.

Why not reuse ``canonical.py:330``
----------------------------------
It looks like the canonical source, and it is not. It is private, unexported,
has a single consumer, and **it omits** ``element`` --
``ProcessParticipantModel.element`` and ``ElementWithStateModel.element``.
Measured by REV-086 over 390 committed payload files: **416 dict-shaped
participant entries carry ``element`` and nothing else**, every one of them in
``transports.elements_with_states``. Reconciling to that tuple verbatim would
leave all 416 unreadable, including the slot where F-125's three invisible
orphans live. The reconciliation target here is instead the
``payload_models.py`` union.

Two constants, not one
----------------------
:data:`PARTICIPANT_NAME_KEYS` alone fixes how a participant entry is *read* and
leaves both readers blind to ``elements_with_states``, which is *where* the
orphans are. :data:`PARTICIPANT_SLOTS` fixes the second half.

The legacy tails are additive on purpose
----------------------------------------
``ref`` and ``id`` (name keys) and the members of
:data:`PARTICIPANT_LEGACY_SLOTS` appear in **no** model. They are kept, and kept
*separate*, because:

* ``payload_models`` declares ``model_config = ConfigDict(extra="allow")``, so a
  key absent from a model can still legitimately reach a reader at runtime;
* **dropping a slot is the dangerous direction in both consumers.** In
  ``identity_admission`` a slot the reader stops seeing turns a used cofactor
  into an unused one and **strips a correct accession** -- F-119's stated failure
  direction. In ``bench.semantic`` it **hides a referential-integrity
  violation** from acceptance priority 3, which is weakening a gate.

So this card *widens* both readers and narrows neither. The schema/legacy split
is what keeps the four corrections REV-086 measured visible and testable:
``modifiers`` is not a ``TransportModel`` field, and ``inputs``, ``outputs``,
``cargo`` and ``transporters`` are not ``ReactionCoupledTransportModel`` fields,
though ``identity_admission.PARTICIPANT_FIELDS`` listed all five as if they were.

What is deliberately NOT here
-----------------------------
* ``processes.interactions`` and its ``entity_1`` / ``entity_2`` /
  ``participants`` slots. Whether an interaction endpoint confers a participant
  role is **D-069 / F-128 / C-091**, a separate ruling on *which buckets confer a
  role*. This module answers *what a key and a slot are*. Adding ``interactions``
  here would implement C-091 as a side effect of a reconciliation.
* ``ReactionCoupledTransportModel.reaction`` and ``.transport``. They are real
  string references, but they name **processes, not entities**; a dangling one is
  a different orphan class and must not be resolved against the entity registry.
  Named in :data:`PROCESS_NAME_SLOTS` so that a later reader cannot mistake them
  for participant slots by omission.

Pure, offline, deterministic: no imports beyond ``typing``, no I/O, no state.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = [
    "PARTICIPANT_NAME_KEYS",
    "PARTICIPANT_SCHEMA_NAME_KEYS",
    "PARTICIPANT_LEGACY_NAME_KEYS",
    "PARTICIPANT_SLOTS",
    "PARTICIPANT_LEGACY_SLOTS",
    "ENZYME_ROLE_SLOTS",
    "PROCESS_NAME_SLOTS",
    "participant_slots",
]

#: Every key ``payload_models.py`` declares as carrying an ENTITY NAME on a
#: participant entry: the union of ``ProcessParticipantModel`` (``name``,
#: ``entity``, ``compound``, ``protein``, ``protein_complex``, ``element``,
#: ``element_collection``, ``nucleic_acid``), ``ActorModel`` (``entity``,
#: ``protein``, ``protein_complex``, ``name``) and ``ElementWithStateModel``
#: (``element``).
#:
#: Ordering is significant for the readers that take the FIRST key present
#: (``bench.semantic._names``): ``name`` then ``entity`` first preserves the
#: order those readers already used, so no existing shape changes hands.
PARTICIPANT_SCHEMA_NAME_KEYS: Tuple[str, ...] = (
    "name",
    "entity",
    "compound",
    "protein",
    "protein_complex",
    "element",
    "element_collection",
    "nucleic_acid",
)

#: Keys ``identity_admission`` has always carried that appear in NO model. Kept
#: as an additive tail, never promoted into the union above: a reader that stops
#: seeing them can only start stripping identities.
PARTICIPANT_LEGACY_NAME_KEYS: Tuple[str, ...] = ("ref", "id")

#: The one participant-key list. Both readers consume THIS.
PARTICIPANT_NAME_KEYS: Tuple[str, ...] = (
    PARTICIPANT_SCHEMA_NAME_KEYS + PARTICIPANT_LEGACY_NAME_KEYS
)

#: Per-bucket ENTITY-name-bearing slots, exactly as ``payload_models.py``
#: declares them. Every name here is a declared field on that bucket's model --
#: pinned by a schema-conformance test so the corrections below cannot silently
#: regress.
#:
#: ``reactions``                    -> ``ReactionModel``   (:355-366)
#: ``transports``                   -> ``TransportModel``  (:381-390)
#: ``reaction_coupled_transports``  -> ``ReactionCoupledTransportModel`` (:369-378)
PARTICIPANT_SLOTS: Dict[str, Tuple[str, ...]] = {
    "reactions": ("inputs", "outputs", "enzymes", "modifiers"),
    "transports": ("cargo", "transporters", "elements_with_states"),
    "reaction_coupled_transports": ("modifiers", "enzymes", "elements_with_states"),
}

#: Slots a reader in this repository already read on that bucket, which the
#: bucket's model does **not** declare. Retained so this card widens and never
#: narrows; separated so the mismatch is a fact a test can assert rather than a
#: belief buried in a tuple.
#:
#: * ``reactions.catalysts`` -- read by ``bench.semantic._enzyme_names``.
#: * ``transports.modifiers`` -- listed by ``PARTICIPANT_FIELDS["transports"]``;
#:   ``TransportModel`` has no ``modifiers``.
#: * ``transports.inputs`` / ``.outputs`` / ``.enzymes`` -- read today because
#:   ``_orphaned_references`` applied one uniform slot list to every bucket.
#: * ``reaction_coupled_transports.inputs`` / ``.outputs`` / ``.cargo`` /
#:   ``.transporters`` -- listed by ``PARTICIPANT_FIELDS``; none exist on
#:   ``ReactionCoupledTransportModel``. This is F-119's "fields that do not
#:   exist", confirmed.
PARTICIPANT_LEGACY_SLOTS: Dict[str, Tuple[str, ...]] = {
    "reactions": ("catalysts",),
    "transports": ("inputs", "outputs", "enzymes", "modifiers", "catalysts"),
    "reaction_coupled_transports": (
        "inputs", "outputs", "cargo", "transporters", "catalysts",
    ),
}

#: Slots whose occupants act ON a process rather than being consumed by it.
#: ``enzymes`` and ``modifiers`` are declared model fields; ``catalysts`` is a
#: legacy tail key present in no model.
ENZYME_ROLE_SLOTS: Tuple[str, ...] = ("enzymes", "modifiers", "catalysts")

#: Slots holding a PROCESS name, not an entity name. Enumerated so they are
#: excluded on the record instead of by oversight -- resolving one of these
#: against the entity registry would report every well-formed coupled transport
#: as an orphan.
PROCESS_NAME_SLOTS: Dict[str, Tuple[str, ...]] = {
    "reaction_coupled_transports": ("reaction", "transport"),
}


def participant_slots(bucket: str, *, include_legacy: bool = True) -> Tuple[str, ...]:
    """Slots to read on ``bucket``; schema slots first, legacy tail after.

    ``include_legacy=False`` gives the strictly schema-conformant list. An
    unknown bucket gives ``()`` -- notably ``"interactions"``, which is C-091's
    to rule on, not this module's to leak in.
    """

    slots = PARTICIPANT_SLOTS.get(bucket, ())
    if include_legacy:
        slots = slots + PARTICIPANT_LEGACY_SLOTS.get(bucket, ())
    return slots

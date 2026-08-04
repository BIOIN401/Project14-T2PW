"""When an entity is substituted, repoint the references that named it.

THE DEFECT, MEASURED
--------------------
``runs/2026-08-02_2130/papers/PMC12856317/strict/final_mapped.json``::

    entities.proteins          : ['Unknown']          uniprot="Unknown", pathbank_protein_id=9659
    entities.protein_complexes : ['ALAS2 homodimer']  components=['Unknown']
    processes.interactions     : 4 rows, every entity_2 == 'ALAS2'

    GATE ERROR /processes | Registry validation failed:
      /processes/interactions/{0,1,2,3}/entity_2 unknown entity: ALAS2

WHAT ACTUALLY HAPPENED (traced and reproduced, not assumed)
-----------------------------------------------------------
The writer is **``_rewrite_reaction_protein_enzymes_to_complexes``**
(``map_ids.py:6010``), not the Unknown fallback, and not the export-policy census.

That leg's merged payload lists the ALAS2 reaction with *two* enzyme actors --
``{protein: ALAS2}`` and ``{protein_complex: ALAS2 homodimer}`` -- because the
paper states both. In ``_rewrite_rows``, ``resolved_protein_count`` counts the
complex as resolved, so the rule "drop an unresolved protein actor when a
resolved one remains in the same list" fires and ``ALAS2`` is deleted from
``entities.proteins`` outright. Everything else follows: the wrapper's component
becomes the ``Unknown`` sentinel, and the four interactions keep naming a protein
that no bucket declares any more.

Reproduced offline from ``runs/2026-08-02_2130``'s own ``merged_payload.json``,
with the resolvers stubbed unmapped -- normalize, then map -- so this is measured
rather than inferred. Two natural readings of the evidence are wrong and cost
time, so they are recorded here:

* It is *not* ``_apply_pathbank_unknown_enzyme_fallback`` skipping a check it
  should have made. That pass has a guard, ``_has_disqualifying_reference``,
  which does read interactions and does retain a protein they reference. In the
  reproduction it retains ``ALAS2`` correctly -- and never gets the chance,
  because the actor is gone before the fallback is reached.
* ``ALAS2`` is not renamed and not quarantined. It leaves no row in
  ``quarantined_proteins``, which is why "what removed it?" is not answerable
  from the artifacts alone.

The Unknown fallback *does* have its own version of this seam -- its reuse branch
removes a superseded wrapper component -- and it is repaired through this module
too. Its direct-wrap branch needs no repair, because there the generated wrapper
takes the protein's own display name and a reference to ``X`` still resolves, to
the complex ``X``.

THE CORRECT TRANSFORMATION
--------------------------
::

    protein entity      : ALAS2 -> Unknown              (the sentinel; identity is gone)
    complex component   : ALAS2 -> Unknown              (already repaired, map_ids.py:7178)
    functional process  : ALAS2 -> ALAS2 homodimer      (the WRAPPER, not the sentinel)

Never repoint a functional reference at ``Unknown``. The sentinel is a single
shared row: several unresolved actors can be backed by it at once, so sending
``heme inhibits ALAS2`` to ``Unknown`` would not merely lose which protein was
meant, it would silently merge that claim with every other unresolved actor's.
The wrapper still names one specific unresolved actor; the sentinel names none.

WHY THE FIELD LIST IS NOT WRITTEN OUT HERE
-------------------------------------------
:data:`REFERENCE_FIELDS` mirrors :func:`process_normalizer.validate_registry_references`
field for field, INCLUDING the aliases it accepts: an interaction side may arrive
as ``left``/``entity_1``/``source``, and a transport's cargo as ``cargo`` or
``cargo_complex``. A repair pass that rewrote ``entity_1`` while the payload
carried ``left`` would validate clean and repair nothing -- the validator reads
whichever alias is present, so the repair must write whichever alias is present.
``entities.nucleic_acids`` is in that validator's registry too, which is why a
reference to one is a legitimate target here rather than an unknown name.

FAILING RATHER THAN GUESSING
----------------------------
Two records disagreeing about one name, a replacement that does not name exactly
one registry entry, a replacement that is the sentinel itself: all raise
:class:`ReferenceRepairError`. An unresolved actor with **no** wrapper is not an
error -- it is simply not repairable, so its references are left alone and strict
validation fails on them clearly, which is a better outcome than a payload that
validates while meaning something else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "REFERENCE_FIELDS",
    "ReferenceRepairError",
    "ReferenceRewrite",
    "ReplacementRecord",
    "repair_entity_references",
]


# ── Normalization, matched to the validator's ──────────────────────────────
# Copied rather than imported: ``process_normalizer`` imports nothing from this
# module today and this module must stay importable from ``mapping.map_ids``,
# which is upstream of the normalizer in the dependency order. The two are
# pinned equal by a test rather than by an import.


def _canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", (str(value or "")).strip())


def _normalize(value: Any) -> str:
    lowered = re.sub(r"\s+", " ", (str(value or "")).strip().casefold())
    return re.sub(r"[^a-z0-9:+ ]+", "", lowered)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


#: The actor-name keys, in the order ``_actor_name_from_row`` consults them. The
#: repair writes back to the key it read from, so an actor spelled ``protein``
#: does not silently acquire a second, conflicting ``entity``.
_ACTOR_NAME_KEYS: Tuple[str, ...] = (
    "entity",
    "protein",
    "protein_name",
    "protein_complex",
    "enzyme",
    "modifier",
    "name",
)

#: Interaction sides, each with every alias the validator accepts. First key
#: present wins -- the same precedence ``validate_registry_references`` uses.
_INTERACTION_SIDES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("entity_1", ("left", "entity_1", "source")),
    ("entity_2", ("right", "entity_2", "target")),
)

#: Transport cargo, same precedence as the validator: ``cargo_complex`` when it
#: carries a non-empty string, otherwise ``cargo``.
_CARGO_KEYS: Tuple[str, ...] = ("cargo_complex", "cargo")

#: Human-readable inventory of what is traversed. Pinned against
#: ``validate_registry_references`` by test, so a field added there and not here
#: fails rather than quietly going unrepaired.
REFERENCE_FIELDS: Tuple[str, ...] = (
    "processes.reactions[].inputs[]",
    "processes.reactions[].outputs[]",
    "processes.reactions[].enzymes[]",
    "processes.reactions[].modifiers[]",
    "processes.transports[].cargo",
    "processes.transports[].cargo_complex",
    "processes.transports[].transporters[]",
    "processes.interactions[].entity_1|left|source",
    "processes.interactions[].entity_2|right|target",
)


class ReferenceRepairError(ValueError):
    """The repair refused to guess. Never raised for "nothing to repair"."""


@dataclass(frozen=True)
class ReplacementRecord:
    """One substitution event, as the code that performed it saw it.

    ``old_name``               the entity that was removed or renamed.
    ``sentinel_name``          what its ENTITY row became (``Unknown``). Recorded
                               so the repair can refuse to repoint a functional
                               reference at it, and so the event is auditable.
    ``functional_replacement`` the wrapper that now carries the actor's function
                               -- what process references must point at. Empty
                               means "there is no wrapper", which is a skip and
                               not an error.
    ``reason``                 why the substitution happened, e.g.
                               ``pathbank_unknown_protein_fallback``.
    """

    old_name: str
    sentinel_name: str
    functional_replacement: str
    reason: str


@dataclass(frozen=True)
class ReferenceRewrite:
    """One rewritten reference. Every field is required for the audit trail."""

    pointer: str
    old_value: str
    new_value: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "json_pointer": self.pointer,
            "old_value": self.old_value,
            "replacement_value": self.new_value,
            "replacement_reason": self.reason,
        }


def _registry_norms(payload: Mapping[str, Any]) -> Dict[str, List[str]]:
    """``{normalized name: [display names]}`` over the same buckets the
    validator's registry is built from, ``nucleic_acids`` included."""

    entities = _safe_dict(payload.get("entities"))
    found: Dict[str, List[str]] = {}
    for bucket in ("compounds", "proteins", "protein_complexes", "nucleic_acids"):
        for row in _safe_list(entities.get(bucket)):
            if not isinstance(row, dict):
                continue
            name = _canonical(row.get("name"))
            if name:
                found.setdefault(_normalize(name), []).append(name)
    return found


def _plan(
    payload: Mapping[str, Any],
    replacements: Sequence[ReplacementRecord],
) -> Dict[str, ReplacementRecord]:
    """``{normalized old name: record}``, after refusing every ambiguity."""

    registry = _registry_norms(payload)
    plan: Dict[str, ReplacementRecord] = {}
    for record in replacements:
        old_norm = _normalize(record.old_name)
        if not old_norm:
            continue
        replacement = _canonical(record.functional_replacement)
        sentinel_norm = _normalize(record.sentinel_name)

        existing = plan.get(old_norm)
        if existing is not None:
            if _normalize(existing.functional_replacement) != _normalize(replacement):
                # Two substitutions claiming the same name for different
                # wrappers. Picking one would attribute a process to whichever
                # record happened to be enumerated first.
                raise ReferenceRepairError(
                    f"ambiguous replacement for {record.old_name!r}: "
                    f"{existing.functional_replacement!r} vs {replacement!r}"
                )
            continue

        if not replacement:
            # No wrapper. Not an error: leaving the reference alone lets strict
            # validation fail on it clearly, which says "this actor is
            # unresolved" instead of quietly attributing it to the sentinel.
            continue
        replacement_norm = _normalize(replacement)
        if replacement_norm == old_norm:
            continue  # a rename to itself is a no-op, not a rewrite
        if sentinel_norm and replacement_norm == sentinel_norm:
            raise ReferenceRepairError(
                f"refusing to repoint references to {record.old_name!r} at the "
                f"sentinel {record.sentinel_name!r}: the sentinel is shared, so "
                "the process would no longer name which actor it meant"
            )
        names = registry.get(replacement_norm, [])
        if len(names) != 1:
            # Zero: the wrapper is not in the registry, so the rewrite would
            # trade one unknown reference for another. More than one: the name
            # does not identify a single entity.
            raise ReferenceRepairError(
                f"replacement {replacement!r} for {record.old_name!r} names "
                f"{len(names)} registry entries; exactly one is required"
            )
        plan[old_norm] = ReplacementRecord(
            old_name=_canonical(record.old_name),
            sentinel_name=_canonical(record.sentinel_name),
            # The registry's spelling, so the rewritten reference matches the
            # declared entity byte for byte rather than merely normalizing equal.
            functional_replacement=names[0],
            reason=record.reason,
        )
    return plan


def _rewrite_token(
    container: Any,
    key: Any,
    pointer: str,
    plan: Mapping[str, ReplacementRecord],
    rewrites: List[ReferenceRewrite],
) -> None:
    """Rewrite one string-valued reference in place, if it is planned."""

    value = container[key] if isinstance(container, list) else container.get(key)
    if not isinstance(value, str):
        return
    canonical = _canonical(value)
    if not canonical:
        return
    record = plan.get(_normalize(canonical))
    if record is None:
        return
    container[key] = record.functional_replacement
    rewrites.append(
        ReferenceRewrite(
            pointer=pointer,
            old_value=canonical,
            new_value=record.functional_replacement,
            reason=record.reason,
        )
    )


def _rewrite_actor(
    actor: Any,
    pointer: str,
    plan: Mapping[str, ReplacementRecord],
    rewrites: List[ReferenceRewrite],
) -> Optional[str]:
    """Rewrite an actor row's name key, or return the replacement for a bare
    string actor so the caller can store it back into its list."""

    if isinstance(actor, str):
        canonical = _canonical(actor)
        record = plan.get(_normalize(canonical))
        if record is None or not canonical:
            return None
        rewrites.append(
            ReferenceRewrite(
                pointer=pointer,
                old_value=canonical,
                new_value=record.functional_replacement,
                reason=record.reason,
            )
        )
        return record.functional_replacement
    if not isinstance(actor, dict):
        return None
    for name_key in _ACTOR_NAME_KEYS:
        if not _canonical(actor.get(name_key, "")):
            continue
        # Write back to the key the name was READ from. An actor spelled
        # ``protein`` that acquired an ``entity`` instead would leave the old
        # value in place and the validator reading the stale one.
        _rewrite_token(actor, name_key, f"{pointer}/{name_key}", plan, rewrites)
        return None
    return None


def repair_entity_references(
    payload: Dict[str, Any],
    replacements: Sequence[ReplacementRecord],
) -> Dict[str, Any]:
    """Repoint every functional process reference named in ``replacements``.

    Mutates ``payload`` in place and returns the audit trail. Idempotent: after
    one pass no reference carries an ``old_name`` any more, so a second pass
    finds nothing and rewrites nothing.

    Entity rows and complex components are deliberately NOT touched. Those go to
    the sentinel and are the substituting code's own business; this pass exists
    for the references that point *at* the substituted entity from elsewhere, and
    those must go to the wrapper instead.
    """

    rewrites: List[ReferenceRewrite] = []
    plan = _plan(payload, replacements)
    if not plan:
        return {
            "rewrites": [],
            "summary": {"references_rewritten": 0, "replacements_applied": 0},
        }

    processes = _safe_dict(payload.get("processes"))

    for ridx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(reaction, dict):
            continue
        for side in ("inputs", "outputs"):
            tokens = _safe_list(reaction.get(side))
            for tidx in range(len(tokens)):
                _rewrite_token(
                    tokens, tidx, f"/processes/reactions/{ridx}/{side}/{tidx}", plan, rewrites
                )
        for actor_key in ("enzymes", "modifiers"):
            actors = _safe_list(reaction.get(actor_key))
            for aidx, actor in enumerate(actors):
                replaced = _rewrite_actor(
                    actor, f"/processes/reactions/{ridx}/{actor_key}/{aidx}", plan, rewrites
                )
                if replaced is not None:
                    actors[aidx] = replaced

    for tidx, transport in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(transport, dict):
            continue
        for cargo_key in _CARGO_KEYS:
            if _canonical(transport.get(cargo_key, "")):
                _rewrite_token(
                    transport, cargo_key, f"/processes/transports/{tidx}/{cargo_key}", plan, rewrites
                )
                # ``cargo_complex`` wins when present, exactly as the validator
                # reads it; ``cargo`` is only consulted when it does not.
                break
        transporters = _safe_list(transport.get("transporters"))
        for xidx, actor in enumerate(transporters):
            replaced = _rewrite_actor(
                actor, f"/processes/transports/{tidx}/transporters/{xidx}", plan, rewrites
            )
            if replaced is not None:
                transporters[xidx] = replaced

    for iidx, interaction in enumerate(_safe_list(processes.get("interactions"))):
        if not isinstance(interaction, dict):
            continue
        for _side_name, aliases in _INTERACTION_SIDES:
            for alias in aliases:
                if not _canonical(interaction.get(alias, "")):
                    continue
                _rewrite_token(
                    interaction, alias, f"/processes/interactions/{iidx}/{alias}", plan, rewrites
                )
                break

    applied = {
        _normalize(rewrite.old_value)
        for rewrite in rewrites
    }
    return {
        "rewrites": [rewrite.to_dict() for rewrite in rewrites],
        "summary": {
            "references_rewritten": len(rewrites),
            "replacements_applied": len(applied),
            "replacements_planned": len(plan),
        },
    }

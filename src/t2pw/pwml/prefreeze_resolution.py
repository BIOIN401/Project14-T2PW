"""Run compound canonicalization BEFORE the canonical payload is frozen.

``DECISIONS.md`` **D-015 (LOCKED)** rules that compound name and identity
canonicalization is part of the canonical biological representation and must
happen deterministically *before* the freeze. ``PRODUCT_CONTRACT.md`` §5 states
the same prohibition from the other side: an exporter must not "add, remove,
resolve or reinterpret biological content after the canonical graph is frozen".

C-040 made the machinery callable from before the freeze
(:mod:`t2pw.pwml.compound_resolution`). **This module is the caller.** It
re-implements none of it: it decides *when* the resolution runs, builds the
explicit rename map that resolution implies, and discharges the one obligation
C-040 documented as the caller's and could not perform itself.

Why the caller has to do the propagation
----------------------------------------
``compound_resolution`` is handed *rows*, not the payload that references them.
``processes.reactions[].inputs`` and friends hold participant references as plain
**name strings**. ``ir.build_pwml_ir`` absorbs a rename only because its
``entity_by_name`` index carries ``raw_name`` and ``synonyms`` as well as
``name`` -- but ``strict_quarantine``, which runs inside the freeze, keys on
``name``/``synonyms`` alone. A rename applied to ``entities.compounds`` and not
propagated therefore silently prunes the compound and breaks every reaction that
referenced it. D-015 clause 3 says the propagation is **atomic**; that is what
:func:`resolve_compounds_prefreeze` guarantees, and it is why the whole operation
is staged on a copy and committed with an undo log.

Failure is loud (D-015 clause 6)
--------------------------------
Ambiguity and broken connectivity raise :class:`PrefreezeResolutionError`, which
carries a machine-readable ``code``. The payload handed in is left **exactly** as
it was -- there is no partially propagated state to observe, and no silent skip.

Extending this to species (C-045)
---------------------------------
:data:`PREFREEZE_CANONICALIZERS` is an ordered tuple of ``(name, callable)``.
Each callable takes ``(payload, context)`` and returns its own summary block.
C-045 adds a species entry beside the compound one; nothing else has to move, and
the Streamlit call site does not change again.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from t2pw.pwml.compound_resolution import _db_id, _resolve_compound_rows, ensure_resolution_report
from t2pw.pwml.name_index import default_name_index

#: ``ir.py``'s compound ``entity_spec`` db-key list and its ``db_field``. ``ir._entity_record``
#: projects ``_db_id(row, COMPOUND_DB_KEYS)`` onto both of these while building the IR --
#: i.e. **after the freeze**. Materializing the same projection here, from ids the row
#: already carries, is what moves that step upstream instead of leaving it to the exporter.
COMPOUND_DB_KEYS: Tuple[str, ...] = ("pathbank_compound_id", "pw_compound_id", "pathwhiz_id")
COMPOUND_DB_FIELD = "pathbank_compound_id"

#: How many times resolution may be re-applied while hunting its fixed point.
_MAX_RESOLUTION_PASSES = 4

__all__ = [
    "PrefreezeResolutionError",
    "PREFREEZE_CANONICALIZERS",
    "resolve_compounds_prefreeze",
    "run_prefreeze_resolution",
]


# ---------------------------------------------------------------------------
# Normalizers. Byte-identical semantics to ``ir._canonical`` / ``ir._norm`` --
# reference matching MUST agree with the index ``ir.resolve_entity`` consults,
# or a rename this module calls propagated would not be the one the exporter
# resolves.
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _norm(value: Any) -> str:
    text = _canonical(value).casefold()
    return re.sub(r"[^a-z0-9:+ ]+", " ", text).strip()


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


class PrefreezeResolutionError(RuntimeError):
    """A named pre-freeze failure. The payload is never left partly propagated."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details: Dict[str, Any] = details


# ---------------------------------------------------------------------------
# The participant-reference table.
#
# Every key below is a place a *name string* refers to an entity. The key orders
# mirror ``ir._coerce_participants`` and ``ir._actor_name_and_hint`` exactly, but
# this module rewrites **every** matching key rather than only the first
# non-empty one: a shadowed alias left holding a pre-rename name is a stale
# reference, and D-015 clause 3 admits no partial propagation.
# ---------------------------------------------------------------------------

#: ``ir._coerce_participants`` -- reaction inputs/outputs, RCT left/right.
_PARTICIPANT_KEYS: Tuple[str, ...] = (
    "name", "compound", "protein", "protein_complex", "element", "entity",
)
#: ``ir._actor_name_and_hint`` -- enzymes, modifiers, transporters.
_ACTOR_KEYS: Tuple[str, ...] = (
    "protein_complex", "protein-complex", "protein", "entity", "name",
)
#: ``ir``'s transport-element reader.
_CARGO_KEYS: Tuple[str, ...] = ("element", "cargo", "name")
#: ``ir``'s interaction endpoint readers, both sides.
_INTERACTION_KEYS: Tuple[str, ...] = (
    "left", "entity_1", "source", "right", "entity_2", "target",
)

#: bucket -> ((list field, key set), ...) plus scalar fields.
_PROCESS_MEMBER_FIELDS: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    "reactions": (
        ("inputs", _PARTICIPANT_KEYS),
        ("outputs", _PARTICIPANT_KEYS),
        ("enzymes", _ACTOR_KEYS),
        ("modifiers", _ACTOR_KEYS),
    ),
    "transports": (
        ("transport_elements", _CARGO_KEYS),
        ("transporters", _ACTOR_KEYS),
    ),
    "reaction_coupled_transports": (
        ("inputs", _PARTICIPANT_KEYS),
        ("outputs", _PARTICIPANT_KEYS),
        ("left", _PARTICIPANT_KEYS),
        ("right", _PARTICIPANT_KEYS),
        ("transport_elements", _CARGO_KEYS),
        ("enzymes", _ACTOR_KEYS),
        ("modifiers", _ACTOR_KEYS),
    ),
}
#: bucket -> scalar name fields directly on the process object.
_PROCESS_SCALAR_FIELDS: Dict[str, Tuple[str, ...]] = {
    "transports": ("cargo", "cargo_complex"),
    "interactions": _INTERACTION_KEYS,
    "reaction_coupled_transports": ("cargo", "cargo_complex"),
}


class _Ref:
    """One rewritable participant reference: a container, a key, a JSON pointer."""

    __slots__ = ("container", "key", "pointer")

    def __init__(self, container: Any, key: Any, pointer: str) -> None:
        self.container = container
        self.key = key
        self.pointer = pointer

    def get(self) -> str:
        value = self.container[self.key]
        return value if isinstance(value, str) else ""

    def set(self, value: str) -> None:
        self.container[self.key] = value


def _iter_refs(payload: Dict[str, Any]) -> Iterator[_Ref]:
    """Yield every process participant reference that is a plain name string.

    Deterministic order: bucket order as declared above, then index order, then
    field order. The pointer is the identity used by the connectivity signature,
    so this ordering is part of the contract, not an implementation detail.
    """

    processes = _safe_dict(payload.get("processes"))
    for bucket in ("reactions", "transports", "interactions", "reaction_coupled_transports"):
        items = _safe_list(processes.get(bucket))
        for pidx, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            base = f"/processes/{bucket}/{pidx}"
            for field in _PROCESS_SCALAR_FIELDS.get(bucket, ()):
                if isinstance(raw.get(field), str) and _canonical(raw.get(field)):
                    yield _Ref(raw, field, f"{base}/{field}")
            for field, keys in _PROCESS_MEMBER_FIELDS.get(bucket, ()):
                members = raw.get(field)
                if not isinstance(members, list):
                    continue
                for midx, member in enumerate(members):
                    slot = f"{base}/{field}/{midx}"
                    if isinstance(member, str):
                        if _canonical(member):
                            yield _Ref(members, midx, slot)
                        continue
                    if not isinstance(member, dict):
                        continue
                    for key in keys:
                        if isinstance(member.get(key), str) and _canonical(member.get(key)):
                            yield _Ref(member, key, f"{slot}/{key}")


# ---------------------------------------------------------------------------
# Entity identity index.
# ---------------------------------------------------------------------------

#: Alias fields ``ir.build_pwml_ir`` feeds into ``entity_by_name`` (``ir.py:940``).
_PRIMARY_ALIAS_FIELDS: Tuple[str, ...] = ("name", "raw_name", "short_name", "common_name")


def _alias_index(payload: Dict[str, Any]) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, Tuple[str, ...]]]:
    """``_norm(alias) -> entity tokens``, once for primary names, once including synonyms.

    A token is ``"<bucket>#<index>"``: positional, so it survives a rename, which
    is exactly the property the connectivity signature needs. Every list bucket
    under ``entities`` is indexed -- a compound whose new name collides with a
    protein is as much a merge as one that collides with another compound.
    """

    primary: Dict[str, List[str]] = {}
    with_synonyms: Dict[str, List[str]] = {}
    entities = _safe_dict(payload.get("entities"))
    for bucket in sorted(entities):
        rows = entities.get(bucket)
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            token = f"{bucket}#{index}"
            for field in _PRIMARY_ALIAS_FIELDS:
                key = _norm(row.get(field))
                if key:
                    if token not in primary.setdefault(key, []):
                        primary[key].append(token)
                    if token not in with_synonyms.setdefault(key, []):
                        with_synonyms[key].append(token)
            for synonym in _safe_list(row.get("synonyms")) + _safe_list(row.get("aliases")):
                key = _norm(synonym)
                if key and token not in with_synonyms.setdefault(key, []):
                    with_synonyms[key].append(token)
    return (
        {key: tuple(sorted(value)) for key, value in primary.items()},
        {key: tuple(sorted(value)) for key, value in with_synonyms.items()},
    )


def _connectivity_signature(payload: Dict[str, Any], index: Dict[str, Tuple[str, ...]]) -> str:
    """The ``processes`` tree with every reference replaced by what it resolves to.

    Comparing this string before and after proves D-015 clause 5 in full: process
    and participant **counts**, participant **order**, **stoichiometry**, roles,
    and the resolved **identity** of every edge are all inside it. A name that
    changes but still resolves to the same entity produces the same signature; a
    name that stops resolving, or starts resolving to something else, does not.
    """

    projected = deepcopy(_safe_dict(payload.get("processes")))
    holder = {"processes": projected}
    for ref in _iter_refs(holder):
        tokens = index.get(_norm(ref.get()), ())
        ref.set("<<" + "|".join(tokens) + ">>" if tokens else "<<UNRESOLVED>>")
    return json.dumps(projected, sort_keys=True, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# The compound canonicalizer.
# ---------------------------------------------------------------------------

_NAME_INDEX_UNSET = object()


def resolve_compounds_prefreeze(
    payload: Dict[str, Any],
    *,
    db_resolver: Any = None,
    strict_db: bool = False,
    name_index: Any = _NAME_INDEX_UNSET,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonicalize ``payload['entities']['compounds']`` in place, pre-freeze.

    Mutates **the object it is given** -- ``payload``, ``payload['entities']``,
    ``payload['processes']`` and every process dict keep their identity, so a
    caller holding ``final_export_payload`` cannot end up holding a
    pre-resolution view of it. The only objects replaced are the compound row
    dicts themselves, which :func:`_resolve_compound_rows` returns fresh.

    ``db_resolver=None`` and an unset ``name_index`` reproduce exactly what
    ``build_pwml_ir`` passes today, so this call -- not the exporter -- is the
    one that opens the PathBank connection (``PRODUCT_CONTRACT`` §8), and the
    resolution outcome is the same one the exporter would have reached.

    Raises :class:`PrefreezeResolutionError` on an ambiguous rename or a
    reference the rename would break. On any raise the payload is unchanged.
    """

    resolution_report = ensure_resolution_report(report if isinstance(report, dict) else {})
    summary: Dict[str, Any] = {
        "applied": False,
        "rows": 0,
        "renamed": 0,
        "rename_map": {},
        "references_updated": [],
        "aliases_preserved": [],
        "resolution_report": resolution_report,
    }
    if not isinstance(payload, dict):
        summary["skipped_reason"] = "payload_not_a_mapping"
        return summary
    entities = payload.get("entities")
    rows = entities.get("compounds") if isinstance(entities, dict) else None
    if not isinstance(rows, list) or not rows:
        summary["skipped_reason"] = "no_compound_rows"
        return summary

    summary["rows"] = len(rows)
    resolved_name_index = default_name_index() if name_index is _NAME_INDEX_UNSET else name_index

    # ---- stage 1: resolve on a copy. The live rows are still untouched. ----
    staged = deepcopy(rows)
    before_names = [_canonical(row.get("name")) if isinstance(row, dict) else "" for row in staged]
    resolved, passes = _resolve_to_fixed_point(
        staged,
        db_resolver=db_resolver,
        strict_db=bool(strict_db),
        report=resolution_report,
        name_index=resolved_name_index,
    )
    summary["resolution_passes"] = passes
    _project_db_identity(resolved, summary)
    if len(resolved) != len(rows):
        raise PrefreezeResolutionError(
            "PREFREEZE_ROW_COUNT_CHANGED",
            f"compound resolution returned {len(resolved)} rows for {len(rows)} inputs",
            expected=len(rows), actual=len(resolved),
        )
    after_names = [_canonical(row.get("name")) if isinstance(row, dict) else "" for row in resolved]

    # ---- stage 2: the explicit rename map (D-015 clause 2) -----------------
    rename_map: Dict[str, str] = {}
    for index, (before, after) in enumerate(zip(before_names, after_names)):
        if not before or not after or before == after:
            continue
        previous = rename_map.get(before)
        if previous is not None and _norm(previous) != _norm(after):
            raise PrefreezeResolutionError(
                "AMBIGUOUS_RENAME_SOURCE",
                f"compound name {before!r} canonicalizes to both {previous!r} and {after!r}",
                name=before, targets=sorted({previous, after}), row=index,
            )
        rename_map[before] = after
    summary["rename_map"] = dict(rename_map)
    summary["renamed"] = len(rename_map)

    primary_before, alias_before = _alias_index(payload)

    if rename_map:
        _reject_ambiguous_renames(rename_map, before_names, after_names, primary_before)

    # ---- stage 3: propagate on a staged copy of the whole payload ----------
    staged_payload = deepcopy(payload)
    staged_payload["entities"]["compounds"] = resolved
    _preserve_original_names(resolved, before_names, after_names, summary)
    staged_updates = _propagate(staged_payload, rename_map)
    _assert_fully_propagated(staged_payload, rename_map)

    _, alias_after = _alias_index(staged_payload)
    signature_before = _connectivity_signature(payload, alias_before)
    signature_after = _connectivity_signature(staged_payload, alias_after)
    if signature_before != signature_after:
        raise PrefreezeResolutionError(
            "PREFREEZE_CONNECTIVITY_BROKEN",
            "renaming changed the participant edge set; the payload was not modified",
            first_divergence=_first_divergence(signature_before, signature_after),
            rename_map=dict(rename_map),
        )

    # ---- stage 4: commit to the LIVE payload, with an undo log ------------
    undo: List[Tuple[Any, Any, Any]] = [(rows, slice(None), list(rows))]
    try:
        rows[:] = resolved
        for ref in _iter_refs(payload):
            target = rename_map.get(_canonical(ref.get()))
            if target is None:
                continue
            undo.append((ref.container, ref.key, ref.get()))
            ref.set(target)
        _assert_fully_propagated(payload, rename_map)
        _, live_alias = _alias_index(payload)
        if _connectivity_signature(payload, live_alias) != signature_after:
            raise PrefreezeResolutionError(
                "PREFREEZE_COMMIT_DIVERGED",
                "the committed payload did not match the validated staged payload",
                rename_map=dict(rename_map),
            )
    except Exception:
        for container, key, value in reversed(undo):
            container[key] = value
        raise

    summary["applied"] = True
    summary["references_updated"] = [
        {"pointer": pointer, "from": before, "to": after} for pointer, before, after in staged_updates
    ]
    return summary


def _rows_fingerprint(rows: Sequence[Dict[str, Any]]) -> str:
    return json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str)


def _resolve_to_fixed_point(
    rows: List[Dict[str, Any]],
    *,
    db_resolver: Any,
    strict_db: bool,
    report: Dict[str, Any],
    name_index: Any,
) -> Tuple[List[Dict[str, Any]], int]:
    """Apply resolution until re-applying it changes nothing.

    D-015 clause 8 -- "finish **all** network-dependent resolution before the
    freeze" -- is a statement about the *end state*, not about running the
    function once. A single pass does not reach it: pass 1 attaches a ``db_row``
    via the offline index, which makes ``_canonicalize_compound_offline`` return
    early on pass 2, which in turn lets the legacy-id branch's ``db_status``
    stand. So one pass leaves a payload on which the very same resolution would
    still produce a different answer -- and the exporter runs it again.

    Iterating to the fixed point is the only thing that makes "the resolution is
    finished" true, and it is a property of *this* call: it says nothing about
    what any other stage did, so it does not become false when C-051 removes the
    downstream call.
    """

    resolved = rows
    fingerprint = _rows_fingerprint(resolved)
    for attempt in range(1, _MAX_RESOLUTION_PASSES + 1):
        resolved = _resolve_compound_rows(
            resolved,
            db_resolver=db_resolver,
            strict_db=strict_db,
            report=report,
            pointer_prefix="/entities/compounds",
            name_index=name_index,
            # D-015 clauses 2-5 require the rename APPLIED; the suppression
            # adapter attaches identifiers without it and would not discharge
            # them. Propagation is handled by the caller, which is this module.
            apply_canonical_name=True,
        )
        current = _rows_fingerprint(resolved)
        if current == fingerprint:
            return resolved, attempt
        fingerprint = current
    raise PrefreezeResolutionError(
        "PREFREEZE_RESOLUTION_UNSTABLE",
        f"compound resolution did not converge within {_MAX_RESOLUTION_PASSES} passes; "
        "the payload cannot be declared canonical",
        passes=_MAX_RESOLUTION_PASSES,
    )


def _project_db_identity(rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Materialize ``ir._entity_record``'s db-id projection before the hash, not after.

    ``_entity_record`` sets ``pathwhiz_id`` and ``pathbank_compound_id`` to
    ``_db_id(row, COMPOUND_DB_KEYS)`` while building the IR. That is identity
    materialization performed by an exporter on a frozen graph, which
    ``PRODUCT_CONTRACT`` §5 forbids and D-015 places upstream.

    **Nothing is invented** (D-015 clause 7 / A7): ``_db_id`` is C-040's verbatim
    copy of the same helper, the value is read out of ids the row already
    carries, and a row from which it finds nothing is left exactly as it is. No
    lookup, no network, no new external claim.
    """

    projected: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        value = _db_id(row, COMPOUND_DB_KEYS)
        if value is None:
            continue
        for field in ("pathwhiz_id", COMPOUND_DB_FIELD):
            if row.get(field) != value:
                row[field] = value
                projected.append({"row": index, "field": field, "value": value})
    summary["identity_projected"] = projected


def _reject_ambiguous_renames(
    rename_map: Dict[str, str],
    before_names: Sequence[str],
    after_names: Sequence[str],
    primary_before: Dict[str, Tuple[str, ...]],
) -> None:
    """D-015 clause 6, checked before anything is written.

    Two conditions are fatal. **The source name is shared**: some entity other
    than the compounds that all canonicalize alike answers to it, so rewriting
    references named for it would redirect references meant for that entity.
    **Two distinct sources share a target**: the rename would merge two
    compounds into one, which is inventing biology, not canonicalizing it.
    """

    by_target: Dict[str, List[str]] = {}
    for old, new in rename_map.items():
        by_target.setdefault(_norm(new), []).append(old)
    for target, sources in sorted(by_target.items()):
        if len({_norm(source) for source in sources}) > 1:
            raise PrefreezeResolutionError(
                "AMBIGUOUS_RENAME_TARGET",
                f"{len(sources)} distinct compound names canonicalize to {target!r}; "
                "applying the rename would merge them",
                target=target, sources=sorted(sources),
            )

    compatible: Dict[str, set] = {}
    for index, (before, after) in enumerate(zip(before_names, after_names)):
        if before:
            compatible.setdefault(_norm(before), set()).add(f"compounds#{index}")
    for old, new in sorted(rename_map.items()):
        owners = primary_before.get(_norm(old), ())
        rogue = [
            owner for owner in owners
            if owner not in compatible.get(_norm(old), set())
            or _norm(after_names[int(owner.split("#")[1])]) != _norm(new)
        ]
        if rogue:
            raise PrefreezeResolutionError(
                "AMBIGUOUS_REFERENCE",
                f"the name {old!r} is also carried by {rogue}; a reference to it cannot be "
                f"unambiguously redirected to {new!r}",
                name=old, target=new, conflicting_entities=sorted(rogue),
            )


def _preserve_original_names(
    resolved: List[Dict[str, Any]],
    before_names: Sequence[str],
    after_names: Sequence[str],
    summary: Dict[str, Any],
) -> None:
    """D-015 clause 4 / A5 -- keep the supported name reachable.

    ``_canonicalize_compound_offline`` records the extraction name under
    ``raw_name`` and ``aliases``, but the canonical payload's synonym field is
    ``synonyms`` and that is the one ``strict_quarantine`` and the exporter's
    ``entity_by_name`` both read. Adding it there is the caller's job, and it is
    the only field this module writes that resolution did not.
    """

    preserved: List[Dict[str, str]] = []
    for index, row in enumerate(resolved):
        before, after = before_names[index], after_names[index]
        if not isinstance(row, dict) or not before or _norm(before) == _norm(after):
            continue
        synonyms = [value for value in _safe_list(row.get("synonyms")) if isinstance(value, str)]
        if _norm(before) not in {_norm(value) for value in synonyms}:
            synonyms.append(before)
            row["synonyms"] = synonyms
            preserved.append({"name": after, "synonym": before})
    summary["aliases_preserved"] = preserved


def _propagate(payload: Dict[str, Any], rename_map: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """Rewrite every participant reference named by the map. Returns the updates."""

    updates: List[Tuple[str, str, str]] = []
    if not rename_map:
        return updates
    for ref in _iter_refs(payload):
        current = _canonical(ref.get())
        target = rename_map.get(current)
        if target is None:
            continue
        updates.append((ref.pointer, ref.get(), target))
        ref.set(target)
    return updates


def _assert_fully_propagated(payload: Dict[str, Any], rename_map: Dict[str, str]) -> None:
    """No reference may still normalize to a name that was renamed away."""

    stale_norms = {_norm(old): (old, new) for old, new in rename_map.items() if _norm(old) != _norm(new)}
    if not stale_norms:
        return
    for ref in _iter_refs(payload):
        hit = stale_norms.get(_norm(ref.get()))
        if hit is not None:
            raise PrefreezeResolutionError(
                "PREFREEZE_RENAME_NOT_PROPAGATED",
                f"{ref.pointer} still refers to {hit[0]!r}, which was renamed to {hit[1]!r}",
                pointer=ref.pointer, name=hit[0], target=hit[1],
            )


def _first_divergence(before: str, after: str) -> str:
    for index, (left, right) in enumerate(zip(before, after)):
        if left != right:
            start = max(0, index - 60)
            return f"...{before[start:index + 60]}  !=  ...{after[start:index + 60]}"
    return f"length {len(before)} != {len(after)}"


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

#: Ordered canonicalizers. C-045 appends ``("species", resolve_species_prefreeze)``.
PREFREEZE_CANONICALIZERS: Tuple[Tuple[str, Callable[..., Dict[str, Any]]], ...] = (
    ("compounds", resolve_compounds_prefreeze),
)


def run_prefreeze_resolution(
    payload: Dict[str, Any],
    *,
    strict_db: bool = False,
    db_resolver: Any = None,
    name_index: Any = _NAME_INDEX_UNSET,
    canonicalizers: Sequence[Tuple[str, Callable[..., Dict[str, Any]]]] = PREFREEZE_CANONICALIZERS,
) -> Dict[str, Any]:
    """Run every pre-freeze canonicalizer over ``payload``, in order, in place.

    Returns the report. Propagates :class:`PrefreezeResolutionError` -- D-015
    clause 6 is a stop condition, and a canonical payload that cannot be made
    canonical must not reach the freeze wearing the name.
    """

    report: Dict[str, Any] = {
        "stage": "prefreeze_resolution",
        "ok": True,
        "canonicalizers": [name for name, _ in canonicalizers],
    }
    for name, canonicalizer in canonicalizers:
        report[name] = canonicalizer(
            payload,
            db_resolver=db_resolver,
            strict_db=strict_db,
            name_index=name_index,
        )
    return report

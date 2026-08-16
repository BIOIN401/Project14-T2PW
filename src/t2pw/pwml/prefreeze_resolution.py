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

Species too (C-045 / D-016)
---------------------------
:data:`PREFREEZE_CANONICALIZERS` is an ordered tuple of ``(name, callable)``.
Each callable takes ``(payload, context)`` and returns its own summary block.
C-045 added :func:`resolve_species_prefreeze` beside the compound one, for the
reason **D-016 (LOCKED)** gives: organism context is canonical biology as much as
compound identity is, and it was being rewritten inside the exporter, after the
freeze. Nothing else moved, and the Streamlit call site did not change again.
The species ladder itself is ``ir._canonicalize_species_offline``, imported --
this module has no second implementation of it, exactly as it has no second
implementation of compound resolution.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from t2pw.pwml.compound_resolution import _db_id, _resolve_compound_rows, ensure_resolution_report
from t2pw.pwml.ir import (
    PREFREEZE_DB_RESOLUTION_FIELD,
    SPECIES_CANONICALIZATION_FIELD,
    _canonicalize_species_offline,
    _dedupe_aliases,
)
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
    "resolve_species_prefreeze",
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

#: ``element_locations`` bucket -> the name-keyed fields a row may use to name its
#: entity. **These are entity references too.** ``canonical._parse_json``
#: (``canonical.py:391``) reads each row with ``(kind, "entity", "name")`` and
#: ``compound`` is inside the graph-hash allowlist (``canonical_hash.py:88-90``),
#: so a location left holding a pre-rename name dangles *and is hashed into*
#: ``canonical_graph_sha256``. Buckets are exactly
#: ``canonical_hash.GRAPH_SECTIONS["element_locations"]``; all four are here, not
#: just compounds, because C-045 renames through the same propagation.
_LOCATION_MEMBER_FIELDS: Dict[str, Tuple[str, ...]] = {
    "compound_locations": ("compound", "entity", "name"),
    "element_collection_locations": ("element_collection", "entity", "name"),
    "nucleic_acid_locations": ("nucleic_acid", "entity", "name"),
    "protein_locations": ("protein", "entity", "name"),
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
    """Yield every entity reference that is a plain name string.

    Two sources, both name-keyed and both inside the graph hash: ``processes``
    (participants) and ``element_locations`` (which entity is where). Walking
    only the first was a hole -- propagation, the undo log,
    :func:`_assert_fully_propagated` and the connectivity signature all read the
    world through this generator, so a reference it does not yield is one the
    module cannot rename, cannot roll back and **cannot see dangling in its own
    output**.

    Deterministic order: processes in the bucket order declared above, then
    ``element_locations``, then index, then field. The pointer is the identity
    the connectivity signature uses, so this ordering is contract.
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

    locations = _safe_dict(payload.get("element_locations"))
    for bucket, keys in _LOCATION_MEMBER_FIELDS.items():
        for lidx, row in enumerate(_safe_list(locations.get(bucket))):
            if not isinstance(row, dict):
                continue
            for field in keys:
                if isinstance(row.get(field), str) and _canonical(row.get(field)):
                    yield _Ref(row, field, f"/element_locations/{bucket}/{lidx}/{field}")


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
    """``processes`` and ``element_locations`` with every reference replaced by
    what it resolves to.

    Comparing this string before and after proves D-015 clause 5 in full: process
    and participant **counts**, participant **order**, **stoichiometry**, roles,
    and the resolved **identity** of every edge are all inside it. A name that
    changes but still resolves to the same entity produces the same signature; a
    name that stops resolving, or starts resolving to something else, does not.

    ``element_locations`` is projected alongside ``processes`` because it is the
    other name-keyed section in the graph hash: a rename that stopped a location
    resolving would otherwise change the hash without changing this signature.
    """

    holder = {
        "processes": deepcopy(_safe_dict(payload.get("processes"))),
        "element_locations": deepcopy(_safe_dict(payload.get("element_locations"))),
    }
    for ref in _iter_refs(holder):
        tokens = index.get(_norm(ref.get()), ())
        ref.set("<<" + "|".join(tokens) + ">>" if tokens else "<<UNRESOLVED>>")
    return json.dumps(holder, sort_keys=True, ensure_ascii=False, default=str)


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
    resolution outcome is the same one the exporter would have reached: the
    identity the loop settles on, recorded with the provenance of the pass that
    actually chose it (:func:`_resolve_to_fixed_point`). Anything else would be
    this module inventing a decision the resolver never made.

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
        "rename_sources_collapsed": [],
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
        _reject_ambiguous_renames(rename_map, before_names, after_names, primary_before, summary)
    # Called unconditionally, and AFTER the guard above: a canonicalization that
    # gives a name to a row that had none needs no rename-map entry to create a
    # duplicate, and ``AMBIGUOUS_RENAME_TARGET`` must keep its own name where it
    # already fires (D-035 clause 8).
    _reject_duplicate_canonical_rows(payload, rename_map, before_names, after_names, rows)

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
        _propagate(payload, rename_map, undo)
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


#: The fields that record **who decided** a compound's identity, and *what it
#: was decided from*, as opposed to what the identity is. A later pass may add
#: identity; it may never rewrite this record, because by then the only thing it
#: can observe is what an earlier pass already wrote. See
#: :func:`_authoritative_provenance`.
#:
#: The set is exactly the unconditional block of
#: ``db_resolver.apply_compound_db_resolution`` (``:450-454``) -- every field it
#: writes *before* its ``matched and admit_identity`` early return, i.e. every
#: field a re-run overwrites on a row it has already resolved:
#:
#: .. code-block:: text
#:
#:     out["raw_name"]    = match.raw_name or row.raw_name or row.name   # :450
#:     out["db_status"]   = ...                                          # :451
#:     out["db_match"]    = match                                        # :452
#:     out["chosen_rule"] = match.chosen_rule                            # :453
#:     out["confidence"]  = match.confidence                             # :454
#:
#: ``raw_name`` was the one member omitted, and the omission is observable, not
#: cosmetic. ``PathWhizCompoundResolver.resolve`` (``db_resolver.py:279``) sets
#: ``match["raw_name"] = row["name"]`` -- *the name it queried on this pass* --
#: and ``:450`` prefers it over the row's own. So pass 1 records the extraction
#: name correctly and pass 2, querying the row pass 1 renamed, overwrites it
#: with the canonical one. Measured on ``glycolate`` -> ``Glycolic acid``:
#: ``raw_name`` came out ``'Glycolic acid'`` through a reachable resolver and
#: ``'glycolate'`` with ``db_resolver=None`` -- the same payload, differing only
#: in whether a second pass ran.
#:
#: That is the extraction-name provenance carrier (``PRODUCT_CONTRACT`` §3):
#: ``ir.py:433`` projects it into every entity record, so the frozen payload and
#: the IR would both record the canonical name *as though it were what the paper
#: said*. ``db_match["raw_name"]`` was already preserved here, by ``db_match``;
#: only the top-level projection of the same fact drifted.
_PROVENANCE_FIELDS: Tuple[str, ...] = (
    "raw_name", "db_status", "chosen_rule", "confidence", "db_match",
)

#: Absence, distinguished from ``None``: a field the authoritative pass did not
#: write must be *removed* again, not set to ``None``.
_ABSENT = object()

#: Every list ``_resolve_compound_rows`` appends to, one entry per row per pass.
#: Re-running the resolution must not make the record claim the resolver was
#: consulted more often than it decided anything.
_REPORT_LOG_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("db_resolution", "compounds"),
    ("unresolved", "db_identities"),
    ("errors",),
    ("warnings",),
)


def _snapshot_provenance(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {field: (row.get(field, _ABSENT) if isinstance(row, dict) else _ABSENT)
         for field in _PROVENANCE_FIELDS}
        for row in rows
    ]


def _authoritative_provenance(
    incoming: List[Dict[str, Any]], first_pass: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Whose account of the decision stands: the payload's, or pass 1's.

    ``db_status`` is the marker that a resolution has already been recorded on a
    row. If the row arrived carrying one, *this call did not make that decision*
    and must not restate it -- that is what makes the whole operation idempotent
    rather than rewriting its own provenance a little further on every re-run.
    Otherwise pass 1 is the pass that chose, and pass 1's account stands.

    ``raw_name`` rides this same arbitration deliberately, absences included.
    A row that arrived already carrying a ``db_status`` was resolved by someone
    else, and the only name this loop can see on it is the one it already wears
    -- which for a renamed row is the *canonical* name, not the extraction one.
    Manufacturing a ``raw_name`` from that would assert a provenance nobody
    recorded, so the incoming account stands in full and ``_ABSENT`` stays
    absent, exactly as it already does for ``db_match``, ``chosen_rule`` and
    ``confidence``.
    """

    out: List[Dict[str, Any]] = []
    for before, after in zip(incoming, first_pass):
        prior = before.get("db_status", _ABSENT)
        out.append(before if prior is not _ABSENT and _canonical(prior) else after)
    return out


def _restore_provenance(rows: Sequence[Any], authoritative: Sequence[Dict[str, Any]]) -> None:
    for row, record in zip(rows, authoritative):
        if not isinstance(row, dict):
            continue
        for field, value in record.items():
            if value is _ABSENT:
                row.pop(field, None)
            else:
                row[field] = value


def _log_lengths(report: Dict[str, Any]) -> Dict[Tuple[str, ...], int]:
    lengths: Dict[Tuple[str, ...], int] = {}
    for path in _REPORT_LOG_PATHS:
        node: Any = report
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        target = node.get(path[-1]) if isinstance(node, dict) else None
        if isinstance(target, list):
            lengths[path] = len(target)
    return lengths


def _truncate_logs(report: Dict[str, Any], lengths: Dict[Tuple[str, ...], int]) -> None:
    for path, length in lengths.items():
        node: Any = report
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        target = node.get(path[-1]) if isinstance(node, dict) else None
        if isinstance(target, list) and len(target) > length:
            del target[length:]


def _resolve_to_fixed_point(
    rows: List[Dict[str, Any]],
    *,
    db_resolver: Any,
    strict_db: bool,
    report: Dict[str, Any],
    name_index: Any,
) -> Tuple[List[Dict[str, Any]], int]:
    """Apply resolution until re-applying it changes nothing, then restore the
    account of *who decided*.

    D-015 clause 8 -- "finish **all** network-dependent resolution before the
    freeze" -- is about the *end state*, not about running the function once. One
    pass does not reach it: pass 1 attaches a ``db_row``, which makes
    ``_canonicalize_compound_offline`` return early on pass 2, which in turn lets
    the legacy-id branch's ``db_status`` stand. So one pass leaves a payload on
    which the same resolution would still produce a different answer.

    But iteration alone **manufactures provenance** -- a D-015 clause 7 / A7
    violation, not a cosmetic one. Measured, one row, same inputs: pass 1 records
    ``db_status='matched' chosen_rule='fuzzy_name' conf=0.65``, pass 2 records
    ``db_status='legacy_id_unverified'
    chosen_rule='legacy_pathwhiz_id_unverified' conf=0.85``. Pass 1 writes
    ``pathwhiz_id``; pass 2 reads it back through the legacy-id branch, whose
    recorded reason is ``"payload_pathwhiz_id"`` -- so the payload would claim the
    id *arrived in the incoming payload*, 0.20 above what the resolver reported,
    with the rule that chose it erased. The offline-index path drifts the same
    way in reverse: ``matched_offline_name_index`` becomes ``unmatched``.

    So the loop settles the **identity**; pass 1 -- or the payload, if it already
    carried a ``db_status`` -- keeps the **account of the decision**. The per-pass
    log entries truncate back for the same reason: a repeat observation is not a
    second consultation.
    """

    incoming = _snapshot_provenance(rows)
    before_logs = _log_lengths(report)
    first_pass: Optional[List[Dict[str, Any]]] = None
    after_first_logs = before_logs

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
        if first_pass is None:
            first_pass = _snapshot_provenance(resolved)
            after_first_logs = _log_lengths(report)
        current = _rows_fingerprint(resolved)
        if current == fingerprint:
            _restore_provenance(resolved, _authoritative_provenance(incoming, first_pass))
            _truncate_logs(report, after_first_logs)
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
    summary: Optional[Dict[str, Any]] = None,
) -> None:
    """D-015 clause 6, checked before anything is written.

    Two conditions are fatal. **The source name is shared**: some entity other
    than the compounds that all canonicalize alike answers to it, so rewriting
    references named for it would redirect references meant for that entity.
    **Two distinct sources share a target**: the rename would merge two
    compounds into one, which is inventing biology, not canonicalizing it.

    "Distinct" is decided on the ``_norm`` form with its interior space runs
    collapsed, and that qualifier is load-bearing rather than cosmetic.
    ``_norm`` is ``_canonical`` -- which collapses whitespace -- followed by
    ``[^a-z0-9:+ ]+ -> " "``, and that substitution can put a **second** space
    back where a separator sat next to one, with nothing left to collapse it::

        _norm('sn -glycerol 3-phosphate') == 'sn  glycerol 3 phosphate'
        _norm('sn-glycerol 3-phosphate')  == 'sn glycerol 3 phosphate'

    Measured on ``runs/2026-07-28_0919/…PMC12444477…/strict``, where the offline
    index maps both spellings to one canonical target: the guard read them as two
    compounds and aborted a real 27-reaction export at both production entry
    points, in both ``strict_db`` modes, producing no PWML at all
    (``PRODUCT_CONTRACT`` §1; permanent merge rule 7). Merging two spellings of
    one molecule is canonicalization, which D-015 **asks for**.

    **This does not widen the guard's notion of sameness** (merge rule 6).
    ``_norm`` already maps every character outside ``[a-z0-9:+ ]`` to a space, so
    it *already* reads ``beta-D-glucose`` and ``beta D glucose`` as one source --
    measured, at base, no raise. The equivalence has always been "the same token
    sequence"; the double space was the one place ``_norm`` failed to apply its
    own rule, because the count of separator characters leaked through. Collapsing
    here restores consistency and adds no class of name that was not already
    equivalent under a single separator. Sources whose tokens differ at all --
    ``glycerol 2-phosphate`` vs ``glycerol 3-phosphate``, ``ATP`` vs ``ADP``,
    ``PEtN-lipid A`` vs ``modified Lipid A`` -- still collide and still raise.

    ``_norm`` and ``_canonical`` themselves are **not** touched. They are
    byte-identical to ``ir._norm`` / ``ir._canonical`` by design, every downstream
    resolver keys on them, and changing them would move the graph hash. The
    collapse is local to this comparison.
    """

    def _collapsed(value: Any) -> str:
        """``_norm`` with its interior space runs squeezed. Never a substitute
        for ``_norm``: nothing outside this comparison may key on it."""

        return re.sub(r" +", " ", _norm(value))

    merged: List[Dict[str, Any]] = []
    by_target: Dict[str, List[str]] = {}
    for old, new in rename_map.items():
        by_target.setdefault(_norm(new), []).append(old)
    for target, sources in sorted(by_target.items()):
        keys = {_collapsed(source) for source in sources}
        if len(keys) > 1:
            raise PrefreezeResolutionError(
                "AMBIGUOUS_RENAME_TARGET",
                f"{len(keys)} distinct compound names canonicalize to {target!r}; "
                "applying the rename would merge them",
                target=target, sources=sorted(sources),
            )
        if len(sources) > 1:
            # One molecule, spelled more than one way. The rename proceeds, but
            # the operator is told which spellings were read as one -- the same
            # shape ``aliases_preserved`` and ``identity_projected`` use.
            merged.append({
                "target": target, "sources": sorted(sources),
                "source_key": sorted(keys)[0],
            })
    if isinstance(summary, dict):
        summary["rename_sources_collapsed"] = merged

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


#: Identifier keys copied into the duplicate-row diagnostic. **Diagnostic only,
#: never a decision input** -- comparing them would be the engine D-036 clause 2
#: defers. Carried because D-035 clause 6 requires "enough diagnostic information
#: for review", and read from the **pre-resolution** rows per F-043: a resolver
#: can stamp one stub id onto two distinct compounds.
_DUPLICATE_ROW_ID_KEYS: Tuple[str, ...] = (
    "pathbank_compound_id", "pw_compound_id", "pathwhiz_id",
    "kegg_id", "chebi_id", "hmdb_id", "pubchem_cid",
)


def _reject_duplicate_canonical_rows(
    payload: Dict[str, Any],
    rename_map: Dict[str, str],
    before_names: Sequence[str],
    after_names: Sequence[str],
    rows: Sequence[Any],
) -> None:
    """D-035 clause 6 / D-036 clause 1 -- the refusal path, named.

    Refuses when canonicalization leaves **two or more compound rows sharing one
    ``_norm`` name**, at least one of them did not already have that name, and a
    participant reference lands on that name.

    ``_norm`` defines the group and nothing else does (**F-040 as corrected**):
    ``entity_by_name`` (``ir.py:1105-1117``) and ``resolve_entity``
    (``ir.py:1371``) key on it, so it alone decides which row a reference in the
    frozen payload reaches. ``._norm`` here is a byte-identical duplicate of
    ``ir._norm``, deliberately, so the two do not compete;
    ``process_normalizer._normalize`` / ``._norm_text`` disagree with both and are
    incomparable rather than coarser.

    **Detection is this function's own** (D-034 clause 5).
    :func:`_reject_ambiguous_renames` groups only over ``rename_map`` *sources*,
    so it is structurally blind to a target landing on a row that is not itself
    renamed; it is neither modified here nor relied on.

    **What it deliberately does not do.**

    * It does **not** consolidate: D-035 clauses 2-5 are deferred by **D-036
      clause 2**, no committed group clearing clause 3.
    * It does **not** compare identifiers to decide. Identifier equality is a
      measurably unsafe trigger (**F-043**: ``PG`` / ``PG phosphate`` / ``(PGP)``
      all carry ``pathbank_compound_id`` 193, which is UDP-glucose).
    * It does **not** fire on a collision the payload already carried, nor on a
      created group **no reference reaches**. D-035 clause 6's subject is "PWML
      with ambiguous connectivity" and D-036 clause 1 charters a *name* for the
      condition that was already refusing. Both excluded shapes export today and
      are then coalesced first-wins by ``ir._dedupe_named_rows`` -- **F-039**,
      routed as **C-050i**, which D-036's "third option" declined to fold in
      here. Measured cost of that boundary: without it, two parametrizations of
      ``test_prefreeze_third_export_seam`` A3 newly refuse.
    * An empty ``_norm`` is not a group -- bucketing names written entirely
      outside ``[a-z0-9:+ ]`` together would merge names that are not the same
      name, as :func:`_match_key` already records.

    Fail-closed and total: only the staged copy has been resolved by this point,
    so the payload is untouched and no PWML comes out of an ambiguous one.
    """

    groups: Dict[str, List[int]] = {}
    for index, after in enumerate(after_names):
        key = _norm(after)
        if key:
            groups.setdefault(key, []).append(index)

    # Where every reference WILL point once the rename is applied, via the same
    # ``_rename_targets`` / ``_match_key`` pair ``_propagate`` rewrites with.
    targets = _rename_targets(rename_map)
    referenced = {
        _norm(targets.get(_match_key(current), current))
        for current in (ref.get() for ref in _iter_refs(payload))
    }

    for key, members in sorted(groups.items()):
        if len(members) < 2 or key not in referenced:
            continue
        arrived = [index for index in members if _norm(before_names[index]) != key]
        if not arrived:
            continue
        detail = []
        for index in members:
            row = _safe_dict(rows[index] if index < len(rows) else None)
            # ``mapped_ids`` first, then the top-level projection, which wins on
            # the one key both carry. The D-034 leg's decisive KEGG disagreement,
            # ``C03189`` against ``C00093``, is in ``mapped_ids`` and nowhere else.
            merged = {**_safe_dict(row.get("mapped_ids")),
                      **{f: row[f] for f in _DUPLICATE_ROW_ID_KEYS if f in row}}
            identifiers = {f: v for f, v in merged.items() if v not in (None, "")}
            detail.append({
                "row": index,
                "name_before": before_names[index],
                "name_after": after_names[index],
                "canonicalized": index in arrived,
                "identifiers": identifiers,
            })
        raise PrefreezeResolutionError(
            "PREFREEZE_DUPLICATE_CANONICAL_ROWS",
            f"canonicalization puts {len(members)} compound rows on the single "
            f"canonical name {key!r} ({[item['name_before'] for item in detail]}); "
            "consolidating them would require proof that they are the same "
            "biological entity, which this stage does not perform",
            canonical_key=key,
            reason="equivalence_not_proven",
            rows=detail,
            canonicalized_rows=list(arrived),
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


def _match_key(value: Any) -> str:
    """The one key a reference and a rename source match on, in both directions.

    ``_norm`` wherever it says anything, ``_canonical`` behind a sentinel
    wherever it does not. ``_norm`` keeps only ``[a-z0-9:+ ]``, so a name written
    **entirely** in other characters -- ``---``, ``??``, Greek, CJK -- normalizes
    to ``""``. Bucketing those under ``""`` merges names that are not the same
    name; *discarding* them leaves them neither rewritten nor audited, so the row
    is renamed, its reference left behind, and the frozen payload carries a
    reference resolving to nothing on reload (``PRODUCT_CONTRACT`` §5, D-015
    clause 3). The sentinel keeps them matchable *and* distinct, and cannot
    collide with a ``_norm`` result.

    **Both** :func:`_rename_targets` and :func:`_assert_fully_propagated` key on
    this, so the rewrite set and the detection set are one set by construction --
    this card's thesis, broken by the original defect in one direction and by
    C-050f round 1 in the other.
    """

    return _norm(value) or "\x00" + _canonical(value)


def _rename_targets(rename_map: Dict[str, str]) -> Dict[str, str]:
    """``_match_key(old) -> new``, refusing a map whose sources collide.

    Two sources sharing a match key are the **same** reference to everything that
    reads the result -- ``_alias_index`` here, ``canonical._Graph.resolve`` and
    ``ir.resolve_entity`` downstream -- so a silent last-wins would rewrite
    references meant for the other one. Like :func:`_reject_ambiguous_renames`,
    this refuses instead -- and it is **reachable**, because that function groups
    by ``_norm(new)`` and so misses targets differing only in case (F-2).
    """

    targets: Dict[str, str] = {}
    for old, new in rename_map.items():
        if not _canonical(old):
            continue
        key = _match_key(old)
        previous = targets.get(key)
        if previous is not None and _canonical(previous) != _canonical(new):
            raise PrefreezeResolutionError(
                "PREFREEZE_RENAME_MAP_COLLISION",
                f"rename sources matching on {key.lstrip(chr(0))!r} target both "
                f"{previous!r} and {new!r}; a reference matching both cannot be "
                "rewritten unambiguously",
                key=key.lstrip(chr(0)), targets=sorted({previous, new}),
            )
        targets[key] = new
    return targets


def _propagate(
    payload: Dict[str, Any],
    rename_map: Dict[str, str],
    undo: Optional[List[Tuple[Any, Any, Any]]] = None,
) -> List[Tuple[str, str, str]]:
    """Rewrite every participant reference named by the map. Returns the updates.

    **Matched by :func:`_match_key`, not ``_canonical``.** The audit that follows
    has always detected staleness by ``_norm``, so a rewriter keyed on
    ``_canonical`` made the detection set *strictly wider* than the rewrite set:
    a reference spelled ``GLY`` for a compound renamed from ``gly`` was never
    rewritten and then always raised ``PREFREEZE_RENAME_NOT_PROPAGATED`` -- a hard
    abort (``PRODUCT_CONTRACT`` §1) on a reference that is not dangling at all,
    since it resolves to the very entity being renamed. D-015 clause 6 is about
    references that cannot be resolved; this one resolves, so failing on it was a
    false positive, and ``_norm`` is the rule every consumer of the frozen payload
    already applies.

    ``undo`` collects ``(container, key, previous)`` for the caller's rollback log
    when the rewrite is committed to the live payload. That pass was an inline
    copy of this match rule, and one copy is how the two rules drifted apart.
    """

    updates: List[Tuple[str, str, str]] = []
    targets = _rename_targets(rename_map)
    if not targets:
        return updates
    for ref in _iter_refs(payload):
        current = ref.get()
        target = targets.get(_match_key(current))
        if target is None or target == current:
            continue
        if undo is not None:
            undo.append((ref.container, ref.key, current))
        updates.append((ref.pointer, current, target))
        ref.set(target)
    return updates


def _assert_fully_propagated(payload: Dict[str, Any], rename_map: Dict[str, str]) -> None:
    """No reference may still normalize to a name that was renamed away.

    A reference is stale when it matches a rename source under
    :func:`_match_key` and is **not spelled as that source's target**. Keying on
    ``_match_key`` makes this set identical to :func:`_propagate`'s rewrite set,
    so nothing can be rewritten without being auditable, or matched by the audit
    without being rewritable.

    This **adds** the pure case changes the old ``_norm(old) != _norm(new)``
    guard skipped. It is **not** the strict superset round 1 claimed: it
    **removes exactly one class, deliberately.** The old form bucketed every
    empty-``_norm`` name under ``""``, so renaming ``---`` made an *unrelated*
    Greek-alpha reference fatal in a message naming ``---`` -- a false positive
    on a reference the rename never touched. Such a reference is dangling on its
    own account and is invisible here exactly as ``Serine`` is, which is the
    standing deferred finding rather than this function's subject. The renamed
    name itself is still matched, rewritten and audited (B-1), and D-015 clause 6
    keeps every genuinely stale reference fatal.
    """

    stale = {_match_key(old): (old, new)
             for old, new in rename_map.items() if _canonical(old)}
    if not stale:
        return
    for ref in _iter_refs(payload):
        current = ref.get()
        hit = stale.get(_match_key(current))
        if hit is not None and _canonical(current) != _canonical(hit[1]):
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
# The species canonicalizer (C-045 / D-016).
# ---------------------------------------------------------------------------


def _reject_ambiguous_species_renames(
    rename_map: Dict[str, str],
    before_names: Sequence[str],
    after_names: Sequence[str],
    primary_before: Dict[str, Tuple[str, ...]],
) -> None:
    """D-015 clause 6 for species, checked before anything is written.

    The **rule** is :func:`_reject_ambiguous_renames`'s, and the two error codes
    are the same two: a shared source name would redirect references meant for
    another entity, and two distinct sources sharing a target would merge two
    organisms into one. Only the bucket differs, and it differs materially --
    that function builds its compatibility set out of ``"compounds#<i>"`` tokens
    while :func:`_alias_index` labels a species row ``"species#<i>"``, so reusing
    it here would find *every* species rename rogue and raise
    ``AMBIGUOUS_REFERENCE`` on a rename that is not ambiguous at all. It is
    C-050's owned function and this card may not retune it, so the bucket-correct
    twin lives here. Unifying the two is a deferred finding, not this card's.
    """

    by_target: Dict[str, List[str]] = {}
    for old, new in rename_map.items():
        by_target.setdefault(_norm(new), []).append(old)
    for target, sources in sorted(by_target.items()):
        if len({_norm(source) for source in sources}) > 1:
            raise PrefreezeResolutionError(
                "AMBIGUOUS_RENAME_TARGET",
                f"{len(sources)} distinct species names canonicalize to {target!r}; "
                "applying the rename would merge them",
                target=target, sources=sorted(sources),
            )

    # ...and a target some OTHER species row already answers to and keeps. The
    # loop above only compares rename sources with each other, so a rename onto
    # an existing, unrenamed row falls between its two branches -- and for
    # species nothing else catches it. ``_LOCATION_MEMBER_FIELDS`` has no species
    # bucket, so ``_iter_refs``, ``_propagate``, ``_assert_fully_propagated`` and
    # ``_connectivity_signature`` are all blind to species rows, and the
    # row-count check sees 2 == 2 because the loss happens later, in the
    # exporter's ``_dedupe_named_rows``. The compound stage fails closed on the
    # identical shape only because compounds are *participants*, so the rename
    # breaks a reference and ``PREFREEZE_CONNECTIVITY_BROKEN`` fires; species are
    # not participants, so the refusal has to be stated.
    #
    # Left unrefused, ``build_pwml_ir`` deduplicates the two now-identically
    # named rows and the second organism is **deleted**, leaving a row that
    # carries the surviving name with the *other* row's ``taxonomy_id`` -- a
    # wrong organism identity, not a lossy one (``PRODUCT_CONTRACT`` §2, §5), and
    # the operator-facing at-risk list shrinks at the moment of the loss. Merge
    # rule 7 keeps incomplete-but-correct content as ``review_required``; it does
    # not license silently dropping an organism.
    #
    # Deliberately narrow, so it cannot over-fire on the cases the ladder is
    # *for*: a target inside the source's own ``_norm`` group is a spelling
    # change that merges nothing, and a row that vacates the target name under
    # its own rename is no longer there to collide with.
    for old, new in sorted(rename_map.items()):
        target = _norm(new)
        if target == _norm(old):
            continue
        occupied = sorted({
            before_names[index] for index in range(len(before_names))
            if before_names[index]
            and _norm(before_names[index]) == target
            and _norm(after_names[index]) == target
        })
        if occupied:
            raise PrefreezeResolutionError(
                "AMBIGUOUS_RENAME_TARGET",
                f"renaming {old!r} to {new!r} would merge it into {occupied}, which "
                "another species row already answers to and keeps",
                target=new, sources=sorted({old, *occupied}),
            )

    compatible: Dict[str, set] = {}
    for index, before in enumerate(before_names):
        if before:
            compatible.setdefault(_norm(before), set()).add(f"species#{index}")
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


def _canonicalize_species_rows(rows: List[Any], *, name_index: Any) -> List[Dict[str, Any]]:
    """Run **the** species ladder over ``rows``, once per surviving row.

    ``ir._canonicalize_species_offline`` is imported, never re-implemented: one
    precedence ladder, one set of statuses, one log shape. What this adds is the
    ordering the exporter used to supply for free.

    ``build_pwml_ir`` canonicalized the output of ``_dedupe_named_rows``, which
    keeps the **first** row of each ``_norm(name)`` group and drops the rest with
    a ``duplicate_named_record`` warning. Running the ladder over every payload
    row instead would let a dropped duplicate be renamed to a name the survivor
    does not share -- and a row that stops being a duplicate becomes a **second
    species** in the IR that the exporter never emitted. That is inventing
    biology, so the group leader is the row that gets canonicalized, and the rest
    of its group follows it **only when the leader's rename moved the group's
    ``_norm``**, which is exactly the condition under which they would otherwise
    stop deduplicating. A leader left alone leaves its group alone.

    Every row that participated carries :data:`ir.SPECIES_CANONICALIZATION_FIELD`
    afterwards, so ``build_pwml_ir`` can rebuild the report it used to compute.
    That marker is **durable**, and the returned list is not: re-running the stage
    over an already-canonical payload produces no fresh log entry -- the ladder
    correctly finds nothing to do -- but the row must keep the account of the pass
    that *did* decide, or a replayed payload would silently lose the rename from
    ``pwml_ir_report.json``. Same distinction ``_authoritative_provenance`` draws
    on the compound side, for the same reason. The returned entries describe what
    **this** call did, so the summary never claims a second consultation.
    """

    fresh: List[Dict[str, Any]] = []
    leaders: Dict[str, int] = {}
    for index, row in enumerate(rows):
        if isinstance(row, dict) and _canonical(row.get("name")):
            leaders.setdefault(_norm(row.get("name")), index)

    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not _canonical(row.get("name")):
            continue  # ``_dedupe_named_rows`` never saw this row either
        leader = leaders[_norm(row.get("name"))]
        prior = _safe_dict(row.get(SPECIES_CANONICALIZATION_FIELD))
        if leader == index:
            log: Dict[str, Any] = {}
            status = _canonicalize_species_offline(row, name_index=name_index, report=log)
            entries = _safe_list(_safe_dict(log.get("name_canonicalization")).get("species"))
            marker: Dict[str, Any] = {"status": status}
            if entries:
                marker["entry"] = entries[0]
                fresh.append(entries[0])
            elif isinstance(prior.get("entry"), dict):
                marker["entry"] = prior["entry"]
            row[SPECIES_CANONICALIZATION_FIELD] = marker
            continue
        leader_row = rows[leader]
        marker = {"status": _safe_dict(leader_row.get(SPECIES_CANONICALIZATION_FIELD)).get("status")}
        target = _canonical(leader_row.get("name"))
        own = _canonical(row.get("name"))
        if target and _norm(target) != _norm(own):
            row.setdefault("raw_name", own)
            aliases = _dedupe_aliases([*_safe_list(row.get("aliases")), own])
            if aliases:
                row["aliases"] = aliases
            row["name"] = target
            marker["followed_leader"] = own
        elif isinstance(prior.get("followed_leader"), str):
            marker["followed_leader"] = prior["followed_leader"]
        row[SPECIES_CANONICALIZATION_FIELD] = marker

    return fresh


def resolve_species_prefreeze(
    payload: Dict[str, Any],
    *,
    db_resolver: Any = None,  # noqa: ARG001 - the ladder is offline; see below
    strict_db: bool = False,  # noqa: ARG001
    name_index: Any = _NAME_INDEX_UNSET,
    report: Optional[Dict[str, Any]] = None,  # noqa: ARG001
) -> Dict[str, Any]:
    """Canonicalize ``payload['entities']['species']`` in place, pre-freeze.

    ``DECISIONS.md`` **D-016 (LOCKED)**: organism context is part of the canonical
    biological representation, so the species rename belongs upstream of the
    freeze for the same reason the compound rename does. This is that move -- the
    ladder is ``ir._canonicalize_species_offline``, unchanged and unduplicated;
    what changes is that it now runs here rather than inside ``build_pwml_ir``.

    Mutates **the object it is given**, row identities included where the ladder
    left them alone, so a caller holding ``final_export_payload`` cannot end up
    holding a pre-canonicalization view of it.

    ``db_resolver`` and ``strict_db`` are accepted because
    :func:`run_prefreeze_resolution` passes them to every canonicalizer, and are
    deliberately **unused**: the species ladder's step 1 (a live resolution-DB
    name) is applied upstream in stage-2 hydration and reaches this function as a
    name already on the row, and steps 2-4 are offline by construction. So this
    stage opens no connection and can report no ``db_unavailable`` -- D-029's
    ``review_required`` outcome is reachable here only through the compound
    canonicalizer beside it, which is where the database actually is.

    Raises :class:`PrefreezeResolutionError` on an ambiguous rename or a
    reference the rename would break. On any raise the payload is unchanged.
    """

    summary: Dict[str, Any] = {
        "applied": False,
        "rows": 0,
        "renamed": 0,
        "rename_map": {},
        "references_updated": [],
        "statuses": {},
        "at_risk": [],
        "name_canonicalization": [],
    }
    if not isinstance(payload, dict):
        summary["skipped_reason"] = "payload_not_a_mapping"
        return summary
    entities = payload.get("entities")
    rows = entities.get("species") if isinstance(entities, dict) else None
    if not isinstance(rows, list) or not rows:
        summary["skipped_reason"] = "no_species_rows"
        return summary

    summary["rows"] = len(rows)
    resolved_name_index = default_name_index() if name_index is _NAME_INDEX_UNSET else name_index
    if resolved_name_index is not None and not hasattr(resolved_name_index, "species_canonical"):
        # A name index that cannot answer a species question is step 2 being
        # *absent*, not step 2 saying "no". Recorded rather than swallowed: the
        # ladder still runs, and its remaining steps still decide whether to
        # APPLY a rename (D-028), but the record says which rung was missing.
        summary["name_index"] = "no_species_canonical"
        resolved_name_index = None

    # ---- stage 1: canonicalize on a copy. The live rows are untouched. ----
    staged = deepcopy(rows)
    before_names = [_canonical(row.get("name")) if isinstance(row, dict) else "" for row in staged]
    fresh_entries = _canonicalize_species_rows(staged, name_index=resolved_name_index)
    if len(staged) != len(rows):
        raise PrefreezeResolutionError(
            "PREFREEZE_ROW_COUNT_CHANGED",
            f"species canonicalization returned {len(staged)} rows for {len(rows)} inputs",
            expected=len(rows), actual=len(staged),
        )
    after_names = [_canonical(row.get("name")) if isinstance(row, dict) else "" for row in staged]

    summary["statuses"] = {
        before_names[index]: _safe_dict(row.get(SPECIES_CANONICALIZATION_FIELD)).get("status")
        for index, row in enumerate(staged)
        if isinstance(row, dict) and before_names[index]
    }
    summary["at_risk"] = [
        after_names[index]
        for index, row in enumerate(staged)
        if isinstance(row, dict)
        and _safe_dict(row.get(SPECIES_CANONICALIZATION_FIELD)).get("status") == "deterministic"
    ]
    summary["name_canonicalization"] = fresh_entries

    # ---- stage 2: the explicit rename map (D-015 clause 2) -----------------
    rename_map: Dict[str, str] = {}
    for index, (before, after) in enumerate(zip(before_names, after_names)):
        if not before or not after or before == after:
            continue
        previous = rename_map.get(before)
        if previous is not None and _norm(previous) != _norm(after):
            raise PrefreezeResolutionError(
                "AMBIGUOUS_RENAME_SOURCE",
                f"species name {before!r} canonicalizes to both {previous!r} and {after!r}",
                name=before, targets=sorted({previous, after}), row=index,
            )
        rename_map[before] = after
    summary["rename_map"] = dict(rename_map)
    summary["renamed"] = len(rename_map)

    primary_before, alias_before = _alias_index(payload)
    if rename_map:
        _reject_ambiguous_species_renames(rename_map, before_names, after_names, primary_before)

    # ---- stage 3: propagate on a staged copy of the whole payload ----------
    # Same traversal, same match rule, same audit as the compound stage --
    # ``_LOCATION_MEMBER_FIELDS`` was written to cover all four buckets for
    # exactly this call.
    staged_payload = deepcopy(payload)
    staged_payload["entities"]["species"] = staged
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
        rows[:] = staged
        _propagate(payload, rename_map, undo)
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


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

#: Ordered canonicalizers. C-045 appended ``("species", resolve_species_prefreeze)``
#: at the seam C-050 declared for it. The two are independent -- neither reads a
#: row the other writes -- so the order is the declaration order and nothing else
#: depends on it.
PREFREEZE_CANONICALIZERS: Tuple[Tuple[str, Callable[..., Dict[str, Any]]], ...] = (
    ("compounds", resolve_compounds_prefreeze),
    ("species", resolve_species_prefreeze),
)

#: A canonicalizer that ran on nothing is not a failure. Any other skip is.
_BENIGN_SKIPS = frozenset({"no_compound_rows", "no_species_rows"})

#: Verdicts that mean "identity was not established", not "the payload is wrong".
#: An unreachable PathBank DB is an infrastructure condition, not a defect in the
#: graph: merge rule 7 keeps incomplete-but-correct content as ``review_required``
#: rather than dropping it, and raising here would turn a PathBank outage into a
#: total export failure. D-015 clause 6's "fail visibly" is about **ambiguous or
#: dangling references** -- structural defects -- and those still raise directly
#: out of :func:`resolve_compounds_prefreeze`, untouched by this.
_REVIEW_REQUIRED_REASONS = frozenset({"resolution_report_not_ok:db_unavailable"})


def _canonicalizer_verdict(summary: Any) -> Tuple[bool, str]:
    """``(ok, reason)`` for one canonicalizer's summary. Reason is "" when ok.

    The reason separates a resolver that was consulted and rejected the row from
    one that was never reachable. Neither may be reported as a success, but only
    the first is a statement about the payload -- so the verdict says which, and
    :data:`_REVIEW_REQUIRED_REASONS` routes the second to review.
    """

    if not isinstance(summary, dict):
        return False, "summary_not_a_mapping"
    skipped = str(summary.get("skipped_reason") or "")
    if skipped and skipped not in _BENIGN_SKIPS:
        return False, skipped
    resolution = _safe_dict(summary.get("resolution_report"))
    if resolution.get("ok") is False:
        if _safe_dict(resolution.get("db_resolution")).get("available") is False:
            return False, "resolution_report_not_ok:db_unavailable"
        return False, "resolution_report_not_ok"
    return True, ""


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

    ``report["ok"]`` is a **verdict, not a stamp.** It used to be set ``True``
    before the canonicalizers ran, with nothing able to falsify it, so a nested
    resolution report carrying ``ok=False`` and error-severity
    ``compound_db_resolution_failed`` entries was handed back as a success. It is
    now the conjunction of every canonicalizer's verdict; a non-benign skip or a
    failed resolution report names its reason under ``report["failures"]``.

    **It does not raise on that verdict**, and the distinction is the point.
    Raising is reserved for the structural failures
    :func:`resolve_compounds_prefreeze` already raises directly --
    ``AMBIGUOUS_RENAME_SOURCE``, ``AMBIGUOUS_REFERENCE``,
    ``PREFREEZE_DUPLICATE_CANONICAL_ROWS``,
    ``PREFREEZE_CONNECTIVITY_BROKEN``, ``PREFREEZE_RENAME_NOT_PROPAGATED`` -- which
    are D-015 clause 6's actual subject: a payload whose references cannot be
    resolved unambiguously. An identity the PathBank DB was never reachable to
    establish is **incomplete, not wrong**; merge rule 7 keeps that as
    ``report["review_required"]`` rather than dropping the pathway, and stopping
    on it would convert a database outage into a total export failure.

    The report is the whole output: nothing here mutates the payload beyond what
    the canonicalizers did, and every canonicalizer runs even if an earlier one
    reported a failure, so the verdict describes the payload as it stands.
    """

    report: Dict[str, Any] = {
        "stage": "prefreeze_resolution",
        "canonicalizers": [name for name, _ in canonicalizers],
    }
    failures: Dict[str, str] = {}
    for name, canonicalizer in canonicalizers:
        summary = canonicalizer(
            payload,
            db_resolver=db_resolver,
            strict_db=strict_db,
            name_index=name_index,
        )
        report[name] = summary
        ok, reason = _canonicalizer_verdict(summary)
        if not ok:
            failures[name] = reason
    report["ok"] = not failures
    report["failures"] = failures
    report["review_required"] = {
        name: reason for name, reason in failures.items()
        if reason in _REVIEW_REQUIRED_REASONS
    }

    # ---- carry the DB-reachability verdict to the exporter ----------------
    #
    # ``ir._emit_canonicalization_preflight`` reads
    # ``report["db_resolution"]["available"]`` to decide whether names may be
    # non-canonical, and D-032 clause 6 (LOCKED) rules ``preflight`` and
    # ``name_canonicalization`` product-visible export content. The value is
    # decided *here*, pre-freeze: ``compound_resolution.py:485`` computes it
    # from ``db_resolver.available()`` after replacing a ``None`` resolver with
    # ``PathBankDbResolver.from_env()``. The exporter may not re-derive it --
    # that would be a post-freeze probe of an external service, which D-015
    # forbids -- and cannot infer it either, because an all-legacy population
    # resolves to byte-identical rows whether or not the DB was reachable. So
    # it is written onto the payload, the one object every export entry point
    # hands to ``build_pwml_ir``.
    #
    # Written only when a canonicalizer actually recorded the fact. A payload
    # this function skipped entirely (no compound rows) records nothing rather
    # than asserting an availability nobody measured, and ``build_pwml_ir``
    # keeps the report shape it already had.
    # ``reason`` rides with ``available``: the preflight prints both, and it
    # defaults an absent reason to ``"db_not_configured"`` -- false of a DB that
    # was configured and down. Only the compound stage records either.
    consulted: List[bool] = []
    reasons: List[str] = []
    for name, _canonicalizer in canonicalizers:
        db_resolution = _safe_dict(
            _safe_dict(_safe_dict(report.get(name)).get("resolution_report")).get("db_resolution")
        )
        if isinstance(db_resolution.get("available"), bool):
            consulted.append(bool(db_resolution["available"]))
        if str(db_resolution.get("reason") or "").strip():
            reasons.append(str(db_resolution["reason"]))
    # D-032 clause 1 exists so this tuple GROWS. The day a second canonicalizer
    # records ``db_resolution``, ``any()`` would OR two different facts and
    # ``reasons[0]`` would pair one stage's availability with another's reason.
    # Asserted rather than assumed, so that fails loudly instead of mispairing.
    assert len(consulted) <= 1, (
        "more than one canonicalizer recorded db_resolution; this carrier has no "
        "rule for combining them: " + str([name for name, _ in canonicalizers])
    )
    if consulted and isinstance(payload, dict):
        marker: Dict[str, Any] = {"available": any(consulted)}
        if reasons:
            marker["reason"] = reasons[0]
        payload[PREFREEZE_DB_RESOLUTION_FIELD] = marker

    return report

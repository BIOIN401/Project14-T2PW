"""Compound DB resolution and offline name canonicalization.

Lifted verbatim out of ``t2pw.pwml.ir`` by C-040 under SPIKE-002's
``LIFT_WITH_ADAPTER`` verdict. **This module is not called from production yet.**
``ir.build_pwml_ir`` still drives the in-IR call site; moving that call
pre-freeze is C-050's, and deleting the in-IR one is C-051's.

Why it moved: ``ir.py:900-908`` documents non-canonical names colliding on
PathWhiz import as the root bug, and the step that fixes them runs *inside the
exporter*, downstream of the canonical freeze -- where an exporter may not
resolve biological content. It has to be callable from before the freeze before
it can be moved there. C-040 makes it callable and changes nothing else.

Entry points are ``_resolve_compound_rows`` and ``_canonicalize_compound_offline``;
``_normalize_compound_external_ids`` and ``_compound_external_ids`` are per-row
helpers, the latter also re-imported by ``ir.py`` for its preflight. All four
keep their leading underscore so the move stays byte-verbatim and reviews as a
move rather than a rewrite.

The three-part adapter contract (SPIKE-002 §2)
----------------------------------------------
1. **Report shape.** ``_resolve_compound_rows`` hard-indexes the nested shape
   ``ir._new_report()`` builds. ``ensure_resolution_report`` seeds those
   containers so a caller holding a bare ``{}`` -- which every pre-freeze caller
   does -- no longer raises ``KeyError``. It is a no-op on a real report.

2. **Rename propagation is the CALLER's obligation**, and both entry points take
   a keyword-only ``apply_canonical_name`` so the caller can decline the rename
   until it has discharged it. ``apply_compound_db_resolution`` and
   ``_canonicalize_compound_offline`` rewrite ``row["name"]``, but
   ``processes.reactions[].inputs`` in the canonical payload are plain name
   *strings*. ``build_pwml_ir`` absorbs a rename only because ``entity_by_name``
   indexes both ``name`` and ``raw_name``; **no pre-freeze consumer does**, and
   ``strict_quarantine`` keys on ``name``/``synonyms`` alone, so an unpropagated
   pre-freeze rename silently prunes the compound and breaks the reaction that
   references it. A caller passing ``apply_canonical_name=True`` pre-freeze must
   propagate the rename across ``processes.reactions[].inputs/outputs``,
   ``transports``, ``interactions`` and ``enzymes[].entity``. This module cannot:
   it is handed rows, not the payload that references them.

3. **Row set and idempotency.** Pre-freeze these functions see the *un-pruned,
   un-deduped* payload rows -- a strict superset of what the in-IR call sees
   today, which runs after ``_dedupe_named_rows`` and after quarantine pruning.
   ``_resolve_compound_rows`` is positional per row and never assumes uniqueness;
   ``_canonicalize_compound_offline`` returns early once a row carries a resolved
   ``db_row`` name, so re-running either over resolved rows is stable.

D-028 -- DB-match admission (C-040a). ``_resolve_compound_rows`` used to log a
sub-threshold match as a *failure* and apply it anyway; :func:`_admit_db_identity`
is now the single place that decision is made. Its call site is shared by the
pre-freeze caller and ``ir.build_pwml_ir``, and D-028 governs both, so it is
deliberately not special-cased by caller.

The nine private helpers below are **verbatim copies** of ``ir.py:43-96``,
``:183-193`` and ``:244-260``; the originals are unmodified. Duplication is
forced -- ``ir.py`` imports this module, so importing back would be a cycle --
so ``tests/test_compound_resolution_extraction.py`` pins every copy to its
original by source equality, which is what stops the two drifting (SPIKE-002 F-3).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from t2pw.pwml.db_resolver import (
    PathWhizCompoundResolver,
    _mapped_ids_from_row,
    apply_compound_db_resolution,
    normalize_chebi_id,
    normalize_hmdb_id,
    normalize_kegg_id,
    normalize_pubchem_cid,
)
from t2pw.pwml.name_index import PathwhizNameIndex


# ---------------------------------------------------------------------------
# Verbatim leaf-helper copies -- ir.py:43-96, :183-193, :244-260.
# Pinned to the originals by source equality; do not edit either side alone.
# ---------------------------------------------------------------------------


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _norm(value: Any) -> str:
    text = _canonical(value).casefold()
    return re.sub(r"[^a-z0-9:+ ]+", " ", text).strip()


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _first_nonempty(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    meta = _safe_dict(row.get("mapping_meta"))
    for key in keys:
        if key in meta and meta.get(key) not in (None, ""):
            return meta.get(key)
    mapped = _safe_dict(row.get("mapped_ids"))
    for key in keys:
        if key in mapped and mapped.get(key) not in (None, ""):
            return mapped.get(key)
    candidates = _safe_list(meta.get("candidates"))
    if candidates and isinstance(candidates[0], dict):
        top = candidates[0]
        for key in keys:
            if key in top and top.get(key) not in (None, ""):
                return top.get(key)
    return None


def _db_id(row: Dict[str, Any], keys: Sequence[str]) -> Optional[int]:
    return _to_int(_first_nonempty(row, keys))


def _dedupe_aliases(values: Iterable[Any]) -> List[str]:
    aliases: List[str] = []
    seen = set()
    for value in values:
        text = _canonical(value)
        norm = _norm(text)
        if not text or not norm or norm in seen:
            continue
        seen.add(norm)
        aliases.append(text)
    return aliases


def _add_issue(
    report: Dict[str, Any],
    severity: str,
    code: str,
    message: str,
    *,
    pointer: str = "",
    **extra: Any,
) -> None:
    issue = {"code": code, "message": message}
    if pointer:
        issue["pointer"] = pointer
    issue.update(extra)
    bucket = "errors" if severity == "error" else "warnings"
    report.setdefault(bucket, []).append(issue)
    if severity == "error":
        report["ok"] = False


# ---------------------------------------------------------------------------
# Adapter part 1 -- report shape.
# ---------------------------------------------------------------------------


def ensure_resolution_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Seed the nested containers ``_resolve_compound_rows`` hard-indexes.

    ``_resolve_compound_rows`` was written against ``ir._new_report()`` and
    indexes its nested shape unguarded in three places: the legacy-id row and
    the non-legacy row both append to ``report["db_resolution"]["compounds"]``,
    and an unresolved row appends to ``report["unresolved"]["db_identities"]``.
    A caller passing a bare ``{}`` raised ``KeyError`` at the first of them, so
    no pre-freeze caller could use the function at all.

    Seeding is ``setdefault``-shaped: on a report that already carries the
    containers -- every in-IR call today -- it writes nothing, replaces nothing
    and reorders nothing. It is deliberately *not* a full ``_new_report()``:
    this module owns only the resolution slice and must not manufacture the
    rest of a run report it knows nothing about.
    """
    db_resolution = report.get("db_resolution")
    if not isinstance(db_resolution, dict):
        db_resolution = report["db_resolution"] = {}
    if not isinstance(db_resolution.get("compounds"), list):
        db_resolution["compounds"] = []

    unresolved = report.get("unresolved")
    if not isinstance(unresolved, dict):
        unresolved = report["unresolved"] = {}
    if not isinstance(unresolved.get("db_identities"), list):
        unresolved["db_identities"] = []
    return report


# ---------------------------------------------------------------------------
# Moved from ir.py:530-555, :558-575, :578-621, :797-897.
# ---------------------------------------------------------------------------


def _normalize_compound_external_ids(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    mapped = dict(_safe_dict(out.get("mapped_ids")))

    chebi = normalize_chebi_id(out.get("chebi_id") or mapped.get("chebi"))
    if chebi:
        out["chebi_id"] = chebi
        mapped["chebi"] = chebi

    for direct_key, mapped_key in [
        ("hmdb_id", "hmdb"),
        ("kegg_id", "kegg"),
        ("pubchem_cid", "pubchem"),
        ("pwc_id", "pwc_id"),
    ]:
        value = _first_nonempty(out, [direct_key, mapped_key])
        if value not in (None, ""):
            text = str(value).strip()
            if direct_key == "kegg_id":
                text = text.removeprefix("cpd:").removeprefix("CPD:")
            out[direct_key] = text
            mapped[mapped_key] = text

    if mapped:
        out["mapped_ids"] = {key: value for key, value in mapped.items() if value not in (None, "")}
    return out


def _compound_external_ids(row: Dict[str, Any]) -> Dict[str, Any]:
    mapped = _safe_dict(row.get("mapped_ids"))

    def _pick(*keys: str) -> Any:
        for key in keys:
            for source in (row, mapped):
                value = source.get(key)
                if value not in (None, ""):
                    return value
        return None

    return {
        "hmdb": _pick("hmdb_id", "hmdb"),
        "kegg": _pick("kegg_id", "kegg"),
        "chebi": _pick("chebi_id", "chebi"),
        "pubchem": _pick("pubchem_cid", "pubchem_id", "pubchem"),
        "drugbank": _pick("drugbank_id", "drugbank"),
    }


def _canonicalize_compound_offline(
    row: Dict[str, Any],
    *,
    name_index: Optional[PathwhizNameIndex],
    report: Dict[str, Any],
    apply_canonical_name: bool = True,
) -> None:
    """Apply the offline id->canonical-name mapping when the live DB didn't.

    Only fires when the row does not already carry a resolved ``db_row`` name
    (i.e. the live PathBank DB was unavailable or produced no confident match)
    and the row's external ids hit a real PathWhiz DB row in the offline index.
    Rows with no id hit are left untouched, preserving novel/extraction names.

    ``apply_canonical_name`` (keyword-only, adapter part 2) is the caller's
    control over the rename. The default ``True`` is today's behaviour exactly.
    ``False`` looks the canonical name up and **records what it found** under
    ``report["name_canonicalization"]["compounds_suppressed"]``, then leaves the
    row untouched -- no rename, no fabricated ``db_row``, no ``db_status``. That
    lets a pre-freeze caller attach identifiers without taking on the reference
    propagation the rename requires, and it leaves a provenance record either
    way, which the applied path does not do when the two names normalize alike.
    """
    if name_index is None:
        return
    existing_db_row = row.get("db_row") if isinstance(row.get("db_row"), dict) else {}
    if _canonical(existing_db_row.get("name")):
        return  # live-DB resolution already canonicalized this compound
    hit = name_index.compound_canonical(**_compound_external_ids(row))
    if not hit:
        return
    canonical = _canonical(hit.get("name"))
    if not canonical:
        return
    extraction_name = _canonical(row.get("name"))
    if not apply_canonical_name:
        report.setdefault("name_canonicalization", {}).setdefault("compounds_suppressed", []).append(
            {
                "from": extraction_name,
                "to": canonical,
                "matched_on": hit.get("matched_on"),
                "db_id": hit.get("id"),
                "source": "pathwhiz_id_db.json",
                "applied": False,
            }
        )
        return
    row.setdefault("raw_name", extraction_name)
    row["name"] = canonical
    # Provide a minimal db_row so the writer emits the canonical name and the
    # compound is treated as a trusted, DB-backed identity.
    row["db_row"] = {"id": hit.get("id"), "name": canonical}
    row["db_status"] = "matched_offline_name_index"
    if extraction_name and _norm(extraction_name) != _norm(canonical):
        aliases = _dedupe_aliases([*_safe_list(row.get("aliases")), extraction_name])
        if aliases:
            row["aliases"] = aliases
        report.setdefault("name_canonicalization", {}).setdefault("compounds", []).append(
            {
                "from": extraction_name,
                "to": canonical,
                "matched_on": hit.get("matched_on"),
                "db_id": hit.get("id"),
                "source": "pathwhiz_id_db.json",
            }
        )


# ---------------------------------------------------------------------------
# D-028 -- which DB matches may rename a row and stamp identifiers on it.
# ---------------------------------------------------------------------------

#: The confidence at or above which ``db_resolver.resolve`` only ever reports an
#: **identifier-backed** match. Already the literal ``0.85`` here; named because
#: D-028 makes it decide whether a resolution is *applied*, not merely whether a
#: failure is *logged* -- the defect this constant now closes.
DB_MATCH_CONFIDENCE_FLOOR = 0.85

#: **D-028 rule 1.** A fuzzy name match may never rename and never stamp
#: identifiers. ``resolve`` admits one on a 0.0006 tie-break (measured: ``OPDA``
#: -> ``Dinor-12-oxo-phytodienoate``, not a PathBank synonym of it), and
#: `PRODUCT_CONTRACT` §1 forbids inventing identities on that evidence.
RECORD_ONLY_MATCH_RULES = frozenset({"fuzzy_name"})

#: **D-028 rule 2.** The exact name rules that may rename and stamp, subject to
#: the short-abbreviation guard below.
EXACT_NAME_MATCH_RULES = frozenset({"exact_name", "exact_short_name_or_synonym"})

#: **D-028 rule 2**, the named constant the decision requires instead of a magic
#: number. A queried name of at most this many characters *after* ``_norm`` --
#: this module's existing normalization, unchanged -- is a short abbreviation,
#: and an exact name/synonym hit on it is not by itself evidence of identity.
#: Measured collisions: ``CL`` (2) -> "Chloride ion" but means cardiolipin;
#: ``THF`` (3) -> "Tetrahydrofuran" in one-carbon metabolism; ``G3P`` (3) ->
#: "3-Phosphoglyceric acid" but means glycerol-3-phosphate; ``PE`` (2) ->
#: "O-Phosphoethanolamine" but means phosphatidylethanolamine. The gold set names
#: the same failure class (``PMC13231680``/``PSA``). ``OPC-8:0`` normalizes to 7
#: and stays admissible.
SHORT_ABBREVIATION_MAX_CHARS = 4

#: **D-028 rule 2.** Exact identifiers that may corroborate a short abbreviation,
#: keyed as ``_compound_external_ids`` already returns them. ``drugbank`` is
#: carried by that helper but is deliberately NOT here: D-028 lists four, and
#: widening the set would be inventing policy.
CORROBORATING_ID_KEYS = ("kegg", "chebi", "pubchem", "hmdb")

#: **D-028 rule 2, corroboration is AGREEMENT, not presence.** Maps each
#: corroborating namespace to the matched PathBank row's column and the shared
#: normalizer both sides are compared through. Measured: ``PE`` carries
#: ``mapped_ids.kegg = C00012``, which is absent from ``compounds.kegg_id`` --
#: which is why no identifier rule matched it in the first place -- so under a
#: presence test its mere existence corroborated a synonym hit on a different
#: compound and admitted the confirmed-wrong ``PE -> O-Phosphoethanolamine``
#: rename. An identifier that disagrees with the matched row, or that has no
#: counterpart on it, corroborates nothing.
_CORROBORATION_COLUMNS = {
    "kegg": ("kegg_id", normalize_kegg_id),
    "chebi": ("chebi_id", normalize_chebi_id),
    "pubchem": ("pubchem_cid", normalize_pubchem_cid),
    "hmdb": ("hmdb_id", normalize_hmdb_id),
}


def _db_match_rule_decision(row: Dict[str, Any], match: Dict[str, Any]) -> Dict[str, Any]:
    """Weigh the MATCH: does this rule, at this confidence, justify an identity?

    D-028's half of :func:`_admit_db_identity`, unchanged. The row's own refusal
    record is **not** consulted here -- that is a property of the row, not of the
    match, and it is applied by the caller below.

    Returns the decision **as a record**, never a bare bool: D-028 rule 4
    requires a refusal to be recorded for review rather than silently dropped,
    and this dict is what the caller files. It never raises -- a refused match
    must not abort an export -- and never mutates ``row`` or ``match``.

    The rule vocabulary is closed. ``resolve`` returns ``status == "matched"``
    under exactly three families: identifier rules (all at or above
    :data:`DB_MATCH_CONFIDENCE_FLOOR` by construction, ``pubchem_cid_exact``
    lowest at 0.85), the two exact name rules (fixed at 0.70) and ``fuzzy_name``
    (hard-capped at 0.65). Rules 1 and 2 cover the last two; the first is
    admitted by the floor, which is what the floor was written to mean.

    Anything below the floor matching no known rule fails **closed** -- not extra
    policy: `PRODUCT_CONTRACT` §8 forbids accepting an identifier because its
    shape is valid, and the module's own floor already called it untrustworthy.
    """
    rule = str(match.get("chosen_rule") or "")
    confidence = float(match.get("confidence") or 0.0)
    queried_name = _canonical(match.get("raw_name") or row.get("name"))
    decision: Dict[str, Any] = {
        "admitted": False,
        "rule": rule,
        "confidence": confidence,
        "queried_name": queried_name,
        "reason": "",
    }

    if match.get("status") != "matched" or not _safe_dict(match.get("chosen")):
        decision["reason"] = "no_match_to_admit"
        return decision

    if rule in RECORD_ONLY_MATCH_RULES:
        decision["reason"] = "fuzzy_name_match_never_admitted"
        return decision

    if rule in EXACT_NAME_MATCH_RULES:
        normalized_length = len(_norm(queried_name))
        decision["normalized_length"] = normalized_length
        if normalized_length > SHORT_ABBREVIATION_MAX_CHARS:
            decision["admitted"] = True
            decision["reason"] = "exact_name_or_synonym_match"
            return decision
        chosen = _safe_dict(match.get("chosen"))
        external_ids = _compound_external_ids(row)
        corroborating: List[str] = []
        disagreeing: List[str] = []
        for key in CORROBORATING_ID_KEYS:
            column, normalize = _CORROBORATION_COLUMNS[key]
            ours = normalize(external_ids.get(key))
            if not ours:
                continue
            theirs = normalize(chosen.get(column))
            if theirs and ours.casefold() == theirs.casefold():
                corroborating.append(key)
            else:
                disagreeing.append(key)
        decision["corroborating_ids"] = corroborating
        decision["disagreeing_ids"] = disagreeing
        decision["admitted"] = bool(corroborating)
        if corroborating:
            decision["reason"] = "short_abbreviation_corroborated_by_exact_identifier"
        elif disagreeing:
            decision["reason"] = "short_abbreviation_identifier_disagrees_with_match"
        else:
            decision["reason"] = "short_abbreviation_without_corroborating_identifier"
        return decision

    if confidence >= DB_MATCH_CONFIDENCE_FLOOR:
        decision["admitted"] = True
        decision["reason"] = "identifier_match_at_or_above_confidence_floor"
        return decision

    decision["reason"] = "unrecognized_rule_below_confidence_floor"
    return decision


#: **F-099 / C-078.** Every namespace ``apply_compound_db_resolution`` can write
#: back onto a row, mapped to the one token both sides are compared through.
#:
#: The two sides spell the same fact differently. A row's refusal record --
#: ``mapping_meta.rejected_mapped_ids`` -- is keyed like ``mapped_ids``
#: (``map_ids._strip_rejected_identifiers`` builds it from that dict plus the
#: ``pathbank_compound_id`` scalar), while the apply writes both ``mapped_ids``
#: *and* the scalar columns ``kegg_id`` / ``chebi_id`` / ``pathwhiz_id`` /
#: ``pw_compound_id`` for the same identifiers. Comparing raw keys would miss a
#: refusal spelled the other way, so both sides normalize through this map first.
#:
#: A key the map does not know normalizes to **itself**, so an unrecognized
#: refusal namespace can never collide by accident, and a namespace the apply
#: never writes can never manufacture a refusal.
_REFUSABLE_NAMESPACE = {
    "pathbank_compound_id": "pathbank_compound_id",
    "pw_compound_id": "pathbank_compound_id",
    "pathwhiz_id": "pathbank_compound_id",
    "db_id": "pathbank_compound_id",
    "hmdb": "hmdb",
    "hmdb_id": "hmdb",
    "kegg": "kegg",
    "kegg_id": "kegg",
    "chebi": "chebi",
    "chebi_id": "chebi",
    "pubchem": "pubchem",
    "pubchem_id": "pubchem",
    "pubchem_cid": "pubchem",
    "cas": "cas",
    "biocyc": "biocyc",
    "biocyc_id": "biocyc",
    "chemspider": "chemspider",
    "chemspider_id": "chemspider",
    "drugbank": "drugbank",
    "drugbank_id": "drugbank",
    "pwc_id": "pwc_id",
}

#: :func:`_admit_db_identity`'s reason when the C-078 veto fires. Distinct from
#: every D-028 rule reason: those say the match was too weak, this says the match
#: may have been strong enough and is refused anyway.
REFUSED_IDENTITY_WOULD_BE_RESTORED = "db_match_would_restore_refused_identifiers"


def _id_namespace(key: Any) -> str:
    return _REFUSABLE_NAMESPACE.get(str(key).strip().casefold(), str(key).strip().casefold())


def _restored_refused_namespaces(row: Dict[str, Any], chosen: Dict[str, Any]) -> List[str]:
    """Namespaces this row already refused that ``chosen`` would write back.

    ``_mapped_ids_from_row`` is the same function the apply uses to rebuild
    ``mapped_ids``, so this asks the question in the apply's own terms -- "which
    of these identifiers would land on the row" -- rather than re-deriving a
    parallel list that could drift from it. It drops empty values, which is why
    a matched DB row carrying nothing in a refused namespace restores nothing and
    is not vetoed.

    Never raises and never mutates either argument.
    """
    rejected = _safe_dict(_safe_dict(row.get("mapping_meta")).get("rejected_mapped_ids"))
    if not rejected or not chosen:
        return []
    refused = {_id_namespace(key) for key in rejected}
    stamped = {_id_namespace(key) for key in _mapped_ids_from_row(chosen)}
    return sorted(refused & stamped)


def _admit_db_identity(row: Dict[str, Any], match: Dict[str, Any]) -> Dict[str, Any]:
    """Decide whether ``match`` may rename ``row`` and stamp identifiers on it.

    Two questions, and until C-078 only the first was asked.
    :func:`_db_match_rule_decision` weighs the **match** under D-028. This adds
    the one fact that belongs to the **row**: *a DB match may not stamp an
    identity that this row's own ``mapping_meta.rejected_mapped_ids`` already
    refused* (F-099). On a collision the match becomes **record-only** -- the
    caller already passes ``admit_identity=admission["admitted"]`` to
    ``apply_compound_db_resolution``, whose one flag and one early return give
    exactly D-028 rule 3's "no rename AND no identifier stamp, never a partial
    apply", and whose ``IDENTITY_REFUSED_STATUS`` keeps the row in ``resolved``
    with its extracted name for review (merge rule 7).

    Why it is needed here, measured: the identity gate withholds an identifier by
    moving it to ``rejected_mapped_ids`` and clearing it from ``mapped_ids`` and
    its scalar column. Clearing the PathBank scalar is what makes ``legacy_id``
    ``None`` at the caller, so the row falls through to a **name-keyed** lookup,
    and ``apply_compound_db_resolution`` then rebuilds ``mapped_ids`` **wholesale**
    from the matched DB row. On ``runs_verify/2026-08-22_2147`` and
    ``.../2026-08-21_2239``, ``Fe3+`` (PMC12096016/research, extracted as *ferric
    iron*) is that path completed: ``rejected_mapped_ids = {kegg: C14819}`` and
    the committed ``final_mapped.json`` ships ``mapped_ids.kegg = C14819`` again,
    byte-equal to ``_mapped_ids_from_row(db_match.chosen)``. Pre-freeze, so the
    restored identity is canonical (``PRODUCT_CONTRACT`` §5), and no later stage
    re-imposes the refusal.

    **The veto only ever overrides an admission.** A decision the match rules
    already refused keeps its own reason -- ``fuzzy_name_match_never_admitted``
    and the short-abbreviation reasons are unchanged, and no refusal is ever
    turned into an admission -- so the only behaviour this changes is the one
    F-099 names. The overridden reason is preserved as ``admitted_by_match_rule``
    so the record still says what the match was worth.
    """
    decision = _db_match_rule_decision(row, match)
    restored = _restored_refused_namespaces(row, _safe_dict(match.get("chosen")))
    if restored:
        decision["refused_namespaces"] = restored
        if decision.get("admitted"):
            decision["admitted"] = False
            decision["admitted_by_match_rule"] = decision["reason"]
            decision["reason"] = REFUSED_IDENTITY_WOULD_BE_RESTORED
    return decision


# ---------------------------------------------------------------------------
# Resolver selection -- the third state (F-129).
# ---------------------------------------------------------------------------

#: The ``db_resolution.reason`` recorded when the caller passed
#: :data:`NO_DB_RESOLVER`. Deliberately **not** one of the reasons already in
#: this vocabulary: ``db_not_configured`` means no ambient PathBank settings were
#: found, ``harvest_db_down`` / ``db_unavailable`` mean a configured resolver
#: answered ``available() is False``, and ``db_resolver_unavailable:<exc>`` means
#: constructing one raised. All three say *a lookup was attempted and did not
#: succeed*. This one says the caller **asked for no lookup at all**, which is
#: not a failure, and a report that spelled it as one would be untrue about the
#: run. It is added in the same spirit as the existing distinction between
#: ``db_not_configured`` and ``harvest_db_down``.
DB_RESOLUTION_DISABLED_REASON = "db_resolution_disabled_by_caller"


class _NoDbResolver:
    """The type of :data:`NO_DB_RESOLVER`; recognised by identity, never by shape.

    It deliberately implements **no** resolver protocol -- no ``available``, no
    ``resolve``, no ``last_error``. :func:`_resolve_compound_rows` recognises the
    **singleton**, by ``is``, and never a shape: giving this class a shape would
    only create a second, quieter way of meaning the same thing, which is the
    defect being closed rather than the fix.

    **What a non-singleton instance actually does -- measured, because the first
    draft of this paragraph guessed and guessed wrong.** It claimed such an
    instance would "fail open, back onto the ambient database", and that anything
    receiving it "must fail visibly". Neither is true, and the truth is worse. A
    fresh ``_NoDbResolver()`` is not the singleton and is not ``None``, so it
    reaches **neither** arm of the selection in :func:`_resolve_compound_rows`;
    the availability ladder below those arms then defaults its missing
    ``available`` to ``True`` -- ``getattr(db_resolver, "available",
    lambda: True)()`` -- wraps the impostor in :class:`PathWhizCompoundResolver`
    and reports (``evidence/c096_impostor_probe.py``, G11 ``C-096/20``, with the
    ambient database stubbed out so nothing else can contribute)::

        singleton       available=False reason='db_resolution_disabled_by_caller'
        fresh impostor  available=True  reason=<ABSENT>
        None            available=False reason='db_not_configured'

    ``available: True`` carrying **no reason at all**: a silent false
    availability, on a population that resolved nothing, which suppresses the
    preflight warning D-032 clause 6 rules product-visible export content. So the
    failure is neither open nor visible -- it is an unresolved population reported
    as though a database had answered it.

    That ``getattr`` default is pre-existing and is not this card's to change,
    which is exactly why identity is defended here instead: copying returns the
    singleton itself, because ``resolve_compounds_prefreeze`` deep-copies the rows
    it resolves and a future caller could as easily deep-copy a kwargs dict.
    ``tests/test_c096_explicit_no_resolver.py`` pins that under ``copy``,
    ``deepcopy`` and ``pickle``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "NO_DB_RESOLVER"

    def __copy__(self) -> "_NoDbResolver":
        return self

    def __deepcopy__(self, _memo: Any) -> "_NoDbResolver":
        return self

    def __reduce__(self) -> Any:
        return (_no_db_resolver, ())


def _no_db_resolver() -> "_NoDbResolver":
    """Pickle/copy hook for :data:`NO_DB_RESOLVER`. Returns the singleton."""
    return NO_DB_RESOLVER


#: **Resolve nothing against a database** -- the third resolver selection, and
#: the only one that says what it means. The full vocabulary is now:
#:
#: ``db_resolver=<resolver>``
#:     use exactly this resolver, unchanged.
#: ``db_resolver=None``
#:     *unspecified* -- open the ambient PathBank connection. **Unchanged, and
#:     load-bearing:** ``PRODUCT_CONTRACT`` §8 forbids an exporter opening one, so
#:     the pre-freeze call is the one that must (D-015, D-032 clause 6, and
#:     ``prefreeze_resolution.resolve_compounds_prefreeze``'s docstring).
#: ``db_resolver=NO_DB_RESOLVER``
#:     resolve against no database; record it as the caller's decision.
#:
#: Before this existed ``None`` carried both the first and third meanings, so the
#: third was unreachable: a caller that deliberately disabled DB resolution was
#: handed the ambient live database instead and nothing in the report said so
#: (F-129). Adding a value fixes that without redefining ``None``, which could
#: not be redefined without pushing the connection into the exporter.
NO_DB_RESOLVER = _NoDbResolver()


def _resolve_compound_rows(
    rows: List[Dict[str, Any]],
    *,
    db_resolver: Any,
    strict_db: bool,
    report: Dict[str, Any],
    pointer_prefix: str,
    name_index: Optional[PathwhizNameIndex] = None,
    apply_canonical_name: bool = True,
) -> List[Dict[str, Any]]:
    ensure_resolution_report(report)
    normalized = [_normalize_compound_external_ids(row) for row in rows]
    resolver: Optional[PathWhizCompoundResolver] = None
    db_reason = ""

    if db_resolver is NO_DB_RESOLVER:
        # The caller said "resolve nothing against a database". Recorded under
        # its own reason so the report never claims a lookup failed, and
        # ``db_reason`` is non-empty from here on, which is what stops either arm
        # of the availability ladder below overwriting it.
        #
        # **The ``elif`` is the load-bearing part, not the order of the arms.**
        # An earlier comment here said the sentinel had to be matched "BEFORE the
        # ambient substitution"; that is not the invariant. Both conditions are
        # identity tests against distinct objects, so they are mutually exclusive
        # and swapping the two arms is a measured no-op. What must never happen is
        # the two becoming independent ``if``s: this arm sets ``db_resolver =
        # None`` deliberately, to rejoin the same downstream ladder an
        # unconfigured ambient reaches, so a second ``if db_resolver is None``
        # would immediately substitute ``PathBankDbResolver.from_env()`` and
        # resolve the explicitly offline caller against the live database after
        # all. Mutation-probed both ways, on this tree: swapping the arms leaves
        # both test files green (22 passed, G11 ``C-096/21``), while splitting the
        # ``elif`` into two ``if``s is killed by 9 tests across them, the four
        # F-129 repairs included (9 failed / 13 passed, G11 ``C-096/22``).
        db_resolver = None
        db_reason = DB_RESOLUTION_DISABLED_REASON
    elif db_resolver is None:
        try:
            from t2pw.mapping.map_ids import PathBankDbResolver

            db_resolver = PathBankDbResolver.from_env()
        except Exception as exc:  # noqa: BLE001
            db_reason = f"db_resolver_unavailable:{exc}"
            db_resolver = None

    available = bool(db_resolver is not None and getattr(db_resolver, "available", lambda: True)())
    if db_resolver is not None and available:
        resolver = PathWhizCompoundResolver(db_resolver)
    elif db_resolver is not None and not db_reason:
        db_reason = str(getattr(db_resolver, "last_error", "") or "db_unavailable")
    elif not db_reason:
        db_reason = "db_not_configured"

    # Surface whether the live resolution DB was actually consulted so the
    # preflight can warn when compound names may be non-canonical.
    db_resolution = report.setdefault("db_resolution", {})
    db_resolution["available"] = resolver is not None
    if db_reason:
        db_resolution["reason"] = db_reason

    resolved: List[Dict[str, Any]] = []
    for idx, row in enumerate(normalized):
        raw_name = _canonical(row.get("raw_name") or row.get("name"))
        legacy_id = _db_id(row, ["pathbank_compound_id", "pw_compound_id", "pathwhiz_id"])
        if legacy_id is not None:
            fallback = dict(row)
            fallback["db_status"] = "legacy_id_unverified"
            fallback["db_id"] = legacy_id
            fallback["chosen_rule"] = "legacy_pathwhiz_id_unverified"
            fallback["confidence"] = max(float(fallback.get("confidence") or 0.0), 0.85)
            report["db_resolution"]["compounds"].append(
                {
                    "raw_name": raw_name,
                    "status": "legacy_id_unverified",
                    "db_id": legacy_id,
                    "chosen_rule": "legacy_pathwhiz_id_unverified",
                    "confidence": fallback["confidence"],
                    "reason": "payload_pathwhiz_id",
                }
            )
            resolved.append(fallback)
            continue

        if resolver is None:
            match = {
                "status": "unmatched",
                "raw_name": raw_name,
                "chosen": None,
                "candidates": [],
                "chosen_rule": "",
                "confidence": 0.0,
                "reason": db_reason,
            }
        else:
            match = resolver.resolve(row)

        report["db_resolution"]["compounds"].append(match)
        # D-028. The gate now decides whether the resolution is APPLIED, not only
        # whether a failure is logged: both arms below used to apply it anyway.
        admission = _admit_db_identity(row, match)
        high_confidence = (
            match.get("status") == "matched"
            and float(match.get("confidence") or 0.0) >= DB_MATCH_CONFIDENCE_FLOOR
        )
        if high_confidence and admission["admitted"]:
            resolved.append(apply_compound_db_resolution(row, match))
            continue

        issue = {
            "entity_type": "compound",
            "raw_name": raw_name,
            "status": match.get("status"),
            "reason": match.get("reason") or "low_confidence_db_match",
            "chosen_rule": match.get("chosen_rule"),
            "confidence": match.get("confidence", 0.0),
            "candidates": match.get("candidates", []),
            # D-028 rule 4: a refusal is recorded for review, never dropped and
            # never raised. A refused row stays in `resolved` with its extracted
            # name, so an incomplete-but-correct pathway survives as
            # review_required (merge rule 7) instead of losing the compound.
            "admission": admission,
        }
        report["unresolved"]["db_identities"].append(issue)
        _add_issue(
            report,
            "error" if strict_db else "warning",
            "compound_db_resolution_failed",
            f"Compound '{raw_name or idx}' did not resolve to a confident PathWhiz DB row.",
            pointer=f"{pointer_prefix}/{idx}",
            **issue,
        )
        resolved.append(
            apply_compound_db_resolution(row, match, admit_identity=admission["admitted"])
        )

    for row in resolved:
        _canonicalize_compound_offline(
            row,
            name_index=name_index,
            report=report,
            apply_canonical_name=apply_canonical_name,
        )
    return resolved

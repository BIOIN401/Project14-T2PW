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

The nine private helpers below are **verbatim copies** of ``ir.py:43-96``,
``:183-193`` and ``:244-260``; the originals are unmodified. Duplication is
forced -- ``ir.py`` imports this module, so importing back would be a cycle --
so ``tests/test_compound_resolution_extraction.py`` pins every copy to its
original by source equality, which is what stops the two drifting (SPIKE-002 F-3).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from t2pw.pwml.db_resolver import PathWhizCompoundResolver, apply_compound_db_resolution, normalize_chebi_id
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

    if db_resolver is None:
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
        if match.get("status") == "matched" and float(match.get("confidence") or 0.0) >= 0.85:
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
        resolved.append(apply_compound_db_resolution(row, match))

    for row in resolved:
        _canonicalize_compound_offline(
            row,
            name_index=name_index,
            report=report,
            apply_canonical_name=apply_canonical_name,
        )
    return resolved

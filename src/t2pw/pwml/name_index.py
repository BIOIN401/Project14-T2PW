"""Offline PathWhiz/PathBank canonical-name index.

The live PathBank MySQL database is the authoritative source for the canonical
name of a compound or species. When that DB is *not* configured (the common
offline case), the pipeline still needs to emit the DB row's canonical name so
that the PathWhiz importer's ``find_by(name + external ids)`` lookup hits the
existing row instead of colliding on a unique key (``compounds.hmdb_id`` /
``species.taxonomy_id``) and failing to save.

This module treats ``data/pathwhiz_id_db.json`` (built by
``build_pathwhiz_id_db.py`` from the real PathBank .pwml corpus) as an offline
source of truth for the ``external id -> canonical name`` mapping. It only ever
provides a canonical name for an entity that *matches a real DB row by id*;
entities with no id hit are left untouched so novel/T2PW-invented entities keep
their extraction names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from t2pw.pwml.db_resolver import (
    normalize_chebi_id,
    normalize_empty,
    normalize_hmdb_id,
    normalize_kegg_id,
)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _hmdb_variants(value: Optional[str]) -> list[str]:
    """HMDB is stored inconsistently (with/without leading zeros)."""
    text = normalize_hmdb_id(value)
    if not text:
        return []
    variants = {text}
    digits = text[4:].lstrip("0")
    if digits:
        variants.add(f"HMDB{digits.zfill(7)}")
        variants.add(f"HMDB{digits}")
    return list(variants)


class PathwhizNameIndex:
    """Resolve canonical names for compounds/species/proteins from the id-db JSON."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = _safe_dict(data)
        self._compounds = _safe_dict(self._data.get("compounds"))
        self._species = _safe_dict(self._data.get("species"))
        self._proteins = _safe_dict(self._data.get("proteins"))

    # ---- compounds -----------------------------------------------------
    def compound_canonical(
        self,
        *,
        hmdb: Any = None,
        kegg: Any = None,
        chebi: Any = None,
        pubchem: Any = None,
        drugbank: Any = None,
    ) -> Optional[Dict[str, Any]]:
        by_id = _safe_dict(self._compounds.get("by_id"))
        if not by_id:
            return None

        # Try id types in descending strength, matching the live-DB resolver.
        lookups = []
        for variant in _hmdb_variants(hmdb):
            lookups.append(("hmdb", variant))
        kegg_norm = normalize_kegg_id(kegg)
        if kegg_norm:
            lookups.append(("kegg", kegg_norm))
        chebi_norm = normalize_chebi_id(chebi)
        if chebi_norm:
            lookups.append(("chebi", chebi_norm))
            lookups.append(("chebi", f"CHEBI:{chebi_norm}"))
        pubchem_norm = normalize_empty(pubchem)
        if pubchem_norm:
            lookups.append(("pubchem", pubchem_norm))
        drugbank_norm = normalize_empty(drugbank)
        if drugbank_norm:
            lookups.append(("drugbank", drugbank_norm))

        for index_key, value in lookups:
            index = _safe_dict(self._compounds.get(index_key))
            cid = index.get(value)
            if cid is None:
                continue
            entry = _safe_dict(by_id.get(str(cid)))
            name = normalize_empty(entry.get("name"))
            if not name:
                continue
            return {
                "id": _to_int(cid),
                "name": name,
                "hmdb_id": normalize_hmdb_id(entry.get("hmdb")),
                "kegg_id": normalize_kegg_id(entry.get("kegg")),
                "chebi_id": normalize_chebi_id(entry.get("chebi")),
                "pubchem_cid": normalize_empty(entry.get("pubchem")),
                "drugbank_id": normalize_empty(entry.get("drugbank")),
                "matched_on": index_key,
            }
        return None

    # ---- species -------------------------------------------------------
    def species_canonical(
        self,
        *,
        taxonomy_id: Any = None,
        pathbank_species_id: Any = None,
        name: Any = None,  # noqa: ARG002 - reserved for future name-based lookup
    ) -> Optional[Dict[str, Any]]:
        by_id = _safe_dict(self._species.get("by_id"))
        if not by_id:
            return None

        sid = _to_int(pathbank_species_id)
        if sid is not None:
            entry = _safe_dict(by_id.get(str(sid)))
            resolved = _species_entry_to_dict(entry)
            if resolved is not None:
                return resolved

        tax = normalize_empty(taxonomy_id)
        if tax:
            tax_index = _safe_dict(self._species.get("taxonomy"))
            cid = tax_index.get(tax)
            if cid is not None:
                entry = _safe_dict(by_id.get(str(cid)))
                resolved = _species_entry_to_dict(entry)
                if resolved is not None:
                    return resolved
        return None

    # ---- proteins (available for future use; not auto-applied) ----------
    def protein_canonical(self, *, uniprot: Any = None) -> Optional[Dict[str, Any]]:
        by_id = _safe_dict(self._proteins.get("by_id"))
        uid = normalize_empty(uniprot)
        if not by_id or not uid:
            return None
        index = _safe_dict(self._proteins.get("uniprot"))
        pid = index.get(uid) or index.get(uid.upper())
        if pid is None:
            return None
        entry = _safe_dict(by_id.get(str(pid)))
        name = normalize_empty(entry.get("name"))
        if not name:
            return None
        return {"id": _to_int(pid), "name": name, "uniprot": normalize_empty(entry.get("uniprot"))}


def _species_entry_to_dict(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = normalize_empty(entry.get("name"))
    if not name:
        return None
    return {
        "name": name,
        "taxonomy_id": normalize_empty(entry.get("taxonomy_id") or entry.get("taxonomy")),
        "classification": normalize_empty(entry.get("classification")),
    }


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def load_name_index(path: Path) -> Optional[PathwhizNameIndex]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing/broken index must never break export
        return None
    if not isinstance(data, dict):
        return None
    return PathwhizNameIndex(data)


_DEFAULT_INDEX_UNSET = object()
_default_index_cache: Any = _DEFAULT_INDEX_UNSET


def _default_index_path() -> Optional[Path]:
    # src/t2pw/pwml/name_index.py -> repo root is parents[3]
    root = Path(__file__).resolve().parents[3]
    candidate = root / "data" / "pathwhiz_id_db.json"
    return candidate if candidate.exists() else None


def default_name_index() -> Optional[PathwhizNameIndex]:
    """Load (and cache) the repo's ``data/pathwhiz_id_db.json`` index if present."""
    global _default_index_cache
    if _default_index_cache is _DEFAULT_INDEX_UNSET:
        path = _default_index_path()
        _default_index_cache = load_name_index(path) if path is not None else None
    return _default_index_cache

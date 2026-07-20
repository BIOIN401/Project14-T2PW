"""Maintainer command: refresh the offline name index from the resolution DB.

The offline canonical-name index (``data/pathwhiz_id_db.json``) is what lets
*offline* generation runs still emit DB-canonical compound/species names. It was
originally built from a small .pwml sample and therefore covers only a handful
of entities. This module repopulates its ``compounds`` and ``species`` sections
directly from the live resolution DB so offline runs canonicalize far more
entities (and the species section -- currently empty -- becomes usable).

Nothing here is invoked during normal generation or in tests; it is a manual
maintenance step. Connection settings come exclusively from the environment /
``.env`` (see ``docs/setup.md``); no host or credential is hardcoded.

Usage::

    python -m t2pw.pwml.refresh_name_index                 # writes data/pathwhiz_id_db.json
    python -m t2pw.pwml.refresh_name_index --out other.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from t2pw.paths import PROJECT_ROOT
from t2pw.pwml.db_resolver import (
    normalize_chebi_id,
    normalize_empty,
    normalize_hmdb_id,
    normalize_kegg_id,
)

DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "pathwhiz_id_db.json"


def _fetch_all(db: Any, sql: str) -> List[Dict[str, Any]]:
    query = getattr(db, "_query", None)
    if not callable(query):
        return []
    return [dict(row) for row in query(sql, tuple()) if isinstance(row, dict)]


def build_compound_section(db: Any) -> Dict[str, Any]:
    section: Dict[str, Any] = {
        "hmdb": {},
        "kegg": {},
        "chebi": {},
        "pubchem": {},
        "drugbank": {},
        "by_id": {},
    }
    rows = _fetch_all(
        db,
        "SELECT id, name, hmdb_id, kegg_id, chebi_id, pubchem_cid, drugbank_id FROM compounds",
    )
    for row in rows:
        cid = normalize_empty(row.get("id"))
        name = normalize_empty(row.get("name"))
        if not cid or not name:
            continue
        hmdb = normalize_hmdb_id(row.get("hmdb_id"))
        kegg = normalize_kegg_id(row.get("kegg_id"))
        chebi = normalize_chebi_id(row.get("chebi_id"))
        pubchem = normalize_empty(row.get("pubchem_cid"))
        drugbank = normalize_empty(row.get("drugbank_id"))
        section["by_id"][cid] = {
            "name": name,
            "hmdb": hmdb,
            "kegg": kegg,
            "chebi": chebi,
            "pubchem": pubchem,
            "drugbank": drugbank,
        }
        if hmdb:
            section["hmdb"][hmdb] = int(cid) if cid.isdigit() else cid
            digits = hmdb[4:].lstrip("0")
            if digits:
                section["hmdb"][f"HMDB{digits.zfill(7)}"] = int(cid) if cid.isdigit() else cid
        if kegg:
            section["kegg"][kegg] = int(cid) if cid.isdigit() else cid
        if chebi:
            section["chebi"][chebi] = int(cid) if cid.isdigit() else cid
            section["chebi"][f"CHEBI:{chebi}"] = int(cid) if cid.isdigit() else cid
        if pubchem:
            section["pubchem"][pubchem] = int(cid) if cid.isdigit() else cid
        if drugbank:
            section["drugbank"][drugbank] = int(cid) if cid.isdigit() else cid
    return section


def build_species_section(db: Any) -> Dict[str, Any]:
    section: Dict[str, Any] = {"taxonomy": {}, "by_id": {}}
    rows = _fetch_all(db, "SELECT id, name, common_name, taxonomy_id FROM species")
    for row in rows:
        sid = normalize_empty(row.get("id"))
        name = normalize_empty(row.get("name"))
        if not sid or not name:
            continue
        taxonomy = normalize_empty(row.get("taxonomy_id"))
        section["by_id"][sid] = {
            "name": name,
            "taxonomy_id": taxonomy,
            "common_name": normalize_empty(row.get("common_name")),
        }
        if taxonomy:
            section["taxonomy"][taxonomy] = int(sid) if sid.isdigit() else sid
    return section


def refresh_name_index(
    *,
    db: Any = None,
    out_path: Optional[Path] = None,
) -> Dict[str, int]:
    """Rebuild the compounds+species sections of the offline index from the DB.

    ``db`` may be supplied for testing; otherwise a live resolver is built from
    the environment. Other sections of the existing JSON are preserved. Returns a
    small summary of the row counts written.
    """
    out = Path(out_path) if out_path is not None else DEFAULT_INDEX_PATH

    if db is None:
        from t2pw.mapping.map_ids import PathBankDbResolver

        db = PathBankDbResolver.from_env()
        if db is None or not db.available():
            reason = getattr(db, "last_error", "") or "db_not_configured"
            raise RuntimeError(f"resolution DB not available for index refresh: {reason}")

    existing: Dict[str, Any] = {}
    if out.exists():
        try:
            loaded = json.loads(out.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:  # noqa: BLE001 - a broken existing file is simply replaced
            existing = {}

    compounds = build_compound_section(db)
    species = build_species_section(db)
    existing["compounds"] = compounds
    existing["species"] = species

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "compounds": len(compounds["by_id"]),
        "species": len(species["by_id"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_INDEX_PATH), help="Output index JSON path")
    args = ap.parse_args()
    summary = refresh_name_index(out_path=Path(args.out))
    print(
        f"Refreshed {args.out}: {summary['compounds']} compounds, "
        f"{summary['species']} species"
    )


if __name__ == "__main__":
    main()

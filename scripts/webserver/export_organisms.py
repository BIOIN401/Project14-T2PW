"""Freeze every PathBank species to ``data/webserver/organisms.json`` (task W2, D-W04).

This is the ONLY code that touches the PathBank database for the organism list;
``t2pw.service.organisms`` reads the committed JSON at runtime.

Source of truth: the ``species`` table (``id, name, taxonomy_id, common_name,
classification``). ``pathways.species_id`` -> ``species.id`` is used only for the
``has_pathways`` flag. Pathway accessions (PW ids) are NOT used here: they resolve
through ``spv.pw_id`` and ``pathways.id`` lookups return the wrong pathways.

Scope string: the pipeline matches ``RequestedScope.requested_organism`` on the
normalised binomial (``rag.eligibility._taxon`` drops strain suffixes), so the
stripped PathBank ``name`` is passed through verbatim -- no remapping.

Re-run (read-only, SELECT only)::

    PYTHONPATH=src python scripts/webserver/export_organisms.py \
        [--env-file <path to .env>] [--out data/webserver/organisms.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = 1
SPECIES_SQL = (
    "SELECT id, name, taxonomy_id, common_name, classification FROM species ORDER BY id"
)
PATHWAY_SPECIES_SQL = "SELECT DISTINCT species_id FROM pathways WHERE species_id IS NOT NULL"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "webserver" / "organisms.json"


def _clean(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text or None


def build_entries(species_rows: Iterable[Dict[str, Any]], pathway_species_ids: Iterable[Any]) -> List[Dict[str, Any]]:
    """Pure transform: DB rows -> sorted, unique-label entries. No I/O."""
    with_pathways = {int(i) for i in pathway_species_ids}
    entries: List[Dict[str, Any]] = []
    for row in species_rows:
        name = _clean(row.get("name"))
        if name is None:  # nothing to put in a drop-down
            continue
        sid = int(row["id"])
        entries.append({
            "id": sid,
            "label": name,
            "scope": name,
            "taxonomy_id": _clean(row.get("taxonomy_id")),
            "common_name": _clean(row.get("common_name")),
            "classification": _clean(row.get("classification")),
            "has_pathways": sid in with_pathways,
        })
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_label[e["label"].casefold()].append(e)
    for group in by_label.values():
        if len(group) > 1:  # PathBank allows duplicate names; a drop-down cannot
            for e in group:
                e["label"] = f"{e['label']} [PathBank id {e['id']}]"
    entries.sort(key=lambda e: (e["label"].casefold(), e["id"]))
    return entries


def fetch_entries(cursor: Any) -> List[Dict[str, Any]]:
    cursor.execute(SPECIES_SQL)
    species = list(cursor.fetchall())
    cursor.execute(PATHWAY_SPECIES_SQL)
    pw_ids = [r["species_id"] for r in cursor.fetchall()]
    return build_entries(species, pw_ids)


def render(entries: List[Dict[str, Any]]) -> str:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "source": "PathBank MySQL: species (+ pathways.species_id for has_pathways)",
        "count": len(entries),
        "organisms": entries,
    }
    head = json.dumps({k: v for k, v in doc.items() if k != "organisms"}, ensure_ascii=False, indent=1)
    # one compact object per species per line: a re-export diffs one line per species
    lines = ",\n".join("  " + json.dumps(e, ensure_ascii=False, separators=(",", ":")) for e in entries)
    return head[:-2] + ',\n "organisms": [\n' + lines + "\n ]\n}\n"


def _safe_error(exc: BaseException) -> str:
    """Exception class name plus an integer errno only; the text can echo host/user."""
    errno = exc.args[0] if exc.args and isinstance(exc.args[0], int) else None
    return type(exc).__name__ + (f" (errno {errno})" if errno is not None else "")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--env-file", type=Path, default=None, help="path to the gitignored .env")
    args = ap.parse_args(argv)
    if args.env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=args.env_file, override=False)
    from t2pw.bench.eval_v2.pathbank_gold import connect  # repo's own helper

    try:
        conn = connect()
    except Exception as exc:  # noqa: BLE001 - pymysql/config texts carry host and user: never echo them
        print(f"export_organisms: BLOCKED: {_safe_error(exc)}", file=sys.stderr)
        return 2
    try:
        entries = fetch_entries(conn.cursor())  # SELECT only
    finally:
        conn.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(entries), encoding="utf-8", newline="\n")
    print(f"export_organisms: wrote {len(entries)} species to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

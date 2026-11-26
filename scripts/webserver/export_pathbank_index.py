"""Build the smart-search PathBank index ``data/webserver/pathbank_index.sqlite`` (task W10).

The ONLY code that touches the PathBank database for the search; ``t2pw.web.pathbank``
reads the SQLite file at runtime. The file is not committed (about 520k public pathways);
re-run this on each server after a PathBank release.

Source: ``super_pathway_visualizations`` (one row per public pathway drawing), filtered to
``named_for_type='Pathway' AND private_pathway=0 AND export=1``, with ``pw_id`` (the public
PW accession), ``smpdb_id``, ``cached_name``, ``cached_species_id`` and ``cached_subject``.
PW accessions are never looked up in ``pathways.id`` (that returns other pathways).

Run (read-only, SELECT only)::

    PYTHONPATH=src python scripts/webserver/export_pathbank_index.py \
        [--env-file <path to .env>] [--out data/webserver/pathbank_index.sqlite]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

SQL = ("SELECT pw_id, smpdb_id AS smp_id, cached_name AS name, cached_species_id AS species_id, "
       "cached_subject AS subject FROM super_pathway_visualizations "
       "WHERE named_for_type='Pathway' AND private_pathway=0 AND export=1 AND pw_id IS NOT NULL "
       "ORDER BY pw_id")


def rows(cursor: Any, batch: int = 20_000) -> Iterator[Dict[str, Any]]:
    cursor.execute(SQL)
    while True:
        got = cursor.fetchmany(batch)
        if not got:
            return
        yield from got


def main(argv: Optional[List[str]] = None) -> int:
    from t2pw.web.pathbank import DEFAULT_INDEX, build_index

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--env-file", type=Path, default=None, help="path to the gitignored .env")
    args = ap.parse_args(argv)
    if args.env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=args.env_file, override=False)
    from t2pw.bench.eval_v2.pathbank_gold import connect  # repo's own helper

    try:
        conn = connect()
    except Exception as exc:  # noqa: BLE001 - pymysql/config texts carry host and user: never echo them
        print(f"export_pathbank_index: BLOCKED: {type(exc).__name__}", file=sys.stderr)
        return 2
    try:
        import pymysql.cursors

        cur = conn.cursor(pymysql.cursors.SSDictCursor)  # streamed: 520k rows
        n = build_index(rows(cur), args.out, meta={
            "source": "PathBank MySQL super_pathway_visualizations (public, export=1)",
            "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
    finally:
        conn.close()
    print(f"export_pathbank_index: wrote {n} pathways to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

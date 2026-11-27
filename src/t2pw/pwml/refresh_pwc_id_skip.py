"""Maintainer command (VIS-9): rebuild ``data/pwc_id_skip.json``, the PathBank compound ids the PWML writer must not
give a derived ``pwc-id`` (``PW_C%06d`` of the row id).

Two groups:

- ``derivation_mismatch``: every ``compounds`` row whose ``pwc_id`` is not ``PW_C%06d(id)`` (read-only SELECT on the
  resolution DB). For those the derived pwc-id names another compound or none.
- ``structure_errors``: rows whose PathBank / MolDB structure is wrong for their name, found by eye and by the VIS-9
  structure audit (InChIKey against ChEBI). Curated below, each with its reason.

It fails closed: it refuses to write when the query returns no rows, when the resolver recorded an error, or when the
mismatch list shrinks below ``MIN_KEEP`` of the current file (``--allow-shrink`` overrides the last one only).

Nothing here runs during generation or in tests. Connection settings come from the environment / ``.env`` exactly as
for ``refresh_name_index``; no host or credential is hardcoded or printed.

Usage::

    python -m t2pw.pwml.refresh_pwc_id_skip                       # live DB, writes data/pwc_id_skip.json
    python -m t2pw.pwml.refresh_pwc_id_skip --snapshot rows.json  # rows from a saved SELECT ({"mismatched": [...]})
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from t2pw.paths import PROJECT_ROOT

DEFAULT_SKIP_PATH = PROJECT_ROOT / "data" / "pwc_id_skip.json"
MISMATCH_SQL = "SELECT id, name, pwc_id FROM compounds WHERE pwc_id <> CONCAT('PW_C', LPAD(id, 6, '0'))"
MIN_KEEP = 0.9  # PathBank rows are not deleted; a list that shrinks more than this is a failed or partial query

STRUCTURE_ERRORS: Dict[int, str] = {
    40747: "1,4-dihydroxy-2-naphthoate: structure is its CoA thioester (C32H38N7O19P3S, InChIKey of 40746)",
    40974: "4'-phosphopantetheine: methyl where the 2-OH is (C12H23N2O6PS, not C11H23N2O7PS)",
    2580: "Geraniol: drawn as the Z isomer (nerol)",
    442: "Cinnamic acid: drawn cis",
    41655: "Ferric enterobactin: lactone C=O drawn as CH(OH), amides drawn as imidic N=C(OH), Fe unbonded",
}


def _error_tag(error: Any) -> str:
    """Only the tag of a resolver error (``db_connect_failed`` ...): the rest can carry the DB host."""
    return str(error or "").split(":", 1)[0] or "db_error"


def _mismatch_rows(db: Any = None, snapshot: Optional[Path] = None) -> List[Dict[str, Any]]:
    if snapshot is not None:
        return list(json.loads(Path(snapshot).read_text(encoding="utf-8"))["mismatched"])
    if db is None:
        from t2pw.mapping.map_ids import PathBankDbResolver

        db = PathBankDbResolver.from_env()
        if db is None or not db.available():
            raise RuntimeError("resolution DB not available: " + _error_tag(getattr(db, "last_error", "") or "db_not_configured"))
    rows = [dict(row) for row in db._query(MISMATCH_SQL, tuple())]
    if getattr(db, "last_error", ""):  # _query returns [] (or a partial answer) on error and records it here
        raise RuntimeError("mismatch query failed: " + _error_tag(db.last_error))
    return rows


def build_skip(rows: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
    mismatch = sorted({int(r["id"]) for r in rows})
    return {
        "source": source,
        "built": datetime.date.today().isoformat(),
        "query": MISMATCH_SQL,
        "derivation_mismatch": mismatch,
        "structure_errors": {str(k): v for k, v in sorted(STRUCTURE_ERRORS.items())},
    }


def refresh(out: Path, db: Any = None, snapshot: Optional[Path] = None, allow_shrink: bool = False) -> Dict[str, Any]:
    rows = _mismatch_rows(db=db, snapshot=snapshot)
    if not rows:
        raise RuntimeError("no mismatch rows: refusing to write an empty skip list")
    data = build_skip(rows, "snapshot " + Path(snapshot).name if snapshot else "live resolution DB")
    try:
        current = len(json.loads(Path(out).read_text(encoding="utf-8"))["derivation_mismatch"])
    except Exception:  # noqa: BLE001 - no usable current file: nothing to compare against
        current = 0
    if not allow_shrink and len(data["derivation_mismatch"]) < MIN_KEEP * current:
        raise RuntimeError(f"mismatch list shrank from {current} to {len(data['derivation_mismatch'])}: "
                           "refusing to write (--allow-shrink to override)")
    Path(out).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_SKIP_PATH))
    ap.add_argument("--snapshot", default=None, help="JSON with the saved SELECT rows under 'mismatched'")
    ap.add_argument("--allow-shrink", action="store_true")
    args = ap.parse_args()
    data = refresh(Path(args.out), snapshot=Path(args.snapshot) if args.snapshot else None, allow_shrink=args.allow_shrink)
    print(f"wrote {args.out}: {len(data['derivation_mismatch'])} derivation mismatches, "
          f"{len(data['structure_errors'])} structure errors")


if __name__ == "__main__":
    main()

"""C-093: name the rows behind the C-030 identity-fallback census.

The census in ``tests/test_c030_canonical_identity_fallback.py`` counts rows by
bucket and key. Attributing a NEW bucket to a card needs the rows themselves --
which entity, which identifier, and which container the exporter reads it from.
This probe prints exactly that for the legs given on the command line (default:
every leg carrying a gap), reusing that module's ``_rows``/``_blind``/``_slot``
ladder so the two can never disagree. Read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import test_c030_canonical_identity_fallback as c030  # noqa: E402
from t2pw.pwml import ir  # noqa: E402


def _detail(relative: str) -> List[Dict[str, Any]]:
    payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    rows = list(c030._rows(payload))
    out: List[Dict[str, Any]] = []
    for _leg, bucket, index, key in [g for g in c030._gap_rows() if g[0] == relative]:
        found_bucket, key_lists, row = rows[index]
        keys = next(k for k in key_lists if key in k)
        container, slot = c030._slot(row, keys)
        meta = row.get("mapping_meta") or {}
        out.append({
            "bucket": found_bucket,
            "row_index_in_ladder": index,
            "name": row.get("name"),
            "key": slot,
            "exported_value": ir._first_nonempty(row, list(keys)),
            "container": "mapping_meta" if container is meta else "candidates[0]",
            "visible_to_the_old_projection": ir._first_nonempty(
                c030._blind(row), list(keys)),
            "record_level_value": row.get(slot),
            "mapped_ids": row.get("mapped_ids"),
            "mapping_meta_keys": sorted(meta) if isinstance(meta, dict) else None,
            "candidate_count": len(meta.get("candidates") or []),
            "match_rule": meta.get("chosen_rule") or meta.get("rule"),
            "confidence": meta.get("confidence"),
            "resolution": meta.get("resolution"),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legs", nargs="*")
    parser.add_argument("--out")
    args = parser.parse_args()
    legs = args.legs or sorted({g[0] for g in c030._gap_rows()})
    result = {leg: _detail(leg) for leg in legs}
    blob = json.dumps(result, indent=1, sort_keys=False, default=repr)
    if args.out:
        Path(args.out).write_text(blob, encoding="utf-8")
        print(f"wrote {args.out} ({len(blob)} bytes)")
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

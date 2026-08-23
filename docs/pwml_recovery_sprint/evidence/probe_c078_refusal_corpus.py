"""C-078: re-derive the F-099 corpus and dump the seam rows, offline.

Reads only committed ``final_mapped.json`` artifacts. No DB, no network, no LLM.

Sections
--------
``corpus``   the four counts C-078 §2d / the F-099 AMENDMENT assert, re-derived
             over the T-104 + T-105 verification runs, plus the "no refusal
             record" complement §6.6 needs a sample size for.
``seam``     one full compound row per distinct (leg, name) on the exact F-099
             seam, so the fixture in the test file is copied from the corpus
             rather than invented.
``protein``  §5's report-only question: which refused rows are proteins, and
             does ``pwml/`` have any admission gate that could see them.

Usage::

    <python> probe_c078_refusal_corpus.py --out <json path> [--root <repo>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

#: The two verification runs C-078 §2d names. T-105 and T-104.
RUN_DIRS = ("runs_verify/2026-08-22_2147", "runs_verify/2026-08-21_2239")

#: Namespaces a refusal may name that mean "a PathBank scalar was withheld",
#: i.e. the precondition for the legacy branch at ``compound_resolution.py:502``
#: being skipped.
PATHBANK_KEYS = ("pathbank_compound_id", "pw_compound_id", "pathwhiz_id")


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _leg(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return rel


def _iter_rows(payload: Dict[str, Any]):
    entities = _safe_dict(payload.get("entities"))
    for bucket, rows in entities.items():
        for index, row in enumerate(_safe_list(rows)):
            if isinstance(row, dict):
                yield bucket, index, row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    artifacts: List[Path] = []
    for run_dir in RUN_DIRS:
        artifacts.extend(sorted((root / run_dir).rglob("final_mapped.json")))

    counts = {
        "artifacts_scanned": len(artifacts),
        "rows_with_mapping_meta": 0,
        "rows_with_refusal": 0,
        "rows_with_refusal_in_compounds": 0,
        "rows_with_refusal_naming_pathbank": 0,
        "rows_with_mapping_meta_and_no_refusal": 0,
    }
    seam: List[Dict[str, Any]] = []
    refused_rows: List[Dict[str, Any]] = []

    for path in artifacts:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {path}: {exc}")
            continue
        if not isinstance(payload, dict):
            continue
        for bucket, index, row in _iter_rows(payload):
            meta = row.get("mapping_meta")
            if not isinstance(meta, dict):
                continue
            counts["rows_with_mapping_meta"] += 1
            rejected = _safe_dict(meta.get("rejected_mapped_ids"))
            if not rejected:
                counts["rows_with_mapping_meta_and_no_refusal"] += 1
                continue
            counts["rows_with_refusal"] += 1
            record = {
                "leg": _leg(path, root),
                "bucket": bucket,
                "index": index,
                "name": str(row.get("name") or ""),
                "rejected": sorted(rejected),
                "rejected_values": {k: str(v) for k, v in sorted(rejected.items())},
                "mapped_ids": _safe_dict(row.get("mapped_ids")),
                "pathbank_scalar_still_on_row": {
                    key: row.get(key) for key in PATHBANK_KEYS if row.get(key) not in (None, "")
                },
            }
            refused_rows.append(record)
            if bucket != "compounds":
                continue
            counts["rows_with_refusal_in_compounds"] += 1
            if not any(key in rejected for key in PATHBANK_KEYS):
                continue
            counts["rows_with_refusal_naming_pathbank"] += 1
            seam.append({**record, "row": row})

    # ---- protein question (§5) --------------------------------------------
    protein_buckets = {"proteins", "protein_complexes", "nucleic_acids"}
    protein_refusals = [r for r in refused_rows if r["bucket"] in protein_buckets]

    result = {
        "probe": "c078_refusal_corpus",
        "root": str(root),
        "run_dirs": list(RUN_DIRS),
        "counts": counts,
        "refused_rows": refused_rows,
        "seam_rows": seam,
        "protein_refusals": protein_refusals,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== C-078 corpus ===")
    for key, value in counts.items():
        print(f"{key:44s}: {value}")
    print()
    print("=== refused rows (all buckets) ===")
    for rec in refused_rows:
        print(
            f"{rec['bucket']:18s} {rec['name'][:38]:38s} "
            f"rejected={','.join(rec['rejected'])} "
            f"pb_scalar_on_row={bool(rec['pathbank_scalar_still_on_row'])} "
            f"[{rec['leg']}]"
        )
    print()
    print(f"=== seam rows (compounds whose refusal names a pathbank id): {len(seam)} ===")
    for rec in seam:
        print(json.dumps({k: v for k, v in rec.items() if k != "row"}, ensure_ascii=False))
    if seam:
        print()
        print("=== one full seam row, verbatim ===")
        print(json.dumps(seam[0]["row"], indent=2, ensure_ascii=False)[:4000])
    print()
    print(f"=== protein-bucket refusals: {len(protein_refusals)} ===")
    for rec in protein_refusals:
        print(json.dumps({k: v for k, v in rec.items() if k != "row"}, ensure_ascii=False))
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

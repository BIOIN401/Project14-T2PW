"""C-093 read-only probe: the committed-corpus census and its attribution.

Measures, over ``git ls-files "*final_mapped.json"``:

* the C-030 identity-fallback census (the same ``_rows``/``_blind``/``_slot``
  ladder ``tests/test_c030_canonical_identity_fallback.py`` uses, imported from
  that module so the two can never drift), per file and in aggregate;
* for every leg, the accompanying ``quarantine_report.json`` verdict, the
  ``degree_zero_exports`` trigger, the accession corroborants C-068 used
  (``enrichment`` / ``ec_number`` occurrence counts, ``prefreeze_db_resolution``);
* the commit that ADDED each leg, so a new census bucket can be attributed to a
  card rather than asserted.

Writes JSON to stdout or ``--out``. Reads only; imports ``ir`` as an oracle and
mutates nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import test_c030_canonical_identity_fallback as c030  # noqa: E402


def _corpus() -> List[str]:
    listed = subprocess.run(["git", "ls-files", "*final_mapped.json"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return sorted(listed.stdout.split())


def _added_by(relative: str) -> Dict[str, str]:
    out = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H%x1f%ad%x1f%s",
         "--date=short", "--", relative],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if not out:
        return {"sha": "", "date": "", "subject": ""}
    sha, date, subject = out.splitlines()[-1].split("\x1f")
    return {"sha": sha[:7], "date": date, "subject": subject}


def _quarantine(relative: str) -> Dict[str, Any]:
    report = (ROOT / relative).parent / "quarantine_report.json"
    if not report.is_file():
        return {"present": False}
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - corrupt artifact
        return {"present": True, "unreadable": repr(exc)}
    strict = data.get("strict_invariants") or {}
    return {
        "present": True,
        "ok": data.get("ok"),
        "refusal_reasons": data.get("refusal_reasons"),
        "degree_zero_exports": strict.get("degree_zero_exports"),
    }


def _leg(relative: str) -> Dict[str, Any]:
    text = (ROOT / relative).read_text(encoding="utf-8")
    payload = json.loads(text)
    entities = payload.get("entities") or {}
    return {
        "leg": relative,
        "added": _added_by(relative),
        "quarantine": _quarantine(relative),
        "enrichment_occurrences": text.count('"enrichment"'),
        "ec_number_occurrences": text.count('"ec_number"'),
        "prefreeze_db_resolution": payload.get("prefreeze_db_resolution"),
        "bucket_sizes": {k: len(v) for k, v in entities.items()
                         if isinstance(v, list) and v},
        "bytes": len(text),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--baseline", nargs="*", default=[],
                        help="legs the pinned census was measured over")
    args = parser.parse_args()

    corpus = _corpus()
    gaps = c030._gap_rows()

    per_file: Dict[str, List[Dict[str, Any]]] = {}
    buckets: Dict[str, int] = {}
    keys: Dict[str, int] = {}
    bucket_key: Dict[str, int] = {}
    for relative, bucket, index, key in gaps:
        per_file.setdefault(relative, []).append(
            {"bucket": bucket, "index": index, "key": key})
        buckets[bucket] = buckets.get(bucket, 0) + 1
        keys[key] = keys.get(key, 0) + 1
        bucket_key[f"{bucket}/{key}"] = bucket_key.get(f"{bucket}/{key}", 0) + 1

    baseline = set(args.baseline)
    result = {
        "corpus_size": len(corpus),
        "gap_rows": len(gaps),
        "files_carrying_a_gap": len(per_file),
        "buckets": dict(sorted(buckets.items())),
        "keys": dict(sorted(keys.items())),
        "bucket_key": dict(sorted(bucket_key.items())),
        "per_file_gaps": {k: v for k, v in sorted(per_file.items())},
        "legs": [_leg(relative) for relative in corpus],
    }
    if baseline:
        new = [c for c in corpus if c not in baseline]
        result["new_legs"] = new
        result["new_leg_count"] = len(new)
        nb: Dict[str, int] = {}
        nk: Dict[str, int] = {}
        for relative, bucket, _index, key in gaps:
            if relative in new:
                nb[bucket] = nb.get(bucket, 0) + 1
                nk[key] = nk.get(key, 0) + 1
        result["buckets_from_new_legs"] = dict(sorted(nb.items()))
        result["keys_from_new_legs"] = dict(sorted(nk.items()))
        ob: Dict[str, int] = {}
        ok_: Dict[str, int] = {}
        for relative, bucket, _index, key in gaps:
            if relative in baseline:
                ob[bucket] = ob.get(bucket, 0) + 1
                ok_[key] = ok_.get(key, 0) + 1
        result["buckets_from_baseline_legs"] = dict(sorted(ob.items()))
        result["keys_from_baseline_legs"] = dict(sorted(ok_.items()))

    blob = json.dumps(result, indent=1, sort_keys=False)
    if args.out:
        Path(args.out).write_text(blob, encoding="utf-8")
        print(f"wrote {args.out} ({len(blob)} bytes)")
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

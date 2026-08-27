"""C-092 probe: the C-030 identity-gap census over the CURRENT committed corpus.

``tests/test_c030_canonical_identity_fallback.py::test_the_census_reproduces_over_
the_committed_corpus`` pins five literals measured when the corpus held 35 legs.
This probe reports the same five over today's corpus, plus the per-file gap
counts and, for every file that carries a gap row, whether that leg's
``quarantine_report.json`` recorded ``ok`` -- which is the justification the
pinned docstring already gives for why the census grew.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tests"))

import test_c030_canonical_identity_fallback as M  # noqa: E402


def main() -> int:
    gaps = M._gap_rows()
    corpus = M._corpus()
    per_file: dict[str, int] = {}
    buckets: dict[str, int] = {}
    keys: dict[str, int] = {}
    for relative, bucket, _index, key in gaps:
        per_file[relative] = per_file.get(relative, 0) + 1
        buckets[bucket] = buckets.get(bucket, 0) + 1
        keys[key] = keys.get(key, 0) + 1

    leg_ok: dict[str, object] = {}
    for relative in sorted(per_file):
        report = (ROOT / relative).parent / "quarantine_report.json"
        if report.is_file():
            leg_ok[relative] = json.loads(report.read_text(encoding="utf-8")).get("ok")
        else:
            leg_ok[relative] = "no-report"

    json.dump(
        {
            "corpus_len": len(corpus),
            "gap_rows": len(gaps),
            "distinct_files": len(per_file),
            "buckets": buckets,
            "keys": keys,
            "per_file": per_file,
            "leg_ok_for_gap_files": leg_ok,
            "ok_true_files_with_gaps": sorted(f for f, ok in leg_ok.items() if ok is True),
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

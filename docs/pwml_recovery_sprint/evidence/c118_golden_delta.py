"""C-118 measurement: exactly which C-061 golden entries this template moves.

Orchestration tooling, not pipeline code, and it decides nothing. It re-runs the
committed ``c061_relation_golden`` corpus (2000 real candidate spans, nine
papers, both legs) against whatever tree ``PYTHONPATH`` names and reports the
delta against the committed base golden in the terms the two pinned assertions
in ``tests/test_rag_multi_relation_spans.py`` use:

* how many entries changed digest, and whether every one of them is a candidate
  the production run REJECTED (never an admission that broke);
* whether every changed entry is now ``ok``;
* how many entries are unchanged;
* the DISTINCT (span, inputs, outputs, enzymes) claims behind the change.

Usage::

    PYTHONPATH=<tree>/src python c118_golden_delta.py [out.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main(argv) -> int:
    import c061_relation_golden as golden

    base = json.loads(
        (HERE / "c061_relation_golden_base.json").read_text(encoding="utf-8")
    )
    tip = golden.build()
    base_digests = dict(base["digests"])

    changed = [
        e["key"] for e in tip["entries"] if base_digests[e["key"]] != e["digest"]
    ]
    changed_set = set(changed)
    changed_entries = [e for e in tip["entries"] if e["key"] in changed_set]

    distinct = {}
    for row in golden._rows():
        key = f"{row['paper']}/{row['leg']}/{row['bucket']}[{row['index']}]"
        if key not in changed_set:
            continue
        distinct[
            (
                row["span"],
                tuple(row["inputs"]),
                tuple(row["outputs"]),
                tuple(row["enzymes"]),
            )
        ] = row

    out = {
        "n": tip["n"],
        "base_n": base["n"],
        "changed": len(changed),
        "unchanged": tip["n"] - len(changed),
        "all_changed_are_rejected": all("/rejected[" in k for k in changed),
        "changed_not_rejected": [k for k in changed if "/rejected[" not in k][:20],
        "all_changed_now_ok": all(e["ok"] for e in changed_entries),
        "changed_not_ok": [e["key"] for e in changed_entries if not e["ok"]][:20],
        "distinct": len(distinct),
        "distinct_rows": [
            {
                "span": k[0],
                "inputs": list(k[1]),
                "outputs": list(k[2]),
                "enzymes": list(k[3]),
                "paper": v["paper"],
                "leg": v["leg"],
            }
            for k, v in distinct.items()
        ],
    }
    text = json.dumps(out, indent=2, ensure_ascii=True)
    print(text)
    if len(argv) > 1:
        Path(argv[1]).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

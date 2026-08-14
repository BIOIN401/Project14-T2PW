"""C-061a clause A3: the 115 flips, measured — not argued.

Orchestration tooling, not pipeline code.

"A post-filter can only remove readings, therefore it cannot regress the flips"
is true and is NOT a proof: removing a reading can also remove a LEGITIMATE one,
and the correct MenA reading is itself a SUPERSET of the drop-readings C-061a
deletes, so a filter written slightly wrong deletes exactly what the card exists
to recover. The safety case is therefore empirical — same corpus, both ends.

Inputs are two FULL goldens from ``c061_relation_golden.py <digests> <full>``, one
built against a base ``src`` export and one against the tip. Exit 0 only when
every A3 threshold holds::

    PYTHONPATH=<tree>/src python c061a_golden_compare.py <base> <tip> [out.json]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c061_relation_golden import _rows  # noqa: E402

#: Clause A3. The flips C-061 measured, by claim, at the counts the bio-audit
#: reproduced. Not a threshold to beat — an equality to hold.
EXPECTED_FLIPS = 115
EXPECTED_SIGNATURES = {
    "['DHNA', 'prenyl diphosphate'] -> ['demethylmenaquinone (DMK)'] [MenA]": 46,
    "['HepPP'] -> ['DMK-7'] [MenA]": 27,
    "['DMK-7'] -> ['MK-7'] [UbiE]": 27,
    "['DMK'] -> ['MK'] [MenG]": 15,
}


def _signature(row) -> str:
    return (
        f"{list(row['inputs'])} -> {list(row['outputs'])} "
        f"[{', '.join(row['enzymes'])}]"
    )


def main(argv) -> int:
    base = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    tip = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    print("base t2pw:", base["t2pw"])
    print("tip  t2pw:", tip["t2pw"])

    corpus = {}
    for order, row in enumerate(_rows()):
        corpus[order] = row
    if not (len(base["entries"]) == len(tip["entries"]) == len(corpus)):
        print(
            f"FAIL corpus size moved: base={len(base['entries'])} "
            f"tip={len(tip['entries'])} rows={len(corpus)}"
        )
        return 1

    gained, lost, churn = [], [], []
    for order, (b, t) in enumerate(zip(base["entries"], tip["entries"])):
        assert b["key"] == t["key"], f"corpus order moved at {order}: {b['key']}/{t['key']}"
        if b["ok"] != t["ok"]:
            (gained if t["ok"] else lost).append((b["key"], order))
        elif b["digest"] != t["digest"]:
            churn.append((b["key"], order, b["reasons"], t["reasons"], b["shim"], t["shim"]))

    signatures = Counter(_signature(corpus[order]) for _, order in gained)
    body = {
        "n": len(corpus),
        "base_sha256": base["sha256"],
        "tip_sha256": tip["sha256"],
        "false_to_true": len(gained),
        "true_to_false": len(lost),
        "unchanged_verdict_changed_answer": len(churn),
        "flip_signatures": dict(sorted(signatures.items())),
        "flip_keys": [key for key, _ in gained],
        "lost_keys": [key for key, _ in lost],
        "churn": churn[:20],
    }
    if len(argv) > 3:
        Path(argv[3]).write_text(
            json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    checks = [
        ("A3 false->true == 115", len(gained) == EXPECTED_FLIPS, len(gained)),
        ("A3 true->false == 0", not lost, len(lost)),
        ("A3 unchanged-verdict answer churn == 0", not churn, len(churn)),
        ("A3 flip signatures unchanged", dict(signatures) == EXPECTED_SIGNATURES, dict(signatures)),
    ]
    for label, good, observed in checks:
        print(f"{'PASS' if good else 'FAIL'} {label}: {observed}")
    for key, _, b_reasons, t_reasons, b_shim, t_shim in churn[:10]:
        print(f"  churn {key}\n    base {b_reasons} {b_shim}\n    tip  {t_reasons} {t_shim}")
    return 0 if all(good for _, good, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

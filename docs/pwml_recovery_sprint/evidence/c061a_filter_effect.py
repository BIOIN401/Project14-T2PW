"""C-061a: what the correction actually removes, over the real corpus.

Orchestration tooling, not pipeline code. Clause A6 asks for monotonicity to be
SHOWN, not argued, so this measures it span by span over the same 2000 committed
candidate spans the A3 golden uses. Two modes, so the two ends can run under
different ``PYTHONPATH`` trees::

    PYTHONPATH=<base>/src python c061a_filter_effect.py dump <base.json>
    PYTHONPATH=<tip>/src  python c061a_filter_effect.py dump <tip.json>
    python c061a_filter_effect.py compare <base.json> <tip.json> [out.json]

``compare`` exits 0 only when the tip's readings are a SUBSET of the base's on
every span (nothing invented, rewritten or reordered), no span loses its last
reading, and every removed reading is one of the two named defect classes.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c061_relation_golden import _rows  # noqa: E402

#: The hyphens and the locant comma a compound name is spelled with. A removed
#: reading is a restart artifact when it names a proper tail of a name a surviving
#: reading spells, cut at one of these.
_CUTS = "-‐‑‒–—―,"


def dump(out: Path) -> int:
    import t2pw
    from t2pw.rag.admission import parse_span_relations

    print("t2pw:", t2pw.__file__)
    spans = []
    for row in _rows():
        spans.append(
            [
                [r.inputs, r.outputs, r.catalysts, bool(r.reversible), r.pattern]
                for r in parse_span_relations(row["span"])
            ]
        )
    out.write_text(
        json.dumps({"t2pw": t2pw.__file__, "spans": spans}, ensure_ascii=False),
        encoding="utf-8",
    )
    print("spans:", len(spans), "readings:", sum(len(s) for s in spans))
    return 0


def _key(reading):
    return json.dumps(reading[:4], sort_keys=True, ensure_ascii=False)


def _fold_set(names):
    return {" ".join(str(n).split()).casefold() for n in names}


def _classify(removed, survivors) -> str:
    r_in, r_out, r_cat = _fold_set(removed[0]), _fold_set(removed[1]), _fold_set(removed[2])
    for s in survivors:
        s_in, s_out, s_cat = _fold_set(s[0]), _fold_set(s[1]), _fold_set(s[2])
        if r_cat <= s_cat and bool(removed[3]) == bool(s[3]):
            if (r_in < s_in and r_out == s_out) or (r_out < s_out and r_in == s_in):
                return "participant_subset"
    for s in survivors:
        for whole in _fold_set(s[0]) | _fold_set(s[1]) | _fold_set(s[2]):
            for frag in r_in | r_out | r_cat:
                if len(frag) < len(whole) and whole.endswith(frag) and whole[-len(frag) - 1] in _CUTS:
                    return "token_internal_restart"
    return "UNCLASSIFIED"


def compare(base_path: Path, tip_path: Path, out: Path | None) -> int:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    tip = json.loads(tip_path.read_text(encoding="utf-8"))
    print("base t2pw:", base["t2pw"])
    print("tip  t2pw:", tip["t2pw"])
    invented, emptied, classes = [], 0, Counter()
    removed_total = 0
    for index, (b, t) in enumerate(zip(base["spans"], tip["spans"])):
        b_keys = [_key(r) for r in b]
        t_keys = [_key(r) for r in t]
        for reading, key in zip(t, t_keys):
            if key not in b_keys:
                invented.append((index, reading))
        if b and not t:
            emptied += 1
        if t_keys != [k for k in b_keys if k in set(t_keys)]:
            invented.append((index, "REORDERED"))
        for reading, key in zip(b, b_keys):
            if key in t_keys:
                continue
            removed_total += 1
            classes[_classify(reading, t)] += 1
    body = {
        "spans": len(base["spans"]),
        "readings_base": sum(len(s) for s in base["spans"]),
        "readings_tip": sum(len(s) for s in tip["spans"]),
        "readings_removed": removed_total,
        "removed_by_class": dict(classes),
        "readings_invented_or_reordered": len(invented),
        "spans_emptied": emptied,
        "examples": invented[:10],
    }
    print(json.dumps(body, indent=1, ensure_ascii=False))
    if out is not None:
        out.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    checks = [
        ("A6 nothing invented, rewritten or reordered", not invented),
        ("A8 no span lost its last reading", emptied == 0),
        ("every removal is a named defect class", not classes["UNCLASSIFIED"]),
    ]
    for label, good in checks:
        print(f"{'PASS' if good else 'FAIL'} {label}")
    return 0 if all(good for _, good in checks) else 1


def main(argv) -> int:
    if argv[1] == "dump":
        return dump(Path(argv[2]))
    return compare(Path(argv[2]), Path(argv[3]), Path(argv[4]) if len(argv) > 4 else None)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Live fill sample (run by the measurement dispatch, not by tests).

  PYTHONPATH=src python -I -X utf8 fill_sample.py --runs 10 --seed 1 --out DIR [--runlist C:/t/ev-cm/runlist.json]

Picks N canonical runs (``q`` is true) with a seeded shuffle that spreads over distinct papers, builds the clean model,
fills it with live network + the on-disk cache (clean_model_cache/), and writes one FillReport JSON per run plus an
aggregate ``fill_sample.json`` / ``fill_sample.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

from t2pw.clean_model import build_from_run, to_payload
from t2pw.clean_model.fill import fill_model


def pick(runlist, n, seed):
    runs = [r for r in json.load(open(runlist, encoding="utf-8")) if r.get("q") is True]
    rng = random.Random(seed)
    rng.shuffle(runs)
    seen, first, rest = set(), [], []
    for r in runs:  # one run per paper first (variety), then the remainder
        paper = os.path.basename(os.path.dirname(r["dir"].rstrip("/\\")))
        (rest if paper in seen else first).append(r)
        seen.add(paper)
    return (first + rest)[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runlist", default="C:/t/ev-cm/runlist.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    agg = {"runs": 0, "payload_unchanged": 0, "fields_filled": 0, "ids_filled": 0, "by_source": Counter(),
           "by_match_type": Counter(), "by_field": Counter(), "ambiguous": 0, "conflicts": 0, "refused": Counter(),
           "lookups_failed": Counter(), "lookups_no_match": Counter(), "requests": 0, "cache_hits": 0, "per_run": []}
    for i, r in enumerate(pick(a.runlist, a.runs, a.seed)):
        m = build_from_run(r["dir"])
        before = to_payload(m)
        rep = fill_model(m).to_dict()
        ok = to_payload(m) == before
        name = f"{i:02d}"
        with open(os.path.join(a.out, f"fill_{name}.json"), "w", encoding="utf-8") as fh:
            json.dump({"run": r["dir"], "payload_unchanged": ok, "report": rep, "conflicts": m.db_layer["conflicts"],
                       "refused": m.db_layer["refused"], "suggestions": m.db_layer["suggestions"]}, fh, indent=1)
        agg["runs"] += 1
        agg["payload_unchanged"] += ok
        for k in ("fields_filled", "ids_filled", "conflicts", "requests", "cache_hits"):
            agg[k] += rep[k]
        agg["ambiguous"] += rep["lookups_ambiguous"]
        for k in ("by_source", "by_match_type", "by_field", "refused", "lookups_failed", "lookups_no_match"):
            agg[k].update(rep[k])
        agg["per_run"].append({"run": r["dir"], "fields": rep["fields_filled"], "ids": rep["ids_filled"],
                               "ambiguous": rep["lookups_ambiguous"], "conflicts": rep["conflicts"]})
        print(f"{name} fields={rep['fields_filled']} ids={rep['ids_filled']} requests={rep['requests']}", file=sys.stderr)
    agg = {k: dict(v) if isinstance(v, Counter) else v for k, v in agg.items()}
    with open(os.path.join(a.out, "fill_sample.json"), "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=1)
    md = [f"# Fill sample ({agg['runs']} runs, seed {a.seed})", "",
          f"payload unchanged after fill: {agg['payload_unchanged']}/{agg['runs']}", "",
          f"fields filled {agg['fields_filled']}, ids filled {agg['ids_filled']}, ambiguous {agg['ambiguous']}, "
          f"conflicts {agg['conflicts']}, requests {agg['requests']}, cache hits {agg['cache_hits']}", ""]
    for title, key in (("By source", "by_source"), ("By match type", "by_match_type"), ("By field", "by_field"),
                       ("Refused", "refused"), ("Lookups failed", "lookups_failed"), ("Lookups with no match", "lookups_no_match")):
        md += [f"## {title}", "", "| key | n |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(agg[key].items())] + [""]
    with open(os.path.join(a.out, "fill_sample.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())

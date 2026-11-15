"""Build the clean model for every canonical run and round-trip its final_mapped.json (P-23).

Run: PYTHONPATH=C:/t/cm/src PY -I build_all.py   (reads runlist.json; copies no run file into the repo)
"""

import json
import os
import sys
import time
import traceback
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5, "src")))  # -I ignores PYTHONPATH

from t2pw.clean_model import build_from_run, summary, to_payload  # noqa: E402
from t2pw.clean_model.vocab import normalise

RUNLIST = "C:/t/ev-cm/runlist.json"
OUT = os.path.join("C:/t/cm/docs/pwml_output/evidence/clean_model")


def first_diff(a, b, path=""):
    if type(a) is not type(b):
        return "%s: type %s != %s" % (path, type(a).__name__, type(b).__name__)
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return "%s/%s: key missing in %s" % (path, k, "model" if k not in b else "payload")
            d = first_diff(a[k], b[k], path + "/" + str(k))
            if d:
                return d
    elif isinstance(a, list):
        if len(a) != len(b):
            return "%s: len %d != %d" % (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, "%s[%d]" % (path, i))
            if d:
                return d
    elif a != b:
        return "%s: %r != %r" % (path, str(a)[:60], str(b)[:60])
    return None


def main():
    runs = [r for r in json.load(open(RUNLIST, encoding="utf-8")) if r.get("q") is True]
    t0 = time.time()
    res = {"runs": len(runs), "built": 0, "roundtrip_equal": 0, "failures": [], "diffs": [],
           "title": 0, "pmid": 0, "pmcid": 0, "doi": 0, "organism": 0, "year": 0}
    tot = Counter()
    kinds, signs, raw_sp, norm_sp, kind_sign = Counter(), Counter(), Counter(), Counter(), Counter()
    for r in runs:
        d = r["dir"]
        try:
            payload = json.load(open(os.path.join(d, "final_mapped.json"), encoding="utf-8"))
            before = json.dumps(payload, sort_keys=True)
            m = build_from_run(d)
            res["built"] += 1
        except Exception:
            res["failures"].append({"dir": d, "error": traceback.format_exc()[-600:]})
            continue
        assert json.dumps(payload, sort_keys=True) == before, "input payload mutated: " + d
        diff = first_diff(to_payload(m), payload)
        if diff is None and to_payload(m) == payload:
            res["roundtrip_equal"] += 1
        else:
            res["diffs"].append({"dir": d, "first_diff": diff})
        s = summary(m)
        for k, v in s.pathway.items():
            res[k] += int(v)
        for k, v in s.counts.items():
            tot[k] += v
        tot["reactions_with_quote"] += s.reactions_with_quote
        tot["transports_with_quote"] += s.transports_with_quote
        tot["interactions_with_quote"] += s.interactions_with_quote
        tot["complexes_with_unknown_member"] += s.complexes_with_unknown_member
        tot["complexes_all_unknown"] += s.complexes_all_unknown
        tot["undeclared_interaction_endpoints"] += s.undeclared_interaction_endpoints
        tot["runs_with_unknown_member_complex"] += int(s.complexes_with_unknown_member > 0)
        for i in m.interactions or []:
            raw = i.relationship.value
            raw_sp[raw] += 1
            norm_sp[normalise(raw)] += 1
            kinds[i.kind.value] += 1
            signs[i.sign.value] += 1
            kind_sign[i.kind.value + "/" + i.sign.value] += 1
    res["totals"] = dict(tot)
    res["interactions"] = {
        "raw_spellings_exact": len(raw_sp), "raw_spellings_normalised": len(norm_sp),
        "kinds": dict(kinds), "kinds_distinct": len(kinds), "signs": dict(signs), "kind_sign": dict(kind_sign),
        "unsigned": signs.get("unsigned", 0), "total": sum(signs.values()),
    }
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "build_all.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=True)
    t = res["totals"]
    md = [
        "# Clean model build over the 152 canonical runs",
        "",
        "Script: `scripts/build_all.py` (local files only, no network, no run file copied here).",
        "",
        "| measure | value |",
        "|---|---|",
        "| runs | %d |" % res["runs"],
        "| built | %d / %d (%.1f%%) |" % (res["built"], res["runs"], 100.0 * res["built"] / res["runs"]),
        "| round trip `to_payload(build(p)) == p` exact | %d / %d |" % (res["roundtrip_equal"], res["runs"]),
        "| round-trip diffs listed | %d |" % len(res["diffs"]),
        "| runs with title | %d |" % res["title"],
        "| runs with PMID | %d |" % res["pmid"],
        "| runs with PMCID | %d |" % res["pmcid"],
        "| runs with DOI | %d |" % res["doi"],
        "| runs with organism | %d |" % res["organism"],
        "| runs with year | %d |" % res["year"],
        "| reactions with evidence quote retained | %d / %d |" % (t.get("reactions_with_quote", 0), t.get("reactions", 0)),
        "| transports with evidence quote retained | %d / %d |" % (t.get("transports_with_quote", 0), t.get("transports", 0)),
        "| interactions with evidence quote retained | %d / %d |" % (t.get("interactions_with_quote", 0), t.get("interactions", 0)),
        "| interaction raw spellings (exact / normalised) | %d / %d |" % (
            res["interactions"]["raw_spellings_exact"], res["interactions"]["raw_spellings_normalised"]),
        "| controlled kinds used | %d |" % res["interactions"]["kinds_distinct"],
        "| interactions unsigned / total | %d / %d |" % (res["interactions"]["unsigned"], res["interactions"]["total"]),
        "| complexes with any Unknown member | %d / %d (in %d runs) |" % (
            t.get("complexes_with_unknown_member", 0), t.get("complexes", 0), t.get("runs_with_unknown_member_complex", 0)),
        "| complexes with only Unknown members | %d |" % t.get("complexes_all_unknown", 0),
        "| interactions with an undeclared endpoint | %d |" % t.get("undeclared_interaction_endpoints", 0),
        "| removed records (view) | %d |" % t.get("removed", 0),
        "| mapping records | %d |" % t.get("mapping_records", 0),
        "",
        "## Interactions by kind and sign",
        "",
        "| kind / sign | count |",
        "|---|---|",
    ]
    md += ["| %s | %d |" % (k, v) for k, v in sorted(kind_sign.items(), key=lambda kv: -kv[1])]
    if res["diffs"]:
        md += ["", "## Round-trip diffs", ""] + ["- `%s`: %s" % (x["dir"], x["first_diff"]) for x in res["diffs"][:40]]
    if res["failures"]:
        md += ["", "## Build failures", ""] + ["- `%s`" % x["dir"] for x in res["failures"][:40]]
    with open(os.path.join(OUT, "build_all.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(json.dumps({k: res[k] for k in ("runs", "built", "roundtrip_equal", "title", "pmid", "pmcid", "doi", "organism", "year", "seconds")}))
    print(json.dumps(res["totals"]))
    print(json.dumps(res["interactions"]["kind_sign"]))
    return 0 if res["roundtrip_equal"] == res["runs"] else 1


if __name__ == "__main__":
    sys.exit(main())

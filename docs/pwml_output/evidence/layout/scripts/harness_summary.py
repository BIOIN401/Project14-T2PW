"""v1 vs v2 PathWhiz import results (same 59 files, same importer). Categories as final_payload_p4/d3/scripts/cat2_p4.py.

usage: harness_summary.py V1_results.jsonl V2_results.jsonl OUT.md
"""
import collections
import json
import os
import sys


def base(k):
    return k.rsplit("_", 1)[0] if k.rsplit("_", 1)[-1].isdigit() else k


def cat(b):
    if b in ("subcellular_location", "cell_type", "tissue"):
        return "vocabulary"
    if b.startswith("nucleic_acid") or b == "interaction_nucleic_acid_visualization":
        return "nucleic acid"
    if b.startswith("transport"):
        return "transport family"
    return "interactions" if b.startswith("interaction") else "other"


def load(p):
    rows = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        rows[os.path.basename(r["file"].replace("\\", "/"))] = r
    return rows


def summary(rows):
    s = collections.Counter()
    for r in rows.values():
        s["files"] += 1
        s["success_true"] += bool(r["success"])
        s["clean (no messages)"] += not r["errors"]
        for k, v in r["errors"].items():
            s["messages"] += len(v)
            s["  " + cat(base(k))] += len(v)
            s["  of which left=right"] += sum("unique biological states" in m for m in v)
        for k in ("transports", "interactions", "reactions", "compounds", "compound_locations", "proteins", "protein_complexes"):
            s[f"{k} created"] += (r.get("created") or {}).get(k, 0) or 0
            s[f"{k} expected"] += (r.get("expected") or {}).get(k, 0) or 0
    return s


a, b = load(sys.argv[1]), load(sys.argv[2])
sa, sb = summary(a), summary(b)
out = ["| | v1 | v2 |", "|---|---:|---:|"] + [f"| {k} | {sa[k]} | {sb[k]} |" for k in sorted(set(sa) | set(sb), key=lambda k: (k.startswith("  "), k))]
diff = []
for name in sorted(set(a) | set(b), key=lambda n: (len(n), n)):
    ea, eb = (a.get(name) or {}).get("errors"), (b.get(name) or {}).get("errors")
    ca, cb = (a.get(name) or {}).get("created"), (b.get(name) or {}).get("created")
    if ea != eb or ca != cb:
        diff.append(f"| {name} | {sum(len(v) for v in (ea or {}).values())} | {sum(len(v) for v in (eb or {}).values())} | "
                    f"{'same' if ca == cb else json.dumps({k: [ca.get(k), cb.get(k)] for k in set(ca or {}) | set(cb or {}) if (ca or {}).get(k) != (cb or {}).get(k)})} |")
msg_diff = sum(1 for n in set(a) | set(b) if (a.get(n) or {}).get("errors") != (b.get(n) or {}).get("errors"))
extra = sorted({k for n in b for k in (b[n].get("created") or {}) if k not in ((a.get(n) or {}).get("created") or {})})
out += ["", f"Files whose importer messages differ (exact text): {msg_diff}", f"Row types created only in v2: {extra}",
        "", f"Files whose messages or created rows differ: {len(diff)}", "", "| file | v1 msgs | v2 msgs | created rows that differ [v1, v2] |", "|---|---:|---:|---|"] + diff
open(sys.argv[3], "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))

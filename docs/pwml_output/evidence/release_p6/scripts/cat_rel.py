"""P5 harness-import comparison: P4 d3 vs P5 (a) importer 03 vs P5 (b) importer 04, same category definitions as P4 (cat2_p4.py).
Usage: python cat_p5.py   (reads results.jsonl of each; writes import_table.md and import_per_file.md next to this script's parent)"""
import collections, json, re, sys
from pathlib import Path

E = Path("C:/t/rel/docs/pwml_output/evidence")
D3I = E / "final_payload_p4/d3/import"
IMP = Path("C:/t/rel-reports/import")
OUT = Path("C:/t/rel-reports/import")


def L(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def idx(r): return int(re.search(r"[\\/](\d+)\.pwml$", r["file"]).group(1))
def base(k): return k.rsplit("_", 1)[0] if k.rsplit("_", 1)[-1].isdigit() else k
def msgs(r): return [(base(k), m) for k, v in r["errors"].items() for m in v]


def cat(b):
    if b in ("subcellular_location", "cell_type", "tissue"): return "vocabulary"
    if b.startswith("nucleic_acid") or b == "interaction_nucleic_acid_visualization": return "nucleic acid"
    if b.startswith("transport"): return "transport family"
    return "interactions" if b.startswith("interaction") else "other"


def summ(rs):
    c = collections.Counter()
    for r in rs:
        for b, m in msgs(r):
            c[cat(b)] += 1
            if "unique biological states" in m: c["left=right"] += 1
    c["clean"] = sum(1 for r in rs if not r["errors"])
    c["msgs"] = sum(c[x] for x in ("vocabulary", "nucleic acid", "transport family", "interactions", "other"))
    c["tr_cr"] = sum((r.get("created") or {}).get("transports", 0) for r in rs)
    c["tr_ex"] = sum((r.get("expected") or {}).get("transports", 0) for r in rs)
    c["ia_cr"] = sum((r.get("created") or {}).get("interactions", 0) for r in rs)
    c["ia_ex"] = sum((r.get("expected") or {}).get("interactions", 0) for r in rs)
    return c


S = {"P4 d3": {idx(r): r for r in L(Path("C:/t/p5scratch/measure/import/m04-out/results.jsonl"))}}  # baseline = P5(b) m04
for n, d in (("REL", "rel-out"),):
    p = IMP / d / "results.jsonl"
    if p.exists(): S[n] = {idx(r): r for r in L(p)}
names = list(S)
sm = {n: summ(list(S[n].values())) for n in names}
lines = ["| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
for lab, k in (("files imported", None), ("clean files", "clean"), ("messages", "msgs"), ("vocabulary", "vocabulary"), ("transport family", "transport family"),
               ("of which left=right", "left=right"), ("interactions", "interactions"), ("nucleic acid", "nucleic acid"), ("other", "other")):
    lines.append("| %s | %s |" % (lab, " | ".join(str(len(S[n])) if k is None else str(sm[n][k]) for n in names)))
lines.append("| transports created / expected | %s |" % " | ".join("%d/%d" % (sm[n]["tr_cr"], sm[n]["tr_ex"]) for n in names))
lines.append("| interactions created / expected | %s |" % " | ".join("%d/%d" % (sm[n]["ia_cr"], sm[n]["ia_ex"]) for n in names))
(OUT / "import_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

# per-file: messages as multiset of (base key, text); worse = more messages or a new message type (base key) compared with P4 d3
def sig(r): return collections.Counter((b, m[:80]) for b, m in msgs(r))
rows = ["| idx | " + " | ".join("msgs " + n for n in names) + " | worse in | transports cr/ex (" + " / ".join(names) + ") |", "|---|" + "---|" * (len(names) + 2)]
worse = {n: [] for n in names[1:]}
for i in sorted(S["P4 d3"]):
    row = [str(i)]
    for n in names:
        r = S[n].get(i)
        row.append("-" if r is None else str(len(msgs(r))))
    w = []
    for n in names[1:]:
        r = S[n].get(i)
        if r is None: continue
        a, b = sig(S["P4 d3"][i]), sig(r)
        new = [k for k in b if b[k] > a.get(k, 0)]
        newtype = {k[0] for k in b} - {k[0] for k in a}
        if len(msgs(r)) > len(msgs(S["P4 d3"][i])) or newtype:
            w.append(n); worse[n].append((i, len(msgs(S["P4 d3"][i])), len(msgs(r)), sorted(newtype), new[:4]))
    row.append(", ".join(w) or "")
    row.append(" / ".join("%s/%s" % ((S[n].get(i) or {}).get("created", {}).get("transports", "-"), (S[n].get(i) or {}).get("expected", {}).get("transports", "-")) for n in names))
    rows.append("| " + " | ".join(row) + " |")
(OUT / "import_per_file.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
for n, v in worse.items():
    print("WORSE", n, len(v))
    for t in v: print("  ", t)
# files better / same
for n in names[1:]:
    better = [i for i in S["P4 d3"] if n in S and i in S[n] and len(msgs(S[n][i])) < len(msgs(S["P4 d3"][i]))]
    print("fewer messages", n, better)

import json, re, sys
sys.path.insert(0, "C:/t/ev-dedup")
import scan_runs as S
gold = {c["paper_id"]: c for c in json.load(open("src/t2pw/bench/gold/pinned_v1.json", encoding="utf-8"))["cases"]}
print("gold cases", len(gold), {k: v["min_connected_reactions"] for k, v in gold.items()})
rows = [json.loads(l) for l in open("C:/t/ev-dedup/out/runs.jsonl", encoding="utf-8")]
hit = 0
for r in rows:
    m = re.search(r"(PMC\d+)", r["path"])
    if not m or m.group(1) not in gold or r["quarantine_ok"] is not True:
        continue
    p = json.load(open(r["path"], encoding="utf-8"))
    pr = p["processes"]
    ik = S.ident_map(p["entities"])
    f = lambda v: frozenset(ik.get(S.nm(x)) or ("?", S.nm(x)) for x in S.names_of(v))
    rx = len(pr.get("reactions", [])); tr = len(pr.get("transports", []))
    dist_rx = len({(f(x.get("inputs")), f(x.get("outputs"))) for x in pr.get("reactions", [])})
    dist_tr = len({(f(x.get("cargo")), frozenset(f(x.get("transporters")))) for x in pr.get("transports", [])})
    flag = "  <-- distinct < gold floor" if dist_rx + dist_tr < gold[m.group(1)]["min_connected_reactions"] <= rx + tr else ""
    if rx + tr != dist_rx + dist_tr or flag:
        hit += 1
        print(m.group(1), r["path"][-60:], "rows rx/tr", rx, tr, "distinct", dist_rx, dist_tr, "gold floor", gold[m.group(1)]["min_connected_reactions"], flag)
print("gold-case canonical runs where distinct < rows:", hit)

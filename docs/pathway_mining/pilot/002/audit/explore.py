"""Scratch exploration of the pilot-002 run directory (read-only). Not part of the audit build."""
import json, sys
from pathlib import Path

RUN = Path("C:/t/mining-pilot-runs2/pilot-002-jasmonate-ath")


def jl(p):
    t = (RUN / p).read_text(encoding="utf-8")
    return [json.loads(x) for x in t.splitlines() if x.strip()]


what = sys.argv[1] if len(sys.argv) > 1 else "calls"
if what == "calls":
    for r in jl("model_calls.jsonl"):
        r = dict(r)
        r["evidence_ids"] = len(r["evidence_ids"])
        print(json.dumps(r, ensure_ascii=True))
    for n in (1, 2, 3):
        for r in jl(f"rounds/round_00{n}/model_raw.jsonl"):
            t = r.pop("raw_text")
            print(n, json.dumps(r, ensure_ascii=True), "RAWLEN", len(t), repr(t[:300]))
elif what == "papers":
    for r in jl("papers.jsonl"):
        if r["exclusion_reasons"] and r["processed_round"] < 0:
            continue
        r.pop("abstract", None)
        print(json.dumps(r, ensure_ascii=True))
        print()
elif what == "keys":
    for f in ("papers.jsonl", "claims.jsonl", "evidence_spans.jsonl", "queries.jsonl", "decisions.jsonl", "frontier.jsonl"):
        rows = jl(f)
        print(f, len(rows), sorted(rows[0].keys()) if rows else None)
    for f in ("candidate_subgraphs.json", "primary_graph.json", "state.json", "checkpoint.json"):
        d = json.loads((RUN / f).read_text(encoding="utf-8"))
        print(f, sorted(d.keys()) if isinstance(d, dict) else type(d))
    for n in (1, 2, 3):
        d = json.loads((RUN / f"rounds/round_00{n}/round.json").read_text(encoding="utf-8"))
        print("round", n, sorted(d.keys()))
        print(json.dumps(d["metrics"], ensure_ascii=True))
        print("errors", d["errors"])
elif what == "json":
    print(json.dumps(json.loads((RUN / sys.argv[2]).read_text(encoding="utf-8")), indent=1, ensure_ascii=True)[: int(sys.argv[3]) if len(sys.argv) > 3 else 20000])
elif what == "jl":
    for r in jl(sys.argv[2]):
        print(json.dumps(r, ensure_ascii=True)[: int(sys.argv[3]) if len(sys.argv) > 3 else 3000])
elif what == "texts":
    st = json.loads((RUN / "state.json").read_text(encoding="utf-8"))
    for pid, t in st["paper_texts"].items():
        p = st["papers"][pid]
        print("=====", pid, len(t), "|", p["title"][:120], "| abstract_len", len(p.get("abstract") or ""), "| text==abstract", t == p.get("abstract"))
        print(t[: int(sys.argv[2]) if len(sys.argv) > 2 else 800])
elif what == "cited":
    ev = {r["evidence_id"]: r for r in jl("evidence_spans.jsonl")}
    ids = []
    for c in jl("claims.jsonl"):
        for e in c["evidence_ids"]:
            if e not in ids:
                ids.append(e)
    for e in ids:
        r = ev[e]
        print("=====", e, r["paper_id"], r["section"], r["locator"], r["char_start"], r["char_end"], "score", r["retrieval_score"])
        print(r["text"])
elif what == "decisions":
    for r in jl("decisions.jsonl"):
        print(json.dumps(r, ensure_ascii=True))
    print(json.dumps(json.loads((RUN / "primary_graph.json").read_text(encoding="utf-8")), indent=1, ensure_ascii=True))
    cs = json.loads((RUN / "candidate_subgraphs.json").read_text(encoding="utf-8"))
    for comp in cs["components"]:
        rx = comp.pop("reactions", [])
        print("COMP", json.dumps(comp, ensure_ascii=True)[:1500])
        for r in rx:
            print("  RX", json.dumps(r, ensure_ascii=True)[:2500])
elif what == "queries":
    papers = {r["paper_id"]: r for r in jl("papers.jsonl")}
    fr = {}
    for n in (1, 2, 3):
        for r in jl(f"rounds/round_00{n}/input_frontier.jsonl"):
            fr.setdefault(r["frontier_item_id"], r)
    for q in jl("queries.jsonl"):
        disc = [p for p, r in papers.items() if r["discovered_via_query_id"] == q["query_id"]]
        inc = [p for p in disc if not papers[p]["exclusion_reasons"]]
        f = fr.get(q["frontier_item_id"], {})
        print(q["round_index"], q["query_id"], q["frontier_item_id"], q["frontier_kind"], "att", q["attempt"], "res", q["result_count"], "new_disc", len(disc), "new_inc", len(q["new_paper_ids"]), inc, "cache", q["cache_hit"], "err", q["error"][:80], "| target:", f.get("target", "?")[:60], "| Q:", q["query_text"][:160], "| src", q["sources"], q["source_terms"])
elif what == "frontier":
    for n in (1, 2, 3):
        print("ROUND", n, "input")
        for r in jl(f"rounds/round_00{n}/input_frontier.jsonl"):
            print(" ", json.dumps(r, ensure_ascii=True)[:400])
    print("FINAL frontier.jsonl")
    for r in jl("frontier.jsonl"):
        print(" ", json.dumps(r, ensure_ascii=True)[:400])
elif what == "graph":
    st = json.loads((RUN / "state.json").read_text(encoding="utf-8"))
    g = st["graph"]
    print(sorted(g.keys()))
    for k, v in g.items():
        if isinstance(v, list):
            print(k, len(v), json.dumps(v[:3], ensure_ascii=True)[:1500])
        elif isinstance(v, dict):
            print(k, len(v), sorted(v.keys())[:8])
    reg = g.get("registry") or {}
    print("REGISTRY keys", sorted(reg.keys()) if isinstance(reg, dict) else type(reg))
    ents = reg.get("entities") if isinstance(reg, dict) else None
    if isinstance(ents, dict):
        for eid, e in ents.items():
            print(" ", eid, json.dumps(e, ensure_ascii=True)[:300])
    elif isinstance(ents, list):
        for e in ents:
            print(" ", json.dumps(e, ensure_ascii=True)[:300])
    print("exhausted", len(st["exhausted_frontier"]))
    print("errors", st["errors"])

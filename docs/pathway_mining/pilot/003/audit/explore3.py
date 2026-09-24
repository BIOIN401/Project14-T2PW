"""Scratch reader for pilot-003 (read-only). Not part of the audit build."""
import json
import sys
from pathlib import Path

RUN = Path("C:/t/mining-pilot-runs3/pilot-003-jasmonate-ath")


def jl(p):
    return [json.loads(x) for x in (RUN / p).read_text(encoding="utf-8").splitlines() if x.strip()]


what = sys.argv[1]
if what == "calls":
    for r in jl("model_calls.jsonl"):
        r = dict(r)
        r["evidence_ids"] = len(r["evidence_ids"])
        print(json.dumps(r, ensure_ascii=False))
    for r in jl("rounds/round_001/model_raw.jsonl"):
        t = r.pop("raw_text")
        print("RAW", json.dumps(r, ensure_ascii=False), "LEN", len(t), repr(t[:150]))
elif what == "papers":
    for r in jl("papers.jsonl"):
        if r["exclusion_reasons"] and r["processed_round"] < 0:
            continue
        r.pop("abstract", None)
        print(json.dumps(r, ensure_ascii=False))
elif what == "texts":
    st = json.loads((RUN / "state.json").read_text(encoding="utf-8"))
    for pid, t in st["paper_texts"].items():
        p = st["papers"][pid]
        print("=====", pid, len(t), "|", p["title"][:120], "| text==abstract", t == p.get("abstract"))
        print(t[:500])
elif what == "claims":
    for c in jl("rounds/round_001/claims_extracted.jsonl"):
        print(c["claim_id"], c["model_call_id"], c["paper_id"], "|", c["subject"], "->", c["object"], "| enz", c["enzyme"], "| sp", c["species_text"], c["species_state"],
              "| attr", c["field_attribution"], "| part_ev", [p["evidence_ids"] for p in c["substrates"] + c["products"]], "| hyp", c["hypothesis"], "| ev", c["evidence_ids"])
elif what == "jl":
    for r in jl(sys.argv[2]):
        print(json.dumps(r, ensure_ascii=False)[: int(sys.argv[3]) if len(sys.argv) > 3 else 3000])
elif what == "json":
    print(json.dumps(json.loads((RUN / sys.argv[2]).read_text(encoding="utf-8")), indent=1, ensure_ascii=False)[: int(sys.argv[3]) if len(sys.argv) > 3 else 20000])
elif what == "frontier":
    from collections import Counter
    rows = jl("frontier.jsonl")
    print(len(rows), Counter(r["kind"] for r in rows), "exhausted", sum(r["exhausted"] for r in rows), "attempts>0", sum(r["attempts"] > 0 for r in rows))
    for r in rows:
        print(r["kind"], r["priority"], r["attempts"], r["exhausted"], "|", r["target"][:110], "|", r["reason"][:80])
    for r in jl("rounds/round_001/input_frontier.jsonl"):
        print("INPUT", r["kind"], r["target"], r["attempts"], r["exhausted"])
elif what == "rx":
    cs = json.loads((RUN / "candidate_subgraphs.json").read_text(encoding="utf-8"))
    for comp in cs["components"]:
        for r in comp["reactions"]:
            print(r["reaction_id"], comp["component_id"], comp["quarantine_reason"], "|", r["label"], "|", r["enzyme_names"], "|", r["species_state"], r["species_text"], "|", r["paper_ids"], r["claim_ids"], r["decision_status"], "| rel", comp["relevance_score"])
    for d in jl("decisions.jsonl"):
        if d["status"] not in ("candidate",):
            print("DEC", json.dumps(d, ensure_ascii=False)[:600])
    st = json.loads((RUN / "state.json").read_text(encoding="utf-8"))
    print("errors", st["errors"])
    print("exhausted", st["exhausted_frontier"])

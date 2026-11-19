"""V8 live proof: literature extension on 3 gold papers' stored runs + 1 human run (read only).

Run through the launcher that loads the primary .env (never printed):
  python C:/t/vis-runs/scratch/vis-extend-db/launch.py <this file> <out_dir> [budget_s] [max_llm_calls] [only,labels]
Writes <out_dir>/<label>.json (report, LLM calls and tokens, added and merged steps with their quotes)
and <out_dir>/handcheck_pool.json (every added / merged step, for the 10-step hand-check).
"""
import json
import sys
import time
from pathlib import Path

from t2pw.llm.client import get_token_stats, reset_token_stats
from t2pw.vision.extend.literature import STAGE, extend_literature
from t2pw.vision.payload import big_from_run, keyed_rows
from t2pw.vision.view import DEFAULT, FULL, unresolved_refs, view

ROOT = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
RUNS = {   # label -> (stored run, input paper PMCID); the first three are the V9 paper-gold set
    "PMC12421875-menaquinone": ("runs_papers/ORCH-759p/legs/menaquinone-strict/2026-10-22_2154/papers/PMC12421875/strict",
                                "PMC12421875"),
    "PMC12096016-enterobactin": ("runs_papers/ORCH-759p/legs/enterobactin-strict/2026-10-22_2101/papers/PMC12096016/strict",
                                 "PMC12096016"),
    "PMC12312563-listeria-mend": ("runs_verify/2026-08-21_2014/papers/PMC12312563/strict", "PMC12312563"),
    "PMC12376012-human": ("runs_verify/2026-09-06_1425/papers/PMC12376012/research", "PMC12376012"),
}
CACHE = "C:/t/vis-runs/cache/vis-extend-lit"


def sizes(p):
    ents = sum(len(v) for v in (p.get("entities") or {}).values() if isinstance(v, list))
    procs = {k: len(v) for k, v in (p.get("processes") or {}).items() if isinstance(v, list)}
    return {"entities": ents, **procs, "unresolved_refs": len(unresolved_refs(p))}


def main():
    out = Path(sys.argv[1])
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    calls = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    out.mkdir(parents=True, exist_ok=True)
    pool = []
    for label, (rel, pmcid) in RUNS.items():
        if only and label not in only:
            continue
        t0 = time.monotonic()
        reset_token_stats()
        big = big_from_run(ROOT + rel, paper_meta={"pmcid": pmcid})
        big2, report = extend_literature(big, budget_s=budget, max_llm_calls=calls, cache_dir=CACHE)
        tokens = get_token_stats()
        steps = []
        for s, k, r, is_base in keyed_rows(big2):
            if not s.startswith("processes."):
                continue
            lit = [x for x in big2["provenance"].get(k, []) if x.get("stage") == STAGE]
            if not lit:
                continue
            src = big2["sources"].get(lit[0]["source"]) or {}
            steps.append({"label": label, "key": k, "merged_into_base": is_base, "hop": big2["hops"].get(k),
                          "name": r.get("name"), "inputs": r.get("inputs"), "outputs": r.get("outputs"),
                          "enzymes": [e.get("entity") for e in r.get("enzymes") or [] if isinstance(e, dict)],
                          "state": r.get("biological_state"), "quote": lit[0]["quote"], "note": lit[0]["note"],
                          "source": lit[0]["source"], "title": src.get("title"), "journal": src.get("journal"),
                          "year": src.get("year"), "records": len(lit)})
        pool += steps
        summary = {"run": rel, "input_paper": pmcid, "seconds": round(time.monotonic() - t0, 1),
                   "llm": {"calls_counted": report["llm"]["calls"], "token_stats": tokens}, "report": report,
                   "base": sizes(big2["base"]), "default": sizes(view(big2, DEFAULT)),
                   "full": sizes(view(big2, FULL)), "steps": steps}
        (out / f"{label}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        print(label, summary["seconds"], "s", "status", report["status"], "llm", report["llm"],
              "tokens", tokens.get("total_tokens"), "placement", json.dumps(report["placement"]),
              "default", summary["default"], "full", summary["full"], flush=True)
    (out / "handcheck_pool.json").write_text(json.dumps(pool, indent=1, default=str), encoding="utf-8")
    print("handcheck pool", len(pool), "steps", flush=True)


if __name__ == "__main__":
    main()

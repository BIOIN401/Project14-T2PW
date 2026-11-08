"""Unique live LLM answers across all R26 segments: tokens and cost, by response id. Also the fixture's counts."""
import glob
import gzip
import json
from pathlib import Path

seen = {}
for path in sorted(glob.glob("C:/t/wsR26tmp/rec1/segments/seg-*.jsonl")):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("kind") == "llm":
            resp = row["response"]
            seen.setdefault(resp.get("id") or row["key"], resp)
tok_in = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in seen.values())
tok_out = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in seen.values())
costs = [(r.get("usage") or {}).get("cost") for r in seen.values()]
print("unique live answers", len(seen), "prompt tokens", tok_in, "completion tokens", tok_out)
print("usage.cost present on", sum(c is not None for c in costs), "total", round(sum(c or 0 for c in costs), 4))
print("models", sorted({r.get("model") for r in seen.values()}))

fx = json.loads(gzip.decompress(Path("C:/t/ws-R26/tests/fixtures/equivalence/farnesol_research.json.gz").read_bytes()).decode("utf-8"))
print("counts", {k: len(v) for k, v in fx["entries"].items()}, "recorded", {k: fx["recorded"][k] for k in ("tree_head", "tree_dirty", "hash_seed", "seconds", "segment", "served_from_prior", "models")})
print("outcome", fx["recorded"]["outcome"]["status"], fx["recorded"]["outcome"]["stage"])
print("env flag", fx["env"].get("RAG_ELIGIBILITY_STAGE0_CONFLICT_ABORTS"), fx["env"].get("RAG_ENABLED"), fx["env"].get("T2PW_EXTRACTION_DRAWS"))
p = fx["recorded"]["canonical_payload"]
print("canonical reactions", len(p["processes"]["reactions"]), [r["name"] for r in p["processes"]["reactions"]])
for row in fx["entries"]["llm"]:
    c = row["response"]["choices"][0]["message"]["content"] or ""
    if '"additions"' in c[:200]:
        try:
            a = json.loads(c)["additions"]
        except Exception:
            continue
        print("stage2 reply additions:", {k: len(v) for k, v in a.items() if isinstance(v, (list, dict)) and v}, [x.get("name") for x in (a.get("processes", {}) or {}).get("reactions", [])][:6])

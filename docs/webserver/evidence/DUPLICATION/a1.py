import json
rows = [json.loads(l) for l in open("C:/t/ev-dedup/out/runs.jsonl", encoding="utf-8")]
c = [r for r in rows if r["quarantine_ok"] is True]
print("canonical: identity-keyed exact", sum(r["a_rx_exact"] for r in c), "name-only exact", sum(r["a_rx_exact_nameonly"] for r in c))

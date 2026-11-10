import json, re, os, time
rows = [json.loads(l) for l in open("C:/t/ev-dedup/out/runs.jsonl", encoding="utf-8")]
def date(r):
    m = re.search(r"(20\d\d-\d\d-\d\d)_\d{4}", r["path"])
    return m.group(1) if m else time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(r["path"])))
out = {}
for label, sel in (("pre-2026-10-20 canonical", lambda r: date(r) < "2026-10-20" and r["quarantine_ok"] is True),
                   ("2026-10-20+ (3-draw era) canonical", lambda r: date(r) >= "2026-10-20" and r["quarantine_ok"] is True)):
    rs = [r for r in rows if sel(r)]
    rx = sum(r["n_reactions"] for r in rs); tr = sum(r["n_transports"] for r in rs)
    d = {"runs": len(rs), "reactions": rx, "transports": tr,
         "rx_near_copy(a2)": sum(r["a_rx_same_enzymes_other_state"] for r in rs),
         "tr_near_copy(a2)": sum(r["a_tr_same_cargo_transporter_other_state"] for r in rs),
         "rx_exact(a1)": sum(r["a_rx_exact"] for r in rs), "tr_exact(a1)": sum(r["a_tr_exact"] for r in rs),
         "runs_with_tr_near_copy": sum(1 for r in rs if r["a_tr_same_cargo_transporter_other_state"] > 0),
         "runs_with_transports": sum(1 for r in rs if r["n_transports"] > 0),
         "runs_with_rx_near_copy": sum(1 for r in rs if r["a_rx_same_enzymes_other_state"] > 0)}
    out[label] = d
    print(label, d)
json.dump(out, open("C:/t/ev-dedup/out/era_summary.json", "w"), indent=1)

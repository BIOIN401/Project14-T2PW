import json, re, os, time
rows = [json.loads(l) for l in open("C:/t/ev-dedup/out/runs.jsonl", encoding="utf-8")]
def date(r):
    m = re.search(r"(20\d\d-\d\d-\d\d)_\d{4}", r["path"])
    return m.group(1) if m else time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(r["path"])))
for r in rows: r["date"] = date(r)
rec = [r for r in rows if r["date"] >= "2026-10-20"]
for r in sorted(rec, key=lambda r: r["date"]):
    print(r["date"], r["quarantine_ok"], r["path"][-70:], "rx", r["n_reactions"], "tr", r["n_transports"], "a1", r["a_rx_exact"] + r["a_tr_exact"], "a2", r["a_rx_same_enzymes_other_state"] + r["a_tr_same_cargo_transporter_other_state"], "a3", r["a_rx_same_participants_other_enzymes"] + r["a_tr_same_cargo_state_other_transporter"], "b2", r["b_id_extra"], "tile", r["c_tile_redraws"], "/", r["c_tile_slots"])

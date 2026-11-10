import json
rows = [json.loads(l) for l in open("C:/t/ev-dedup/out/runs.jsonl", encoding="utf-8")]
pw = [r for r in rows if "pwml" in r and r["quarantine_ok"] is True]
L = sum(r["pwml"]["n_locations"] for r in pw); E = sum(r["pwml"]["entities_located"] for r in pw)
S = sum(r["pwml"]["c_same_state_redraws"] for r in pw); M = sum(r["pwml"]["d_entities_in_multi_states"] for r in pw)
print("canonical PWML files", len(pw), "locations", L, "entities_located", E, "same-state redraws", S, "%.1f%%" % (100 * S / L), "multi-state ents", M, "%.1f%%" % (100 * M / E))
print("gold: locations 116 entities_located 85 same-state redraws 8 (%.1f%%) multi-state ents 21 (%.1f%%)" % (100 * 8 / 116, 100 * 21 / 85))
b = [r for r in pw if r["pwml"]["b_id_extra_records"] > 0]
print("PWML with same-accession duplicate records:", len(b), "records", sum(r["pwml"]["b_id_extra_records"] for r in b), "of entities", sum(r["pwml"]["n_entities"] for r in pw))
x = [r for r in pw if r["pwml"]["a_reaction_exact_dups"] > 0]
for r in x: print("  rx exact dup PWML", r["path"][-90:])

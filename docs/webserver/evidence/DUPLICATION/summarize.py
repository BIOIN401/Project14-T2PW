import json, sys, os
from collections import Counter
rows = [json.loads(l) for l in open(sys.argv[1] + "/runs.jsonl", encoding="utf-8")]
def agg(rs, label):
    n = len(rs)
    withp = [r for r in rs if r["n_reactions"] + r["n_transports"] > 0]
    P = sum(r["n_reactions"] + r["n_transports"] for r in rs)
    E = sum(r["n_entities"] for r in rs)
    def aff(keys, pool=None):
        pool = pool if pool is not None else rs
        return sum(1 for r in pool if sum(r[k] for k in keys) > 0)
    def tot(keys):
        return sum(sum(r[k] for k in keys) for r in rs)
    out = {"label": label, "runs": n, "runs_with_processes": len(withp), "processes": P, "entities": E}
    kinds = {
      "a1_exact_process_dup(inputs/outputs/enzymes/state, identity-keyed)": ["a_rx_exact", "a_tr_exact"],
      "a2_near_copy_same_participants_and_enzyme_other_state": ["a_rx_same_enzymes_other_state", "a_tr_same_cargo_transporter_other_state"],
      "a3_same_participants_other_enzymes_or_transporter": ["a_rx_same_participants_other_enzymes", "a_tr_same_cargo_state_other_transporter"],
      "b1_same_name_two_records": ["b_name_extra"],
      "b2_same_accession_two_records(alias)": ["b_id_extra"],
      "b3_name_is_another_records_synonym(unproven)": ["b_alias_overlap_rows"],
      "c1_tile_redraw(entity drawn once per process tile)": ["c_tile_redraws"],
      "c2_same_entity_same_state_location_rows(payload)": ["c_same_state_location_rows"],
      "d_entity_located_in_multiple_states(payload)": ["d_entities_in_multi_states"],
    }
    out["kinds"] = {}
    for name, keys in kinds.items():
        den = P if name.startswith("a") else (E if name[0] in "bd" else sum(r["c_tile_slots"] for r in rs))
        if name.startswith("d"): den = sum(r["c_distinct_entities_in_processes"] for r in rs)
        out["kinds"][name] = {"runs_affected": aff(keys), "share_runs": round(aff(keys) / n, 3) if n else None,
                              "extra_rows": tot(keys), "share_of_denominator": round(tot(keys) / den, 4) if den else None, "denominator": den}
    pw = [r for r in rs if "pwml" in r]
    if pw:
        out["pwml_files"] = len(pw)
        for k in ("c_same_state_redraws", "d_entities_in_multi_states", "b_name_extra_records", "b_id_extra_records", "a_reaction_exact_dups", "a_transport_exact_dups"):
            out["pwml_" + k] = {"runs_affected": sum(1 for r in pw if r["pwml"][k] > 0), "total": sum(r["pwml"][k] for r in pw)}
        out["pwml_locations"] = sum(r["pwml"]["n_locations"] for r in pw)
        out["pwml_entities"] = sum(r["pwml"]["n_entities"] for r in pw)
    return out
res = {"all": agg(rows, "all scanned"),
       "quarantine_ok_true": agg([r for r in rows if r["quarantine_ok"] is True], "quarantine_ok=true (canonical)"),
       "quarantine_ok_false": agg([r for r in rows if r["quarantine_ok"] is False], "quarantine_ok=false (final_mapped is pre-quarantine fallback)"),
       "quarantine_unknown": agg([r for r in rows if r["quarantine_ok"] is None], "no quarantine_report.json (older runs)")}
res["by_root"] = {}
for root in sorted({r["root"] for r in rows}):
    res["by_root"][root] = {"runs": sum(1 for r in rows if r["root"] == root), "q_ok_true": sum(1 for r in rows if r["root"] == root and r["quarantine_ok"] is True)}
json.dump(res, open(sys.argv[1] + "/summary.json", "w"), indent=1)
for k in ("all", "quarantine_ok_true", "quarantine_ok_false", "quarantine_unknown"):
    a = res[k]
    print("==", a["label"], "runs", a["runs"], "with_processes", a["runs_with_processes"], "processes", a["processes"], "entities", a["entities"])
    for name, v in a["kinds"].items():
        print("  %-70s runs %3d (%5.1f%%)  extra %4d  /%d = %s" % (name, v["runs_affected"], 100 * (v["share_runs"] or 0), v["extra_rows"], v["denominator"], v["share_of_denominator"]))
    if "pwml_files" in a:
        print("  PWML files", a["pwml_files"], {k: v for k, v in a.items() if k.startswith("pwml_") and k != "pwml_files"})
print(res["by_root"])

"""V1 proof: identity of view(big0, DEFAULT/FULL) with base on the 152 stored payloads (read only),
plus paper-only stats. Usage: proof_152.py <runlist.json> <out summary json>
Run as: PYTHONPATH=<tree>/src bash C:/t/vis-runs/bw.sh vis-model model-proof-1 900 <json> 2 -- <venv python>
        proof_152.py C:/t/ev-cm/runlist.json docs/vision/evidence/V1/identity_152.json"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

from t2pw.vision.payload import big_from_run, keyed_rows, norm
from t2pw.vision.view import DEFAULT, FULL, PAPER_ONLY, unresolved_refs, view, view_stats


def main(runlist, out_path):
    runs = [r for r in json.loads(Path(runlist).read_text(encoding="utf-8")) if r.get("q")]
    t0 = time.time()
    rows, kinds_any, kinds_sig, drop_why, rec_kinds = [], Counter(), Counter(), Counter(), Counter()
    tot = Counter()
    failures = []
    for r in runs:
        d = Path(r["dir"])
        raw = (d / "final_mapped.json").read_text(encoding="utf-8")
        base = json.loads(raw)
        big = big_from_run(d)
        row = {"dir": "/".join(d.parts[-6:])}
        ok_base = big["base"] == base
        ok = {}
        for name, scope in (("default", DEFAULT), ("full", FULL)):
            v = view(big, scope)
            ok[name] = (v == base) and (json.dumps(v, ensure_ascii=False) == json.dumps(base, ensure_ascii=False))
        row["identity"] = ok_base and ok["default"] and ok["full"]
        if not row["identity"]:
            failures.append(row["dir"])
        po = view(big, PAPER_ONLY)
        base_unres = unresolved_refs(base)
        before = {(k, norm(n)) for _, k, n in base_unres}
        new_dangling = [x for x in unresolved_refs(po) if (x[1], norm(x[2])) not in before]
        st = view_stats(big, PAPER_ONLY)
        unclassified = 0
        for section, key, x, _ in keyed_rows(big):
            recs = big["provenance"][key]
            ks = sorted({rr["kind"] for rr in recs})
            if ks == ["unclassified"]:
                unclassified += 1
                tot["unclassified_" + section.split(".")[0]] += 1
            if section == "processes.reactions":
                for rr in recs:
                    rec_kinds[rr["kind"]] += 1
                for k in ks:
                    kinds_any[k] += 1
                kinds_sig["+".join(ks)] += 1
                if "paper" not in ks:
                    has_lin = bool(x.get("provenance_lineage"))
                    drop_why[("lineage" if has_lin else "no lineage") + ": " + "+".join(ks)] += 1
        rx = st.get("processes.reactions", {"total": 0, "kept": 0, "dropped": 0})
        proc = {s: v for s, v in st.items() if s.startswith("processes.")}
        row.update({
            "rx_total": rx["total"], "rx_kept_paper_only": rx["kept"],
            "proc_total": sum(v["total"] for v in proc.values()),
            "proc_kept_paper_only": sum(v["kept"] for v in proc.values()),
            "rows_total": sum(v["total"] for v in st.values()),
            "rows_kept_paper_only": sum(v["kept"] for v in st.values()),
            "unclassified_rows": unclassified,
            "base_unresolved_refs": len(base_unres),
            "paper_only_new_dangling": len(new_dangling),
        })
        if new_dangling:
            row["new_dangling_sample"] = new_dangling[:5]
        for k in ("rx_total", "rx_kept_paper_only", "proc_total", "proc_kept_paper_only", "rows_total",
                  "rows_kept_paper_only", "unclassified_rows", "base_unresolved_refs", "paper_only_new_dangling"):
            tot[k] += row[k]
        tot["identity_pass"] += int(row["identity"])
        rows.append(row)
    summary = {
        "runs": len(runs), "seconds": round(time.time() - t0, 1),
        "identity_pass": tot["identity_pass"], "identity_failures": failures,
        "totals": dict(tot),
        "reactions_with_kind": dict(kinds_any.most_common()),
        "reaction_kind_sets": dict(kinds_sig.most_common()),
        "reaction_records_by_kind": dict(rec_kinds.most_common()),
        "paper_only_dropped_reactions_why": dict(drop_why.most_common()),
        "per_run": rows,
    }
    Path(out_path).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_run"}, ensure_ascii=False, indent=1))
    print("PROOF_OK" if tot["identity_pass"] == len(runs) and tot["paper_only_new_dangling"] == 0 else "PROOF_FAIL")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

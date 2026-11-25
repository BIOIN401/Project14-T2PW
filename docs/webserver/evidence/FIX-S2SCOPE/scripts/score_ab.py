"""FIX-S2SCOPE live A/B scoring. usage: score_ab.py <tree> <out.json>
Per leg in C:/t/acc-runs/legs-s2scope: status, PWML or not, gold matched (final_mapped, accscore), rows,
precision, lactate row present in merged/final, stage2_scope_drops.json contents, cost."""
import json
import re
import sys
from pathlib import Path

tree, outp = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(Path(tree) / "src"))
sys.path.insert(0, "C:/t/acc-runs/scripts")
import accscore as S  # noqa: E402

A = Path("C:/t/acc-runs")


def lactate(p):
    return [S.label(r) for r in S.rows_of(p) if any("lactate" in str(n).casefold() for n in (r.get("outputs") or []) + (r.get("inputs") or []))]


res = {}
for legdir in sorted((A / "legs-s2scope").iterdir()):
    leg = legdir.name
    arm, paper = re.match(r"([a-z]+)\d-(PMC\d+)", leg).groups()
    rec = {"arm": arm, "paper": paper}
    runs = sorted(p for p in legdir.glob("*") if p.is_dir() and (p / "papers").is_dir())
    pdir = next(iter(sorted((runs[-1] / "papers").glob("*"))), None) if runs else None
    if pdir is None:
        res[leg] = dict(rec, outcome="nothing ran")
        continue
    sd = pdir / "strict"
    tx = pdir / "01_source_text.txt"
    text = tx.read_text(encoding="utf-8", errors="replace") if tx.exists() else ""
    rt = (sd / "RESULT.txt").read_text(encoding="utf-8", errors="replace") if (sd / "RESULT.txt").exists() else ""
    rec["status"] = (re.search(r"^status\s*:\s*(\S+)", rt, re.M) or [None, None])[1]
    rec["pwml"] = [p.name for p in sd.glob("*.pwml")]
    for fn in ("merged_payload.json", "final_mapped.json"):
        if (sd / fn).exists():
            rec["lactate_rows_" + fn.split(".")[0]] = lactate(json.load(open(sd / fn, encoding="utf-8")))
    s1 = sd / "stage1_payload.json"
    if s1.exists():
        rec["stage1_out_of_scope"] = [S.label(r) for r in S.rows_of(json.load(open(s1, encoding="utf-8")))
                                      if str(r.get("scope_membership") or "").casefold() == "out_of_scope"]
    dr = sd / "stage2_scope_drops.json"
    rec["drops_report_present"] = dr.exists()
    if dr.exists():
        d = json.load(open(dr, encoding="utf-8"))
        rec["drops"] = [S.label(x["stage2_row"]) for x in d.get("drops") or []]
    fm = sd / "final_mapped.json"
    if fm.exists() and text:
        p = json.load(open(fm, encoding="utf-8"))
        f = S.score_payload(paper, p, text, S.leg_lineage(tree, sd, p), tree)
        rec.update(matched=f["matched"], n_matched=len(f["matched"]), n_gold=f["n_gold"], rows=f["rows"],
                   tp_rows=f["tp_rows"], precision=f["precision"], fps=[x["label"] for x in f["fps"]])
    u = A / "logs-s2scope" / (leg + ".usage.jsonl")
    if u.exists():
        rec["cost"] = round(sum((json.loads(x).get("cost") or 0) for x in open(u, encoding="utf-8")), 4)
    res[leg] = rec

json.dump(res, open(outp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("| leg | status | PWML | gold | rows | precision | lactate in final | drops |")
for leg, r in res.items():
    print(f"| {leg} | {r.get('status')} | {'yes' if r.get('pwml') else 'no'} | {r.get('n_matched')}/{r.get('n_gold')} | "
          f"{r.get('rows')} | {r.get('precision')} | {bool(r.get('lactate_rows_final_mapped'))} | {r.get('drops')} |")
print("SCORE_DONE")

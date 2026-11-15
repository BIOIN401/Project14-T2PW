"""Step 1/2 comparisons. Usage: python cmp_p4.py  (run from anywhere)."""
import json, hashlib, collections
from pathlib import Path
E = Path("C:/t/fp/docs/pwml_output/evidence")
def load(p):
    return {json.loads(l)["idx"]: json.loads(l) for l in open(p, encoding="utf-8")}
def sha(d, i): return hashlib.sha256((E / d / "pwml" / f"{i}.pwml").read_bytes()).hexdigest()
w = lambda r: {i for i, x in r.items() if x["pwml_written"]}
out = []
p = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))
o3, o4 = load(E/"final_payload_p3/replay/off/results_0.jsonl"), load(E/"final_payload_p4/d2/replay/off/results_0.jsonl")
p("OFF written p3/p4", len(w(o3)), len(w(o4)), "same set", w(o3) == w(o4))
diff = [i for i in w(o3) & w(o4) if sha("final_payload_p3/replay/off", i) != sha("final_payload_p4/d2/replay/off", i)]
p("OFF identical", len(w(o3) & w(o4)) - len(diff), "different", sorted(diff))
try:
    n3, n4 = load(E/"final_payload_p3/replay/on/results_0.jsonl"), load(E/"final_payload_p4/d2/replay/on/results_0.jsonl")
except FileNotFoundError:
    n4 = None
if n4:
    p("ON written p3/p4", len(w(n3)), len(w(n4)), "same set", w(n3) == w(n4), "only p3", sorted(w(n3)-w(n4)), "only p4", sorted(w(n4)-w(n3)))
    ch = sorted(i for i in w(n3) & w(n4) if sha("final_payload_p3/replay/on", i) != sha("final_payload_p4/d2/replay/on", i))
    p("ON files changed vs P3", len(ch), "of", len(w(n3) & w(n4)), "idx", ch)
    errs = [(i, x["error"][:80]) for i, x in n4.items() if x["error"] and x["stage"] != "stage3_revalidation"]
    p("ON non-s3 errors", errs[:5], "fill_error", [i for i, x in n4.items() if x.get("fill_error")])
    tot = collections.Counter(); ldc = collections.Counter(); ont = collections.Counter(); nuc = collections.Counter(); kinds = collections.Counter(); names = set()
    for i, x in n4.items():
        for k, v in (x.get("layout_debug_counts") or {}).items():
            if k in ("locations_dropped_no_id", "nucleic_acids_type_unset") or "nucleic" in k or "ontology" in k: ldc[k] += v
        for k, v in ((x.get("ontology_report") or {}).get("counts") or {}).items(): ont[k] += v
        nr = x.get("nucleic_types_report") or {}
        for k, v in nr.items():
            if isinstance(v, (int, float)): nuc[k] += v
            elif isinstance(v, dict): nuc.update({k + "." + kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))})
        for e in x.get("fill_entries") or []:
            if e.get("field") == "ontology_id":
                kinds[e["kind"]] += 1; names.add((e["kind"], e["entity"]))
        if x["pwml_written"] is False and x.get("fill_entries"): tot["entries_in_unwritten_runs"] += len(x["fill_entries"])
    p("layout_debug_counts (written runs)", dict(ldc))
    p("ontology report counts", dict(ont)); p("ontology_id entries by kind", dict(kinds), "distinct names", len(names))
    p("nucleic_types_report sums", dict(nuc))
    p("by_field sum", dict(sum((collections.Counter((x.get("fill_summary") or {}).get("by_field") or {}) for x in n4.values()), collections.Counter())))
    p("live requests", sum(x.get("live_requests", 0) for x in n4.values()), "fetch errors", sum(x.get("fetch_errors", 0) for x in n4.values()))
    p("paper_text_found", sum(1 for x in n4.values() if x.get("paper_text_found")), "of", len(n4))
    lo = E/"final_payload_p4/d2/import"; lo.mkdir(exist_ok=True)
    open(lo/"list_on.txt", "w").write("\n".join(str(E/"final_payload_p4/d2/replay/on/pwml"/f"{i}.pwml").replace("\\", "/") for i in sorted(w(n4))) + "\n")
open(E/"final_payload_p4/d2/cmp_p4.out.txt", "w").write("\n".join(out) + "\n")

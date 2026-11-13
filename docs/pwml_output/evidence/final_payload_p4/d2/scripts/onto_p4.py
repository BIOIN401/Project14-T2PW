import json, re
from pathlib import Path
E = Path("C:/t/fp/docs/pwml_output/evidence/final_payload_p4/d2")
rows = [json.loads(l) for l in open(E/"replay/on/results_0.jsonl", encoding="utf-8")]
md = [l.split("|") for l in open(E/"a_ontology_lookup.md", encoding="utf-8") if l.startswith("| ")][2:]
names = {(c[2].strip().casefold()): (c[3].strip(), c[4].strip()) for c in md}
out = []
ent = []
for r in rows:
    for e in r.get("fill_entries") or []:
        if e.get("field") == "ontology_id": ent.append((r["idx"], r["pwml_written"], e["kind"], e["entity"], e["accession"]))
dn = {}
for i, w, k, n, a in ent: dn.setdefault((k, n.casefold()), set()).add(a)
out.append("ontology_id entries (all 152 runs): %d; in written runs: %d; distinct (kind,name): %d" % (len(ent), sum(1 for x in ent if x[1]), len(dn)))
for (k, n), a in sorted(dn.items()): out.append("  %-9s %-28s %s  in_A27=%s" % (k, n, sorted(a), n in names))
hit = {n for (k, n) in dn if n in names}
out.append("of the 27 A names, filled in replay: %d (A resolved offline-lookup: %d)" % (len(hit), sum(1 for v in names.values() if v[0])))
out.append("A-resolved names not filled in any replay run: " + str(sorted(n for n, v in names.items() if v[0] and n not in hit)))
ld = sum((r.get("layout_debug_counts") or {}).get("locations_dropped_no_id", 0) for r in rows)
out.append("locations_dropped_no_id total (written runs) %d; nucleic_acids_type_unset %d" % (ld, sum((r.get("layout_debug_counts") or {}).get("nucleic_acids_type_unset", 0) for r in rows)))
nt = [r for r in rows if r.get("nucleic_types_report") and r["nucleic_types_report"].get("candidates")]
out.append("runs with nucleic candidates %d; resolved rows %d (of %d candidates) in written runs: %d" % (len(nt), sum(r["nucleic_types_report"].get("resolved", 0) for r in nt), sum(r["nucleic_types_report"]["candidates"] for r in nt), sum(r["nucleic_types_report"].get("resolved", 0) for r in nt if r["pwml_written"])))
open(E/"onto_p4.out.txt", "w").write("\n".join(out)+"\n"); print("\n".join(out))

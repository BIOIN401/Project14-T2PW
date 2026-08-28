"""C-101 probe: the 16/5 split, the invariant, and the PMC12444477 findings.

Static re-score of committed archived artifacts. No live run, no LLM, no DB.
ASCII-only output on purpose.
"""
import glob
import io
import json
import os
import sys

TREE = sys.argv[1] if len(sys.argv) > 1 else "."
os.chdir(TREE)

from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path
from t2pw.bench.semantic import validate_semantic_coverage

gs = load_gold_set(pinned_gold_set_path())
cases = {c.paper_id: c for c in gs.cases}
print("TREE:", TREE)
print("GOLD CASES:", len(cases))
sentinel_entry = cases["PMC12444477"].unknown_backed_tolerated_sentinel
print("SENTINEL TOLERANCE ON PMC12444477:", sentinel_entry.to_dict() if sentinel_entry else None)
print("NAME-KEYED SCOPE (unchanged):",
      [t.name for t in cases["PMC12444477"].unknown_backed_tolerated_entities])
print("OTHER CASES DECLARING A SENTINEL:",
      [p for p, c in cases.items() if c.unknown_backed_tolerated_sentinel and p != "PMC12444477"])
print()

tot = {"pb": 0, "sent": 0, "wrap": 0, "other": 0, "ok": 0, "recov": 0}
broken = []
for fp in sorted(glob.glob("runs/2026-08-02_2130/papers/*/*/final_mapped.json")):
    p = fp.replace(os.sep, "/")
    parts = p.split("/")
    pid, mode = parts[-3], parts[-2]
    if pid not in cases:
        continue
    payload = json.load(io.open(p, encoding="utf-8"))
    r = validate_semantic_coverage(cases[pid], payload, mode=mode)
    c = r.identity_census
    pb = r.scientific_errors["placeholder_backed_proteins"]
    s = c["placeholder_sentinel_rows"]
    w = c["placeholder_generated_wrappers"]
    o = c["placeholder_other_rows"]
    tot["pb"] += pb
    tot["sent"] += s
    tot["wrap"] += w
    tot["other"] += o
    tot["ok"] += c["withheld_identity_correct"]
    tot["recov"] += c["withheld_identity_recoverable"]
    if pb != s + w + o:
        broken.append(p)
    if pb or c["withheld_identity_correct"] or c["withheld_identity_recoverable"]:
        print("%-12s/%-9s pb=%2d  sentinel=%d wrappers=%2d other=%d  invariant=%s"
              "   F141 correct=%2d recoverable=%d evaluated=%s"
              % (pid, mode, pb, s, w, o, "OK" if pb == s + w + o else "BROKEN",
                 c["withheld_identity_correct"], c["withheld_identity_recoverable"],
                 c["withheld_identity_evaluated"]))

print()
print("PINNED TOTALS: placeholder_backed=%d  sentinel=%d  wrappers=%d  other=%d"
      % (tot["pb"], tot["sent"], tot["wrap"], tot["other"]))
print("INVARIANT placeholder_backed == sentinel + wrappers + other :",
      tot["pb"] == tot["sent"] + tot["wrap"] + tot["other"], "| broken legs:", broken)
print("F-141 pinned: correct=%d recoverable=%d" % (tot["ok"], tot["recov"]))
print()

print("PMC12444477/strict tolerance findings, row by row:")
p = "runs/2026-08-02_2130/papers/PMC12444477/strict/final_mapped.json"
payload = json.load(io.open(p, encoding="utf-8"))
r = validate_semantic_coverage(cases["PMC12444477"], payload, mode="strict")
findings = [f for f in r.checks["placeholder_identities_distinguished"].findings]
for f in findings:
    print("    %-28s %-14s %s" % (f["pointer"], f["name"], f["kind"]))
print("    TOTAL FINDINGS:", len(findings))
names = sorted(f["name"] for f in findings)
print("    LpxH still a finding:", "LpxH" in names)
print("    Unknown still a finding:", "Unknown" in names)
print("    summary:", r.checks["placeholder_identities_distinguished"].summary)
print("PROBE OK")

"""ORCH-772 dev check: coverage stats over the replayed legs that write a PWML, and 0% counts, from a corpus JSON."""
import json
import statistics
import sys
from pathlib import Path

doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = [r for r in doc["legs"] if r["recorded"] != "aborted" and r["replay"] and not r["replay"].get("error")]
for label, ex in (("headline", False), ("excluding_by_design", True)):
    vals = []
    for r in rows:
        c = r["replay"]["pwml_coverage"]
        if r["replay"]["pwml_written"]:
            t = c["total"] - (c["by_design_withheld"] if ex else 0)
            vals.append(100 * c["drawn"] / t if t else 0.0)
    print(f"PWML-writing legs {label}: n={len(vals)} mean={statistics.fmean(vals):.1f} median={statistics.median(vals):.1f} "
          f">=75: {sum(v >= 75 for v in vals)}/{len(vals)} at 100%: {sum(v == 100 for v in vals)}")
zero = [r["id"] for r in rows if r["replay"]["pwml_coverage"]["drawn"] == 0]
print("replayed legs at 0%:", len(zero), "of", len(rows), "; no PWML:", sum(not r["replay"]["pwml_written"] for r in rows))
tot = sum(r["replay"]["pwml_coverage"]["total"] for r in rows)
drawn = sum(r["replay"]["pwml_coverage"]["drawn"] for r in rows)
print(f"pooled over replayed legs: {drawn} of {tot} processes drawn ({100 * drawn / tot:.1f}%)")

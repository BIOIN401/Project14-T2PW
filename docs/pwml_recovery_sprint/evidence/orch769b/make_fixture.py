"""ORCH-769b: build tests/fixtures/orch769b/ from the ORCH-767 chlorophyll-strict leg (read-only).

Writes the leg's final_mapped.json verbatim, gzipped (level 9, mtime 0), and the Stage-0
requested_context copied out of its quarantine_report.json. Prints the hashes the test and
README pin, and the five reactions the diagnosis names. Stdlib only; imports no t2pw.

    python make_fixture.py <worktree root>
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

LEG = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/"
           "ORCH-767/legs/chlorophyll-strict/2026-10-23_2328/papers/PMC5130987/strict")
OUT = Path(sys.argv[1]) / "tests" / "fixtures" / "orch769b"
STEM = "chlorophyll_PMC5130987_767"

raw = (LEG / "final_mapped.json").read_bytes()
qraw = (LEG / "quarantine_report.json").read_bytes()
context = json.loads(qraw)["coverage"]["requested_context"]

OUT.mkdir(parents=True, exist_ok=True)
gz = gzip.compress(raw, compresslevel=9, mtime=0)
(OUT / f"{STEM}_final_mapped.json.gz").write_bytes(gz)
ctx_text = json.dumps(context, indent=2, ensure_ascii=False) + "\n"
(OUT / f"{STEM}_requested_context.json").write_text(ctx_text, encoding="utf-8", newline="\n")

print("final_mapped.json bytes  :", len(raw))
print("final_mapped.json sha256 :", hashlib.sha256(raw).hexdigest())
print("gzip bytes               :", len(gz))
print("gzip sha256              :", hashlib.sha256(gz).hexdigest())
print("quarantine_report sha256 :", hashlib.sha256(qraw).hexdigest())
print("requested_context sha256 :", hashlib.sha256(ctx_text.encode("utf-8")).hexdigest())
assert gzip.decompress((OUT / f"{STEM}_final_mapped.json.gz").read_bytes()) == raw

payload = json.loads(raw)
for i in (17, 18, 21, 22, 23):
    r = payload["processes"]["reactions"][i]
    names = lambda slot: [a.get("entity") if isinstance(a, dict) else a for a in r.get(slot) or []]  # noqa: E731
    print(f"r{i} {r.get('name')!r} enzymes={names('enzymes')} modifiers={names('modifiers')}")
ents = payload["entities"]
for name in ("PIF1", "GBP", "FAR1", "CPO1", "PPO1"):
    where = [b for b in ("compounds", "proteins", "protein_complexes", "nucleic_acids", "element_collections")
             for row in ents.get(b) or [] if isinstance(row, dict)
             and (row.get("name") == name or name in (row.get("synonyms") or []))]
    print(f"{name}: declared in {where}")

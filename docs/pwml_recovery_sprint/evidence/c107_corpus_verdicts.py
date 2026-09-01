"""C-107: dump the accept/refuse verdict for every real actor row in the corpus.

The enumeration is REV-105's `rev105_r3_corpus_dump.py`, reproduced so the row
population and the 692 count stay comparable across SHAs. Every row is driven
through the REAL public seam `apply_patch_with_policy`, not through the private
predicate.

Usage::  <python> c107_corpus_verdicts.py <repo-root> <out.json>
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
sys.path.insert(0, str(REPO / "src"))

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402

CONT = ("enzymes", "modifiers", "modifiers_or_enzymes", "catalysts",
        "transporters", "cargo", "cargo_complex")
EVK = ("evidence", "evidence_quote", "source_evidence", "source_text")


def nm_of(r):
    if isinstance(r, str):
        return r.strip()
    if not isinstance(r, dict):
        return ""
    for f in ("entity", "protein", "protein_name", "protein_complex", "enzyme",
              "modifier", "name"):
        v = r.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def ev_of(r):
    if not isinstance(r, dict):
        return ""
    for k in EVK:
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def seam(cont, row, bucket):
    n = nm_of(row)
    proc = {"name": "p", "inputs": ["A"], "outputs": ["B"], "evidence": "chem only", cont: []}
    pl = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                       "proteins": [{"name": n}], "protein_complexes": [],
                       "nucleic_acids": []},
          "processes": {bucket: [proc]}}
    op = {"op": "add", "path": f"/processes/{bucket}/0/{cont}/-", "value": row,
          "confidence": 1.0}
    _r, rep = apply_patch_with_policy(pl, [op], stage="probe")
    return rep["summary"]["accepted_count"] == 1


out = {}
seen = set()
for f in sorted(glob.glob(str(REPO / "runs/**/final_mapped.json"), recursive=True) +
                glob.glob(str(REPO / "runs_verify/**/final_mapped.json"), recursive=True)):
    try:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
    except Exception:
        continue
    procs = d.get("processes") or {}
    for bucket in ("reactions", "transports", "reaction_coupled_transports"):
        for rxn in (procs.get(bucket) or []):
            if not isinstance(rxn, dict):
                continue
            for cont in CONT:
                rows = rxn.get(cont)
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    n, ev = nm_of(row), ev_of(row)
                    if not n or not ev:
                        continue
                    role = str(row.get("role") if isinstance(row, dict) else "")
                    key = f"{cont}|{bucket}|{n}|{role}|{ev}"
                    if key in seen:
                        continue
                    seen.add(key)
                    out[key] = seam(cont, row, bucket)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print("rows:", len(out), " accepted:", sum(out.values()),
      " refused:", len(out) - sum(out.values()))

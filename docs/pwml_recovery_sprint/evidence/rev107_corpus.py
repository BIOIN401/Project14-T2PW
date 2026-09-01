"""REV-107's own corpus enumeration. Data root and CODE root are SEPARATE
arguments, so base and tip are scored over a provably identical row population
and the only variable is the source tree.

Row population definition follows REV-105's (the same one c107_corpus_verdicts.py
reproduces), so the 692 count stays comparable. Every row is driven through the
REAL public seam apply_patch_with_policy.

Usage:  <python> rev107_corpus.py <data-root> <code-root> <out.json>
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

DATA = Path(sys.argv[1]).resolve()
CODE = Path(sys.argv[2]).resolve()
OUT = Path(sys.argv[3])
sys.path.insert(0, str(CODE / "src"))

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402
import t2pw.curation.apply_audit_patch as _m  # noqa: E402

print("code loaded from:", _m.__file__, file=sys.stderr)

CONT = ("enzymes", "modifiers", "modifiers_or_enzymes", "catalysts",
        "transporters", "cargo", "cargo_complex")
EVK = ("evidence", "evidence_quote", "source_evidence", "source_text")
NAMEK = ("entity", "protein", "protein_name", "protein_complex", "enzyme",
         "modifier", "name")


def nm_of(r):
    if isinstance(r, str):
        return r.strip()
    if not isinstance(r, dict):
        return ""
    for f in NAMEK:
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


def verdict(cont, row, bucket):
    n = nm_of(row)
    proc = {"name": "p", "inputs": ["A"], "outputs": ["B"],
            "evidence": "chem only", cont: []}
    pl = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                       "proteins": [{"name": n}], "protein_complexes": [],
                       "nucleic_acids": []},
          "processes": {bucket: [proc]}}
    op = {"op": "add", "path": f"/processes/{bucket}/0/{cont}/-", "value": row,
          "confidence": 1.0}
    _r, rep = apply_patch_with_policy(pl, [op], stage="rev107")
    return rep["summary"]["accepted_count"] == 1


files = sorted(
    glob.glob(str(DATA / "runs/**/final_mapped.json"), recursive=True)
    + glob.glob(str(DATA / "runs_verify/**/final_mapped.json"), recursive=True)
)
print("final_mapped.json files:", len(files), file=sys.stderr)

out = {}
meta = {}
seen = set()
for f in files:
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
                    out[key] = verdict(cont, row, bucket)
                    meta[key] = {"file": f[len(str(DATA)):], "len_ev": len(ev)}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"verdicts": out, "meta": meta}, ensure_ascii=False),
               encoding="utf-8")
print("ROWS:", len(out), " ACCEPTED:", sum(out.values()),
      " REFUSED:", len(out) - sum(out.values()))
print("SPANS OVER 5000 CHARS:", sum(1 for k in meta if meta[k]["len_ev"] > 5000),
      " MAX SPAN:", max((meta[k]["len_ev"] for k in meta), default=0))

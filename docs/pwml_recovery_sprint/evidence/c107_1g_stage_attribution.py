"""C-107 1g: which stage writes the oversized actor-evidence spans.

Registered, NOT fixed -- 1g is outside this card's boundary. This only locates
the producer, so the Lead can size a card for it.

For every run directory that contributes an actor-evidence span over 5,000
characters, it walks the pipeline's committed artifacts in stage order and
reports the longest ACTOR-ROW evidence string each one carries. The first
artifact that already holds the oversized span is the stage that produces it.

Usage::  <python> c107_1g_stage_attribution.py <repo-root>
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()

CONT = ("enzymes", "modifiers", "modifiers_or_enzymes", "catalysts",
        "transporters", "cargo", "cargo_complex")
EVK = ("evidence", "evidence_quote", "source_evidence", "source_text")

# Pipeline artifacts in the order the stages write them.
STAGE_ORDER = [
    ("stage1_payload.json", "Stage 1 extraction"),
    ("merged_payload.json", "merge / normalize"),
    ("final.json", "post-audit"),
    ("final.audited.json", "post-audit"),
    ("final_mapped.json", "post-mapping (the corpus artifact)"),
]


def actor_evidence_lengths(payload):
    out = []
    procs = (payload or {}).get("processes") or {}
    for bucket in ("reactions", "transports", "reaction_coupled_transports"):
        for rxn in (procs.get(bucket) or []):
            if not isinstance(rxn, dict):
                continue
            rxn_ev = rxn.get("evidence")
            if isinstance(rxn_ev, str):
                out.append(("<reaction.evidence>", len(rxn_ev)))
            for cont in CONT:
                rows = rxn.get(cont)
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    for k in EVK:
                        v = row.get(k)
                        if isinstance(v, str) and v.strip():
                            out.append((f"{cont}.{k}", len(v)))
                            break
    return out


# Locate the run directories that hold an oversized ACTOR span.
targets = set()
for f in sorted(glob.glob(str(REPO / "runs/**/final_mapped.json"), recursive=True) +
                glob.glob(str(REPO / "runs_verify/**/final_mapped.json"), recursive=True)):
    try:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
    except Exception:
        continue
    if any(n > 5000 for _k, n in actor_evidence_lengths(d)):
        targets.add(Path(f).parent)

print("=" * 78)
print("1g. STAGE ATTRIBUTION FOR THE OVERSIZED ACTOR-EVIDENCE SPANS")
print("=" * 78)
print(f"run directories contributing an oversized actor span: {len(targets)}")
for d in sorted(targets):
    print(f"\n{d.parent.parent.name} / {d.parent.name} / {d.name}")
    present = {p.name for p in d.iterdir() if p.is_file()}
    for fname, stage in STAGE_ORDER:
        if fname not in present:
            print(f"  {fname:24s} {'(absent)':>10s}   {stage}")
            continue
        try:
            obj = json.loads((d / fname).read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  {fname:24s} unreadable ({exc.__class__.__name__})")
            continue
        lens = actor_evidence_lengths(obj)
        if not lens:
            print(f"  {fname:24s} {'no actor rows':>13s}   {stage}")
            continue
        longest = max(n for _k, n in lens)
        over = sum(1 for _k, n in lens if n > 5000)
        mark = "  <<< ALREADY OVERSIZED" if over else ""
        print(f"  {fname:24s} rows={len(lens):>4d}  longest={longest:>7d}  "
              f"over5k={over:>3d}   {stage}{mark}")

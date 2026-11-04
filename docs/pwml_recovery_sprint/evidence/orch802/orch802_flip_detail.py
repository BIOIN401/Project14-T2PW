"""ORCH-802: for each predicate flip of noloss_scan.json, print the row's actor entries so the
origin of an ``inferred`` attachment (cue attacher vs Stage-2 LLM) can be read off.

A cue-attacher row is recognisable: Stage-2 injector writes confidence 0.9 with
``source_refs == [evidence]``; the Stage-5 normalizer writes confidence 0.75 with role catalyst.
Usage: python orch802_flip_detail.py <noloss_scan.json> <repo> [actor ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
scan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
repo = Path(sys.argv[2])
only = set(sys.argv[3:])
for flip in scan["A_predicate"]["flips"]:
    if only and flip["actor"] not in only:
        continue
    for occ in flip["occurrences"]:
        p = Path(occ["file"])
        p = p if p.is_absolute() else repo / p
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows = (payload.get("processes") or {}).get("reactions") or payload.get("reactions") or []
        rows = [r for r in rows if isinstance(r, dict)]
        row = rows[occ["reaction_index"]]
        print(f"== {flip['actor']} | {occ['file']} r{occ['reaction_index']} | {row.get('name')!r}")
        print(f"   evidence: {str(row.get('evidence'))[:300]!r}")
        for b in ("enzymes", "modifiers"):
            for it in row.get(b) or []:
                if isinstance(it, dict):
                    nm = it.get("entity") or it.get("protein") or it.get("protein_complex")
                    attacher = (it.get("confidence") == 0.9 and it.get("source_refs") == [it.get("evidence")]) or (
                        it.get("confidence") == 0.75 and it.get("role") == "catalyst" and it.get("provenance") == "inferred")
                    print(f"   {b}: {nm!r} prov={it.get('provenance')} conf={it.get('confidence')} "
                          f"cue_attacher_shape={attacher} ev={str(it.get('evidence'))[:120]!r}")

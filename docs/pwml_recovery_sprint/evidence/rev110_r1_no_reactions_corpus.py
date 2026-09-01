"""REV-110 round 1 -- how reachable is the surviving `no_reactions` relabelling?

Round 0's finding was blocking partly because `contract` was the DOMINANT
bucket (55 legs) and needed only ANY issue code. The residual after round 1
needs a much rarer conjunction. This measures it on the committed corpus:

  of the legs the driver labelled `no_reactions`, how many carry ZERO issue
  codes AND a message/detail matching a PROVIDER marker -- i.e. how many could
  be a provider casualty wearing the decline label?

T-107 IS EXCLUDED BY NAME. `runs_verify/2026-08-28_1816` is not opened, and
the exclusion is printed so it can be seen to be non-vacuous. Nothing is
re-scored; this reads rows and counts.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

SRC = Path(sys.argv[1]); ROOT = Path(sys.argv[2]); sys.path.insert(0, str(SRC))
from t2pw.batch.driver import _NETWORK_MARKERS, _LLM_MARKERS, _NO_REACTION_MARKERS

T107 = "2026-08-28_1816"   # NOT OPENED
manifests, skipped = [], []
for base in ("runs", "runs_verify"):
    for path in (ROOT / base).glob("*/manifest.jsonl"):
        (skipped if T107 in str(path) else manifests).append(path)

print(f"manifests read    : {len(manifests)}")
print(f"EXCLUDED BY RULE  : {[str(p.relative_to(ROOT)) for p in skipped] or '(none)'}   <- T-107, never opened")
print()

kinds, no_rx, suspicious, acq = {}, [], [], []
for m in sorted(manifests):
    for line in m.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line: continue
        try: row = json.loads(line)
        except ValueError: continue
        kind = str(row.get("failure_kind") or "")
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind != "no_reactions": continue
        codes = [c for c in (row.get("issue_codes") or []) if c]
        text = f"{row.get('message','')} {row.get('detail','')}".lower()
        files = len([f for f in (row.get("files") or ()) if f])
        counts = row.get("counts") or {}
        rx = int(counts.get("reactions", 0) or 0) + int(counts.get("transports", 0) or 0)
        tag = f"{m.parent.name}/{row.get('paper_id')}/{row.get('mode')}"
        entry = dict(tag=tag, codes=len(codes), files=files, status=row.get("status"),
                     rx=rx, net=[k for k in _NETWORK_MARKERS if k in text][:3],
                     llm=[k for k in _LLM_MARKERS if k in text][:3])
        no_rx.append(entry)
        if not codes and (entry["net"] or entry["llm"]) and row.get("status") != "error":
            suspicious.append(entry)
        if "no full text" in text or "nothing to extract" in text:
            acq.append(entry)

print("failure_kind distribution across the corpus (T-107 excluded):")
for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
    print(f"    {k or '(empty)':<26} {v:>4}")
print()
print(f"no_reactions legs total                       : {len(no_rx)}")
print(f"  ... with >=1 preserved file                 : {sum(1 for e in no_rx if e['files'])}")
print(f"  ... with ZERO issue codes                   : {sum(1 for e in no_rx if not e['codes'])}")
print()
print(f"SUSPICIOUS (no codes + provider wording + status!=error) : {len(suspicious)}")
for e in suspicious:
    print(f"    {e['tag']}  status={e['status']} files={e['files']} net={e['net']} llm={e['llm']}")
print()
print(f"ACQUISITION-shaped ('no full text' / 'nothing to extract') : {len(acq)}")
for e in acq:
    print(f"    {e['tag']}  status={e['status']} files={e['files']} codes={e['codes']}")
print()
print("A leg only reaches PASS_NEGATIVE_CONTROL if it is ALSO on a negative-control")
print("or context_only gold case with zero released rows -- so these counts are an")
print("UPPER BOUND on the exposure, not the exposure itself.")

"""ORCH-759a: cut tests/fixtures/orch759a/enterobactin_final_mapped.excerpt.json from the leg.

Usage: python make_fixture.py <leg final_mapped.json> <out path>

The excerpt is the leg's canonical payload with two things removed, and nothing else:

* every entity row's ``enrichment`` block (UniProt/ChEBI text written by Stage 7, after
  Stage 6; no gate and no part of the fold reads it), and
* the top-level ``source_text_index`` (a normalized copy of the paper). Without it the
  identity-admission pass A is ``not_evaluated``, which keeps every accession (fail-open,
  PRODUCT_CONTRACT section 8) -- the same admission outcome the leg recorded.

Offline, stdlib only, deterministic (sorted keys are NOT applied: key order is kept).
"""

import hashlib
import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
raw = src.read_bytes()
payload = json.loads(raw.decode("utf-8"))
payload.pop("source_text_index", None)
for rows in payload.get("entities", {}).values():
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            row.pop("enrichment", None)
out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("source sha256 (file bytes):", hashlib.sha256(raw).hexdigest())
print("excerpt sha256 (file bytes):", hashlib.sha256(out.read_bytes()).hexdigest())
print("excerpt bytes:", out.stat().st_size)

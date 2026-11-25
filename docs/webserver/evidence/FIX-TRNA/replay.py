"""FIX-TRNA offline replay: deterministic audit over every stored merged payload, flag off vs on.

usage: python -I replay.py <src dir> <out.json> [--flag]   (flag toggled in-process: off, then on)
Dumps, per payload: the full deterministic audit result (patch + report) with the flag off and on.
"""

import glob
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
from t2pw.curation.audit_json_llm import audit_payload  # noqa: E402

FLAG = "T2PW_AUDIT_NA_DECLARED"
os.environ.pop(FLAG, None)

paths = []
for pat in (
    "C:/t/acc-runs/legs*/**/merged_payload.json",
    "C:/t/t2pw-server-data/work/item-*/attempt-1/out/**/merged_payload.json",
):
    paths += glob.glob(pat, recursive=True)
paths = sorted({p.replace("\\", "/") for p in paths})

out = {}
for p in paths:
    payload = json.load(open(p, encoding="utf-8"))
    rec = {}
    os.environ.pop(FLAG, None)
    rec["off"] = json.dumps(audit_payload(json.loads(json.dumps(payload)), use_llm=False), sort_keys=True)
    os.environ[FLAG] = "1"
    rec["on"] = json.dumps(audit_payload(json.loads(json.dumps(payload)), use_llm=False), sort_keys=True)
    os.environ.pop(FLAG, None)
    out[p] = rec
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"))
print(len(out), "payloads")

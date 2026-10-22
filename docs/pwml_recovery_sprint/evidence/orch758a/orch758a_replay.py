"""Scratch replay: strict normalize_process_payload on an ORCH-755 sour-taste payload."""
import json
import sys
import traceback

from t2pw.pipeline.process_normalizer import normalize_process_payload, _has_plus_token
from t2pw.pipeline.stage_contracts import validate_post_normalization

for s in ["Ca++", "Ca ++", "Fe+++", "Ca2+", "Ca 2+", "NADH + H+", "A + B", "EntB + EntE",
          "voltage-gated Ca ++ channels", "voltage-gated Ca++ channels", "H+", "NAD+"]:
    print(repr(s), _has_plus_token(s))

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
try:
    out, rep = normalize_process_payload(payload, mode="pathwhiz")
except Exception:
    traceback.print_exc()
    sys.exit(3)
print("gate ok:", (rep.get("gate") or {}).get("ok"), "errors:", (rep.get("gate") or {}).get("errors"))
print("plus remaining", rep["summary"].get("n_plus_tokens_remaining"))
cr = validate_post_normalization(out, rep.get("gate") or {})
codes = {}
for k in ("errors", "issues", "findings"):
    for e in cr.get(k, []) if isinstance(cr, dict) else []:
        c = e.get("code") if isinstance(e, dict) else str(e)
        codes[c] = codes.get(c, 0) + 1
print("contract keys", list(cr.keys()) if isinstance(cr, dict) else type(cr))
print("contract codes", codes)
print("transports:")
for t in out.get("processes", {}).get("transports", []):
    print("  ", t.get("cargo"), t.get("transporters"))
print("compounds:", [c.get("name") for c in out["entities"]["compounds"]])
print("proteins:", [c.get("name") for c in out["entities"]["proteins"]])
print("complexes:", [c.get("name") for c in out["entities"]["protein_complexes"]])

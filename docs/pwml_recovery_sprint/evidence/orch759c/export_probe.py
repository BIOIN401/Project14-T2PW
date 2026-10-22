"""ORCH-759c probe: past the final gate, would the leg's quarantined payload export?

Not the app's ``run_pwml_export``. It reuses Chunk E's own approximation of the
exporter (``tests/test_strict_quarantine_real_artifact_replay.py``:
``_required_contract_codes`` -- the required-field contract with the exporter's
metadata -- and ``_ir_codes`` -- pre-freeze compound resolution, ``build_pwml_ir``,
blocking IR issues, ``validate_pwml_ir``) on the tip's quarantine result for the
chlorophyll-strict leg. Prints only.
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests"))

from test_strict_quarantine_real_artifact_replay import _as_export_payload, _ir_codes, _required_contract_codes  # noqa: E402
from t2pw.pwml.ir import validate_required_pwml_contract  # noqa: E402
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "orch759c"
payload = json.loads(gzip.decompress((FIX / "chlorophyll_PMC5130987_final_mapped.json.gz").read_bytes()))
context = json.loads((FIX / "chlorophyll_PMC5130987_requested_context.json").read_text(encoding="utf-8"))
result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
print("quarantine ok:", result.ok, "| release:", (result.quarantine_report.get("release") or {}).get("status"))
print("required-field contract error codes:", _required_contract_codes(result.payload))
print("IR error codes:", _ir_codes(result.payload))
for issue in validate_required_pwml_contract(_as_export_payload(result.payload), strict_db=True).get("errors", []):
    print("  contract error:", json.dumps({k: issue.get(k) for k in ("code", "pointer", "message")}, ensure_ascii=False))
for bucket in ("reactions",):
    for i, row in enumerate(result.payload["processes"][bucket]):
        for enzyme in row.get("enzymes") or []:
            name = enzyme.get("entity") if isinstance(enzyme, dict) else enzyme
            if name not in {c["name"] for c in result.payload["entities"]["protein_complexes"]}:
                print(f"  /processes/{bucket}/{i} {row.get('name')!r} enzyme={json.dumps(enzyme, ensure_ascii=False)}")

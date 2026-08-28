"""C-103: measure, for every stored strict-failure case, the fields the corrected
fixture will pin -- so no expectation in ``cases.json`` is written from a
document rather than from the code under test."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

TREE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(TREE / "src"))

from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402
from t2pw.pipeline.process_normalizer import (  # noqa: E402
    GateValidationError,
    run_strict_post_normalization_gates,
)
from t2pw.pwml.ir import validate_required_pwml_contract  # noqa: E402
import t2pw  # noqa: E402

print("MEASURED_TREE t2pw =", t2pw.__file__)


def strict_ok(payload):
    reasons = []
    try:
        run_strict_post_normalization_gates(deepcopy(payload), enforce_all_proteins_connected=True)
    except GateValidationError as exc:
        reasons.extend(str(r.get("reason", "")) for r in (exc.details.get("errors") or []))
    contract = validate_required_pwml_contract(deepcopy(payload), strict_db=True)
    if not contract.get("ok"):
        reasons.extend(str(i.get("code", "")) for i in contract.get("errors", []))
    return (not reasons), reasons


def process_count(payload):
    return sum(len(r) for r in (payload.get("processes") or {}).values() if isinstance(r, list))


def entity_count(payload):
    return sum(len(r) for r in (payload.get("entities") or {}).values() if isinstance(r, list))


cases = json.loads(
    (TREE / "tests/fixtures/strict_failures/cases.json").read_text(encoding="utf-8")
)["cases"]

for case in cases:
    cid = case["id"]
    expect = case["expect"]
    payload = case["payload"]
    result = quarantine_and_close(deepcopy(payload), strict_db=True)
    report = result.quarantine_report
    release = report["release"]
    shrank = (
        process_count(result.payload) < process_count(payload)
        or entity_count(result.payload) < entity_count(payload)
    )
    after_ok, after_reasons = strict_ok(result.payload)
    print("=" * 74)
    print(cid)
    print("  fixture recovers / smaller  :", expect["recovers"], "/", expect["smaller"])
    print("  MEASURED ok                 :", result.ok)
    print("  MEASURED shrank             :", shrank, " -> (ok and shrank) =", result.ok and shrank)
    print("  MEASURED strict-after-ok    :", after_ok, ("" if after_ok else after_reasons))
    print("  MEASURED release.status     :", release["status"])
    print("  MEASURED strict_acc_elig    :", release["strict_acceptance_eligible"])
    print("  MEASURED completeness       :", release["completeness"])
    print("  MEASURED review_reasons     :", json.dumps(report["review_reasons"]))
    print("  MEASURED refusal_reasons    :", json.dumps(report["refusal_reasons"]))
    print("  MEASURED coverage.reasons   :", json.dumps(result.coverage.get("reasons")))
    print("  MEASURED reactions surviving:", len((result.payload.get("processes") or {}).get("reactions") or []))

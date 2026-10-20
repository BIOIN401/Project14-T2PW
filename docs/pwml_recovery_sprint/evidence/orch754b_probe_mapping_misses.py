"""Does the interactive curator's mapping-miss context see this card's Stage-6 actions?"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "tests")
import test_pwml_interaction_protein_wrappers as w  # noqa: E402
from t2pw.curation.interactive_curator import compact_mapping_misses  # noqa: E402

payload = w._payload()
report = w._stage6(payload)
print("actions:", [a["type"] for a in report["actions"]])
misses = compact_mapping_misses({"summary": report["summary"], "actions": report["actions"]})
print("misses picked from the wrapper report:", json.dumps(misses)[:800])

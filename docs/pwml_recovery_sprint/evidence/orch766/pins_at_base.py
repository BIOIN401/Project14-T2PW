"""ORCH-766: print the byte-identity pins of tests/test_orch766_unreferenced_species.py.

Run on the BASE tree (src identical to d6849c9e), through bounded_run.py with
PYTHONPATH=<tree>/src. Each pin is ``_digest`` of the full quarantine output for one
synthetic payload; the test asserts the tip reproduces it. Each payload is run twice
to show the digest is deterministic. Prints only.
"""
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(TREE / "tests"))

import test_orch766_unreferenced_species as T  # noqa: E402
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402


def run(kind):
    payload = T._base() if kind == "base" else T._kind_payload(kind)
    context = None if kind == "base" else T._context(kind)
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
    return T._digest(result), T._names(result.payload), result.ok


print("PINS = {")
for kind in ("base", *T.ALL_KINDS):
    first, second = run(kind), run(kind)
    assert first == second, kind
    print(f'    "{kind}": "{first[0]}",  # species out {first[1]}, ok={first[2]}')
print("}")
print("PINS DONE")

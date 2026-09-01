"""REV-110 -- is the condition-2 hole REACHABLE from the shipped driver?

The adversarial probe showed three constructed rows earn PASS_NEGATIVE_CONTROL
that should not. A constructed row is only a finding if the pipeline can
actually WRITE it. This asks the driver itself, with no fixtures:

1. ``driver._classify`` -- does a leg whose text is a PROVIDER failure come back as
   ``contract`` when issue codes happen to be present? The branch order at
   ``driver.py:1262`` is ``if contract_signal or issue_codes: return
   KIND_CONTRACT``, tested BEFORE the network/llm markers below it.

2. ``KIND_UNKNOWN`` -- what status, message and codes does the shipped call
   site emit alongside it?

Nothing is scored here and no run directory is opened.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/t/rev110/src")
sys.path.insert(0, str(SRC))

from t2pw.batch import driver  # noqa: E402
from t2pw.bench import acceptance  # noqa: E402

print("=" * 78)
print("1. driver._classify -- BRANCH ORDER: issue codes outrank the provider markers")
print("=" * 78)
print(f"  signature: {inspect.signature(driver._classify)}")

NETWORK_TEXT = "connection reset by peer"
LLM_TEXT = "the model returned an empty completion"
for marker_name, text in (("_NETWORK_MARKERS", NETWORK_TEXT), ("_LLM_MARKERS", LLM_TEXT)):
    markers = getattr(driver, marker_name, ())
    hit = [m for m in markers if m in text.lower()]
    print(f"  {marker_name} matched by {text!r}: {hit}")

for label, codes in (("no issue codes", []), ("ONE issue code", ["gate.pwml_required_field"])):
    for text_label, text in (("network text", NETWORK_TEXT), ("llm text", LLM_TEXT)):
        kind = driver._classify(
            text=text, issue_codes=list(codes), contract_signal=False,
            ambiguous=False, no_reactions=False, crashed=False,
        )
        verdict = "DECLINE" if kind in acceptance._NC_DECLINE_KINDS else (
            "CASUALTY" if kind in acceptance._NC_CASUALTY_KINDS else "other")
        flag = "!! " if verdict == "DECLINE" else "ok "
        print(f"  {flag}{text_label:<13} + {label:<15} -> failure_kind={kind!r}  "
              f"C-110 reads this as: {verdict}")

print()
print("  => A PROVIDER casualty carrying ANY issue code is written to the manifest")
print("     as failure_kind='contract', which C-110 lists as a DECLARED DECLINE.")

print()
print("=" * 78)
print("2. KIND_UNKNOWN -- the shipped call site's status, message and codes")
print("=" * 78)
src = inspect.getsource(driver)
idx = src.find("kind=KIND_UNKNOWN")
if idx < 0:
    idx = src.find("kind=KIND_UNKNOWN")
start = src.rfind("_fail(", 0, idx)
print(src[start:src.find(")", idx) + 1])
print(f"  KIND_UNKNOWN literal              : {driver.KIND_UNKNOWN!r}")
print(f"  in C-110 _NC_INDETERMINATE_KINDS  : "
      f"{driver.KIND_UNKNOWN in acceptance._NC_INDETERMINATE_KINDS}")
print(f"  in C-110 _NC_CASUALTY_KINDS       : "
      f"{driver.KIND_UNKNOWN in acceptance._NC_CASUALTY_KINDS}")

print()
print("=" * 78)
print("3. THE GUARD THAT LETS BOTH THROUGH")
print("=" * 78)
fn = inspect.getsource(acceptance.negative_control_outcome)
for line in fn.splitlines():
    if "declared = " in line or "or bool(codes)" in line or "_NC_DECLINE_KINDS" in line \
            or "SCIENTIFICALLY_UNRECOVERABLE" in line or "_NC_INDETERMINATE_KINDS" in line \
            or "stated = " in line:
        print(f"    {line.strip()}")
print()
print("  `bool(codes)` alone satisfies `declared`, and the INDETERMINATE branch")
print("  is guarded by `not codes` -- so one issue code both supplies the")
print("  'stated reason' AND suppresses the indeterminate refusal.")
print("=" * 78)

"""ORCH-758: does D-104 rule 3 (merge_on_mapped_identity) apply in the batch's post-pipeline run?

Offline, no model, no network. Drives orch758_rule3_probe_app.py with AppTest exactly as
batch/driver.py drives the app: one script run for "Run pipeline", a second for the
post-pipeline button. Prints what merge_on_mapped_identity sees in each run.
Usage: PYTHONPATH=src python orch758_rule3_probe.py <scratch dir for diagnostics>
"""
import json
import os
import sys
import threading
from pathlib import Path

from streamlit.testing.v1 import AppTest

os.environ["ORCH758_DIAG"] = sys.argv[1]
app = Path(__file__).with_name("orch758_rule3_probe_app.py")
at = AppTest.from_file(str(app), default_timeout=120)
at.run()
at.button[0].click()
at.run()
at.button[1].click()
at.run()
out = {"python": sys.version.split()[0], "main_thread": threading.get_ident(),
       "same_run": dict(at.session_state["same_run"]), "next_run": dict(at.session_state["next_run"]),
       "exceptions": [str(e.value) for e in at.exception]}
print(json.dumps(out, indent=1))
ok = out["same_run"]["rule3_applied"] and not out["next_run"]["rule3_applied"]
print("VERDICT:", "rule 3 applies only inside the Stage-1 script run; the post-pipeline run sees "
      "a detached recorder" if ok else "hypothesis NOT confirmed")

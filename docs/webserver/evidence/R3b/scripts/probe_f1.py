"""R3b review F1 probe: does a patch of the diagnostics names on their SOURCE module reach
the moved code? Prints one line per name. usage: probe_f1.py (PYTHONPATH = the tree's src)."""
import tempfile
from pathlib import Path

import t2pw.paths as paths
import t2pw.pipeline.extraction_diagnostics as diag
import t2pw.service.extraction as ex

print("service module:", ex.__file__)
paths.PROJECT_ROOT = Path(tempfile.mkdtemp(prefix="r3b-probe-"))
hits = []
real_activate, real_current = diag.activate, diag.current
diag.payload_hash = lambda v: hits.append("payload_hash") or "sha256:probe"
diag.activate = lambda **kw: hits.append("activate") or real_activate(**kw)
diag.current = lambda: hits.append("current") or real_current()
ex._start_extraction_diagnostics("text", state={})
ex._sync_extraction_diagnostics(state={})
for name in ("payload_hash", "activate", "current"):
    print(f"diag.{name} patch reaches service: {name in hits}")
diag.deactivate()

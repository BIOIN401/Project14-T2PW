import sys
import t2pw.service.extraction as ex
print("streamlit loaded:", "streamlit" in sys.modules)
print("t2pw.app loaded:", any(m.startswith("t2pw.app") for m in sys.modules))
import t2pw.pipeline.extraction_diagnostics as diag
import t2pw.paths as paths, tempfile, pathlib
paths.PROJECT_ROOT = pathlib.Path(tempfile.mkdtemp(prefix="r3brev-"))
hit = []
ra, rc, rh = diag.activate, diag.current, diag.payload_hash
diag.payload_hash = lambda *a, **k: (hit.append("payload_hash"), "sha256:x")[1]
diag.activate = lambda **k: (hit.append("activate"), ra(**k))[1]
diag.current = lambda: (hit.append("current"), rc())[1]
class R:
    def spinner(self, t, **k):
        import contextlib; return contextlib.nullcontext()
    def info(self, b): pass
    def warning(self, b): pass
ex.run_stage_zero("short", temperature=0.0, user_task_context=None, state={}, reporter=R(),
                  preprocess_fn=lambda t, **k: {"pathway_name": "p"})
print("patches reached:", hit)
print("service has old aliases:", [n for n in ("diagnostic_payload_hash","activate_extraction_diagnostics","current_extraction_diagnostics") if hasattr(ex, n)])
diag.deactivate()

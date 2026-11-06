import sys, inspect
import t2pw.service.extraction as ex
print("streamlit loaded:", "streamlit" in sys.modules)
print("t2pw.app loaded:", any(m.startswith("t2pw.app") for m in sys.modules))
sig = inspect.signature(ex.run_stage_zero)
for n in ("state", "reporter", "preprocess_fn"):
    p = sig.parameters[n]
    print(n, p.kind.name, "default=", p.default)
for f in (ex._start_extraction_diagnostics, ex._sync_extraction_diagnostics):
    p = inspect.signature(f).parameters["state"]
    print(f.__name__, "state", p.kind.name, "default=", p.default)
import t2pw.pipeline.extraction_diagnostics as diag
orig = diag.payload_hash
diag.payload_hash = lambda *a, **k: "PATCHED"
print("diag.payload_hash patch reaches service:", ex.diagnostic_payload_hash is diag.payload_hash)
diag.payload_hash = orig

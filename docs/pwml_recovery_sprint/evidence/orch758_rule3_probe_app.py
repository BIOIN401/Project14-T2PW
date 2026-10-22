"""ORCH-758 probe app (driven by orch758_rule3_probe.py through AppTest; never served).

Mirrors the batch path: click 1 runs "Stage 1" (activates the diagnostics recorder the way
streamlit_app._start_extraction_diagnostics does and records a 3/3 stage1_ensemble boundary,
the way run_ensemble does); click 2 runs the "post-pipeline", where Stage 6 calls
merge_on_mapped_identity. Each click is a separate AppTest script run, as in batch/driver.py.
"""
import os

import streamlit as st

from t2pw.pipeline import extraction_diagnostics as xd
from t2pw.pipeline import stage_one_ensemble as ens

os.environ[ens.ENV_VAR] = "3"
PAYLOAD = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}]},
           "processes": {"reactions": [{"name": "r", "inputs": ["A"], "outputs": ["B"]}]}}


def probe() -> dict:
    rec = xd.current()
    result = ens.merge_on_mapped_identity({**PAYLOAD}, report={})
    return {"artifact_dir": str(rec.artifact_dir or ""), "ensemble_boundaries":
            sum(1 for b in rec.boundaries if b.get("boundary") == ens.BOUNDARY_STAGE1_ENSEMBLE),
            "pooled_draws": ens._pooled_draws(), "rule3_applied": ens.IDENTITY_REPORT_KEY in result}


if st.button("run pipeline"):
    rec = xd.activate(run_id="orch758-probe", artifact_dir=os.environ["ORCH758_DIAG"], unique=True)
    rec.record_boundary(stage="extraction", boundary=ens.BOUNDARY_STAGE1_ENSEMBLE, attempts=3,
                        outcome=xd.OUTCOME_OK, ensemble={"draws_requested": 3, "draws_succeeded": 3})
    st.session_state["same_run"] = probe()
if st.button("post pipeline"):
    st.session_state["next_run"] = probe()

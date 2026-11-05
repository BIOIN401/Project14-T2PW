"""NEW ACCEPTANCE TEST (W0, new capability).

Pins the frozen ``t2pw.service.pathway`` interface: the field sets and defaults
of ``PathwayRequest`` / ``PathwayResult``, the stub's fixed return, that the
module imports without Streamlit / the app / the batch driver, and that its
mirrored vocabulary equals ``t2pw.batch.driver``'s.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from t2pw.rag.eligibility import RequestedScope
from t2pw.service import pathway
from t2pw.service.pathway import PathwayRequest, PathwayResult, run_pathway


def _fields(cls):
    return [f.name for f in dataclasses.fields(cls)]


def test_request_fields_and_defaults():
    assert _fields(PathwayRequest) == [
        "texts", "pdf_paths", "pmids", "scope", "export_mode", "out_dir", "deadline",
    ]
    req = PathwayRequest()
    assert req.texts == () and req.pdf_paths == () and req.pmids == ()
    assert req.scope == RequestedScope()
    assert req.export_mode == pathway.MODE_STRICT
    assert req.out_dir is None and req.deadline is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.export_mode = pathway.MODE_RESEARCH


def test_result_fields_and_defaults():
    assert _fields(PathwayResult) == [
        "status", "stage", "failure_kind", "message", "release_status",
        "pwml_path", "image_path", "report_paths", "attempt_log",
    ]
    a, b = PathwayResult(), PathwayResult()
    assert (a.status, a.stage, a.failure_kind, a.message) == ("error", "init", "", "")
    assert a.release_status is None and a.pwml_path is None and a.image_path is None
    assert a.report_paths == {} and a.attempt_log == []
    assert a.report_paths is not b.report_paths and a.attempt_log is not b.attempt_log


def test_stub_returns_fixed_error_and_writes_nothing(tmp_path):
    out = tmp_path / "out"
    req = PathwayRequest(
        texts=("some text",),
        scope=RequestedScope("glycolysis", "Homo sapiens"),
        out_dir=out,
    )
    result = run_pathway(req)
    assert result == run_pathway(PathwayRequest())  # ignores the request
    assert result == PathwayResult(
        status="error",
        stage="init",
        failure_kind="unknown",
        message="run_pathway is a W0 stub: no pipeline ran. R5 implements it.",
    )
    assert not out.exists()


def test_import_pulls_no_streamlit_app_or_driver():
    src = Path(pathway.__file__).resolve().parents[2]
    code = (
        "import sys, t2pw.service.pathway\n"
        "bad = sorted(m for m in sys.modules if m == 'streamlit' or m.startswith(("
        "'streamlit.', 't2pw.app', 't2pw.batch', 't2pw.llm')))\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(src))
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _vocab(module, prefix):
    return {
        name[len(prefix):]: value
        for name, value in vars(module).items()
        if name.startswith(prefix) and isinstance(value, str)
    }


@pytest.mark.parametrize(
    "driver_prefix, service_prefix",
    [("MODE_", "MODE_"), ("STAGE_", "STAGE_"), ("_STATUS_", "STATUS_"), ("KIND_", "KIND_")],
)
def test_vocabulary_matches_batch_driver(driver_prefix, service_prefix):
    from t2pw.batch import driver

    expected = _vocab(driver, driver_prefix)
    assert expected, driver_prefix
    assert _vocab(pathway, service_prefix) == expected

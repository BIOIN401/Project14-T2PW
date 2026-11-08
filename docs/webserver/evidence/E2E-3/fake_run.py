"""Fake run function for E2E-3 phase 1: no pipeline, no network, no LLM.

Selected with ``--run-fn fake_run:run_pathway`` (PYTHONPATH holds this folder). The
pathway name picks the behaviour so one browser submission can show every result kind:

* "success"  -> pass + release_ready, PWML, PNG diagram, three reports
* "review"   -> pass + review_required, PWML, PNG, warnings
* "fail"     -> fail / contract, no PWML, one diagnostic report
* "flaky"    -> attempt 1: error / network (the worker retries); attempt 2: success
* anything else -> instant success (used by the filler submissions)
"""
from __future__ import annotations

import struct
import time
import zlib
from pathlib import Path

from t2pw.pipeline.release_status import ReleaseStatus
from t2pw.service.pathway import (
    KIND_CONTRACT, KIND_NETWORK, STAGE_PWML_EXPORT, STAGE_STAGE1, STATUS_ERROR,
    STATUS_FAIL, STATUS_PASS, PathwayRequest, PathwayResult,
)

SLEEP_S = 14.0


def _png(path: Path, w: int = 360, h: int = 200) -> None:
    rows = b"".join(b"\x00" + bytes(v for x in range(w) for v in (30 + x * 200 // w, 90 + y * 120 // h, 180))
                    for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _rs(status: str) -> ReleaseStatus:
    return ReleaseStatus(status=status, pipeline_executed=True, strict_gates_passed=status == "release_ready")


def run_pathway(req: PathwayRequest) -> PathwayResult:
    name = req.scope.requested_pathway.lower()
    out = Path(req.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    attempt = int(out.parent.name.split("-")[-1])
    reports = {}

    def rep(fname: str, text: str) -> None:
        (out / fname).write_text(text, encoding="utf-8")
        reports[fname] = out / fname

    if "fail" in name:
        time.sleep(SLEEP_S)
        rep("gate_fail_report.json", '{"gate": "stage3", "failed": ["no_reactions_closed"]}\n')
        return PathwayResult(status=STATUS_FAIL, stage=STAGE_STAGE1, failure_kind=KIND_CONTRACT,
                             message=f"fake contract failure for {req.scope.requested_pathway}",
                             report_paths=reports)
    if "flaky" in name:
        time.sleep(6)
        if attempt == 1:
            return PathwayResult(status=STATUS_ERROR, stage=STAGE_STAGE1, failure_kind=KIND_NETWORK,
                                 message="fake network error on attempt 1")
    elif "success" in name or "review" in name:
        time.sleep(SLEEP_S)

    review = "review" in name
    pwml = out / ("pathway.review_required.pwml" if review else "pathway.pwml")
    pwml.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<pathway><name>%s</name>'
                    '<note>E2E-3 fake output</note></pathway>\n' % req.scope.requested_pathway, encoding="utf-8")
    png = out / "pathway.png"
    _png(png)
    rep("final_stage3_gate_report.json", '{"status": "ok"}\n')
    rep("coverage_summary.json", '{"coverage": 0.9}\n')
    rep("pwml_qa.json", '{"qa": "fake"}\n')
    rep("research_pathway_report.txt", "Fake report for E2E-3.\nLine two.\n")
    rep("research_pathway_elements.csv", "element,kind\nglucose,compound\n")
    return PathwayResult(
        status=STATUS_PASS, stage=STAGE_PWML_EXPORT,
        message="fake run complete",
        warnings=(["Fake warning: two reactions have no EC number.", "Fake warning: one compound unmapped."]
                  if review else []),
        release_status=_rs("review_required" if review else "release_ready"),
        pwml_path=pwml, image_path=png, report_paths=reports,
        counts={"reactions": 7})

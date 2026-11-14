"""One live case for the T2PW_TRANSPORT_SIDES test (D-W21c): real Stage 0, Stage 1 (+ boundary,
+ the flagged repair hook), Stage 2 with feedback loop, merge, then the offline normalizer.

usage: python run_case.py --paper PMC12096016 --arm off|on --rep 1 --out <dir> [--draws 1] [--stage1-only]

Runs the same functions a production leg runs (service.extraction.run_extraction) on the stored
paper text, with RAG off, T2PW_FINAL_PAYLOAD unset and the offline curator off. Writes, in
<out>/<paper>/<arm>_<rep>/ :
  stage1_pre.json   the Stage-1 payload as extracted (on arm: the input of the repair step)
  stage1_post.json  the Stage-1 payload Stage 2 receives (on arm: after the repair step)
  transport_sides_report.json (on arm only)
  merged_payload.json, normalized_payload.json (unless --stage1-only)
  summary.json      llm requests by stage (count, seconds), wall time, errors
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

SRC = "C:/t/fp/src"
RUNS = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
PAPERS = {  # id -> (stored run dir with 01_source_text.txt, topic, organism)
    "PMC12096016": ("runs_verify/2026-09-02_2052/papers/PMC12096016", "enterobactin biosynthesis", "Escherichia coli"),
    "PMC6112128": ("runs_smoke/2026-09-08_1528/papers/PMC6112128", "tubercidin biosynthesis", "Streptomyces tubercidicus"),
    "PMC7059786": ("runs_papers/ORCH-759p/legs/sour_taste_v2-strict/2026-10-22_2233/papers/PMC7059786", "Taste transduction: sour", "Mus musculus"),
    "PMC12421875": ("runs_verify/2026-09-02_2052/papers/PMC12421875", "menaquinone biosynthesis", "Bacillus subtilis"),
    "PMC12444477": ("runs_verify/2026-09-02_2052/papers/PMC12444477", "lipid A biosynthesis", "Escherichia coli"),
    "PMC11487621": ("runs_smoke/2026-09-07_2323/papers/PMC11487621", "coenzyme A biosynthesis", "Bacillus subtilis"),
    "PMC12452463": ("runs_verify/2026-09-02_2052/papers/PMC12452463", "enterobactin biosynthesis", "Escherichia coli"),
}


class StopRun(Exception):
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    ap.add_argument("--arm", required=True, choices=["off", "on"])
    ap.add_argument("--rep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", default="1")
    ap.add_argument("--stage1-only", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, SRC)
    import t2pw.llm.client as client  # loads .env (override=True) -- everything below is set AFTER this

    for k in ("T2PW_FINAL_PAYLOAD", "T2PW_OFFLINE_CURATOR", "T2PW_TRANSPORT_SIDES"):
        os.environ.pop(k, None)
    os.environ["T2PW_EXTRACTION_DRAWS"] = str(a.draws)
    if a.arm == "on":
        os.environ["T2PW_TRANSPORT_SIDES"] = "1"

    out = Path(a.out) / a.paper / f"{a.arm}_{a.rep}"
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "scratch"
    (scratch / "tmp").mkdir(parents=True, exist_ok=True)

    import t2pw.config as config
    import t2pw.paths as paths
    import t2pw.pipeline.pipeline as pl
    from t2pw.pipeline.export_mode import PATHWHIZ
    from t2pw.pipeline.process_normalizer import normalize_process_payload
    from t2pw.service import defaults as d
    from t2pw.service import extraction as ex
    from t2pw.service import transport_sides as ts
    from t2pw.service.reporter import LoggingReporter
    from t2pw.batch import driver

    paths.PROJECT_ROOT = scratch  # run_stage_one's artefact dir
    pl.TMP_DIR = scratch / "tmp"  # draft_graph.json goes here, not into the tracked tmp/
    real_rag = config.rag_config
    config.rag_config = lambda *x, **k: dict(real_rag(*x, **k), enabled=False)  # no RAG seed (4-6 min, not part of this test)

    # --- instrumentation: every provider request, by stage name -------------------------------
    calls = []
    real_create = client._create

    def counted(kwargs, stage):
        t0 = time.time()
        err = None
        try:
            return real_create(kwargs, stage)
        except BaseException as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"[:200]
            raise
        finally:
            calls.append({"stage": stage, "seconds": round(time.time() - t0, 1), "error": err})

    client._create = counted

    # --- snapshots of the Stage-1 payload either side of the repair hook ----------------------
    snaps = {}
    real_apply = ts.apply_to_stage_one

    def apply_wrap(stage_one, text, state, diagnostics_dir):
        snaps["pre"] = copy.deepcopy(stage_one)
        return real_apply(stage_one, text, state, diagnostics_dir)

    ts.apply_to_stage_one = apply_wrap
    real_filter = pl.filter_out_of_scope_reactions

    def filter_wrap(payload):
        snaps["post"] = copy.deepcopy(payload)
        snaps.setdefault("pre", snaps["post"])
        if a.stage1_only:
            raise StopRun()
        return real_filter(payload)

    pl.filter_out_of_scope_reactions = filter_wrap

    run_dir = RUNS + PAPERS[a.paper][0]
    text = Path(run_dir, "01_source_text.txt").read_text(encoding="utf-8")
    paper = {"topic": PAPERS[a.paper][1], "organism": PAPERS[a.paper][2]}
    state = {}
    request = ex.ExtractionRequest(
        text_entries=[text], pdfs=[], uses_pdf_input=False, input_mode="Paste text", pdf_page_range="",
        pdf_skip_refs=True, pdf_ocr=False, user_task_context=driver._extraction_focus(paper), export_mode=PATHWHIZ,
        temperature=d.TEMPERATURE, run_inference=d.RUN_INFERENCE, enable_chunking=d.ENABLE_CHUNKING,
        chunk_size=d.CHUNK_SIZE, chunk_overlap=d.CHUNK_OVERLAP, extract_attempts=d.EXTRACT_ATTEMPTS,
        infer_attempts=d.INFER_ATTEMPTS, infer_rounds=d.INFER_ROUNDS, extract_tokens=d.extract_tokens(),
        infer_tokens=d.infer_tokens(), rag_incomplete_flag=False)

    t0 = time.time()
    summary = {"paper": a.paper, "arm": a.arm, "rep": a.rep, "draws": a.draws, "stage1_only": a.stage1_only,
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "failure": None, "exception": None}
    result = None
    try:
        result = ex.run_extraction(request, reporter=LoggingReporter(), state=state)
        if result.failure is not None:
            summary["failure"] = f"{result.failure.kind}: {result.failure.message}"[:500]
    except StopRun:
        pass
    except Exception as exc:  # noqa: BLE001
        summary["exception"] = f"{type(exc).__name__}: {exc}"[:500]
    summary["wall_seconds"] = round(time.time() - t0, 1)

    def dump(name, obj):
        (out / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    if "pre" in snaps:
        dump("stage1_pre.json", snaps["pre"])
        dump("stage1_post.json", snaps["post"])
        summary["stage1_identical_pre_post"] = json.dumps(snaps["pre"], sort_keys=True) == json.dumps(snaps["post"], sort_keys=True)
    if "transport_sides" in state:
        dump("transport_sides_report.json", state["transport_sides"])
        summary["repair_calls"] = state["transport_sides"].get("calls")
        summary["deficient"] = len(state["transport_sides"].get("deficient", []))
        summary["repairs"] = len(state["transport_sides"].get("repairs", []))
        summary["repair_error"] = state["transport_sides"].get("error")
    if result is not None and result.failure is None and result.final_payload is not None:
        dump("merged_payload.json", result.final_payload)
        try:
            normalized, report = normalize_process_payload(copy.deepcopy(result.final_payload), mode=PATHWHIZ)
            dump("normalized_payload.json", normalized)
        except Exception as exc:  # noqa: BLE001
            summary["normalizer_error"] = f"{type(exc).__name__}: {exc}"[:300]

    by = {}
    for c in calls:
        row = by.setdefault(c["stage"], {"requests": 0, "seconds": 0.0, "errors": 0})
        row["requests"] += 1
        row["seconds"] = round(row["seconds"] + c["seconds"], 1)
        row["errors"] += 1 if c["error"] else 0
    summary["llm_by_stage"] = by
    summary["llm_requests_total"] = len(calls)
    summary["llm_errors"] = [c["error"] for c in calls if c["error"]][:10]
    dump("summary.json", summary)
    print(json.dumps({k: summary[k] for k in summary if k != "llm_errors"}, sort_keys=True))
    return 0 if summary["exception"] is None and summary["failure"] is None else 3


if __name__ == "__main__":
    sys.exit(main())

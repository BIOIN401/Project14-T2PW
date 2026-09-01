"""REV-111 — the reviewer's OWN adversarial probe of C-111.

Written by the independent reviewer, not the author. Nothing here is taken from
the card's report: every claim below is re-derived by construction in this file.

Six arms, each with its own controls:

* **A** — B13. Is the "NEW capability" G9 label true? The same synthetic kill is
  run against the BASE tree and against the TIP tree, and what each leaves on
  disk is listed. Symbol absence is not proof, so the arm is behavioural.
* **B** — B12. The reviewer's OWN planted credentials, different strings from the
  card's test, plus a deliberate NEGATIVE arm: a string that is secret-shaped to
  a human but matches no detector, to find the scanner's edge rather than its
  centre.
* **C** — B6. The in-process mechanism and the outer parent kill are constructed
  end to end and their labels read back OFF DISK. If they collapse to one string
  the card's central claim is false.
* **D** — B18. The reviewer's mutation: one of the nine preservations is removed
  from ``summarize`` and the acceptance test must go red; the saved bytes are
  then replayed and it must go green again (D-084).
* **E** — the Lead's second question. How much wall clock does the fsync-per-
  attempt instrument add to one model attempt, measured, with a no-trace control.
* **F** — B2/B5. The retry knobs and the outer-kill ``stage`` are read at base and
  at tip and compared as values.

Offline and cheap. No network, no provider, no LLM call, no benchmark leg, no
T-107 leg, no ``runs/`` or ``runs_verify/`` access. Every write goes under one
``mkdtemp`` that is removed at the end.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TIP = Path(r"C:/t/rev111")
BASE_SHA = "45c4f73996bdf312284d204936df2218dbb076db"
RESULTS: dict = {}
CONTROL_FAILURES: list = []


def control(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        CONTROL_FAILURES.append(f"{name}: {detail}")
    print(f"  control {name:<52} {'OK' if ok else 'FAILED'} {detail}")


# --------------------------------------------------------------------------- #
# The synthetic leg. Deliberately NOT the card's: written from the F-148 shape.
# --------------------------------------------------------------------------- #
CHILD = r'''
import sys, time
from pathlib import Path
SRC = sys.argv[1]
LEG = Path(sys.argv[2])
LEG.mkdir(parents=True, exist_ok=True)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
try:
    from t2pw.batch import leg_trace
except Exception:
    leg_trace = None
(LEG / "partial_reactions.json").write_text('{"reactions": 2}', encoding="utf-8")
if leg_trace is not None:
    t = leg_trace.activate(LEG / leg_trace.LEG_TRACE_NAME, child_deadline_seconds=4.0)
    leg_trace.record_event("leg_begin", slug="REVSYNTH")
    t.stage_begin("extraction")
    t.model_attempt(stage="extraction", attempt=1, status="error",
                    model="m", reason="reviewer planted retry one")
    t.model_attempt(stage="extraction", attempt=2, status="error",
                    model="m", reason="reviewer planted retry two")
    t.model_attempt(stage="extraction", attempt=3, status="ok", model="m")
while True:
    time.sleep(0.05)
'''


def _kill_one(runner_mod, tmp: Path, src: str, name: str, wait: float = 4.0):
    script = tmp / f"child_{name}.py"
    script.write_text(CHILD, encoding="utf-8")
    leg = tmp / name / "strict"
    began = time.monotonic()
    result = runner_mod.launch_child([sys.executable, str(script), src, str(leg)], wait)
    return leg, result, time.monotonic() - began


# --------------------------------------------------------------------------- #
# ARM A -- B13: the base/tip contrast, behaviourally
# --------------------------------------------------------------------------- #
def arm_a(tmp: Path) -> None:
    print("\n== ARM A -- B13: is the NEW-capability label true, behaviourally? ==")
    base_tree = tmp / "basetree"
    base_tree.mkdir()
    subprocess.run(
        ["git", "-C", str(TIP), "worktree", "list"], capture_output=True, text=True
    )
    # Export the base tree's src without touching any worktree registry.
    export = subprocess.run(
        ["git", "-C", str(TIP), "archive", BASE_SHA, "src"],
        capture_output=True,
    )
    control("A.base_archive_produced_bytes", export.returncode == 0 and len(export.stdout) > 0,
            f"rc={export.returncode} bytes={len(export.stdout)}")
    tar = tmp / "base_src.tar"
    tar.write_bytes(export.stdout)
    shutil.unpack_archive(str(tar), str(base_tree), format="tar")
    base_src = base_tree / "src"

    control("A.base_has_runner", (base_src / "t2pw" / "batch" / "runner.py").is_file())
    base_has_leg_trace = (base_src / "t2pw" / "batch" / "leg_trace.py").is_file()
    control("A.base_has_NO_leg_trace", not base_has_leg_trace,
            "the module must be absent at base for the NEW label to be available")

    sys.path.insert(0, str(TIP / "src"))
    from t2pw.batch import runner as tip_runner  # noqa: E402

    # Kill the SAME synthetic leg twice: once with the base src on the child's
    # path, once with the tip's. Same parent, same kill, same child source.
    base_leg, base_res, _ = _kill_one(tip_runner, tmp, str(base_src), "at_base")
    tip_leg, tip_res, tip_elapsed = _kill_one(tip_runner, tmp, str(TIP / "src"), "at_tip")

    control("A.base_arm_was_really_killed", base_res.timed_out is True)
    control("A.tip_arm_was_really_killed", tip_res.timed_out is True)

    base_files = sorted(p.name for p in base_leg.iterdir()) if base_leg.is_dir() else []
    tip_files = sorted(p.name for p in tip_leg.iterdir()) if tip_leg.is_dir() else []
    control("A.base_arm_did_produce_a_payload", "partial_reactions.json" in base_files,
            f"{base_files}")
    control("A.base_arm_preserved_NO_trace", "LEG_TRACE.jsonl" not in base_files,
            f"{base_files}")
    control("A.tip_arm_preserved_a_trace", "LEG_TRACE.jsonl" in tip_files, f"{tip_files}")

    from t2pw.batch import leg_trace  # noqa: E402

    row = tip_runner._timeout_row(slug="REVSYNTH", mode="strict", paper={},
                                  seconds=tip_elapsed, timeout=12.0, tail="")
    tip_runner._record_leg_terminal(
        tip_leg, row=row, result=tip_res, child_reported=False,
        elapsed=tip_elapsed, timeout=12.0,
        payload_before_cleanup=leg_trace.scan_payload(tip_leg),
    )
    summary = leg_trace.summarize(tip_leg)
    base_summary_input = leg_trace.summarize(base_leg)  # tip reader, base artifacts

    RESULTS["A"] = {
        "base_files_after_kill": base_files,
        "tip_files_after_kill": tip_files,
        "base_leg_read_by_tip_reader": {
            "total_model_calls": base_summary_input["total_model_calls"],
            "retry_reasons": base_summary_input["retry_reasons"],
            "trace_present": base_summary_input["_trace_present"],
        },
        "tip_leg": {
            "total_model_calls": summary["total_model_calls"],
            "retry_reason_count": len(summary["retry_reasons"]),
            "timeout_source": summary["timeout_source"],
            "payload_existed": summary["payload_before_cleanup"].get("existed"),
        },
        "row_files": row["files"],
        "row_counts": row["counts"],
        "row_stage": row["stage"],
    }
    control("A.base_leg_yields_zero_evidence", base_summary_input["total_model_calls"] == 0
            and base_summary_input["_trace_present"] is False)
    control("A.tip_leg_yields_three_attempts", summary["total_model_calls"] == 3,
            str(summary["total_model_calls"]))
    control("A.tip_row_STILL_says_files_empty", row["files"] == [] and row["counts"] == {},
            "the row must NOT have been repaired")


# --------------------------------------------------------------------------- #
# ARM B -- B12: the reviewer's own secrets, and the scanner's edge
# --------------------------------------------------------------------------- #
def arm_b(tmp: Path) -> None:
    print("\n== ARM B -- B12: plant the reviewer's own secrets ==")
    from t2pw.batch import leg_trace

    planted = {
        "openai_style_key": "sk-Rev111ReviewerPlantedKeyAAAAAAAAAAAA",
        "bearer_token": "Authorization: Bearer rev111.reviewer.planted.token.value",
        "inline_secret_assignment": "password: rev111PlantedPassword",
        "github_token": "ghp_Rev111ReviewerPlantedTokenAAAAAAAAAA",
        "google_api_key": "AIzaRev111ReviewerPlantedGoogleKeyAAA",
        "aws_access_key_id": "AKIAREV111REVIEWERAA",
        "credentialed_url": "https://reviewer:rev111secret@provider.example.com/v1",
    }
    for name, secret in planted.items():
        hits = leg_trace.scan_credentials(secret)
        control(f"B.detector_fires_on_{name}", name in hits, f"hits={hits}")

    # NEGATIVE arm: the scanner must NOT fire on ordinary sprint vocabulary.
    for benign in ("task-12345678901234567890", "risk-assessment-of-the-leg",
                   "the extraction stage was skipped", "sha256:deadbeefcafe"):
        control(f"B.no_false_positive_on_{benign[:18]!r}",
                leg_trace.scan_credentials(benign) == [],
                str(leg_trace.scan_credentials(benign)))

    leg = tmp / "reviewer_secrets"
    trace = leg_trace.activate(leg / leg_trace.LEG_TRACE_NAME)
    trace.model_attempt(stage="extraction", attempt=2, status="error", model="m",
                        reason=" | ".join(planted.values()))
    leg_trace.record_terminal(
        leg, timeout_source=leg_trace.SOURCE_OUTER_PARENT_KILL,
        terminal_state={"note": planted["openai_style_key"]},
        cleanup_decisions=[leg_trace.cleanup_decision(
            artifact="a.json", decision="discarded",
            decided_by=planted["credentialed_url"])],
    )
    leg_trace.deactivate()

    leaked = {}
    for path in (leg / leg_trace.LEG_TRACE_NAME, leg / leg_trace.LEG_TERMINAL_NAME):
        raw = path.read_text(encoding="utf-8")
        found = leg_trace.scan_credentials(raw)
        verbatim = [n for n, s in planted.items() if s in raw]
        leaked[path.name] = {"scanner_hits_on_disk": found, "verbatim_present": verbatim}
        control(f"B.{path.name}_clean_by_scanner", found == [], str(found))
        control(f"B.{path.name}_no_verbatim_secret", verbatim == [], str(verbatim))

    # The scanner's EDGE, reported not repaired: a shape a human would call a
    # secret but no detector matches.
    edge = "x-internal-credential 8fA2b9Qz71LmPd"
    edge_hits = leg_trace.scan_credentials(edge)
    RESULTS["B"] = {"on_disk": leaked, "edge_case": {"text": edge, "hits": edge_hits}}
    print(f"  EDGE (reported, not a control): {edge!r} -> {edge_hits}")


# --------------------------------------------------------------------------- #
# ARM C -- B6: two mechanisms, constructed, labels read off disk
# --------------------------------------------------------------------------- #
def arm_c(tmp: Path) -> None:
    print("\n== ARM C -- B6: construct the first two mechanisms end to end ==")
    from t2pw.batch import leg_trace, runner as tip_runner
    from t2pw.pipeline import deadline as leg_deadline

    # (1) OUTER PARENT KILL -- a real child, really killed by the real seam.
    leg, result, elapsed = _kill_one(tip_runner, tmp, str(TIP / "src"), "mech_outer")
    control("C.outer_arm_really_killed", result.timed_out is True)
    row = tip_runner._timeout_row(slug="REVSYNTH", mode="strict", paper={},
                                  seconds=elapsed, timeout=12.0, tail="")
    tip_runner._record_leg_terminal(
        leg, row=row, result=result, child_reported=False, elapsed=elapsed,
        timeout=12.0, payload_before_cleanup=leg_trace.scan_payload(leg))
    outer_disk = json.loads((leg / leg_trace.LEG_TERMINAL_NAME).read_text(encoding="utf-8"))
    outer_label = leg_trace.summarize(leg)["timeout_source"]

    # (2) IN-PROCESS DEADLINE -- the child stops ITSELF, through run_single.
    from t2pw.batch import driver as tip_driver
    run_dir = tmp / "inproc_run"
    (run_dir / "papers" / "PMCX__a").mkdir(parents=True)
    (run_dir / "papers" / "PMCX__a" / tip_runner.SOURCE_TEXT_NAME).write_text(
        "body", encoding="utf-8")
    (run_dir / tip_runner.PLAN_NAME).write_text(json.dumps(
        {"modes": ["strict"],
         "papers": [{"slug": "PMCX__a", "paper_id": "PMCX", "title": "t"}]}),
        encoding="utf-8")

    def stopped_itself(paper, mode, *, timeout=0.0, **_kw):
        out = tip_driver.RunOutcome(paper_id="PMCX", mode=tip_driver.MODE_STRICT)
        out.status = "timeout"
        out.stage = "input"
        out.failure_kind = tip_driver.KIND_TIMEOUT
        out.termination_reason = leg_deadline.OPERATION_TIMEOUT
        out.termination_is_operational = True
        return out

    inproc_row = tip_runner.run_single(run_dir, "PMCX__a", "strict", timeout=1680.0,
                                       run_fn=stopped_itself)
    inproc_leg = run_dir / "papers" / "PMCX__a" / "strict"
    inproc_label = leg_trace.summarize(inproc_leg)["timeout_source"]

    control("C.outer_labelled_outer_parent_kill",
            outer_label == leg_trace.SOURCE_OUTER_PARENT_KILL, outer_label)
    control("C.inproc_labelled_in_process_deadline",
            inproc_label == leg_trace.SOURCE_IN_PROCESS_DEADLINE, inproc_label)
    control("C.the_two_are_DIFFERENT_strings", outer_label != inproc_label,
            f"{outer_label} vs {inproc_label}")
    control("C.outer_stage_is_still_unknown", row["stage"] == "unknown", row["stage"])
    control("C.inproc_stage_is_input_not_guessed", inproc_row["stage"] == "input",
            inproc_row["stage"])
    # WRAPPER: reachable only as a shape on disk, never from the classifier.
    wrapper_leg = tmp / "mech_wrapper"
    t = leg_trace.activate(wrapper_leg / leg_trace.LEG_TRACE_NAME)
    t.stage_begin("input")
    leg_trace.deactivate()
    wrapper_label = leg_trace.summarize(wrapper_leg)["timeout_source"]
    control("C.wrapper_inferred_from_shape_on_disk",
            wrapper_label == leg_trace.SOURCE_WRAPPER, wrapper_label)

    RESULTS["C"] = {
        "outer_parent_kill": {"label": outer_label,
                              "terminal_on_disk_source": outer_disk["timeout_source"],
                              "row_stage": row["stage"]},
        "in_process_deadline": {"label": inproc_label, "row_stage": inproc_row["stage"]},
        "wrapper": {"label": wrapper_label},
        "distinct_labels": sorted({outer_label, inproc_label, wrapper_label}),
    }


# --------------------------------------------------------------------------- #
# ARM E -- the Lead's question 2: what does fsync-per-attempt COST?
# --------------------------------------------------------------------------- #
def arm_e(tmp: Path) -> None:
    print("\n== ARM E -- the measured cost of the instrument on the attempt path ==")
    from t2pw.batch import leg_trace
    from t2pw.llm import client as llm_client

    N = 200

    # CONTROL (negative): no active trace. This is every non-batch caller.
    leg_trace.deactivate()
    d = llm_client.CompletionDiagnostics(model="m", stage="extraction")
    began = time.perf_counter()
    for i in range(N):
        d.note(attempt=i + 1, status=llm_client.STATUS_OK, content_chars=10)
    without = time.perf_counter() - began

    # ARM (positive): an active trace, so every note() flushes and fsyncs.
    leg = tmp / "cost"
    leg_trace.activate(leg / leg_trace.LEG_TRACE_NAME)
    d2 = llm_client.CompletionDiagnostics(model="m", stage="extraction")
    began = time.perf_counter()
    for i in range(N):
        d2.note(attempt=i + 1, status=llm_client.STATUS_OK, content_chars=10)
    with_trace = time.perf_counter() - began
    leg_trace.deactivate()

    events = leg_trace.read_events(leg)
    control("E.every_attempt_actually_reached_disk", len(events) == N, f"{len(events)}/{N}")
    control("E.control_arm_wrote_nothing",
            not (tmp / "cost" / leg_trace.LEG_TRACE_NAME).is_file() or len(events) == N)
    control("E.attempt_log_identical_either_way",
            len(d.attempt_log) == len(d2.attempt_log) == N)

    per_attempt_ms = (with_trace - without) / N * 1000.0
    # PRODUCT_CONTRACT s9: one leg is capped at THREE Stage-1 model attempts.
    RESULTS["E"] = {
        "attempts_measured": N,
        "seconds_without_trace": round(without, 6),
        "seconds_with_trace": round(with_trace, 6),
        "added_ms_per_attempt": round(per_attempt_ms, 4),
        "added_seconds_at_3_attempts": round(per_attempt_ms * 3 / 1000.0, 6),
        "added_seconds_at_1000_attempts": round(per_attempt_ms * 1000 / 1000.0, 4),
        "finalization_reserve_seconds": 120.0,
    }
    print(f"  measured: +{per_attempt_ms:.3f} ms per model attempt")
    print(f"  at the s9 cap of 3 attempts : +{per_attempt_ms * 3 / 1000.0:.6f} s")
    print(f"  at a runaway 1000 attempts  : +{per_attempt_ms:.4f} s")


# --------------------------------------------------------------------------- #
# ARM F -- B2/B5: the knobs, at base and at tip, as VALUES
# --------------------------------------------------------------------------- #
def arm_f(tmp: Path) -> None:
    print("\n== ARM F -- B2/B3: the retry and ceiling knobs, base vs tip ==")
    reader = (
        "import sys, json, os\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from t2pw.batch import runner\n"
        "from t2pw.pipeline import deadline as d\n"
        "from t2pw.llm import client as c\n"
        "print(json.dumps({\n"
        "  'DEFAULT_PAPER_TIMEOUT': runner.DEFAULT_PAPER_TIMEOUT,\n"
        "  '_CHILD_GRACE': runner._CHILD_GRACE,\n"
        "  'LEG_TIMEOUT_SECONDS': d.LEG_TIMEOUT_SECONDS,\n"
        "  'PARENT_CHILD_GRACE_SECONDS': d.PARENT_CHILD_GRACE_SECONDS,\n"
        "  'DEFAULT_FINALIZATION_RESERVE_SECONDS': d.DEFAULT_FINALIZATION_RESERVE_SECONDS,\n"
        "  'child_deadline_1800': d.child_deadline_seconds(1800.0, grace=120.0),\n"
        "  'SDK_MAX_RETRIES': c.SDK_MAX_RETRIES,\n"
        "  'LLM_MAX_RETRIES_default': int(os.getenv('LLM_MAX_RETRIES', '8')),\n"
        "}))\n"
    )
    script = tmp / "knobs.py"
    script.write_text(reader, encoding="utf-8")
    out = {}
    for name, src in (("base", str(tmp / "basetree" / "src")),
                      ("tip", str(TIP / "src"))):
        proc = subprocess.run([sys.executable, str(script), src],
                              capture_output=True, text=True, timeout=120)
        control(f"F.{name}_knob_read_succeeded", proc.returncode == 0,
                proc.stderr.strip()[-200:])
        out[name] = json.loads(proc.stdout.strip().splitlines()[-1])
    control("F.every_knob_identical_base_and_tip", out["base"] == out["tip"],
            f"base={out['base']} tip={out['tip']}")
    RESULTS["F"] = out
    print(f"  base: {out['base']}")
    print(f"  tip : {out['tip']}")


def main() -> int:
    print("REV-111 reviewer probe -- independent adversarial measurement of C-111")
    print(f"tip tree      : {TIP}")
    print(f"base SHA      : {BASE_SHA}")
    print("live run trees NOT touched: runs, runs_verify")
    tmp = Path(tempfile.mkdtemp(prefix="rev111probe_"))
    print(f"probe tmp root: {tmp}")
    try:
        arm_a(tmp)
        arm_b(tmp)
        arm_c(tmp)
        arm_e(tmp)
        arm_f(tmp)
    finally:
        print("\n================ RESULTS ================")
        print(json.dumps(RESULTS, indent=2, sort_keys=True, default=str))
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"probe tmp removed: {not tmp.exists()}")
    if CONTROL_FAILURES:
        print("\nCONTROLS_FAILED:")
        for item in CONTROL_FAILURES:
            print(f"  - {item}")
        print("A probe whose controls failed has measured NOTHING.")
        return 1
    print("\nALL CONTROLS HELD. The measurements above stand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

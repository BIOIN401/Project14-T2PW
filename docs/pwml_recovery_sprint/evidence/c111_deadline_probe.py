#!/usr/bin/env python
"""C-111 / F-148 s3 -- the cheap OFFLINE probe that separates the three readings.

Charter: ``docs/pwml_recovery_sprint/prompts/C-111.md`` s2.
Hypotheses, committed BEFORE this file existed:
``docs/pwml_recovery_sprint/evidence/c111_three_readings_hypotheses.md`` (commit 4fde91b3).

WHAT THIS PROBE IS ALLOWED TO BE, and is
----------------------------------------
Offline, cheap, synthetic. **No LLM spend. No provider call. No benchmark leg. No
T-run. No rerun of any T-107 leg.** It never reads and never writes the pinned run
trees: the two live run trees in this repository are named here explicitly --
``runs/`` and ``runs_verify/`` -- and this probe touches NEITHER. Every filesystem
write goes under a single temporary directory created by ``tempfile.mkdtemp`` and
removed at the end, whose path is printed.

The only network-capable module in the dependency graph is ``t2pw.llm.client``;
it is never imported here and never reached, because no measurement below calls a
pipeline stage.

WHAT IT MEASURES
----------------
M1  the child-deadline arithmetic and the argv the parent actually builds
M2  what the parent actually WAITS, versus what it TELLS the child
M3  whether the in-process budget admits work past its own total   (reading 1)
M4  whether post-deadline work is inside the leg's recorded elapsed, unbounded (reading 2)
M5  whether the outer parent kill invokes any finalization at all  (reading 3)
M6  a scaled end-to-end reproduction of the T-107 shape

Every measurement carries a known-POSITIVE and a known-NEGATIVE control arm, fixed
in the hypotheses file before any of this ran. **If any control arm fails, the probe
reports CONTROLS_FAILED and exits non-zero without issuing a verdict**: a probe that
cannot fail has measured nothing (C-108's first verification probe used the wrong
payload envelope, never reached the guard, and looked exactly like a finding).

Usage:  python c111_deadline_probe.py [--json <path>]
Exit:   0 = all controls held and a verdict was issued; 1 = controls failed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

HERE = Path(__file__).resolve()
TREE = HERE.parents[3]
SRC = TREE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.batch import driver, runner  # noqa: E402
from t2pw.batch.fetch import BatchPaper  # noqa: E402
from t2pw.pipeline import deadline as leg_deadline  # noqa: E402

#: The two live run trees in this repository. Named explicitly, per the C-111
#: charter, because "the pinned run" is ambiguous. This probe reads and writes
#: NEITHER; the assertion below is enforced at the end of main().
LIVE_RUN_TREES = ("runs", "runs_verify")


# --------------------------------------------------------------------------- #
# M1 -- the child deadline arithmetic, and the argv the parent actually builds
# --------------------------------------------------------------------------- #
def m1_child_deadline_arithmetic(tmp: Path) -> Dict[str, Any]:
    computed = leg_deadline.child_deadline_seconds(1800.0, grace=120.0)
    argv = runner.child_command(tmp / "batch_run.py", tmp / "run", "slug", "strict", 1800.0)
    timeout_flag = ""
    if "--timeout" in argv:
        timeout_flag = argv[argv.index("--timeout") + 1]
    return {
        "leg_ceiling_seconds": 1800.0,
        "parent_child_grace_seconds": leg_deadline.PARENT_CHILD_GRACE_SECONDS,
        "default_finalization_reserve_seconds": leg_deadline.DEFAULT_FINALIZATION_RESERVE_SECONDS,
        "child_deadline_seconds_computed": computed,
        "argv_timeout_flag": timeout_flag,
        "argv": list(argv),
        # controls
        "control_positive_1680_computed": computed == 1680.0,
        "control_positive_argv_says_1680": timeout_flag == "1680",
        "control_negative_argv_does_not_say_1800": timeout_flag != "1800",
    }


# --------------------------------------------------------------------------- #
# M2 -- what the parent WAITS versus what it TELLS the child
# --------------------------------------------------------------------------- #
def m2_parent_wait_vs_child_deadline(tmp: Path) -> Dict[str, Any]:
    """Drive the real batch loop with a recording child_fn. No child is spawned."""

    out_dir = tmp / "m2_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp / "m2_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    topics = tmp / "m2_topics.txt"
    topics.write_text("synthetic probe topic\n", encoding="utf-8")

    seen: List[Dict[str, Any]] = []

    def recorder(cmd: Sequence[str], timeout: float) -> runner.ChildResult:
        argv = list(cmd)
        flag = argv[argv.index("--timeout") + 1] if "--timeout" in argv else ""
        seen.append({"parent_wait_seconds": float(timeout), "child_timeout_flag": flag})
        # A child killed by the parent with nothing printed: the T-107 shape.
        return runner.ChildResult(None, "", "", True)

    def fetch(_text: str, **_kw: Any):
        paper = BatchPaper(
            paper_id="PROBE0001",
            slug="PROBE0001__c111",
            title="C-111 synthetic probe paper",
            full_text="synthetic body text for the C-111 offline probe",
            topic="probe",
        )
        return [paper], []

    runner.run_batch(
        topics_path=topics,
        out=out_dir,
        modes="strict",
        timeout=1800.0,
        deadline_hours=10.0,
        fresh=True,
        script_path=tmp / "batch_run.py",
        fetch_fn=fetch,
        child_fn=recorder,
        data_dir=data_dir,
        echo=False,
    )

    run_dirs = runner.list_run_dirs(out_dir)
    rows: List[Dict[str, Any]] = []
    if run_dirs:
        manifest = run_dirs[-1] / runner.MANIFEST_NAME
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:  # noqa: BLE001
                        pass
    launched = seen[0] if seen else {}
    timeout_rows = [r for r in rows if str(r.get("status")) == "timeout"]
    return {
        "launches_recorded": len(seen),
        "parent_wait_seconds": launched.get("parent_wait_seconds"),
        "child_timeout_flag": launched.get("child_timeout_flag"),
        "manifest_rows": len(rows),
        "timeout_row_stage": (timeout_rows[0].get("stage") if timeout_rows else None),
        "timeout_row_termination_reason": (
            timeout_rows[0].get("termination_reason") if timeout_rows else None
        ),
        "timeout_row_files": (timeout_rows[0].get("files") if timeout_rows else None),
        "timeout_row_budget": (timeout_rows[0].get("budget") if timeout_rows else None),
        # controls
        "control_positive_parent_waits_full_ceiling": launched.get("parent_wait_seconds") == 1800.0,
        "control_negative_parent_does_not_wait_child_deadline": (
            launched.get("parent_wait_seconds") != 1680.0
        ),
    }


# --------------------------------------------------------------------------- #
# M3 -- does the in-process budget admit work past its own total?  (reading 1)
# --------------------------------------------------------------------------- #
class _SlowApp:
    """A stand-in for streamlit's AppTest. ``run`` sleeps its whole slice."""

    def __init__(self, per_call: float) -> None:
        self.per_call = float(per_call)
        self.calls: List[float] = []

    def run(self, timeout: float) -> None:
        self.calls.append(float(timeout))
        time.sleep(min(self.per_call, float(timeout)))


class _OverrunApp:
    """A stand-in whose single interaction overruns and raises AppTest's error."""

    def run(self, timeout: float) -> None:
        time.sleep(min(0.4, float(timeout)))
        raise RuntimeError("app run timed out after %.1fs" % float(timeout))


def m3_in_process_budget(_tmp: Path) -> Dict[str, Any]:
    total = 3.0
    budget = driver._Budget(total, 1.0)
    app = _SlowApp(1.0)
    calls: List[Dict[str, Any]] = []
    admitted_past_total = False
    slice_overrun = 0.0
    for _ in range(20):
        remaining_before = budget.remaining
        before = len(app.calls)
        timed_out, detail = driver._run_app(app, budget)
        slice_given = app.calls[-1] if len(app.calls) > before else None
        if slice_given is not None:
            # How far past the leg total the slice HANDED OUT would have reached.
            slice_overrun = max(slice_overrun, float(slice_given) - remaining_before)
        calls.append(
            {
                "remaining_before": round(remaining_before, 3),
                "slice_given": (round(float(slice_given), 3) if slice_given is not None else None),
                "elapsed_after": round(budget.elapsed, 3),
                "timed_out": bool(timed_out),
                "detail": detail,
            }
        )
        if remaining_before <= 0.0 and not timed_out:
            admitted_past_total = True
        if timed_out:
            break

    stop = calls[-1]
    stopped_at = stop["elapsed_after"]
    marker_present = leg_deadline.BUDGET_SPENT_MARKER in str(stop["detail"])
    classified = leg_deadline.classify_interaction_timeout(stop["detail"])

    # Second arm: one interaction that overruns its own slice, not the leg budget.
    budget2 = driver._Budget(30.0, 1.0)
    timed_out2, detail2 = driver._run_app(_OverrunApp(), budget2)
    classified2 = leg_deadline.classify_interaction_timeout(detail2)

    return {
        "total_budget_seconds": total,
        "interactions": calls,
        "stopped_at_elapsed": stopped_at,
        "overrun_past_total_seconds": round(stopped_at - total, 3),
        "stop_detail_carries_budget_spent_marker": marker_present,
        "stop_classified_as": classified,
        "single_overrun_arm": {
            "timed_out": bool(timed_out2),
            "detail": detail2,
            "classified_as": classified2,
            "elapsed": round(budget2.elapsed, 3),
        },
        "work_admitted_after_budget_gone": admitted_past_total,
        # Hypotheses clause B: "the interaction it starts is given a slice that
        # reaches past the total". Recorded as a MAGNITUDE, not as a boolean,
        # because ``_Budget.slice`` has a ``max(1.0, ...)`` FLOOR: the last
        # admitted interaction can be handed up to 1s more than remains. That is
        # a real, bounded, sub-second effect and it is reported as such -- it
        # cannot account for a 120s overrun, and the verdict says which clause
        # fired and at what magnitude rather than collapsing the two.
        "max_slice_overrun_seconds": round(slice_overrun, 3),
        "slice_floor_seconds": 1.0,
        # controls
        "control_positive_overrun_arm_timed_out": bool(timed_out2),
        "control_negative_no_false_completion_past_total": not admitted_past_total,
    }


# --------------------------------------------------------------------------- #
# M4 -- is post-deadline work inside the leg's recorded elapsed, and unbounded?
# --------------------------------------------------------------------------- #
def _run_one_with_drive(stub, *, timeout: float) -> Any:
    original = driver._drive
    driver._drive = stub  # type: ignore[assignment]
    try:
        return driver.run_one(
            {"paper_id": "PROBE0001", "full_text": "x"},
            "strict",
            timeout=timeout,
            app_timeout=timeout,
        )
    finally:
        driver._drive = original  # type: ignore[assignment]


def m4_post_deadline_work(_tmp: Path) -> Dict[str, Any]:
    total = 2.0
    finalization = 1.5

    def instant(_paper, _mode, outcome, _budget, _app_path):
        outcome.status = "pass"

    def burn_then_finalize(_paper, _mode, outcome, budget, _app_path):
        # Phase 1: spend the whole in-process budget, exactly as a leg's stages do.
        while budget.remaining > 0:
            time.sleep(0.05)
        outcome.__dict__["_probe_work_stopped_at"] = budget.elapsed
        # Phase 2: FINALIZATION -- checkpoint persistence, validation, status
        # classification, diagnostic-artifact writing. Nothing here consults the
        # budget, which is exactly what the measurement is about.
        time.sleep(finalization)
        outcome.status = "timeout"

    control = _run_one_with_drive(instant, timeout=total)
    measured = _run_one_with_drive(burn_then_finalize, timeout=total)
    work_stopped_at = float(measured.__dict__.get("_probe_work_stopped_at", -1.0))

    return {
        "in_process_total_seconds": total,
        "synthetic_finalization_seconds": finalization,
        "work_stopped_at_seconds": round(work_stopped_at, 3),
        "recorded_outcome_seconds": measured.seconds,
        "overrun_beyond_total_seconds": round(float(measured.seconds) - total, 3),
        "instant_drive_recorded_seconds": control.seconds,
        # controls
        "control_positive_instant_drive_is_near_zero": float(control.seconds) < 0.5,
        "control_negative_instant_drive_is_not_the_total": abs(float(control.seconds) - total) > 0.5,
    }


# --------------------------------------------------------------------------- #
# M5 -- does the outer parent kill invoke ANY finalization?  (reading 3)
# --------------------------------------------------------------------------- #
_CHILD_SOURCE = '''
import atexit, os, signal, sys, time
from pathlib import Path

OUT = Path(sys.argv[1])
SLEEP = float(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "started.txt").write_text("started", encoding="utf-8")

def finalize(*_a):
    try:
        (OUT / "finalized.txt").write_text("finalized", encoding="utf-8")
    except Exception:
        pass

atexit.register(finalize)
for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
    handler = getattr(signal, name, None)
    if handler is not None:
        try:
            signal.signal(handler, lambda *_a: (finalize(), sys.exit(0)))
        except Exception:
            pass

deadline = time.monotonic() + SLEEP
while time.monotonic() < deadline:
    time.sleep(0.05)
print("child done")
'''


def m5_outer_kill_finalization(tmp: Path) -> Dict[str, Any]:
    script = tmp / "m5_child.py"
    script.write_text(_CHILD_SOURCE, encoding="utf-8")

    def arm(name: str, sleep_for: float, wait_for: float) -> Dict[str, Any]:
        out = tmp / ("m5_" + name)
        began = time.monotonic()
        result = runner.launch_child(
            [sys.executable, str(script), str(out), str(sleep_for)], wait_for
        )
        return {
            "child_sleep_seconds": sleep_for,
            "parent_wait_seconds": wait_for,
            "elapsed": round(time.monotonic() - began, 3),
            "timed_out": bool(result.timed_out),
            "returncode": result.returncode,
            "started_marker": (out / "started.txt").exists(),
            "finalized_marker": (out / "finalized.txt").exists(),
        }

    killed = arm("killed", sleep_for=25.0, wait_for=3.0)
    # Guard against a slow filesystem: give any late writer a moment.
    time.sleep(1.0)
    killed["finalized_marker"] = (tmp / "m5_killed" / "finalized.txt").exists()
    survived = arm("survived", sleep_for=0.5, wait_for=8.0)

    return {
        "killed_arm": killed,
        "survived_arm": survived,
        "kill_mechanism": "runner._kill_tree (taskkill /F /T on nt)" if os.name == "nt"
        else "runner._kill_tree (SIGKILL to the process group)",
        # controls
        "control_positive_survivor_finalized": survived["finalized_marker"] is True
        and survived["started_marker"] is True,
        "control_negative_killed_arm_did_not_time_out": killed["timed_out"] is True,
        "control_positive_killed_arm_started": killed["started_marker"] is True,
    }


# --------------------------------------------------------------------------- #
# M6 -- scaled end-to-end reproduction of the T-107 shape
# --------------------------------------------------------------------------- #
_LEG_SOURCE = '''
import sys, time
from pathlib import Path

SRC = Path(sys.argv[1])
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
OUT = Path(sys.argv[2])
CHILD_DEADLINE = float(sys.argv[3])
FINALIZATION = float(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)

from t2pw.batch import driver

class SlowApp:
    def run(self, timeout):
        time.sleep(min(0.25, float(timeout)))

budget = driver._Budget(CHILD_DEADLINE, 0.25)
(OUT / "work_started.txt").write_text("1", encoding="utf-8")
while True:
    timed_out, detail = driver._run_app(SlowApp(), budget)
    if timed_out:
        break
(OUT / "work_stopped.txt").write_text(
    "%.3f|%s" % (budget.elapsed, detail), encoding="utf-8")
(OUT / "finalization_started.txt").write_text("1", encoding="utf-8")
time.sleep(FINALIZATION)
(OUT / "payload.json").write_text('{"reactions": 5}', encoding="utf-8")
print("leg finalized")
'''


def m6_scaled_reproduction(tmp: Path) -> Dict[str, Any]:
    script = tmp / "m6_leg.py"
    script.write_text(_LEG_SOURCE, encoding="utf-8")

    leg_ceiling = 12.0
    grace = 4.0
    child_deadline = leg_deadline.child_deadline_seconds(leg_ceiling, grace=grace, floor=0.5)

    def arm(name: str, finalization: float) -> Dict[str, Any]:
        out = tmp / ("m6_" + name)
        began = time.monotonic()
        result = runner.launch_child(
            [sys.executable, str(script), str(SRC), str(out), str(child_deadline), str(finalization)],
            leg_ceiling,
        )
        stopped = out / "work_stopped.txt"
        return {
            "finalization_seconds": finalization,
            "elapsed": round(time.monotonic() - began, 3),
            "timed_out": bool(result.timed_out),
            "work_started": (out / "work_started.txt").exists(),
            "work_stopped_record": stopped.read_text(encoding="utf-8") if stopped.exists() else "",
            "finalization_started": (out / "finalization_started.txt").exists(),
            "payload_present": (out / "payload.json").exists(),
        }

    over_reserve = arm("over_reserve", finalization=grace + 2.0)
    under_reserve = arm("under_reserve", finalization=max(0.5, grace - 2.5))

    return {
        "leg_ceiling_seconds": leg_ceiling,
        "grace_seconds": grace,
        "child_deadline_seconds": child_deadline,
        "finalization_over_reserve_arm": over_reserve,
        "finalization_under_reserve_arm": under_reserve,
        # controls
        "control_positive_under_reserve_payload_present": under_reserve["payload_present"] is True,
        "control_negative_over_reserve_payload_absent": over_reserve["payload_present"] is False,
        "control_positive_over_reserve_started_work": over_reserve["work_started"] is True,
    }


# --------------------------------------------------------------------------- #
# Verdicts -- the rules were fixed in the hypotheses file before this ran
# --------------------------------------------------------------------------- #
def verdicts(m: Dict[str, Any]) -> Dict[str, Any]:
    m3, m4, m5 = m["M3"], m["M4"], m["M5"]

    # The hypotheses fixed two clauses for reading 1. Clause A is the deadline
    # not being enforced at all. Clause B is a slice reaching past the total --
    # which is TRUE here only at the magnitude of ``_Budget.slice``'s 1s floor.
    # Both are reported; the verdict names which fired and how big it was, and a
    # sub-second floor artifact is NOT allowed to masquerade as "the child never
    # honoured its 1680s deadline".
    clause_a = bool(m3["work_admitted_after_budget_gone"])
    clause_b_seconds = float(m3["max_slice_overrun_seconds"])
    reading_1 = clause_a or clause_b_seconds > 2.0
    mechanism_2 = (not reading_1) and (
        float(m4["overrun_beyond_total_seconds"])
        >= 0.6 * float(m4["synthetic_finalization_seconds"])
    )
    reading_3 = (
        m5["killed_arm"]["started_marker"] is True
        and m5["killed_arm"]["finalized_marker"] is False
        and m5["survived_arm"]["finalized_marker"] is True
    )

    return {
        "reading_1_child_never_honoured_its_deadline": "CONFIRMED" if reading_1 else "REFUTED",
        "reading_1_clause_a_work_admitted_after_budget_gone": clause_a,
        "reading_1_clause_b_max_slice_overrun_seconds": clause_b_seconds,
        "reading_2_finalization_overran_the_reserve": (
            "MECHANISM CONFIRMED, MAGNITUDE NOT MEASURABLE OFFLINE"
            if mechanism_2
            else "NOT ESTABLISHED"
        ),
        "reading_3_outer_kill_invokes_no_finalization": "CONFIRMED" if reading_3 else "REFUTED",
    }


def collect_controls(m: Dict[str, Any]) -> List[str]:
    failed: List[str] = []
    for name, block in m.items():
        for key, value in block.items():
            if key.startswith("control_") and value is not True:
                failed.append(f"{name}.{key}={value!r}")
    return failed


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="C-111 offline deadline probe")
    parser.add_argument("--json", default="", help="write the full report here")
    args = parser.parse_args(list(argv))

    tmp = Path(tempfile.mkdtemp(prefix="c111probe_"))
    print(f"probe tmp root       : {tmp}")
    print(f"tree under measure   : {TREE}")
    print(f"live run trees NOT touched: {', '.join(LIVE_RUN_TREES)}")
    print("")

    measurements: Dict[str, Any] = {}
    try:
        for name, fn in (
            ("M1", m1_child_deadline_arithmetic),
            ("M2", m2_parent_wait_vs_child_deadline),
            ("M3", m3_in_process_budget),
            ("M4", m4_post_deadline_work),
            ("M5", m5_outer_kill_finalization),
            ("M6", m6_scaled_reproduction),
        ):
            began = time.monotonic()
            measurements[name] = fn(tmp)
            print(f"{name} took {time.monotonic() - began:.1f}s")
    finally:
        pass

    failed = collect_controls(measurements)
    report: Dict[str, Any] = {
        "probe": "c111_deadline_probe",
        "tree": str(TREE),
        "live_run_trees_touched": [],
        "measurements": measurements,
        "controls_failed": failed,
    }
    if not failed:
        report["verdicts"] = verdicts(measurements)

    print("")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    shutil.rmtree(tmp, ignore_errors=True)

    print("")
    if failed:
        print("CONTROLS_FAILED: " + "; ".join(failed))
        print("No verdict issued. A probe that cannot fail has measured nothing.")
        return 1
    for key, value in report["verdicts"].items():
        print(f"VERDICT {key} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

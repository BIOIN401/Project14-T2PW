"""REV-110 -- independent adversarial probe of C-110 condition 3.

INDEPENDENT OF THE AUTHOR'S TESTS. Nothing here imports
``tests/test_c110_negative_control_status.py``; the row shapes are rebuilt from
``batch/runner.py::_timeout_row`` / ``_crash_row`` and ``batch/driver.py::_kind``
directly, so a fixture bug in the author's file cannot propagate into this
verdict.

THE CRUX: the card can only fail in one direction that matters -- by rewarding
an empty result that was a CASUALTY rather than a DECISION. Everything below
attacks that.

HARNESS SELF-CHECK FIRST. REV-108's verification probe was wrong because it
built the wrong payload envelope, never reached the guard, and every case came
back permissive -- which looked like a finding. So this probe asserts a
KNOWN-POSITIVE and a KNOWN-NEGATIVE before it reports anything else, and
refuses to print a verdict if either fails.

NO COMMITTED RUN IS OPENED. T-107 (``runs_verify/2026-08-28_1816``) is never
constructed, named as a path, or read. Every run directory here is built in a
fresh temp dir from literal row dicts.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/t/rev110/src")
sys.path.insert(0, str(SRC))

from t2pw.bench.acceptance import (  # noqa: E402
    NEGATIVE_CONTROL_NOT_AWARDED,
    NEGATIVE_CONTROL_PASS,
    score_run,
)
from t2pw.bench.goldset import load_gold_set  # noqa: E402

GOLD = load_gold_set()

NEG = "PMC13231680"        # is_negative_control True   -- arm 1
CTX = "PMC12180156"        # context_only, min 0, neg False -- arm 2
POS = "PMC12096016"        # core, min_connected 4      -- positive control

DIAG = [{"name": "extraction_diagnostics.json", "bytes": 812}]


def payload(n):
    rows = [
        {"name": f"step {i}", "inputs": [f"m{i}"], "outputs": [f"m{i+1}"],
         "evidence": "quoted"}
        for i in range(n)
    ]
    names = {x for r in rows for x in (*r["inputs"], *r["outputs"])}
    return {
        "entities": {"compounds": [{"name": x} for x in sorted(names)],
                     "proteins": [], "protein_complexes": []},
        "processes": {"reactions": rows, "transports": [], "interactions": []},
    }


def declined(pid=NEG, mode="strict", **over):
    """The clean DECLINE. Every adversarial row below is this, minimally mutated."""
    row = {
        "paper_id": pid, "slug": pid, "mode": mode,
        "status": "fail", "stage": "stage1", "failure_kind": "no_reactions",
        "message": "extraction produced no reactions: nothing lipid-A-related is present",
        "issue_codes": [], "counts": {"reactions": 0, "transports": 0},
        "files": list(DIAG),
    }
    row.update(over)
    return {k: v for k, v in row.items() if v is not _ABSENT}


class _Absent:
    def __repr__(self):
        return "<absent>"


_ABSENT = _Absent()


def score(rows, payloads=None):
    tmp = Path(tempfile.mkdtemp(prefix="rev110_"))
    try:
        run = tmp / "2026-01-01_0000"
        (run / "papers").mkdir(parents=True)
        for (pid, mode), pl in (payloads or {}).items():
            d = run / "papers" / pid / mode
            d.mkdir(parents=True, exist_ok=True)
            (d / "merged_payload.json").write_text(json.dumps(pl), encoding="utf-8")
        (run / "manifest.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return score_run(run, GOLD)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def record(row, pid=None, mode="strict", payloads=None):
    rows = [row] if row is not None else []
    rep = score(rows, payloads)
    pid = pid or (row or {}).get("paper_id") or NEG
    paper = next(p for p in rep.papers if p.paper_id == pid)
    return paper.leg(mode).negative_control


def verdict(rec):
    if rec is None:
        return "NONE (rule does not apply)", []
    return rec["status"], list(rec["blocked_by"])


# ---------------------------------------------------------------------------
# 0. HARNESS SELF-CHECK -- refuse to report anything if these two do not hold.
# ---------------------------------------------------------------------------
print("=" * 78)
print("0. HARNESS SELF-CHECK  (a probe that cannot reach the guard proves nothing)")
print("=" * 78)

kp, kp_blocked = verdict(record(declined()))
kn_row = declined(status="timeout", stage="unknown", failure_kind="timeout",
                  termination_reason="budget_exhausted", operational_failure=True,
                  counts={}, files=[])
kn, kn_blocked = verdict(record(kn_row))
print(f"  known-POSITIVE (clean decline)   -> {kp}   blocked={kp_blocked}")
print(f"  known-NEGATIVE (outer-kill row)  -> {kn}   blocked={kn_blocked}")
ok = kp == NEGATIVE_CONTROL_PASS and kn == NEGATIVE_CONTROL_NOT_AWARDED
print(f"  HARNESS REACHES THE GUARD IN BOTH DIRECTIONS : {ok}")
if not ok:
    print("\n!! HARNESS INVALID -- no verdict below may be trusted. STOP.")
    raise SystemExit(2)

results = []


def case(label, rec, must_pass, note=""):
    status, blocked = verdict(rec)
    if must_pass:
        good = status == NEGATIVE_CONTROL_PASS
    else:
        good = status != NEGATIVE_CONTROL_PASS
    results.append((label, status, blocked, must_pass, good, note))
    flag = "ok " if good else "!! "
    want = "MUST PASS" if must_pass else "MUST REFUSE"
    print(f"  {flag}{label:<52} {want:<11} -> {status}")
    if blocked:
        print(f"      blocked_by = {blocked}")
    if note:
        print(f"      {note}")


# ---------------------------------------------------------------------------
# 1. CONDITION 3 ISOLATION -- one casualty signal at a time, other readings
#    clean, artifacts preserved, reason stated. Proves the AND, not the OR.
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("1. CONDITION-3 ISOLATION -- exactly ONE reading says casualty")
print("   (all others clean, >=1 artifact preserved, reason stated)")
print("=" * 78)

case("only operational_failure=True",
     record(declined(operational_failure=True)), False)
case("only termination_reason=budget_exhausted",
     record(declined(termination_reason="budget_exhausted")), False)
case("only termination_reason=operation_timeout",
     record(declined(termination_reason="operation_timeout")), False)
case("only status=timeout",
     record(declined(status="timeout")), False)
case("only status=error",
     record(declined(status="error")), False)
case("only failure_kind=timeout",
     record(declined(failure_kind="timeout")), False)
case("only failure_kind=crash",
     record(declined(failure_kind="crash")), False)
case("only failure_kind=network",
     record(declined(failure_kind="network")), False)
case("only failure_kind=llm",
     record(declined(failure_kind="llm")), False)

# ---------------------------------------------------------------------------
# 2. THE LEAD'S SHARPEST CASE -- a CASUALTY that PRESERVED ARTIFACTS and
#    STATED A REASON. The artifact condition alone must not be what saves us.
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("2. CASUALTIES THAT PRESERVED ARTIFACTS AND STATED A REASON")
print("=" * 78)

case("timeout, files preserved, reason stated",
     record(declined(status="timeout", failure_kind="timeout",
                     termination_reason="budget_exhausted",
                     operational_failure=True,
                     message="killed at the wall-clock ceiling (budget_exhausted)",
                     files=list(DIAG))), False,
     "the Lead's #2: >=1 artifact is NOT what discriminates here")
case("crash, files preserved, reason stated",
     record(declined(status="error", stage="unknown", failure_kind="crash",
                     message="the child process exited with code 3221225477",
                     files=list(DIAG))), False)
case("B15: empty + reason string + timed out",
     record(declined(status="timeout", failure_kind="timeout",
                     termination_reason="operation_timeout",
                     operational_failure=True,
                     message="a reason string that is nonetheless a timeout",
                     counts={"reactions": 0}, files=list(DIAG))), False,
     "re-derived independently of the author's B15 test")
case("F-148 outer kill: budget_exhausted only, stage=unknown",
     record(declined(stage="unknown", termination_reason="budget_exhausted",
                     message="the child was still running after 1800s and was killed",
                     files=list(DIAG))), False,
     "F-148's real shape, but with the runner's status/kind stripped off")

# ---------------------------------------------------------------------------
# 3. OLD ROW SHAPE -- `operational_failure` and `termination_reason` ABSENT
#    rather than False. Author claims absence is treated as indeterminate.
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("3. OLD ROW SHAPE -- operational_failure / termination_reason ABSENT")
print("=" * 78)

old_timeout = {"paper_id": NEG, "slug": NEG, "mode": "strict", "status": "timeout",
               "stage": "unknown", "failure_kind": "timeout",
               "message": "killed at the ceiling", "issue_codes": [],
               "counts": {}, "files": []}
case("old-shape TIMEOUT (no termination/operational keys)",
     record(old_timeout), False,
     "caught by status/kind/boundary, not by the two absent keys")

old_declined = {"paper_id": NEG, "slug": NEG, "mode": "strict", "status": "fail",
                "stage": "stage1", "failure_kind": "no_reactions",
                "message": "extraction produced no reactions",
                "issue_codes": [], "counts": {"reactions": 0},
                "files": list(DIAG)}
case("old-shape DECLINE (no termination/operational keys)",
     record(old_declined), True,
     "TESTS THE AUTHOR'S CLAIM that absent keys => indeterminate")

bare = {"paper_id": NEG, "slug": NEG, "mode": "strict", "status": "fail",
        "counts": {}, "files": []}
case("bare row: no kind, no message, no files",
     record(bare), False)

# ---------------------------------------------------------------------------
# 4. THE ATTACK -- driver._kind relabels. `contract` is the catch-all bucket
#    for "there were issue codes", checked BEFORE the network/llm markers.
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("4. ATTACK -- driver._kind:1265  `if contract_signal or issue_codes:")
print("   return KIND_CONTRACT`  is tested BEFORE the network/llm markers")
print("=" * 78)

case("provider failure relabelled contract by issue codes",
     record(declined(failure_kind="contract",
                     issue_codes=["gate.pwml_required_field"],
                     stage="stage2",
                     message="connection reset by peer while calling the provider",
                     files=list(DIAG))), False,
     "a NETWORK casualty that driver._kind files as `contract`")

case("failure_kind=unknown WITH an issue code",
     record(declined(failure_kind="unknown", stage="stage2",
                     issue_codes=["some.code"],
                     message="no research report was produced and no reason was given",
                     files=list(DIAG))), False,
     "driver KIND_UNKNOWN == 'we could not tell'; card says default to FAIL")

case("failure_kind=unknown, NO issue code",
     record(declined(failure_kind="unknown", stage="stage2", issue_codes=[],
                     message="no reason was given", files=list(DIAG))), False)

case("failure_kind=ambiguous_review_scope with codes",
     record(declined(failure_kind="ambiguous_review_scope", stage="stage2",
                     issue_codes=["scope.conflict"],
                     message="scope ambiguity between two pathways",
                     files=list(DIAG))), False,
     "a scope conflict is a declared stop -- but is it a DECLINE?")

# ---------------------------------------------------------------------------
# 5. THE OTHER RULING CONDITIONS
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("5. CONDITIONS 1 AND 2, AND THE POPULATIONS THAT MUST NOT PASS")
print("=" * 78)

case("B4: empty, artifacts preserved, NO message",
     record(declined(message="")), False)
case("empty, message stated, NO classification at all",
     record(declined(failure_kind="", issue_codes=[])), False)
case("missing artifact: clean decline but files=[]",
     record(declined(files=[])), False)
case("not attempted: no manifest row at all",
     record(None, pid=NEG), False)
case("reactions in the row's own counts",
     record(declined(counts={"reactions": 2})), False)
case("transports in the row's own counts",
     record(declined(counts={"reactions": 0, "transports": 3})), False)
case("a PWML artifact was produced",
     record(declined(pwml_artifact="pathway.pwml")), False)

# ---------------------------------------------------------------------------
# 6. B16 / B17 -- over-retention, and BOTH arms of _empty_is_correct
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("6. B16 over-retention, B17 both arms, and the positive control")
print("=" * 78)

case("B16: negative control that EXTRACTED reactions anyway",
     record(declined(status="pass", failure_kind="", message="exported 1 reaction",
                     counts={"reactions": 1}),
            payloads={(NEG, "strict"): payload(1)}), False,
     "over-retention must not be absolved by the new status")

case("B17 arm 2: context_only min_connected==0 declines",
     record(declined(pid=CTX), pid=CTX), True)
case("B17 arm 2: context_only that timed out",
     record(declined(pid=CTX, status="timeout", failure_kind="timeout",
                     termination_reason="budget_exhausted",
                     operational_failure=True, counts={}, files=[]), pid=CTX), False)

pos = record(declined(pid=POS), pid=POS)
print(f"  {'ok ' if pos is None else '!! '}"
      f"{'positive control gets NO record at all':<52} MUST BE NONE -> {verdict(pos)[0]}")
results.append(("positive control gets no record", verdict(pos)[0], [], False,
                pos is None, "rule must not touch a relevant paper"))

# ---------------------------------------------------------------------------
# 7. B5 -- raw outcome and rejection reason preserved beside the adjusted view
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("7. B5 -- RAW PRESERVED BESIDE THE ADJUSTED VIEW")
print("=" * 78)

raw_row = declined(status="timeout", failure_kind="timeout",
                   termination_reason="budget_exhausted", operational_failure=True,
                   message="killed at the wall-clock ceiling", counts={}, files=[])
rep = score([raw_row])
leg = next(p for p in rep.papers if p.paper_id == NEG).leg("strict")
ser = leg.to_dict()
print(f"  leg.status (raw, untouched)         : {leg.status!r}")
print(f"  serialized top-level 'status'       : {ser.get('status')!r}")
print(f"  serialized negative_control.status  : {ser['negative_control']['status']!r}")
print(f"  serialized negative_control.raw     : {json.dumps(ser['negative_control']['raw'])}")
print(f"  rejection_reason preserved verbatim : "
      f"{ser['negative_control']['rejection_reason'] == raw_row['message']}")
print(f"  leg.passed (the scored verdict)     : {leg.passed}")

# ---------------------------------------------------------------------------
# 8. SUMMARY
# ---------------------------------------------------------------------------
print()
print("=" * 78)
bad = [r for r in results if not r[4]]
print(f"CASES: {len(results)}   UNEXPECTED: {len(bad)}")
for label, status, blocked, must_pass, _good, note in bad:
    want = "MUST PASS" if must_pass else "MUST REFUSE"
    print(f"  !! {label}  ({want}) -> {status}  blocked={blocked}")
    if note:
        print(f"     {note}")
print("=" * 78)

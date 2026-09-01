"""REV-110 round 1 -- rows the author has NOT constructed.

Round 0 was caught by building rows the author had not. This does the same
against the corrected `declared`, which now has exactly two routes:

    declared = kind in ("no_reactions",) or termination == scientifically_unrecoverable

So the questions are: can a CASUALTY wear either label, and can the new
`classified` OR readmit anything the left arm excludes?

HARNESS SELF-CHECK FIRST -- a probe that cannot reach the guard proves nothing.
No committed run is opened; T-107 is never named or read.
"""
from __future__ import annotations
import json, shutil, sys, tempfile
from pathlib import Path

SRC = Path(sys.argv[1]); sys.path.insert(0, str(SRC))
from t2pw.bench.acceptance import (NEGATIVE_CONTROL_NOT_AWARDED, NEGATIVE_CONTROL_PASS, score_run)
from t2pw.bench.goldset import load_gold_set
from t2pw.pipeline.deadline import TERMINATION_REASONS
from t2pw.batch import driver

GOLD = load_gold_set()
NEG, CTX = "PMC13231680", "PMC12180156"
DIAG = [{"name": "extraction_diagnostics.json", "bytes": 812}]

def declined(pid=NEG, mode="strict", **over):
    row = {"paper_id": pid, "slug": pid, "mode": mode, "status": "fail", "stage": "stage1",
           "failure_kind": "no_reactions",
           "message": "extraction produced no reactions: nothing lipid-A-related is present",
           "issue_codes": [], "counts": {"reactions": 0, "transports": 0},
           "files": list(DIAG)}
    row.update(over); return row

def record(row, pid=None, mode="strict"):
    tmp = Path(tempfile.mkdtemp(prefix="rev110r1_"))
    try:
        run = tmp / "2026-01-01_0000"; (run / "papers").mkdir(parents=True)
        rows = [row] if row is not None else []
        (run / "manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
        rep = score_run(run, GOLD)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    paper = next(p for p in rep.papers if p.paper_id == (pid or row["paper_id"]))
    return paper.leg(mode).negative_control

def verdict(rec):
    return ("NONE", []) if rec is None else (rec["status"], list(rec["blocked_by"]))

print("="*78); print("0. HARNESS SELF-CHECK"); print("="*78)
kp,_ = verdict(record(declined()))
kn,_ = verdict(record(declined(status="timeout", failure_kind="timeout",
                              termination_reason="budget_exhausted",
                              operational_failure=True, counts={}, files=[])))
print(f"  known-POSITIVE -> {kp}\n  known-NEGATIVE -> {kn}")
ok = kp == NEGATIVE_CONTROL_PASS and kn == NEGATIVE_CONTROL_NOT_AWARDED
print(f"  REACHES THE GUARD BOTH WAYS : {ok}")
if not ok: print("!! HARNESS INVALID"); raise SystemExit(2)

res = []
def case(label, rec, must_pass, note=""):
    st, bl = verdict(rec)
    good = (st == NEGATIVE_CONTROL_PASS) if must_pass else (st != NEGATIVE_CONTROL_PASS)
    res.append((label, st, bl, must_pass, good, note))
    print(f"  {'ok ' if good else '!! '}{label:<54} {'MUST PASS' if must_pass else 'MUST REFUSE':<11} -> {st}")
    if bl: print(f"      blocked_by = {bl}")
    if note: print(f"      {note}")

print(); print("="*78)
print("1. THE SECOND ROUTE -- termination == scientifically_unrecoverable")
print("="*78)
case("scientifically_unrecoverable, kind unknown, clean",
     record(declined(failure_kind="unknown", stage="stage2",
                     termination_reason="scientifically_unrecoverable",
                     message="the source does not support a defensible pathway")), True,
     "D-005 scientific statement -- the ruling's own condition 2. Legitimate.")
case("scientifically_unrecoverable BUT it timed out",
     record(declined(failure_kind="timeout", status="timeout",
                     termination_reason="scientifically_unrecoverable",
                     message="claims a scientific verdict, but status says killed")), False,
     "condition 3 must still veto a scientific-sounding label")
case("scientifically_unrecoverable BUT operational_failure=True",
     record(declined(termination_reason="scientifically_unrecoverable",
                     operational_failure=True,
                     message="the source does not support a defensible pathway")), False)
case("scientifically_unrecoverable BUT no artifacts",
     record(declined(termination_reason="scientifically_unrecoverable", files=[],
                     message="the source does not support a defensible pathway")), False)

print(); print("="*78)
print("2. THE `classified` OR -- can the RIGHT arm readmit what the LEFT excludes?")
print("   classified = kind not in ('','unknown') or termination in TERMINATION_REASONS")
print("="*78)
for reason in TERMINATION_REASONS:
    must = reason == "scientifically_unrecoverable"
    case(f"kind=unknown + termination={reason}",
         record(declined(failure_kind="unknown", stage="stage2", termination_reason=reason,
                         message="a stated message with an unknown kind")), must,
         "only the scientific reason is a decline" if must else "")

print(); print("="*78)
print("3. driver._classify -- is `no_reactions` relabellable the way `contract` was?")
print("="*78)
BOTH = "the provider connection was refused and the run produced no reactions"
for label, codes in (("no codes", []), ("one code", ["x.y"])):
    kind = driver._classify(text=BOTH, issue_codes=list(codes), contract_signal=False,
                            ambiguous=False, no_reactions=False, crashed=False)
    print(f"  network+no-reaction wording, {label:<9} -> failure_kind={kind!r}")
case("provider text that ALSO says 'no reactions', no codes",
     record(declined(message=BOTH, stage="stage2")), False,
     "_NO_REACTION_MARKERS are tested BEFORE _NETWORK_MARKERS -- F-159's shape")

print(); print("="*78)
print("4. THE ACQUISITION FAILURE -- driver.py:2217 labels it no_reactions")
print("="*78)
ACQ = "paper has no full text, so there was nothing to extract"
case("no full text, artifacts NOT preserved (the shipped shape)",
     record(declined(message=ACQ, files=[], counts={})), False,
     "refused -- but by the ARTIFACT guard, not by understanding it")
case("no full text, artifacts preserved (counterfactual)",
     record(declined(message=ACQ)), False,
     "if that path ever preserved a file, would it earn the status?")

print(); print("="*78)
print("5. LABEL HYGIENE -- case, whitespace, and codes as mere evidence")
print("="*78)
case("failure_kind='No_Reactions' (mixed case)", record(declined(failure_kind="No_Reactions")), True)
case("termination='  scientifically_unrecoverable  '",
     record(declined(failure_kind="unknown",
                     termination_reason="  scientifically_unrecoverable  ",
                     message="padded reason")), True)
case("legitimate decline that ALSO carries issue codes",
     record(declined(issue_codes=["gate.something", "other.code"])), True,
     "codes must now be evidence only -- neither granting nor blocking")
case("kind='' with a message and files", record(declined(failure_kind="")), False)

print(); print("="*78)
bad = [r for r in res if not r[4]]
print(f"CASES: {len(res)}   UNEXPECTED: {len(bad)}")
for label, st, bl, mp, _g, note in bad:
    print(f"  !! {label} ({'MUST PASS' if mp else 'MUST REFUSE'}) -> {st} blocked={bl}")
    if note: print(f"     {note}")
print("="*78)

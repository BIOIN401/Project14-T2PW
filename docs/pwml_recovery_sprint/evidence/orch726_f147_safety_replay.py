"""ORCH-726: the F-147 safety replay. Would lifting the stale-report block let bad content out?

READ-ONLY and EVALUATION-ONLY. Reruns no pipeline leg, takes no LLM draw, writes no
cache, edits no production code. It loads ARCHIVED canonical payloads and pushes them
through the CURRENT HEAD deterministic validators -- the same functions the production
pre-export path calls.

The question
------------
`F-147` (FINDINGS.md:6150) is registered HIGH and deliberately NOT chartered. The reason
is a safety precondition, quoted verbatim:

    "If the driver stopped honouring superseded audit_round reports, both would PASS, and
     both would export content their own gold forbids"

  * PMC12452463/strict -- `enterobactin synthase complex` (forbidden_identifier),
    `RyhB inhibits EntC`/`EntF` (a small RNA, never an enzyme), an `Enterobactin
    secretion` transport the gold says is never described, and an Unknown-backed protein
    where `unknown_backed_proteins_acceptable: false`.
  * PMC12180156/strict -- the `ferrochelatase reaction` built on `protoporphyrin IX`,
    the gold's own "HALLUCINATION TEST: zero occurrences in the entire 67,304-character
    file".

That precondition was measured against the code of 2026-08-29. `F-179` (the reaction
support rule) and `C-118` have landed since. This probe asks whether the CURRENT gates
refuse that content on the CURRENT archived payloads.

What is replayed
----------------
For every leg, at HEAD, in the order production runs them:

  1. ``process_normalizer.run_strict_post_normalization_gates``  -- the stage-3 gate
  2. ``stage_contracts.validate_post_normalization``             -- the contract
  3. ``stage_contracts.validate_pre_export``                     -- THE FINAL SEMANTIC
     GATE, which embeds F-179 at ``stage_contracts.py:364-380`` and whose inner
     ``pwml_contract_report.ok`` is what ``run_pwml_export`` actually decides on
  4. ``reaction_support.evaluate_reaction_support`` / ``reaction_support_issue``
     -- F-179 alone, itemised per row

and then simulates ``batch/driver.py::_blocking_reports`` over the ARCHIVED
``contract_reports.json`` twice: once as the driver does it today, and once with the
``phase: audit_round`` snapshot excluded. The difference between those two is exactly
the F-147 fix, evaluated with no code change.

WHAT THIS CANNOT ANSWER, and it must not be read as answering it
----------------------------------------------------------------
The archived development payloads were produced by PRE-C-118 code (C-118 merged
`a5ffdbeb`, 2026-09-06; the newest archive of these two legs is 2026-09-02). C-118
APPENDS a relation template, so at HEAD these papers could admit MORE reactions than the
archive holds. This probe therefore answers *"do the current gates refuse the archived
content"* and NOT *"what would the current pipeline produce for these papers"*. The
second question needs a fresh run, which the charter forbids and which this probe does
not substitute for.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: (paper, mode, archived run, cohort, why it is here)
LEGS: Tuple[Tuple[str, str, str, str, str], ...] = (
    # --- the two cases that made F-147 unsafe ---
    ("PMC12452463", "strict", "runs_verify/2026-09-02_2052", "development",
     "F-147 case A: enterobactin synthase complex / RyhB / secretion / Unknown-backed"),
    ("PMC12180156", "strict", "runs_verify/2026-09-02_2052", "development",
     "F-147 case B: ferrochelatase reaction on protoporphyrin IX (gold hallucination test)"),
    # --- the same two, at the run F-147 was actually measured on ---
    ("PMC12452463", "strict", "runs_verify/2026-08-28_1816", "development(T-107 archive)",
     "the payload F-147 was registered from"),
    ("PMC12180156", "strict", "runs_verify/2026-08-28_1816", "development(T-107 archive)",
     "the payload F-147 was registered from"),
    # --- the three substantive unseen legs F-147 currently blocks ---
    ("PMC7232280", "strict", "runs_verify/2026-09-06_1425", "unseen",
     "Moco biosynthesis, Neurospora crassa"),
    ("PMC8510960", "strict", "runs_verify/2026-09-06_1425", "unseen",
     "MIA biosynthesis, Catharanthus roseus"),
    ("PMC12376012", "strict", "runs_verify/2026-09-06_1425", "unseen",
     "sphingolipid metabolism, Homo sapiens"),
    # --- controls ---
    ("PMC12071552", "strict", "runs_verify/2026-09-06_1425", "unseen(control)",
     "the pilot's only PWML -- must stay exportable"),
    ("PMC11172790", "strict", "runs_verify/2026-09-06_1425", "unseen(thin)",
     "0 reactions -- reported, not counted as a recovery"),
)

#: Tokens whose PRESENCE the F-147 precondition names. Reported, never used as a gate:
#: the production rule decides, this is only so a reader can see what is in the payload.
FORBIDDEN_TOKENS: Dict[str, Tuple[str, ...]] = {
    "PMC12452463": ("enterobactin synthase complex", "RyhB", "enterobactin secretion"),
    "PMC12180156": ("protoporphyrin", "ferrochelatase"),
}

#: Two error codes ``validate_pre_export`` raises on a bare canonical payload that the
#: PRODUCTION path never sees, because ``run_pwml_export`` attaches pathway metadata
#: before calling the gate. Calibrated, not assumed: the real
#: ``pwml_required_field_gate_report.json`` of BOTH legs that actually exported
#: (`2026-09-06_1425/PMC12071552/strict`, `2026-09-02_2052/PMC12180156/strict`) records
#: ``ok: true, errors: 0`` with neither code present. Counting them would mark the
#: control leg -- which demonstrably shipped a PWML -- as blocked, so they are instrument
#: noise and are subtracted. Every other pre-export error is kept.
EXPORT_TIME_METADATA_CODES = ("pathway_missing_name", "pathway_missing_subject")

AUDIT_ROUND = "audit_round"
BLOCKING_SUFFIX = "_contract_report"
RUNTIME_SCHEMA_SUFFIX = "_runtime_schema_report"


def load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def issue_text(issue: Any, width: int = 150) -> str:
    if isinstance(issue, str):
        return issue[:width]
    if isinstance(issue, dict):
        where = issue.get("pointer") or issue.get("path") or ""
        what = issue.get("code") or issue.get("reason") or issue.get("message") or ""
        extra = issue.get("message") if issue.get("code") else ""
        return f"{where} {what} {extra}".strip()[:width]
    return str(issue)[:width]


def simulate_blocking_reports(reports: Dict[str, Any], *, drop_audit_round: bool) -> Tuple[int, List[str]]:
    """Reimplements ``driver._blocking_reports`` + ``_collect_issue_codes``' error count.

    ``drop_audit_round=False`` is today's behaviour. ``drop_audit_round=True`` is the
    proposed fix, evaluated without touching the driver.
    """
    total = 0
    names: List[str] = []
    for key, value in (reports or {}).items():
        if not isinstance(value, dict) or not value:
            continue
        if key.endswith(RUNTIME_SCHEMA_SUFFIX) or not key.endswith(BLOCKING_SUFFIX):
            continue
        if drop_audit_round and str(value.get("phase") or "") == AUDIT_ROUND:
            continue
        errors = value.get("errors") or []
        if errors:
            total += len(errors)
            names.append(f"{key}(phase={value.get('phase')}, {len(errors)} errors)")
    return total, names


def provenance_completeness(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Row-level provenance, using the same fields F-179 reads."""
    processes = payload.get("processes") or {}
    rows = [r for r in (processes.get("reactions") or []) if isinstance(r, dict)]
    detail: List[Dict[str, Any]] = []
    for row in rows:
        lineage = row.get("provenance_lineage") or []
        stages = [
            str((entry or {}).get("stage") or "")
            for entry in lineage
            if isinstance(entry, dict)
        ]
        origins = [
            str((entry or {}).get("origin") or "")
            for entry in lineage
            if isinstance(entry, dict)
        ]
        detail.append(
            {
                "name": row.get("name"),
                "lineage_entries": len(lineage),
                "stages": stages,
                "origins": origins,
                "has_evidence": bool(row.get("evidence") or row.get("source_quote")),
                "provenance": row.get("provenance"),
                "rag_added": bool(row.get("rag_added")),
            }
        )
    return {
        "reactions": len(rows),
        "rows_with_no_lineage": sum(1 for d in detail if d["lineage_entries"] == 0),
        "detail": detail,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    root = args[0] if args else "."
    sys.path.insert(0, os.path.join(root, "src"))

    from t2pw.pipeline.process_normalizer import (
        GateValidationError,
        run_strict_post_normalization_gates,
    )
    from t2pw.pipeline.stage_contracts import (
        StageContractError,
        validate_post_normalization,
        validate_pre_export,
    )
    from t2pw.pipeline.reaction_support import (
        evaluate_reaction_support,
        reaction_support_class,
        reaction_support_issue,
    )

    print("=" * 100)
    print("ORCH-726 -- F-147 SAFETY REPLAY. Archived payloads, CURRENT HEAD validators.")
    print("=" * 100)
    print(f"repo root : {root}")
    print("validators: run_strict_post_normalization_gates | validate_post_normalization |")
    print("            validate_pre_export (embeds F-179) | reaction_support (F-179 itemised)")
    print()

    verdicts: List[Dict[str, Any]] = []

    for paper, mode, run, cohort, note in LEGS:
        base = os.path.join(root, run, "papers", paper, mode)
        payload = load(os.path.join(base, "final_mapped.json"))
        print("#" * 100)
        print(f"{paper}/{mode}   [{cohort}]   {run}")
        print(f"   {note}")
        if payload is None:
            print("   NO ARCHIVED final_mapped.json -- cannot replay this leg")
            verdicts.append({"leg": f"{paper}/{mode}@{run}", "cohort": cohort, "verdict": "no_payload"})
            print()
            continue

        # ---- what the leg actually did, from the archive -------------------
        terminal = load(os.path.join(base, "LEG_TERMINAL.json")) or {}
        recorded = (terminal.get("terminal_state_before_cleanup") or {}).get("status", "?")
        shipped_pwml = [f for f in os.listdir(base) if f.endswith(".pwml")]
        print(f"   archived outcome: status={recorded}  pwml={shipped_pwml or 'none'}")

        # ---- 1. stage-3 gate ------------------------------------------------
        try:
            gate_details = run_strict_post_normalization_gates(
                payload, enforce_all_proteins_connected=True
            )
            gate_report = {"ok": True, **gate_details}
        except GateValidationError as exc:
            gate_report = {"ok": False, **(getattr(exc, "details", None) or {})}
        gate_errors = gate_report.get("errors") or []
        print(f"   [1] stage-3 gate            ok={gate_report.get('ok')} errors={len(gate_errors)}")
        for err in gate_errors[:6]:
            print(f"          - {issue_text(err)}")

        # ---- 2. post-normalization contract ---------------------------------
        try:
            contract = validate_post_normalization(payload, gate_report)
            contract_ok, contract_errors = contract.get("ok"), contract.get("errors") or []
        except StageContractError as exc:
            contract = getattr(exc, "report", None) or {}
            contract_ok, contract_errors = False, contract.get("errors") or []
        print(f"   [2] post_normalization      ok={contract_ok} errors={len(contract_errors)}")
        for err in contract_errors[:6]:
            print(f"          - {issue_text(err)}")

        # ---- 3. THE FINAL SEMANTIC GATE (embeds F-179) ----------------------
        pre_export_ok: Optional[bool] = None
        inner_ok: Optional[bool] = None
        inner_errors: List[Any] = []
        try:
            pre_export = validate_pre_export(payload, strict_db=False)
        except StageContractError as exc:
            pre_export = getattr(exc, "report", None) or {}
        except Exception as exc:  # noqa: BLE001 -- a replay reports, it does not raise
            pre_export = {"ok": None, "_replay_error": f"{type(exc).__name__}: {exc}"}
        if pre_export.get("_replay_error"):
            print(f"   [3] pre_export              COULD NOT REPLAY: {pre_export['_replay_error']}")
        else:
            pre_export_ok = pre_export.get("ok")
            inner = pre_export.get("pwml_contract_report") or {}
            inner_ok = inner.get("ok")
            inner_errors = inner.get("errors") or []
            substantive = [
                e for e in inner_errors
                if not (isinstance(e, dict) and e.get("code") in EXPORT_TIME_METADATA_CODES)
            ]
            artifacts_n = len(inner_errors) - len(substantive)
            inner_ok = not substantive
            print(
                f"   [3] pre_export (FINAL)      raw_ok={inner.get('ok')}  "
                f"errors={len(inner_errors)} (of which {artifacts_n} export-time metadata artifacts)  "
                f"-> SUBSTANTIVE ok={inner_ok}"
            )
            print("       ^ pwml_contract_report.ok is what run_pwml_export decides on")
            for err in inner_errors[:8]:
                tag = "  [instrument artifact]" if (
                    isinstance(err, dict) and err.get("code") in EXPORT_TIME_METADATA_CODES
                ) else ""
                print(f"          - {issue_text(err)}{tag}")

        # ---- 4. F-179 itemised ----------------------------------------------
        support = evaluate_reaction_support(payload)
        support_issue = reaction_support_issue(payload)
        print(
            f"   [4] F-179 reaction_support  verdict={support.get('verdict') or support.get('status')}  "
            f"issue={'YES -> ' + str(support_issue.get('code')) if support_issue else 'none'}"
        )
        for key in ("counts", "summary", "supported", "unsupported", "indeterminate"):
            if key in support:
                print(f"          {key}: {json.dumps(support[key])[:180]}")
        rows = [r for r in ((payload.get("processes") or {}).get("reactions") or []) if isinstance(r, dict)]
        for row in rows:
            print(f"          row '{str(row.get('name'))[:44]:<46} -> support_class={reaction_support_class(row)}")

        # ---- 5. provenance completeness -------------------------------------
        prov = provenance_completeness(payload)
        print(
            f"   [5] provenance              reactions={prov['reactions']}  "
            f"rows with NO lineage={prov['rows_with_no_lineage']}"
        )
        for d in prov["detail"]:
            print(
                f"          {str(d['name'])[:42]:<44} lineage={d['lineage_entries']} "
                f"origins={d['origins']} evidence={d['has_evidence']} rag_added={d['rag_added']}"
            )

        # ---- 6. forbidden-token presence (reported, not a gate) --------------
        tokens = FORBIDDEN_TOKENS.get(paper)
        if tokens:
            blob = json.dumps(payload).casefold()
            print("   [6] F-147 precondition tokens in the archived FINAL payload:")
            for token in tokens:
                print(f"          {token!r:36} present={token.casefold() in blob}")

        # ---- 7. the F-147 simulation ----------------------------------------
        reports = load(os.path.join(base, "contract_reports.json")) or {}
        today, today_names = simulate_blocking_reports(reports, drop_audit_round=False)
        fixed, fixed_names = simulate_blocking_reports(reports, drop_audit_round=True)
        print(f"   [7] driver simulation       today: error_count={today}  {today_names}")
        print(f"                               with audit_round dropped: error_count={fixed}  {fixed_names}")

        would_export = (
            fixed == 0
            and gate_report.get("ok") is not False
            and inner_ok is not False
            and support_issue is None
        )
        blockers: List[str] = []
        if fixed:
            blockers.append(f"non-stale contract errors ({fixed})")
        if gate_report.get("ok") is False:
            blockers.append("stage-3 gate")
        if inner_ok is False:
            blockers.append("pre_export (substantive)")
        if support_issue is not None:
            blockers.append(f"F-179 {support_issue.get('code')}")
        print(
            f"   => IF ONLY THE STALE REPORT WERE DROPPED: "
            f"{'WOULD EXPORT' if would_export else 'STILL BLOCKED by ' + ', '.join(blockers)}"
        )
        print()

        verdicts.append(
            {
                "leg": f"{paper}/{mode}",
                "run": run,
                "cohort": cohort,
                "archived_status": recorded,
                "stage3_ok": gate_report.get("ok"),
                "post_norm_ok": contract_ok,
                "pre_export_inner_ok": inner_ok,
                "f179_issue": (support_issue or {}).get("code"),
                "reactions": prov["reactions"],
                "rows_no_lineage": prov["rows_with_no_lineage"],
                "stale_dropped_error_count": fixed,
                "would_export": would_export,
                "blockers": blockers,
            }
        )

    print("=" * 100)
    print("REPLAY SUMMARY")
    print("=" * 100)
    header = (
        f"{'leg':<22}{'cohort':<26}{'rx':>4}{'s3':>6}{'pre':>6}{'F179':>22}"
        f"{'stale-dropped':>15}{'  would export?'}"
    )
    print(header)
    for v in verdicts:
        if v.get("verdict") == "no_payload":
            print(f"{v['leg']:<22}{v['cohort']:<26}   -- no archived payload --")
            continue
        print(
            f"{v['leg']:<22}{v['cohort']:<26}{v['reactions']:>4}"
            f"{str(v['stage3_ok']):>6}{str(v['pre_export_inner_ok']):>6}"
            f"{str(v['f179_issue'] or 'clean'):>22}{v['stale_dropped_error_count']:>15}"
            f"  {'YES' if v['would_export'] else 'NO -- ' + ', '.join(v['blockers'])}"
        )
    print()
    dev = [v for v in verdicts if v.get("cohort", "").startswith("development") and v.get("verdict") != "no_payload"]
    unseen = [v for v in verdicts if v.get("cohort") == "unseen"]
    print(f"  development cases that WOULD export with the stale report dropped: "
          f"{sum(1 for v in dev if v['would_export'])} of {len(dev)}")
    print(f"  unseen substantive cases that WOULD export: "
          f"{sum(1 for v in unseen if v['would_export'])} of {len(unseen)}")
    print()
    print("  F-147's precondition is CLEARED only if the development count is 0 while the")
    print("  unseen count is positive. Any development case that would export means the")
    print("  precondition stands and the fix stays unsafe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

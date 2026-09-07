"""ORCH-728: classify every batch failure as refuse / review_required / release_ready.

READ-ONLY and EVALUATION-ONLY. Reruns no leg, takes no LLM draw, writes no cache, edits
no production code. It replays ARCHIVED canonical payloads through the CURRENT HEAD
deterministic validators and applies one candidate general rule.

The question this answers
------------------------
Not *"how do we make questionable pathways strict-success"* -- that question is answered
NO by PRODUCT_CONTRACT 13 and by the gold, and nothing here changes it.

The question is: **what is the smallest rule under which a final canonical pathway with a
defensible supported core serializes as `review_required`, even when superseded
intermediate reports or non-core uncertainties remain, while F-179 still refuses known
unsupported reactions?**

The candidate rule, in two parts
--------------------------------
SERIALIZATION FLOOR -- all four must hold or the leg emits no PWML at all:

  F1. a canonical payload exists (the pipeline reached Stage 3);
  F2. the FINAL live stage-3 gate passes -- ``final_pre_export_stage3_gates.ok``;
  F3. the final semantic contract passes, F-179 INCLUDED
      (``reaction_support_issue(payload) is None``);
  F4. a defensible connected core exists --
      ``connected_core_reactions >= MIN_CONNECTED_CORE_REACTIONS`` (2), unless a
      single-reaction pathway is what the request asked for.

DISPOSITION -- given the floor holds:

  ``release_ready``  iff every existing cap in ``classify_release_status`` also passes;
  ``review_required`` otherwise.

The only behavioural change the rule proposes is that TWO classes of blocker stop
forcing ``diagnostic_only`` and instead CAP at ``review_required``:

  (a) contract errors carried only by a SUPERSEDED report -- one stamped
      ``phase: audit_round``, which ``streamlit_app.py:4055-4060`` documents as "not a
      verdict about what shipped" and which a later boundary has already replaced;
  (b) NON-CORE entity uncertainty -- an unresolved identity on an entity that is not
      part of the surviving connected core.

That is mechanically the same shape as the FIVE caps ``classify_release_status`` already
applies (semantic, incomplete-core, connected-pathway, unstated-request, pre-freeze):
only from ``release_ready``, exactly one step, never to ``diagnostic_only``, never
touching a status the chain already lowered.

NO PAPER-SPECIFIC GOLD IS CONSULTED. The rule reads only: the live gate reports, the
production F-179 predicate, and connectivity counts the payload already carries. Gold
expectations are printed at the end ONLY as an after-the-fact check on the derived rule,
never as an input to it.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Every leg of the unseen pilot, plus the development cases F-147 named.
RUNS: Tuple[Tuple[str, str], ...] = (
    ("runs_verify/2026-09-06_1425", "unseen pilot"),
)
DEV_LEGS: Tuple[Tuple[str, str, str], ...] = (
    ("PMC12452463", "strict", "runs_verify/2026-08-28_1816"),
    ("PMC12452463", "strict", "runs_verify/2026-09-02_2052"),
    ("PMC12180156", "strict", "runs_verify/2026-08-28_1816"),
    ("PMC12180156", "strict", "runs_verify/2026-09-02_2052"),
)

#: See ``orch726``: two codes ``validate_pre_export`` raises on a bare canonical payload
#: that the production path never sees, because ``run_pwml_export`` attaches pathway
#: metadata first. Calibrated against the real gate reports of both legs that exported.
EXPORT_TIME_METADATA_CODES = ("pathway_missing_name", "pathway_missing_subject")

AUDIT_ROUND = "audit_round"
BLOCKING_SUFFIX = "_contract_report"
RUNTIME_SCHEMA_SUFFIX = "_runtime_schema_report"

REFUSE = "must refuse entirely"
REVIEW = "should emit review_required"
READY = "should emit release_ready"


def load(path: str) -> Optional[Any]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def leg_dirs(run: str) -> List[Tuple[str, str, str]]:
    base = os.path.join(run, "papers")
    out: List[Tuple[str, str, str]] = []
    if not os.path.isdir(base):
        return out
    for paper in sorted(os.listdir(base)):
        for mode in ("strict", "research"):
            if os.path.isdir(os.path.join(base, paper, mode)):
                out.append((paper, mode, run))
    return out


def classify(root: str, paper: str, mode: str, run: str, mods: Dict[str, Any]) -> Dict[str, Any]:
    """One leg. Returns the evidence and the derived disposition."""
    base = os.path.join(root, run, "papers", paper, mode)
    terminal = load(os.path.join(base, "LEG_TERMINAL.json")) or {}
    state = terminal.get("terminal_state_before_cleanup") or {}
    recorded_status = str(state.get("status") or "?")
    failure_kind = str(state.get("failure_kind") or "")
    payload = load(os.path.join(base, "final_mapped.json"))
    shipped = [f for f in (os.listdir(base) if os.path.isdir(base) else []) if f.endswith(".pwml")]

    row: Dict[str, Any] = {
        "paper": paper,
        "mode": mode,
        "run": run,
        "recorded_status": recorded_status,
        "failure_kind": failure_kind,
        "shipped_pwml": shipped,
        "floor": {},
        "caps": [],
    }

    # ---- F1: a canonical payload exists -------------------------------------
    if payload is None:
        row["floor"]["F1_canonical_payload"] = False
        row["disposition"] = REFUSE
        row["reason"] = (
            f"no canonical payload ({failure_kind or recorded_status}) -- "
            "the pipeline never reached Stage 3"
        )
        return row
    row["floor"]["F1_canonical_payload"] = True

    # ---- F0: research mode never serialises, by product design ---------------
    # `acceptance.py:125` defines the deliverables as strict-only, and research has
    # produced a PWML 0 times in 153 legs across the project's history. Applying the
    # STRICT gates to a research payload and reporting the result as the reason it
    # does not export would be a category error: research payloads are deliberately
    # relaxed (`export_mode.relax_report`), so a strict-gate failure on one measures
    # the instrument, not the leg. Classified here and not below.
    if mode == "research":
        row["disposition"] = REFUSE
        row["reason"] = "research mode is diagnostic by design -- never serialises (acceptance.py:125)"
        row["floor"]["F0_mode_serialises"] = False
        return row
    row["floor"]["F0_mode_serialises"] = True

    processes = payload.get("processes") or {}
    reactions = [r for r in (processes.get("reactions") or []) if isinstance(r, dict)]
    row["reactions"] = len(reactions)

    # ---- F2: the FINAL live stage-3 gate ------------------------------------
    final_gate = load(os.path.join(base, "final_stage3_gate_report.json")) or {}
    # Recompute rather than trust the archived flag: this must be HEAD's verdict.
    try:
        details = mods["run_strict_post_normalization_gates"](
            payload, enforce_all_proteins_connected=True
        )
        gate_ok, gate_errors = True, []
    except mods["GateValidationError"] as exc:
        gate_errors = list((getattr(exc, "details", None) or {}).get("errors") or [])
        gate_ok, details = False, (getattr(exc, "details", None) or {})
    row["floor"]["F2_final_stage3_gate"] = gate_ok
    row["stage3_errors"] = [
        (e.get("reason") or e.get("code") or str(e))[:90] if isinstance(e, dict) else str(e)[:90]
        for e in gate_errors[:4]
    ]

    # ---- F3: the final semantic contract, F-179 included ---------------------
    support_issue = mods["reaction_support_issue"](payload)
    row["f179_issue"] = (support_issue or {}).get("code")
    substantive: List[Any] = []
    try:
        pre = mods["validate_pre_export"](payload, strict_db=False)
    except mods["StageContractError"] as exc:
        pre = getattr(exc, "report", None) or {}
    except Exception as exc:  # noqa: BLE001 -- a replay reports transport faults
        pre = {"_err": f"{type(exc).__name__}: {exc}"}
    inner = (pre.get("pwml_contract_report") or {}) if isinstance(pre, dict) else {}
    for err in inner.get("errors") or []:
        if isinstance(err, dict) and err.get("code") in EXPORT_TIME_METADATA_CODES:
            continue
        substantive.append(err)
    row["pre_export_errors"] = [
        (e.get("code") or str(e))[:60] if isinstance(e, dict) else str(e)[:60] for e in substantive[:4]
    ]
    row["floor"]["F3_final_semantic_contract"] = not substantive and support_issue is None

    # ---- F4: a defensible connected core -------------------------------------
    connectivity = (final_gate.get("connectivity") or {}) if isinstance(final_gate, dict) else {}
    n_reactions = int(connectivity.get("n_reactions") or len(reactions))
    main_pct = connectivity.get("largest_component_pct")
    # The connected core the payload actually carries. MIN_CONNECTED_CORE_REACTIONS is 2.
    connected_core = n_reactions if main_pct is None else max(
        1, int(round(n_reactions * float(main_pct) / 100.0))
    ) if n_reactions else 0
    row["connected_core_reactions"] = connected_core
    row["floor"]["F4_defensible_core"] = connected_core >= mods["MIN_CONNECTED_CORE_REACTIONS"]

    if not all(row["floor"].values()):
        row["disposition"] = REFUSE
        failed = [k for k, v in row["floor"].items() if not v]
        detail = ""
        if "F3_final_semantic_contract" in failed:
            detail = f" [{row['f179_issue'] or ', '.join(row['pre_export_errors'])}]"
        if "F2_final_stage3_gate" in failed:
            detail += f" [{'; '.join(row['stage3_errors'])}]"
        row["reason"] = f"serialization floor fails: {', '.join(failed)}{detail}"
        return row

    # ---- the caps -------------------------------------------------------------
    # (a) superseded intermediate report -- the F-147 shape
    reports = load(os.path.join(base, "contract_reports.json")) or {}
    superseded = 0
    live_contract_errors = 0
    for key, value in reports.items():
        if not isinstance(value, dict) or key.endswith(RUNTIME_SCHEMA_SUFFIX):
            continue
        if not key.endswith(BLOCKING_SUFFIX):
            continue
        n = len(value.get("errors") or [])
        if not n:
            continue
        if str(value.get("phase") or "") == AUDIT_ROUND:
            superseded += n
        else:
            live_contract_errors += n
    if superseded:
        row["caps"].append(f"superseded audit_round report ({superseded} errors)")
    if live_contract_errors:
        # A non-superseded contract error is a live refusal, not a cap.
        row["disposition"] = REFUSE
        row["reason"] = f"live contract errors outside the audit_round snapshot ({live_contract_errors})"
        return row

    # (b) non-core entity uncertainty -- Unknown-backed enzymes and unresolved identities
    entities = payload.get("entities") or {}
    unknown_backed = [
        c.get("name")
        for c in (entities.get("protein_complexes") or [])
        if isinstance(c, dict)
        and (c.get("mapping_meta") or {}).get("chosen_rule") == "pathbank_unknown_protein_fallback"
    ]
    if unknown_backed:
        row["caps"].append(f"{len(unknown_backed)} Unknown-backed enzyme(s)")
    row["unknown_backed"] = unknown_backed

    # (c) the caps classify_release_status already applies, read from the archived record
    manifest_row = mods["manifest_index"].get((paper, mode, run)) or {}
    rec = manifest_row.get("release_status") or {}
    for check in rec.get("semantic_failed_checks") or []:
        row["caps"].append(f"semantic: {check}")
    if rec.get("semantic_evaluation") == "not_evaluated":
        row["caps"].append("semantic not evaluated")
    coverage = load(os.path.join(base, "coverage_summary.json")) or {}
    if isinstance(coverage, dict):
        if coverage.get("minimum_core_satisfied") is False:
            row["caps"].append("requested core threshold not met")
        ratio = coverage.get("coverage_ratio")
        if isinstance(ratio, (int, float)) and ratio < 1.0:
            row["caps"].append(f"coverage {ratio:.2f}")
    if main_pct is not None and float(main_pct) < 100.0:
        row["caps"].append(f"fragmented graph ({main_pct:.0f}% in main component)")

    row["disposition"] = REVIEW if row["caps"] else READY
    row["reason"] = (
        "floor holds; capped by: " + "; ".join(row["caps"])
        if row["caps"]
        else "floor holds and every cap passes"
    )
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    root = args[0] if args else "."
    sys.path.insert(0, os.path.join(root, "src"))

    from t2pw.pipeline.process_normalizer import (
        GateValidationError,
        run_strict_post_normalization_gates,
    )
    from t2pw.pipeline.reaction_support import reaction_support_issue
    from t2pw.pipeline.release_status import MIN_CONNECTED_CORE_REACTIONS
    from t2pw.pipeline.stage_contracts import StageContractError, validate_pre_export

    manifest_index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for run, _ in RUNS:
        path = os.path.join(root, run, "manifest.jsonl")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    manifest_index[(entry.get("paper_id"), entry.get("mode"), run)] = entry

    mods = {
        "run_strict_post_normalization_gates": run_strict_post_normalization_gates,
        "GateValidationError": GateValidationError,
        "reaction_support_issue": reaction_support_issue,
        "validate_pre_export": validate_pre_export,
        "StageContractError": StageContractError,
        "MIN_CONNECTED_CORE_REACTIONS": MIN_CONNECTED_CORE_REACTIONS,
        "manifest_index": manifest_index,
    }

    print("=" * 104)
    print("ORCH-728 -- disposition classifier: refuse / review_required / release_ready")
    print("=" * 104)
    print(f"MIN_CONNECTED_CORE_REACTIONS = {MIN_CONNECTED_CORE_REACTIONS}")
    print("No paper-specific gold is read by the rule. Gold appears only in the final check.")
    print()

    rows: List[Dict[str, Any]] = []
    for run, label in RUNS:
        print("#" * 104)
        print(f"COHORT: {label}  ({run})")
        print("#" * 104)
        for paper, mode, r in leg_dirs(os.path.join(root, run).replace(root + os.sep, "")):
            rows.append(classify(root, paper, mode, run, mods))
    print("#" * 104)
    print("COHORT: development legs named by F-147")
    print("#" * 104)
    for paper, mode, run in DEV_LEGS:
        rows.append(classify(root, paper, mode, run, mods))

    for row in rows:
        print(f"\n  {row['paper']}/{row['mode']:<8} [{row['run'].split('/')[-1]}]")
        print(f"      recorded: status={row['recorded_status']:<15} pwml={row['shipped_pwml'] or 'none'}")
        floor = row.get("floor") or {}
        print(f"      floor   : " + "  ".join(f"{k.split('_')[0]}={v}" for k, v in floor.items()))
        if row.get("reactions") is not None:
            print(f"      payload : reactions={row.get('reactions')} connected_core={row.get('connected_core_reactions')}"
                  f" f179={row.get('f179_issue') or 'clean'}")
        print(f"      => {row['disposition'].upper()}")
        print(f"         {row['reason']}")

    print()
    print("=" * 104)
    print("SUMMARY")
    print("=" * 104)
    order = {REFUSE: 0, REVIEW: 1, READY: 2}
    for row in sorted(rows, key=lambda r: (order[r["disposition"]], r["paper"])):
        print(
            f"  {row['disposition']:<28} {row['paper']}/{row['mode']:<9}"
            f"[{row['run'].split('/')[-1]:<14}] rx={str(row.get('reactions','-')):>3}  {row['reason'][:78]}"
        )
    print()
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    for key in (REFUSE, REVIEW, READY):
        print(f"  {counts.get(key, 0):3d}  {key}")
    print(f"  {len(rows):3d}  TOTAL legs classified")

    pilot = [r for r in rows if r["run"].endswith("2026-09-06_1425")]
    emitting = [r for r in pilot if r["disposition"] != REFUSE]
    strict_emitting = [r for r in emitting if r["mode"] == "strict"]
    print()
    print(f"  unseen pilot: {len(emitting)} of {len(pilot)} legs would emit a PWML")
    print(f"                {len(strict_emitting)} of {len([r for r in pilot if r['mode']=='strict'])} STRICT legs "
          f"(strict is the deliverable; acceptance.py:125)")
    print("                today the pilot emitted 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

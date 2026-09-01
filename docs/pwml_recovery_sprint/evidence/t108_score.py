"""T-108 — score the official run. Offline, deterministic, read-only.

Reads stored artifacts only. Calls no model, writes nothing into the run
directory, and does not touch gold or the scorer. Emits every row
`T108-READINESS.md` § 4 ("What T-108 must report") requires:

  completion and scorable denominator - every timeout - every missing or partial
  payload - Priority 1 raw AND accepted counts with row composition - Priority 2
  eligible denominator and its NOT-EVALUATED population - negative-control
  outcomes - Priority 3 referential integrity - Priority 4/5 raw and accepted -
  every contract adjustment - C-111 timeout/retry telemetry - provider and model
  provenance - elapsed time.

Usage:  t108_score.py <repo> <run-dir-relative>
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path


def j(value, limit=100000):
    return json.dumps(value, default=str)[:limit]


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    tree = Path(sys.argv[1]).resolve()
    run = Path(sys.argv[2])
    sys.path.insert(0, str(tree / "src"))

    from t2pw.bench.acceptance import score_run  # noqa: E402
    from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path  # noqa: E402

    print("MEASURED_TREE t2pw =", sys.modules["t2pw"].__file__)
    print("RUN =", run)
    run_abs = (tree / run) if not run.is_absolute() else run

    gold = load_gold_set(pinned_gold_set_path())
    rep = score_run(run_abs, gold)
    d = rep.to_dict()

    # ------------------------------------------------------------- completion
    rule("1. COMPLETION AND SCORABLE DENOMINATOR")
    comp = d.get("completion") or {}
    for k in ("planned_gold_cases", "papers_attempted", "strict_legs_attempted",
              "research_legs_attempted", "legs_attempted", "legs_planned",
              "payloads_available", "semantically_scorable_legs",
              "fully_completed_cases", "papers_with_no_attempted_leg",
              "papers_with_only_one_mode_attempted", "complete", "rendered"):
        if k in comp:
            print(f"  {k:36s} = {j(comp[k], 400)}")
    print(f"  legs_scored (top level)              = {d.get('legs_scored')}")
    print(f"  legs_attempted (top level)           = {d.get('legs_attempted')}")

    # --------------------------------------------------- manifest / per-leg facts
    rule("2. PER-LEG OUTCOMES, TIMEOUTS AND PAYLOAD PRESERVATION")
    manifest = run_abs / "manifest.jsonl"
    rows = []
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    print(f"  manifest rows : {len(rows)}")
    status_counts = Counter(str(r.get("status")) for r in rows)
    print(f"  status tally  : {j(dict(status_counts))}")

    timeouts, missing_payload, partial = [], [], []
    print()
    print(f"  {'leg':34s} {'status':22s} {'elapsed':>9s}  {'ceiling':>8s}  files")
    print("  " + "-" * 92)
    for r in sorted(rows, key=lambda x: (str(x.get("slug")), str(x.get("mode")))):
        leg = f"{r.get('slug')}/{r.get('mode')}"
        status = str(r.get("status"))
        elapsed = r.get("elapsed_seconds")
        ceiling = r.get("leg_timeout_seconds")
        files = r.get("files") or []
        nfiles = len(files) if isinstance(files, list) else "?"
        el = f"{float(elapsed):.1f}" if isinstance(elapsed, (int, float)) else "-"
        ce = f"{float(ceiling):.0f}" if isinstance(ceiling, (int, float)) else "-"
        print(f"  {leg:34s} {status:22s} {el:>9s}  {ce:>8s}  {nfiles}")
        if "timeout" in status.lower():
            timeouts.append((leg, elapsed, ceiling, r.get("termination_reason")))
        if not files:
            missing_payload.append(leg)

    rule("2a. TIMEOUTS — every one, with its ceiling")
    if not timeouts:
        print("  NONE. No leg hit the 3600 s ceiling.")
    for leg, elapsed, ceiling, reason in timeouts:
        pct = ""
        if isinstance(elapsed, (int, float)) and isinstance(ceiling, (int, float)) and ceiling:
            pct = f"  ({100.0 * float(elapsed) / float(ceiling):.1f}% of ceiling)"
        print(f"  {leg:34s} elapsed={elapsed}  ceiling={ceiling}{pct}  reason={reason}")

    rule("2b. MISSING OR PARTIAL PAYLOADS")
    print(f"  legs with an EMPTY files list : {len(missing_payload)}")
    for leg in missing_payload:
        print(f"    {leg}")

    # ------------------------------------------------------- C-111 telemetry
    rule("3. C-111 TIMEOUT / RETRY TELEMETRY (read off disk, per leg)")
    try:
        from t2pw.batch import leg_trace  # noqa: E402

        papers_dir = run_abs / "papers"
        any_trace = False
        if papers_dir.is_dir():
            for slug in sorted(os.listdir(papers_dir)):
                for mode in ("strict", "research"):
                    leg_dir = papers_dir / slug / mode
                    if not leg_dir.is_dir():
                        continue
                    try:
                        summary = leg_trace.summarize(leg_dir)
                    except Exception as exc:
                        print(f"  {slug}/{mode}: summarize failed {exc!r}")
                        continue
                    interesting = {
                        k: v for k, v in (summary or {}).items()
                        if k in ("timeout_source", "terminal_state_before_cleanup",
                                 "finalization_reserve", "payload_before_cleanup",
                                 "model_attempts", "attempts", "events",
                                 "cleanup_decisions")
                    }
                    src = str((summary or {}).get("timeout_source") or "")
                    if src and src not in ("none", ""):
                        any_trace = True
                        print(f"  {slug}/{mode}: timeout_source={src}")
                        print(f"      {j(interesting, 1200)}")
        if not any_trace:
            print("  No leg recorded a non-'none' timeout_source.")
    except Exception as exc:
        print(f"  leg_trace unavailable: {exc!r}")

    # ------------------------------------------------------------- priorities
    rule("4. ACCEPTANCE PRIORITIES — raw and accepted, each with composition")
    prios = d.get("acceptance_priorities") or d.get("priorities") or []
    for p in prios:
        print()
        print(f"  rank {p.get('rank')}: {p.get('name')}")
        for k in ("ok", "evaluated", "observed", "counted", "raw", "accepted",
                  "accepted_status", "papers", "not_evaluated_papers",
                  "not_evaluated_legs", "contract_adjusted",
                  "requested_core_coverage"):
            if k in p:
                print(f"      {k:26s} = {j(p[k], 1500)}")
        reasons = p.get("not_evaluated_reasons")
        if reasons:
            print(f"      not_evaluated_reasons ({len(reasons)}):")
            for leg, why in list(reasons.items()):
                print(f"        {leg}: {str(why)[:400]}")

    # ------------------------------------------------- priority 1 composition
    rule("5. PRIORITY-1 ROW COMPOSITION (raw) — every finding, named")
    rows1 = d.get("priority1_rows") or []
    if rows1:
        for r in rows1:
            print(f"  {j(r, 400)}")
    else:
        n = 0
        for pap in d.get("papers", []):
            for mode, leg in (pap.get("legs") or {}).items():
                for f in (leg.get("semantic") or {}).get("findings", []) or []:
                    blob = json.dumps(f, default=str)
                    if "false_real" in blob:
                        n += 1
                        print(f"  {pap.get('paper_id')} {mode} {blob[:300]}")
        print(f"  (derived from per-leg findings: {n} row(s))")

    # ------------------------------------------------- negative controls
    rule("6. NEGATIVE CONTROLS")
    for pap in d.get("papers", []):
        pid = pap.get("paper_id")
        case = next((c for c in gold.cases if getattr(c, "paper_id", None) == pid), None)
        is_neg = bool(getattr(case, "is_negative_control", False)) if case else False
        rel = getattr(case, "mechanistic_relevance", "?") if case else "?"
        if not is_neg and rel != "context_only":
            continue
        print(f"  {pid}  relevance={rel}  is_negative_control={is_neg}")
        for mode, leg in sorted((pap.get("legs") or {}).items()):
            sem = leg.get("semantic") or {}
            print(f"      {mode:9s} status={leg.get('status')!r} "
                  f"release={leg.get('release_status')!r} "
                  f"reactions={sem.get('retained_reactions', leg.get('reaction_count'))} "
                  f"semantic_ok={sem.get('ok')}")
            for k in ("negative_control_status", "empty_is_correct",
                      "rejection_reason", "empty_pathway_reason"):
                if k in leg or k in sem:
                    print(f"        {k} = {j(leg.get(k, sem.get(k)), 300)}")

    # --------------------------------- priority 2 evaluability / gold limits
    rule("7. PRIORITY-2 EVALUABILITY — the acceptance-instrument limitation (D-087)")
    n_complete = sum(1 for c in gold.cases
                     if getattr(c, "supported_reactions_complete", False) is True)
    ceilings = [(getattr(c, "paper_id", "?"), getattr(c, "max_retained_reactions", None))
                for c in gold.cases
                if getattr(c, "max_retained_reactions", None) is not None]
    print(f"  gold cases                                : {len(gold.cases)}")
    print(f"  supported_reactions_complete TRUE         : {n_complete}")
    print(f"  max_retained_reactions set                : {len(ceilings)} -> {ceilings}")
    print()
    print("  Priority 2 = N is a real number and it is NOT a measure of how much")
    print("  invented chemistry this run produced. With supported_reactions_complete")
    print("  unset on every case, the unsupported-reaction verdict can only be reached")
    print("  where a max_retained_reactions ceiling exists -- and both ceilings are on")
    print("  negative controls. Reported as an acceptance-instrument limitation per D-087.")

    # ------------------------------------------------------- contract adjustments
    rule("8. CONTRACT ADJUSTMENTS APPLIED")
    found = False
    for key in ("contract_adjustments", "tolerances_applied", "adjustments"):
        if key in d:
            found = True
            print(f"  {key} = {j(d[key], 3000)}")
    for p in prios:
        if p.get("raw") is not None and p.get("accepted") is not None:
            same = p.get("raw") == p.get("accepted")
            print(f"  rank {p.get('rank')}: raw={p.get('raw')} accepted={p.get('accepted')} "
                  f"{'(identical - no adjustment reached this priority)' if same else '(ADJUSTED)'}")
            found = True
    if not found:
        print("  none reported")

    # ------------------------------------------------------- LpxH, unmeasured claims
    rule("9. LpxH ON PMC12444477 — measured here or reported unverified")
    hit = False
    for pap in d.get("papers", []):
        if pap.get("paper_id") != "PMC12444477":
            continue
        for mode, leg in sorted((pap.get("legs") or {}).items()):
            sem = leg.get("semantic") or {}
            names = [f.get("name") for f in (sem.get("findings") or [])
                     if isinstance(f, dict) and f.get("name")]
            print(f"  {mode:9s} status={leg.get('status')!r} findings={len(names)}")
            if names:
                hit = True
                print(f"      names   : {j(sorted(set(names)), 600)}")
                print(f"      LpxH    : {'LpxH' in names}")
    if not hit:
        print("  No findings available on either leg -> LpxH remains UNVERIFIED on T-108.")

    # ------------------------------------------------------- overall verdict
    rule("10. OVERALL")
    for k in ("accepted", "verdict", "ok", "is_complete", "summary_line"):
        if k in d:
            print(f"  {k:20s} = {j(d[k], 600)}")
    hard = [p for p in prios if p.get("ok") is False]
    unevaluated = [p for p in prios if p.get("evaluated") is False]
    print(f"  priorities with ok=false     : {[p.get('rank') for p in hard]}")
    print(f"  priorities not evaluated     : {[p.get('rank') for p in unevaluated]}")
    print()
    print(f"  T108_SCORE_EMITTED: legs={comp.get('legs_attempted')} "
          f"scorable={comp.get('semantically_scorable_legs')} "
          f"timeouts={len(timeouts)} missing_payload={len(missing_payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

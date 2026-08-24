"""C-080 section 5.6 -- does making the PRODUCTION accession-collision rule
kind-aware flip any committed leg to ``release_ready``?

This is the merge-rule-6 safety property of C-080, re-derived over the WHOLE
committed corpus rather than over the three legs ``f094_reopen_probe.py`` names.
It is deterministic and offline: no network, no model call, no pipeline
execution. It reads only committed artifacts.

WHAT IT MEASURES, per leg that carries all three of ``quarantine_report.json``,
``coverage_summary.json`` and ``final_mapped.json``:

1. **Reproduction.** Run the tree's OWN ``evaluate_production_semantics`` over the
   leg's committed payload and compare the verdict of
   ``no_real_id_or_name_conflict`` against what the run RECORDED in
   ``release.semantic_failed_checks``. A counterfactual whose base measurement
   does not reproduce the record is not trustworthy, so this is reported first.
2. **Faithful replay.** ``classify_release_status`` with the leg's own recorded
   coverage verdict and recorded semantic fields must return the recorded status.
3. **Measured-at-this-tree replay.** The same call, but with
   ``no_real_id_or_name_conflict`` dropped from the failed set exactly when THIS
   TREE's ``_audit_entities`` says the check passes. Run on base and on tip, the
   difference between the two is the whole observable effect of this card.
4. **Conservative upper bound.** The same call with the check dropped
   unconditionally. The real change can only ever turn that check from failing to
   passing, never the reverse, so a corpus with no flip HERE has no flip under any
   kind-aware rule whatsoever -- including one stricter than the one implemented.

It also records the placeholder arm (``placeholder_identities_distinguished``),
the identity census and the forgery counts on every leg, so a patch that made
collisions kind-aware while quietly dropping a forgery finding shows up as a
corpus-wide diff rather than as an untested claim.

Usage::

    python c080_release_flip_probe.py [--out <json>] [--root <repo>]
"""

import argparse
import json
import os
import sys

import t2pw
from t2pw.bench.semantic import CHECK_ID_CONFLICT, CHECK_PLACEHOLDER_IDENTITY
from t2pw.bench.semantic_production import evaluate_production_semantics
from t2pw.pipeline.release_status import (
    RELEASE_READY,
    SEMANTIC_FAILED,
    SEMANTIC_GATING_CHECKS,
    SEMANTIC_NOT_EVALUATED,
    classify_release_status,
)

COLLISION = "accession_claimed_by_multiple_entities"
FORGERY = "placeholder_claims_real_identity"


def _legs(root):
    out = []
    base = os.path.join(root, "runs_verify")
    for run in sorted(os.listdir(base)):
        papers = os.path.join(base, run, "papers")
        if not os.path.isdir(papers):
            continue
        for paper in sorted(os.listdir(papers)):
            for mode in sorted(os.listdir(os.path.join(papers, paper))):
                leg = os.path.join(papers, paper, mode)
                need = ("quarantine_report.json", "coverage_summary.json", "final_mapped.json")
                if all(os.path.isfile(os.path.join(leg, n)) for n in need):
                    out.append((run, "%s/%s" % (paper, mode), leg))
    return out


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _classify(cov, rel, evaluation, failed):
    return classify_release_status(
        coverage=cov,
        pipeline_executed=bool(rel.get("pipeline_executed")),
        strict_gates_passed=bool(rel.get("strict_gates_passed")),
        semantic_check_evaluability=rel.get("semantic_check_evaluability") or [],
        retrieval_attempts=rel.get("retrieval_attempts"),
        expansion_blocked_reason=rel.get("expansion_blocked_reason") or "",
        semantic_evaluation=evaluation,
        semantic_not_evaluated_reason=rel.get("semantic_not_evaluated_reason") or "",
        semantic_failed_checks=failed,
    ).to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = os.path.abspath(args.root)

    print("MEASURED TREE  t2pw : %s" % t2pw.__file__)
    print("MEASURED TREE  cwd  : %s" % os.getcwd())
    print("CORPUS ROOT         : %s" % root)
    print("CHECK_ID_CONFLICT   : %r" % CHECK_ID_CONFLICT)
    print("GATING?             : %s" % (CHECK_ID_CONFLICT in SEMANTIC_GATING_CHECKS))
    print()

    rows = []
    legs = _legs(root)
    for run, leg, path in legs:
        rel = _load(os.path.join(path, "quarantine_report.json")).get("release") or {}
        cov = _load(os.path.join(path, "coverage_summary.json"))
        payload = _load(os.path.join(path, "final_mapped.json"))

        report = evaluate_production_semantics(payload)
        id_check = report.checks.get(CHECK_ID_CONFLICT)
        ph_check = report.checks.get(CHECK_PLACEHOLDER_IDENTITY)
        findings = list(getattr(id_check, "findings", ()) or ())
        collisions = [f for f in findings if f.get("kind") == COLLISION]
        forgeries = [f for f in findings if f.get("kind") == FORGERY]
        ph_findings = list(getattr(ph_check, "findings", ()) or ())

        recorded_status = rel.get("status")
        recorded_eval = str(rel.get("semantic_evaluation") or SEMANTIC_NOT_EVALUATED)
        recorded_failed = [str(n) for n in (rel.get("semantic_failed_checks") or ())]
        recorded_id_failed = CHECK_ID_CONFLICT in recorded_failed
        measured_id_failed = not bool(getattr(id_check, "ok", True))

        # (1) reproduction of the RECORDED id verdict by this tree's measurement
        if recorded_eval != SEMANTIC_FAILED:
            reproduces = "n/a(%s)" % recorded_eval
        else:
            reproduces = "MATCH" if recorded_id_failed == measured_id_failed else "DIVERGES"

        # (2) faithful replay
        replay = _classify(cov, rel, recorded_eval, recorded_failed)

        # (3) measured at this tree
        if measured_id_failed:
            failed_now = list(recorded_failed)
        else:
            failed_now = [n for n in recorded_failed if n != CHECK_ID_CONFLICT]
        eval_now = recorded_eval
        if recorded_eval == SEMANTIC_FAILED and not failed_now:
            eval_now = "passed"
        measured = _classify(cov, rel, eval_now, failed_now)

        # (4) conservative upper bound: the check always passes
        upper_failed = [n for n in recorded_failed if n != CHECK_ID_CONFLICT]
        upper_eval = recorded_eval
        if recorded_eval == SEMANTIC_FAILED and not upper_failed:
            upper_eval = "passed"
        upper = _classify(cov, rel, upper_eval, upper_failed)

        row = {
            "run": run,
            "leg": leg,
            "recorded_status": recorded_status,
            "recorded_strict_acceptance_eligible": rel.get("strict_acceptance_eligible"),
            "recorded_semantic_evaluation": recorded_eval,
            "recorded_failed_checks": recorded_failed,
            "recorded_id_check_failed": recorded_id_failed,
            "measured_id_check_failed": measured_id_failed,
            "measured_id_check_reproduces_record": reproduces,
            "measured_collision_findings": [
                "%s:%s <- %s" % (f.get("namespace"), f.get("identifier"), f.get("entities"))
                for f in collisions
            ],
            "measured_forgery_findings": [f.get("pointer") for f in forgeries],
            "measured_id_finding_total": len(findings),
            "measured_placeholder_check_ok": bool(getattr(ph_check, "ok", True)),
            "measured_placeholder_findings": [f.get("pointer") for f in ph_findings],
            "measured_identity_census": dict(report.identity_census or {}),
            "measured_scientific_errors": dict(report.scientific_errors or {}),
            "replay_as_recorded_status": replay.get("status"),
            "replay_reproduces_record": replay.get("status") == recorded_status,
            "measured_status": measured.get("status"),
            "measured_strict_acceptance_eligible": measured.get("strict_acceptance_eligible"),
            "measured_reasons": measured.get("reasons") or [],
            "upper_bound_status": upper.get("status"),
            "upper_bound_strict_acceptance_eligible": upper.get("strict_acceptance_eligible"),
            "upper_bound_reasons": upper.get("reasons") or [],
        }
        rows.append(row)

        print("==== %-26s (%s)" % (leg, run))
        print("   recorded status / elig      : %-16s %s"
              % (recorded_status, rel.get("strict_acceptance_eligible")))
        print("   recorded failed checks      : %s" % recorded_failed)
        print("   measured id-check failed?   : %-6s  reproduces record: %s"
              % (measured_id_failed, reproduces))
        print("   measured collisions         : %s" % (row["measured_collision_findings"] or "none"))
        print("   measured forgeries          : %s" % (row["measured_forgery_findings"] or "none"))
        print("   measured placeholder ok     : %-6s findings: %s"
              % (row["measured_placeholder_check_ok"], row["measured_placeholder_findings"] or "none"))
        print("   census                      : %s" % row["measured_identity_census"])
        print("   replay as recorded          : %-16s %s"
              % (replay.get("status"), "MATCHES" if row["replay_reproduces_record"] else "!!! DIVERGES !!!"))
        print("   measured-at-this-tree       : %-16s elig=%s"
              % (measured.get("status"), measured.get("strict_acceptance_eligible")))
        print("   conservative upper bound    : %-16s elig=%s"
              % (upper.get("status"), upper.get("strict_acceptance_eligible")))
        if upper.get("status") == RELEASE_READY and recorded_status != RELEASE_READY:
            print("   *** UPPER BOUND FLIPS THIS LEG TO release_ready ***")
        print("   reasons (upper bound)       : %s" % (upper.get("reasons") or []))
        print()

    diverged = [r for r in rows if not r["replay_reproduces_record"]]
    id_diverged = [r for r in rows if r["measured_id_check_reproduces_record"] == "DIVERGES"]
    flips_measured = [
        r for r in rows
        if r["measured_status"] == RELEASE_READY and r["recorded_status"] != RELEASE_READY
    ]
    flips_upper = [
        r for r in rows
        if r["upper_bound_status"] == RELEASE_READY and r["recorded_status"] != RELEASE_READY
    ]
    elig = [r for r in rows if r["upper_bound_strict_acceptance_eligible"]]
    ready_recorded = [r for r in rows if r["recorded_status"] == RELEASE_READY]
    ready_upper = [r for r in rows if r["upper_bound_status"] == RELEASE_READY]

    print("SUMMARY  (sample size: %d legs)" % len(rows))
    print("  legs whose faithful replay DIVERGES from the record : %d %s"
          % (len(diverged), [r["leg"] for r in diverged]))
    print("  legs whose id-check measurement DIVERGES            : %d %s"
          % (len(id_diverged), [r["leg"] for r in id_diverged]))
    print("  legs recorded release_ready                          : %d" % len(ready_recorded))
    print("  legs release_ready under the MEASURED rule           : %d"
          % len([r for r in rows if r["measured_status"] == RELEASE_READY]))
    print("  legs release_ready under the UPPER BOUND             : %d" % len(ready_upper))
    print("  FLIPS to release_ready, measured                     : %d %s"
          % (len(flips_measured), [r["leg"] for r in flips_measured]))
    print("  FLIPS to release_ready, upper bound                  : %d %s"
          % (len(flips_upper), [r["leg"] for r in flips_upper]))
    print("  legs strict_acceptance_eligible under upper bound    : %d %s"
          % (len(elig), [r["leg"] for r in elig]))
    print("  total collision findings across corpus              : %d"
          % sum(len(r["measured_collision_findings"]) for r in rows))
    print("  total forgery findings across corpus                : %d"
          % sum(len(r["measured_forgery_findings"]) for r in rows))
    print("  total placeholder findings across corpus            : %d"
          % sum(len(r["measured_placeholder_findings"]) for r in rows))
    print()
    for row in rows:
        mark = ""
        if row["upper_bound_status"] != row["recorded_status"]:
            mark = "  <== STATUS MOVES"
        print("  %-26s %-16s -> measured %-16s upper %-16s%s"
              % (row["leg"], row["recorded_status"], row["measured_status"],
                 row["upper_bound_status"], mark))

    verdict = "NO LEG FLIPS TO release_ready" if not flips_upper else "AT LEAST ONE LEG FLIPS"
    print()
    print("VERDICT: %s (upper bound over %d legs)" % (verdict, len(rows)))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"t2pw": t2pw.__file__, "legs": rows}, handle, indent=1, sort_keys=True)
        print("wrote %s" % args.out)
    return 0 if not flips_upper and not diverged else 1


if __name__ == "__main__":
    sys.exit(main())

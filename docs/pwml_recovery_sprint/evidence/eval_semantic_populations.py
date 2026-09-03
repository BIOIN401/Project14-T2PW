"""F-177 -- the semantic evaluator that reports POPULATIONS instead of one number.

EVALUATION-ONLY. Read-only over archived run directories; writes nothing into them,
imports no production writer, and produces NO acceptance verdict. Running it is not
a re-score of T-107/T-108/T-109 and does not touch their dispositions.

THE DEFECT IT EXISTS FOR (F-177). ``bench.acceptance`` resolves a leg's payload
through ``_PAYLOAD_FILES = ("final_mapped.json", "merged_payload.json")`` and falls
back SILENTLY. Those two files are not the same kind of object:

  * ``final_mapped.json``  -- CANONICAL. Post-mapping, post-enrichment, the payload
    the Stage-3 gate is bound to by ``payload_sha256`` and the one an export is made
    from. A verdict here is a statement about a PRODUCT.
  * ``merged_payload.json`` -- FALLBACK. PRE-mapping and PRE-quarantine. It exists so
    a leg that died early still leaves something an audit can read, which is a good
    reason and not the defect. A verdict here is a statement about a DRAFT, and on
    T-109 every fallback leg exported nothing at all.

Summing them produces a corpus number nobody can act on, and several check reason
strings say "the exported payload" while scoring a leg that produced no export.

WHAT THIS MODULE REFUSES TO DO. It will not print a combined pass/fail rate without
naming its denominator and its populations. Every count below is per-population.

WHAT IT MUST NOT BECOME. A way to make results look better by quarantining the
failures into a population nobody reads. Every population is always printed, always
with its failures enumerated, including the empty ones -- "fallback: 0 evaluated" is
itself a fact about the run. Attribution, not favourable reporting.

Usage:
  python eval_semantic_populations.py <repo-root> <run-dir> [<run-dir> ...] [--json OUT]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: Population identity, keyed by the filename the payload actually came from.
#: An unknown source is its own population rather than being folded into either --
#: guessing is what F-177 is about.
CANONICAL = "canonical"
FALLBACK = "fallback"
UNKNOWN = "unknown_source"
NO_PAYLOAD = "no_payload"

POPULATION_BY_FILE: Dict[str, str] = {
    "final_mapped.json": CANONICAL,
    "merged_payload.json": FALLBACK,
}

#: The order populations are always printed in. Fixed, so a reader comparing two
#: reports is never comparing differently-ordered tables.
POPULATION_ORDER: Tuple[str, ...] = (CANONICAL, FALLBACK, UNKNOWN, NO_PAYLOAD)

#: The five verdict states, kept apart because collapsing any two of them is the
#: error this whole finding family records. ``APPLICABLE`` is NOT one of them: it is
#: derived (``PASSED`` or ``FAILED``), and never a synonym for ``PASSED``.
PASSED = "passed"
FAILED = "failed"
INAPPLICABLE = "inapplicable"
MISSING = "artifact_missing"
MALFORMED = "artifact_malformed"
VERDICT_ORDER: Tuple[str, ...] = (PASSED, FAILED, INAPPLICABLE, MISSING, MALFORMED)


def classify_population(payload_source: str, payload: Any) -> str:
    """Which population a leg's verdicts belong to. Never guesses."""

    if not isinstance(payload, dict) or not payload:
        return NO_PAYLOAD
    return POPULATION_BY_FILE.get(payload_source or "", UNKNOWN)


def classify_verdict(check: Any) -> str:
    """One of :data:`VERDICT_ORDER` for one ``CheckResult``.

    ``inapplicable_reason`` is consulted BEFORE ``ok``, deliberately: an inapplicable
    check carries ``ok=True`` by design (absence of evidence is never a failure), so
    reading ``ok`` first would silently score every unevaluable check as a pass. That
    is precisely the misreading F-176 recorded.

    ``artifact_missing`` and ``artifact_malformed`` are split out of ``inapplicable``
    by matching on the corrected reason text from ``bench.semantic``. They are
    different facts about different components -- one is a caller that supplied
    nothing, the other an artifact that arrived broken -- and only the second is a
    defect in the evidence.
    """

    reason = (getattr(check, "inapplicable_reason", "") or "").lower()
    if reason:
        if "malformed" in reason or "carries no 'rejected' key" in reason:
            return MALFORMED
        if "supplied to this evaluation" in reason or "no payload" in reason:
            return MISSING
        return INAPPLICABLE
    ok = getattr(check, "ok", None)
    if ok is None:
        return INAPPLICABLE
    return PASSED if ok else FAILED


class Tally:
    """Per-check, per-population counts plus the failures themselves.

    The failures travel with the counts on purpose. A population report that gave
    only numbers would let a reader see "fallback: 2 failed" and never learn what
    failed, which is how a real defect hides inside a correctly-attributed bucket.
    """

    def __init__(self) -> None:
        self.counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.failures: List[Dict[str, Any]] = []
        self.legs: Dict[str, List[str]] = {name: [] for name in POPULATION_ORDER}
        #: Legs that carried a payload but produced no evaluated report. Counted
        #: rather than skipped: a module about silently-absorbed populations that
        #: silently absorbs one would be worthless.
        self.not_evaluated: List[Dict[str, str]] = []

    def record_not_evaluated(self, population: str, leg: str, reason: str) -> None:
        self.not_evaluated.append(
            {"population": population, "leg": leg, "reason": reason})

    def record_leg(self, population: str, leg: str) -> None:
        self.legs.setdefault(population, []).append(leg)

    def record(self, check: str, population: str, verdict: str) -> None:
        per_check = self.counts.setdefault(check, {})
        per_pop = per_check.setdefault(population, {name: 0 for name in VERDICT_ORDER})
        per_pop[verdict] += 1

    def record_failure(self, check: str, population: str, leg: str,
                       findings: Any) -> None:
        self.failures.append({
            "check": check, "population": population, "leg": leg,
            "findings": [f for f in (findings or ())][:5],
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "legs_by_population": {k: sorted(v) for k, v in self.legs.items() if v},
            "counts": self.counts,
            "failures": self.failures,
            "not_evaluated": self.not_evaluated,
        }


def evaluate_run(repo: Path, run: Path, tally: Tally) -> None:
    """Score every leg of one archived run into ``tally``. Read-only."""

    from t2pw.bench.acceptance import (  # noqa: PLC0415
        _PAYLOAD_FILES, _QUARANTINE_FILES, _ADMISSION_FILES,
        _first_existing, _paper_text,
    )
    from t2pw.bench.goldset import load_gold_set  # noqa: PLC0415
    from t2pw.bench.semantic import validate_semantic_coverage  # noqa: PLC0415

    cases = {case.paper_id: case for case in load_gold_set().cases}
    papers = run / "papers"
    if not papers.is_dir():
        print(f"  {run}: no papers/ directory -- skipped")
        return

    for paper_dir in sorted(p for p in papers.iterdir() if p.is_dir()):
        slug = paper_dir.name
        case = cases.get(slug)
        paper_text = _paper_text(run, slug)
        for mode_dir in sorted(p for p in paper_dir.iterdir() if p.is_dir()):
            leg = f"{slug}/{mode_dir.name}"
            payload, payload_source, _ = _first_existing(mode_dir, _PAYLOAD_FILES)
            admission, _, _ = _first_existing(mode_dir, _ADMISSION_FILES)
            quarantine, _, _ = _first_existing(mode_dir, _QUARANTINE_FILES)

            population = classify_population(payload_source, payload)
            tally.record_leg(population, leg)
            if case is None or population == NO_PAYLOAD:
                continue

            report = validate_semantic_coverage(
                case,
                payload if isinstance(payload, dict) else None,
                mode=mode_dir.name,
                admission=admission if isinstance(admission, dict) else None,
                quarantine_report=quarantine if isinstance(quarantine, dict) else None,
                paper_text=paper_text,
            )
            if not report.evaluated:
                # COUNTED, NOT DROPPED. An earlier draft `continue`d here with no
                # counter, in a module whose entire thesis is that a silently
                # absorbed leg is the defect. It never fired on T-109, which is
                # exactly why it would have gone unnoticed.
                tally.record_not_evaluated(
                    population, leg, report.not_evaluated_reason or "(no reason given)")
                continue
            for name, check in report.checks.items():
                verdict = classify_verdict(check)
                tally.record(name, population, verdict)
                if verdict == FAILED:
                    tally.record_failure(name, population, leg,
                                         getattr(check, "findings", None))


def render(tally: Tally) -> None:
    print("\n" + "=" * 78)
    print("F-177 SEMANTIC POPULATIONS -- never one number, always a denominator")
    print("=" * 78)
    print("  canonical  = final_mapped.json  : post-mapping, gate-bound, export-eligible.")
    print("               A verdict here is about a PRODUCT.")
    print("  fallback   = merged_payload.json: PRE-mapping, PRE-quarantine draft from a")
    print("               leg that produced no export. A verdict here is about a DRAFT.")
    print("  These are NOT the same measurement and are never summed below.\n")

    for population in POPULATION_ORDER:
        legs = tally.legs.get(population) or []
        print(f"  {population:<14} legs = {len(legs)}")
        if legs:
            print(f"      {', '.join(sorted(legs))}")

    print("\n  PER CHECK, PER POPULATION")
    print(f"    {'check':<46} {'population':<12} "
          + " ".join(f"{v:>13}" for v in VERDICT_ORDER))
    for check in sorted(tally.counts):
        for population in POPULATION_ORDER:
            row = tally.counts[check].get(population)
            if not row:
                continue
            evaluated = sum(row.values())
            print(f"    {check:<46} {population:<12} "
                  + " ".join(f"{row[v]:>13}" for v in VERDICT_ORDER)
                  + f"   (n={evaluated})")

    if tally.not_evaluated:
        print("")
        print("  LEGS WITH A PAYLOAD BUT NO EVALUATED REPORT -- counted, not skipped:")
        for row in tally.not_evaluated:
            print(f"    [{row['population']}] {row['leg']}  {row['reason']}")

    print("\n  EVERY FAILURE, ATTRIBUTED. A population report that hid these would be")
    print("  worse than the mixed number it replaces.")
    if not tally.failures:
        print("    none")
    for failure in tally.failures:
        print(f"    [{failure['population']}] {failure['leg']}  {failure['check']}")
        for finding in failure["findings"][:2]:
            print(f"        {json.dumps(finding)[:200]}")


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    out_json: Optional[str] = None
    args = list(argv[1:])
    if "--json" in args:
        index = args.index("--json")
        out_json = args[index + 1]
        del args[index:index + 2]

    repo = Path(args[0]).resolve()
    sys.path.insert(0, str(repo / "src"))
    import t2pw  # noqa: PLC0415
    print(f"MEASURED_TREE t2pw = {Path(t2pw.__file__).resolve()}")
    # THE GOLD BLOB IS PART OF THE MEASUREMENT. The first run of this reporter was
    # committed beside a gold set it had NOT been measured against (D-091 moved the
    # blob hours later), and nothing in the output said so. Stamping it makes a
    # stale report self-evidently stale instead of quietly wrong.
    import subprocess  # noqa: PLC0415
    try:
        blob = subprocess.run(
            ["git", "-C", str(repo), "hash-object", "src/t2pw/bench/gold/pinned_v1.json"],
            capture_output=True, text=True, timeout=60, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        blob = "(git unavailable -- THIS REPORT IS UNSTAMPED)"
    print(f"GOLD BLOB          = {blob}")

    tally = Tally()
    for entry in args[1:]:
        run = Path(entry)
        if not run.is_absolute():
            run = (repo / entry).resolve()
        print(f"RUN = {run}")
        evaluate_run(repo, run, tally)

    render(tally)

    if out_json:
        Path(out_json).write_text(
            json.dumps(tally.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        print(f"\n  structured report written to {out_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))

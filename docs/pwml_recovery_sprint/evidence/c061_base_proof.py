"""C-061 G9 proof: behavioural, and callable at the dispatch base.

Uses ONLY API that exists at ``472293c`` — ``validate_evidence_span`` and the
``SpanVerdict`` it returns — so the difference it reports is a difference in
BEHAVIOUR, not in symbols. ``parse_span_relations`` is deliberately not imported:
"the new function does not exist at base" is not a proof of anything.

Run it against the base source export and against the branch tip::

    PYTHONPATH=<base-export>/src python c061_base_proof.py   # expect FAIL
    PYTHONPATH=<worktree>/src   python c061_base_proof.py   # expect PASS

Exit code 0 only when every clause holds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEG = ROOT / "runs" / "2026-08-02_2130" / "papers" / "PMC12421875" / "research"

SPAN = (
    "Subsequently, MenA joins DHNA and prenyl diphosphate to produce "
    "demethylmenaquinone (DMK), and MenG demethylates DMK to generate MK ( Fig. 1A )."
)


def main() -> int:
    import t2pw
    from t2pw.rag.admission import validate_evidence_span

    print("t2pw:", t2pw.__file__)

    def check(label, expect_ok, **claim):
        verdict = validate_evidence_span(SPAN, **claim)
        good = bool(verdict.ok) is expect_ok
        print(
            f"{'PASS' if good else 'FAIL'} {label}: ok={verdict.ok} "
            f"expected={expect_ok} reasons={verdict.reasons}"
        )
        return good, verdict

    results = []

    # A1 — the correction. The claim is the span's first clause, verbatim.
    ok, _ = check(
        "A1 MenA reaction admitted",
        True,
        inputs=["DHNA", "prenyl diphosphate"],
        outputs=["demethylmenaquinone (DMK)"],
        enzymes=["MenA"],
    )
    results.append(ok)

    # A2 — the second clause, refused at base over the figure callout.
    ok, verdict = check(
        "A2 MenG reaction admitted", True, inputs=["DMK"], outputs=["MK"], enzymes=["MenG"]
    )
    results.append(ok)

    # A6 — behavioural: the relation the verdict carries names a compound.
    relation = verdict.relation
    products = list(relation.outputs) if relation is not None else []
    good = bool(products) and not any("fig" in p.casefold() for p in products)
    print(f"{'PASS' if good else 'FAIL'} A6 product is a compound: {products}")
    results.append(good)

    # A3/A4/A5 — the refusals must hold at BOTH ends. A proof that only shows the
    # widening would not show that the gate still refuses.
    results.append(
        check(
            "A3 catalyst crossover refused",
            False,
            inputs=["DHNA", "prenyl diphosphate"],
            outputs=["demethylmenaquinone (DMK)"],
            enzymes=["MenG"],
        )[0]
    )
    results.append(
        check(
            "A4 reversal refused",
            False,
            inputs=["demethylmenaquinone (DMK)"],
            outputs=["DHNA", "prenyl diphosphate"],
            enzymes=["MenA"],
        )[0]
    )
    results.append(
        check(
            "A5 dropped substrate refused",
            False,
            inputs=["DHNA"],
            outputs=["demethylmenaquinone (DMK)"],
            enzymes=["MenA"],
        )[0]
    )

    # A7 — the committed production candidate, replayed.
    report = json.loads(
        (LEG / "rag_admission_report.json").read_text(encoding="utf-8")
    )
    candidate = report["rejected"][1]
    assert candidate["evidence"]["span"] == SPAN, "the committed span moved"
    replay = validate_evidence_span(
        candidate["evidence"]["span"],
        inputs=candidate["inputs"],
        outputs=candidate["outputs"],
        enzymes=candidate["enzymes"],
        reversible=bool(candidate.get("reversible")),
    )
    good = bool(replay.ok)
    print(f"{'PASS' if good else 'FAIL'} A7 committed rejected[1] replays as admitted")
    results.append(good)

    print(f"\n{sum(results)}/{len(results)} clauses hold")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

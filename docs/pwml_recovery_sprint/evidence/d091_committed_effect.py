"""D-091 -- what the completeness flag ACTUALLY does to the committed corpus.

READ-ONLY. Written because the first draft of D-091 asserted that the flag "changes
no acceptance verdict at all on the committed corpus", gave as its reason that both
of this paper's legs are ``scope_conflict`` with no ``final_mapped.json`` -- and that
reason described the T-109 run, which is UNTRACKED. Caught by review.

``git ls-files`` says the committed corpus holds, for PMC12312563, THREE canonical
legs and nineteen fallback ones. So the effect is measurable, it was not measured,
and the claim was unproven. This measures it.

The A/B is the flag and nothing else: the same gold case, the same payload, the same
paper text, scored twice with ``supported_reactions_complete`` False and True.

Usage:  python d091_committed_effect.py <repo-root>
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

CASE_ID = "PMC12312563"


def committed(repo: Path, pattern: str) -> List[Path]:
    listed = subprocess.run(
        ["git", "-C", str(repo), "ls-files"], capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return [repo / line for line in listed if CASE_ID in line and line.endswith(pattern)]


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    repo = Path(argv[1]).resolve()
    sys.path.insert(0, str(repo / "src"))

    from t2pw.bench.acceptance import _paper_text  # noqa: E402
    from t2pw.bench.goldset import load_gold_set  # noqa: E402
    from t2pw.bench.semantic import (  # noqa: E402
        CHECK_SUPPORTED_REACTIONS, ERR_UNSUPPORTED_REACTIONS,
        validate_semantic_coverage,
    )

    import t2pw  # noqa: E402
    print(f"MEASURED_TREE t2pw = {Path(t2pw.__file__).resolve()}")
    blob = subprocess.run(
        ["git", "-C", str(repo), "hash-object", "src/t2pw/bench/gold/pinned_v1.json"],
        capture_output=True, text=True, check=True).stdout.strip()
    print(f"GOLD BLOB          = {blob}")

    case_off = {c.paper_id: c for c in load_gold_set().cases}[CASE_ID]
    # D-092 WITHDREW the flag, so the shipped tree is the OFF arm and the ON arm is
    # synthesised. That is the right way round now: this probe answers "what WOULD
    # the flag do", which is the question the withdrawal rests on.
    assert case_off.supported_reactions_complete is False, (
        "the flag is set in this tree; D-092 says it must not be")
    case_on = dataclasses.replace(case_off, supported_reactions_complete=True)

    canonical = committed(repo, "final_mapped.json")
    fallback = committed(repo, "merged_payload.json")
    print(f"\nCOMMITTED legs for {CASE_ID}: "
          f"{len(canonical)} canonical, {len(fallback)} fallback")
    print("The T-109 run (runs_verify/2026-09-02_2052) is UNTRACKED and is NOT below.\n")

    moved = 0
    rows: List[Dict[str, Any]] = []
    for population, paths in (("canonical", canonical), ("fallback", fallback)):
        for path in sorted(paths):
            leg = path.parent
            rel = leg.relative_to(repo).as_posix()
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            run_dir = leg.parent.parent.parent
            slug = leg.parent.name
            text = _paper_text(run_dir, slug)

            out: Dict[str, Any] = {"leg": rel, "population": population}
            for label, case in (("off", case_off), ("on", case_on)):
                report = validate_semantic_coverage(
                    case, payload, mode=leg.name, paper_text=text)
                check = report.checks.get(CHECK_SUPPORTED_REACTIONS)
                out[label] = {
                    "evaluated": bool(report.support.get("unsupported_verdict_evaluated")),
                    "ok": getattr(check, "ok", None),
                    "applicable": getattr(check, "applicable", None),
                    "unsupported": report.scientific_errors.get(ERR_UNSUPPORTED_REACTIONS),
                    "n_reactions": int((report.graph or {}).get("n_reactions", 0)),
                }
            changed = out["off"] != out["on"]
            out["changed"] = changed
            moved += 1 if changed else 0
            rows.append(out)
            flag = "MOVED" if changed else "same "
            print(f"  [{population:<9}] {flag} {rel}")
            print(f"        rows={out['on']['n_reactions']:<3} "
                  f"OFF evaluated={out['off']['evaluated']!s:<5} ok={out['off']['ok']!s:<5} "
                  f"unsupported={out['off']['unsupported']}")
            print(f"        {'':<7} ON  evaluated={out['on']['evaluated']!s:<5} "
                  f"ok={out['on']['ok']!s:<5} unsupported={out['on']['unsupported']}")

    print(f"\nLEGS WHOSE SUPPORTED-REACTION VERDICT MOVED: {moved} of {len(rows)}")
    print("A leg that MOVED went from a WITHHELD zero to a MEASURED verdict. That is the")
    print("flag doing its job, and it is the opposite of 'changes no acceptance verdict'.")
    print("\nNOTE ON SCOPE: this measures the SEMANTIC check only. Whether a moved verdict")
    print("changes an ACCEPTANCE priority depends on bench/acceptance.py's corpus and its")
    print("contract adjustments, which are not re-run here and are not claimed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))

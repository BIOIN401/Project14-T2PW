"""F-172 G9 proof -- the DEFAULT verdict of ``check_report`` is unchanged.

WHY THIS PROOF AND NOT A BEHAVIOURAL BASE FAILURE. F-172's change is two things
at once, and they need different evidence (merge gate G9):

  * a NEW capability -- the rule-10 reader that was never built. New capability
    carries an explicitly labelled acceptance test and needs no fabricated base
    failure. Those live in ``g11_evidence.py selftest``.
  * a change to a surface EVERY CARD DEPENDS ON. ``bounded_run.py``'s build hash
    appears in every G11 report, and F-163 is the standing reason not to touch
    that casually. So the obligation here is the opposite of a base failure: the
    default verdict must be PROVABLY IDENTICAL, artifact by artifact, across the
    entire committed evidence tree. Anything else silently re-scores 5000+
    committed reports and destroys the comparability the deferral protected.

HOW. Both modules are imported by path -- the BASE copy out of a git worktree at
the base SHA, the TIP copy out of the primary checkout -- and
``check_report(path)`` is called on the SAME list of committed artifacts with no
keyword arguments, i.e. exactly as every existing caller calls it. Any artifact
whose violation list differs is a comparability break and fails this proof.

The flagged rules are then exercised on the same corpus to show they are not
inert: a proof that nothing changed is worthless if nothing would have changed
under any setting.

Usage:
  python f172_default_verdict_equivalence.py <base-g11_evidence.py> <tip-g11_evidence.py>
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, List


def load(path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    base = load(os.path.abspath(argv[1]), "g11_base")
    tip = load(os.path.abspath(argv[2]), "g11_tip")

    # The BASE module's own reports root is the base tree's; the TIP module's is
    # the primary's. The corpus under test is the TIP tree's committed artifacts,
    # because that is the set a reader will run ``check`` over. Base is asked
    # about those same paths explicitly, so both see identical bytes.
    paths = tip.iter_reports()
    print(f"corpus            : {len(paths)} committed artifact(s)")
    print(f"base module       : {argv[1]}")
    print(f"tip module        : {argv[2]}")

    differing = []
    for path in paths:
        before = base.check_report(path)
        after = tip.check_report(path)
        if before != after:
            differing.append((path, before, after))

    print(f"\nDEFAULT-VERDICT EQUIVALENCE: {len(paths) - len(differing)} identical, "
          f"{len(differing)} DIFFERENT")
    for path, before, after in differing[:20]:
        print(f"  DIFF {path}\n       base={before}\n       tip ={after}")

    # Now prove the new rules are not inert. If these all read zero the proof
    # above is vacuous -- "nothing changed" would be true of a no-op patch too.
    n_pin = sum(1 for p in paths if tip.check_report(p, require_pin=True) !=
                tip.check_report(p))
    n_lbl = sum(1 for p in paths if tip.check_report(p, require_label_match=True) !=
                tip.check_report(p))
    n_fgn = sum(1 for p in paths if tip.check_report(p, forbid_foreign_src=True) !=
                tip.check_report(p))
    n_ref = sum(1 for p in paths if tip.check_report(p, forbid_refused_pin=True) !=
                tip.check_report(p))
    print("\nTHE NEW RULES ARE NOT INERT -- artifacts whose verdict CHANGES under each flag:")
    print(f"  --require-pin          : {n_pin}")
    print(f"  --require-label-match  : {n_lbl}")
    print(f"  --forbid-foreign-src   : {n_fgn}")
    print(f"  --forbid-refused-pin   : {n_ref}")

    stats = tip.audit_rule10(paths)
    print("\nRULE-10 AUDIT over the same corpus:")
    for key in ("reports", "direct_pytest", "with_pin", "without_pin",
                "pin_refused", "pin_with_violations", "pin_with_foreign_src",
                "label_mismatch"):
        print(f"  {key:<24} = {stats[key]}")

    ok = not differing and (n_pin or n_lbl or n_fgn or n_ref)
    print("\nVERDICT: " + (
        "PASS -- default verdict identical on every artifact, and the new rules bite."
        if ok else
        "FAIL -- either the default verdict moved, or every new rule is inert."))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))

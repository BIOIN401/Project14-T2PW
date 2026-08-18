"""C-054 G9 CORRECTION proof: an omitted gold classification, base vs tip.

The defect. ``goldset._case`` read the two classification fields as
``canonical_text(raw.get(...)) or CONST``. A *present but invalid* value was
refused by the membership check below it; an *absent* one never reached that
check, so the case loaded carrying a classification nobody authored.
``expected_export`` decides membership of the strict-PWML denominator, so a case
that lost the key left that denominator silently.

Why this script and not a bare pytest. Merge gate G9 requires a proof that fails
*behaviourally* at the base SHA, and the shared block adds that "no exception" is
not enough here -- the failure worth showing is the **value the loader invented**.
So this asserts on ``case.expected_export`` / ``case.mechanistic_relevance`` as
loaded, and states its expectation per label:

* ``--label base`` PASSES only if both omissions LOAD and read ``partial_only`` /
  ``partial`` -- the silent downgrade, demonstrated.
* ``--label tip``  PASSES only if both omissions are REFUSED with ``missing '<f>'``.

Run it against the *other* label and it exits 1. It is a differential, not a
tautology.

Two controls travel with it. A fully declared case must still load and keep both
values verbatim, so the tip arm cannot be satisfied by a loader that refuses
everything; and the pinned set is loaded on both sides, because the whole card
rests on the claim that its exposure is **zero** -- ten cases, both fields
declared, identical strict ids at base and at tip.

No DB, no network, no LLM: ``goldset`` is JSON parsing and string folding, so
F-051's ``.env`` reachability hazard does not reach this measurement. The tree
under test is selected by ``--tree`` and verified by the resolved ``__file__`` of
the module actually imported, never assumed from cwd.

Usage::

    <python> c054_required_classification_probe.py --tree <root> --label base|tip
        --out docs/pwml_recovery_sprint/evidence/c054_g9_base.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

FIELDS = ("expected_export", "mechanistic_relevance")

#: What the removed fallback substituted. The base arm asserts these exact values
#: because "it loaded" alone would not show that a classification was invented.
BASE_DEFAULTS = {"expected_export": "partial_only", "mechanistic_relevance": "partial"}

#: A raw case that is valid in every other respect, so the only thing under test
#: is the deleted key.
VALID_CASE: Dict[str, Any] = {
    "paper_id": "PROBE1",
    "requested_pathway": "menaquinone biosynthesis",
    "requested_organism": "Bacillus subtilis",
    "expected_pathway_anchors": ["menaquinone biosynthesis"],
    "mechanistic_relevance": "core",
    "expected_export": "strict_exportable",
}


def _import_goldset(tree: Path):
    src = (tree / "src").resolve()
    if not src.is_dir():
        raise SystemExit("FAILED: no src/ under %s" % tree)
    sys.path.insert(0, str(src))
    from t2pw.bench import goldset  # noqa: E402  (path set above, on purpose)

    resolved = Path(goldset.__file__).resolve()
    if src not in resolved.parents:
        raise SystemExit("FAILED: imported %s, which is not under %s" % (resolved, src))
    return goldset, resolved


def _write(tmp: Path, name: str, case: Dict[str, Any]) -> Path:
    target = tmp / (name + ".json")
    target.write_text(json.dumps({"version": "c054-probe", "cases": [case]}), encoding="utf-8")
    return target


def _load_one(goldset, path: Path) -> Dict[str, Any]:
    try:
        gold = goldset.load_gold_set(path)
    except goldset.GoldSetError as exc:
        return {"outcome": "refused", "error": str(exc)}
    case = gold.cases[0]
    return {
        "outcome": "loaded",
        "expected_export": case.expected_export,
        "mechanistic_relevance": case.mechanistic_relevance,
        "is_strict_expected": case.is_strict_expected,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, help="root of the tree under test")
    parser.add_argument("--label", required=True, choices=("base", "tip"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    tree = Path(args.tree).resolve()
    goldset, module_file = _import_goldset(tree)

    report: Dict[str, Any] = {
        "task": "C-054",
        "label": args.label,
        "arm": "G9 CORRECTION",
        "tree": str(tree),
        "goldset_module": str(module_file),
        "omitted": {},
        "control_declared": {},
        "pinned": {},
        "verdict": {},
    }
    failures: List[str] = []

    with tempfile.TemporaryDirectory(prefix="c054probe") as raw_tmp:
        tmp = Path(raw_tmp)

        for field in FIELDS:
            case = dict(VALID_CASE)
            case.pop(field)
            observed = _load_one(goldset, _write(tmp, "omit_" + field, case))
            report["omitted"][field] = observed
            if args.label == "base":
                # The behavioural failure is not "it loaded" but "it was classified".
                if observed["outcome"] != "loaded":
                    failures.append(
                        "base: omitting %s was refused, expected a silent load" % field)
                elif observed[field] != BASE_DEFAULTS[field]:
                    failures.append(
                        "base: omitting %s loaded as %r, expected the invented default %r"
                        % (field, observed[field], BASE_DEFAULTS[field]))
            else:
                if observed["outcome"] != "refused":
                    failures.append(
                        "tip: omitting %s loaded as %r instead of being refused"
                        % (field, observed.get(field)))
                elif ("missing '" + field + "'") not in observed["error"]:
                    failures.append(
                        "tip: the refusal did not name the field: %r" % observed["error"])

        control = _load_one(goldset, _write(tmp, "declared", dict(VALID_CASE)))
        report["control_declared"] = control
        if control["outcome"] != "loaded":
            failures.append("a fully declared case must still load, got %r" % (control,))
        elif not (control["expected_export"] == "strict_exportable"
                  and control["mechanistic_relevance"] == "core"
                  and control["is_strict_expected"]):
            failures.append("a declared case must survive verbatim, got %r" % (control,))

    pinned_raw = json.loads(goldset.pinned_gold_set_path().read_text(encoding="utf-8"))
    pinned = goldset.load_gold_set()
    declaring = sum(
        1 for c in pinned_raw["cases"]
        if str(c.get("expected_export") or "").strip()
        and str(c.get("mechanistic_relevance") or "").strip()
    )
    report["pinned"] = {
        "cases": len(pinned),
        "declaring_both_fields": declaring,
        "strict_expected_ids": sorted(pinned.strict_expected_ids),
        "exposure": "zero -- every pinned case declares both fields, so this change "
                    "removes a latent fail-open and moves no score",
    }
    if declaring != len(pinned):
        failures.append("the pinned set does NOT declare both fields on every case; "
                        "exposure is not zero and the card's premise has changed")

    report["verdict"] = {"ok": not failures, "failures": failures}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("label / tree     : %s -> %s" % (args.label, tree))
    print("goldset module   : %s" % module_file)
    for field in FIELDS:
        print("omit %-22s: %s" % (field, report["omitted"][field]))
    print("declared control : %s" % (report["control_declared"],))
    print("pinned           : %d cases, %d declare both, strict=%s"
          % (report["pinned"]["cases"], declaring, report["pinned"]["strict_expected_ids"]))
    print("report written   : %s" % out)
    if failures:
        for line in failures:
            print("FAILED: %s" % line)
        return 1
    print("VERIFIED: the '%s' expectation holds on this tree" % args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())

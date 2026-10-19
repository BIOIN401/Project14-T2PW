"""ORCH-747 -- the exact delta the interaction export puts into pinned IR digests.

``tests/test_compound_resolution_extraction.py::test_build_pwml_ir_matches_the_pre_extraction_golden``
and ``tests/test_pwml_ir_duplicate_row_refusal.py::test_new_acceptance_a_non_colliding_leg_is_byte_identical_to_the_base``
pin sha256 digests of ``build_pwml_ir``'s ``(ir, report)`` over committed leg fixtures.
ORCH-747 makes the IR annotate every interaction (``export_status``, a ``key`` on each
side, a ``withheld`` record on rows PathWhiz cannot draw, and locations / edges / a
process visualization for the rows it can), so those digests move on every leg that
carries an interaction. Permanent merge rule 4 allows a pinned baseline to move only
**deliberately, with an exact documented delta** -- this script is that delta.

For every pinned leg it builds the IR twice under one deterministic, DB-down
configuration (the R3 test's own): once with ``src/t2pw`` AS COMMITTED at the base
revision (``git archive`` into a temp dir, imported by a child interpreter that verifies
which ``t2pw`` it got), once with the working tree. It then lists every JSON pointer at
which the two differ, generalised over list indices, and checks that each one lies under
a prefix the change owns. A pointer outside that set means something other than the
interaction export moved, and the script exits non-zero.

Usage::

    python docs/pwml_recovery_sprint/evidence/orch747_ir_digest_delta.py [--rev HEAD] [--out FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

REPO = Path(__file__).resolve().parents[3]
PACKAGE_REL = "src/t2pw"
DEFAULT_OUT = Path(__file__).resolve().parent / "orch747_ir_digest_delta.json"

#: Where the interaction export is allowed to move an (ir, report) pair.
OWNED_PREFIXES = (
    "/ir/processes/interactions/[]/key",
    "/ir/processes/interactions/[]/export_status",
    "/ir/processes/interactions/[]/withheld",
    "/ir/processes/interactions/[]/interaction_type",   # spelling only: Activation / Inhibition
    "/ir/processes/interactions/[]/left/key",
    "/ir/processes/interactions/[]/right/key",
    "/ir/locations",                                    # endpoints of drawn interactions
    "/ir/edges",
    "/ir/process_visualizations",
    "/ir/protein_complex_visualizations",               # a complex endpoint's visualization
    "/report/warnings",                                 # interaction_withheld_from_pwml
    "/report/counts/locations",
    "/report/counts/edges",
    "/report/counts/process_visualizations",
    "/report/ir_validation/counts",
)


class _DownDb:
    last_error = "harvest_db_down"

    def available(self) -> bool:
        return False


# --------------------------------------------------------------------------- child
def _build_legs(expect_tree: Path, legs_file: Path, out_file: Path) -> int:
    import t2pw
    from t2pw.pwml.ir import build_pwml_ir
    from t2pw.pwml.prefreeze_resolution import PrefreezeResolutionError, run_prefreeze_resolution

    resolved = Path(t2pw.__file__).resolve()
    if (expect_tree / PACKAGE_REL).resolve() not in resolved.parents:
        print(json.dumps({"error": "T2PW_MEASUREMENT_TREE_REFUSED", "t2pw_file": str(resolved)}))
        return 98
    results: Dict[str, Any] = {"t2pw_file": str(resolved), "legs": {}}
    for leg in json.loads(legs_file.read_text(encoding="utf-8")):
        payload = json.loads((REPO / leg).read_text(encoding="utf-8"))
        try:
            run_prefreeze_resolution(payload, strict_db=True, db_resolver=_DownDb(), name_index=None)
        except PrefreezeResolutionError as exc:
            results["legs"][leg] = {"stop": f"prefreeze:{exc.code}"}
            continue
        try:
            ir, report = build_pwml_ir(payload, db_resolver=_DownDb(), strict_db=True, name_index=None)
        except Exception as exc:  # noqa: BLE001 -- recorded, compared across trees
            results["legs"][leg] = {"stop": f"build:{type(exc).__name__}:{exc}"}
            continue
        results["legs"][leg] = {"ir": ir, "report": report}
    out_file.write_text(json.dumps(results, default=repr), encoding="utf-8")
    print(json.dumps({"t2pw_file": str(resolved), "legs": len(results["legs"])}))
    return 0


# -------------------------------------------------------------------------- parent
def _export_base_tree(rev: str) -> Path:
    out = subprocess.run(["git", "archive", "--format=tar", rev, PACKAGE_REL], cwd=REPO,
                         capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"git archive {rev} failed: {out.stderr.decode(errors='replace').strip()}")
    tree = Path(tempfile.mkdtemp(prefix="orch747_delta_base_"))
    with tarfile.open(fileobj=io.BytesIO(out.stdout)) as tar:
        tar.extractall(tree)
    return tree


def _run_child(tree: Path, legs_file: Path, out_file: Path) -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree / "src")
    env["T2PW_OFFLINE_CURATOR"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--build-legs",
                          "--expect-tree", str(tree), "--legs-file", str(legs_file),
                          "--out-file", str(out_file)],
                         cwd=tree, env=env, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"child failed in {tree} (exit {out.returncode}):\n{out.stdout}\n{out.stderr}")
    return json.loads(out_file.read_text(encoding="utf-8"))


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, indent=1, default=repr).encode()).hexdigest()


def _diff_paths(a: Any, b: Any, path: str, out: Set[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a) | set(b):
            if key not in a or key not in b or a[key] != b[key]:
                _diff_paths(a.get(key), b.get(key), f"{path}/{key}", out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.add(f"{path}")
            return
        for x, y in zip(a, b):
            if x != y:
                _diff_paths(x, y, f"{path}/[]", out)
        return
    out.add(path)


def _generalise(path: str) -> str:
    return re.sub(r"/\d+(?=/|$)", "/[]", path)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--build-legs", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--expect-tree", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--legs-file", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-file", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.build_legs:
        return _build_legs(Path(args.expect_tree), Path(args.legs_file), Path(args.out_file))

    # The pinned legs, read from the two test modules that pin them.
    sys.path.insert(0, str(REPO / "tests"))
    sys.path.insert(0, str(REPO / "src"))
    import test_compound_resolution_extraction as golden_mod  # noqa: E402
    import test_pwml_ir_duplicate_row_refusal as r3_mod  # noqa: E402

    legs: List[str] = sorted(set(golden_mod.GOLDEN) | {r3_mod.R3_CONTROL_LEG})
    work = Path(tempfile.mkdtemp(prefix="orch747_delta_"))
    legs_file = work / "legs.json"
    legs_file.write_text(json.dumps(legs), encoding="utf-8")

    sha = subprocess.run(["git", "rev-parse", args.rev], cwd=REPO, capture_output=True,
                         text=True).stdout.strip()
    base_tree = _export_base_tree(args.rev)
    base = _run_child(base_tree, legs_file, work / "base.json")
    tip = _run_child(REPO, legs_file, work / "tip.json")

    per_leg: Dict[str, Any] = {}
    unowned: Dict[str, List[str]] = {}
    moved = 0
    reasons: Counter = Counter()
    for leg in legs:
        b, t = base["legs"][leg], tip["legs"][leg]
        if "stop" in b or "stop" in t:
            per_leg[leg] = {"base": b.get("stop", "built"), "tip": t.get("stop", "built"),
                            "same_stop": b.get("stop") == t.get("stop")}
            continue
        changed: Set[str] = set()
        _diff_paths({"ir": b["ir"], "report": b["report"]}, {"ir": t["ir"], "report": t["report"]}, "", changed)
        paths = sorted({_generalise(p) for p in changed})
        outside = [p for p in paths if not p.startswith(OWNED_PREFIXES)]
        rows = t["ir"]["processes"]["interactions"]
        row_reasons = Counter((r.get("withheld") or {}).get("reason", "exportable") for r in rows)
        reasons.update(row_reasons)
        entry = {
            "interactions": len(rows),
            "by_outcome": dict(row_reasons),
            "base_ir_sha256": _sha(b["ir"]),
            "tip_ir_sha256": _sha(t["ir"]),
            "base_report_sha256": _sha(b["report"]),
            "tip_report_sha256": _sha(t["report"]),
            "ir_moved": _sha(b["ir"]) != _sha(t["ir"]),
            "report_moved": _sha(b["report"]) != _sha(t["report"]),
            "changed_paths": paths,
            "added_warning_codes": sorted(
                Counter(w.get("code") for w in t["report"]["warnings"]).items()
                - Counter(w.get("code") for w in b["report"]["warnings"]).items()
            ),
            "errors_unchanged": [e.get("code") for e in b["report"]["errors"]] == [e.get("code") for e in t["report"]["errors"]],
            "ok_unchanged": b["report"].get("ok") == t["report"].get("ok"),
        }
        if entry["ir_moved"] or entry["report_moved"]:
            moved += 1
        if outside:
            unowned[leg] = outside
        per_leg[leg] = entry

    confined = not unowned and all(
        e.get("errors_unchanged", True) and e.get("ok_unchanged", True) and e.get("same_stop", True)
        for e in per_leg.values()
    )
    summary = {
        "base_rev": args.rev,
        "base_sha": sha,
        "base_t2pw_file": base["t2pw_file"],
        "tip_t2pw_file": tip["t2pw_file"],
        "configuration": "run_prefreeze_resolution + build_pwml_ir, strict_db=True, db_resolver=_DownDb(), name_index=None (the R3 test's)",
        "legs": len(legs),
        "legs_moved": moved,
        "interaction_outcomes_on_tip": dict(reasons),
        "owned_prefixes": list(OWNED_PREFIXES),
        "paths_outside_owned_prefixes": unowned,
        "confined_to_interaction_export": confined,
        "per_leg": per_leg,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"base {args.rev} = {sha}; t2pw from {base['t2pw_file']}")
    print(f"tip  t2pw from {tip['t2pw_file']}")
    print(f"legs: {len(legs)}  moved: {moved}  interaction outcomes on tip: {dict(reasons)}")
    all_paths = Counter(p for e in per_leg.values() for p in e.get("changed_paths", []))
    for path, n in sorted(all_paths.items()):
        print(f"  {n:3d} legs  {path}")
    if unowned:
        print("PATHS OUTSIDE THE OWNED PREFIXES:")
        for leg, paths in unowned.items():
            print(f"  {leg}: {paths}")
    print(f"errors / ok / stops unchanged on every leg: {confined}")
    print(f"written: {args.out}")
    print("VERDICT: " + ("delta confined to the interaction export" if confined
                         else "NOT confined -- something else moved"))
    return 0 if confined else 1


if __name__ == "__main__":
    sys.exit(main())

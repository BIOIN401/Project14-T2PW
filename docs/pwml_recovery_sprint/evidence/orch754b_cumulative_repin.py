"""ORCH-754b -- the cumulative, deliberate re-pin of the two IR digest pins (merge rule 4).

The pins:

* ``tests/test_compound_resolution_extraction.py::test_build_pwml_ir_matches_the_pre_extraction_golden``
  -- ``GOLDEN``, one digest per committed leg over FIVE configurations (``_leg_digest``);
* ``tests/test_pwml_ir_duplicate_row_refusal.py::test_new_acceptance_a_non_colliding_leg_is_byte_identical_to_the_base``
  -- ``R3_BASE_IR_DIGEST`` / ``R3_BASE_REPORT_DIGEST`` on one leg, one configuration.

Both last passed at ``c9265fb8``; ORCH-747/749 (``70d05e03``) moved them and the re-pin was
owed; ORCH-754b moves them again. This measures the WHOLE move, ``c9265fb8`` -> working tree,
in one pass, for every pinned leg and every one of the five configurations:

1. builds ``(ir, report)`` on both trees, each in its own child that verifies which ``t2pw``
   it imported, with the test module's own ``_configs()`` and -- the part an out-of-pytest
   harness used to miss -- the module's autouse ``_no_live_db`` monkeypatch applied by hand,
   so configuration E sees ``harvest_forced_unavailable`` exactly as it does under pytest and
   nothing can reach a live database;
2. lists every JSON pointer that differs, generalised over list indices, and checks each lies
   under an interaction annotation: the interaction rows themselves, the geometry drawn for
   the ones PathWhiz can carry (locations, edges, the interaction visualizations, a complex
   endpoint's visualization) and the warnings / counts that report them. Separately it
   checks that every BIOLOGICAL part of the IR -- entities, reactions, transports,
   biological states, species, every non-interaction process visualization -- is identical,
   that ``errors`` and ``ok`` did not move, and that no configuration's stop moved;
3. computes ``_leg_digest`` out of pytest on the tip and compares it with the digests pytest
   itself reported (``--pytest-output``, the failing run's log), so the new pins are the
   authority's numbers and the instrument is checked against it rather than trusted.

The ``type="integer"`` side effect ORCH-747 recorded is in the WRITER's XML, not in the IR,
so it cannot reach these digests; that is checked here only in the sense that nothing outside
the interaction prefixes moved.

Usage::

    python docs/pwml_recovery_sprint/evidence/orch754b_cumulative_repin.py \\
        --pytest-output <log of the failing pinned run> [--rev c9265fb8] [--out FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Set

REPO = Path(__file__).resolve().parents[3]
PACKAGE_REL = "src/t2pw"
DEFAULT_OUT = Path(__file__).resolve().parent / "orch754b_cumulative_repin.json"

#: Where an interaction annotation may move an (ir, report) pair. ORCH-747's set, which
#: covers ORCH-749 (spelling only) and ORCH-754b (the withheld record) too.
OWNED_PREFIXES = (
    "/ir/processes/interactions",
    "/ir/locations",
    "/ir/edges",
    "/ir/process_visualizations",
    "/ir/protein_complex_visualizations",
    "/report/warnings",
    "/report/counts/locations",
    "/report/counts/edges",
    "/report/counts/process_visualizations",
    "/report/ir_validation/counts",
)
#: Must be byte-identical: the biology, and the verdicts.
BIOLOGICAL_IR_KEYS = ("entities", "species", "subcellular_locations", "biological_states", "pathway",
                      "cell_types", "tissues")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, indent=1, default=repr).encode()).hexdigest()


# --------------------------------------------------------------------------- child
def _child(expect_tree: Path, legs_file: Path, out_dir: Path) -> int:
    logging.disable(logging.CRITICAL)
    import t2pw
    import t2pw.mapping.map_ids as map_ids

    resolved = Path(t2pw.__file__).resolve()
    if (expect_tree / PACKAGE_REL).resolve() not in resolved.parents:
        print(json.dumps({"error": "T2PW_MEASUREMENT_TREE_REFUSED", "t2pw_file": str(resolved)}))
        return 98

    class _ForcedUnavailable:  # the test module's autouse ``_no_live_db``, verbatim in effect
        @classmethod
        def from_env(cls) -> Any:
            raise RuntimeError("harvest_forced_unavailable")

    map_ids.PathBankDbResolver = _ForcedUnavailable
    sys.path.insert(0, str(REPO / "tests"))
    import test_compound_resolution_extraction as golden  # noqa: E402
    from t2pw.pwml.ir import DuplicateNamedRowError, build_pwml_ir
    from t2pw.pwml.prefreeze_resolution import PrefreezeResolutionError, run_prefreeze_resolution

    digests: Dict[str, Any] = {}
    for i, leg in enumerate(json.loads(legs_file.read_text(encoding="utf-8"))):
        payload = json.loads((REPO / leg).read_text(encoding="utf-8"))
        digests[leg] = list(golden._leg_digest(payload))
        for name, kwargs in golden._configs():
            staged = deepcopy(payload)
            try:
                run_prefreeze_resolution(staged, strict_db=kwargs["strict_db"], db_resolver=kwargs["db_resolver"],
                                         name_index=kwargs["name_index"])
                ir, report = build_pwml_ir(staged, **kwargs)
                result: Dict[str, Any] = {"ir": ir, "report": report}
            except PrefreezeResolutionError as stop:
                result = {"stop": f"prefreeze:{stop.code}"}
            except DuplicateNamedRowError as refusal:
                result = {"stop": f"ir_refusal:{refusal.code}"}
            (out_dir / f"{i}__{name}.json").write_text(json.dumps(result, default=golden._nonjson), encoding="utf-8")
    (out_dir / "digests.json").write_text(json.dumps(digests), encoding="utf-8")
    print(json.dumps({"t2pw_file": str(resolved), "legs": len(digests)}))
    return 0


# -------------------------------------------------------------------------- parent
def _export_base_tree(rev: str) -> Path:
    # The package plus the offline name index ``_configs()`` asserts on (configurations A, C, D).
    out = subprocess.run(["git", "archive", "--format=tar", rev, PACKAGE_REL, "data/pathwhiz_id_db.json"],
                         cwd=REPO, capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"git archive {rev} failed: {out.stderr.decode(errors='replace').strip()}")
    tree = Path(tempfile.mkdtemp(prefix="orch754b_repin_base_"))
    with tarfile.open(fileobj=io.BytesIO(out.stdout)) as tar:
        tar.extractall(tree)
    return tree


def _run_child(tree: Path, legs_file: Path, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(tree / "src"), T2PW_OFFLINE_CURATOR="1", PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", "--expect-tree", str(tree),
                          "--legs-file", str(legs_file), "--out-dir", str(out_dir)],
                         cwd=tree, env=env, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"child failed in {tree} (exit {out.returncode}):\n{out.stdout}\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])["t2pw_file"]


def _diff_paths(a: Any, b: Any, path: str, out: Set[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a) | set(b):
            if key not in a or key not in b or a[key] != b[key]:
                _diff_paths(a.get(key), b.get(key), f"{path}/{key}", out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.add(path)
            return
        for x, y in zip(a, b):
            if x != y:
                _diff_paths(x, y, f"{path}/[]", out)
        return
    out.add(path)


def _pytest_actuals(log: Path) -> Dict[str, Dict[str, str]]:
    """``{leg: {"expected", "actual"}}`` from the failing GOLDEN test's own message."""
    text = log.read_text(encoding="utf-8", errors="replace")
    found = re.findall(r"E\s+(\S+final_mapped\.json)\s*\nE\s+expected ([0-9a-f]{64})\s*\nE\s+actual\s+([0-9a-f]{64})", text)
    return {leg: {"expected": e, "actual": a} for leg, e, a in found}


def _r3_actuals(log: Path) -> List[str]:
    text = log.read_text(encoding="utf-8", errors="replace")
    return re.findall(r"E\s+\+\s+([0-9a-f]{64})", text)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--rev", default="c9265fb8")
    ap.add_argument("--pytest-output", required=False)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--expect-tree", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--legs-file", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-dir", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return _child(Path(args.expect_tree), Path(args.legs_file), Path(args.out_dir))

    sys.path.insert(0, str(REPO / "tests"))
    sys.path.insert(0, str(REPO / "src"))
    import test_compound_resolution_extraction as golden_mod  # noqa: E402
    import test_pwml_ir_duplicate_row_refusal as r3_mod  # noqa: E402

    configs = [name for name, _ in golden_mod._configs()]
    legs: List[str] = sorted(set(golden_mod.GOLDEN) | {r3_mod.R3_CONTROL_LEG})
    work = Path(tempfile.mkdtemp(prefix="orch754b_repin_"))
    legs_file = work / "legs.json"
    legs_file.write_text(json.dumps(legs), encoding="utf-8")
    sha = subprocess.run(["git", "rev-parse", args.rev], cwd=REPO, capture_output=True, text=True).stdout.strip()
    base_file = _run_child(_export_base_tree(args.rev), legs_file, work / "base")
    tip_file = _run_child(REPO, legs_file, work / "tip")

    per_leg: Dict[str, Any] = {}
    path_legs: Counter = Counter()
    unowned: Dict[str, List[str]] = {}
    biology_moved: Dict[str, List[str]] = {}
    verdicts_moved: Dict[str, List[str]] = {}
    for i, leg in enumerate(legs):
        entry: Dict[str, Any] = {"configs": {}}
        leg_paths: Set[str] = set()
        for name in configs:
            b = json.loads((work / "base" / f"{i}__{name}.json").read_text(encoding="utf-8"))
            t = json.loads((work / "tip" / f"{i}__{name}.json").read_text(encoding="utf-8"))
            if "stop" in b or "stop" in t:
                entry["configs"][name] = {"base": b.get("stop", "built"), "tip": t.get("stop", "built")}
                if b.get("stop") != t.get("stop"):
                    verdicts_moved.setdefault(leg, []).append(f"{name}: stop {b.get('stop')} -> {t.get('stop')}")
                continue
            changed: Set[str] = set()
            _diff_paths(b, t, "", changed)
            paths = sorted({re.sub(r"/\d+(?=/|$)", "/[]", p) for p in changed})
            leg_paths.update(paths)
            outside = [p for p in paths if not p.startswith(OWNED_PREFIXES)]
            if outside:
                unowned.setdefault(leg, []).extend(f"{name}: {p}" for p in outside)
            bio = [k for k in BIOLOGICAL_IR_KEYS if b["ir"].get(k) != t["ir"].get(k)]
            for key in ("reactions", "transports", "reaction_coupled_transports", "sub_pathways"):
                if b["ir"]["processes"].get(key) != t["ir"]["processes"].get(key):
                    bio.append(f"processes/{key}")
            non_interaction = lambda ir: [v for v in ir.get("process_visualizations") or []  # noqa: E731
                                          if v.get("type") != "interaction_visualization"]
            if non_interaction(b["ir"]) != non_interaction(t["ir"]):
                bio.append("process_visualizations (non-interaction)")
            if bio:
                biology_moved.setdefault(leg, []).extend(f"{name}: {k}" for k in bio)
            if ([e.get("code") for e in b["report"]["errors"]] != [e.get("code") for e in t["report"]["errors"]]
                    or b["report"].get("ok") != t["report"].get("ok")):
                verdicts_moved.setdefault(leg, []).append(f"{name}: errors/ok")
            entry["configs"][name] = {"moved": bool(paths), "changed_paths": paths,
                                      "base_ir_sha256": _sha(b["ir"]), "tip_ir_sha256": _sha(t["ir"]),
                                      "base_report_sha256": _sha(b["report"]), "tip_report_sha256": _sha(t["report"])}
        per_leg[leg] = entry
        for p in leg_paths:
            path_legs[p] += 1

    base_digests = json.loads((work / "base" / "digests.json").read_text(encoding="utf-8"))
    tip_digests = json.loads((work / "tip" / "digests.json").read_text(encoding="utf-8"))
    golden_rows = {}
    for leg in sorted(golden_mod.GOLDEN):
        golden_rows[leg] = {
            "committed_pin": golden_mod.GOLDEN[leg],
            "base_out_of_pytest": base_digests[leg][0],
            "tip_out_of_pytest": tip_digests[leg][0],
            "moved": base_digests[leg][0] != tip_digests[leg][0],
        }
    instrument_agrees_at_base = all(r["committed_pin"] == r["base_out_of_pytest"] for r in golden_rows.values())

    pytest_check: Dict[str, Any] = {}
    if args.pytest_output:
        log = Path(args.pytest_output)
        actuals = _pytest_actuals(log)
        pytest_check = {
            "log": str(log),
            "legs_pytest_reported_moved": len(actuals),
            "legs_this_instrument_reports_moved": sum(r["moved"] for r in golden_rows.values()),
            "same_set": set(actuals) == {leg for leg, r in golden_rows.items() if r["moved"]},
            "pytest_actual_equals_out_of_pytest_tip": all(actuals[leg]["actual"] == golden_rows[leg]["tip_out_of_pytest"]
                                                         for leg in actuals),
            "pytest_expected_equals_committed_pin": all(actuals[leg]["expected"] == golden_rows[leg]["committed_pin"]
                                                       for leg in actuals),
            "r3_pytest_reported_new_digests": _r3_actuals(log),
        }
    r3 = per_leg[r3_mod.R3_CONTROL_LEG]["configs"]["B_dbdown_noindex_strict"]
    r3_check = {
        "committed_ir_pin": r3_mod.R3_BASE_IR_DIGEST, "committed_report_pin": r3_mod.R3_BASE_REPORT_DIGEST,
        "base_ir_sha256": r3["base_ir_sha256"], "base_report_sha256": r3["base_report_sha256"],
        "tip_ir_sha256": r3["tip_ir_sha256"], "tip_report_sha256": r3["tip_report_sha256"],
        "instrument_agrees_at_base": (r3["base_ir_sha256"], r3["base_report_sha256"])
        == (r3_mod.R3_BASE_IR_DIGEST, r3_mod.R3_BASE_REPORT_DIGEST),
        "changed_paths": r3["changed_paths"],
    }
    confined = not unowned and not biology_moved and not verdicts_moved
    summary = {
        "base_rev": args.rev, "base_sha": sha, "base_t2pw_file": base_file, "tip_t2pw_file": tip_file,
        "configurations": configs, "legs": len(legs),
        "golden": {"legs": len(golden_rows), "moved": sum(r["moved"] for r in golden_rows.values()),
                   "instrument_agrees_with_committed_pins_at_base": instrument_agrees_at_base},
        "pytest_cross_check": pytest_check,
        "r3": r3_check,
        "changed_paths_by_leg_count": dict(sorted(path_legs.items())),
        "paths_outside_interaction_prefixes": unowned,
        "biological_ir_moved": biology_moved,
        "errors_ok_or_stops_moved": verdicts_moved,
        "confined_to_interaction_annotations": confined,
        "golden_old_to_new": golden_rows,
        "per_leg": per_leg,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)  # ~900 per-configuration IR dumps; the summary keeps the hashes
    print(f"base {args.rev} = {sha}\n  base t2pw {base_file}\n  tip  t2pw {tip_file}")
    print(f"legs {len(legs)} x {len(configs)} configurations; GOLDEN moved {summary['golden']['moved']} of {len(golden_rows)}")
    print(f"instrument == committed GOLDEN at base: {instrument_agrees_at_base}; "
          f"== committed R3 pins at base: {r3_check['instrument_agrees_at_base']}")
    if pytest_check:
        print(f"pytest cross-check: {json.dumps({k: v for k, v in pytest_check.items() if k != 'log'})}")
    for p, n in sorted(path_legs.items()):
        print(f"  {n:3d} legs  {p}")
    print(f"outside interaction prefixes: {unowned or 'none'}")
    print(f"biological IR moved: {biology_moved or 'none'}")
    print(f"errors / ok / stops moved: {verdicts_moved or 'none'}")
    print(f"R3 control: IR {r3_check['base_ir_sha256'][:12]} -> {r3_check['tip_ir_sha256'][:12]}, "
          f"report {r3_check['base_report_sha256'][:12]} -> {r3_check['tip_report_sha256'][:12]}")
    print(f"written: {args.out}")
    print("VERDICT: " + ("the whole move is interaction annotation" if confined else "NOT confined"))
    return 0 if confined else 1


if __name__ == "__main__":
    sys.exit(main())

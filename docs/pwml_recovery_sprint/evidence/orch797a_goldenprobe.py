"""ORCH-797a: what moved in ``test_build_pwml_ir_matches_the_pre_extraction_golden`` on its one drifted leg.

Usage: python orch797a_goldenprobe.py --src <tree>/src --tests <worktree>/tests --out <file.json>
Runs the test's own five configurations (``_configs``) over the leg and writes, per configuration, the stop code or
the IR (``build_pwml_ir`` output, JSON). Run base and tip; diff the two files.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--leg", default="runs_verify/2026-08-24_1428/papers/PMC12444477/strict/final_mapped.json")
args = ap.parse_args()
sys.path.insert(0, str(Path(args.src).resolve()))
import t2pw  # noqa: E402

print("t2pw resolved:", Path(t2pw.__file__).resolve().parent)
sys.path.insert(1, str(Path(args.tests).resolve()))
import test_compound_resolution_extraction as T  # noqa: E402
from t2pw.pwml.ir import DuplicateNamedRowError, build_pwml_ir  # noqa: E402
from t2pw.pwml.prefreeze_resolution import PrefreezeResolutionError, run_prefreeze_resolution  # noqa: E402

payload = json.loads((Path(args.tests).resolve().parent / args.leg).read_text(encoding="utf-8"))
out = {}
for name, kwargs in T._configs():
    staged = copy.deepcopy(payload)
    try:
        report = run_prefreeze_resolution(staged, strict_db=kwargs["strict_db"], db_resolver=kwargs["db_resolver"],
                                          name_index=kwargs["name_index"])
        built = build_pwml_ir(staged, **kwargs)
        out[name] = {"prefreeze": report, "ir": list(built)}
    except (PrefreezeResolutionError, DuplicateNamedRowError) as stop:
        out[name] = {"stop": f"{type(stop).__name__}:{getattr(stop, 'code', '')}"}
Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str, ensure_ascii=False), encoding="utf-8")
print("written", args.out, {k: ("stop" if "stop" in v else "ir") for k, v in out.items()})

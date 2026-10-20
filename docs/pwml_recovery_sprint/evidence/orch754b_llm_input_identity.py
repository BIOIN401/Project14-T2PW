"""ORCH-754b repair round 1 -- every LLM-bound input is the base's, byte for byte (D-104, 2026-10-20).

Runs ``tests/test_pwml_interaction_protein_wrappers._llm_bound_digests`` -- the reaction
summary (read by the curator and the enrichment agent), the draft graph and QA report it is
built from, and the Stage-6 report + payload for a payload whose signed rows cannot be
wrapped (the report feeds the interactive curator's "Mapping Misses") -- over the same fixed
payloads under three trees, each in its own child interpreter that verifies every ``t2pw``
module it loaded came from the tree under measurement:

* ``base``  -- ``f9892cdf`` (a clean worktree), the reference the ruling names;
* ``prev``  -- ``d73e9bc8``, the tip the review rejected (expected to DIFFER: G9 for the test);
* ``tip``   -- this worktree.

The base digests are what ``LLM_BOUND_BASE_DIGESTS`` in the test file pins.

Usage::

    python orch754b_llm_input_identity.py --base-tree <dir> --prev-tree <dir> [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path(__file__).resolve().parent / "orch754b_llm_input_identity.json"


def _child(tree: Path) -> int:
    import t2pw  # first, so every later t2pw import resolves through this tree's package

    sys.path.insert(0, str(REPO / "tests"))
    import test_pwml_interaction_protein_wrappers as w  # noqa: E402

    digests = w._llm_bound_digests()
    foreign = sorted(
        str(getattr(m, "__file__", "")) for name, m in sys.modules.items()
        if name.startswith("t2pw") and getattr(m, "__file__", None)
        and (tree / "src" / "t2pw").resolve() not in Path(m.__file__).resolve().parents
    )
    print(json.dumps({"t2pw_file": str(Path(t2pw.__file__).resolve()), "foreign_t2pw_modules": foreign,
                      "digests": digests}))
    return 0


def _run(tree: Path) -> dict:
    env = dict(os.environ, PYTHONPATH=str(tree / "src"), T2PW_OFFLINE_CURATOR="1", PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(tree)],
                         cwd=str(tree), env=env, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"child failed in {tree}:\n{out.stdout}\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--base-tree")
    ap.add_argument("--prev-tree")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    if args.child:
        return _child(Path(args.child))

    runs = {"base": _run(Path(args.base_tree)), "prev": _run(Path(args.prev_tree)), "tip": _run(REPO)}
    clean = all(not r["foreign_t2pw_modules"] for r in runs.values())
    tip_equals_base = runs["tip"]["digests"] == runs["base"]["digests"]
    base = runs["base"]["digests"]
    prev_differs = {k: ([f for f in v if v[f] != base[k].get(f)] if isinstance(v, dict) else [k])
                    for k, v in runs["prev"]["digests"].items() if v != base.get(k)}
    record = {
        "trees": {k: v["t2pw_file"] for k, v in runs.items()},
        "every_module_from_its_own_tree": clean,
        "tip_equals_base": tip_equals_base,
        "prev_differs_from_base": {k: v for k, v in prev_differs.items() if v},
        "base_digests": runs["base"]["digests"],
        "tip_digests": runs["tip"]["digests"],
        "prev_digests": runs["prev"]["digests"],
    }
    Path(args.out).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("trees", "every_module_from_its_own_tree", "tip_equals_base",
                                             "prev_differs_from_base")}, indent=1))
    print("VERDICT: " + ("tip LLM inputs == base; d73e9bc8 differed" if clean and tip_equals_base and record["prev_differs_from_base"]
                         else "FAILED"))
    return 0 if clean and tip_equals_base else 1


if __name__ == "__main__":
    sys.exit(main())

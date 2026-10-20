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

Round 2 adds the WRAPPED case. The tip runs first and writes the payload its Stage 6 produced
for the fixture; the base and ``d73e9bc8`` children read THAT payload for the
``wrapped_post_stage6`` texts, so every tree reads the same post-Stage-6 payload and the only
permitted difference -- the wrapper itself -- is held constant. The Stage-6 report and the
curator's Mapping Misses for that fixture are each tree's own, from the same input.

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
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path(__file__).resolve().parent / "orch754b_llm_input_identity.json"


def _child(tree: Path, post_stage6: str, write_post_stage6: str) -> int:
    import t2pw  # first, so every later t2pw import resolves through this tree's package

    sys.path.insert(0, str(REPO / "tests"))
    import test_pwml_interaction_protein_wrappers as w  # noqa: E402

    if write_post_stage6:
        Path(write_post_stage6).write_text(json.dumps(w._wrapped_post_stage6()), encoding="utf-8")
    shared = json.loads(Path(post_stage6).read_text(encoding="utf-8")) if post_stage6 else None
    digests = w._llm_bound_digests(shared)
    foreign = sorted(
        str(getattr(m, "__file__", "")) for name, m in sys.modules.items()
        if name.startswith("t2pw") and getattr(m, "__file__", None)
        and (tree / "src" / "t2pw").resolve() not in Path(m.__file__).resolve().parents
    )
    print(json.dumps({"t2pw_file": str(Path(t2pw.__file__).resolve()), "foreign_t2pw_modules": foreign,
                      "digests": digests}))
    return 0


def _run(tree: Path, post_stage6: str = "", write_post_stage6: str = "") -> dict:
    env = dict(os.environ, PYTHONPATH=str(tree / "src"), T2PW_OFFLINE_CURATOR="1", PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(tree),
                          "--post-stage6", post_stage6, "--write-post-stage6", write_post_stage6],
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
    ap.add_argument("--post-stage6", default="", help=argparse.SUPPRESS)
    ap.add_argument("--write-post-stage6", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return _child(Path(args.child), args.post_stage6, args.write_post_stage6)

    shared = Path(tempfile.mkdtemp(prefix="orch754b_ps6_")) / "wrapped_post_stage6.json"
    tip = _run(REPO, write_post_stage6=str(shared))
    runs = {"base": _run(Path(args.base_tree), post_stage6=str(shared)),
            "prev": _run(Path(args.prev_tree), post_stage6=str(shared)), "tip": tip}
    wrapped = json.loads(shared.read_text(encoding="utf-8"))
    clean = all(not r["foreign_t2pw_modules"] for r in runs.values())
    tip_equals_base = runs["tip"]["digests"] == runs["base"]["digests"]
    base = runs["base"]["digests"]
    prev_differs = {k: ([f for f in v if v[f] != base[k].get(f)] if isinstance(v, dict) else [k])
                    for k, v in runs["prev"]["digests"].items() if v != base.get(k)}
    record = {
        "wrapped_post_stage6_has_generated_wrappers": sorted(
            r["name"] for r in wrapped["entities"]["protein_complexes"] if r.get("generated")),
        "trees": {k: v["t2pw_file"] for k, v in runs.items()},
        "every_module_from_its_own_tree": clean,
        "tip_equals_base": tip_equals_base,
        "prev_differs_from_base": {k: v for k, v in prev_differs.items() if v},
        "base_digests": runs["base"]["digests"],
        "tip_digests": runs["tip"]["digests"],
        "prev_digests": runs["prev"]["digests"],
    }
    Path(args.out).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("wrapped_post_stage6_has_generated_wrappers", "trees",
                                             "every_module_from_its_own_tree", "tip_equals_base",
                                             "prev_differs_from_base")}, indent=1))
    print("VERDICT: " + ("tip LLM inputs == base; d73e9bc8 differed" if clean and tip_equals_base and record["prev_differs_from_base"]
                         else "FAILED"))
    return 0 if clean and tip_equals_base else 1


if __name__ == "__main__":
    sys.exit(main())

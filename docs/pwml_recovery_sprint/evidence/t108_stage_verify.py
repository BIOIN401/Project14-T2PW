"""T-108 staged-directory verification — read-only, run BEFORE launch.

Proves, against the *exact* staged run directory and nothing else:

 1. ``find_resumable()`` returns that exact path, so continuing without
    ``--fresh`` cannot silently pick a different directory;
 2. the plan holds 20 pairs, **20 pending, 0 legs present** — the discriminator
    that makes ``--fresh`` unnecessary AND wrong here (T-107 ledger: the
    runner's own "rerun the same command WITHOUT --stage-only" hint still
    carries ``--fresh``, which would discard the verified staging);
 3. no live pipeline leg ran during the stage-only preflight — no manifest row,
    no leg directory, no RESULT.txt anywhere;
 4. the gold set the run will be scored against is the pinned blob;
 5. the per-leg ceiling resolves to **3600 s with NO override**, so the manifest
    will carry ``leg_timeout_overridden: false`` and there is no empty
    ``leg_timeout_override_reason`` for PRODUCT_CONTRACT § 9 to catch.

Point 5 is verified *before* launch, as T108-READINESS § 2.1 requires, rather
than read out of the manifest afterwards.

Usage:  t108_stage_verify.py <repo> <run-dir-relative>
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys


EXPECTED_GOLD_BLOB = "36f4b7b690b577f72882c3045ca6728d1ec8d9d1"
EXPECTED_PAIRS = 20
EXPECTED_LEG_TIMEOUT = 3600.0


def main() -> int:
    repo = os.path.abspath(sys.argv[1])
    rel = sys.argv[2]
    run_dir = os.path.abspath(os.path.join(repo, rel))
    sys.path.insert(0, os.path.join(repo, "src"))

    problems: list[str] = []

    print("== target ==")
    print(f"repo    : {repo}")
    print(f"run dir : {run_dir}")
    print(f"exists  : {os.path.isdir(run_dir)}")
    if not os.path.isdir(run_dir):
        print("REFUSED: staged directory absent")
        return 1

    # ---------------------------------------------------------------- 1. resumable
    print()
    print("== 1. find_resumable() ==")
    from t2pw.batch import runner  # noqa: E402

    resumable = runner.find_resumable(os.path.join(repo, "runs_verify"))
    got = os.path.abspath(str(resumable)) if resumable else None
    print(f"returned : {got}")
    print(f"expected : {run_dir}")
    same = got == run_dir
    print(f"MATCH    : {same}")
    if not same:
        problems.append(f"find_resumable returned {got!r}, not the verified {run_dir!r}")

    # ---------------------------------------------------------------- 2. plan shape
    print()
    print("== 2. plan pairs / pending / legs present ==")
    plan_path = os.path.join(run_dir, "plan.json")
    plan = json.load(open(plan_path, encoding="utf-8"))
    papers = plan.get("papers") or plan.get("entries") or []
    modes = plan.get("modes") or []
    if isinstance(modes, str):
        modes = [m.strip() for m in modes.split(",") if m.strip()]
    pairs = plan.get("pairs")
    if pairs is None:
        pairs = [(p, m) for p in papers for m in modes]
    n_pairs = len(pairs)
    print(f"papers in plan : {len(papers)}")
    print(f"modes in plan  : {modes}")
    print(f"plan pairs     : {n_pairs}")
    if n_pairs != EXPECTED_PAIRS:
        problems.append(f"plan holds {n_pairs} pairs, expected {EXPECTED_PAIRS}")

    manifest = os.path.join(run_dir, "manifest.jsonl")
    recorded = 0
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as fh:
            recorded = sum(1 for line in fh if line.strip())
    print(f"manifest rows (already recorded) : {recorded}")
    print(f"pending                          : {n_pairs - recorded}")
    if recorded != 0:
        problems.append(f"{recorded} manifest row(s) already recorded — this is not a clean staging")

    papers_dir = os.path.join(run_dir, "papers")
    leg_dirs: list[str] = []
    results: list[str] = []
    if os.path.isdir(papers_dir):
        for slug in sorted(os.listdir(papers_dir)):
            slug_path = os.path.join(papers_dir, slug)
            if not os.path.isdir(slug_path):
                continue
            for mode in sorted(os.listdir(slug_path)):
                mode_path = os.path.join(slug_path, mode)
                if os.path.isdir(mode_path):
                    leg_dirs.append(f"{slug}/{mode}")
                    if os.path.isfile(os.path.join(mode_path, "RESULT.txt")):
                        results.append(f"{slug}/{mode}")
    print(f"leg directories present : {len(leg_dirs)}")
    print(f"RESULT.txt present      : {len(results)}")
    if leg_dirs:
        problems.append(f"{len(leg_dirs)} leg director(ies) already exist: {leg_dirs[:5]}")
    if results:
        problems.append(f"{len(results)} RESULT.txt already exist: {results[:5]}")

    # -------------------------------------------------------- 3. all 20 legs expected
    print()
    print("== 3. expected strict/research legs ==")
    slugs = []
    for p in papers:
        if isinstance(p, dict):
            slugs.append(p.get("slug") or p.get("paper_id") or p.get("pmcid") or "?")
        else:
            slugs.append(str(p))
    print(f"papers : {len(slugs)}")
    missing_mode = [m for m in ("strict", "research") if m not in modes]
    print(f"modes  : {modes}  missing={missing_mode or 'none'}")
    if missing_mode:
        problems.append(f"plan is missing mode(s) {missing_mode}")
    expected_legs = sorted(f"{s}/{m}" for s in slugs for m in ("strict", "research"))
    print(f"expected legs : {len(expected_legs)}")
    for leg in expected_legs:
        print(f"  {leg}")
    if len(expected_legs) != EXPECTED_PAIRS:
        problems.append(f"{len(expected_legs)} expected legs, not {EXPECTED_PAIRS}")

    # ---------------------------------------------------------------- 4. gold hash
    print()
    print("== 4. gold set identity ==")
    gold_rel = "src/t2pw/bench/gold/pinned_v1.json"
    gold_abs = os.path.join(repo, gold_rel)
    blob = subprocess.run(
        ["git", "hash-object", gold_rel],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()
    head_blob = subprocess.run(
        ["git", "rev-parse", f"HEAD:{gold_rel}"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()
    sha = hashlib.sha256(open(gold_abs, "rb").read()).hexdigest()
    gold = json.load(open(gold_abs, encoding="utf-8"))
    cases = gold.get("cases", [])
    print(f"working-tree blob : {blob}")
    print(f"HEAD blob         : {head_blob}")
    print(f"expected blob     : {EXPECTED_GOLD_BLOB}")
    print(f"sha256            : {sha}")
    print(f"gold version      : {gold.get('version')}")
    print(f"gold cases        : {len(cases)}")
    if blob != EXPECTED_GOLD_BLOB:
        problems.append(f"gold working-tree blob {blob} != pinned {EXPECTED_GOLD_BLOB}")
    if head_blob != EXPECTED_GOLD_BLOB:
        problems.append(f"gold HEAD blob {head_blob} != pinned {EXPECTED_GOLD_BLOB}")
    if blob != head_blob:
        problems.append("gold differs between working tree and HEAD — uncommitted gold edit")

    n_complete = sum(1 for c in cases if c.get("supported_reactions_complete") is True)
    n_ceiling = sum(1 for c in cases if c.get("max_retained_reactions") is not None)
    print(f"supported_reactions_complete TRUE : {n_complete} / {len(cases)}")
    print(f"max_retained_reactions set        : {n_ceiling} / {len(cases)}")

    # ------------------------------------------------------- 5. leg ceiling, no override
    print()
    print("== 5. per-leg ceiling resolves with NO override ==")
    from t2pw.pipeline import deadline as dl  # noqa: E402

    print(f"deadline.LEG_TIMEOUT_SECONDS        : {dl.LEG_TIMEOUT_SECONDS}")
    print(f"runner.DEFAULT_PAPER_TIMEOUT        : {runner.DEFAULT_PAPER_TIMEOUT}")
    ceiling = dl._ceiling(EXPECTED_LEG_TIMEOUT)
    record = ceiling.to_dict()
    print(f"_ceiling({EXPECTED_LEG_TIMEOUT}).to_dict()      : {record}")
    print(f"child deadline (leg - 120 grace)    : {dl.child_deadline_seconds(EXPECTED_LEG_TIMEOUT)}")
    if record.get("leg_timeout_overridden") is not False:
        problems.append(f"leg_timeout_overridden is {record.get('leg_timeout_overridden')!r}, expected False")
    if "leg_timeout_override_reason" in record:
        problems.append("leg_timeout_override_reason present — there must be no override at 3600 s")
    if float(record.get("leg_timeout_seconds", 0)) != EXPECTED_LEG_TIMEOUT:
        problems.append(f"leg_timeout_seconds is {record.get('leg_timeout_seconds')!r}")
    if float(runner.DEFAULT_PAPER_TIMEOUT) != EXPECTED_LEG_TIMEOUT:
        problems.append(f"runner default is {runner.DEFAULT_PAPER_TIMEOUT}, not {EXPECTED_LEG_TIMEOUT}")
    if float(dl.LEG_TIMEOUT_SECONDS) != EXPECTED_LEG_TIMEOUT:
        problems.append(f"deadline default is {dl.LEG_TIMEOUT_SECONDS}, not {EXPECTED_LEG_TIMEOUT}")

    # ------------------------------------------------------------------- verdict
    print()
    print("== VERDICT ==")
    if problems:
        print(f"T108_STAGE_VERIFY: FAIL ({len(problems)})")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("T108_STAGE_VERIFY: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

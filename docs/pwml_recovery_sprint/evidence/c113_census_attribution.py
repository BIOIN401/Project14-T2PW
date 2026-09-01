"""C-113: attribute the census movement caused by the F-150 half-1 gold edit, PER LEG.

The two pins that move (`with_matched_forbidden` in test 13, and the
`affected_papers` set-equality in test 10) are pinned *literals*. A literal that
is read off a pytest failure message and pasted back in is a fitted pin: it will
absorb the next real regression in silence. So this probe measures the population
INDEPENDENTLY of the failing asserts.

It walks the SAME leg population the two tests walk -- `git ls-files
*quarantine_report.json`, filtered to legs whose paper is in the gold set and
whose coverage block `contract_accepted_coverage` accepts -- TWICE:

  arm PRE   the gold blob as it stands at the base of this card
            (`aee8cb4f1da3d417f36206407867585622b741c0`)
  arm POST  the gold blob with F-150 half 1 applied
            (`36f4b7b690b577f72882c3045ca6728d1ec8d9d1`)

Everything else is held fixed: same tree, same interpreter, same
`contract_accepted_coverage`, same leg list, same iteration order. The only
difference between the arms is the gold file handed to `load_gold_set`.

It then prints, BY LEG PATH and BY RUN TREE (`runs/...` and `runs_verify/...` are
both live, so "the pinned run" is ambiguous and every row is named in full):

  * every leg that carries a matched forbidden term in POST but not in PRE,
  * every leg that puts a paper into `affected_papers` in POST but not in PRE,
  * the per-leg `excluded_count` delta that drives `withheld`,
  * and the four scalar quantities the two tests pin, in both arms.

`withheld` is measured here, not assumed: test 10's `:422` set-equality aborts
before `assert withheld == 97` ever runs, so at the base of this card nobody
knows whether it moved. This probe answers that.

It asserts nothing and changes nothing. It is a measurement.

Usage::

    <venv python> c113_census_attribution.py <repo-root> <pre-edit-gold> <post-edit-gold>
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
PRE_GOLD = Path(sys.argv[2]).resolve()
POST_GOLD = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(REPO / "src"))

from t2pw.bench.acceptance import contract_accepted_coverage        # noqa: E402
from t2pw.bench.goldset import load_gold_set                        # noqa: E402

#: Kept identical to the tuple in tests/test_c102_coverage_denominator.py.
F132_PAPERS = (
    "PMC12096016", "PMC12312563", "PMC12444477",
    "PMC12452463", "PMC12782028", "PMC12856317",
)


def git_blob_sha(path: Path) -> str:
    """The git blob id of a file, so each arm names the gold it actually read."""

    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def run_tree_of(rel: str) -> str:
    """`runs/<stamp>` or `runs_verify/<stamp>` -- the run tree, named in full."""

    parts = Path(rel).parts
    return "/".join(parts[:2]) if len(parts) > 2 else "?"


listed = subprocess.run(
    ["git", "ls-files", "*quarantine_report.json"],
    cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", check=True,
)
PATHS = sorted(line.strip() for line in listed.stdout.splitlines() if line.strip())

print(f"repo                : {REPO}")
print(f"tracked quarantine_report.json (the `paths` both tests build) : {len(PATHS)}")
print(f"arm PRE  gold       : {PRE_GOLD}")
print(f"         blob       : {git_blob_sha(PRE_GOLD)}")
print(f"arm POST gold       : {POST_GOLD}")
print(f"         blob       : {git_blob_sha(POST_GOLD)}")


def walk(gold_path: Path) -> dict:
    """Reproduce, exactly, the loops in tests 10 and 13 -- minus their asserts."""

    gold = {case.paper_id: case for case in load_gold_set(gold_path).cases}

    legs = 0
    withheld = 0
    with_matched_forbidden = 0
    affected_papers: dict[str, int] = {}
    cleared: list[str] = []
    # per-leg detail, keyed by the tracked path so the two arms line up exactly
    per_leg: dict[str, dict] = {}

    for rel in PATHS:
        leg_dir = (REPO / rel).parent
        case = gold.get(leg_dir.parent.name)
        if case is None:
            continue
        coverage = json.load(io.open(REPO / rel, encoding="utf-8")).get("coverage") or {}
        out = contract_accepted_coverage(case, coverage)
        if out is None:
            continue
        legs += 1
        matched_forbidden = any(e["matched_in_raw"] for e in out["excluded_terms"])
        if out["excluded_count"]:
            affected_papers[case.paper_id] = affected_papers.get(case.paper_id, 0) + 1
            withheld += out["excluded_count"]
        if out["cleared_by_reconciliation"]:
            cleared.append(f"{case.paper_id}:{leg_dir.name}")
        if matched_forbidden:
            with_matched_forbidden += 1
        per_leg[rel] = {
            "paper": case.paper_id,
            "mode": leg_dir.name,
            "run_tree": run_tree_of(rel),
            "excluded_count": out["excluded_count"],
            "excluded_terms": [e["term"] for e in out["excluded_terms"]],
            "matched_terms": [e["term"] for e in out["excluded_terms"] if e["matched_in_raw"]],
            "matched_forbidden": matched_forbidden,
            "raw_matched": out["raw_matched"],
            "accepted_matched": out["accepted_matched"],
            "raw_denominator": out["raw_denominator"],
            "accepted_denominator": out["accepted_denominator"],
        }

    return {
        "legs": legs,
        "checked": legs,          # test 13's `checked` is the same loop body
        "withheld": withheld,
        "with_matched_forbidden": with_matched_forbidden,
        "affected_papers": affected_papers,
        "cleared": cleared,
        "per_leg": per_leg,
    }


PRE = walk(PRE_GOLD)
POST = walk(POST_GOLD)

print("\n" + "=" * 78)
print("THE FOUR PINNED SCALARS, IN BOTH ARMS")
print("=" * 78)
rows = [
    ("test 10  legs                   (pinned == 72)", "legs"),
    ("test 13  checked                (pinned == 72)", "checked"),
    ("test 10  withheld               (pinned == 97)", "withheld"),
    ("test 13  with_matched_forbidden (pinned == 26)", "with_matched_forbidden"),
]
for label, key in rows:
    moved = "MOVED" if PRE[key] != POST[key] else "unmoved"
    print(f"  {label:48s}  PRE {PRE[key]:>4}   POST {POST[key]:>4}   {moved}")
print(f"  {'test 10  cleared                (pinned == [])':48s}"
      f"  PRE {PRE['cleared']}   POST {POST['cleared']}")

print("\n  affected_papers PRE  : "
      f"{dict(sorted(PRE['affected_papers'].items()))}")
print("  affected_papers POST : "
      f"{dict(sorted(POST['affected_papers'].items()))}")
print(f"  outside F132_PAPERS  PRE  : {sorted(set(PRE['affected_papers']) - set(F132_PAPERS))}")
print(f"  outside F132_PAPERS  POST : {sorted(set(POST['affected_papers']) - set(F132_PAPERS))}")
print(f"  F132_PAPERS all present   : PRE {set(F132_PAPERS) <= set(PRE['affected_papers'])}"
      f"   POST {set(F132_PAPERS) <= set(POST['affected_papers'])}")

print("\n" + "=" * 78)
print("PER-LEG ATTRIBUTION -- every leg whose classification differs between arms")
print("=" * 78)

all_rels = sorted(set(PRE["per_leg"]) | set(POST["per_leg"]))
newly_matched: list[str] = []
newly_affected: list[str] = []
withheld_delta_legs: list[str] = []

for rel in all_rels:
    a = PRE["per_leg"].get(rel)
    b = POST["per_leg"].get(rel)
    if a is None or b is None:
        print(f"\n  LEG PRESENT IN ONLY ONE ARM (this would be a defect): {rel}")
        continue
    if a == b:
        continue
    print(f"\n  {rel}")
    print(f"      run tree              : {b['run_tree']}")
    print(f"      paper : mode          : {b['paper']} : {b['mode']}")
    for field in ("excluded_count", "excluded_terms", "matched_terms",
                  "matched_forbidden", "raw_matched", "accepted_matched",
                  "raw_denominator", "accepted_denominator"):
        if a[field] != b[field]:
            print(f"      {field:22s}: PRE {a[field]!r}  ->  POST {b[field]!r}")
    if b["matched_forbidden"] and not a["matched_forbidden"]:
        newly_matched.append(rel)
    if b["excluded_count"] and not a["excluded_count"]:
        newly_affected.append(rel)
    if a["excluded_count"] != b["excluded_count"]:
        withheld_delta_legs.append(rel)

print("\n" + "=" * 78)
print("THE ARITHMETIC")
print("=" * 78)

print(f"\nwith_matched_forbidden : {PRE['with_matched_forbidden']} -> "
      f"{POST['with_matched_forbidden']}   "
      f"(+{POST['with_matched_forbidden'] - PRE['with_matched_forbidden']})")
print(f"  legs that NEWLY carry a matched forbidden term : {len(newly_matched)}")
for rel in newly_matched:
    b = POST["per_leg"][rel]
    print(f"    {rel}")
    print(f"        run tree {b['run_tree']}  paper {b['paper']}  mode {b['mode']}"
          f"  matched {b['matched_terms']}")
lost = [r for r in all_rels
        if r in PRE["per_leg"] and r in POST["per_leg"]
        and PRE["per_leg"][r]["matched_forbidden"]
        and not POST["per_leg"][r]["matched_forbidden"]]
print(f"  legs that STOP carrying one : {len(lost)}  {lost}")

per_run_new: dict[str, int] = {}
for rel in newly_matched:
    tree = POST["per_leg"][rel]["run_tree"]
    per_run_new[tree] = per_run_new.get(tree, 0) + 1
print("  new matched-forbidden legs, by run tree:")
for tree in sorted(per_run_new):
    print(f"    {tree:44s} +{per_run_new[tree]}")

print(f"\nwithheld : {PRE['withheld']} -> {POST['withheld']}   "
      f"(+{POST['withheld'] - PRE['withheld']})")
print(f"  legs whose excluded_count changed : {len(withheld_delta_legs)}")
for rel in withheld_delta_legs:
    a, b = PRE["per_leg"][rel], POST["per_leg"][rel]
    print(f"    {rel}  {a['excluded_count']} -> {b['excluded_count']}"
          f"   (+{b['excluded_count'] - a['excluded_count']})")

print("\naffected_papers :")
gained = set(POST["affected_papers"]) - set(PRE["affected_papers"])
droppd = set(PRE["affected_papers"]) - set(POST["affected_papers"])
print(f"  papers gained : {sorted(gained)}")
print(f"  papers lost   : {sorted(droppd)}")
for paper in sorted(gained):
    print(f"  legs that put {paper} into affected_papers:")
    for rel in sorted(POST["per_leg"]):
        b = POST["per_leg"][rel]
        if b["paper"] == paper and b["excluded_count"]:
            a = PRE["per_leg"].get(rel, {})
            print(f"    {rel}")
            print(f"        run tree {b['run_tree']}  mode {b['mode']}"
                  f"  excluded_count PRE {a.get('excluded_count')} -> POST {b['excluded_count']}")
            print(f"        excluded_terms POST {b['excluded_terms']}")

print("\n" + "=" * 78)
print("WHAT THE PINS MUST BECOME (measured, not read off a failure message)")
print("=" * 78)
print(f"  assert legs == {POST['legs']}")
print(f"  assert checked == {POST['checked']}")
print(f"  assert withheld == {POST['withheld']}")
print(f"  assert with_matched_forbidden == {POST['with_matched_forbidden']}")
print("  assert set(affected_papers) - set(F132_PAPERS) == "
      f"{sorted(set(POST['affected_papers']) - set(F132_PAPERS))}")
print(f"  assert cleared == {POST['cleared']}")

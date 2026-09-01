"""REV-113: INDEPENDENT re-derivation of the census movement C-113 attributes to F-150 half 1.

Written by the reviewer, from `tests/test_c102_coverage_denominator.py` itself, NOT
adapted from `c113_census_attribution.py`. Its job is to answer REV-113 section 3
without trusting the author's probe or the author's log:

  * do the three legs the diff names actually exist, belong to PMC12180156, and
    carry a MATCHED forbidden term under the post-edit gold and NOT under the
    pre-edit gold?
  * is `withheld` really 97 -> 100, MEASURED by running the whole loop to
    completion in both arms rather than back-computed as "+1 per moved leg"?
  * is the FOURTH leg the author volunteered -- 2026-08-24_1402 research, which
    draws the ALAS2 ENZYME in the same Greek spelling -- genuinely NOT in the
    moved set, and is that because `forbidden_match` refuses containment?

Two differences from the author's probe, both deliberate:

  1. Both arms are materialised with `git cat-file blob <sha>`, so each arm's
     bytes ARE the reviewed blob bytes and the sha1 printed below is the GIT
     BLOB ID. (The author's probe hashed the CRLF working-tree file and so
     printed 2b2c7931... for an arm whose git blob is 36f4b7b6...; content
     identical, provenance line misleading.)
  2. `withheld` is accumulated by the same full loop in both arms and the two
     totals are printed side by side, so the number is a measurement and not
     an arithmetic consequence of the per-leg diff.

Asserts nothing about the verdict. It is a measurement.

Usage::

    <venv python> rev113_attribution_rederive.py <repo-root> <pre-sha> <post-sha> <scratch-dir>
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
PRE_SHA = sys.argv[2]
POST_SHA = sys.argv[3]
SCRATCH = Path(sys.argv[4]).resolve()
SCRATCH.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "src"))

from t2pw.bench.acceptance import (            # noqa: E402
    contract_accepted_coverage,
    forbidden_coverage_match,
)
from t2pw.bench.goldset import load_gold_set   # noqa: E402

F132_PAPERS = (
    "PMC12096016", "PMC12312563", "PMC12444477",
    "PMC12452463", "PMC12782028", "PMC12856317",
)

#: The three legs the diff's comments name. Re-derived, not assumed.
CLAIMED = (
    "runs_verify/2026-08-21_2239/papers/PMC12180156/research/quarantine_report.json",
    "runs_verify/2026-08-21_2239/papers/PMC12180156/strict/quarantine_report.json",
    "runs_verify/2026-08-28_1816/papers/PMC12180156/research/quarantine_report.json",
)
#: The fourth leg the author volunteered as a NON-mover.
FOURTH = "runs_verify/2026-08-24_1402/papers/PMC12180156/research/quarantine_report.json"


def materialise(sha: str, name: str) -> Path:
    """Write a git blob to disk BYTE FOR BYTE, so the arm is provably that blob."""

    blob = subprocess.run(
        ["git", "cat-file", "blob", sha],
        cwd=str(REPO), capture_output=True, check=True,
    ).stdout
    path = SCRATCH / name
    path.write_bytes(blob)
    return path


def blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


listed = subprocess.run(
    ["git", "ls-files", "*quarantine_report.json"],
    cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", check=True,
)
PATHS = sorted(line.strip() for line in listed.stdout.splitlines() if line.strip())

PRE_PATH = materialise(PRE_SHA, "rev113_gold_pre.json")
POST_PATH = materialise(POST_SHA, "rev113_gold_post.json")

print("REV-113 INDEPENDENT RE-DERIVATION")
print(f"repo                         : {REPO}")
print(f"tracked quarantine_report.json (the population both tests walk) : {len(PATHS)}")
print(f"arm PRE   requested sha      : {PRE_SHA}")
print(f"          materialised blob  : {blob_id(PRE_PATH)}   MATCH={blob_id(PRE_PATH) == PRE_SHA}")
print(f"arm POST  requested sha      : {POST_SHA}")
print(f"          materialised blob  : {blob_id(POST_PATH)}   MATCH={blob_id(POST_PATH) == POST_SHA}")

runs_prefixes = sorted({p.split("/")[0] for p in PATHS})
print(f"top-level dirs contributing legs : {runs_prefixes}")


def walk(gold_path: Path) -> dict:
    """The loop bodies of tests 10 and 13, verbatim, with their asserts removed."""

    gold = {case.paper_id: case for case in load_gold_set(gold_path).cases}

    legs = 0
    withheld = 0
    with_matched_forbidden = 0
    affected: dict[str, int] = {}
    cleared: list[str] = []
    detail: dict[str, dict] = {}

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
        if out["excluded_count"]:
            affected[case.paper_id] = affected.get(case.paper_id, 0) + 1
            withheld += out["excluded_count"]
        if out["cleared_by_reconciliation"]:
            cleared.append(f"{case.paper_id}:{leg_dir.name}")
        if any(e["matched_in_raw"] for e in out["excluded_terms"]):
            with_matched_forbidden += 1
        detail[rel] = {
            "paper": case.paper_id,
            "mode": leg_dir.name,
            "excluded_count": out["excluded_count"],
            "excluded_terms": [e["term"] for e in out["excluded_terms"]],
            "matched": any(e["matched_in_raw"] for e in out["excluded_terms"]),
        }

    return {
        "legs": legs, "withheld": withheld,
        "with_matched_forbidden": with_matched_forbidden,
        "affected": affected, "cleared": cleared, "detail": detail,
    }


PRE = walk(PRE_PATH)
POST = walk(POST_PATH)

print("\n" + "=" * 78)
print("THE SCALARS, EACH ACCUMULATED BY A COMPLETE LOOP IN ITS OWN ARM")
print("=" * 78)
for key in ("legs", "withheld", "with_matched_forbidden"):
    tag = "MOVED" if PRE[key] != POST[key] else "unmoved"
    print(f"  {key:24s}  PRE {PRE[key]:>5}   POST {POST[key]:>5}   {tag}")
print(f"  {'cleared':24s}  PRE {PRE['cleared']}   POST {POST['cleared']}")
print(f"  affected outside F132  PRE  {sorted(set(PRE['affected']) - set(F132_PAPERS))}")
print(f"  affected outside F132  POST {sorted(set(POST['affected']) - set(F132_PAPERS))}")

# `withheld` measured, not back-computed: show the sum over the legs that did NOT
# move, in each arm. If the pin were fitted as "97 + 3" this line would not hold.
unmoved_pre = sum(v["excluded_count"] for k, v in PRE["detail"].items() if k not in CLAIMED)
unmoved_post = sum(v["excluded_count"] for k, v in POST["detail"].items() if k not in CLAIMED)
claimed_pre = sum(PRE["detail"][k]["excluded_count"] for k in CLAIMED if k in PRE["detail"])
claimed_post = sum(POST["detail"][k]["excluded_count"] for k in CLAIMED if k in POST["detail"])
print("\n  withheld decomposition (measured per arm, not assumed):")
print(f"    the 69 other legs        PRE {unmoved_pre:>4}   POST {unmoved_post:>4}")
print(f"    the 3 named legs         PRE {claimed_pre:>4}   POST {claimed_post:>4}")
print(f"    total                    PRE {unmoved_pre + claimed_pre:>4}"
      f"   POST {unmoved_post + claimed_post:>4}")

print("\n" + "=" * 78)
print("THE THREE LEGS THE DIFF NAMES -- do they exist and do they move?")
print("=" * 78)
for rel in CLAIMED:
    on_disk = (REPO / rel).is_file()
    tracked = rel in PATHS
    a, b = PRE["detail"].get(rel), POST["detail"].get(rel)
    print(f"\n  {rel}")
    print(f"      file on disk / git-tracked : {on_disk} / {tracked}")
    if a is None or b is None:
        print("      NOT IN THE WALKED POPULATION IN ONE ARM -- this would be a defect")
        continue
    print(f"      paper : mode               : {b['paper']} : {b['mode']}")
    print(f"      excluded_count             : PRE {a['excluded_count']} -> POST {b['excluded_count']}")
    print(f"      excluded_terms             : PRE {a['excluded_terms']} -> POST {b['excluded_terms']}")
    print(f"      matched forbidden in raw   : PRE {a['matched']} -> POST {b['matched']}")

print("\n  every OTHER leg, both arms compared field by field:")
differing = [r for r in PATHS
             if PRE["detail"].get(r) != POST["detail"].get(r) and r not in CLAIMED]
print(f"      legs differing between arms and NOT among the three named : {len(differing)}")
for rel in differing:
    print(f"      UNATTRIBUTED MOVER: {rel}")

print("\n" + "=" * 78)
print("THE FOURTH LEG -- the enzyme, volunteered by the author as a NON-mover")
print("=" * 78)
print(f"  {FOURTH}")
print(f"      git-tracked                : {FOURTH in PATHS}")
a, b = PRE["detail"].get(FOURTH), POST["detail"].get(FOURTH)
print(f"      PRE  : {a}")
print(f"      POST : {b}")
print(f"      classification changed     : {a != b}")

post_gold = {case.paper_id: case for case in load_gold_set(POST_PATH).cases}
pre_gold = {case.paper_id: case for case in load_gold_set(PRE_PATH).cases}
case_post = post_gold["PMC12180156"]
case_pre = pre_gold["PMC12180156"]
cov = json.load(io.open(REPO / FOURTH, encoding="utf-8")).get("coverage") or {}
print("\n      requested_core_terms on this leg, and what the gate says about each:")
for term in cov.get("requested_core_terms", []):
    hp = forbidden_coverage_match(case_post, term)
    hq = forbidden_coverage_match(case_pre, term)
    print(f"        {term!r}")
    print(f"            PRE-edit  forbidden_coverage_match : {getattr(hq, 'name', hq)}")
    print(f"            POST-edit forbidden_coverage_match : {getattr(hp, 'name', hp)}")

print("\n      the separation, asked of the gate directly under the POST-edit gold:")
for probe in ("delta-aminolevulinic acid",
              "δ-aminolevulinic acid",
              "δ-aminolevulinic acid synthase",
              "δ-aminolevulinic acid synthase (ALAS2)",
              "erythroid delta-aminolevulinic acid synthase",
              "aminolevulinic acid",
              "5-aminolevulinic acid"):
    hp = forbidden_coverage_match(case_post, probe)
    hq = forbidden_coverage_match(case_pre, probe)
    print(f"        {probe!r:52s} PRE {getattr(hq, 'name', hq)!r:28s} POST {getattr(hp, 'name', hp)!r}")

print("\n      the same case's acceptable_enzymes, for the internal-consistency claim:")
for enz in getattr(case_post, "acceptable_enzymes", []) or []:
    print(f"        name={getattr(enz, 'name', '?')!r} aliases={getattr(enz, 'aliases', None)!r}")

print("\n" + "=" * 78)
print("WHAT THE PINS WOULD HAVE TO BE, DERIVED HERE AND NOWHERE ELSE")
print("=" * 78)
print(f"  assert legs == {POST['legs']}")
print(f"  assert withheld == {POST['withheld']}")
print(f"  assert with_matched_forbidden == {POST['with_matched_forbidden']}")
print("  assert set(affected_papers) - set(F132_PAPERS) == "
      f"{sorted(set(POST['affected']) - set(F132_PAPERS))}")
print(f"  assert cleared == {POST['cleared']}")
print("\n  under the PRE-edit gold the same pins would be:")
print(f"  assert withheld == {PRE['withheld']}")
print(f"  assert with_matched_forbidden == {PRE['with_matched_forbidden']}")
print("  assert set(affected_papers) - set(F132_PAPERS) == "
      f"{sorted(set(PRE['affected']) - set(F132_PAPERS))}")

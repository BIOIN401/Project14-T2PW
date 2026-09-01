"""REV-F150 probe A -- reproduce REV-F150.md section 1, then scan the NAMED run tree.

Independent reviewer's own reproduction. Reading the number in the charter is not
verification, so every value here is recomputed from the tree under measurement.

Usage:
    <py> revf150_probeA_reproduce_and_scan.py <repo-root> <gold-json-path> <label>

<gold-json-path> is passed explicitly rather than resolved through
``pinned_gold_set_path()`` so that the SAME probe can be pointed at the pre-edit
and the post-edit gold without editing the probe between arms.

The run tree is NAMED EXPLICITLY below and printed in the header. Both ``runs/``
and ``runs_verify/`` are live and both hold ``papers/*/*/final_mapped.json``;
"the pinned run" is ambiguous and has cost a full rescan before.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
GOLD_PATH = Path(sys.argv[2]).resolve()
LABEL = sys.argv[3] if len(sys.argv) > 3 else "unlabelled"

sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from t2pw.bench.goldset import load_gold_set, normalize_name, canonical_text  # noqa: E402
from t2pw.bench import acceptance  # noqa: E402

#: T-107's artifacts. NAMED, not inferred. Confirmed to exist by probe A itself.
T107_TREE = REPO / "runs_verify" / "2026-08-28_1816"

CASE_ID = "PMC12180156"
DELTA_ASCII = "delta-aminolevulinic acid"
DELTA_GREEK = "δ-aminolevulinic acid"

print("=" * 78)
print(f"REV-F150 probe A   arm={LABEL}")
print(f"  repo       : {REPO}")
print(f"  gold file  : {GOLD_PATH}")
print(f"  run tree   : {T107_TREE}   exists={T107_TREE.is_dir()}")
print("=" * 78)

gs = load_gold_set(GOLD_PATH)
cases = {c.paper_id: c for c in gs.cases}
case = cases[CASE_ID]

# ---------------------------------------------------------------------------
# 1. REV-F150.md section 1, reproduced.
# ---------------------------------------------------------------------------
print("\n--- 1. reproduction of REV-F150.md section 1 ---")
for term in ("5-aminolevulinic acid", DELTA_ASCII, DELTA_GREEK):
    hit = case.forbidden_match(term)
    print(f"  forbidden_match({term!r:34s}) -> {getattr(hit, 'name', None)!r}")
print(f"  forbidden_identifiers[0].name    : {case.forbidden_identifiers[0].name!r}")
print(f"  forbidden_identifiers[0].aliases : {list(case.forbidden_identifiers[0].aliases)}")

# ---------------------------------------------------------------------------
# 2. Are the two proposed strings actually two aliases, or one?
# ---------------------------------------------------------------------------
print("\n--- 2. normalization: are the two proposed strings distinct? ---")
n_ascii, n_greek = normalize_name(DELTA_ASCII), normalize_name(DELTA_GREEK)
print(f"  normalize_name({DELTA_ASCII!r}) = {n_ascii!r}")
print(f"  normalize_name({DELTA_GREEK!r}) = {n_greek!r}")
print(f"  IDENTICAL AFTER NORMALIZATION: {n_ascii == n_greek}")
print(f"  normalize_name('5-aminolevulinic acid') = {normalize_name('5-aminolevulinic acid')!r}")
print(f"  normalize_name('ALA')                   = {normalize_name('ALA')!r}")

# ---------------------------------------------------------------------------
# 3. V1 -- the internal-inconsistency claim, confirmed from the file itself.
# ---------------------------------------------------------------------------
print("\n--- 3. V1: does the gold author already use the delta spelling in THIS case? ---")
for i, enz in enumerate(case.all_enzymes):
    al = list(getattr(enz, "aliases", ()) or ())
    if any("aminolevulinic" in str(a).lower() for a in al) or "aminolevulinic" in str(enz.name).lower():
        print(f"  acceptable/expected enzyme [{i}] name={enz.name!r}")
        print(f"      aliases = {al}")
        print(f"      quote   = {getattr(enz, 'quote', None)!r}")

raw = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
raw_case = [c for c in raw["cases"] if c.get("paper_id") == CASE_ID][0]
print(f"\n  raw acceptable_enzymes[1].name    = {raw_case['acceptable_enzymes'][1].get('name')!r}")
print(f"  raw acceptable_enzymes[1].aliases = {raw_case['acceptable_enzymes'][1].get('aliases')!r}")

print("\n  -- every literal occurrence of 'aminolevulinic' in this case's JSON --")
blob = json.dumps(raw_case, ensure_ascii=False, indent=1)
for line in blob.splitlines():
    if "aminolevulinic" in line.lower():
        print(f"    {line.strip()}")
print(f"\n  literal 'delta-aminolevulinic acid' present in case JSON : "
      f"{DELTA_ASCII in blob}")
print(f"  literal '{DELTA_GREEK}' present in case JSON : {DELTA_GREEK in blob}")

# ---------------------------------------------------------------------------
# 4. V3 -- would the edit condemn an ACCEPTABLE enzyme? (containment check)
# ---------------------------------------------------------------------------
print("\n--- 4. V3: does the edit reach any acceptable enzyme or expected term? ---")
for label, terms in (
    ("acceptable/expected enzymes", case.all_enzymes),
    ("expected_substrates", case.expected_substrates),
    ("expected_products", case.expected_products),
    ("expected_pathway_anchors", case.expected_pathway_anchors),
):
    for t in terms:
        names = [t.name, *list(getattr(t, "aliases", ()) or ())]
        for nm in names:
            hit = case.forbidden_match(nm)
            if hit is not None:
                print(f"  !! {label}: {nm!r} -> forbidden {hit.name!r}")
print("  (no '!!' lines above means the edit condemns nothing the case accepts)")

# ---------------------------------------------------------------------------
# 5. The NAMED run tree -- every entity whose name reaches the new aliases.
# ---------------------------------------------------------------------------
print(f"\n--- 5. corpus scan of {T107_TREE} ---")
TARGET = {normalize_name(DELTA_ASCII), normalize_name(DELTA_GREEK),
          normalize_name("5-aminolevulinic acid"), normalize_name("ALA")}

def external_ids(row):
    try:
        return acceptance_ids(row)
    except Exception:
        return {}

from t2pw.bench.semantic import _external_ids as acceptance_ids  # noqa: E402

hits = 0
legs = 0
for fm in sorted(T107_TREE.glob("papers/*/*/final_mapped.json")):
    legs += 1
    paper = fm.parent.parent.name
    leg = fm.parent.name
    try:
        payload = json.loads(fm.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        print(f"  UNREADABLE {paper}/{leg}: {exc}")
        continue
    ents = payload.get("entities") or {}
    if not isinstance(ents, dict):
        continue
    for bucket, rows in ents.items():
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            nm = canonical_text(row.get("name"))
            norm = normalize_name(nm)
            if norm in TARGET or "aminolevulinic" in norm:
                ids = acceptance_ids(row)
                gcase = cases.get(paper)
                fb = gcase.forbidden_match(nm) if gcase else None
                hits += 1
                print(f"  {paper}/{leg} /entities/{bucket}/{idx}")
                print(f"      name={nm!r} norm={norm!r}")
                print(f"      external_ids={dict(ids)}")
                print(f"      forbidden_match -> {getattr(fb, 'name', None)!r}"
                      f"   PRIORITY-1 ELIGIBLE={bool(fb is not None and ids)}")
print(f"\n  legs scanned: {legs}   matching entity rows: {hits}")

# ---------------------------------------------------------------------------
# 6. D-072 seam -- does the edit reach any requested-core coverage draw?
# ---------------------------------------------------------------------------
print(f"\n--- 6. D-072 coverage-denominator seam on {T107_TREE} ---")
cov_hits = 0
for fm in sorted(T107_TREE.glob("papers/*/*/final_mapped.json")):
    paper = fm.parent.parent.name
    leg = fm.parent.name
    gcase = cases.get(paper)
    if gcase is None:
        continue
    try:
        payload = json.loads(fm.read_text(encoding="utf-8"))
    except Exception:
        continue
    # find any coverage block carrying a term list, wherever it lives
    stack = [payload]
    seen_terms = []
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("requested_core_terms", "terms", "core_terms",
                         "requested_core", "unmatched_terms", "matched_terms") \
                        and isinstance(v, list):
                    seen_terms.extend([t for t in v if isinstance(t, str)])
                stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    for t in set(seen_terms):
        if "aminolevulinic" in normalize_name(t):
            hit = acceptance.forbidden_coverage_match(gcase, t)
            cov_hits += 1
            print(f"  {paper}/{leg}  term={t!r} -> "
                  f"forbidden_coverage_match={getattr(hit, 'name', None)!r}")
print(f"  coverage-term matches touching 'aminolevulinic': {cov_hits}")
print("\n=== probe A complete ===")

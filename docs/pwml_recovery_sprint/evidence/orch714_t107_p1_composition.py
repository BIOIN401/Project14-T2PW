"""Priority-1 composition + the LpxH preservation check, at the merged tip.
Shape read from the live report, not guessed."""
from __future__ import annotations
import json, sys
from pathlib import Path
TREE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(TREE / "src"))
from t2pw.bench.acceptance import score_run  # noqa: E402
from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path  # noqa: E402
print("MEASURED_TREE =", sys.modules["t2pw"].__file__)
gold = load_gold_set(pinned_gold_set_path())

def names_of(chk):
    out = []
    for f in (chk or {}).get("findings", []) or []:
        if isinstance(f, dict):
            n = f.get("name") or f.get("entity") or f.get("identifier") or f.get("missing_anchor")
            if n: out.append(str(n))
    return out

for run in sys.argv[2:]:
    print("\n" + "#" * 78)
    print("RUN:", run)
    print("#" * 78)
    d = score_run(Path(run), gold).to_dict()
    total_raw = 0
    print("\n-- Priority-1 (false_real_identifiers) per leg --")
    for pap in d.get("papers", []):
        pid = pap.get("paper_id")
        for mode, leg in (pap.get("legs") or {}).items():
            sem = leg.get("semantic") or {}
            se = sem.get("scientific_errors") or {}
            n = int(se.get("false_real_identifiers", 0) or 0)
            if not n: continue
            total_raw += n
            chk = (sem.get("checks") or {}).get("no_real_id_or_name_conflict")
            print(f"   {pid:14s} {mode:9s} false_real={n}  names={json.dumps(sorted(set(names_of(chk))))}")
    print(f"   RAW TOTAL false_real_identifiers = {total_raw}")

    print("\n-- PMC12444477 placeholder findings (LpxH must remain, Unknown must be gone) --")
    for pap in d.get("papers", []):
        if pap.get("paper_id") != "PMC12444477": continue
        for mode, leg in (pap.get("legs") or {}).items():
            sem = leg.get("semantic") or {}
            chk = (sem.get("checks") or {}).get("placeholder_identities_distinguished")
            if chk is None: continue
            nm = sorted(set(names_of(chk)))
            cen = sem.get("identity_census") or {}
            print(f"   {mode:9s} findings={len(nm)} {json.dumps(nm)}")
            print(f"            LpxH present : {'LpxH' in nm}   Unknown absent : {'Unknown' not in nm}")
            print(f"            census: backed={cen.get('placeholder_backed_proteins')} "
                  f"sentinel_rows={cen.get('placeholder_sentinel_rows')} "
                  f"wrappers={cen.get('placeholder_generated_wrappers')} "
                  f"other={cen.get('placeholder_other_rows')} "
                  f"withheld_correct={cen.get('withheld_identity_correct')} "
                  f"withheld_recoverable={cen.get('withheld_identity_recoverable')} "
                  f"withheld_other={cen.get('withheld_identity_other')}")

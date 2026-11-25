"""FIX-S2SCOPE review step 3: every out_of_scope Stage-1 row in the stored runs, with every field Stage 1
recorded. usage: oosdump.py <tree> <out.json>. Read-only."""
import collections
import json
import sys
from pathlib import Path

tree, outp = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(Path(tree) / "src"))
sys.path.insert(0, "C:/t/acc-runs/scripts")
import accscore as S  # noqa: E402


def load(p):
    return json.load(open(p, encoding="utf-8"))


def runs():
    for s1 in sorted(Path("C:/t/acc-runs").glob("legs*/**/papers/*/strict/stage1_payload.json")):
        yield s1.parent, s1.parent.parent.name
    for i in range(1, 10):
        sd = Path(f"C:/t/t2pw-server-data/work/item-{i}/attempt-1/out")
        if (sd / "stage1_payload.json").exists():
            yield sd, "PMC12096016"


out, keysets = [], collections.Counter()
for sd, paper in runs():
    s1 = load(sd / "stage1_payload.json")
    vr = load(sd / "reaction_verifier_report.json") if (sd / "reaction_verifier_report.json").exists() else {}
    vrows = {x.get("index"): x for x in ((vr.get("reactions") or {}).get("rows") or [])}
    for i, r in enumerate(S.rows_of(s1)):
        if str(r.get("scope_membership") or "").strip().casefold() != "out_of_scope":
            continue
        v = vrows.get(i)
        by_verifier = bool(v and v.get("action") not in ("kept", "repaired", "flagged"))
        keysets[(by_verifier, tuple(sorted(r)))] += 1
        out.append({
            "paper": paper, "run": str(sd).replace("\\", "/"), "index": i, "label": S.label(r),
            "by_verifier": by_verifier, "verifier_action": (v or {}).get("action"),
            "keys": sorted(r), "biological_state": r.get("biological_state"),
            "locked_reaction_id": r.get("locked_reaction_id"), "name": r.get("name"),
            "evidence": r.get("evidence"),
            "enzymes": [{k: e.get(k) for k in ("protein", "provenance", "confidence", "evidence")} for e in r.get("enzymes") or []],
            "lineage": [{k: x.get(k) for k in ("stage", "origin", "support", "paper_explicit", "review_required")}
                        for x in r.get("provenance_lineage") or []],
            "inference": r.get("inference")})
json.dump(out, open(outp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("rows", len(out), "extractor", sum(not x["by_verifier"] for x in out))
for (bv, ks), n in keysets.most_common():
    print("KEYS", "verifier" if bv else "extractor", n, ks)
print("--- extractor-marked rows, distinct steps")
seen = collections.defaultdict(list)
for x in out:
    if not x["by_verifier"]:
        seen[(x["paper"], x["label"])].append(x)
for (paper, lab), xs in sorted(seen.items()):
    bs = sorted({str(x["biological_state"]) for x in xs})
    ep = sorted({f"{e['provenance']}/{e['confidence']}" for x in xs for e in x["enzymes"]})
    ln = sorted({f"{l['origin']}/{l['paper_explicit']}/{l['support']}" for x in xs for l in x["lineage"]})
    ev = (xs[0]["evidence"] or (xs[0]["enzymes"][0]["evidence"] if xs[0]["enzymes"] else "") or "")[:110]
    print(f"STEP {paper} n={len(xs)} | {lab} | bio_state={bs} | enzyme={ep} | lineage={ln} | name={xs[0]['name']!r} | ev={ev!r}")
print("OOSDUMP_DONE")

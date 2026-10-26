"""ORCH-775 read-only probe: trace every enzyme-less reaction of the ORCH-773 tip replay back
through the leg own Stage-1, merged and final_mapped payloads.

Inputs: the ORCH-768 corpus manifest (committed) and the ORCH-773 reviewer tip replay
outputs (C:/t/rev773r1/out-tip, scratch; PWML digests are in evidence/g11/ORCH-773/112,123).
Writes nothing.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
TIP = Path("C:/t/rev773r1/out-tip")
MAN = ROOT / "docs/pwml_recovery_sprint/evidence/orch768_corpus_manifest.json"
SLUGS = [
    "ORCH-756_enterobactin-strict_2026-10-21_2020",
    "ORCH-767_chlorophyll-strict_2026-10-23_2328",
    "runs_verify_2026-08-18_1328_PMC12452463",
    "runs_verify_2026-08-21_2239_PMC12452463",
    "runs_verify_2026-08-22_2147_PMC12452463",
    "runs_verify_2026-08-24_1203_PMC12096016",
    "runs_verify_2026-08-27_1341_PMC12782028",
    "runs_verify_2026-08-28_1816_PMC12452463",
    "runs_verify_2026-09-06_1425_PMC12376012",
    "runs_verify_2026-09-16_2152_PMC13170813",
]
CARRIERS = ("rag_provenance", "source_papers", "rag_confidence", "provenance", "origin")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rxns(d):
    return ((d.get("processes") or {}).get("reactions")) or []


def ents(field):
    out = []
    for a in field or []:
        if isinstance(a, dict):
            out.append(a.get("entity") or a.get("protein_complex") or a.get("protein") or a.get("name") or json.dumps(a)[:80])
        else:
            out.append(str(a))
    return out


def norm(xs):
    return frozenset(str(x).casefold().strip() for x in xs or [])


def brief(r):
    lin = r.get("provenance_lineage") or []
    return {
        "name": r.get("name"),
        "in": r.get("inputs"),
        "out": r.get("outputs"),
        "enzymes": ents(r.get("enzymes")),
        "modifiers": ents(r.get("modifiers")),
        "evidence": r.get("evidence"),
        "spontaneous": r.get("spontaneous"),
        "carriers": {k: r.get(k) for k in CARRIERS if r.get(k) not in (None, "", [])},
        "lineage": [(x.get("stage"), x.get("origin"), x.get("paper_explicit")) for x in lin],
    }


def matches(rs, r):
    name = (r.get("name") or "").casefold()
    o, i = norm(r.get("outputs")), norm(r.get("inputs"))
    hits = []
    for x in rs:
        if (x.get("name") or "").casefold() == name:
            hits.append(("name", x))
        elif norm(x.get("outputs")) == o and (norm(x.get("inputs")) & i or not i):
            hits.append(("io", x))
    return hits


man = load(MAN)
by_slug = {re.sub(r"[^A-Za-z0-9._-]+", "_", l["id"]): l for l in man["legs"]}
total = 0
for slug in SLUGS:
    leg = by_slug[slug]
    legdir = ROOT / leg["leg_dir"]
    print("=" * 100)
    print(slug, "| paper", leg["paper"], "| checkpoint", leg.get("checkpoint"), "| leg_dir", leg["leg_dir"])
    print("  recorded quarantine_ok:", (leg.get("recorded") or {}).get("quarantine_ok"))
    pw = TIP / slug / "outputs" / "pathway.pwml"
    if pw.exists():
        print("  tip pwml sha256", hashlib.sha256(pw.read_bytes()).hexdigest())
    can = load(TIP / slug / "canonical.json")
    crx = rxns(can)
    none = [(k, r) for k, r in enumerate(crx) if not ents(r.get("enzymes"))]
    print("  tip canonical reactions", len(crx), "enzyme-less", len(none))
    stages = {}
    for fn in ("stage1_payload.json", "merged_payload.json", "final_mapped.json"):
        p = legdir / fn
        stages[fn] = rxns(load(p)) if p.exists() else None
    rem = legdir / "removed_entity_report.json"
    rem_txt = rem.read_text(encoding="utf-8") if rem.exists() else ""
    for k, r in none:
        total += 1
        print("-" * 100)
        print("  [%d] canonical idx %d pwml id %d" % (total, k, 50001 + k))
        print("   CANON", json.dumps(brief(r), ensure_ascii=False))
        sib = [x for x in crx if x is not r and norm(x.get("outputs")) == norm(r.get("outputs"))]
        for x in sib:
            print("   CANON-SIBLING(same outputs)", json.dumps(brief(x), ensure_ascii=False))
        for fn, rs in stages.items():
            if rs is None:
                print("   ", fn, "MISSING")
                continue
            hs = matches(rs, r)
            if not hs:
                print("   ", fn, "no match")
            for how, x in hs:
                print("   ", fn, how, json.dumps(brief(x), ensure_ascii=False))
        nm = (r.get("name") or "")
        if nm and nm.casefold() in rem_txt.casefold():
            for line in rem_txt.splitlines():
                if nm.casefold() in line.casefold():
                    print("    removed_entity_report:", line.strip()[:200])
print("TOTAL enzyme-less tip reactions:", total)

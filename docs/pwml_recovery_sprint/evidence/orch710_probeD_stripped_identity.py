"""Probe D — the STRIPPED-IDENTITY loss class, counted exactly.

CRITERION, fixed before counting, derived from the measured P22557 case:

  STRIPPED-IDENTITY LOSS := a protein / protein_complex row whose
  mapping_meta.identity_verdict.identity names a real UniProt accession, while
  the row's MATERIALIZED identity (mapped_ids.uniprot / uniprot_id) does not
  carry it.

  i.e. the pipeline HELD the correct accession and did not ship it.  This is
  MASTER_PLAN 1.4's "identity ladder strips a correct accession" pattern.  It is
  a DIFFERENT SEAM from the Unknown-placeholder fallback, and membership in the
  pinned 21 is reported, never assumed.
"""
import json, os, re, glob, sys, io
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.environ["T2PW_SRC"])
from t2pw.pipeline.entity_identity import identity_status  # noqa

PINNED = "runs/2026-08-02_2130"
ACC = re.compile(r"uniprot:([A-Z0-9]{6,10})")
REAL = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}$")
pinned = json.load(io.open(os.environ["PINNED_JSON"], encoding="utf-8"))["placeholders"]
pin_keys = {(p["leg"], p["pointer"]) for p in pinned}

def scan(root, label):
    out = []
    for fp in sorted(glob.glob(f"{root}/papers/*/*/final_mapped.json")):
        p = fp.replace("\\", "/"); parts = p.split("/")
        leg = f"{parts[-3]}/{parts[-2]}"
        d = json.load(io.open(p, encoding="utf-8")); ents = d.get("entities") or {}
        for bucket in ("proteins", "protein_complexes"):
            for idx, r in enumerate(ents.get(bucket) or []):
                if not isinstance(r, dict): continue
                meta = r.get("mapping_meta") or {}
                iv = meta.get("identity_verdict") or {}
                held = [a for a in ACC.findall(json.dumps(iv)) if REAL.match(a)]
                if not held: continue
                mids = r.get("mapped_ids") or {}
                shipped = str(mids.get("uniprot") or r.get("uniprot_id") or r.get("uniprot") or "")
                if any(a == shipped for a in held): continue
                out.append({
                    "leg": leg, "pointer": f"/entities/{bucket}/{idx}", "name": r.get("name"),
                    "held": held, "shipped": shipped or "(none)",
                    "verdict_reason": iv.get("reason"), "checks": iv.get("checks"),
                    "identity_status": identity_status(r),
                    "in_pinned_21": (leg, f"/entities/{bucket}/{idx}") in pin_keys,
                })
    return out

print("=" * 78); print("D1. STRIPPED-IDENTITY LOSSES in the pinned run")
print("=" * 78)
losses = scan(PINNED, "pinned")
print(f"*** COUNT = {len(losses)} ***\n")
for l in losses:
    print(f"  {l['leg']:26s} {l['pointer']:30s} {str(l['name'])[:22]:22s}")
    print(f"       held={l['held']}  shipped={l['shipped']}  identity_status={l['identity_status']}")
    print(f"       reason={l['verdict_reason']}  checks={l['checks']}")
    print(f"       IN PINNED 21 = {l['in_pinned_21']}")
inside = [l for l in losses if l["in_pinned_21"]]
print(f"\n  inside the pinned 21 : {len(inside)}")
print(f"  outside the 21       : {len(losses) - len(inside)}")
print(f"  papers               : {sorted({l['leg'].split('/')[0] for l in losses})}")
print(f"  distinct reasons     : {sorted({str(l['verdict_reason']) for l in losses})}")

print(); print("=" * 78); print("D2. Same criterion across the WHOLE committed corpus (context only)")
print("=" * 78)
allr = []
for root in sorted({p.replace("\\", "/").split("/papers/")[0]
                    for p in glob.glob("runs*/**/papers/*/*/final_mapped.json", recursive=True)}):
    allr += scan(root, root)
print(f"  corpus-wide stripped-identity rows: {len(allr)}")
byp = {}
for l in allr: byp.setdefault(l["leg"].split("/")[0], set()).add(str(l["name"]))
for k in sorted(byp): print(f"    {k:14s} {sorted(byp[k])}")

print(); print("=" * 78); print("D3. EC 1.3.1.28 — dropped or not?")
print("=" * 78)
for fp in sorted(glob.glob(f"{PINNED}/papers/PMC12096016/*/final_mapped.json")):
    leg = fp.replace("\\", "/").split("/")[-2]
    blob = io.open(fp, encoding="utf-8").read()
    print(f"  {leg:9s} 'EC 1.3.1.28' in payload = {'EC 1.3.1.28' in blob}   "
          f"'1.3.1.28' = {'1.3.1.28' in blob}")
g = json.load(io.open("src/t2pw/bench/gold/pinned_v1.json", encoding="utf-8"))["cases"]
gc = next((c for c in g if c.get("paper_id") == "PMC12096016"), {})
s = json.dumps(gc)
i = s.find("1.3.1.28")
print(f"  gold context: ...{s[max(0,i-260):i+90]}...")

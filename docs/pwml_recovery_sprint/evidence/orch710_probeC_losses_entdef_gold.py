"""Probe C — genuine losses by the pipeline's own standard, EntD/E/F, gold flags.

LOSS CRITERION, fixed before counting and chosen because the P22557 case
demonstrates it:  a paper is a GENUINE LOSS where the SAME RUN's research leg
resolved a real UniProt accession for an entity and the strict leg shipped a
placeholder instead.  The evidence was reachable by the pipeline itself, on the
same input, in the same run -- so it is a loss, not an absence of evidence.
This is independent of the gold and of any name matching.
"""
import json, os, re, glob, sys, io
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.environ["T2PW_SRC"])
from t2pw.pipeline.entity_identity import identity_status  # noqa

PINNED = "runs/2026-08-02_2130"
UNIPROT = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}$")
pinned = json.load(io.open(os.environ["PINNED_JSON"], encoding="utf-8"))["placeholders"]
gold = json.load(io.open("src/t2pw/bench/gold/pinned_v1.json", encoding="utf-8"))["cases"]

def accs_of(path):
    """{name -> set(real uniprot)} over proteins+complex components."""
    out = {}
    d = json.load(io.open(path, encoding="utf-8")); ents = d.get("entities") or {}
    def add(nm, v):
        v = str(v or "").strip()
        if v and UNIPROT.match(v):
            out.setdefault(str(nm), set()).add(v)
    for r in ents.get("proteins") or []:
        if isinstance(r, dict):
            add(r.get("name"), (r.get("mapped_ids") or {}).get("uniprot") or r.get("uniprot_id"))
    for r in ents.get("protein_complexes") or []:
        if not isinstance(r, dict): continue
        for c in r.get("components") or []:
            if isinstance(c, dict):
                add(c.get("name"), (c.get("mapped_ids") or {}).get("uniprot") or c.get("uniprot"))
    return out

print("=" * 78); print("C1. GENUINE LOSSES — strict shipped a placeholder where research resolved a real accession")
print("=" * 78)
papers = sorted({r["leg"].split("/")[0] for r in pinned})
losses = []
for paper in papers:
    s = f"{PINNED}/papers/{paper}/strict/final_mapped.json"
    r = f"{PINNED}/papers/{paper}/research/final_mapped.json"
    if not (os.path.exists(s) and os.path.exists(r)):
        print(f"  {paper}: research leg absent -- cannot apply criterion"); continue
    sa, ra = accs_of(s), accs_of(r)
    strict_all = {a for v in sa.values() for a in v}
    gained = {n: v for n, v in ra.items() if v - strict_all}
    ph = [x for x in pinned if x["leg"] == f"{paper}/strict"]
    print(f"  {paper}: strict placeholders={len(ph)}  research resolved {len(gained)} name(s) strict lacks")
    for n, v in sorted(gained.items()):
        print(f"       research '{n}' -> {sorted(v)}   (absent from strict payload)")
        losses.append({"paper": paper, "entity": n, "accessions": sorted(v)})
print(f"\n  *** GENUINE-LOSS ENTITIES (research resolved, strict did not): {len(losses)} ***")
print("  papers involved:", sorted({l['paper'] for l in losses}))

print(); print("=" * 78); print("C2. Are those losses INSIDE or OUTSIDE the pinned 21?")
print("=" * 78)
pin_by_paper = {}
for x in pinned: pin_by_paper.setdefault(x["leg"].split("/")[0], []).append(x)
for l in losses:
    names = [str(p["name"]) for p in pin_by_paper.get(l["paper"], [])]
    hit = [n for n in names if l["entity"].lower() in n.lower() or n.lower() in l["entity"].lower()]
    print(f"  {l['paper']:14s} {l['entity'][:28]:28s} {l['accessions']}")
    print(f"       pinned-21 rows on this leg: {names}")
    print(f"       name-level correspondence : {hit or 'NONE (the placeholder is the Unknown sentinel or a differently-named wrapper)'}")

print(); print("=" * 78); print("C3. P22557 / EC 1.3.1.28")
print("=" * 78)
for paper, tok in (("PMC12856317", "P22557"), ("PMC12096016", "EC 1.3.1.28")):
    t = glob.glob(f"{PINNED}/papers/{paper}/01_source_text.txt")
    txt = io.open(t[0], encoding="utf-8", errors="replace").read() if t else ""
    print(f"  {paper}: '{tok}' in source text x{txt.count(tok)}")
    for fp in sorted(glob.glob(f"{PINNED}/papers/{paper}/*/final_mapped.json")):
        leg = fp.replace("\\", "/").split("/")[-2]
        print(f"      {leg:9s} present_in_payload={tok in io.open(fp, encoding='utf-8').read()}")

print(); print("=" * 78); print("C4. EntD / EntE / EntF — loss or superset inheritance?")
print("=" * 78)
for fp in sorted(glob.glob("runs*/**/final_mapped.json", recursive=True)):
    p = fp.replace("\\", "/")
    if "PMC12096016" not in p and "PMC12452463" not in p: continue
    parts = p.split("/"); d = json.load(io.open(p, encoding="utf-8")); ents = d.get("entities") or {}
    rows = []
    for r in ents.get("protein_complexes") or []:
        if not isinstance(r, dict): continue
        comps = [str(c.get("name")) for c in (r.get("components") or []) if isinstance(c, dict)]
        if any(e in comps or e in str(r.get("name")) for e in ("EntD", "EntE", "EntF")):
            rows.append((str(r.get("name")), r.get("pathbank_protein_complex_id"), comps))
    prot = [str(r.get("name")) for r in ents.get("proteins") or [] if isinstance(r, dict)]
    seen = {e: (e in prot or any(e in n or e in c for n, _, c in rows)) for e in ("EntD", "EntE", "EntF")}
    if not rows and not any(seen.values()): continue
    print(f"  {parts[0]+'/'+parts[1]:24s} {parts[-3]}/{parts[-2]:9s} present={seen}")
    for n, cid, comps in rows:
        print(f"        {n[:34]:34s} complex_id={cid} comps={comps}")

print(); print("=" * 78); print("C5. GOLD unknown_backed flag vs rationale, every case")
print("=" * 78)
for c in gold:
    pid, flag = c.get("paper_id"), c.get("unknown_backed_proteins_acceptable")
    rat = (c.get("unknown_backed_rationale") or "").strip()
    print(f"  {pid:14s} acceptable={str(flag):6s} rationale={'YES' if rat else 'none'}")
    if rat: print(f"       {rat}")

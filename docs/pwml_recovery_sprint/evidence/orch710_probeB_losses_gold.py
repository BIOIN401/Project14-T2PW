"""Probe B — losses, EntD/E/F, EC 1.3.1.28, P22557, and the gold tolerance flags.

Criteria fixed before counting:

 GENUINE LOSS := a row counted in the pinned 21 whose correct real accession is
   DISCOVERABLE IN THE PAPER'S OWN SOURCE TEXT, i.e. the pipeline shipped a
   placeholder where the evidence to do better was present in its own input.
   Operationalised as: the wrapper's functional name appears in the gold's
   expected/acceptable enzymes for that case, AND >=1 UniProt-shaped accession
   occurs in 01_source_text.txt within the same paper.  Reported per row with
   the accessions found, so the reader can judge rather than trust a boolean.

 This is deliberately DIFFERENT from "identity_status == placeholder", which is
 the 21.  A loss is a row that COULD have resolved.  Membership relative to the
 21 is reported, not assumed.
"""
import json, os, re, glob, sys, io

sys.path.insert(0, os.environ["T2PW_SRC"])
from t2pw.pipeline.entity_identity import identity_status, is_pathbank_unknown_protein  # noqa

PINNED = "runs/2026-08-02_2130"
UNIPROT = re.compile(r"\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b|\b[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}\b")
EC = re.compile(r"\bEC\s*\d+\.\d+\.\d+\.\d+\b")

pinned = json.load(io.open(os.environ["PINNED_JSON"], encoding="utf-8"))["placeholders"]
gold = json.load(io.open("src/t2pw/bench/gold/pinned_v1.json", encoding="utf-8"))["cases"]
gold_by = {c.get("paper_id"): c for c in gold}

print("=" * 78)
print("B1. THE SIX SPECIES-BEARING WRAPPERS vs THE PINNED 21")
print("=" * 78)
SENT_SP, SENT_ID = "arabidopsis thaliana", 4
six = []
for path in sorted(glob.glob("runs*/**/final_mapped.json", recursive=True)):
    p = path.replace("\\", "/"); parts = p.split("/")
    run, leg = parts[0] + "/" + parts[1], f"{parts[-3]}/{parts[-2]}"
    try: d = json.load(io.open(p, encoding="utf-8"))
    except Exception: continue
    for r in (d.get("entities") or {}).get("protein_complexes") or []:
        if not isinstance(r, dict) or not r.get("generated"): continue
        comps = r.get("components") or []
        if not any(isinstance(c, dict) and (c.get("pathbank_protein_id") == 9659
                   or str(c.get("name") or "").strip().lower() == "unknown") for c in comps): continue
        meta = r.get("mapping_meta") or {}; ref = r.get("species_ref") or meta.get("species_resolution") or {}
        cand = {"species_name": r.get("species_name"), "taxonomy_id": r.get("taxonomy_id"),
                "species_ref.name": ref.get("name"), "meta.species_id": meta.get("species_id")}
        good = [f"{k}={v}" for k, v in cand.items()
                if v not in (None, "") and str(v).lower() != SENT_SP and str(v) != "3702" and v != SENT_ID]
        if good:
            six.append({"run": run, "leg": leg, "name": r.get("name"), "evidence": good,
                        "in_pinned_run": run == PINNED, "ref_source": ref.get("source")})
print(f"wrappers with resolved species evidence underneath : {len(six)}")
pin_named = {(x['leg'], str(x['name'])) for x in pinned}
for s in six:
    inside = s["in_pinned_run"] and (s["leg"], str(s["name"])) in pin_named
    print(f"  {'IN PINNED 21' if inside else 'outside 21  '} | {s['run']:24s} {s['leg']:34s} {str(s['name'])[:30]:30s}")
    print(f"       source={s['ref_source']}  {'; '.join(s['evidence'])[:120]}")
n_in = sum(1 for s in six if s["in_pinned_run"] and (s["leg"], str(s["name"])) in pin_named)
print(f"\n  *** of the six, IN the pinned 21 : {n_in} ; outside : {len(six) - n_in} ***")

print()
print("=" * 78)
print("B2. GENUINE-LOSS CANDIDATES among the pinned 21")
print("=" * 78)
src_cache = {}
def source_text(paper):
    if paper not in src_cache:
        hits = glob.glob(f"{PINNED}/papers/{paper}/01_source_text.txt")
        src_cache[paper] = io.open(hits[0], encoding="utf-8", errors="replace").read() if hits else ""
    return src_cache[paper]

losses = []
for row in pinned:
    paper = row["leg"].split("/")[0]
    txt = source_text(paper)
    name = str(row["name"] or "")
    g = gold_by.get(paper) or {}
    enz = [str(x) for x in (g.get("expected_enzymes") or []) + (g.get("acceptable_enzymes") or [])]
    named = [e for e in enz if e and (e.lower() in name.lower() or name.lower() in e.lower())]
    accs = sorted(set(UNIPROT.findall(txt)))
    if named and accs:
        losses.append({"leg": row["leg"], "pointer": row["pointer"], "name": name,
                       "sentinel": row["sentinel"], "gold_enzyme": named[:3], "accessions_in_paper": accs[:8]})
print(f"pinned-21 rows whose name matches a gold enzyme AND whose paper carries UniProt accessions: {len(losses)}")
for l in losses:
    print(f"  {l['leg']:34s} {l['pointer']:32s} {l['name'][:26]:26s} sentinel={l['sentinel']}")
    print(f"       gold_enzyme={l['gold_enzyme']}  accs_in_paper={l['accessions_in_paper']}")

print()
print("=" * 78)
print("B3. P22557 and EC 1.3.1.28")
print("=" * 78)
for paper, tok, rx in (("PMC12856317", "P22557", None), ("PMC12096016", "EC 1.3.1.28", EC)):
    txt = source_text(paper)
    n = txt.count(tok)
    print(f"  {paper}: '{tok}' occurs {n} time(s) in 01_source_text.txt")
    if n:
        i = txt.find(tok); print(f"      ...{txt[max(0,i-90):i+70].replace(chr(10),' ')}...")
    shipped = []
    for fp in sorted(glob.glob(f"{PINNED}/papers/{paper}/*/final_mapped.json")):
        d = json.load(io.open(fp, encoding="utf-8"))
        blob = json.dumps(d)
        shipped.append((fp.replace("\\", "/").split("/")[-2], tok in blob))
    print(f"      shipped in payload? {shipped}")
gA = gold_by.get("PMC12096016") or {}
print(f"  gold PMC12096016 expected_enzymes: {gA.get('expected_enzymes')}")

print()
print("=" * 78)
print("B4. EntD / EntE / EntF — loss, or superset inheritance?")
print("=" * 78)
for fp in sorted(glob.glob("runs*/**/PMC12096016/*/final_mapped.json", recursive=True)) + \
          sorted(glob.glob("runs*/**/PMC12452463/*/final_mapped.json", recursive=True)):
    p = fp.replace("\\", "/"); parts = p.split("/")
    d = json.load(io.open(p, encoding="utf-8")); ents = d.get("entities") or {}
    prot = [str(r.get("name")) for r in ents.get("proteins") or [] if isinstance(r, dict)]
    cx = [(str(r.get("name")), r.get("pathbank_protein_complex_id"),
           [str(c.get("name")) for c in (r.get("components") or []) if isinstance(c, dict)])
          for r in ents.get("protein_complexes") or [] if isinstance(r, dict)]
    present = {e: (any(e in x for x in prot) or any(e in n or any(e in c for c in comps) for n, _, comps in cx))
               for e in ("EntD", "EntE", "EntF")}
    if not any(present.values()) and "PMC12096016" not in p: continue
    print(f"  {parts[0]+'/'+parts[1]:24s} {parts[-3]}/{parts[-2]:10s} present={present}")
    for n, cid, comps in cx:
        if any(e in comps for e in ("EntD", "EntE", "EntF")) or any(e in n for e in ("EntD", "EntE", "EntF")):
            print(f"        complex {n[:30]:30s} id={cid} comps={comps}")

print()
print("=" * 78)
print("B5. GOLD TOLERANCE FLAG vs RATIONALE, every case")
print("=" * 78)
for c in gold:
    pid = c.get("paper_id")
    flag = c.get("unknown_backed_proteins_acceptable")
    rat = (c.get("unknown_backed_rationale") or "").strip()
    mark = "  <== PMC12444477" if pid == "PMC12444477" else ""
    print(f"  {pid:14s} unknown_backed_proteins_acceptable={str(flag):6s} rationale_len={len(rat)}{mark}")
    if rat:
        print(f"       rationale: {rat}")

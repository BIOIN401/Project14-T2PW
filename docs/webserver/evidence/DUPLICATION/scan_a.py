"""Scan finished runs for duplication kinds a-d in final_mapped.json (and the PWML beside it for kinds c/d).
Usage: scan_runs.py OUTDIR root...   Writes OUTDIR/runs.jsonl and OUTDIR/scan_meta.json. Read-only on inputs."""
import sys, json, re, os, hashlib
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pwml_dup

def nm(s):
    return re.sub(r"\s+", " ", str(s or "").strip().casefold())

def names_of(v):
    out = []
    if isinstance(v, str):
        out.append(v)
    elif isinstance(v, dict):
        for k in ("entity", "protein", "protein_complex", "complex", "compound", "name", "element"):
            if isinstance(v.get(k), str) and v[k].strip():
                out.append(v[k])
                break
    elif isinstance(v, list):
        for x in v:
            out += names_of(x)
    return out

def acc(row, kind):
    ids = row.get("mapped_ids") if isinstance(row.get("mapped_ids"), dict) else {}
    keys = []
    if kind == "compounds":
        for k in ("chebi", "hmdb", "kegg", "pubchem"):
            v = str(ids.get(k) or "").strip()
            if v:
                keys.append((k, re.sub(r"^chebi:", "", v.casefold())))
        v = row.get("pathbank_compound_id") or ids.get("pathbank_compound_id")
        if v not in (None, ""):
            keys.append(("pathbank", str(v)))
    else:
        v = str(ids.get("uniprot") or row.get("uniprot") or "").strip()
        if v and v.casefold() != "unknown":
            keys.append(("uniprot", v.casefold()))
    return keys

def ident_map(ent):
    m = {}
    for bucket in ("compounds", "proteins", "protein_complexes", "nucleic_acids", "element_collections"):
        for row in ent.get(bucket) or []:
            if not isinstance(row, dict) or not str(row.get("name") or "").strip():
                continue
            ak = acc(row, bucket) if bucket in ("compounds", "proteins") else []
            pref = [a for a in ak if a[0] in ("pathbank", "uniprot")] or ak
            key = (bucket, pref[0]) if pref else (bucket, ("name", nm(row["name"])))
            m.setdefault(nm(row["name"]), key)
    return m

def extra(sigs):
    c = Counter(sigs)
    return sum(v - 1 for v in c.values() if v > 1)

"""ORCH-790: one-off read-only UniProt probe of the family queries the D-119 terms produce.

usage: python probe_family.py <out.json> [fields]. Prints gene/accession per query; saves the raw TSV answers.
"""
import json
import sys
import time

import requests

FIELDS = sys.argv[2] if len(sys.argv) > 2 else "accession,gene_primary,organism_id"
Q = [
    ('family:"G family"', 9606), ('family:"GDI family"', 9606), ('family:"Rho GDI family"', 9606),
    ('family:"JNK family"', 9606), ('family:"JNK subfamily"', 9606), ('family:"ROCK family"', 9606),
    ('family:"ROCK subfamily"', 9606), ('family:"Orai family"', 9606), ('family:"PLC family"', 9606),
    ('family:"PKC subfamily"', 9606), ('family:"DELLA family"', 3702), ('family:"DELLA subfamily"', 3702),
    ('family:"PIF family"', 3702), ('family:"PIF subfamily"', 3702), ('family:"5-HT 3 receptor family"', 9606),
    ('family:"5-HT3 receptor family"', 9606), ('family:"CAMK2 family"', 9606), ('family:"CaMKII subfamily"', 9606),
    ('family:"G family"', 3702), ('family:"GDI family"', 7955),
]
out = {}
s = requests.Session()
s.headers["User-Agent"] = "Project14-T2PW-ORCH790-probe/1.0"
for q, tax in Q:
    full = f"{q} AND organism_id:{tax} AND reviewed:true"
    r = s.get("https://rest.uniprot.org/uniprotkb/search",
              params={"query": full, "fields": FIELDS, "format": "tsv", "size": 25}, timeout=60)
    out[full] = {"status": r.status_code, "text": r.text}
    rows = [line.split("\t") for line in r.text.splitlines()[1:]]
    print(r.status_code, full, "->", [tuple(x[1:]) + (x[0],) for x in rows if len(x) > 1])
    time.sleep(0.3)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1, sort_keys=True)

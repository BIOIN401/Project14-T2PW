import json,sys,glob
p=json.load(open(sys.argv[1],encoding="utf-8"))
for c in p["entities"]["compounds"][:2]:
    print({k:v for k,v in c.items() if k in("name","mapped_ids","db_status","db_id","pathwhiz_id","pathbank_compound_id","synonyms","short_name","type")})
for c in p["entities"].get("protein_complexes",[])[:2]:
    print({k:v for k,v in c.items() if k not in("provenance_lineage","mapping_meta","evidence")})

import json, sys
i, pat = sys.argv[1], sys.argv[2].lower()
d = json.load(open('C:/t/corr-assess/base/%s/final_payload.json' % i, encoding='utf-8'))
for k in ('compounds', 'proteins', 'protein_complexes', 'nucleic_acids'):
    for c in d['entities'].get(k, []):
        if pat in c['name'].lower():
            print(k, {x: c.get(x) for x in ('name', 'db_id', 'pathwhiz_id', 'pathbank_compound_id', 'chosen_rule', 'db_status', 'hmdb_id', 'kegg_id', 'chebi_id')})

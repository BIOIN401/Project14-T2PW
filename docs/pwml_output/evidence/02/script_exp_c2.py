import json,re,urllib.parse,collections
import netc
from ccause import compound_cause
def chebi_info(cid):
    cid=str(cid).replace('CHEBI:','')
    t=netc.get('https://www.ebi.ac.uk/chebi/backend/api/public/compound/CHEBI:%s/'%cid,'chebi')
    try: d=json.loads(t)
    except Exception: return None
    ds=d.get('default_structure') or {}
    return dict(label=d.get('ascii_name'),inchikey=ds.get('standard_inchi_key'),stars=d.get('stars'),formula=(d.get('chemical_data') or {}).get('formula'),charge=(d.get('chemical_data') or {}).get('charge'))
def opsin(name):
    t=netc.get('https://opsin.ch.cam.ac.uk/opsin/'+urllib.parse.quote(name)+'.json','opsin')
    try: d=json.loads(t)
    except Exception: return None
    return d.get('stdinchikey') if d.get('status')=='SUCCESS' else None
R=json.load(open('results_c.json',encoding='utf-8'))
for r in R:
    r['ccause']=compound_cause(r['name'])
    best=None
    if r['chebi']:
        q=re.sub(r'[^a-z0-9]+','',r['chebi'][0]['matched_variant'].lower())
        ex=[x for x in r['chebi'] if re.sub(r'[^a-z0-9]+','',(x['label'] or '').lower())==q]
        best=(ex or r['chebi'])[0]
        info=chebi_info(best['id']); r['chebi_best']=dict(id=best['id'],**(info or {}))
        r['n_chebi_exact']=len(r['chebi'])
    # OPSIN for unresolved, plausible chemicals
    if not r['chebi'] and r['ccause'] in ('specific_named_compound','name_with_abbreviation','class_or_generic_term','abbreviation'):
        nm=re.sub(r'\s*\([^()]*\)\s*$','',r['name']) if r['ccause']=='name_with_abbreviation' else r['name']
        if r['ccause'] in ('specific_named_compound','name_with_abbreviation') and not re.search(r'mrna|operon',nm.lower()):
            r['opsin_inchikey']=opsin(nm); r['opsin_tried']=True
json.dump(R,open('results_c2.json','w',encoding='utf-8'),ensure_ascii=False,indent=0)
print(netc.ncached('chebi'),netc.ncached('opsin'))

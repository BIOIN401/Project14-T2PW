import json, sys, collections
import xml.etree.ElementTree as ET
sys.path.insert(0, 'C:/t/corr-assess/scripts')
import a1_check as c

out = {}
for s in json.load(open('C:/t/corr-assess/sample.json', encoding='utf-8')):
    i = s['idx']
    root = ET.parse('%s/%d.pwml' % (c.PWML, i)).getroot()
    pv = root.find('.//pathway-visualization')
    sp = {c.txt(x, 'id'): c.txt(x, 'name') for x in pv.find('species')}
    use = collections.Counter()
    for tag in ('biological-states', 'protein-complexes', 'proteins'):
        for e in pv.find(tag):
            sid = c.txt(e, 'species-id')
            use[(tag, sp.get(sid, '?' + sid))] += 1
    pw = pv.find('pathway')
    use[('pathway', sp.get(c.txt(pw, 'species-id'), '?'))] += 1
    out[i] = {'organism': s['organism'], 'species_defined': list(sp.values()), 'usage': {'%s|%s' % k: v for k, v in use.items()}}
    print(i, s['organism'], '->', dict(use))
json.dump(out, open('C:/t/corr-assess/a1/species_usage.json', 'w', encoding='utf-8'), indent=1)

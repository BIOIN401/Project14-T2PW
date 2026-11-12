import json
import xml.etree.ElementTree as ET
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
for l in open(R):
    r=json.loads(l); t=ET.parse(r['file']).getroot()
    for c in t.iter('protein-complex'):
        if c.find('protein_complex-proteins') is None: continue
        if not c.findall('protein_complex-proteins/protein-complex-protein'):
            print(r['file'], c.findtext('id'), c.findtext('name'))

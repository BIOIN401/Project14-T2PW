import json
d = json.load(open('C:/t/corr-assess/base/134/final_payload.json', encoding='utf-8'))
for s in d['entities']['species']:
    print(json.dumps({k: v for k, v in s.items() if k != 'provenance_lineage'}, ensure_ascii=False)[:700])
import xml.etree.ElementTree as ET
r = ET.parse('C:/t/corr-assess/replay/on_v2/pwml/134.pwml').getroot()
for s in r.find('.//pathway-visualization/species'):
    print(ET.tostring(s, encoding='unicode').replace('\n', '')[:300])

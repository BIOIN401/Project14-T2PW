"""Date of each imported file (from run folder) vs ORCH-747 (2026-10-19): which invalid-type interactions are stale?"""
import json, re, collections
import xml.etree.ElementTree as ET
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
rows=[json.loads(l) for l in open(R)]
OK={'Activation','Inhibition'}
C=collections.Counter()
for r in rows:
    f=r['file'].replace(chr(92),'/')
    m=re.search(r'(20\d\d-\d\d-\d\d)_\d{4}',f)
    d=m.group(1) if m else 'e2e5(unknown)'
    t=ET.parse(r['file']).getroot()
    bad=sum(1 for i in t.iter('interaction') if i.find('interaction-type') is not None and i.findtext('interaction-type') not in OK)
    C[('file_after_747' if d>='2026-10-19' and m else 'file_before_747/unknown')]+=1
    C[('invalid-type interactions in file_after_747' if d>='2026-10-19' and m else 'invalid-type interactions in file_before_747/unknown')]+=bad
    if bad and m and d>='2026-10-19': print('AFTER 747 with invalid:',f)
    # also withheld side-file next to the pwml?
for k,v in C.items(): print(k,v)

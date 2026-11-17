import json, sys, collections
import xml.etree.ElementTree as ET
sys.path.insert(0, 'C:/t/corr-assess/scripts')
import a1_check as c

res = {}
for s in json.load(open('C:/t/corr-assess/sample.json', encoding='utf-8')):
    i = s['idx']
    pv = ET.parse('%s/%d.pwml' % (c.PWML, i)).getroot().find('.//pathway-visualization')
    bad = {}
    for sec in pv:
        if len(sec) and sec.tag not in ('pathway',):
            ids = collections.Counter(c.txt(e, 'id') for e in sec if e.find('id') is not None)
            dups = {k: v for k, v in ids.items() if v > 1}
            if dups:
                bad[sec.tag] = dups
    res[i] = bad
    if bad:
        print(i, bad)
json.dump(res, open('C:/t/corr-assess/a1/dup_ids.json', 'w'), indent=1)

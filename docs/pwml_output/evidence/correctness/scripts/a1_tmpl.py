import collections, sys
import xml.etree.ElementTree as ET
sys.path.insert(0, 'C:/t/corr-assess/scripts')
import a1_check as c
for i in (58, 9, 68):
    pv = ET.parse('%s/%d.pwml' % (c.PWML, i)).getroot().find('.//pathway-visualization')
    cnt = collections.Counter(c.txt(l, 'visualization-template-id') for l in pv.find('compound-locations'))
    names = collections.defaultdict(list)
    cm = {c.txt(x, 'id'): c.txt(x, 'name') for x in pv.find('compounds')}
    for l in pv.find('compound-locations'):
        names[c.txt(l, 'visualization-template-id')].append(cm.get(c.txt(l, 'compound-id')))
    print(i, dict(cnt), {k: sorted(set(v))[:5] for k, v in names.items() if k not in ('81',)})

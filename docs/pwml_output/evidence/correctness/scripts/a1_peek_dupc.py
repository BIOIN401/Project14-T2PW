import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, 'C:/t/corr-assess/scripts')
import a1_check as c
for i, cid in ((128, '1099'), (116, '120'), (129, '479')):
    pv = ET.parse('%s/%d.pwml' % (c.PWML, i)).getroot().find('.//pathway-visualization')
    print(i, [(c.txt(e, 'name'), c.txt(e, 'hmdb-id'), c.txt(e, 'kegg-id')) for e in pv.find('compounds') if c.txt(e, 'id') == cid])
    for r in pv.find('reactions'):
        for e in r.iter('element-id'):
            if e.text == cid:
                print('   used in reaction', c.txt(r, 'id'), [ (x.tag, ) for x in r.iter('element-id') if x.text == cid][:1])
for i in (134, 114):
    pv = ET.parse('%s/%d.pwml' % (c.PWML, i)).getroot().find('.//pathway-visualization')
    print(i, [(c.txt(e, 'id'), c.txt(e, 'name'), c.txt(e, 'taxonomy-id')) for e in pv.find('species')])
    print('   states species-id', [(c.txt(b, 'name'), c.txt(b, 'species-id')) for b in pv.find('biological-states')])

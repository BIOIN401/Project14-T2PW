import sys, xml.etree.ElementTree as ET
f=sys.argv[1]
t=ET.parse(f).getroot()
for pv in t.iter('pathway-visualization'):
    edges={}
    for e in pv.find('edges'):
        edges[e.findtext('id')]=e
    own={}
    for el in pv.iter():
        ei=el.find('edge-id')
        if ei is not None and ei.text:
            own.setdefault(ei.text,[]).append(el.tag)
    for eid,e in edges.items():
        tpl=e.findtext('visualization-template-id'); h=e.findtext('hidden'); p=e.findtext('path')
        o=','.join(own.get(eid,['NONE']))
        print(eid,'tpl',tpl,'hid',h,'z',e.findtext('zindex'),'len',len(p),o, p[:70] if not (tpl in('5','83') and h=='false') else '')

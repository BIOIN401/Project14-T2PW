import sys, xml.etree.ElementTree as ET
t=ET.parse(sys.argv[1]).getroot()
for pv in t.iter('pathway-visualization'):
    for iv in pv.iter('interaction-visualization'):
        print('IV',iv.findtext('id'),'interaction',iv.findtext('interaction-id'))
        for ch in iv:
            if ch.tag.startswith('interaction_'):
                for m in ch:
                    print('   ',ch.tag[12:30],m.findtext('side'),'edge',m.findtext('edge-id'), [ (x.tag,x.text) for x in m if x.tag.endswith('location-id') or x.tag.endswith('visualization-id')])
for i in t.iter('interaction'):
    if i.find('interaction-type') is not None or i.find('name') is not None:
        print('INT',i.findtext('id'),i.findtext('interaction-type'), (i.findtext('name') or '')[:60])

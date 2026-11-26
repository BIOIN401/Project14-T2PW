"""Edge template ids by what refers to the edge (reaction / transport / interaction / complex cofactor member, or nothing).
usage: edge_templates.py DIR [DIR ...]   (*.pwml or */pathway.pwml)"""
import collections
import sys
from pathlib import Path

from lxml import etree as ET

P = ET.XMLParser(recover=True, huge_tree=True)
KINDS = {"reaction-visualization": "reaction", "transport-visualization": "transport", "interaction-visualization": "interaction",
         "protein-complex-visualization": "complex", "reaction-coupled-transport-visualization": "coupled",
         "sub-pathway-visualization": "subpathway"}
for d in sys.argv[1:]:
    files = sorted(Path(d).glob("*.pwml")) or sorted(Path(d).glob("*/pathway.pwml"))
    c = collections.Counter()
    for f in files:
        t = ET.parse(str(f), P).getroot()
        pv = t.find(".//pathway-visualization")
        kind = {}
        for tag, k in KINDS.items():
            for v in pv.iter(tag):
                for e in v.iter("edge-id"):
                    kind.setdefault(e.text, k)
        for e in pv.iter("edge"):
            c[(kind.get(e.findtext("id"), "none"), e.findtext("visualization-template-id"))] += 1
    print(d, len(files), "files")
    for (k, tpl), n in sorted(c.items()):
        print("   %-12s template %-4s %d" % (k, tpl, n))

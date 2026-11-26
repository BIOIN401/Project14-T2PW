"""Which compounds use compound template 157 in the reference PWMLs, with the pathway name and the compound's DrugBank id.
usage: mine_157.py DIR"""
import sys
from pathlib import Path

from lxml import etree as ET

P = ET.XMLParser(recover=True, huge_tree=True)
for f in sorted(Path(sys.argv[1]).glob("*.pwml")):
    t = ET.parse(str(f), P).getroot()
    comp = {c.findtext("id"): c for c in t.iter("compound") if c.findtext("id")}
    pname = t.findtext(".//pathway/name") or ""
    for cl in t.iter("compound-location"):
        if cl.findtext("visualization-template-id") == "157":
            c = comp.get(cl.findtext("compound-id"))
            print(f.stem, "|", pname[:50], "|", c.findtext("name") if c is not None else "?", "| drugbank", c.findtext("drugbank-id") if c is not None else "?")

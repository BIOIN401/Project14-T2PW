"""Compound-location templates in the reference PWMLs: per template, count, sizes, and whether the compound has a DrugBank id.
usage: mine_templates.py DIR"""
import collections
import sys
from pathlib import Path

from lxml import etree as ET

P = ET.XMLParser(recover=True, huge_tree=True)
by_t = collections.defaultdict(collections.Counter)
names = collections.defaultdict(list)
files = collections.defaultdict(set)
for f in sorted(Path(sys.argv[1]).glob("*.pwml")):
    t = ET.parse(str(f), P).getroot()
    comp = {}
    for c in t.iter("compound"):
        if c.findtext("id"):
            comp[c.findtext("id")] = c
    for cl in t.iter("compound-location"):
        tpl = cl.findtext("visualization-template-id")
        c = comp.get(cl.findtext("compound-id"))
        db = bool(c is not None and (c.findtext("drugbank-id") or "").strip())
        by_t[tpl]["drugbank" if db else "no_drugbank"] += 1
        by_t[tpl]["size %sx%s" % (cl.findtext("width"), cl.findtext("height"))] += 1
        files[tpl].add(f.stem)
        if len(names[tpl]) < 6 and c is not None:
            names[tpl].append(c.findtext("name"))
for tpl, cnt in sorted(by_t.items(), key=lambda kv: -sum(v for k, v in kv[1].items() if "drugbank" in k)):
    n = cnt["drugbank"] + cnt["no_drugbank"]
    if n < 3:
        continue
    print(tpl, n, dict(cnt), "files", len(files[tpl]), names[tpl])

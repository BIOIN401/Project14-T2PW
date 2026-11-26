"""Mine template-id -> PathBank style from the 16 reference PWML/SVG pairs."""
import collections
import re
import sys
from pathlib import Path
from lxml import etree as ET

REF = Path("C:/t/vis-ref/super_pathway_visualizations/super_pathway_visualizations")
P = ET.XMLParser(recover=True, huge_tree=True)
SVGNS = "{http://www.w3.org/2000/svg}"

def norm(d):
    return " ".join(re.findall(r"-?\d+(?:\.\d+)?", d or ""))

memb = collections.defaultdict(collections.Counter)
edge = collections.defaultdict(collections.Counter)
comp = collections.defaultdict(collections.Counter)
prot = collections.defaultdict(collections.Counter)
lab = collections.defaultdict(collections.Counter)
other = collections.defaultdict(collections.Counter)
for f in sorted((REF / "pwml_markups").glob("*.pwml")):
    pw = ET.parse(str(f), P).getroot()
    sv = ET.parse(str(REF / "full_vector_images" / (f.stem + ".svg")), P).getroot()
    mt = {m.findtext("id"): m.findtext("visualization-template-id") for m in pw.iter("membrane-visualization")}
    et = {}
    for e in pw.iter("edge"):
        if e.findtext("path"):
            et[norm(e.findtext("path"))] = e.findtext("visualization-template-id")
    loc = {}
    for kind, tag in (("compound", "compound-location"), ("protein", "protein-location"),
                      ("nucleic_acid", "nucleic-acid-location"), ("element_collection", "element-collection-location")):
        for c in pw.iter(tag):
            loc[(kind, c.findtext("id"))] = (c.findtext("visualization-template-id"), c.findtext("width"), c.findtext("height"))
    for g in sv.iter(SVGNS + "g"):
        gid = g.get("id")
        if gid == "membrane_visualization":
            t = mt.get(g.get("data-location-id"))
            paths = g.findall(".//" + SVGNS + "path")
            sig = tuple((p.get("stroke"), p.get("stroke-width"), p.get("stroke-dasharray"), p.get("fill")) for p in paths)
            uses = len(g.findall(".//" + SVGNS + "use"))
            memb[t][(g.get("data-complex"), sig, uses)] += 1
        elif gid == "location":
            k = g.get("data-element-type")
            key = (k, g.get("data-location-id"))
            if key in loc:
                t, w, h = loc[key]
                kids = tuple(c.tag.replace(SVGNS, "") + ":" + str(c.get("rx") or "") for c in g if isinstance(c.tag, str) and c.get("class") != "selection-box")
                tgt = {"compound": comp, "protein": prot}.get(k, other)
                tgt[(k, t)][(kids[:4], w, h, g.get("data-default-color"))] += 1
            elif k is None and "scale" in (g.get("transform") or "") and "rotate" not in (g.get("transform") or ""):
                r = g.find(SVGNS + "rect[@data-colored]")
                if r is not None:
                    lab["label"][(r.get("fill"), r.get("stroke"), r.get("stroke-width"), r.get("height"), r.get("width"))] += 1
    for p in sv.iter(SVGNS + "path"):
        if p.get("id") == "arrow_path":
            continue
        t = et.get(norm(p.get("d")))
        if t is not None:
            edge[t][(p.get("stroke"), p.get("stroke-width"), p.get("stroke-dasharray"))] += 1
    for lt in pw.iter("label-location"):
        lab["label-tmpl"][lt.findtext("visualization-template-id")] += 1

for name, d in (("MEMBRANE", memb), ("EDGE", edge), ("COMPOUND", comp), ("PROTEIN", prot), ("OTHER", other), ("LABEL", lab)):
    print("=====", name)
    for t, c in sorted(d.items(), key=lambda kv: str(kv[0])):
        for k, n in c.most_common(4):
            print(f"  {t}: {n} x {str(k)[:330]}")

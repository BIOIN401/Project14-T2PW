"""Reactants and products per reaction of a PathWhiz SBML export, next to the PWML's own roles.

usage: sbml_reactions.py SBML PWML OUT.md   (prints the table too)
SBML species names come from the export; the PWML side is read from <reaction-left-elements> / <reaction-right-elements>
(compound names via compound-id). A reaction is correct when the SBML reactants equal the PWML substrates.
"""
import sys
from lxml import etree

sbml, pwml, out = sys.argv[1:4]
s = etree.parse(sbml).getroot()
ns = {"s": s.nsmap.get(None) or "http://www.sbml.org/sbml/level3/version1/core"}
name = {e.get("id"): e.get("name") for e in s.iter("{%s}species" % ns["s"])}


def names(rx, tag):
    return sorted(name.get(r.get("species"), r.get("species")) for r in rx.iter("{%s}speciesReference" % ns["s"])
                  if r.getparent().tag.endswith(tag))


sb = []
for rx in s.iter("{%s}reaction" % ns["s"]):
    sb.append((rx.get("name") or rx.get("id"), names(rx, "listOfReactants"), names(rx, "listOfProducts")))

p = etree.parse(pwml).getroot()
cname = {c.findtext("id"): c.findtext("name") for c in p.iter("compound")}
px = []
for r in p.iter("reaction"):
    if r.find("reaction-left-elements") is None:
        continue
    def side(lst):
        return sorted(cname.get(e.findtext("element-id"), "?") for e in r.iterfind(lst + "/*") if e.findtext("element-type") == "Compound")
    px.append((side("reaction-left-elements"), side("reaction-right-elements")))

lines = ["| SBML reaction | reactants (SBML) | products (SBML) |", "|---|---|---|"]
for n, a, b in sb:
    lines.append(f"| {n} | {'; '.join(a)} | {'; '.join(b)} |")
lines += ["", "PWML roles (reaction-left-elements / reaction-right-elements), compounds only:", "",
          "| substrates (PWML) | products (PWML) |", "|---|---|"]
for a, b in px:
    lines.append(f"| {'; '.join(a)} | {'; '.join(b)} |")
asc = lambda l: sorted("".join(c for c in t if ord(c) < 128) for t in l)  # the export mangles non-ASCII names
ok = sum(1 for _, a, b in sb if any(asc(a) == asc(x) and asc(b) == asc(y) for x, y in px))
lines += ["", f"SBML reactions whose reactants and products equal a PWML reaction's substrates and products: {ok} of {len(sb)}"]
open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))

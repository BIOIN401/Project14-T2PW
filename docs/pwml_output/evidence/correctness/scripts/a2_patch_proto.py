p = "C:/t/corr-assess/a2/proto/t2pw_diagrams.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep("from PIL import ImageFont\n", "from PIL import ImageFont\n\n# A2 mock-up switches; all off = the original prototype behaviour\nOPTS = {}\n")
rep("    def inst(etype, ekey, ck):\n        key = (etype, ekey, ck)\n",
    "    def inst(etype, ekey, ck, sfx=None):\n        key = (etype, ekey, ck) if sfx is None else (etype, ekey, ck, sfx)\n")
rep('        enz = [inst(x["entity_type"], x["entity_key"], ck) for x in r.get("enzymes", [])]\n',
    '        enz = [inst(x["entity_type"], x["entity_key"], ck, r["key"] if OPTS.get("enz_clone") else None) for x in r.get("enzymes", [])]\n')
rep("""    m._ent = ent
    lay = layout(m, style)
    svg = draw(m, lay, ident)
    return svg, m, lay
""", """    m._ent = ent
    m.hidden = hide_regulation(m) if OPTS.get("hide_reg") else None
    lay = layout(m, style)
    if OPTS.get("legend"):
        lay["H"] += 56
    svg = draw(m, lay, ident)
    if OPTS.get("legend"):
        svg = svg.replace("</svg>", legend_svg(m, lay) + "</svg>")
    return svg, m, lay


REG_KINDS = ("act", "inh", "assoc", "enz")


def hide_regulation(m):
    \"\"\"A2 mock: move regulation/association edges, and molecules that exist only for them, out of the
    default view. Nothing is deleted from the data: the hidden sets are returned for a 'show' toggle.\"\"\"
    hid_e = [e for e in m.edges if e.kind in REG_KINDS]
    m.edges = [e for e in m.edges if e.kind not in REG_KINDS]
    used = {id(x) for e in m.edges for x in (e.a, e.b)}
    hid_n = [n for n in m.nodes.values() if n.kind != "junction" and id(n) not in used]
    for n in hid_n:
        del m.nodes[n.id]
        m.comps[n.comp]["nodes"].remove(n)
    return dict(edges=len(hid_e), nodes=len(hid_n), act=sum(e.kind == "act" for e in hid_e),
                inh=sum(e.kind == "inh" for e in hid_e))


def legend_svg(m, lay):
    \"\"\"A2 mock: one-line legend at the bottom of the canvas.\"\"\"
    y = lay["H"] - 30
    x = 16
    out = [f'<g class="legend"><rect x="0" y="{y - 22}" width="{lay["W"]:.0f}" height="52" fill="#fff" opacity=".92"/>']
    def t(s, dx):
        nonlocal x
        out.append(f'<text x="{x}" y="{y + 4}" font-size="12" fill="#333">{escape(s)}</text>')
        x += dx
    def sw(shape, dx=10):
        nonlocal x
        out.append(shape.format(x=x, y=y))
        x += dx
    sw('<rect x="{x}" y="{y}" width="26" height="10" rx="5" transform="translate(0,-7)" fill="#FCE9C4" stroke="#9A6A12"/>', 30); t("molecule", 72)
    sw('<rect x="{x}" y="{y}" width="26" height="10" rx="2" transform="translate(0,-7)" fill="#D7E6F6" stroke="#2F5F96"/>', 30); t("enzyme / protein", 120)
    sw('<rect x="{x}" y="{y}" width="26" height="12" rx="6" transform="translate(0,-8)" fill="#fff" stroke="#333"/>', 30); t("R# = reaction step (hover for name)", 250)
    sw('<path d="M{x},{y} h26" stroke="#555" marker-end="url(#arr)"/>', 34); t("substrate \u2192 product", 150)
    sw('<path d="M{x},{y} h26" stroke="#555" marker-end="url(#cat)"/>', 34); t("catalyses", 84)
    sw('<path d="M{x},{y} h26" stroke="#2E9E6B" stroke-width="1.9" marker-end="url(#act)"/>', 34); t("activates", 76)
    sw('<path d="M{x},{y} h26" stroke="#D0583F" stroke-width="1.9" marker-end="url(#inh)"/>', 34); t("inhibits", 70)
    sw('<path d="M{x},{y} h26" stroke="#888" stroke-dasharray="5 4"/>', 34); t("binds / other link", 130)
    h = getattr(m, "hidden", None)
    if h and h["edges"]:
        out.append(f'<text x="16" y="{y + 22}" font-size="12" fill="#1F5F8B" font-weight="600">'
                   f'{h["edges"]} regulation/binding links ({h["act"]} activate, {h["inh"]} inhibit) and {h["nodes"]} molecules '
                   f'that appear only in them are hidden \u2014 \u201cShow regulation\u201d toggle</text>')
    out.append("</g>")
    return "".join(out)
""")
open(p, "w", encoding="utf-8").write(s)
print("patched")

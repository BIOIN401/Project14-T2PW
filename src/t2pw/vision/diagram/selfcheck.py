"""Self-check: does the drawn SVG show everything the IR says?  Reads the IR and the SVG independently
(only the data-* attributes the renderer writes), never the renderer model. check_svg(ir, svg) returns a list of issues.
Moved from the v2 prototype; the batch CLI was dropped."""
import re
import xml.etree.ElementTree as ET

NS = "{http://www.w3.org/2000/svg}"
DRAWABLE = ("compound", "protein", "protein_complex", "nucleic_acid")


def _cls(el):
    return (el.get("class") or "").split()


def _ref(x):
    return f'{x["entity_type"]}:{x["entity_key"]}'


def comp_names(ir):
    loc = {l["key"]: l["name"] for l in ir.get("subcellular_locations", [])}
    out = {}
    for s in ir.get("biological_states", []):
        out[s["key"]] = "unplaced (auto state)" if s["name"].startswith("__auto") else loc.get(s.get("subcellular_location_key"), s["name"])
    return out


def view_of(root, name):
    for g in root.iter(NS + "g"):
        if name in _cls(g):
            return g
    return None


def scan(view):
    jn, edges, nodes = {}, [], []
    for el in view.iter():
        c = _cls(el)
        if el.tag == NS + "g" and "jn" in c and el.get("data-k"):
            jn.setdefault(el.get("data-p"), []).append(el)
        elif el.tag == NS + "g" and "n" in c and el.get("data-ent"):
            nodes.append(el)
        elif el.tag == NS + "path" and ("e" in c) and (el.get("data-p") or el.get("data-ik") is not None):
            edges.append(el)
    return jn, edges, nodes


def check_view(ir, root, view, label, hidden_keys, issues):
    P = ir["processes"]
    ents = {}
    for k, lst in ir["entities"].items():
        for e in lst:
            ents[e["key"]] = e
    cn = comp_names(ir)
    jn, edges, nodes = scan(view)
    by_key = {}
    for tag, els in jn.items():
        if len(els) > 1:
            issues.append(f"{label}: junction tag {tag} drawn {len(els)} times")
        for k in els[0].get("data-k").split():
            by_key[k] = (tag, els[0])
    pe = {}
    for e in edges:
        if e.get("data-p"):
            pe.setdefault((e.get("data-p"), e.get("data-ek")), []).append(e.get("data-e"))
    node_set = {(n.get("data-ent"), n.get("data-comp"), tok) for n in nodes for tok in (n.get("data-p") or "").split()}
    node_comps = {n.get("data-comp") for n in nodes}

    def need(tag, kind, ref, comp, what):
        if ref not in pe.get((tag, kind), []):
            issues.append(f"{label}: {what}: no {kind} edge for {ref} on {tag}")
        elif (ref, comp, tag) not in node_set:
            issues.append(f"{label}: {what}: {ref} not drawn in compartment {comp!r} for {tag}")

    # reactions: substrates, products, every enzyme of every copy, in the reaction's compartment
    merged = {}
    for r in P.get("reactions", []):
        hit = by_key.get(r["key"])
        if hit is None:
            issues.append(f'{label}: reaction {r["key"]} ({r.get("name")}) not drawn')
            continue
        tag, el = hit
        merged.setdefault(tag, []).append(r)
        comp = cn.get(r.get("biological_state_key"), "unplaced (no state)")
        for x in r["left"]:
            need(tag, "sub", _ref(x), comp, f'reaction {r["key"]} substrate')
        for x in r["right"]:
            need(tag, "prod", _ref(x), comp, f'reaction {r["key"]} product')
        for x in r.get("enzymes", []):
            need(tag, "cat", _ref(x), comp, f'reaction {r["key"]} enzyme')
        if len(r.get("left", [])) == 0 or len(r.get("right", [])) == 0:
            issues.append(f'{label}: reaction {r["key"]} has an empty side in the IR')
    for tag, rs in merged.items():
        if len(rs) > 1:
            txt = "".join(el.itertext())
            if not re.search(rf"\u00d7{len(rs)} in data", " ".join(t for g in view.iter(NS + "g") if g.get("data-p") == tag for t in g.itertext())):
                issues.append(f"{label}: {tag} stands for {len(rs)} identical steps but has no x{len(rs)} badge")
    # transports: cargo on both sides in the right compartments, every transporter
    for t in P.get("transports", []):
        hit = by_key.get(t["key"])
        if hit is None:
            issues.append(f'{label}: transport {t["key"]} not drawn')
            continue
        tag, el = hit
        for x in t.get("transport_elements", []):
            ca = cn.get(x.get("left_biological_state_key"), "unplaced (no state)")
            cb = cn.get(x.get("right_biological_state_key"), "unplaced (no state)")
            need(tag, "xin", _ref(x), ca, f'transport {t["key"]} cargo (from)')
            need(tag, "xout", _ref(x), cb, f'transport {t["key"]} cargo (to)')
        for x in t.get("transporters", []):
            if _ref(x) not in pe.get((tag, "cat"), []):
                issues.append(f'{label}: transport {t["key"]} transporter {_ref(x)} not drawn')
    # interactions
    drawn_ik = {}
    for e in edges:
        if e.get("data-ik") is not None:
            drawn_ik.setdefault(e.get("data-ik"), []).append(e)
    for it in P.get("interactions", []):
        k = it["key"]
        es = drawn_ik.get(k)
        if es:
            e = es[0]
            if (e.get("data-ea"), e.get("data-eb")) != (_ref(it["left"]), _ref(it["right"])):
                issues.append(f"{label}: interaction {k} drawn between the wrong molecules")
            continue
        if k not in hidden_keys:
            issues.append(f'{label}: interaction {k} ({it.get("interaction_type")}) neither drawn nor in the hidden list')
    for k in hidden_keys:
        if k in drawn_ik and label != "full":
            issues.append(f"{label}: interaction {k} is both drawn and listed as hidden")
    # compartments of every state a process refers to
    used = set()
    for r in P.get("reactions", []):
        used.add(r.get("biological_state_key"))
    for t in P.get("transports", []):
        for x in t.get("transport_elements", []):
            used.update((x.get("left_biological_state_key"), x.get("right_biological_state_key")))
        for x in t.get("transporters", []):
            used.add(x.get("biological_state_key"))
    for it in P.get("interactions", []):
        if label == "full" or it["key"] not in hidden_keys:
            used.add(it.get("biological_state_key"))
    texts = {t.strip() for t in view.itertext() if t.strip()}
    for s in used:
        name = cn.get(s, "unplaced (no state)")
        if name not in node_comps and not any(name == (g.get("data-comp") or "") for g in view.iter(NS + "g")):
            issues.append(f"{label}: compartment {name!r} (state {s}) not represented")


def check_svg(ir, svg):
    """Returns a list of issue strings (empty = every IR item is represented)."""
    issues = []
    if re.search(r"[ ,\"]NaN[ ,\"]", svg):
        issues.append("svg contains NaN coordinates")
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        return [f"svg is not well-formed XML: {e}"]
    P = ir["processes"]
    full, core = view_of(root, "view-full"), view_of(root, "view-core")
    hidden = [k for k in (root.get("data-hidden") or "").split() if k]
    if core is None:
        return ["svg has no view-core group (old renderer output)"] + issues
    if full is None:
        check_view(ir, root, core, "plain", set(), issues)
    else:
        check_view(ir, root, full, "full", set(), issues)
        check_view(ir, root, core, "core", set(hidden), issues)
        for k in hidden:
            if k not in {i["key"] for i in P.get("interactions", [])}:
                issues.append(f"hidden list names {k}, which is not an interaction of the IR")
        txt = " ".join(t for t in core.itertext())
        m = re.search(r"(\d+) regulation links? hidden", txt)
        if not m or int(m.group(1)) != len(hidden):
            issues.append(f"count line says {m.group(1) if m else 'nothing'} hidden, list has {len(hidden)}")
        if not any("regtoggle" in _cls(g) for g in core.iter(NS + "g")):
            issues.append("regulation links are hidden but the core view has no toggle")
    return issues

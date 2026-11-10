"""Per-PWML duplication metrics. Usage: pwml_dup.py file...  (prints JSON lines).
Kinds: a=process dup, b=entity-record dup (same name / same id), c=redraw (same entity+state located >1x),
d=same entity in >1 state."""
import sys, json, re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

TAGPFX = re.compile(r"<(/?)\s*[A-Za-z_][\w.-]*:")
ATTRPFX = re.compile(r"(\s)[A-Za-z_][\w.-]*:([A-Za-z_][\w.-]*=)")
CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
AMP = re.compile(r"&(?!(amp|lt|gt|apos|quot|#\d+|#x[0-9A-Fa-f]+);)")

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def t(e, n):
    c = e.find(n)
    return (c.text or "").strip() if c is not None and c.text and c.get("nil") != "true" else ""

IDK = ["hmdb-id", "kegg-id", "chebi-id", "uniprot-id", "pubchem-cid", "drugbank-id"]

def load(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    txt = TAGPFX.sub(r"<\1", txt)
    txt = ATTRPFX.sub(r"\1\2", txt)
    txt = CTRL.sub("", txt)
    txt = AMP.sub("&amp;", txt)
    if txt.startswith("<?xml"):
        txt = txt.split("?>", 1)[1]
    return ET.fromstring(txt.encode("utf-8"))

def analyze(path):
    root = load(path)
    out = {"file": str(path)}
    ents = {}
    for tag, kind in [("compound", "C"), ("protein", "P"), ("protein-complex", "X"), ("nucleic-acid", "N")]:
        for e in root.iter(tag):
            if e.find("id") is None or e.find("name") is None:
                continue
            ents[(kind, t(e, "id"))] = (t(e, "name"), {k: t(e, k) for k in IDK if t(e, k)})
    out["n_entities"] = len(ents)
    byname = defaultdict(set)
    byid = defaultdict(set)
    for k, (n, ids) in ents.items():
        byname[(k[0], norm(n))].add(k)
        for ik, iv in ids.items():
            byid[(k[0], ik, iv)].add(k)
    nd = {k: v for k, v in byname.items() if len(v) > 1 and k[1]}
    idd = {k: v for k, v in byid.items() if len(v) > 1}
    out["b_name_groups"] = len(nd)
    out["b_name_extra_records"] = sum(len(v) - 1 for v in nd.values())
    out["b_id_groups"] = len(idd)
    out["b_id_extra_records"] = sum(len(v) - 1 for v in idd.values())
    out["b_name_examples"] = [k[1] for k in list(nd)[:5]]
    loc = defaultdict(list)
    for tag, kind, idtag in [("compound-location", "C", "compound-id"), ("protein-location", "P", "protein-id"),
                             ("protein-complex-location", "X", "protein-complex-id"),
                             ("nucleic-acid-location", "N", "nucleic-acid-id")]:
        for e in root.iter(tag):
            loc[(kind, t(e, idtag))].append(t(e, "biological-state-id"))
    nloc = sum(len(v) for v in loc.values())
    out["n_locations"] = nloc
    out["entities_located"] = len(loc)
    out["c_same_state_redraws"] = sum(sum(c - 1 for c in Counter(v).values() if c > 1) for v in loc.values())
    out["c_entities_with_multi_locations"] = sum(1 for v in loc.values() if len(v) > 1)
    out["d_entities_in_multi_states"] = sum(1 for v in loc.values() if len(set(v)) > 1)

    def side(e, tag):
        return tuple(sorted((t(x, "element-id"), t(x, "element-type")) for x in e.iter(tag)))

    rx_sig = Counter(); rx_sig_en = Counter(); nrx = 0
    for e in root.iter("reaction"):
        if e.find("reaction-left-elements") is None:
            continue
        nrx += 1
        s = (side(e, "reaction-left-element"), side(e, "reaction-right-element"))
        en = tuple(sorted(t(x, "protein-complex-id") or t(x, "protein-id") for x in e.iter("reaction-enzyme")))
        rx_sig[s] += 1
        rx_sig_en[s + (en,)] += 1
    tr_sig = Counter(); tr_cargo = Counter(); ntr = 0
    for e in root.iter("transport"):
        if e.find("transport-elements") is None:
            continue
        ntr += 1
        el = tuple(sorted((t(x, "element-id"), t(x, "element-type"), t(x, "left-biological-state-id"),
                           t(x, "right-biological-state-id")) for x in e.iter("transport-element")))
        en = tuple(sorted(t(x, "protein-complex-id") or t(x, "protein-id") for x in e.iter("transport-transporter")))
        tr_sig[el + (en,)] += 1
        tr_cargo[tuple((a, b) for a, b, _, _ in el)] += 1
    out["n_reactions"] = nrx
    out["n_transports"] = ntr
    out["a_reaction_exact_dups"] = sum(c - 1 for c in rx_sig_en.values() if c > 1)
    out["a_reaction_same_participants_diff_enzyme"] = sum(c - 1 for c in rx_sig.values() if c > 1) - out["a_reaction_exact_dups"]
    out["a_transport_exact_dups"] = sum(c - 1 for c in tr_sig.values() if c > 1)
    out["a_transport_same_cargo_other_state_or_transporter"] = sum(c - 1 for c in tr_cargo.values() if c > 1) - out["a_transport_exact_dups"]
    return out

if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            print(json.dumps(analyze(p)))
        except Exception as ex:
            print(json.dumps({"file": p, "error": repr(ex)[:200]}))

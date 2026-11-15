"""P5 writerA comparison: writerA flag-on replay vs P4 d3 flag-on replay, plus a wildcard checker on the written PWML.
Usage: python cmp_wa.py   (writes docs/pwml_output/evidence/final_payload_p5/writerA/cmp_wa.out.txt)"""
import collections
import hashlib
import json
from pathlib import Path

from lxml import etree

E = Path("C:/t/fp/docs/pwml_output/evidence")
D3 = E / "final_payload_p4/d3/replay/on"
D3OFF = E / "final_payload_p4/d3/replay/off"
WA = Path("C:/t/ev-fp/p5/replay/writerA_on")
out = []


def p(*a):
    print(*a)
    out.append(" ".join(str(x) for x in a))


def load(path):
    return {json.loads(l)["idx"]: json.loads(l) for l in open(path, encoding="utf-8")}


def sha(d, i):
    return hashlib.sha256((d / "pwml" / f"{i}.pwml").read_bytes()).hexdigest()


def written(r):
    return {i for i, x in r.items() if x["pwml_written"]}


def states_of(path):
    """per state: (species id, tissue token, cell-type token, location token); token = (name, ontology-id) of the row, None = omitted."""
    root = etree.parse(str(path)).getroot()
    rows = {}
    for sec in ("tissues", "cell-types", "subcellular-locations"):
        for n in root.findall(f".//{sec}/{sec[:-1]}"):
            rows[(sec, (n.findtext("id") or "").strip())] = ((n.findtext("name") or "").casefold(), (n.findtext("ontology-id") or "").strip() or None)
    res = []
    for st in root.findall(".//biological-states/biological-state"):
        sp = (st.findtext("species-id") or "").strip() or None
        comps = [sp]
        for sec, tag in (("tissues", "tissue-id"), ("cell-types", "cell-type-id"), ("subcellular-locations", "subcellular-location-id")):
            v = (st.findtext(tag) or "").strip()
            comps.append(rows.get((sec, v), ("?" + v, None)) if v else None)
        res.append(tuple(comps))
    return res


def check(dirpath, idxs):
    """species-only states (no tissue, cell type or location), and states whose set components all equal another state's
    (wildcard-bindable, includes identical)."""
    sp_only = bindable = files_sp = files_bind = 0
    for i in sorted(idxs):
        sts = states_of(dirpath / "pwml" / f"{i}.pwml")
        a = sum(1 for s in sts if all(c is None for c in s[1:]))
        b = sum(1 for k, s in enumerate(sts) if any(j != k and all(x is None or x == y for x, y in zip(s, t)) for j, t in enumerate(sts)))
        sp_only += a
        bindable += b
        files_sp += bool(a)
        files_bind += bool(b)
    return {"species_only_states": sp_only, "files_with_species_only": files_sp, "wildcard_bindable_states": bindable, "files_with_bindable": files_bind}


d3, wa = load(D3 / "results_0.jsonl"), load(WA / "results_0.jsonl")
off = load(D3OFF / "results_0.jsonl")
p("written d3-on / writerA-on", len(written(d3)), len(written(wa)), "same set", written(d3) == written(wa),
  "only d3", sorted(written(d3) - written(wa)), "only writerA", sorted(written(wa) - written(d3)))
both = written(d3) & written(wa)
ch = sorted(i for i in both if sha(D3, i) != sha(WA, i))
p("files changed vs d3 on:", len(ch), "of", len(both), "idx", ch)
errs = [(i, x["error"][:80], x["stage"]) for i, x in wa.items() if x["error"] and x["stage"] != "stage3_revalidation"]
p("writerA non-stage3 errors", errs[:5])
for k in ("locations_dropped_no_id", "tissues_dropped_no_id", "cell_types_dropped_no_id", "components_kept_wildcard_guard", "locations_kept_state_collision"):
    tot = {name: sum((x.get("layout_debug_counts") or {}).get(k, 0) for x in r.values()) for name, r in (("d3", d3), ("writerA", wa))}
    nf = {name: sum(1 for x in r.values() if (x.get("layout_debug_counts") or {}).get(k)) for name, r in (("d3", d3), ("writerA", wa))}
    p("counter", k, "sum", tot, "files", nf)
p("wildcard check d3 on     ", check(D3, written(d3)))
p("wildcard check writerA on", check(WA, written(wa)))
p("wildcard check P4 off    ", check(D3OFF, written(off)))
# of the changed files: which counters moved
c = collections.Counter()
for i in ch:
    a, b = d3[i].get("layout_debug_counts") or {}, wa[i].get("layout_debug_counts") or {}
    for k in set(a) | set(b):
        if a.get(k, 0) != b.get(k, 0) and ("dropped" in k or "kept" in k or "wildcard" in k):
            c[k] += 1
p("changed files by moved counter", dict(c))
open("C:/t/fp/docs/pwml_output/evidence/final_payload_p5/writerA/cmp_wa.out.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")

# section-level comparison: only the sections the guard can touch (cell-types, tissues, subcellular-locations, biological-states)
SECS = ("cell-types", "tissues", "subcellular-locations", "biological-states")


def sections(path):
    root = etree.parse(str(path)).getroot()
    return [etree.tostring(n) for s in SECS for n in root.findall(f".//{s}")]


sec_ch = sorted(i for i in both if sections(D3 / "pwml" / f"{i}.pwml") != sections(WA / "pwml" / f"{i}.pwml"))
p("files whose cell-type/tissue/location/state sections changed vs d3 on:", len(sec_ch), "of", len(both), "idx", sec_ch)
p("files changed only outside those sections (compound names etc., not the writer guard):", len(set(ch) - set(sec_ch)))
open("C:/t/fp/docs/pwml_output/evidence/final_payload_p5/writerA/cmp_wa.out.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")

"""Benchmark scorer over scoped views (docs/VISION.md section 4; design in docs/vision/BENCHMARK.json).

Two sets, never pooled:

* ``kind = "paper"``: ``view(big, PAPER_ONLY)`` against what the paper covers (gold written from the
  paper, every row quoted). A paper that covers part of a pathway has gold for that part only.
* ``kind = "full"``: ``view(big, FULL)`` against the whole curated PathBank pathway, with recall over
  the curated elements, precision of the extension rows, and a breakdown by provenance kind.

Matching is the evaluation-v2 matcher (``bench/eval_v2``): reactions by defining participants
(``reactions.match_reaction``), names by the strict key (``matcher``), plus an identifier bridge
(a predicted participant whose grounded ids meet a gold term's ids counts as that term). Nothing
here edits or re-implements ``bench/``. No network: the one database read is ``build_full_gold``,
and its extractor is injectable.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from t2pw.bench.eval_v2 import matcher as M
from t2pw.bench.eval_v2.gold import Reaction, Term
from t2pw.bench.eval_v2.reactions import _row_key, match_reaction  # _row_key: the duplicate-row key of score_leg
from t2pw.vision.payload import grounded_ids, iter_sections, keyed_rows, signature
from t2pw.vision.view import ScopeSpec, view

SCHEMA_GOLD = "t2pw.vision.bench_gold/1"
SCHEMA_REPORT = "t2pw.vision.bench_report/1"
TYPES = ("reactions", "enzymes", "transports", "interactions", "compartments")
SECTION = {"reactions": "processes.reactions", "transports": "processes.transports",
           "interactions": "processes.interactions", "compartments": "entities.subcellular_locations"}
KIND_ORDER = ("paper", "other_paper", "database", "inferred", "curation", "unclassified")

# ponytail: a handful of spellings, enough for the compartments PathBank and the pipeline share;
# extend when a case needs another.
_COMP = {"cytosol": "cytosol", "cytoplasm": "cytosol", "cytoplasmic": "cytosol",
         "extracellular space": "extracellular", "extracellular": "extracellular",
         "extracellular region": "extracellular", "nucleus": "nucleus", "mitochondrion": "mitochondrion",
         "mitochondrial matrix": "mitochondrion", "periplasm": "periplasm", "periplasmic space": "periplasm"}
_REL = (("activat", "activation"), ("stimulat", "activation"), ("inhibit", "inhibition"),
        ("repress", "inhibition"), ("bind", "binding"))


def _ratio(a: int, b: int) -> Optional[float]:
    return round(a / b, 4) if b else None


# ---------------------------------------------------------------- ids and names

def _canon(ns: str, v: str) -> Tuple[str, str]:
    if ns == "hmdb":  # HMDB00646 == HMDB0000646
        digits = v.lower().replace("hmdb", "")
        return ns, str(int(digits)) if digits.isdigit() else v
    return ns, v


def row_ids(row: Mapping[str, Any]) -> frozenset:
    return frozenset(_canon(ns, v) for ns, v in grounded_ids(dict(row)))


def gold_ids(identifiers: Mapping[str, Any]) -> List[List[str]]:
    """PathBank identifier dict (hmdb_id, kegg_id, chebi_id, uniprot_id ...) as sorted [ns, value]."""
    return sorted([ns, v] for ns, v in
                  (_canon(n, x) for n, x in grounded_ids({**identifiers, "mapped_ids": dict(identifiers)})))


def _comp(name: Any) -> str:
    k = M.strict_key(name)
    return _COMP.get(k, k)


def _rel(value: Any) -> Optional[str]:
    v = str(value or "").lower()
    return next((canon for stem, canon in _REL if stem in v), None)


class _Ctx:
    """Name -> grounded ids, complexes and state locations of one view payload."""

    def __init__(self, vw: Mapping[str, Any]):
        self.cplx = M.complex_index(vw)
        self.ids: Dict[str, frozenset] = {}
        ents = [(s, r) for s, rows in iter_sections(vw) if s.startswith("entities.") for r in rows
                if isinstance(r, dict) and r.get("name")]
        for _, r in ents:
            for n in [r["name"], *(r.get("synonyms") or [])]:
                k = M.strict_key(n)
                self.ids[k] = self.ids.get(k, frozenset()) | row_ids(r)
        for s, r in ents:  # a complex stands for its components' ids
            if s != "entities.protein_complexes":
                continue
            k = M.strict_key(r["name"])
            for c in r.get("components") or []:
                cn = c if isinstance(c, str) else (c.get("entity") or c.get("protein") or c.get("name") or "")
                self.ids[k] |= self.ids.get(M.strict_key(cn), frozenset())
        self.loc = {M.strict_key(s["name"]): s.get("subcellular_location")
                    for s in vw.get("biological_states") or [] if isinstance(s, dict) and s.get("name")}

    def ids_of(self, name: Any) -> frozenset:
        return self.ids.get(M.strict_key(name), frozenset())

    def location(self, state: Any) -> Optional[str]:
        return self.loc.get(M.strict_key(state))


def _tids(t: Mapping[str, Any]) -> frozenset:
    return frozenset((a, b) for a, b in t.get("ids") or [])


def _hit(t: Mapping[str, Any], name: Any, ctx: _Ctx, enzyme: bool = False) -> bool:
    """Does a predicted entity name stand for gold term ``t``: strict name or alias, or a shared id."""
    keys = M.term_keys(t["name"], t.get("aliases") or ())
    got = M.enzyme_keys(name, ctx.cplx) if enzyme else M.candidate_keys(name)
    return bool(got & keys) or bool(_tids(t) & ctx.ids_of(name))


# ---------------------------------------------------------------- gold builders

def term(name: str, aliases: Sequence[str] = (), ids: Sequence[Sequence[str]] = (), role: str = "defining") -> Dict[str, Any]:
    return {"name": name, "aliases": [a for a in dict.fromkeys(aliases) if a and a != name],
            "ids": [list(i) for i in ids], "role": role}


def _terms(raw: Any) -> List[Dict[str, Any]]:
    out = []
    for t in raw if isinstance(raw, list) else [raw]:
        if isinstance(t, dict) and str(t.get("name") or "").strip():
            out.append(term(t["name"], t.get("aliases") or (), t.get("ids") or (), t.get("role") or "defining"))
    return out


def _sides_defining(terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if terms and not any(t["role"] == "defining" for t in terms):
        return [{**t, "role": "defining"} for t in terms]
    return terms


def _reaction_gold(raw: Mapping[str, Any], rid: str) -> Optional[Dict[str, Any]]:
    ins, outs = _sides_defining(_terms(raw.get("inputs"))), _sides_defining(_terms(raw.get("outputs")))
    if not ins or not outs:
        return None
    enz = _terms(raw["enzyme"]) if raw.get("enzyme") else []
    return {"id": raw.get("id") or rid, "label": raw.get("label") or "", "inputs": ins, "outputs": outs,
            "enzyme": enz[0] if enz else None, "reversible": bool(raw.get("reversible"))}


def _enzymes(reactions: Sequence[Mapping[str, Any]], extra: Sequence[Mapping[str, Any]] = ()) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for e in [r["enzyme"] for r in reactions if r.get("enzyme")] + list(extra):
        seen.setdefault(M.strict_key(e["name"]), {"id": f"E{len(seen) + 1}", **e})
    return list(seen.values())


def gold_from_supplement(raw_case: Mapping[str, Any], case_id: str) -> Dict[str, Any]:
    """A paper-kind gold case from one case of ``gold_v2_supported_reactions.json`` or a paper-gold
    supplement. Reactions are complete only when the case says ``supported_reactions_complete``;
    interactions and transports (supplements that have them) and the derived enzymes are recall-only.
    Off-pathway and summary statements the paper makes are kept apart (``off_scope_reactions``)."""
    rx = [g for i, r in enumerate(raw_case.get("reactions") or [], 1) if (g := _reaction_gold(r, f"R{i}"))]
    off = [g for i, r in enumerate([*(raw_case.get("off_pathway_reactions") or []), *(raw_case.get("summary_statements") or [])], 1)
           if (g := _reaction_gold(r, f"O{i}"))]
    inter = [{"id": r.get("id") or f"I{i}", "a": _terms(r.get("source")), "b": _terms(r.get("target")),
              "relationship": _rel(r.get("effect") or r.get("relationship"))}
             for i, r in enumerate(raw_case.get("interactions") or [], 1)]
    tr = [{"id": r.get("id") or f"T{i}", "cargo": _terms(r.get("entity")), "from": r.get("from_compartment"),
           "to": r.get("to_compartment")} for i, r in enumerate(raw_case.get("transports") or [], 1)]
    return {"schema": SCHEMA_GOLD, "id": case_id, "kind": "paper", "species": raw_case.get("species") or "",
            "complete": {"reactions": bool(raw_case.get("supported_reactions_complete"))},
            "reactions": rx, "off_scope_reactions": off, "enzymes": _enzymes(rx),
            "transports": [t for t in tr if t["cargo"]], "interactions": [i for i in inter if i["a"] and i["b"]],
            "compartments": []}


def _pb_term(p: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not p.get("name"):
        return None
    comps = p.get("components") or []
    ids = list(gold_ids(p.get("identifiers") or {}))
    for c in comps:
        ids += gold_ids({"uniprot_id": c.get("uniprot")}) if c.get("uniprot") else []
    aliases = [p.get("short_name"), p.get("gene_name"), *(p.get("synonyms") or [])]
    aliases += [x for c in comps for x in (c.get("gene_name"), c.get("name"))]
    return term(p["name"].strip(), [a.strip() for a in aliases if isinstance(a, str)][:16], ids,
                "cofactor" if p.get("currency") else "defining")


def _side(parts: Sequence[Mapping[str, Any]], side: str) -> List[Dict[str, Any]]:
    return [t for t in (_pb_term(x) for x in parts if x.get("side") == side) if t]


def _pb_enzyme(catalysts: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    t = _pb_term(catalysts[0]) if catalysts else None
    return {**t, "role": "defining"} if t else None


def gold_from_pathbank(extract: Mapping[str, Any], case_id: str) -> Dict[str, Any]:
    """A full-kind gold case from a ``pathbank_gold.extract`` payload: every reaction, transport,
    interaction, enzyme and compartment of the curated pathway. All five types count as complete."""
    rx, tr, inter, comps, skipped = [], [], [], [], []
    for p in extract.get("processes") or []:
        ptype, parts = p.get("process_type"), p.get("participants") or []
        loc = (p.get("biological_state") or {}).get("location")
        comps.append(loc)
        name = p.get("accession") or p.get("process_id")
        if ptype in ("Reaction", "ReactionCoupledTransport"):
            g = _reaction_gold({"id": name, "inputs": _side(parts, "left"), "outputs": _side(parts, "right"),
                                "enzyme": _pb_enzyme(p.get("catalysts") or []),
                                "reversible": str(p.get("direction") or "") in ("<>", "=", "<=>", "<->", "both")},
                               f"R{len(rx) + 1}")
            if g:
                rx.append(g)
            else:
                skipped.append(name)
        elif ptype == "Interaction":
            a, b = _side(parts, "left"), _side(parts, "right")
            if a and b:
                inter.append({"id": name, "a": a, "b": b, "relationship": _rel(p.get("interaction_type"))})
            else:
                skipped.append(name)
        elif ptype == "Transport":
            cargo = [t for t in map(_pb_term, parts) if t]
            fr = next(((x.get("from_state") or {}).get("location") for x in parts if x.get("from_state")), None)
            to = next(((x.get("to_state") or {}).get("location") for x in parts if x.get("to_state")), None)
            comps += [fr, to]
            if cargo:
                tr.append({"id": name, "cargo": cargo, "from": fr, "to": to,
                           "transporter": _pb_enzyme(p.get("catalysts") or [])})
            else:
                skipped.append(name)
    cs = {_comp(c): c for c in comps if c}
    compartments = [{"id": f"C{i}", **term(name)} for i, name in enumerate(cs.values(), 1)]
    return {"schema": SCHEMA_GOLD, "id": case_id, "kind": "full", "species": (extract.get("species") or {}).get("name") or "",
            "pathbank": {"accession": extract.get("accession"), "pathway_id": extract.get("pathway_id"),
                         "name": extract.get("name"), "content_sha256": extract.get("content_sha256")},
            "complete": {t: True for t in TYPES}, "reactions": rx, "off_scope_reactions": [],
            "enzymes": _enzymes(rx, [t["transporter"] for t in tr if t.get("transporter")]),
            "transports": tr, "interactions": inter, "compartments": compartments, "skipped_processes": skipped}


def _live_extract(accession: str) -> Dict[str, Any]:  # the only database access in this module
    from t2pw.bench.eval_v2 import pathbank_gold
    conn = pathbank_gold.connect()
    try:
        return pathbank_gold.extract(conn, accession)
    finally:
        conn.close()


def build_full_gold(accession: str, extract_fn: Callable[[str], Mapping[str, Any]] = _live_extract) -> Dict[str, Any]:
    """Full-view gold for one PathBank accession (PW id). ``extract_fn`` is injected in tests."""
    return gold_from_pathbank(extract_fn(accession), accession)


# ---------------------------------------------------------------- predicted elements of a view

def _dump(row: Any) -> str:
    return json.dumps(row, sort_keys=True, default=str)


def _items(big: Mapping[str, Any], vw: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Per element type, the deduplicated predicted elements of ``vw`` with key, base flag, kinds."""
    pool: Dict[Tuple[str, str], deque] = defaultdict(deque)
    for s, k, r, is_base in keyed_rows(big):
        pool[(s, _dump(r))].append((k, is_base))
    prov = big.get("provenance") or {}
    found: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TYPES}

    def add(t, dedup, base, kinds, **kw):
        for it in found[t]:
            if it["dedup"] == dedup:
                it["is_base"] |= base
                it["kinds"] |= kinds
                return
        found[t].append({"dedup": dedup, "is_base": base, "kinds": set(kinds), **kw})

    by_section = {s: rows for s, rows in iter_sections(vw)}
    for t, section in SECTION.items():
        for row in by_section.get(section, []):
            q = pool[(section, _dump(row))]
            key, is_base = q.popleft() if q else (None, True)
            kinds = {r["kind"] for r in prov.get(key, [])} or {"unclassified"}
            dedup = (_row_key(row) if t == "reactions" else signature(section, row)) if isinstance(row, dict) else _dump(row)
            add(t, dedup, is_base, kinds, row=row)
            if t in ("reactions", "transports") and isinstance(row, dict):  # enzymes come from these rows
                names = M.enzyme_names(row) if t == "reactions" else M.participant_names(row.get("transporters"))
                for n in names:
                    add("enzymes", M.strict_key(n), is_base, kinds, row=n)
    return found


def _primary(kinds: set) -> str:
    return next((k for k in KIND_ORDER if k in kinds), "unclassified")


# ---------------------------------------------------------------- matchers

def _bridged(row: Mapping[str, Any], g: Mapping[str, Any], ctx: _Ctx) -> Dict[str, Any]:
    """The row with the gold term names its participants' ids point to added next to them."""
    out = dict(row)
    terms = [*g["inputs"], *g["outputs"]]
    for side in ("inputs", "outputs"):
        names = M.participant_names(row.get(side))
        extra = [t["name"] for n in names for t in terms if _tids(t) & ctx.ids_of(n)]
        out[side] = names + extra
    return out


def _m_reaction(g, objs, row, ctx) -> bool:
    return match_reaction(objs[g["id"]], _bridged(row, g, ctx))[0] == "full"


def _m_transport(g, row, ctx) -> bool:
    if not any(_hit(t, row.get("cargo"), ctx) for t in g["cargo"]):
        return False
    for want, state in ((g.get("from"), row.get("from_biological_state")), (g.get("to"), row.get("to_biological_state"))):
        a, b = _comp(want) if want else None, _comp(ctx.location(state)) if ctx.location(state) else None
        if a in _COMP.values() and b in _COMP.values() and a != b:
            return False
    return True


def _m_interaction(g, row, ctx) -> bool:
    e1, e2 = row.get("entity_1"), row.get("entity_2")
    ok = any(all(any(_hit(t, e, ctx) for t in side) for side, e in ((g["a"], x), (g["b"], y))) for x, y in ((e1, e2), (e2, e1)))
    r1, r2 = g.get("relationship"), _rel(row.get("relationship"))
    return ok and {r1, r2} != {"activation", "inhibition"}  # binding fits either: a paper says "binds", PathBank "activation"


def _m_compartment(g, row, ctx) -> bool:
    name = row.get("name") if isinstance(row, dict) else row
    return _comp(name) in {_comp(g["name"]), *(_comp(a) for a in g.get("aliases") or [])}


def _gold_elems(gold: Mapping[str, Any], t: str, scope: ScopeSpec) -> List[Dict[str, Any]]:
    """Gold elements of a type; with a paper-only scope, those flagged ``in_paper: false`` drop out."""
    return [e for e in gold.get(t) or [] if e.get("in_paper", True) or not scope.paper_only]


def _type_stats(gold_n: int, items: List[Dict[str, Any]], hits: List[List[int]], complete: bool) -> Dict[str, Any]:
    recalled = {j for h in hits for j in h}
    base_recalled = {j for it, h in zip(items, hits) if it["is_base"] for j in h}
    correct = sum(1 for h in hits if h)
    return {"gold": gold_n, "recalled": len(recalled), "predicted": len(items), "correct": correct, "complete": complete,
            "recall": _ratio(len(recalled), gold_n), "precision": _ratio(correct, len(items)) if complete else None,
            "recalled_base_only": len(base_recalled), "recovered_by_extension": len(recalled - base_recalled)}


def score_view(big: Mapping[str, Any], scope: ScopeSpec, gold_case: Mapping[str, Any]) -> Dict[str, Any]:
    """Score ``view(big, scope)`` against one gold case. Pure: no I/O, ``big`` is not changed.

    types[t]: gold, recalled (gold elements matched by at least one predicted element), predicted
    (deduplicated), correct (predicted elements matching some gold element), recall, precision (only
    when the case declares that type complete), and how many recalled elements only extension rows
    recover. extension[t]: extension rows in the view and how many match gold. provenance: per type,
    predicted elements and correct ones by provenance kind, both over all kinds an element carries
    ("any") and over its single highest-ranked kind ("primary", which partitions the elements)."""
    vw = view(big, scope)
    ctx = _Ctx(vw)
    items = _items(big, vw)
    gold = {t: _gold_elems(gold_case, t, scope) for t in TYPES}
    objs = {g["id"]: Reaction(g["id"], g.get("label") or "", tuple(Term(t["name"], tuple(t["aliases"]), t["role"]) for t in g["inputs"]),
                              tuple(Term(t["name"], tuple(t["aliases"]), t["role"]) for t in g["outputs"]), None, bool(g.get("reversible")), "")
            for g in [*gold["reactions"], *(gold_case.get("off_scope_reactions") or [])]}
    match: Dict[str, Callable[[Any, Any], bool]] = {
        "reactions": lambda g, p: _m_reaction(g, objs, p["row"], ctx),
        "enzymes": lambda g, p: _hit(g, p["row"], ctx, enzyme=True),
        "transports": lambda g, p: _m_transport(g, p["row"], ctx),
        "interactions": lambda g, p: _m_interaction(g, p["row"], ctx),
        "compartments": lambda g, p: _m_compartment(g, p["row"], ctx)}
    complete = gold_case.get("complete") or {}
    types, hit_by_type = {}, {}
    for t in TYPES:
        hits = [[j for j, g in enumerate(gold[t]) if match[t](g, p)] for p in items[t]]
        hit_by_type[t] = hits
        types[t] = _type_stats(len(gold[t]), items[t], hits, bool(complete.get(t)))
    off = [g for g in gold_case.get("off_scope_reactions") or []]
    types["reactions"]["paper_stated_off_scope"] = sum(
        1 for p, h in zip(items["reactions"], hit_by_type["reactions"]) if not h and any(_m_reaction(g, objs, p["row"], ctx) for g in off))
    ext, prov = {}, {}
    for t in TYPES:
        e = [h for it, h in zip(items[t], hit_by_type[t]) if not it["is_base"]]
        ext[t] = {"rows": len(e), "correct": sum(1 for h in e if h), "precision": _ratio(sum(1 for h in e if h), len(e))}
        any_k, prim = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for it, h in zip(items[t], hit_by_type[t]):
            for k in it["kinds"]:
                any_k[k][0] += 1
                any_k[k][1] += bool(h)
            prim[_primary(it["kinds"])][0] += 1
            prim[_primary(it["kinds"])][1] += bool(h)
        prov[t] = {"any": {k: {"rows": a, "correct": b} for k, (a, b) in any_k.items()},
                   "primary": {k: {"rows": a, "correct": b} for k, (a, b) in prim.items()}}
    tot = {"rows": sum(v["rows"] for v in ext.values()), "correct": sum(v["correct"] for v in ext.values())}
    ext["all"] = {**tot, "precision": _ratio(tot["correct"], tot["rows"])}
    return {"case": gold_case.get("id"), "kind": gold_case.get("kind"),
            "scope": {"paper_only": scope.paper_only, "max_hops": scope.max_hops},
            "types": types, "extension": ext, "provenance": prov}


# ---------------------------------------------------------------- the benchmark

def _micro(legs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sums numerators and denominators over legs (eval_v2 convention). Precision uses only legs
    whose case declares the type complete."""
    out: Dict[str, Any] = {"legs": len(legs), "types": {}, "extension": {}, "provenance": {}}
    for t in TYPES:
        s = Counter()
        for l in legs:
            x = l["types"][t]
            s["gold"] += x["gold"]
            s["recalled"] += x["recalled"]
            s["recovered_by_extension"] += x["recovered_by_extension"]
            if x["complete"]:
                s["p_predicted"] += x["predicted"]
                s["p_correct"] += x["correct"]
        out["types"][t] = {"recall": _ratio(s["recalled"], s["gold"]), "recall_num_den": [s["recalled"], s["gold"]],
                           "precision": _ratio(s["p_correct"], s["p_predicted"]),
                           "precision_num_den": [s["p_correct"], s["p_predicted"]],
                           "recovered_by_extension": s["recovered_by_extension"]}
        e = Counter()
        pk: Dict[str, Counter] = defaultdict(Counter)
        for l in legs:
            e.update({"rows": l["extension"][t]["rows"], "correct": l["extension"][t]["correct"]})
            for k, v in l["provenance"][t]["primary"].items():
                pk[k].update(v)
        out["extension"][t] = {"rows": e["rows"], "correct": e["correct"], "precision": _ratio(e["correct"], e["rows"])}
        out["provenance"][t] = {k: dict(v) for k, v in pk.items()}
    return out


def run_benchmark(cases: Sequence[Mapping[str, Any]], runs_by_case: Mapping[str, Mapping[str, Mapping[str, Any]]],
                  scope: ScopeSpec) -> Dict[str, Any]:
    """Score every stored run (``{case id: {run label: big payload}}``) of every case at ``scope``.
    The aggregate is micro-averaged over legs and kept apart per case kind (paper / full)."""
    legs, missing = [], []
    for case in cases:
        runs = runs_by_case.get(case["id"]) or {}
        missing += [] if runs else [case["id"]]
        legs += [{"run": label, **score_view(big, scope, case)} for label, big in runs.items()]
    kinds = sorted({l["kind"] for l in legs})
    return {"schema": SCHEMA_REPORT, "scope": {"paper_only": scope.paper_only, "max_hops": scope.max_hops},
            "legs": legs, "aggregate": {k: _micro([l for l in legs if l["kind"] == k]) for k in kinds},
            "cases_without_runs": missing}

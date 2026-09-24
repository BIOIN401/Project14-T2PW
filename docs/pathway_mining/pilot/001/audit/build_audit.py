"""Build PILOT-001-AUDIT.md and pilot_001_audit.json from the pilot-001 artifacts.

Read-only over:
  C:/t/mining-pilot-runs/pilot-001-jasmonate-ath/**      (run artifacts)
  C:/t/mining-pilot/data/rag_index/acquire_cache/*.json   (search results the run cached)
  C:/t/mining-pilot-runs/audit/frontier_replay.json       (output of frontier_replay.py)
Writes only into C:/t/mining-pilot-runs/audit/. No network, no model call, no imports
from the mining package (so this script needs no PYTHONPATH).

Run order: frontier_replay.py first, then this script.
"""
import collections
import datetime as dt
import json
import re
from pathlib import Path

RUN = Path("C:/t/mining-pilot-runs/pilot-001-jasmonate-ath")
CACHE = Path("C:/t/mining-pilot/data/rag_index/acquire_cache")
OUT = Path("C:/t/mining-pilot-runs/audit")
CODE = "C:/t/mining-pilot"  # frozen worktree @ 23e1ae83; file:line refs below are into it


def jl(p):
    p = RUN / p
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def js(p):
    return json.loads((RUN / p).read_text(encoding="utf-8"))


def squash(s):
    return re.sub(r"\s+", "", str(s or "")).lower()


# ---------------------------------------------------------------- load artifacts
summary, manifest, state = js("summary.json"), js("run_manifest.json"), js("state.json")
primary, cand = js("primary_graph.json"), js("candidate_subgraphs.json")
preflight = js("preflight.json")
papers = jl("papers.jsonl")
P = {p["paper_id"]: p for p in papers}
queries = jl("queries.jsonl")
claims_top = jl("claims.jsonl")
decisions_top = jl("decisions.jsonl")
spans = {s["evidence_id"]: s for s in jl("evidence_spans.jsonl")}
model_calls = jl("model_calls.jsonl")
r1 = {n: jl(f"rounds/round_001/{n}.jsonl") for n in
      ("claims_extracted", "decisions", "rejections", "queries", "input_frontier", "new_frontier",
       "papers_considered", "passages_retrieved", "errors")}
r2 = {n: jl(f"rounds/round_002/{n}.jsonl") for n in
      ("claims_extracted", "decisions", "rejections", "queries", "input_frontier", "new_frontier",
       "papers_considered", "passages_retrieved", "errors")}
m1, m2 = js("rounds/round_001/metrics.json"), js("rounds/round_002/metrics.json")
replay = json.loads((OUT / "frontier_replay.json").read_text(encoding="utf-8"))
paper_text = state["paper_texts"]["PMC:PMC13364333"]
errors = state["errors"]
usage = manifest and summary["usage"]
budgets = manifest["config"]["budgets"]
C = r1["claims_extracted"]  # model output order: claim[i] in the drop messages == C[i]
assert len(C) == 21 and max(int(re.match(r"claim\[(\d+)\]", e).group(1)) for e in errors) == 20

reactions = [r for comp in cand["components"] for r in comp["reactions"]]
comp_of = {r["reaction_id"]: comp["component_id"] for comp in cand["components"] for r in comp["reactions"]}
cid_to_idx = collections.defaultdict(list)
for i, c in enumerate(C):
    cid_to_idx[c["claim_id"]].append(i)

# ---------------------------------------------------------------- uncited drops
DROP_RE = re.compile(r"^claim\[(\d+)\]\.(\w+)=(.*) \(no valid citation\)$")
drops = []
for e in errors:
    m = DROP_RE.match(e)
    if m:
        drops.append({"claim_index": int(m.group(1)), "field": m.group(2),
                      "value": m.group(3)[1:-1], "message": e})
drop_by_field = collections.Counter(d["field"] for d in drops)


def variants(field, value):
    """Exact value first, then looser forms, each reported separately."""
    out = [("exact", value)]
    if field == "species":
        out += [("genus", value.split()[0]), ("abbrev", "M. polymorpha")]
    elif field == "enzyme" and value.startswith("Mp"):
        out += [("without Mp prefix", value[2:])]
    return out


shown_ids = list(spans)
for d in drops:
    c = C[d["claim_index"]]
    cited = c["evidence_ids"]
    d["claim_id"] = c["claim_id"]
    d["claim_label"] = f'{c["subject"]} -> {c["object"]}'
    d["claim_cited_evidence_ids"] = cited
    checks = {}
    for kind, v in variants(d["field"], d["value"]):
        key = squash(v)
        checks[kind] = {
            "value": v,
            "in_claim_cited_spans": [e for e in cited if key in squash(spans[e]["text"])],
            "in_any_shown_span": [e for e in shown_ids if key in squash(spans[e]["text"])],
            "in_full_paper_text": key in squash(paper_text),
        }
    d["presence"] = checks
    ex = checks["exact"]
    if ex["in_claim_cited_spans"]:
        d["verdict"] = "value is in a passage this claim already cites; a valid citation existed and was not attached to the field"
    elif ex["in_any_shown_span"]:
        d["verdict"] = "value is in another passage shown to the model; a valid citation existed but was not given"
    elif ex["in_full_paper_text"]:
        d["verdict"] = "value is in the paper but in no passage shown to the model; drop is correct enforcement"
    else:
        d["verdict"] = "value appears nowhere in the paper text; drop is correct enforcement (value not from the evidence)"
verdict_counts = collections.Counter(d["verdict"] for d in drops)
verdict_by_field = collections.Counter((d["field"], d["verdict"]) for d in drops)

# ---------------------------------------------------------------- seed-connection curation
# Auditor assessment (name-level, NOT a graph edge: the run has no seed entity in its graph).
ROLE = {
    "(9s,13s)-opda": ("seed", "(9S,13S)-OPDA is cis-(+)-12-OPDA, the seed metabolite 12-OPDA"),
    "13-hpot": ("chain", "13-HPOT: the 13-LOX product and the canonical AOS substrate"),
    "alleneoxide(12,13-eot)": ("chain", "12,13-EOT: the AOS product and AOC substrate that cyclises to 12-OPDA"),
    "ala(18:3n3)": ("precursor", "ALA (alpha-linolenic acid): the C18 fatty acid 13-LOX oxygenates to 13-HPOT"),
    "hta(16:3n3)": ("precursor", "HTA (hexadecatrienoic acid): the C16 precursor of dn-OPDA"),
    "alleneoxide(from11-hpht)": ("c16", "C16 allene oxide made by AOS from 11-HPHT"),
    "(9s,13s)-dn-cis-opda": ("c16", "dn-cis-OPDA: the C16 analogue of 12-OPDA"),
    "dn-cis-opda": ("c16", "dn-cis-OPDA: the C16 analogue of 12-OPDA"),
    "dn-iso-opda": ("c16", "dn-iso-OPDA: bryophyte bioactive jasmonate derived from dn-cis-OPDA"),
    "δ4-dn-cis-opda": ("c16", "Delta4-dn-cis-OPDA: C16 OPDA-type cyclopentenone"),
    "δ4-dn-iso-opda": ("c16", "Delta4-dn-iso-OPDA: bryophyte bioactive jasmonate"),
    "dn-opda-glu": ("c16", "dn-OPDA amino-acid conjugate"),
    "dn-opda-gln": ("c16", "dn-OPDA amino-acid conjugate"),
    "dn-opda-his": ("c16", "dn-OPDA amino-acid conjugate"),
    "ara(20:4n6)": ("c20_precursor", "ARA: C20 PUFA from which 15-HPETE arises by LOX (that LOX step is not in the graph)"),
    "epa(20:5n3)": ("c20_precursor", "EPA: C20 PUFA from which 15-HPEPE arises by LOX (that LOX step is not in the graph)"),
    "15-hpepe": ("c20", "15-HPEPE: C20 (EPA-derived) LOX product; not on the Arabidopsis route"),
    "alleneoxide(14,15-eepe)": ("c20", "14,15-EEPE: C20 allene oxide; not on the Arabidopsis route"),
    "15-hpete": ("c20", "15-HPETE: C20 (ARA-derived) LOX product; not on the Arabidopsis route"),
    "alleneoxide": ("c20", "unqualified 'allene oxide' (here the AOS product of 15-HPETE, a C20 oxylipin)"),
}
TIER_TEXT = {
    "seed": "DIRECT - produces the seed metabolite 12-OPDA",
    "aos_canonical": "DIRECT - the canonical AOS step (13-HPOT -> 12,13-EOT)",
    "chain": "jasmonate/oxylipin chain metabolite",
    "c16": "oxylipin chain, C16 dn-OPDA branch (parallel to the 12-OPDA route)",
    "c20": "C20 (eicosanoid) oxylipin branch; AOS-type chemistry but not the Arabidopsis 12-OPDA route",
    "precursor": "upstream fatty-acid precursor of the chain (produces it)",
    "precursor_consumed": "consumes a chain precursor (ALA) and leads away from 12-OPDA, toward C20 PUFAs",
    "c20_precursor": "C20 PUFA feeding the C20 oxylipin branch; no LOX step in the graph links it",
    "none": "no direct link; fatty-acid desaturation/elongation upstream of the oxylipin substrates",
}
TIER_ORDER = ["seed", "aos_canonical", "chain", "c16", "c20", "precursor", "precursor_consumed",
              "c20_precursor", "none"]


def seed_connection(r, dropped_enzymes):
    parts = [(n, ROLE.get(squash(n))) for n in r["substrate_names"] + r["product_names"]]
    hits = [(n, role) for n, role in parts if role]
    tiers = {role[0] for _, role in hits}
    subs = {squash(n) for n in r["substrate_names"]}
    prods = {squash(n) for n in r["product_names"]}
    if "13-hpot" in subs and "alleneoxide(12,13-eot)" in prods:
        tiers.add("aos_canonical")
    if "precursor" in tiers and not any(ROLE.get(p, ("",))[0] == "precursor" for p in prods):
        tiers.discard("precursor")
        tiers.add("precursor_consumed")
    tier = next((t for t in TIER_ORDER if t in tiers), "none")
    aos = sorted({v for v in dropped_enzymes if "AOS" in v.upper()})
    via = [f"{n} ({role[1]})" for n, role in hits]
    return {
        "tier": tier,
        "summary": TIER_TEXT[tier],
        "shared_chain_metabolites": via,
        "involves_AOS_enzyme": (
            f"only via DROPPED enzyme value(s) {aos} (no valid citation); retained enzyme: none" if aos
            else "no (no enzyme retained; no AOS value asserted)"),
        "connects": tier != "none",
        "basis": "auditor name-level assessment; the run graph holds no seed entity and no primary graph, so no graph edge links any reaction to 12-OPDA or AOS",
    }


# ---------------------------------------------------------------- verbatim excerpts
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\u0394])")


SYN = {"ARA (20:4n6)": {"AA", "arachidonic acid"}}  # the span writes 'converts DGLA to AA'


def name_tokens(name):
    toks = {name} | SYN.get(name, set())
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", name)
    if m:
        toks |= {m.group(1), m.group(2)}
    return {t for t in toks if len(squash(t)) >= 3}


def tok_re(t):
    """Whitespace-tolerant, boundary-anchored: 'dn-cis-OPDA' matches 'dn- cis -OPDA'; 'LA' never matches 'PLA'."""
    core = r"\s*".join(re.escape(ch) for ch in re.sub(r"\s+", "", t))
    return re.compile(r"(?<![A-Za-z0-9])" + core + r"(?![A-Za-z0-9])")


def excerpt(text, subs, prods, limit=3):
    sents = SENT.split(text)
    st = [tok_re(t) for n in subs for t in name_tokens(n)]
    pt = [tok_re(t) for n in prods for t in name_tokens(n)]
    has = lambda s, rx: any(r.search(s) for r in rx)  # noqa: E731
    glossary = lambda s: s.count(";") >= 4  # abbreviation lists name everything  # noqa: E731
    both = [s for s in sents if has(s, st) and has(s, pt)]
    pick = ([s for s in both if not glossary(s)] or [s for s in sents if has(s, pt) and not glossary(s)]
            or [s for s in sents if has(s, st) and not glossary(s)] or both)
    out = pick[:limit]
    assert all(s in text for s in out)
    how = "substrate+product" if out and out[0] in both else "product-or-substrate" if out else "none"
    return out, how


# ---------------------------------------------------------------- per-reaction audit
paper_meta = P["PMC:PMC13364333"]
latest_decision = {d["claim_identity"]: d for d in decisions_top}
species_in_span = lambda e: {k: len(re.findall(k, spans[e]["text"])) for k in ("Marchantia", "M. polymorpha", "Arabidopsis")}  # noqa: E731
rx_rows = []
for r in sorted(reactions, key=lambda r: min(i for c in r["claim_ids"] for i in cid_to_idx[c])):
    idxs = sorted(i for c in r["claim_ids"] for i in cid_to_idx[c])
    dropped = [d for d in drops if d["claim_index"] in idxs]
    denz = [d["value"] for d in dropped if d["field"] == "enzyme"]
    dec = latest_decision.get(r["claim_identity"], {})
    ev = []
    for e in r["evidence_ids"]:
        ex, how = excerpt(spans[e]["text"], r["substrate_names"], r["product_names"])
        ev.append({"evidence_id": e, "section": spans[e]["section"], "locator": spans[e]["locator"],
                   "char_start": spans[e]["char_start"], "char_end": spans[e]["char_end"],
                   "excerpt_sentences_verbatim": ex, "excerpt_match": how,
                   "species_mentions_in_span": species_in_span(e)})
    claim = next(c for c in claims_top if c["claim_id"] in r["claim_ids"])
    rid = r["reaction_id"]
    label = {x["reaction_id"]: x["label"] for x in reactions}
    conns = [{"other_reaction_id": c["right_reaction_id"] if c["left_reaction_id"] == rid else c["left_reaction_id"],
              "other_equation": label[c["right_reaction_id"] if c["left_reaction_id"] == rid else c["left_reaction_id"]],
              "via": c["via_entity_name"], "kind": c["kind"],
              "same_component": comp_of[c["left_reaction_id"]] == comp_of[c["right_reaction_id"]]}
             for c in state["graph"]["connections"] if rid in (c["left_reaction_id"], c["right_reaction_id"])]
    rx_rows.append({
        "graph_connections": conns,
        "reaction_id": r["reaction_id"],
        "graph_status": r["graph_status"],
        "component_id": comp_of[r["reaction_id"]],
        "equation": r["label"],
        "substrates": r["substrate_names"], "products": r["product_names"],
        "direction": r["direction"], "reversible": r["reversible"],
        "enzymes_retained": r["enzyme_names"], "gene_retained": r["gene"],
        "dropped_uncited_values": [{"claim_index": d["claim_index"], "field": d["field"], "value": d["value"]} for d in dropped],
        "model_claim_indices": idxs,
        "claim_ids": r["claim_ids"],
        "species_state": r["species_state"], "species_value": r["species_text"],
        "species_reason": claim["species_reason"],
        "source_paper": {"paper_id": paper_meta["paper_id"], "pmcid": paper_meta["pmcid"],
                         "pmid": paper_meta["pmid"], "doi": paper_meta["doi"], "url": paper_meta["url"],
                         "title": paper_meta["title"], "year": paper_meta["year"],
                         "evidence_type": paper_meta["evidence_type"]},
        "evidence": ev,
        "seed_connection": seed_connection(r, denz),
        "decision_status": r["decision_status"], "decision_reason": r["decision_reason"],
        "decision_id": dec.get("decision_id", ""), "evidence_level": dec.get("evidence_level", ""),
        "quarantine_reason": r["quarantine_reason"],
        "history_events": [h["event"] for h in r["history"]],
    })
assert len(rx_rows) == 19 and primary["reaction_count"] == 0

# ---------------------------------------------------------------- screen
FN_RE = re.compile(r"\b(LOX\d*|ZmLOX\d*|lipoxygenase|AOS\d*|allene oxide|AOC\d*|OPR\d*|ACX\d*|JAR1|"
                   r"13-HPOT|OPDA|oxo-?phytodienoic)", re.I)
OXY_BIO_RE = re.compile(r"oxylipin.*(biosynth|convert|synthes|formation)|(biosynth|convert|synthes|formation).*oxylipin", re.I)


def screen_flag(p):
    if not p["exclusion_reasons"]:
        return "", []
    hits = sorted({h if isinstance(h, str) else h[0] for h in FN_RE.findall(p["title"])}, key=str.lower)
    if OXY_BIO_RE.search(p["title"]):
        hits.append("oxylipin with conversion/biosynthesis wording")
    if hits:
        return "possible screen false negative", hits
    if re.search(r"jasmonate biosynthe", p["title"], re.I):
        return "borderline (names jasmonate biosynthesis, no enzyme/intermediate term)", []
    return "", []


screen = []
for p in papers:
    flag, hits = screen_flag(p)
    screen.append({"paper_id": p["paper_id"], "title": p["title"], "year": p["year"],
                   "discovered_round": p["discovered_round"],
                   "discovered_via_query_id": p["discovered_via_query_id"],
                   "decision": "included" if not p["exclusion_reasons"] else "excluded",
                   "screening_outcome": p["screening_outcome"], "screening_score": p["screening_score"],
                   "exact_reason": (p["exclusion_reasons"] or p["inclusion_reasons"] or [""])[0],
                   "flag": flag, "flag_terms": hits})

# ---------------------------------------------------------------- round-2 diagnosis via the acquire cache
cache_by_query, cache_mtime = {}, {}
for f in sorted(CACHE.glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    cache_by_query[d["query"]["europepmc_query"]] = d
    cache_mtime[d["query"]["europepmc_query"]] = dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")


def pid(cand_id):
    s = str(cand_id).strip()
    if s.upper().startswith("PMC"):
        return "PMC:" + s.upper()
    if s.isdigit():
        return "PMID:" + s
    return "DOI:" + s


def target_terms(item):
    t = item["target"]
    if item["kind"] == "bridge_search":
        t = t.split("|")[0]
    parts = re.split(r"\s*(?:->|\+)\s*", t)
    return {x for p in parts for x in name_tokens(p.strip())}


fr2 = {f["frontier_item_id"]: f for f in r2["input_frontier"]}
seen, q_rows, slot_sources = set(), [], collections.Counter()
r1_ids = []
for q in r1["queries"]:
    ids = [pid(c["id"]) for c in cache_by_query[q["query_text"]]["candidates"]]
    r1_ids.append(ids)
    seen |= set(ids)
seen_after_r1 = set(seen)
r1_texts = {q["query_text"] for q in r1["queries"]}
returned_count = collections.Counter()
for q in r2["queries"]:
    cands = cache_by_query[q["query_text"]]["candidates"]
    ids = [pid(c["id"]) for c in cands]
    slot_sources.update(c["source"] for c in cands)
    returned_count.update(ids)
    new = [i for i in ids if i not in seen]
    item = fr2[q["frontier_item_id"]]
    terms = target_terms(item)
    spec = sum(any(squash(t) in squash(c["title"] + " " + c["abstract"]) for t in terms) for c in cands)
    q_rows.append({
        "query_id": q["query_id"], "frontier_kind": q["frontier_kind"], "target": item["target"],
        "result_count": q["result_count"], "returned_ids": ids,
        "known_before_query": len(ids) - len(new), "new_to_run": new,
        "new_screen_outcomes": {i: (P[i]["exclusion_reasons"] or ["INCLUDED"])[0] for i in new},
        "results_mentioning_frontier_target": spec,
        "identical_text_to_round1_query": q["query_text"] in r1_texts,
        "new_paper_ids_recorded": q["new_paper_ids"], "cache_hit_recorded": q["cache_hit"],
        "cache_file_mtime_local": cache_mtime[q["query_text"]],
    })
    seen |= set(ids)
r2_new = sorted(seen - seen_after_r1)
r2_unique = sorted({i for row in q_rows for i in row["returned_ids"]})
disc_consistency = all(P[i]["discovered_round"] == 2 and
                       next(r["query_id"] for r in q_rows if i in r["new_to_run"]) == P[i]["discovered_via_query_id"]
                       for i in r2_new)
top_recurring = [{"paper_id": i, "returned_by_n_round2_queries": n, "title": P[i]["title"] if i in P else "",
                  "screen": (P[i]["exclusion_reasons"] or ["INCLUDED"])[0] if i in P else ""}
                 for i, n in returned_count.most_common(12)]
spec_hist = collections.Counter(r["results_mentioning_frontier_target"] for r in q_rows)
distinct_lists = len({tuple(sorted(r["returned_ids"])) for r in q_rows})
round2 = {
    "queries_executed": len(q_rows),
    "result_count_distribution": dict(collections.Counter(r["result_count"] for r in q_rows)),
    "result_slots": sum(r["result_count"] for r in q_rows),
    "unique_paper_ids_returned": len(r2_unique),
    "unique_ids_already_known_from_round1": len(set(r2_unique) & seen_after_r1),
    "unique_ids_new_to_run": len(r2_new),
    "new_to_run_ids": r2_new,
    "new_to_run_included": sum(1 for i in r2_new if not P[i]["exclusion_reasons"]),
    "new_to_run_excluded": sum(1 for i in r2_new if P[i]["exclusion_reasons"]),
    "queries_returning_any_new_paper": sum(1 for r in q_rows if r["new_to_run"]),
    "queries_with_nonempty_new_paper_ids_recorded": sum(1 for r in q_rows if r["new_paper_ids_recorded"]),
    "distinct_result_sets_among_60_queries": distinct_lists,
    "slots_by_source": dict(slot_sources),
    "queries_by_results_mentioning_their_frontier_target": dict(sorted(spec_hist.items())),
    "queries_identical_to_a_round1_query": [r["query_id"] for r in q_rows if r["identical_text_to_round1_query"]],
    "top_recurring_papers": top_recurring,
    "papers_discovered_ledger_round1_to_round2": [m1["usage"]["papers_discovered"], m2["usage"]["papers_discovered"]],
    "discovery_attribution_consistent_with_papers_jsonl": disc_consistency,
    "cap_per_query": "max_papers=min(10, remaining) (controller.py:393) -> search_candidates cap; results deduped then sliced [:cap] (rag/acquire.py:913-914); EuropePMC is queried first with pageSize 20 (acquire.py:887-894), so its top-10 fill the cap before NCBI results are reached",
    "no_paging": "each frontier item is queried once per round with the same top-10; there is no offset/page parameter (rag/acquire.py:462-467), and a repeat of identical query text is served from the on-disk acquire cache (acquire.py:861-883)",
    "frontier_replay": replay,
    "per_query": q_rows,
}
r1_summary = {"queries": len(r1_ids), "slots": sum(len(x) for x in r1_ids),
              "unique_ids": len(seen_after_r1), "per_query_new": []}
_s = set()
for ids in r1_ids:
    r1_summary["per_query_new"].append(len([i for i in ids if i not in _s]))
    _s |= set(ids)

# ---------------------------------------------------------------- counts (each with its source)
species_states = collections.Counter(r["species_state"] for r in reactions)
claim_species = collections.Counter(c["species_state"] for c in C)
components_x = [(c["left_reaction_id"], c["right_reaction_id"], c["via_entity_name"]) for c in state["graph"]["connections"]]
cross = [x for x in components_x if comp_of.get(x[0]) != comp_of.get(x[1])]
counts = {
    "papers_discovered_retrieved": {"value": summary["papers"]["discovered"], "source": "summary.json papers.discovered; usage.papers_discovered=%d; papers.jsonl rows=%d" % (usage["papers_discovered"], len(papers)),
                                    "note": "metadata retrieved for 21; full text retrieved for %d (papers.jsonl full_text_chars>0)" % sum(1 for p in papers if p["full_text_chars"] > 0)},
    "papers_considered_screened": {"value": summary["papers"]["screened"], "source": "summary.json papers.screened; papers.jsonl rows with screening_outcome set=%d" % sum(1 for p in papers if p["screening_outcome"])},
    "papers_included": {"value": summary["papers"]["included"], "source": "summary.json papers.included; papers.jsonl rows with empty exclusion_reasons"},
    "papers_excluded": {"value": summary["papers"]["excluded"], "source": "summary.json papers.excluded"},
    "papers_processed": {"value": summary["papers"]["processed"], "source": "summary.json papers.processed; papers.jsonl processed_round>=0 (PMC:PMC13364333, round 1); usage.papers_processed=%d" % usage["papers_processed"]},
    "candidate_reactions": {"value": summary["graph"]["candidate_reactions"], "source": "summary.json graph.candidate_reactions; candidate_subgraphs.json sum(len(components[].reactions))=%d in count=%d components" % (len(reactions), cand["count"])},
    "admitted_primary_reactions": {"value": primary["reaction_count"], "source": "primary_graph.json reaction_count; summary.json graph.primary_reactions=%d; primary_component_id=''" % summary["graph"]["primary_reactions"]},
    "rejected_claims_by_reason": {"value": summary["claims"]["rejected_by_reason"], "source": "summary.json claims.rejected_by_reason ({} = no claim has status 'rejected')",
                                  "note": "rounds/round_001/rejections.jsonl holds %d rows, all status %s: the controller logs every NON-ADMITTED claim there (controller.py:591-602). Per extracted claim by species state: %s" % (len(r1["rejections"]), dict(collections.Counter(x["status"] for x in r1["rejections"])), dict(claim_species))},
    "duplicates_merged": {"value": 21 - summary["claims"]["distinct_identities"], "source": "usage.claims_extracted=21 vs summary.json claims.distinct_identities=19; graph_delta round 1 updated_reaction_ids=%d; reaction history event 'duplicate_merged' on %d reactions" % (len(state["deltas"][0]["updated_reaction_ids"]), sum(1 for r in reactions if any(h["event"] == "duplicate_merged" for h in r["history"]))),
                          "note": "claim[14] merged into claim[11] (13-HPOT -> allene oxide (12,13-EOT)) and claim[15] into claim[12] (15-HPEPE -> allene oxide (14,15-EEPE)); same paper, same span ev_089f6733d93f6ac7, identical claim_id; the dropped enzyme values differed (MpAOS1 vs MpAOS2). multi_paper_reactions=%d" % summary["graph"]["multi_paper_reactions"]},
    "species_mismatches_by_state": {"value": dict(species_states), "source": "candidate_subgraphs.json reactions[].species_state (== summary.json graph.species_states)",
                                    "note": "non_target_species=0, multiple_species=0, target_species_supported=0. Per extracted claim (21): %s. The model asserted species 'Marchantia polymorpha' on all 21 claims; every one was dropped as uncited, so no claim reached the non_target_species test" % dict(claim_species)},
    "budget_stops": {"value": "none", "source": "run_manifest.json stop_reason='%s', stop_detail='%s', completion_state='%s'" % (manifest["stop_reason"], manifest["stop_detail"], manifest["completion_state"]),
                     "note": "frontier_exhausted is not a budget; no budget limit was reached (see budget table). The max_frontier_size cap (60) WAS hit at the end of round 1 but is not a stop reason; it silently cut %d generated items" % replay["cut_by_cap_count"]},
    "empty_or_failed_model_calls": {"value": len(summary["model_calls"]["failed"]), "source": "model_calls.jsonl (1 row, status=ok, finish_reason=stop, parse_error='', schema_error=''); summary.json model_calls.by_status=%s" % summary["model_calls"]["by_status"],
                                    "note": "the preflight reachability probe (preflight.json model.reachable: 'model answered (2 chars)') is an extra model call outside model_calls.jsonl and costs.json"},
    "evidence_spans_retrieved_extracted": {"value": {"retrieved": summary["passages"]["retrieved"], "extracted": summary["passages"]["extracted"]}, "source": "summary.json passages; evidence_spans.jsonl rows=%d; retrieval.chunks=%d" % (len(spans), summary["retrieval"]["chunks"]),
                                           "note": "the paper was indexed as exactly 20 chunks and max_passages_per_paper=20, so every chunk was sent (89,689 span chars vs 98,172 paper chars); cited by at least one claim: %d spans" % len({e for c in C for e in c["evidence_ids"]})},
    "uncited_field_drops": {"value": {"total": len(drops), "enzyme": drop_by_field["enzyme"], "gene": drop_by_field["gene"], "species": drop_by_field["species"], "other": len(drops) - drop_by_field["enzyme"] - drop_by_field["gene"] - drop_by_field["species"]},
                            "source": "state.json errors (59 rows) == rounds/round_001/errors.jsonl; paper note '59 asserted field(s) dropped for lack of a valid citation'",
                            "note": "summary.json errors is truncated to 50 and report.md prints 40, with no truncation marker"},
}
budget_table = [
    ("rounds", usage["rounds"], budgets["max_rounds"]),
    ("papers discovered", usage["papers_discovered"], budgets["max_papers_discovered"]),
    ("papers processed", usage["papers_processed"], budgets["max_papers_processed"]),
    ("LLM calls", usage["llm_calls"], budgets["max_llm_calls"]),
    ("input tokens", usage["input_tokens"], budgets["max_input_tokens"]),
    ("output tokens", usage["output_tokens"], budgets["max_output_tokens"]),
    ("est. cost USD", usage["estimated_cost_usd"], budgets["max_cost_usd"]),
    ("wall clock s", usage["elapsed_seconds"], budgets["max_wall_clock_seconds"]),
    ("new reactions in a round", m1["usage"]["last_round_new_reactions"], budgets["max_new_reactions_per_round"]),
    ("candidate reactions", len(reactions), budgets["max_candidate_reactions"]),
    ("candidate components", cand["count"], budgets["max_candidate_components"]),
    ("frontier size (end of round 1)", len(r2["input_frontier"]), budgets["max_frontier_size"]),
    ("unproductive-round streak", usage["unproductive_streak"], budgets["stop_after_unproductive_rounds"]),
]

# ---------------------------------------------------------------- anomalies
not_stated_spans = sorted({e for r in reactions if r["species_state"] == "species_not_stated" for e in r["evidence_ids"]})
ns_marchantia = {e: species_in_span(e) for e in not_stated_spans}
review_quote = re.search(r"This review is[^.]*\.", paper_text).group(0)
anomalies = [
    {"id": "A1", "severity": "medium", "title": "report.md misstates claims extracted",
     "evidence": "report.md: 'Claims extracted: **19** across **19** distinct reaction identities'; usage.claims_extracted=21, rounds/round_001/metrics.json claims=21, rounds/round_001/claims_extracted.jsonl has 21 rows",
     "code": "src/t2pw/mining/report.py:149 ('extracted': len(store.claims)) rendered at report.py:275"},
    {"id": "A2", "severity": "medium", "title": "report.md attributes whole-graph connections/entities to the primary graph",
     "evidence": "report.md: 'Primary graph: **0** reactions, **10** chemical connections, **30** entities'; primary_graph.json has connections=[] and entity_ids=[]; the 10/30 are state.json graph.connections / registry for the candidate reactions (%d of the 10 connections join reactions in DIFFERENT candidate components: %s)" % (len(cross), [x[2] for x in cross]),
     "code": "src/t2pw/mining/report.py:159-160 (len(graph.registry), len(graph.connections)) rendered at report.py:298-299"},
    {"id": "A3", "severity": "low", "title": "error list silently truncated",
     "evidence": "state.json errors has 59 entries; summary.json errors has %d; report.md prints 40; neither says how many were omitted and report.md never states the total drop count" % len(summary["errors"]),
     "code": "src/t2pw/mining/report.py:194 ([:50]) and report.py:409 ([:40])"},
    {"id": "A4", "severity": "medium", "title": "queries.jsonl cache_hit is never set",
     "evidence": "all 63 queries record cache_hit=false; round-2 query %s repeats the exact text of round-1 query qr_4c01525fdec4ab02 (same cache key), so it was served from the acquire cache (its cache file's mtime is %s local, i.e. written in round 1 and not rewritten in round 2); artifacts.md lists 'cache hit' as a key field" % (round2["queries_identical_to_a_round1_query"], next(r["cache_file_mtime_local"] for r in q_rows if r["identical_text_to_round1_query"])),
     "code": "src/t2pw/mining/models.py:573 (default False, no assignment anywhere in src/t2pw/mining); acquisition.py:366-368 calls search() without the status dict that rag/acquire.py:861-883 fills with from_cache"},
    {"id": "A5", "severity": "high", "title": "screened-out papers vanish from query and round provenance",
     "evidence": "9 papers have discovered_round=2 in papers.jsonl, yet every round-2 query records new_paper_ids=[], rounds/round_002/papers_considered.jsonl is empty and round_002 metrics papers_discovered=0 (round_001 metrics says papers_discovered=1 while its usage says 12). artifacts.md says each round records 'papers considered'",
     "code": "acquisition.py:386-389 (excluded papers never appended to query.new_paper_ids; only included ones at acquisition.py:418); controller.py:401-404 and 518-521 (round_papers holds included papers only); controller.py:530 (metrics papers_discovered=len(set(round_papers)))"},
    {"id": "A6", "severity": "high", "title": "species_not_stated label contradicts its own reason on this run",
     "evidence": "8 reactions carry reason 'no species is stated anywhere in the supporting evidence', but their cited spans name Marchantia: %s. species_policy.md defines species_not_stated as 'no species anywhere in the supporting evidence'. The state is computed only from the (dropped) model species field and paper observed_organisms (empty), never from the span text" % ns_marchantia,
     "code": "src/t2pw/mining/species.py:283-287; extraction.py:326-337 (observed = cited species or paper observed_organisms)"},
    {"id": "A7", "severity": "medium", "title": "species mismatch hidden by the citation drop",
     "evidence": "the model asserted species='Marchantia polymorpha' for all 21 claims (21 species drops); with the value dropped, 0 reactions are non_target_species and the census shows only inferred_only/species_not_stated/in_vitro_or_cell_free. The only included paper is a bryophyte (Marchantia) review, not Arabidopsis evidence",
     "code": "extraction.py:296-306 (keep) feeding extraction.py:326-337"},
    {"id": "A8", "severity": "medium", "title": "review article labelled 'unknown' evidence type",
     "evidence": "the included paper's own text says %r; papers.jsonl evidence_type='unknown', decisions evidence_level='single_paper'. evidence_policy.md: 'A reaction supported only by a review gets evidence_level: review_article_only'. The record also has pmid='' and doi='' although the text carries PMID 42452298 and DOI 10.3390/plants15132102 (DOI only survives inside url)" % review_quote,
     "code": "acquisition.py:106-128 (review only from declared article_type or title/abstract[:400] markers); rag/acquire.py:486-496 (EuropePMC pubType not captured)"},
    {"id": "A9", "severity": "high", "title": "'frontier_exhausted' reached with 62 generated questions never asked",
     "evidence": "frontier_replay.json: round 1 generated %d items (109 non-bridge + 12 bridge); the 60-item cap kept 59 plus the carried 12-OPDA seed item and cut %d (%s). They were never queried, never written to any artifact, and were not regenerated in round 2 because generate() derives them only from the round's delta. limits_and_stopping.md lists frontier size as a budget but no stop reason names it" % (replay["nonbridge_generated_uncapped"] + replay["round1_recorded_bridge_items"], replay["cut_by_cap_count"], replay["cut_by_cap_by_kind"]),
     "code": "frontier.py:335-336 and 353-356 (silent cap); frontier.py:222-301 (downstream/upstream/branch/direction/compartment only from 'touched'/delta); controller.py:327-331 (stop)"},
    {"id": "A10", "severity": "low", "title": "bridge frontier targets depend on in-memory order",
     "evidence": "replaying frontier.generate() on the graph restored from state.json anchors bridges differently: recorded-only %s, replay-only %s. A resumed run would issue bridge queries the uninterrupted run never issued, contrary to resume_and_cancellation.md 'a resumed run mints the same id for the same thing'. The live run also produced 12 bridge items for 14 components (two anchor collisions)" % (replay["bridge_targets_recorded_not_in_replay"], replay["bridge_targets_replay_not_recorded"]),
     "code": "frontier.py:173-185 (anchor = comp_names[0] from component reaction order)"},
    {"id": "A11", "severity": "low", "title": "model_calls 'retries' holds the attempt count",
     "evidence": "model_calls.jsonl retries=1 with a single attempt in attempt_log (attempt 1, status ok)",
     "code": "src/t2pw/mining/modelclient.py:341"},
    {"id": "A12", "severity": "low", "title": "raw model output is not persisted",
     "evidence": "model_calls.jsonl keeps only response_hash 'sha256:a5efe667780d2a11' and response_chars=10814; whether the model returned empty or malformed *_evidence_ids for enzyme/gene/species cannot be determined from the artifacts (artifacts.md does not promise the raw text, so this is a diagnosability gap, not a contradiction)",
     "code": "modelclient.py:336-345 (hashes kept, text discarded)"},
    {"id": "A13", "severity": "low", "title": "abstract span has no offsets and carries raw HTML",
     "evidence": "ev_b585c188b3cdd16b (abstract#0): char_start/char_end null, text contains '<i>Marchantia polymorpha</i>' markup and is the metadata abstract, not a substring of the indexed paper text; the other 19 spans match paper_text[char_start:char_end] exactly. evidence_policy.md lists char_start/char_end as preserved per span",
     "code": "src/t2pw/mining/retrieval.py:177 (abstract indexed from record.abstract)"},
    {"id": "A14", "severity": "info", "title": "all candidate components score relevance 0.0",
     "evidence": "candidate_subgraphs.json relevance_score=0.0 for all 14 components, including the one producing (9S,13S)-OPDA; topic relevance reads entity/enzyme/pathway names and every enzyme and pathway_names value is empty",
     "code": "graph_semantics.md 'Choosing the primary component' (topic relevance definition)"},
]

# ---------------------------------------------------------------- round-2 classification text
r2_diag = (
    "A frontier of 60 produced 60 queries, each returning exactly 10 results (600 slots), which collapsed to "
    f"{round2['unique_paper_ids_returned']} unique papers across only {distinct_lists} distinct result sets: "
    f"{round2['unique_ids_already_known_from_round1']} of them were already known from round 1 and "
    f"{round2['unique_ids_new_to_run']} were new. All {round2['new_to_run_excluded']} new papers were excluded "
    "by the eligibility screen, so no paper was loaded and the recorded new_paper_ids are empty (they list included "
    "papers only). Every query is an OR-clause in which the frontier target is one optional term beside the topic "
    "phrase and generic gap words, AND 'Arabidopsis thaliana', capped to the top 10 hits with no paging, so the same "
    "high-ranking jasmonate-biosynthesis papers come back regardless of the frontier item "
    f"({spec_hist.get(0, 0)} of 60 queries returned no result whose title/abstract even mentions the frontier target). "
    "Each item was then marked exhausted after one attempt because it yielded no newly INCLUDED paper, and "
    "generate() had nothing new to add from an empty delta, so the frontier emptied. "
    "'frontier_exhausted' matches its code definition, but the literature was not exhausted: the items were "
    "never asked specifically, and 62 more generated items had already been cut by the frontier cap."
)
classification = "product_defect"
classification_basis = [
    "src/t2pw/mining/queries.py:108-118 + src/t2pw/rag/acquire.py:371-402: the frontier target is OR-ed with the run topic "
    "and the kind's generic gap terms, so the query does not require the frontier item's own term",
    "src/t2pw/mining/controller.py:393 + src/t2pw/rag/acquire.py:887-914: top-10 per query, EuropePMC first, no paging",
    "src/t2pw/mining/controller.py:406-409: an item is marked exhausted when it yields no newly INCLUDED paper, so "
    "'10 results, all already seen or screened out' after one attempt counts as exhausted",
    "src/t2pw/mining/controller.py:327-331: sets STOP_FRONTIER_EXHAUSTED when the post-merge frontier is empty",
    "src/t2pw/mining/frontier.py:335-336, 353-356: 62 generated items cut by the cap and never recorded or regenerated",
]

# ---------------------------------------------------------------- uncited-drop assessment
n_cited = sum(1 for d in drops if d["presence"]["exact"]["in_claim_cited_spans"])
n_shown = sum(1 for d in drops if d["presence"]["exact"]["in_any_shown_span"])
n_paper = sum(1 for d in drops if d["presence"]["exact"]["in_full_paper_text"])
drops_assessment = (
    f"All 59 drops were checked (sample requirement: 10). Exact dropped value present in the claim's own cited "
    f"passage(s): {n_cited}/59; in any of the 20 passages shown to the model: {n_shown}/59; anywhere in the paper "
    f"text: {n_paper}/59. By field: " + "; ".join(f"{f}: " + ", ".join(
        f"{sum(1 for d in drops if d['field']==f and d['presence']['exact']['in_claim_cited_spans'])} in cited span / "
        f"{sum(1 for d in drops if d['field']==f and d['presence']['exact']['in_any_shown_span'])} in a shown span / "
        f"{sum(1 for d in drops if d['field']==f)} dropped" for _ in [0]) for f in ("enzyme", "gene", "species")) +
    ". The check (extraction.py:174-183 _valid_ids, 271-278 cited(), 296-306 keep()) only tests that the field's "
    "*_evidence_ids list is non-empty and names an id that was shown; it never inspects passage content, so it "
    "cannot be 'too strict' about content. The same response cited valid ids for every substrate and product "
    "(field_evidence has substrates/products on all 21 claims), yet 0 of 21 claims kept any enzyme, gene or species "
    "citation. Pattern: the model failed to fill the per-field *_evidence_ids lists (or filled them with ids "
    "that do not match exactly; the raw response is not persisted, so which of the two cannot be shown). "
    "Where the value is absent from every shown passage, the drop is correct enforcement."
)

key_findings = [
    f"Nothing was admitted because the one processed paper is a Marchantia review: every reaction is a species-held lead "
    f"({dict(species_states)}), and 0 are target-species evidence. Seed-relevant leads do exist: "
    f"{sum(1 for r in rx_rows if r['seed_connection']['tier'] in ('seed', 'aos_canonical'))} direct (the AOS step "
    "13-HPOT -> 12,13-EOT and the AOC step 12,13-EOT -> (9S,13S)-OPDA, chemically connected in the graph but split "
    "across two quarantine components).",
    f"All {len(drops)} enzyme/gene/species values were dropped as uncited, yet every one occurs (case- and "
    f"whitespace-insensitive) in a passage the model was shown ({n_cited} in the claim's own cited passage). The check "
    "behaved as written; no valid per-field citation reached it (the raw response is not persisted, so empty vs "
    "malformed id lists cannot be told apart).",
    f"Round 2 asked 60 questions and got 600 result slots back, but only {round2['unique_paper_ids_returned']} unique "
    f"papers ({round2['unique_ids_new_to_run']} new, all screened out); {spec_hist.get(0, 0)} of 60 result lists never "
    "mention the frontier target. Classified product_defect (query formulation + top-10/no paging + one-shot "
    "exhaustion rule).",
    f"The screen excluded {len([s for s in screen if s['flag'] == 'possible screen false negative'])} titles that "
    "plainly name LOX/AOS/OPDA/oxylipin biosynthesis, including PMC12280011 on 12-OPDA metabolites in Arabidopsis "
    "thaliana itself.",
    "report.md misstates two numbers (claims extracted; primary-graph connections/entities) and several provenance "
    "fields are empty or never set (Section 6).",
]

# ---------------------------------------------------------------- JSON
audit = {
    "audit": "ORCH-742a pilot-001 evaluation (read-only)",
    "run_dir": str(RUN), "code": f"{CODE} @ {manifest['git']['commit']}",
    "generated_by": ["frontier_replay.py", "build_audit.py"],
    "headline": {"stop_reason": manifest["stop_reason"], "stop_detail": manifest["stop_detail"],
                 "completion_state": manifest["completion_state"], "rounds": manifest["rounds_completed"],
                 "elapsed_seconds": usage["elapsed_seconds"], "estimated_cost_usd": usage["estimated_cost_usd"]},
    "key_findings": key_findings,
    "counts": counts,
    "budget_table": [{"budget": b, "used": u, "limit": l} for b, u, l in budget_table],
    "reactions": rx_rows,
    "screen": screen,
    "screen_false_negatives": [s for s in screen if s["flag"] == "possible screen false negative"],
    "uncited_drops": {"counts": dict(drop_by_field), "verdict_counts": dict(verdict_counts),
                      "verdict_by_field": {f"{k[0]}|{k[1]}": v for k, v in verdict_by_field.items()},
                      "check_code": ["src/t2pw/mining/extraction.py:174-183", "src/t2pw/mining/extraction.py:271-285",
                                     "src/t2pw/mining/extraction.py:296-306"],
                      "assessment": drops_assessment, "rows": drops},
    "round1_search": r1_summary,
    "round2": round2,
    "round2_diagnosis": r2_diag,
    "round2_classification": classification,
    "round2_classification_basis": classification_basis,
    "anomalies": anomalies,
    "cited_spans_verbatim": {e: spans[e] for e in sorted({e for r in reactions for e in r["evidence_ids"]})},
}
(OUT / "pilot_001_audit.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")

# ---------------------------------------------------------------- Markdown
L = []
a = L.append
esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")  # noqa: E731
a("# PILOT-001 audit: pathway mining, jasmonate biosynthesis, *Arabidopsis thaliana*\n")
a(f"Read-only evaluation of `{RUN}` (code `{CODE}` @ `{manifest['git']['commit'][:8]}`). "
  "Every number below is taken from a named artifact field. Nothing was re-run and no model was called. "
  "Rebuild with `frontier_replay.py` (PYTHONPATH=C:/t/mining-pilot/src, PYTHONDONTWRITEBYTECODE=1), then "
  "`build_audit.py`, both through `bounded_run.py`. Machine-readable twin: `pilot_001_audit.json`.\n")
a(f"**Headline.** `{manifest['completion_state']}` / `{manifest['stop_reason']}` (\"{manifest['stop_detail']}\"), "
  f"{manifest['rounds_completed']} rounds, {usage['elapsed_seconds']} s, est. ${usage['estimated_cost_usd']}. "
  "1 of 21 papers included (a *Marchantia polymorpha* review), 21 claims -> 19 candidate reactions in 14 "
  "quarantined components, **0 admitted reactions**.\n")

a("**Key findings.**\n")
for k in key_findings:
    a("- " + k)
a("")
a("## 1. Pre-registered counts\n")
a("| metric | value | artifact field | note |\n|---|---|---|---|")
for k, v in counts.items():
    a(f"| {k} | `{esc(json.dumps(v['value'], ensure_ascii=False))}` | {esc(v['source'])} | {esc(v.get('note', ''))} |")
a("\nBudgets (used / limit, from `summary.json` usage and `run_manifest.json` config.budgets):\n")
a("| budget | used | limit |\n|---|---|---|")
for b, u, l in budget_table:
    a(f"| {b} | {u} | {l} |")

a("\n## 2. Per-reaction audit (all 19; primary graph is empty)\n")
a("Seed connection is the auditor's name-level assessment; the run graph contains no seed entity, so no graph edge "
  "links any reaction to 12-OPDA or AOS. Every enzyme/gene/species value the model asserted was dropped as uncited; "
  "dropped values are shown only to explain what was lost.\n")
a("| # | reaction_id | component | equation | species state | enzyme kept | dropped enzyme (uncited) | seed connection | decision |\n|---|---|---|---|---|---|---|---|---|")
for n, r in enumerate(rx_rows, 1):
    de = sorted({d["value"] for d in r["dropped_uncited_values"] if d["field"] == "enzyme"})
    a(f"| {n} | `{r['reaction_id']}` | `{r['component_id']}` | {esc(r['equation'])} | `{r['species_state']}` | "
      f"{', '.join(r['enzymes_retained']) or 'none'} | {', '.join(de) or '-'} | {esc(r['seed_connection']['summary'])} | `{r['decision_status']}` |")
for n, r in enumerate(rx_rows, 1):
    a(f"\n### {n}. `{r['reaction_id']}` - {esc(r['equation'])}\n")
    a(f"- **Component:** `{r['component_id']}` (quarantine_reason `{r['quarantine_reason']}`); history: {', '.join(r['history_events'])}")
    a(f"- **Claims:** {', '.join('`'+c+'`' for c in r['claim_ids'])} (model output index {r['model_claim_indices']})")
    a(f"- **Enzyme retained:** {', '.join(r['enzymes_retained']) or 'none'}; **gene retained:** {r['gene_retained'] or 'none'}")
    if r["dropped_uncited_values"]:
        a("- **Dropped as uncited:** " + "; ".join(f"claim[{d['claim_index']}].{d['field']}='{d['value']}'" for d in r["dropped_uncited_values"]))
    a(f"- **Species:** state `{r['species_state']}`, value `{r['species_value'] or '(empty)'}`; reason: \"{r['species_reason']}\"")
    sp = r["source_paper"]
    a(f"- **Source paper:** {sp['paper_id']} (PMCID {sp['pmcid']}; PMID {sp['pmid'] or '(not recorded)'}; DOI {sp['doi'] or '(not recorded; url ' + sp['url'] + ')'}) - \"{sp['title']}\" ({sp['year']}), evidence_type `{sp['evidence_type']}`")
    sc = r["seed_connection"]
    a(f"- **Seed connection:** {sc['summary']}. Via: {'; '.join(sc['shared_chain_metabolites']) or 'no chain metabolite'}. AOS: {sc['involves_AOS_enzyme']}.")
    a("- **In-graph chemical connections (state.json graph.connections):** " + (
        "; ".join(f"{c['kind']} via {c['via']} with `{c['other_reaction_id']}` ({c['other_equation']}), "
                  f"{'same' if c['same_component'] else 'DIFFERENT'} component" for c in r["graph_connections"])
        or "none"))
    a(f"- **Decision:** `{r['decision_status']}` (`{r['decision_id']}`, evidence_level `{r['evidence_level']}`): \"{r['decision_reason']}\"")
    for e in r["evidence"]:
        a(f"- **Span `{e['evidence_id']}`** ({e['section']}, {e['locator']}, chars {e['char_start']}-{e['char_end']}; "
          f"species mentions {e['species_mentions_in_span']}; excerpt = {e['excerpt_match']} sentences, verbatim; full text in Appendix A):")
        for s in e["excerpt_sentences_verbatim"]:
            a(f"  > {s}")

a("\n## 3. Screen: all 21 papers\n")
a("Flag rule (title only, nothing re-run): 'possible screen false negative' when an excluded title names LOX / "
  "lipoxygenase / AOS / allene oxide / AOC / OPR / ACX / JAR1 / 13-HPOT / OPDA / oxo-phytodienoic, or 'oxylipin' "
  "with biosynthesis/conversion wording; "
  "'borderline' when it names jasmonate biosynthesis without such a term.\n")
a("| paper_id | year | round | decision | score | exact reason | flag | title |\n|---|---|---|---|---|---|---|---|")
for s in screen:
    flag = s["flag"] + (f" ({', '.join(s['flag_terms'])})" if s["flag_terms"] else "")
    a(f"| {s['paper_id']} | {s['year']} | {s['discovered_round']} | {s['decision']} | {s['screening_score']} | {esc(s['exact_reason'])} | {esc(flag)} | {esc(s['title'])} |")

a("\n## 4. Uncited-field drops\n")
a(f"Counts by field: {dict(drop_by_field)} (total {len(drops)}; other fields 0).\n")
a("Code that performs the check (frozen worktree): `src/t2pw/mining/extraction.py:174-183` (`_valid_ids`: keeps only ids "
  "that were shown), `:271-285` (`cited()` per field), `:296-306` (`keep()`: a non-empty value with no valid id is "
  "blanked and logged as `(no valid citation)`).\n")
a(drops_assessment + "\n")
a("| claim | field | dropped value | claim-cited spans | exact value in cited span | in any shown span | in paper text | looser form found in cited span | verdict |\n|---|---|---|---|---|---|---|---|---|")
for d in drops:
    p = d["presence"]
    loose = "; ".join(f"{k} '{v['value']}': {len(v['in_claim_cited_spans'])}" for k, v in p.items() if k != "exact") or "-"
    a(f"| {d['claim_index']} | {d['field']} | `{d['value']}` | {', '.join(d['claim_cited_evidence_ids'])} | "
      f"{', '.join(p['exact']['in_claim_cited_spans']) or 'no'} | {len(p['exact']['in_any_shown_span'])} of 20 | "
      f"{'yes' if p['exact']['in_full_paper_text'] else 'no'} | {esc(loose)} | {d['verdict']} |")

a("\n## 5. Round 2 diagnosis\n")
a(r2_diag + "\n")
a(f"**Classification: `{classification}`.** Basis:\n")
for b in classification_basis:
    a(f"- {b}")
a("\nNumbers (from `rounds/round_002/queries.jsonl`, `input_frontier.jsonl`, and the run's own acquire-cache files "
  "`C:/t/mining-pilot/data/rag_index/acquire_cache/*.json`, matched on `europepmc_query == query_text`):\n")
for k in ("queries_executed", "result_count_distribution", "result_slots", "unique_paper_ids_returned",
          "unique_ids_already_known_from_round1", "unique_ids_new_to_run", "new_to_run_included", "new_to_run_excluded",
          "queries_returning_any_new_paper", "queries_with_nonempty_new_paper_ids_recorded",
          "distinct_result_sets_among_60_queries", "slots_by_source",
          "queries_by_results_mentioning_their_frontier_target", "queries_identical_to_a_round1_query",
          "papers_discovered_ledger_round1_to_round2", "discovery_attribution_consistent_with_papers_jsonl"):
    a(f"- `{k}`: {json.dumps(round2[k], ensure_ascii=False)}")
a(f"- round 1 for comparison: {r1_summary['queries']} queries, {r1_summary['slots']} slots, {r1_summary['unique_ids']} unique papers, new per query {r1_summary['per_query_new']}")
a(f"- cap: {round2['cap_per_query']}")
a(f"- paging: {round2['no_paging']}")
a(f"- frontier cap (replay): {replay['nonbridge_generated_uncapped'] + replay['round1_recorded_bridge_items']} items generated at the end of round 1, 59 kept + 1 carried, {replay['cut_by_cap_count']} cut {replay['cut_by_cap_by_kind']}; see `frontier_replay.json`.")
a("\nMost-returned papers in round 2:\n")
a("| paper_id | returned by n of 60 queries | screen | title |\n|---|---|---|---|")
for t in top_recurring:
    a(f"| {t['paper_id']} | {t['returned_by_n_round2_queries']} | {esc(t['screen'])} | {esc(t['title'])} |")
a("\nPer query:\n")
a("| query_id | kind | target | results | known | new (screen) | results naming target | same as round-1 text | recorded new_paper_ids | recorded cache_hit |\n|---|---|---|---|---|---|---|---|---|---|")
for r in q_rows:
    newtxt = "; ".join(f"{i} ({v.split(':')[0]})" for i, v in r["new_screen_outcomes"].items()) or "-"
    a(f"| {r['query_id']} | {r['frontier_kind']} | {esc(r['target'])} | {r['result_count']} | {r['known_before_query']} | "
      f"{esc(newtxt)} | {r['results_mentioning_frontier_target']} | {'yes' if r['identical_text_to_round1_query'] else 'no'} | "
      f"{r['new_paper_ids_recorded']} | {r['cache_hit_recorded']} |")
a("\n**Is `frontier_exhausted` accurate?** It is set at `src/t2pw/mining/controller.py:327-331` when the merged "
  "frontier is empty. It is literally true (all 60 items carry exhausted=true; `frontier.jsonl` is empty), but "
  "misleading as a description of the run: 'exhausted' means 'one top-10 search returned no newly included paper' "
  "(controller.py:406-409), not 'no more literature', and 62 generated items were discarded by the cap without "
  "ever being asked (A9).\n")

a("## 6. Other anomalies\n")
a("| id | severity | finding | evidence | code |\n|---|---|---|---|---|")
for x in anomalies:
    a(f"| {x['id']} | {x['severity']} | {esc(x['title'])} | {esc(x['evidence'])} | {esc(x['code'])} |")

a("\n## Appendix A: cited evidence spans, verbatim\n")
a("Exact `text` from `evidence_spans.jsonl`, one block per span cited by a reaction.\n")
for e, s in audit["cited_spans_verbatim"].items():
    a(f"<details><summary><code>{e}</code> - {s['section']} / {s['locator']} (chars {s['char_start']}-{s['char_end']})</summary>\n")
    a("```text\n" + s["text"] + "\n```\n</details>\n")

(OUT / "PILOT-001-AUDIT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("reactions", len(rx_rows), "drops", len(drops), "screen FN", len(audit["screen_false_negatives"]),
      "r2 unique", round2["unique_paper_ids_returned"], "r2 new", round2["unique_ids_new_to_run"],
      "distinct sets", distinct_lists, "spec hist", dict(spec_hist), "verdicts", dict(verdict_counts))

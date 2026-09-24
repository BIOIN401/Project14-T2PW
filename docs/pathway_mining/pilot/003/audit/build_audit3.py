"""PILOT-003 audit builder (ORCH-743 iteration 2). Read-only over the run directory and the frozen tree.

Run through C:/t/mining-pilot3/.../bounded_run.py with PYTHONPATH=C:/t/mining-pilot3/src and
PYTHONDONTWRITEBYTECODE=1. Writes only PILOT-003-AUDIT.md and pilot_003_audit.json next to this file.
No model call, no network. Frozen modules are imported only for pure functions: the title-organism
detector, the full-text title check, the species marker tests, and frontier.generate (replayed on
state.json with the frontier cap lifted, to count what the cap cut).
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from types import SimpleNamespace

RUN = Path("C:/t/mining-pilot-runs3/pilot-003-jasmonate-ath")
LOG = Path("C:/t/mining-pilot-runs3/pilot-003.log")
P1 = Path("C:/t/mining-pilot/docs/pathway_mining/pilot/001/run")
P2 = Path("C:/t/mining-pilot-runs2/pilot-002-jasmonate-ath")
P2_AUDIT = Path("C:/t/mining-pilot-runs2/audit/pilot_002_audit.json")
OUT = Path(__file__).resolve().parent
CHARS_PER_TOKEN = 3.5  # ponytail: content-token estimate for JSON answers; the provider reports only the total


def jl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


summary = js(RUN / "summary.json")
manifest = js(RUN / "run_manifest.json")
config = js(RUN / "config.json")
costs = js(RUN / "costs.json")
state = js(RUN / "state.json")
papers = OrderedDict((r["paper_id"], r) for r in jl(RUN / "papers.jsonl"))
claims = jl(RUN / "claims.jsonl")
extracted = jl(RUN / "rounds/round_001/claims_extracted.jsonl")
decisions = jl(RUN / "decisions.jsonl")
calls = jl(RUN / "model_calls.jsonl")
raw_rows = jl(RUN / "rounds/round_001/model_raw.jsonl")
queries = jl(RUN / "queries.jsonl")
evidence = {r["evidence_id"]: r for r in jl(RUN / "evidence_spans.jsonl")}
cands = js(RUN / "candidate_subgraphs.json")
primary = js(RUN / "primary_graph.json")
frontier_final = jl(RUN / "frontier.jsonl")
round1 = js(RUN / "rounds/round_001/round.json")
log_text = LOG.read_text(encoding="utf-8", errors="replace")
graph = state["graph"]
budgets = config["budgets"]
MAX_TOKENS = config["model"]["max_tokens"]

from t2pw.mining import frontier as fm  # noqa: E402
from t2pw.mining.acquisition import Acquirer, _SENTENCE, full_text_matches_title  # noqa: E402
from t2pw.mining.claims import ClaimStore  # noqa: E402
from t2pw.mining.graph import ResearchGraph  # noqa: E402
from t2pw.mining.models import FrontierItem, GraphDelta  # noqa: E402
from t2pw.mining.species import _IN_VITRO_MARKERS, detect_inference_markers  # noqa: E402

acq = object.__new__(Acquirer)
acq.config = SimpleNamespace(topic=config["topic"], target_species=config["target_species"],
                             target_species_normalized=config["target_species_normalized"])

reactions = OrderedDict()
for comp in cands["components"]:
    for r in comp["reactions"]:
        reactions[r["reaction_id"]] = dict(r, _component=comp)
claims_by_id = {c["claim_id"]: c for c in claims}
dec_by_claim = defaultdict(list)
for d in decisions:
    dec_by_claim[d["claim_id"]].append(d)
conn_by_rx = defaultdict(list)
for c in graph["connections"]:
    conn_by_rx[c["left_reaction_id"]].append(c)
    conn_by_rx[c["right_reaction_id"]].append(c)
included = [p for p in papers.values() if not p["exclusion_reasons"]]


def route_of(p):
    if p["screening_outcome"] != "mining_rescue":
        return "production"
    reason = " ".join(p["inclusion_reasons"])
    if "pathway-anchor rule off" in reason:
        return "mining_rescue route 1"
    if "in one sentence" in reason:
        return "mining_rescue route 2"
    return "mining_rescue (unclassified)"


def title_key(t):
    t = re.sub(r"<[^>]+>", "", html.unescape(html.unescape(t or "")))
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def best_sentence(text, names):
    """The sentence of ``text`` naming the most participant names (verbatim substring)."""
    keys = []
    for n in names:
        base = re.sub(r"\s*\([^()]*\)\s*$", "", n).strip()
        for k in {n, base}:
            if len(k) >= 2:
                keys.append(k.lower())
    best, score = "", 0
    for s in _SENTENCE.split(text):
        low = s.lower()
        hit = sum(1 for k in set(keys) if k in low)
        if hit > score:
            best, score = s, hit
    assert not best or best in text
    return best, score


def marker_sentence(text, markers):
    for s in _SENTENCE.split(text):
        if any(m in s.lower() for m in markers):
            return s
    return ""


# ================================================================= 1. counts
raw_by_call = defaultdict(list)
for r in raw_rows:
    raw_by_call[r["call_id"]].append(r)
provider_requests = sum(len(r["provider_requests"]) for r in raw_rows)
assert len(raw_rows) == summary["usage"]["llm_calls"]
assert sum(c["input_tokens"] for c in calls) == costs["input_tokens"]
assert sum(c["output_tokens"] for c in calls) == costs["output_tokens"]
retry_lines = re.findall(r"finish_reason=(\w+)\) on attempt (\d)/(\d); retrying", log_text)
giving_up = log_text.count("giving up and returning")
dup_events = [(rid, h) for rid, r in reactions.items() for h in r["history"] if h["event"] == "duplicate_merged"]
attr_extracted = Counter((f, h) for c in extracted for f, h in c["field_attribution"].items())
attr_stored = Counter((f, h) for c in claims for f, h in c["field_attribution"].items())
rejections = round1["rejections"]

counts = OrderedDict()


def put(key, value, source, note=""):
    counts[key] = {"value": value, "source": source, "note": note}


put("papers_discovered", len(papers), "summary papers.discovered=%d; usage.papers_discovered=%d; papers.jsonl rows=%d"
    % (summary["papers"]["discovered"], summary["usage"]["papers_discovered"], len(papers)),
    "by source %s; distinct normalized titles %d" % (dict(Counter(p["acquisition_source"] for p in papers.values())),
                                                    len({title_key(p["title"]) for p in papers.values()})))
put("papers_screened", sum(1 for p in papers.values() if p["screening_outcome"]), "summary papers.screened=%d" % summary["papers"]["screened"])
put("papers_included", len(included), "summary papers.included=%d; papers.jsonl empty exclusion_reasons" % summary["papers"]["included"],
    "by route %s; by source %s" % (dict(Counter(route_of(p) for p in included)), dict(Counter(p["acquisition_source"] for p in included))))
put("papers_excluded", summary["papers"]["excluded"], "summary papers.excluded")
put("papers_processed", sum(1 for p in included if p["processed_round"] >= 0),
    "papers.jsonl processed_round>=0; usage.papers_processed=%d" % summary["usage"]["papers_processed"])
put("claims_extracted", len(extracted), "usage.claims_extracted=%d; rounds/round_001/claims_extracted.jsonl rows=%d"
    % (summary["usage"]["claims_extracted"], len(extracted)),
    "claims.jsonl stored=%d (= summary claims.extracted=%d, the STORED count); distinct identities=%d; latest status %s"
    % (len(claims), summary["claims"]["extracted"], summary["claims"]["distinct_identities"], summary["claims"]["by_latest_status"]))
put("primary_reactions", primary["reaction_count"], "primary_graph.json reaction_count; summary graph.primary_reactions=%d" % summary["graph"]["primary_reactions"])
put("candidate_reactions", len(reactions), "candidate_subgraphs.json reactions over count=%d components; summary graph.candidate_reactions=%d"
    % (cands["count"], summary["graph"]["candidate_reactions"]),
    "decision status %s; the 2 `incomplete` claims (no substrate) have no reaction node"
    % dict(Counter(r["decision_status"] for r in reactions.values())))
put("rejected_by_reason", summary["claims"]["rejected_by_reason"], "summary claims.rejected_by_reason",
    "rejections.jsonl %d rows (every non-admitted claim): %s" % (len(rejections), dict(Counter(x["status"] for x in rejections))))
put("duplicates_merged", len(dup_events), "reaction history 'duplicate_merged'; claims extracted - distinct identities = %d"
    % (summary["usage"]["claims_extracted"] - summary["claims"]["distinct_identities"]),
    "%s, across two batches of the same paper" % [reactions[rid]["label"] for rid, _ in dup_events])
put("species_states_reaction_nodes", dict(Counter(r["species_state"] for r in reactions.values())),
    "candidate_subgraphs.json reactions[].species_state (== summary graph.species_states)",
    "per extracted claim (38): %s" % dict(Counter(c["species_state"] for c in extracted)))
put("budget_stops", manifest["stop_reason"], "run_manifest stop_reason; stop_detail='%s'" % manifest["stop_detail"],
    "the only limit reached; see section 7")
put("model_calls", dict(Counter(c["status"] for c in calls)), "model_calls.jsonl status (%d logical calls = %d ladder rungs = usage.llm_calls)"
    % (len(calls), len(raw_rows)),
    "provider requests %d (model_raw provider_requests); log: %d 'retrying' lines (%s), %d 'giving up'"
    % (provider_requests, len(retry_lines), [f"{a} {b}/{c}" for a, b, c in retry_lines], giving_up))
put("uncited_field_drops", len(state["errors"]), "state.json errors; rounds/round_001/errors.jsonl rows=%d" % len(round1["errors"]))
put("fields_kept_by_attribution", {f"{f}:{h}": n for (f, h), n in sorted(attr_extracted.items())},
    "claims_extracted.jsonl field_attribution (38 claims)",
    "stored claims: %s; verbatim_cited_span = %d" % ({f"{f}:{h}": n for (f, h), n in sorted(attr_stored.items())},
                                                     sum(n for (f, h), n in attr_extracted.items() if h == "verbatim_cited_span")))
put("passages", summary["passages"], "summary passages; evidence_spans.jsonl rows=%d" % len(evidence))

# ================================================================= 2. reactions
SEED = {
    "rx_ce92a2647a229cc6528e": ("upstream, 2 steps", "13-LOX step (makes the AOS substrate 13-HPOT); the pilot-002 primary reaction's chemistry"),
    "rx_41c97e790d2e17c19477": ("upstream, 2 steps", "13-LOX step again (full text of the same 13-LOX paper)"),
    "rx_2889b990927302948b2e": ("upstream, 2 steps", "13-LOX step from the Marchantia review"),
    "rx_6570fd856f1200fa7a05": ("collapsed cascade to the seed", "LOX + AOS + AOC written as one reaction with products 13-HPOT and 12-OPDA; not a single step"),
    "rx_eba8e16b5409af0ff75a": ("upstream, 3 steps", "lipase release of alpha-linolenic acid from galactolipids"),
    "rx_9dc809e3d64d7a6dbcf8": ("upstream precursor", "makes ALA, the 13-LOX substrate (FAD7 desaturation)"),
    "rx_60e130db9d1a0aa23dce": ("DIRECT (makes seed 12-OPDA)", "AOC step, Marchantia MpAOC"),
    "rx_8ffac9975e3a977a75f0": ("DIRECT (makes seed 12-OPDA)", "AOC step again, MpAOC, other passage"),
    "rx_d438a2fdb9dede27f743": ("DIRECT (makes seed 12-OPDA)", "AOC step, moss PpAOC1 (cis-OPDA = 12-OPDA)"),
    "rx_14eb3a43a495d7ce851f": ("DIRECT (consumes seed 12-OPDA)", "OPR3 reduces OPDA to OPC-8:0, the canonical Arabidopsis step"),
    "rx_e818daf35c555fd843da": ("parallel C16 branch", "AOC step to dn-cis-OPDA (bryophyte route)"),
    "rx_f6e5d9f747dbe8e43554": ("parallel C16 branch", "dn-cis-OPDA isomerisation"),
    "rx_d86306a16865c7a08cec": ("parallel C16 branch", "Delta4-dn-OPDA isomerisation"),
    "rx_c8eec14062ae5544f820": ("parallel C16 branch", "dn-iso-OPDA amino-acid conjugates (MpGH3A)"),
    "rx_cbf86a95dc4f74026267": ("parallel C16 branch", "OPR3 on dn-cis-OPDA -> OPC-6:0"),
    "rx_642b4a34498c459fb3f2": ("C20 branch, not the route", "AOC-type cyclisation of a C20 allene oxide (PpAOC2)"),
    "rx_742cac9b32cba9f3bff8": ("downstream, 4 steps", "JA -> JA-Ile by GH3.10 (JAR1-type conjugation)"),
    "rx_4764f832e88bebda27cb": ("downstream, 4 steps", "the same JA -> JA-Ile step written with Ile and ATP"),
    "rx_05534dc3edb77ff33a5e": ("downstream side branch", "JA -> JA-Gln conjugate"),
    "rx_2f53a30ba9d7528b9b3c": ("downstream side branch", "JA -> JA-Asn conjugate"),
    "rx_edc1ad1fa1676f1685fd": ("downstream side branch", "JA -> JA-Glu conjugate"),
    "rx_9a6326c17c7be6ac5412": ("downstream, turnover branch", "12-OH-JA -> 12-OH-JA-Ile (GH3.10)"),
    "rx_90a150f373302278fb2f": ("downstream, turnover branch", "the same 12-OH-JA -> 12-OH-JA-Ile step written with Ile and ATP"),
    "rx_3216468cdedfd13f06ef": ("downstream, turnover branch", "12-OH-JA -> an unnamed conjugate"),
}
OFF = ("off the route", "fatty-acid desaturation/elongation (C16 or C20 LCPUFA), not the 12-OPDA route")
for rid in ("rx_30cb458fb147c3de7897", "rx_3a229b45b4b5cfa65ddc", "rx_0d8d43b8211e3ff44724", "rx_354a031228c7b0e2dc3c",
            "rx_78d3551bbaba2ff2d5f0", "rx_d0557d1ccb1153bdae8a", "rx_a723aaa413ec6bc29766", "rx_9ebcd41f3b14a7c49564",
            "rx_f7233e1139027ba46f53"):
    SEED[rid] = OFF
SEED["rx_3dc94e3847d88fa54c12"] = ("upstream C16 precursor", "makes HTA, the C16 precursor of the dn-OPDA branch")
SEED["rx_eab7ca60441693867cca"] = ("upstream C16 precursor", "HDA -> HTA again, no species")
assert set(SEED) == set(reactions), set(reactions) ^ set(SEED)
#: auditor's name-level grouping of the 35 nodes into distinct conversions (for the stop count)
SAME = [
    ["rx_ce92a2647a229cc6528e", "rx_41c97e790d2e17c19477", "rx_2889b990927302948b2e"],
    ["rx_60e130db9d1a0aa23dce", "rx_8ffac9975e3a977a75f0", "rx_d438a2fdb9dede27f743"],
    ["rx_742cac9b32cba9f3bff8", "rx_4764f832e88bebda27cb"],
    ["rx_9a6326c17c7be6ac5412", "rx_90a150f373302278fb2f"],
    ["rx_3dc94e3847d88fa54c12", "rx_eab7ca60441693867cca"],
]
distinct_conversions = len(reactions) - sum(len(g) - 1 for g in SAME)

reaction_rows = []
for rid, r in reactions.items():
    cl = [claims_by_id[c] for c in r["claim_ids"]]
    c0 = cl[0]
    ev_text = "\n".join(evidence[e]["text"] for e in r["evidence_ids"])
    inf = detect_inference_markers(ev_text)
    vit = [m for m in _IN_VITRO_MARKERS if m in ev_text.lower()]
    names = r["substrate_names"] + r["product_names"]
    first = r["evidence_ids"][0]
    sent, score = best_sentence(evidence[first]["text"], names)
    part = [{"name": p["name"], "role": role, "evidence_ids": p["evidence_ids"]}
            for c in cl for role, ps in (("substrate", c["substrates"]), ("product", c["products"])) for p in ps]
    d = dec_by_claim[c0["claim_id"]][-1]
    reaction_rows.append({
        "reaction_id": rid, "component_id": r["component_id"], "quarantine_reason": r["_component"]["quarantine_reason"],
        "equation": r["label"], "enzymes": [{"name": n, "attribution": c0["field_attribution"].get("enzyme", "")} for n in r["enzyme_names"]],
        "gene": {"value": r["gene"], "attribution": c0["field_attribution"].get("gene", "")},
        "species_state": r["species_state"], "species_value": r["species_text"],
        "species_attribution": c0["field_attribution"].get("species", ""), "species_reason": c0["species_reason"],
        "species_marker": ({"inference": inf, "sentence": marker_sentence(ev_text, inf)} if r["species_state"] == "inferred_only" and inf else
                           {"in_vitro": vit, "sentence": marker_sentence(ev_text, vit)} if r["species_state"] == "in_vitro_or_cell_free" else {}),
        "paper_id": r["paper_ids"][0], "paper_title": papers[r["paper_ids"][0]]["title"],
        "claim_ids": r["claim_ids"], "model_calls": sorted({c["model_call_id"] for c in cl}),
        "evidence_ids": r["evidence_ids"], "excerpt_span": first, "excerpt": sent, "excerpt_names_matched": score,
        "participants": part, "participants_all_cited": all(p["evidence_ids"] for p in part),
        "participants_cited": sum(1 for p in part if p["evidence_ids"]), "participants_total": len(part),
        "seed_connection": SEED[rid][0], "seed_how": SEED[rid][1],
        "graph_neighbours": sorted({(c["kind"], c["via_entity_name"], c["left_reaction_id"] if c["right_reaction_id"] == rid else c["right_reaction_id"])
                                    for c in conn_by_rx[rid]}),
        "decision_status": r["decision_status"], "decision_reason": r["decision_reason"], "decision_id": d["decision_id"],
        "duplicate_merges": sum(1 for h in r["history"] if h["event"] == "duplicate_merged"), "hypothesis": r["hypothesis"],
    })
incomplete = [{"claim_id": c["claim_id"], "equation": f"{c['subject'] or '?'} -> {c['object']}", "paper_id": c["paper_id"],
               "model_call": c["model_call_id"], "decision": dec_by_claim[c["claim_id"]][-1]["reason"]}
              for c in claims if dec_by_claim[c["claim_id"]][-1]["status"] == "incomplete"]

# ================================================================= 3/8. papers and identity
p2_papers = {title_key(p["title"]): p for p in jl(P2 / "papers.jsonl")}
ft_check = {}
included_rows = []
for p in included:
    text = state["paper_texts"].get(p["paper_id"], "")
    calls_p = [c for c in calls if c["paper_id"] == p["paper_id"]]
    p2 = p2_papers.get(title_key(p["title"]))
    included_rows.append({
        "paper_id": p["paper_id"], "pmid": p["pmid"], "pmcid": p["pmcid"], "title": p["title"], "year": p["year"],
        "source": p["acquisition_source"], "url": p["url"], "route": route_of(p), "inclusion_reason": p["inclusion_reasons"][0],
        "production_exclusion": next((n for n in p["notes"] if n.startswith("the production screen excluded it")), ""),
        "score": p["screening_score"], "title_other_organisms": acq._other_organisms_in_title(p["title"]),
        "observed_organisms": p["observed_organisms"], "text_chars": len(text), "text_is_abstract": text == p["abstract"],
        "full_text_matches_title": full_text_matches_title(p["title"], text),
        "pmid_in_text_head": bool(p["pmid"]) and p["pmid"] in text[:600],
        "text_head": re.sub(r"\s+", " ", text[:260]),
        "calls": [c["call_id"] for c in calls_p], "passages": sum(len(c["evidence_ids"]) for c in calls_p),
        "claims": sum(1 for c in extracted if c["paper_id"] == p["paper_id"]),
        "notes": p["notes"], "evidence_type": p["evidence_type"],
        "pilot002_id_same_title": (p2["paper_id"], p2["acquisition_source"]) if p2 else None,
    })
refused = [(p["paper_id"], n) for p in papers.values() for n in p["notes"] if "full text refused" in n]
ncbi = [p for p in papers.values() if p["acquisition_source"] == "ncbi"]
ncbi_ids = {"total": len(ncbi), "pmcid_resolved": sum(1 for p in ncbi if p["pmcid"]),
            "pmid_only": sum(1 for p in ncbi if not p["pmcid"]),
            "id_consistent": sum(1 for p in ncbi if p["paper_id"] == (f"PMC:{p['pmcid']}" if p["pmcid"] else f"PMID:{p['pmid']}")),
            "url_is_pubmed_of_own_pmid": sum(1 for p in ncbi if p["url"] == f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/"),
            "with_doi": sum(1 for p in ncbi if p["doi"])}
groups = defaultdict(list)
for p in papers.values():
    groups[title_key(p["title"])].append(p["paper_id"])
dup_titles = {k: v for k, v in groups.items() if len(v) > 1}
cross = []
for p in papers.values():
    q = p2_papers.get(title_key(p["title"]))
    if q and q["paper_id"] != p["paper_id"]:
        cross.append({"title": p["title"][:110], "pilot003": p["paper_id"], "pilot003_source": p["acquisition_source"],
                      "pilot002": q["paper_id"], "pilot002_source": q["acquisition_source"]})
same_id_both = sum(1 for p in papers.values() if (q := p2_papers.get(title_key(p["title"]))) and q["paper_id"] == p["paper_id"])
# A pilot-003 NCBI id checked against the id EuropePMC itself gave the same article in pilot-002.
p2_epmc = defaultdict(set)
for q in jl(P2 / "papers.jsonl"):
    if q["acquisition_source"] == "europepmc":
        p2_epmc[title_key(q["title"])].add(q["paper_id"])
ncbi_vs_p2_epmc = [{"title": p["title"][:100], "pilot003_ncbi_id": p["paper_id"], "pilot002_europepmc_id": sorted(p2_epmc[title_key(p["title"])]),
                    "agree": p["paper_id"] in p2_epmc[title_key(p["title"])]}
                   for p in papers.values() if p["acquisition_source"] == "ncbi" and p2_epmc.get(title_key(p["title"]))]

# ================================================================= 4. participants and fields in raw completions
raw_part = {}
raw_fields = Counter()
for r in raw_rows:
    if not r["raw_text"]:
        continue
    rows = json.loads(r["raw_text"]).get("claims", [])
    parts = [p for row in rows for p in (*row.get("substrates", []), *row.get("products", []))]
    raw_part[r["call_id"]] = {"claims": len(rows), "participants": len(parts),
                              "cited": sum(1 for p in parts if p.get("evidence_ids")),
                              "fields_filled": sorted({f for row in rows for f in ("enzyme", "gene", "species", "compartment") if str(row.get(f) or "").strip()})}
    for row in rows:
        for f in ("enzyme", "gene", "species", "compartment"):
            if str(row.get(f) or "").strip():
                own = row.get(f"{f}_evidence_ids")
                raw_fields[(f, "own list present" if own else ("own list missing" if own is None else "own list empty"))] += 1

# ================================================================= 5. batching
call_rows = []
order = defaultdict(int)
for c in calls:
    order[c["paper_id"]] += 1
    reqs = [pr for r in raw_by_call[c["call_id"]] for pr in r["provider_requests"]]
    empty_at_cap = sum(1 for pr in reqs if pr.get("finish_reason") == "length" and not pr.get("content_chars"))
    last_out = c["output_tokens"] - empty_at_cap * MAX_TOKENS  # the answering request, if the empties spent the cap
    content_est = round(c["response_chars"] / CHARS_PER_TOKEN)
    call_rows.append({
        "call_id": c["call_id"], "paper_id": c["paper_id"], "batch": order[c["paper_id"]],
        "passages": raw_by_call[c["call_id"]][0]["passages_sent"], "chars_sent": raw_by_call[c["call_id"]][0]["chars_sent"],
        "rungs": len(raw_by_call[c["call_id"]]), "provider_requests": len(reqs),
        "request_outcomes": [f"{pr.get('finish_reason')}/{pr.get('content_chars')}ch" for pr in reqs],
        "input_tokens": c["input_tokens"], "output_tokens": c["output_tokens"],
        "answering_request_output_tokens": last_out, "share_of_budget": round(last_out / MAX_TOKENS, 3),
        "response_chars": c["response_chars"], "est_content_tokens": content_est,
        "est_reasoning_tokens": last_out - content_est,
        "near_zero_reasoning": last_out - content_est < 100,
        "latency_s": round(c["latency_ms"] / 1000, 1),
        "claims": sum(1 for x in extracted if x["model_call_id"] == c["call_id"]),
        "raw_fields": raw_part.get(c["call_id"], {}),
        "provider_recorded": c["provider"] or "(empty)", "actual_model": c["actual_model"],
        "raw_text": raw_by_call[c["call_id"]][-1]["raw_text"] if c["response_chars"] <= 700 else "",
        "status": c["status"],
    })
calls_per_paper = Counter(c["paper_id"] for c in calls)

# ================================================================= 6. search
q_rows = []
cum = 0
for i, q in enumerate(queries, 1):
    new = [p for p in papers.values() if p["discovered_via_query_id"] == q["query_id"] and p["discovered_via_frontier_id"] == q["frontier_item_id"]]
    cum += len(new)
    q_rows.append({"n": i, "query_id": q["query_id"], "target": q["source_terms"], "attempt": q["attempt"],
                   "window": min(budgets["max_results_per_query"], budgets["max_papers_discovered"] - (cum - len(new))),
                   "result_count": q["result_count"], "new_discovered": len(new),
                   "new_by_source": dict(Counter(p["acquisition_source"] for p in new)),
                   "new_included": q["new_paper_ids"], "cumulative": cum, "query_text": q["query_text"],
                   "executed_at": q["executed_at"], "cache_hit": q["cache_hit"]})
assert cum == len(papers)

# ================================================================= 7. stop and frontier replay
r1_in = [FrontierItem.from_dict(d) for d in round1["input_frontier"]]
delta = GraphDelta.from_dict(state["deltas"][0])
g = ResearchGraph.from_dict(state["graph"])
store = ClaimStore.from_dict(state["store"])
cfg = lambda cap: SimpleNamespace(topic=config["topic"], target_species=config["target_species"],  # noqa: E731
                                  budgets=SimpleNamespace(max_frontier_size=cap))
uncapped = fm.generate(graph=g, store=store, delta=delta, config=cfg(10 ** 6), round_index=1, previous=r1_in, exhausted_ids=set())
capped = fm.merge_frontier([i for i in r1_in if not i.exhausted], fm.generate(graph=g, store=store, delta=delta, config=cfg(60), round_index=1,
                                                                            previous=r1_in, exhausted_ids=set()), cap=60)
rec = {(i["kind"], i["target"]) for i in frontier_final}
rep = {(i.kind, i.target) for i in capped}
recorded_kinds = Counter(i["kind"] for i in frontier_final)
# Bridge targets depend on in-memory component order (pilot-001 A10), so only non-bridge items are compared.
nb = lambda s: {x for x in s if x[0] not in ("bridge_search", "seed_expansion")}  # noqa: E731
# The replay's bridge count can vary between processes (anchor collisions follow set order), so the check
# is on non-bridge items: the recorded ones must be exactly the top-ranked non-bridge items that fit in the
# slots the live run left after its 3 seeds and its recorded bridges.
uncapped_nb = [i for i in uncapped if i.kind != "bridge_search"]
slots = budgets["max_frontier_size"] - recorded_kinds["seed_expansion"] - recorded_kinds["bridge_search"]
assert {(i.kind, i.target) for i in uncapped_nb[:slots]} == nb(rec), ({(i.kind, i.target) for i in uncapped_nb[:slots]} ^ nb(rec))
cut = uncapped_nb[slots:]
replay_bridges = sum(1 for i in uncapped if i.kind == "bridge_search")
replay = {"uncapped_generated": len(uncapped_nb) + recorded_kinds["bridge_search"],
          "uncapped_nonbridge": len(uncapped_nb), "replay_bridges_this_process": replay_bridges,
          "uncapped_by_kind": dict(Counter(i.kind for i in uncapped_nb)) | {"bridge_search (recorded)": recorded_kinds["bridge_search"]},
          "recorded_frontier": len(frontier_final), "recorded_by_kind": dict(recorded_kinds),
          "replay_nonbridge_equals_recorded": {(i.kind, i.target) for i in uncapped_nb[:slots]} == nb(rec),
          "bridge_anchor_differences": {"recorded_only": sorted(t for k, t in rec - rep), "replay_only": sorted(t for k, t in rep - rec)},
          "cut_by_cap": len(cut), "cut_by_kind": dict(Counter(i.kind for i in cut)),
          "cut_items": [{"kind": i.kind, "target": i.target, "priority": i.priority} for i in cut],
          "lowest_kept_priority": min(i["priority"] for i in frontier_final)}
stop = {"stop_reason": manifest["stop_reason"], "stop_detail": manifest["stop_detail"],
        "added_reaction_ids": len(delta.added_reaction_ids), "added_candidate_reaction_ids": len(delta.added_candidate_reaction_ids),
        "updated_reaction_ids": len(delta.updated_reaction_ids), "last_round_new_reactions": summary["usage"]["last_round_new_reactions"],
        "limit": budgets["max_new_reactions_per_round"], "distinct_conversions_by_reading": distinct_conversions,
        "by_paper": dict(Counter(reactions[r]["paper_ids"][0] for r in delta.added_candidate_reaction_ids)),
        "by_species_state": dict(Counter(reactions[r]["species_state"] for r in delta.added_candidate_reaction_ids)),
        "round1_elapsed_s": summary["usage"]["elapsed_seconds"]}

# ================================================================= 9. three-way side by side
def basic(run):
    s = js(run / "summary.json")
    m = js(run / "run_manifest.json")
    ps = jl(run / "papers.jsonl")
    cs = jl(run / "model_calls.jsonl")
    qs = jl(run / "queries.jsonl")
    errs = js(run / "state.json")["errors"]
    inc = [p for p in ps if not p["exclusion_reasons"]]
    ex = [c for d in sorted((run / "rounds").iterdir()) if (d / "claims_extracted.jsonl").exists() for c in jl(d / "claims_extracted.jsonl")]
    raw = [r for d in sorted((run / "rounds").iterdir()) if (d / "model_raw.jsonl").exists() for r in jl(d / "model_raw.jsonl")]
    attr = Counter(h for c in ex for h in (c.get("field_attribution") or {}).values())
    reqs = sum(len(r["provider_requests"]) if isinstance(r.get("provider_requests"), list) else 0 for r in raw)
    return OrderedDict([
        ("code", m["git"]["commit"][:8]), ("stop_reason", m["stop_reason"]), ("rounds", s["rounds_completed"]),
        ("elapsed s", s["costs"]["elapsed_seconds"]), ("queries (rows)", len(qs)),
        ("papers discovered (distinct titles)", "%d (%d)" % (len(ps), len({title_key(p["title"]) for p in ps}))),
        ("NCBI papers discovered", sum(1 for p in ps if p["acquisition_source"] == "ncbi")),
        ("included: production / route 1 / route 2", "%d: %d / %d / %d" % (len(inc), sum(route_of(p) == "production" for p in inc),
                                                                         sum(route_of(p) == "mining_rescue route 1" for p in inc),
                                                                         sum(route_of(p) == "mining_rescue route 2" for p in inc))),
        ("processed", s["papers"]["processed"]), ("passages", s["passages"]["extracted"]),
        ("logical model calls (by status)", "%d %s" % (len(cs), json.dumps(s["model_calls"]["by_status"]))),
        ("ladder rungs (usage.llm_calls)", s["usage"]["llm_calls"]),
        ("max passages per call / max_tokens", "%s / %s" % (max((len(c["evidence_ids"]) for c in cs), default=0),
                                                            m["config"]["model"]["max_tokens"])),
        ("claims extracted -> identities", "%d -> %d" % (s["usage"]["claims_extracted"], s["claims"]["distinct_identities"])),
        ("primary / candidate reactions / components", "%d / %d / %d" % (s["graph"]["primary_reactions"], s["graph"]["candidate_reactions"],
                                                                       s["graph"]["candidate_components"])),
        ("species states (nodes)", json.dumps(dict(sorted(s["graph"]["species_states"].items())))),
        ("uncited field drops", len(errs)),
        ("fields: model_citation / verbatim_cited_span", "%d / %d" % (attr.get("model_citation", 0), attr.get("verbatim_cited_span", 0))),
        ("input / output tokens", "%d / %d" % (s["costs"]["input_tokens"], s["costs"]["output_tokens"])),
        ("est. cost USD", s["costs"]["estimated_cost_usd"]),
        ("evidence type of included", json.dumps(s["papers"]["by_evidence_type"])),
    ])


side = OrderedDict([("pilot-001", basic(P1)), ("pilot-002", basic(P2)), ("pilot-003", basic(RUN))])
p2a = js(P2_AUDIT)
side["pilot-001"]["provider requests"] = "1"
side["pilot-002"]["provider requests"] = str(p2a["empty_completion_summary"]["provider_requests"])
side["pilot-003"]["provider requests"] = str(provider_requests)
side["pilot-001"]["empty requests (finish_reason=length)"] = "0"
side["pilot-002"]["empty requests (finish_reason=length)"] = "21"
side["pilot-003"]["empty requests (finish_reason=length)"] = str(sum(1 for r in raw_rows for pr in r["provider_requests"] if pr.get("finish_reason") == "length"))

SRC = Path("C:/t/mining-pilot3/src/t2pw/mining")


def quote(fname, a, b):
    lines = (SRC / fname).read_text(encoding="utf-8").splitlines()
    return {"file": f"src/t2pw/mining/{fname}", "lines": f"{a}-{b}", "text": "\n".join(lines[a - 1:b])}


code_quotes = OrderedDict([
    ("record_round", quote("budget.py", 155, 159)),
    ("limit", quote("budget.py", 187, 191)),
    ("check_before_round", quote("budget.py", 193, 225)),
    ("round_loop", quote("controller.py", 319, 333)),
    ("admit_delta", quote("controller.py", 691, 697)),
    ("close_round", quote("controller.py", 552, 572)),
])
assert "last_round_new_reactions = max(0, int(new_reactions))" in code_quotes["record_round"]["text"]
assert "added_candidate_reaction_ids" in code_quotes["close_round"]["text"]
assert "STOP_MAX_NEW_REACTIONS_PER_ROUND" in code_quotes["check_before_round"]["text"]

result = OrderedDict(
    audit="ORCH-743 pilot-003 artifact audit (read-only)", run_dir=str(RUN), code_quotes=code_quotes,
    code="C:/t/mining-pilot3 @ %s" % manifest["git"]["commit"],
    headline={k: manifest[k] for k in ("stop_reason", "stop_detail", "completion_state", "rounds_completed")} | {
        "elapsed_seconds": costs["elapsed_seconds"], "estimated_cost_usd": costs["estimated_cost_usd"]},
    counts=counts, reactions=reaction_rows, incomplete_claims=incomplete, distinct_conversions=distinct_conversions, same_conversion_groups=SAME,
    included_papers=included_rows, full_text_refused=refused, ncbi_identity=ncbi_ids, duplicate_titles=dup_titles,
    cross_run_id_changes=cross, same_id_in_both_runs=same_id_both, ncbi_vs_pilot002_europepmc=ncbi_vs_p2_epmc,
    raw_participants_by_call=raw_part, raw_field_citation_lists={f"{a}:{b}": n for (a, b), n in sorted(raw_fields.items())},
    batching={"calls": call_rows, "calls_per_paper": dict(calls_per_paper), "max_tokens": MAX_TOKENS,
              "provider_requests": provider_requests, "log_retry_lines": retry_lines, "log_giving_up": giving_up},
    search={"queries": q_rows, "cache_hit_true": sum(1 for q in queries if q["cache_hit"]),
            "round_metrics_papers_discovered": round1["metrics"]["papers_discovered"]},
    stop=stop, frontier_replay=replay,
    frontier_final=[{"kind": i["kind"], "target": i["target"], "priority": i["priority"], "attempts": i["attempts"]} for i in frontier_final],
    side_by_side=side,
    report_md_lines=[ln.strip() for ln in (RUN / "report.md").read_text(encoding="utf-8").splitlines()
                     if any(k in ln for k in ("Claims extracted", "Primary graph", "Model calls", "never asked"))],
    duplicate_span_texts=sum(n - 1 for n in Counter(e["text"] for e in evidence.values()).values() if n > 1),
    component_relevance=dict(Counter(c["relevance_score"] for c in cands["components"])),
    input_frontier_attempts_as_written=[(i["target"], i["attempts"]) for i in round1["input_frontier"]],
    span_texts={e: evidence[e]["text"] for e in sorted({x["excerpt_span"] for x in reaction_rows})},
)
(OUT / "pilot_003_audit.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

from md3 import render  # noqa: E402

md = []
render(md.append, result)
(OUT / "PILOT-003-AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("wrote", OUT / "PILOT-003-AUDIT.md")
print(json.dumps({k: v["value"] for k, v in counts.items()}, ensure_ascii=False, default=str))
print(json.dumps(replay | {"cut_items": len(replay["cut_items"])}, ensure_ascii=False))
print(json.dumps(stop, ensure_ascii=False))
for c in call_rows:
    print(c["call_id"], c["paper_id"], c["batch"], c["passages"], c["output_tokens"], c["answering_request_output_tokens"],
          c["share_of_budget"], c["response_chars"], c["est_reasoning_tokens"], c["near_zero_reasoning"], c["latency_s"], c["claims"], c["raw_fields"])
for x in reaction_rows:
    print(x["reaction_id"], x["species_state"], x["species_marker"], x["participants_cited"], "/", x["participants_total"], x["excerpt_names_matched"])
print(json.dumps(included_rows, ensure_ascii=False, default=str)[:3000])
print(json.dumps(ncbi_ids), dup_titles, len(cross), same_id_both)
print(json.dumps(cross, ensure_ascii=False)[:3000])

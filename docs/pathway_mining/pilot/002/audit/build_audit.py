"""PILOT-002 audit builder (ORCH-743). Read-only over the run directory and the frozen tree.

Run through bounded_run.py with PYTHONPATH=C:/t/mining-pilot2/src and PYTHONDONTWRITEBYTECODE=1.
Writes only PILOT-002-AUDIT.md and pilot_002_audit.json next to this file.
No model call, no network: frozen modules are imported only for pure functions
(render_passages / extract_claims with a capturing stub client, the title-organism
detector, the PubMed XML parser on a synthetic record).
"""
from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

RUN = Path("C:/t/mining-pilot-runs2/pilot-002-jasmonate-ath")
LOG = Path("C:/t/mining-pilot-runs2/pilot-002.log")
P1 = Path("C:/t/mining-pilot/docs/pathway_mining/pilot/001/run")
FT_CACHE = Path("C:/t/mining-pilot2/data/rag_index/acquire_cache/fulltext")
OUT = Path(__file__).resolve().parent


def jl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell(value) -> str:
    return " ".join(str(value if value is not None else "").split()).replace("|", "\\|")


# ---------------------------------------------------------------- load pilot-002
summary = js(RUN / "summary.json")
manifest = js(RUN / "run_manifest.json")
config = js(RUN / "config.json")
costs = js(RUN / "costs.json")
state = js(RUN / "state.json")
papers = OrderedDict((r["paper_id"], r) for r in jl(RUN / "papers.jsonl"))
claims = jl(RUN / "claims.jsonl")
decisions = jl(RUN / "decisions.jsonl")
calls = jl(RUN / "model_calls.jsonl")
queries = jl(RUN / "queries.jsonl")
evidence = {r["evidence_id"]: r for r in jl(RUN / "evidence_spans.jsonl")}
primary = js(RUN / "primary_graph.json")
cands = js(RUN / "candidate_subgraphs.json")
rounds = {n: js(RUN / f"rounds/round_{n:03d}/round.json") for n in (1, 2, 3)}
raw_rows = [r for n in (1, 2, 3) for r in jl(RUN / f"rounds/round_{n:03d}/model_raw.jsonl")]
extracted = [c for n in (1, 2, 3) for c in jl(RUN / f"rounds/round_{n:03d}/claims_extracted.jsonl")]
log_text = LOG.read_text(encoding="utf-8", errors="replace")
graph = state["graph"]
budgets = config["budgets"]
MAX_TOKENS = config["model"]["max_tokens"]

reactions = {r["reaction_id"]: r for r in primary["reactions"]}
for comp in cands["components"]:
    for r in comp["reactions"]:
        reactions[r["reaction_id"]] = dict(r, _component=comp)
for r in primary["reactions"]:
    reactions[r["reaction_id"]]["_component"] = {k: v for k, v in primary.items() if k != "reactions"}

# ---------------------------------------------------------------- frozen code (pure functions only)
from t2pw.mining import extraction as fx  # noqa: E402
from t2pw.mining.acquisition import Acquirer, _FOREIGN_HOST, _SENTENCE  # noqa: E402
from t2pw.mining.models import EvidenceSpan  # noqa: E402
from t2pw.rag.acquire import _parse_pubmed_article  # noqa: E402

acq = object.__new__(Acquirer)
acq.config = SimpleNamespace(topic=config["topic"], target_species=config["target_species"],
                             target_species_normalized=config["target_species_normalized"])


def rebuild_user(call):
    """The exact user message extract_claims sent on rung 1, via the frozen code and a capturing stub."""
    captured = {}

    class Stub:
        def call_json(self, **kw):
            captured.update(kw)
            return SimpleNamespace(ok=False, status="captured", record=None)

    spans = [EvidenceSpan.from_dict(evidence[e]) for e in call["evidence_ids"]]
    p = papers[call["paper_id"]]
    fx.extract_claims(client=Stub(), spans=spans, paper_id=p["paper_id"], paper_title=p["title"],
                      target_species=config["target_species"],
                      target_species_normalized=config["target_species_normalized"],
                      round_index=call["round_index"], topic=config["topic"],
                      paper_observed_organisms=p["observed_organisms"])
    return captured["system"], captured["user"]


def sentences_with(text, needles):
    parts = _SENTENCE.split(text)
    out = []
    for needle in needles:
        hit = [s for s in parts if needle in s]
        assert hit, f"needle not found: {needle!r}"
        if hit[0] not in out:
            out.append(hit[0])
    for s in out:
        assert s in text
    return out


# ================================================================= 1. counts
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


routes = Counter(route_of(p) for p in included)
processed = [p for p in included if p["processed_round"] >= 0]
dup_events = [(rid, h) for rid, r in reactions.items() for h in r["history"] if h["event"] == "duplicate_merged"]
species_nodes = Counter(r["species_state"] for r in reactions.values())
species_claims = Counter(c["species_state"] for c in extracted)
rejections = [x for n in (1, 2, 3) for x in rounds[n]["rejections"]]
all_errors = state["errors"]
drops = Counter(m.group(1) for e in all_errors for m in [re.match(r"claim\[\d+\]\.(\w+)=", e)] if m)
attr_stored = Counter((f, how) for c in claims for f, how in c["field_attribution"].items())
attr_extracted = Counter((f, how) for c in extracted for f, how in c["field_attribution"].items())
status_calls = Counter(c["status"] for c in calls)
rung_outcomes = Counter(r["parse_outcome"] for r in raw_rows)
giving_up = log_text.count("giving up and returning the empty string")
retrying = re.findall(r"finish_reason=(\w+)\) on attempt (\d)/(\d); retrying", log_text)
inner_max = int(re.search(r"on (\d+) of (\d+) attempts; giving up", log_text).group(2))

# provider requests per rung: an empty rung spent every inner attempt; the last rung's inner attempts are in attempt_log
per_call = OrderedDict()
for c in calls:
    rows = [r for r in raw_rows if r["call_id"] == c["call_id"]]
    rung_requests = []
    for i, r in enumerate(rows):
        if r["parse_outcome"] == "empty_completion":
            rung_requests.append(inner_max)
        elif i == len(rows) - 1:
            rung_requests.append(len(c["attempt_log"]))
        else:
            rung_requests.append(1)
    per_call[c["call_id"]] = {"rows": rows, "rung_requests": rung_requests}
provider_requests = sum(sum(v["rung_requests"]) for v in per_call.values())
ladder_rungs = len(raw_rows)
assert ladder_rungs == summary["usage"]["llm_calls"], (ladder_rungs, summary["usage"]["llm_calls"])
assert sum(c["input_tokens"] for c in calls) == costs["input_tokens"]
assert sum(c["output_tokens"] for c in calls) == costs["output_tokens"]

counts = OrderedDict()


def put(key, value, source, note=""):
    counts[key] = {"value": value, "source": source, "note": note}


put("papers_discovered", len(papers), "summary.json papers.discovered=%d; usage.papers_discovered=%d; papers.jsonl rows=%d"
    % (summary["papers"]["discovered"], summary["usage"]["papers_discovered"], len(papers)),
    "by source: %s" % dict(Counter(p["acquisition_source"] for p in papers.values())))
put("papers_screened", sum(1 for p in papers.values() if p["screening_outcome"]),
    "summary.json papers.screened=%d; papers.jsonl rows with screening_outcome set" % summary["papers"]["screened"])
put("papers_included", len(included), "summary.json papers.included=%d; papers.jsonl rows with empty exclusion_reasons"
    % summary["papers"]["included"],
    "by route (papers.jsonl screening_outcome + inclusion_reasons): %s; by source: %s"
    % (dict(routes), dict(Counter(p["acquisition_source"] for p in included))))
put("papers_excluded", summary["papers"]["excluded"], "summary.json papers.excluded")
put("papers_processed", len(processed), "papers.jsonl processed_round>=0; usage.papers_processed=%d; summary papers.processed=%d"
    % (summary["usage"]["papers_processed"], summary["papers"]["processed"]),
    "the 11th included paper, PMC:PMC5104190, was never processed: its notes say 'round 2: no passages retrieved for this query'")
put("claims_extracted", len(extracted), "usage.claims_extracted=%d; rounds/*/claims_extracted.jsonl rows=%d"
    % (summary["usage"]["claims_extracted"], len(extracted)),
    "claims.jsonl holds %d stored claims (= summary.json claims.extracted=%d, which is the STORED count, not the extracted one); distinct identities=%d"
    % (len(claims), summary["claims"]["extracted"], summary["claims"]["distinct_identities"]))
put("primary_reactions", primary["reaction_count"], "primary_graph.json reaction_count; summary graph.primary_reactions=%d"
    % summary["graph"]["primary_reactions"], "component %s" % primary["component_id"])
put("candidate_reactions", sum(len(c["reactions"]) for c in cands["components"]),
    "candidate_subgraphs.json sum(len(components[].reactions)) over count=%d components; summary graph.candidate_reactions=%d"
    % (cands["count"], summary["graph"]["candidate_reactions"]),
    "decision status of the 10: %s" % dict(Counter(r["decision_status"] for r in reactions.values() if not r.get("admitted"))))
put("rejected_by_reason", summary["claims"]["rejected_by_reason"], "summary.json claims.rejected_by_reason ({} = no claim has status 'rejected')",
    "rounds/*/rejections.jsonl holds %d rows (every NON-ADMITTED claim, controller.py _admit): statuses %s"
    % (len(rejections), dict(Counter(x["status"] for x in rejections))))
put("duplicates_merged", len(dup_events), "reaction history events 'duplicate_merged' in primary_graph.json + candidate_subgraphs.json; usage.claims_extracted - distinct identities = %d"
    % (summary["usage"]["claims_extracted"] - summary["claims"]["distinct_identities"]),
    "all 5 are the model re-emitting a claim inside call mc_001_extract_claims_0004 (same paper, same span ev_f7c56801adf6792d): %s"
    % dict(Counter(reactions[rid]["label"] for rid, _ in dup_events)))
put("species_states_reaction_nodes", dict(species_nodes), "candidate_subgraphs.json + primary_graph.json reactions[].species_state (== summary.json graph.species_states)",
    "per extracted claim (16): %s" % dict(species_claims))
put("budget_stops", manifest["stop_reason"], "run_manifest.json stop_reason / stop_detail='%s' / completion_state=%s"
    % (manifest["stop_detail"], manifest["completion_state"]),
    "the only limit reached. It was recorded during round 2; round 3 then started and ran 0 queries (see anomaly A5)")
put("model_calls_by_status", dict(status_calls), "model_calls.jsonl status (%d logical calls); summary.json model_calls.by_status" % len(calls),
    "ladder rungs (model_raw.jsonl rows, = usage.llm_calls): %d, by parse_outcome %s; provider HTTP requests: %d (+1 preflight probe)"
    % (ladder_rungs, dict(rung_outcomes), provider_requests))
put("empty_completion_log_lines", giving_up, "pilot-002.log lines 'giving up and returning the empty string'",
    "one line per ladder rung whose %d provider requests all came back empty: 3 rungs of mc_001_..._0002 + 3 of _0003 + 1 of mc_002_..._0006 = 7. model_calls.jsonl shows only 2 failed calls because _0006 recovered on rung 2 (status ok). Other log retries: %s"
    % (inner_max, [f"finish_reason={a} attempt {b}/{c}" for a, b, c in retrying if a != "length"]))
put("uncited_field_drops", {"total": len(all_errors), **dict(drops)}, "state.json errors (%d rows); rounds/*/errors.jsonl rows=%d"
    % (len(all_errors), sum(len(rounds[n]["errors"]) for n in (1, 2, 3))), "pilot-001 dropped 59 of 59")
put("fields_kept_by_attribution", {f"{f}:{h}": n for (f, h), n in sorted(attr_stored.items())},
    "claims.jsonl field_attribution (11 stored claims)",
    "over the 16 extracted claims: %s; verbatim_cited_span = %d"
    % ({f"{f}:{h}": n for (f, h), n in sorted(attr_extracted.items())},
       sum(n for (f, h), n in attr_extracted.items() if h == "verbatim_cited_span")))
put("passages", summary["passages"], "summary.json passages; evidence_spans.jsonl rows=%d" % len(evidence))

budget_rows = [
    ("rounds", summary["usage"]["rounds"], budgets["max_rounds"]),
    ("papers discovered", summary["usage"]["papers_discovered"], budgets["max_papers_discovered"]),
    ("papers processed", summary["usage"]["papers_processed"], budgets["max_papers_processed"]),
    ("LLM calls (ladder rungs)", summary["usage"]["llm_calls"], budgets["max_llm_calls"]),
    ("input tokens", costs["input_tokens"], budgets["max_input_tokens"]),
    ("output tokens", costs["output_tokens"], budgets["max_output_tokens"]),
    ("est. cost USD", costs["estimated_cost_usd"], budgets["max_cost_usd"]),
    ("wall clock s", costs["elapsed_seconds"], budgets["max_wall_clock_seconds"]),
    ("frontier size (start of round 2 / 3)", "%d / %d" % (len(rounds[2]["input_frontier"]), len(rounds[3]["input_frontier"])), budgets["max_frontier_size"]),
    ("unproductive streak", summary["usage"]["unproductive_streak"], budgets["stop_after_unproductive_rounds"]),
]

# ================================================================= 2. reactions
SEED = {  # auditor's seed-connection reading, yardstick: 13-LOX -> AOS -> AOC -> OPR3 -> beta-ox -> JA -> JAR1
    "rx_98656adf98c1fca1946a": ("upstream, 2 steps", "13-LOX step: makes 13-HPOT, the AOS substrate; 12-OPDA is AOS then AOC away"),
    "rx_b30980df6dd1f29b4504": ("upstream, 2 steps", "same 13-LOX step as the primary reaction (pepper introduction, species left empty)"),
    "rx_c1d050d4171b110ad2d4": ("upstream, 2 steps", "same 13-LOX step again, same span, species Capsicum annuum: a second node for the same sentence"),
    "rx_86b33c076d39e423f2fa": ("DIRECT (seed enzyme AOS)", "the AOS step 13-HPOT -> 12,13-EOT, catalysed by the seed enzyme"),
    "rx_afb91c7037581c9fadc2": ("DIRECT (makes seed 12-OPDA)", "the AOC step; its product (9S,13S)-OPDA is the seed 12-OPDA"),
    "rx_f8664da4761606d51db8": ("DIRECT (consumes seed 12-OPDA)", "OPR3 reduces the seed 12-OPDA to OPC-8 (downstream)"),
    "rx_312345b466fea5ca097c": ("downstream, 2 steps", "OPC-8 -> JA-CoA; the span collapses CoA activation and three beta-oxidation cycles into one step"),
    "rx_da81fbd8f5b34dc5ec32": ("downstream, 3 steps", "JA-CoA -> JA (thioesterase step; no enzyme stated)"),
    "rx_355aa30ff85f270c9a58": ("downstream, 4 steps", "JAR1 conjugates JA to JA-Ile"),
    "rx_85236be236d5f1e8a0f1": ("not a metabolic step", "JA-Ile binding the SCF-COI1 receptor complex is perception, not a conversion; held as hypothesis_only"),
    "rx_febac937577910d44f45": ("upstream, 2 steps", "the same 13-LOX step (LOX2) from the NAD+ paper's abstract"),
}
NEEDLES = {
    "rx_98656adf98c1fca1946a": ["is converted to 13-S-hydroperoxyoctadeca", "This study aimed for characterizing the four annotated A. thaliana", "in cascade with 12-OPDA pathway enzymes"],
    "rx_b30980df6dd1f29b4504": ["jasmonate biosynthesis starts in the chloroplast", "The fatty acid undergoes oxidation"],
    "rx_c1d050d4171b110ad2d4": ["jasmonate biosynthesis starts in the chloroplast", "The fatty acid undergoes oxidation"],
    "rx_86b33c076d39e423f2fa": ["is catalyzed by allene oxide synthase (AOS)"],
    "rx_afb91c7037581c9fadc2": ["is reconfigured by allene oxide cyclase (AOC)"],
    "rx_f8664da4761606d51db8": ["with the reduction of OPDA by OPDA reductase 3 (OPR3)"],
    "rx_312345b466fea5ca097c": ["OPC-8 undergoes"],
    "rx_da81fbd8f5b34dc5ec32": ["is transformed into jasmonic acid"],
    "rx_355aa30ff85f270c9a58": ["by jasmonate resistant 1 (JAR1)"],
    "rx_85236be236d5f1e8a0f1": ["JA-Ile binds to the Skp1/Cullin1/F-box protein"],
    "rx_febac937577910d44f45": ["hydroperoxidation of"],
}
PEPPER_SPECIES = ["We chose pepper ( Capsicum annuum ) as our study organism"]
PEPPER_INVITRO = ["in vitro studies confirmed"]

claims_by_id = {c["claim_id"]: c for c in claims}
dec_by_claim = defaultdict(list)
for d in decisions:
    dec_by_claim[d["claim_id"]].append(d)
conn_by_rx = defaultdict(list)
for c in graph["connections"]:
    conn_by_rx[c["left_reaction_id"]].append(c)
    conn_by_rx[c["right_reaction_id"]].append(c)
ent = graph["entities"]
seed_named = {eid: e.get("name") or e.get("canonical_name") for eid, e in ent.items()
              if re.search(r"OPDA|oxo-phytodienoic|\bAOS\b|allene oxide synthase", json.dumps(e, ensure_ascii=False))}

reaction_rows = []
for rid in [primary["reactions"][0]["reaction_id"]] + [r["reaction_id"] for comp in cands["components"] for r in comp["reactions"]]:
    r = reactions[rid]
    cl = [claims_by_id[c] for c in r["claim_ids"]]
    c0 = cl[0]
    spans = []
    for eid in r["evidence_ids"]:
        text = evidence[eid]["text"]
        spans.append({"evidence_id": eid, "section": evidence[eid]["section"], "locator": evidence[eid]["locator"],
                      "char_start": evidence[eid]["char_start"], "char_end": evidence[eid]["char_end"],
                      "excerpt": sentences_with(text, NEEDLES[rid]) if eid == r["evidence_ids"][0] else [],
                      "same_text_as": [e for e in r["evidence_ids"] if e != eid and evidence[e]["text"] == text]})
    neighbours = sorted({(c["kind"], c["via_entity_name"], c["left_reaction_id"] if c["right_reaction_id"] == rid else c["right_reaction_id"])
                         for c in conn_by_rx[rid]})
    d = dec_by_claim[c0["claim_id"]][-1]
    reaction_rows.append({
        "reaction_id": rid, "graph_status": r["graph_status"], "component_id": r["component_id"],
        "quarantine_reason": r.get("quarantine_reason", ""), "equation": r["label"],
        "enzymes": [{"name": n, "attribution": c0["field_attribution"].get("enzyme", "")} for n in r["enzyme_names"]],
        "gene": {"value": r["gene"], "attribution": c0["field_attribution"].get("gene", "")},
        "species_state": r["species_state"], "species_value": r["species_text"],
        "species_attribution": c0["field_attribution"].get("species", ""),
        "species_evidence_ids": c0["field_evidence"].get("species", []),
        "species_reason": c0["species_reason"],
        "paper_id": r["paper_ids"][0], "paper_title": papers[r["paper_ids"][0]]["title"],
        "claim_ids": r["claim_ids"], "model_call_id": c0["model_call_id"],
        "spans": spans, "seed_connection": SEED[rid][0], "seed_how": SEED[rid][1],
        "graph_neighbours": [{"kind": k, "via": v, "reaction_id": o} for k, v, o in neighbours],
        "decision_status": r["decision_status"], "decision_reason": r["decision_reason"],
        "decision_id": d["decision_id"], "evidence_level": d["evidence_level"],
        "duplicate_merges": sum(1 for h in r["history"] if h["event"] == "duplicate_merged"),
        "hypothesis": r["hypothesis"],
    })

prim = reaction_rows[0]
prim_raw = next(r for r in raw_rows if r["call_id"] == prim["model_call_id"])
prim_payload = json.loads(prim_raw["raw_text"])["claims"][0]
assert prim["species_attribution"] == "model_citation"
assert prim_payload.get("species_evidence_ids") == prim["species_evidence_ids"]
prim_same_text = evidence[prim["species_evidence_ids"][0]]["text"] == evidence[prim["species_evidence_ids"][1]]["text"]
pepper_text = evidence["ev_f7c56801adf6792d"]["text"]
pepper_species_sentence = sentences_with(pepper_text, PEPPER_SPECIES)
pepper_invitro_sentence = sentences_with(pepper_text, PEPPER_INVITRO)
prim_text = evidence[prim["species_evidence_ids"][0]]["text"]

# ================================================================= 3. included papers
ncbi_proof = {}
xml = ("<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>111</PMID><Article><ArticleTitle>Own title"
       "</ArticleTitle><Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>"
       "<Abstract><AbstractText>own abstract</AbstractText></Abstract></Article></MedlineCitation><PubmedData>"
       "<ArticleIdList><ArticleId IdType=\"pubmed\">111</ArticleId><ArticleId IdType=\"pmc\">PMC999001</ArticleId>"
       "<ArticleId IdType=\"doi\">10.9999/own</ArticleId></ArticleIdList><ReferenceList><Reference><Citation>cited"
       "</Citation><ArticleIdList><ArticleId IdType=\"doi\">10.1073/pnas.76.9.4350</ArticleId><ArticleId "
       "IdType=\"pmc\">PMC411572</ArticleId></ArticleIdList></Reference></ReferenceList></PubmedData></PubmedArticle>"
       "</PubmedArticleSet>")
parsed = _parse_pubmed_article(ElementTree.fromstring(xml).find(".//PubmedArticle"))
ncbi_proof = {"synthetic_record_own_ids": {"pmid": "111", "pmc": "PMC999001", "doi": "10.9999/own"},
              "cited_reference_ids": {"pmc": "PMC411572", "doi": "10.1073/pnas.76.9.4350"},
              "parser_returned": {"id": parsed.id, "source_uri": parsed.source_uri, "title": parsed.title}}
assert parsed.id == "PMC411572"

ft_cache = {}
for f in FT_CACHE.glob("*.json"):
    d = js(f)
    ft_cache[d["id"]] = d.get("full_text", "")


def fetched_header(text):
    m = re.search(r"10\.\d{4,9}/\S+\s+(?:\d+\s+)?(?:Research (?:Paper|Article)\s+)?(.{0,140})", text[:1500])
    return m.group(1) if m else text[:140]


included_rows = []
for p in included:
    ident = p["pmcid"] or p["pmid"]
    text = state["paper_texts"].get(p["paper_id"], "")
    cached = ft_cache.get(ident)
    folded_title = " ".join(re.sub(r"<[^>]+>|&lt;.*?&gt;", "", p["title"]).lower().split()).rstrip(".")
    text_kind = "abstract only" if text == p["abstract"] else "full text %d chars" % len(text)
    title_in_text = folded_title[:60] in " ".join(text[:3000].lower().split())
    reason = p["inclusion_reasons"][0] if p["inclusion_reasons"] else ""
    m = re.search(r"names '([^']+)' and '([^']+)' in one sentence", reason)
    route2_sentence = ""
    if m:
        for s in _SENTENCE.split(f"{p['title']}. {p['abstract']}"):
            if m.group(2) in s and m.group(1) in " ".join(s.lower().split()):
                route2_sentence = s
                break
    n_claims = sum(1 for c in extracted if c["paper_id"] == p["paper_id"])
    call = next((c for c in calls if c["paper_id"] == p["paper_id"]), None)
    included_rows.append({
        "paper_id": p["paper_id"], "title": p["title"], "year": p["year"], "source": p["acquisition_source"],
        "doi_recorded": p["doi"], "url_recorded": p["url"], "route": route_of(p), "inclusion_reason": reason,
        "matched_term": m.group(2) if m else "", "matched_species_form": m.group(1) if m else "",
        "route2_sentence": route2_sentence,
        "production_exclusion": next((n for n in p["notes"] if n.startswith("the production screen excluded it")), ""),
        "screening_score": p["screening_score"], "title_other_organisms": acq._other_organisms_in_title(p["title"]),
        "foreign_host_in_title": bool(_FOREIGN_HOST.search(p["title"])),
        "foreign_host_in_abstract": (_FOREIGN_HOST.search(p["abstract"]).group(0) if _FOREIGN_HOST.search(p["abstract"]) else ""),
        "observed_organisms": p["observed_organisms"], "processed_round": p["processed_round"],
        "text_used": text_kind, "fulltext_cache_chars": len(cached) if cached is not None else None,
        "fetched_text_names_this_title": title_in_text if text_kind != "abstract only" else None,
        "fetched_text_header": fetched_header(text) if text_kind != "abstract only" else "",
        "passages": len(call["evidence_ids"]) if call else 0, "model_call": call["call_id"] if call else "",
        "claims_extracted": n_claims, "evidence_type": p["evidence_type"],
        "notes": p["notes"],
    })

# excluded titles that name the pathway: recall check (pilot-001 F5)
PATH_TERMS = re.compile(r"lipoxygenase|\bLOX\d*\b|allene oxide|\bAOS\b|\bAOC\d*\b|OPDA|oxophytodienoic|OPR3|JAR1|"
                        r"jasmon\w* biosynth|oxylipin", re.IGNORECASE)
excluded_pathway_titles = [
    {"paper_id": p["paper_id"], "title": p["title"], "source": p["acquisition_source"], "score": p["screening_score"],
     "reason": p["exclusion_reasons"][0], "names_arabidopsis": "arabidopsis" in p["title"].lower(),
     "title_other_organisms": acq._other_organisms_in_title(p["title"])}
    for p in papers.values() if p["exclusion_reasons"] and PATH_TERMS.search(p["title"])]

# the same article under two paper ids (normalized title), and what that proves about NCBI ids
import html  # noqa: E402


def title_key(t):
    t = html.unescape(html.unescape(t or ""))
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"[^a-z0-9]+", "", t.lower())


groups = defaultdict(list)
for p in papers.values():
    groups[title_key(p["title"])].append(p)
duplicate_articles = []
for key, ps in groups.items():
    if len(ps) < 2:
        continue
    epmc = [p for p in ps if p["acquisition_source"] == "europepmc" and p["pmcid"]]
    duplicate_articles.append({
        "title": ps[0]["title"],
        "records": [{"paper_id": p["paper_id"], "source": p["acquisition_source"], "year": p["year"], "url": p["url"],
                     "included": not p["exclusion_reasons"], "score": p["screening_score"],
                     "outcome": p["screening_outcome"]} for p in ps],
        "ncbi_pmcid_differs_from_europepmc": bool(epmc) and any(
            p["acquisition_source"] == "ncbi" and p["pmcid"] and p["pmcid"] != epmc[0]["pmcid"] for p in ps),
    })
ncbi_papers = [p for p in papers.values() if p["acquisition_source"] == "ncbi"]

# ================================================================= 4. verbatim fallback (N-1)
raw_fields = Counter()
participant_cites = Counter()
for r in raw_rows:
    if not r["raw_text"]:
        continue
    for row in json.loads(r["raw_text"]).get("claims", []):
        for f in ("enzyme", "gene", "species", "compartment", "direction"):
            if str(row.get(f) or "").strip() and not (f == "direction" and row.get(f) == "unspecified"):
                own = row.get(f"{f}_evidence_ids")
                raw_fields[(f, "own list present" if own else ("own list missing" if own is None else "own list empty"))] += 1
        for part in (*row.get("substrates", []), *row.get("products", [])):
            participant_cites["cited" if part.get("evidence_ids") else "uncited"] += 1
verbatim_values = [(c["claim_id"], f, c.get(f)) for c in extracted for f, h in c["field_attribution"].items()
                   if h == "verbatim_cited_span"]

# ================================================================= 5. empty completions
call_rows = []
for c in calls:
    system, user = rebuild_user(c)
    rungs = per_call[c["call_id"]]
    fr = [1.0, 0.6, 0.35][: len(rungs["rows"])]
    rung_chars = [len(system) + (len(user) if f >= 1.0 else max(200, int(len(user) * f))) for f in fr]
    req_chars = [ch for ch, n in zip(rung_chars, rungs["rung_requests"]) for _ in range(n)]
    tok_per_char = c["input_tokens"] / sum(req_chars)
    call_rows.append({
        "call_id": c["call_id"], "round": c["round_index"], "paper_id": c["paper_id"],
        "passages": len(c["evidence_ids"]), "system_chars": len(system), "user_chars_rung1": len(user),
        "est_input_tokens_per_request_rung1": round(tok_per_char * rung_chars[0]),
        "rungs": [{"rung": r["attempt"], "shrink_fraction": f, "chars_sent_system_plus_user": ch,
                   "schema": "tight" if i == 2 else "full", "provider_requests": n,
                   "finish_reason": r["finish_reason"], "parse_outcome": r["parse_outcome"],
                   "raw_text_chars": len(r["raw_text"]), "raw_text": r["raw_text"] if len(r["raw_text"]) <= 40 else r["raw_text"][:40] + "...",
                   "est_input_tokens_per_request": round(tok_per_char * ch)}
                  for i, (r, f, ch, n) in enumerate(zip(rungs["rows"], fr, rung_chars, rungs["rung_requests"]))],
        "provider_requests": sum(rungs["rung_requests"]), "input_tokens": c["input_tokens"],
        "output_tokens": c["output_tokens"], "output_tokens_per_request": round(c["output_tokens"] / sum(rungs["rung_requests"]), 1),
        "status": c["status"], "finish_reason": c["finish_reason"], "response_chars": c["response_chars"],
        "claims": sum(1 for x in extracted if x["model_call_id"] == c["call_id"]),
        "latency_s": round(c["latency_ms"] / 1000, 1), "cost_usd": c["estimated_cost_usd"],
        "attempt_log_len": len(c["attempt_log"]), "retries_field": c["retries"],
        "chars_per_token": round(sum(req_chars) / c["input_tokens"], 2),
    })
failed = [r for r in call_rows if r["status"] != "ok" or any(x["parse_outcome"] != "ok" for x in r["rungs"])]
wasted_cost = sum(r["cost_usd"] for r in call_rows if r["status"] != "ok")
wasted_time = sum(r["latency_s"] for r in call_rows if r["status"] != "ok")
empty_requests_at_cap = [r["call_id"] for r in call_rows if r["status"] != "ok" and r["output_tokens"] == MAX_TOKENS * r["provider_requests"]]

p1_calls = jl(P1 / "model_calls.jsonl")
p1_same = next((c for c in p1_calls if c["paper_id"] == "PMC:PMC13364333"), None)
p2_same = next(c for c in calls if c["paper_id"] == "PMC:PMC13364333")
same_passages = p1_same is not None and set(p1_same["evidence_ids"]) == set(p2_same["evidence_ids"])

# ================================================================= 6. search
by_qid_item = defaultdict(list)
for p in papers.values():
    by_qid_item[(p["discovered_via_query_id"], p["discovered_via_frontier_id"])].append(p)
fr_items = {}
for n in (1, 2, 3):
    for it in rounds[n]["input_frontier"]:
        fr_items.setdefault(it["frontier_item_id"], it)
q_rows = []
cum = 0
seen_text = {}
for i, q in enumerate(queries, 1):
    remaining = budgets["max_papers_discovered"] - cum
    key = (q["round_index"], q["frontier_item_id"])
    deeper = q["attempt"] == 2 and seen_text.get(key) == q["query_text"]
    seen_text.setdefault(key, q["query_text"])
    window = min((2 if deeper else 1) * budgets["max_results_per_query"], remaining)
    new = by_qid_item[(q["query_id"], q["frontier_item_id"])]
    cum += len(new)
    q_rows.append({
        "n": i, "round": q["round_index"], "query_id": q["query_id"], "frontier_item_id": q["frontier_item_id"],
        "kind": q["frontier_kind"], "target": fr_items.get(q["frontier_item_id"], {}).get("target", ""),
        "attempt": q["attempt"], "deeper": deeper, "window": window, "result_count": q["result_count"],
        "window_filled": q["result_count"] >= window > 0, "new_discovered": len(new),
        "new_included": [p["paper_id"] for p in new if not p["exclusion_reasons"]],
        "new_by_source": dict(Counter(p["acquisition_source"] for p in new)),
        "cumulative_discovered": cum, "query_text": q["query_text"], "executed_at": q["executed_at"],
        "cache_hit": q["cache_hit"],
    })
assert cum == len(papers), (cum, len(papers))
dup_qids = {k: v for k, v in Counter(q["query_id"] for q in queries).items() if v > 1}
item_rows = []
for fid in dict.fromkeys(q["frontier_item_id"] for q in queries):
    qs = [r for r in q_rows if r["frontier_item_id"] == fid]
    for rnd in sorted({r["round"] for r in qs}):
        rq = [r for r in qs if r["round"] == rnd]
        it = fr_items[fid]
        exhausted_now = fid in state["exhausted_frontier"] and rnd == 2
        if exhausted_now and any(r["result_count"] == 0 for r in rq):
            why = "zero results (exhausted at once)"
        elif exhausted_now and all(r["window"] < budgets["max_results_per_query"] for r in rq):
            why = "every attempt's window was cut to %s by the remaining discovery budget" % "/".join(str(r["window"]) for r in rq)
        elif exhausted_now:
            why = "attempts found no new included paper"
        else:
            why = "open"
        item_rows.append({"round": rnd, "frontier_item_id": fid, "kind": it["kind"], "target": it["target"],
                          "attempts": [("wider window %d" % r["window"]) if r["deeper"] else ("attempt %d" % r["attempt"]) for r in rq],
                          "results": [r["result_count"] for r in rq], "windows": [r["window"] for r in rq],
                          "new_discovered": sum(r["new_discovered"] for r in rq),
                          "new_included": [p for r in rq for p in r["new_included"]], "outcome": why})
round_papers = {n: Counter(p["acquisition_source"] for p in papers.values() if p["discovered_round"] == n) for n in (1, 2, 3)}
unasked_r2 = [it for it in rounds[2]["input_frontier"] if it["frontier_item_id"] not in {q["frontier_item_id"] for q in queries if q["round_index"] == 2}]
asked_r2 = len(rounds[2]["input_frontier"]) - len(unasked_r2)
cap_query = next(r for r in q_rows if r["cumulative_discovered"] >= budgets["max_papers_discovered"])
first_throttled = next(r for r in q_rows if r["window"] < budgets["max_results_per_query"] and not r["deeper"])
window1 = [r for r in q_rows if r["window"] == 1]
exhaust_why = Counter(r["outcome"].split(" (")[0] if r["outcome"].startswith("zero") else
                      ("window cut by discovery budget" if r["outcome"].startswith("every attempt's window") else r["outcome"])
                      for r in item_rows if r["round"] == 2 and r["outcome"] != "open")


def secs(ts):
    from datetime import datetime
    return (datetime.fromisoformat(ts) - datetime.fromisoformat(manifest["started_at"])).total_seconds()


# ================================================================= 7. pilot-001 side by side
def basic(run):
    s = js(run / "summary.json")
    m = js(run / "run_manifest.json")
    ps = jl(run / "papers.jsonl")
    cs = jl(run / "model_calls.jsonl")
    qs = jl(run / "queries.jsonl")
    errs = js(run / "state.json")["errors"]
    inc = [p for p in ps if not p["exclusion_reasons"]]
    ex = [c for d in sorted((run / "rounds").iterdir()) if (d / "claims_extracted.jsonl").exists()
          for c in jl(d / "claims_extracted.jsonl")]
    attr = Counter(h for c in ex for h in (c.get("field_attribution") or {}).values())
    return OrderedDict([
        ("code", m["git"]["commit"][:8]), ("prompt_version", m["prompt_version"]),
        ("stop_reason", m["stop_reason"]), ("rounds", s["rounds_completed"]),
        ("elapsed_s", s["costs"]["elapsed_seconds"]), ("queries (rows)", len(qs)),
        ("papers discovered", s["papers"]["discovered"]), ("papers screened", s["papers"]["screened"]),
        ("papers included", len(inc)),
        ("  by production / route 1 / route 2", "%d / %d / %d" % (sum(route_of(p) == "production" for p in inc),
                                                                 sum(route_of(p) == "mining_rescue route 1" for p in inc),
                                                                 sum(route_of(p) == "mining_rescue route 2" for p in inc))),
        ("papers processed", s["papers"]["processed"]), ("passages", s["passages"]["extracted"]),
        ("claims extracted", s["usage"]["claims_extracted"]), ("distinct identities", s["claims"]["distinct_identities"]),
        ("duplicates merged", s["usage"]["claims_extracted"] - s["claims"]["distinct_identities"]),
        ("primary reactions", s["graph"]["primary_reactions"]), ("candidate reactions", s["graph"]["candidate_reactions"]),
        ("candidate components", s["graph"]["candidate_components"]),
        ("rejected (status)", json.dumps(s["claims"]["rejected_by_reason"])),
        ("species states (nodes)", json.dumps(dict(sorted(s["graph"]["species_states"].items())))),
        ("model calls (logical, by status)", json.dumps(s["model_calls"]["by_status"])),
        ("LLM calls (ladder rungs, budgeted)", s["usage"]["llm_calls"]),
        ("input / output tokens", "%d / %d" % (s["costs"]["input_tokens"], s["costs"]["output_tokens"])),
        ("est. cost USD", s["costs"]["estimated_cost_usd"]),
        ("uncited field drops", len(errs)),
        ("fields kept: model_citation / verbatim_cited_span", "%d / %d" % (attr.get("model_citation", 0), attr.get("verbatim_cited_span", 0))),
        ("evidence type of included", json.dumps(s["papers"]["by_evidence_type"])),
    ])


side = {"pilot-001": basic(P1), "pilot-002": basic(RUN)}
side["pilot-001"]["fields kept: model_citation / verbatim_cited_span"] += " (field_attribution did not exist; 0 fields kept)"

# ================================================================= write JSON
result = OrderedDict(
    audit="ORCH-743 pilot-002 artifact audit (read-only)",
    run_dir=str(RUN), code="C:/t/mining-pilot2 @ %s" % manifest["git"]["commit"],
    generated_by=["build_audit.py"],
    headline={k: manifest[k] for k in ("stop_reason", "stop_detail", "completion_state", "rounds_completed")} | {
        "elapsed_seconds": costs["elapsed_seconds"], "estimated_cost_usd": costs["estimated_cost_usd"]},
    counts=counts, budgets=[{"budget": a, "used": b, "limit": c} for a, b, c in budget_rows],
    reactions=reaction_rows,
    primary_species={"evidence_ids": prim["species_evidence_ids"], "attribution": prim["species_attribution"],
                     "raw_model_species": prim_payload.get("species"), "raw_species_evidence_ids": prim_payload.get("species_evidence_ids"),
                     "two_spans_identical_text": prim_same_text,
                     "species_sentences": sentences_with(prim_text, ["This study aimed for characterizing the four annotated A. thaliana", "in cascade with 12-OPDA pathway enzymes"])},
    pepper_span={"species_sentence": pepper_species_sentence, "in_vitro_trigger_sentence": pepper_invitro_sentence},
    seed_named_entities=seed_named,
    included_papers=included_rows, ncbi_identity_proof=ncbi_proof,
    duplicate_articles=duplicate_articles,
    ncbi_counts={"discovered": len(ncbi_papers), "with_pmcid": sum(1 for p in ncbi_papers if p["pmcid"]),
                 "included": sum(1 for p in ncbi_papers if not p["exclusion_reasons"]),
                 "unique_articles_discovered": len(groups)},
    excluded_titles_naming_pathway=excluded_pathway_titles,
    verbatim_fallback={"values": verbatim_values, "raw_field_citation_lists": {f"{a}:{b}": n for (a, b), n in sorted(raw_fields.items())},
                       "raw_participant_citations": dict(participant_cites)},
    model_calls=call_rows,
    empty_completion_summary={"logical_calls": len(calls), "ladder_rungs": ladder_rungs, "provider_requests": provider_requests,
                              "rungs_empty": rung_outcomes.get("empty_completion", 0), "log_giving_up_lines": giving_up,
                              "inner_retry_limit": inner_max, "calls_failed": [r["call_id"] for r in call_rows if r["status"] != "ok"],
                              "failed_calls_output_tokens_equal_max_tokens_x_requests": empty_requests_at_cap,
                              "failed_cost_usd": round(wasted_cost, 6), "failed_latency_s": round(wasted_time, 1),
                              "pilot001_same_paper_same_passages": same_passages,
                              "pilot001_same_paper_call": {k: p1_same[k] for k in ("input_tokens", "output_tokens", "response_chars", "finish_reason", "prompt_version")} if p1_same else None},
    search={"queries": q_rows, "items": item_rows, "duplicate_query_ids": dup_qids,
            "papers_discovered_by_round_and_source": {str(n): dict(v) for n, v in round_papers.items()},
            "round2_frontier": len(rounds[2]["input_frontier"]), "round2_items_asked": asked_r2,
            "round2_items_never_asked": [(it["kind"], it["target"]) for it in unasked_r2],
            "round2_exhaustions": dict(exhaust_why),
            "cap_reached_at_query": cap_query["n"], "cap_reached_seconds_after_start": secs(cap_query["executed_at"]),
            "first_throttled_query": first_throttled["n"], "queries_with_window_1": len(window1),
            "cache_hit_true": sum(1 for q in queries if q["cache_hit"])},
    side_by_side=side,
    span_texts={sp["evidence_id"]: evidence[sp["evidence_id"]]["text"] for x in reaction_rows for sp in x["spans"]},
)
(OUT / "pilot_002_audit.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

# ================================================================= write MD
md = []
add = md.append
from md_text import render  # noqa: E402  (narrative lives in md_text.py; numbers come from `result`)

render(add, result, cell)
(OUT / "PILOT-002-AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("wrote", OUT / "PILOT-002-AUDIT.md", "and", OUT / "pilot_002_audit.json")
print(json.dumps({k: v["value"] for k, v in counts.items()}, ensure_ascii=False, default=str, indent=1))

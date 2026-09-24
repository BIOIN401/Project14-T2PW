"""Markdown renderer for the PILOT-002 audit. Numbers come from build_audit.py's result dict."""
from __future__ import annotations

import json


def render(add, r, cell):
    c = r["counts"]
    h = r["headline"]
    ec = r["empty_completion_summary"]
    s = r["search"]
    calls = {x["call_id"]: x for x in r["model_calls"]}
    inc = r["included_papers"]
    rx = r["reactions"]
    prim = rx[0]
    ps = r["primary_species"]
    v = lambda k: c[k]["value"]  # noqa: E731
    dup = r["duplicate_articles"]
    nc = r["ncbi_counts"]
    ex2 = s["round2_exhaustions"]
    q = s["queries"]
    cap_q = q[s["cap_reached_at_query"] - 1]
    q149 = next(x for x in q if x["cumulative_discovered"] >= 149)
    wrong_ncbi = sorted({rec["paper_id"] for d in dup if d["ncbi_pmcid_differs_from_europepmc"] for rec in d["records"] if rec["source"] == "ncbi"}
                        | {p["paper_id"] for p in inc if p["fetched_text_names_this_title"] is False})

    add("# PILOT-002 audit: pathway mining, jasmonate biosynthesis, *Arabidopsis thaliana*")
    add("")
    add(f"Read-only evaluation of `{r['run_dir']}` (code `{r['code']}`, prompt `mining-extract/1.1.0`), for ORCH-743. "
        "Nothing was re-run and no model was called. Every number below is read from a named artifact field by "
        "[`build_audit.py`](build_audit.py) (narrative in [`md_text.py`](md_text.py)), run through `bounded_run.py` with "
        "`PYTHONPATH=C:/t/mining-pilot2/src` and `PYTHONDONTWRITEBYTECODE=1`. The frozen tree is imported only for pure "
        "functions: the extraction prompt builder (with a capturing stub client, to measure the exact input), the title "
        "organism detector `Acquirer._other_organisms_in_title`, and the PubMed XML parser on a synthetic record. "
        "Machine-readable twin: [`pilot_002_audit.json`](pilot_002_audit.json).")
    add("")
    add(f"**Headline.** `{h['completion_state']}` / `{h['stop_reason']}` ({h['stop_detail']}), {h['rounds_completed']} rounds "
        f"(the third asked nothing), {h['elapsed_seconds']} s, est. ${h['estimated_cost_usd']}. {v('papers_discovered')} papers discovered "
        f"({nc['unique_articles_discovered']} distinct articles), {v('papers_included')} included, {v('papers_processed')} processed. "
        f"{v('claims_extracted')} claims -> 11 reaction nodes: **1 primary** (`target_species_supported`) and 10 candidates.")
    add("")
    add("**Key findings.**")
    add("")
    add(f"1. **The one primary reaction is real biology on a mislabelled paper.** It is the 13-LOX step, alpha-linolenic acid -> 13-HPOT, "
        f"catalysed by the four *A. thaliana* 13-LOXs (LOX2/3/4/6). Its species is **model-cited** (`species_evidence_ids` present "
        f"in the raw completion; attribution `model_citation`), not recovered by the fallback. But its two cited spans are the same "
        f"1,775-character abstract indexed twice (`abstract#0` and `body#0`), and the paper id it cites, `PMC:PMC411572` with URL "
        f"`doi.org/10.1073/pnas.76.9.4350`, is not this 2021 study: that DOI is PNAS volume 76 (1979) (A1, A4).")
    add(f"2. **NCBI records carry another article's PMCID.** `rag/acquire.py:536` reads `.//ArticleIdList/ArticleId` over the whole "
        f"PubMed record, reference list included, so the last cited reference's `pmc` and `doi` win. The frozen parser does exactly "
        f"that on a synthetic record ({r['ncbi_identity_proof']['parser_returned']['id']} returned for a record whose own PMCID is "
        f"PMC999001). In this run {len(wrong_ncbi)} NCBI ids are provably wrong. For 2 included papers the full-text fetch then pulled "
        f"the wrong article, and extraction read it as the included paper. The NAD+ paper was read as a cotton-bollworm effector "
        f"paper, and 'Arabidopsis GH3.10 conjugates jasmonates' as a 2016 wound-amidohydrolase paper. "
        f"{len(dup)} articles were discovered twice under two ids. This is production code that pilot-001 never reached (NCBI was new in item 3).")
    add(f"3. **Empty completions are output-budget exhaustion.** {len(ec['calls_failed'])} calls failed outright. They were the two "
        f"20-passage inputs, about 22.9k and 17.6k input tokens per request. All 9 provider requests of each ended "
        f"`finish_reason=length` with **zero** content and exactly {r['model_calls'][1]['output_tokens_per_request']:.0f} output tokens "
        f"(= `max_tokens`). The hidden reasoning spent the whole budget before any JSON. The shrink ladder (60%, 35% + tight schema) "
        f"did not help. Those two calls cost ${ec['failed_cost_usd']} of ${h['estimated_cost_usd']} "
        f"({100 * ec['failed_cost_usd'] / h['estimated_cost_usd']:.0f}%) and {ec['failed_latency_s']} s of {h['elapsed_seconds']} s. "
        f"Pilot-001 extracted from **the same 20 passages** of PMC13364333 under prompt 1.0.0 in one request with 3,629 output tokens.")
    add(f"4. **Citation repair (item 2) worked.** 0 uncited-field drops (pilot-001: 59 of 59). 53 kept fields are all "
        f"`model_citation`; `verbatim_cited_span` was never used, so bio watch item N-1 has nothing to adjudicate.")
    add("5. **Species labels are still keyword artefacts (F3, new form).** All 9 pepper-paper reactions are `in_vitro_or_cell_free`. "
        "The model cited *Capsicum annuum* for 8 of them. The label came from one incidental sentence in the introduction, 'in vitro "
        "studies confirmed that the unphosphorylatable AtLOX2 ...', which describes other work. Conversely the primary's abstract describes "
        "enzyme kinetics and a reconstituted cascade, but matches no in-vitro marker. Nothing wrong was admitted, but the labels and the "
        "primary's state need a bio-auditor ruling (A3).")
    add(f"6. **`max_papers_discovered=150` is now the binding stop, and it starves the search.** 3 round-1 queries brought in 66 papers. "
        f"By query {q149['n']} of 58 the run had 149. From query {s['first_throttled_query']} on, each query's window was cut to the "
        f"remaining budget (11, then 1), so **{s['queries_with_window_1']} queries asked for a single result**. "
        f"{ex2.get('window cut by discovery budget', 0)} of the {sum(ex2.values())} items exhausted in round 2 were exhausted on windows "
        f"the budget had cut. 15 of 44 round-2 items were never asked. Round 3 then started and asked nothing (A5, A6).")
    add("7. **Round 2 processed 5 papers and got 0 claims.** One was a wheat transcriptome paper (empty completion, then "
        "`{ \"claims\": [] }` at 60%). Four were abstract-only rescues. Two of those are on-topic (the AOC family of *A. thaliana*; "
        "AtLOX2 kinetics), but their abstracts name no complete substrate -> product conversion. The model returned `{\"claims\": []}`, "
        "which the prompt calls a correct answer.")
    add(f"8. **The rescue (item 4) did its job once.** Route 2 re-admitted the paper that gave the only primary reaction (matched term "
        f"`12-OPDA`). Route 1 re-admitted 5 papers. One of them is a grape paper where *A. thaliana* is only an ectopic-expression host, "
        f"a false rescue inside the documented ceiling. It was never processed (A8).")
    add("")

    # ------------------------------------------------------------ 1
    add("## 1. Pre-registered counts")
    add("")
    add("| metric | value | source field | note |")
    add("|---|---|---|---|")
    for k, row in c.items():
        add(f"| {k} | `{cell(json.dumps(row['value'], ensure_ascii=False))}` | {cell(row['source'])} | {cell(row['note'])} |")
    add("")
    add("Included papers by route: " + ", ".join(f"**{k}** {n}" for k, n in
                                                  sorted({x['route']: sum(1 for y in inc if y['route'] == x['route']) for x in inc}.items())) + ".")
    add("")
    add(f"**Model calls, three ways.** `model_calls.jsonl` has {ec['logical_calls']} logical calls ({v('model_calls_by_status')}). "
        f"`usage.llm_calls` = {ec['ladder_rungs']} counts shrink-ladder rungs, one row each in `model_raw.jsonl`; that is the budgeted "
        f"'LLM calls'. Every rung goes to `t2pw.llm.client.chat_detailed`. It re-sends an identical request up to {ec['inner_retry_limit']} "
        f"times on an empty 200, which gives **{ec['provider_requests']} provider requests** (+1 preflight probe). The log has "
        f"{ec['log_giving_up_lines']} 'giving up' lines, one per rung whose {ec['inner_retry_limit']} requests all came back empty: 3 rungs of "
        f"`mc_001_extract_claims_0002`, 3 of `_0003` and 1 of `mc_002_extract_claims_0006`. The status file shows only 2 failures because "
        f"`_0006` recovered on rung 2 and is recorded `ok`. The log's one `finish_reason=error` line is `_0009`'s first provider request; "
        f"its second succeeded.")
    add("")
    add("Budgets (used / limit):")
    add("")
    add("| budget | used | limit |")
    add("|---|---|---|")
    for b in r["budgets"]:
        add(f"| {b['budget']} | {b['used']} | {b['limit']} |")
    add("")

    # ------------------------------------------------------------ 2
    add("## 2. Every reaction (1 primary + 10 candidates)")
    add("")
    add("The seed connection is the auditor's name-level reading against the canonical route (13-LOX -> AOS -> AOC -> OPR3 -> "
        "beta-oxidation -> JA -> JAR1), used only as a yardstick. 'Graph' lists the run's own chemical connections. No entity is named "
        "`12-OPDA`; the seed metabolite exists only as `(9S,13S)-12-oxo-phytodienoic acid (OPDA)` "
        f"(`{[k for k, n in r['seed_named_entities'].items() if 'OPDA)' in (n or '')][0]}`), and AOS as the enzyme entity "
        "`allene oxide synthase (AOS)`.")
    add("")
    add("| # | reaction id | graph | equation (as recorded) | enzyme (attribution) | species state / value | paper | seed connection | decision |")
    add("|---|---|---|---|---|---|---|---|---|")
    for i, x in enumerate(rx, 1):
        enz = "; ".join(f"{e['name']} ({e['attribution']})" for e in x["enzymes"]) or "none"
        add(f"| {i} | `{x['reaction_id']}` | {x['graph_status']} `{x['component_id']}` | {cell(x['equation'])} | {cell(enz)} | "
            f"`{x['species_state']}` / {cell(x['species_value'] or '(empty)')} | `{x['paper_id']}` | {cell(x['seed_connection'])} | "
            f"`{x['decision_status']}` |")
    add("")
    add("### The primary reaction in full")
    add("")
    add(f"- **Reaction:** `{prim['reaction_id']}` in component `{prim['component_id']}` (the primary component; 0 chemical connections). "
        f"Equation: {prim['equation']}.")
    add(f"- **Enzyme:** {prim['enzymes'][0]['name']} (`{prim['enzymes'][0]['attribution']}`); gene {prim['gene']['value']} "
        f"(`{prim['gene']['attribution']}`). Direction `unspecified`, no compartment extracted, though the abstract states plastid targeting.")
    add(f"- **Species:** `{prim['species_state']}`, value \"{prim['species_value']}\", attribution **`{ps['attribution']}`**. The raw completion "
        f"(`rounds/round_001/model_raw.jsonl`, `{prim['model_call_id']}`, attempt 1) carries `\"species\": \"{ps['raw_model_species']}\"` and "
        f"`\"species_evidence_ids\": {json.dumps(ps['raw_species_evidence_ids'])}`. So the model cited it: this is not the "
        f"`verbatim_cited_span` fallback. Reason recorded: \"{prim['species_reason']}\".")
    add(f"- **The span that backs the species:** `{ps['evidence_ids'][0]}` (`abstract#0`, chars 0-1775) and `{ps['evidence_ids'][1]}` "
        f"(`body#0`, chars 0-1775). Identical text: **{ps['two_spans_identical_text']}**. The paper has no full text, so its abstract "
        f"was indexed as both abstract and body. The 'two evidence spans' are one passage. The sentences that tie *A. thaliana* to "
        f"the enzymes, verbatim:")
    for sent in ps["species_sentences"]:
        add(f"  > {sent}")
    add("")
    add("  The span never writes \"Arabidopsis thaliana\" in full; the model expanded \"A. thaliana\", and `normalize_species` folds both.")
    add(f"- **Source paper:** `{prim['paper_id']}`, \"{prim['paper_title'].rstrip('.')}\". The recorded id and URL belong to another article (A1). "
        f"The title and abstract are internally consistent: both come from the PubMed record.")
    add(f"- **Seed connection:** {prim['seed_connection']}: {prim['seed_how']}. It is **not connected** in the graph. Its product "
        "'13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT)' and the candidate AOS reaction's substrate "
        "'(13S)-hydroperoxy-octadecatrienoic acid (13-HPOT)' are two lexical entities (A9).")
    add(f"- **Decision:** `{prim['decision_status']}` (`{prim['decision_id']}`, evidence level `{prim['evidence_level']}`): "
        f"\"{prim['decision_reason']}\".")
    add("- **Auditor's note on the species state (not a verdict).** The abstract reports kinetics (\"Vmax app ... Km\") and says the "
        "13-LOXs \"in cascade with 12-OPDA pathway enzymes, synthesized 12-OPDA\". `species_policy.md` defines `in_vitro_or_cell_free` "
        "as a \"purified-enzyme\" or \"reconstituted system\". No `_IN_VITRO_MARKERS` phrase occurs, so the keyword test passed it. The "
        "biology is textbook-correct for *A. thaliana*. Whether recombinant characterisation of the target's own enzymes counts as "
        "target-species evidence is a `policy_disagreement` question for `pwml-bio-auditor`.")
    add("")
    add("### Per-reaction detail")
    add("")
    for i, x in enumerate(rx, 1):
        add(f"#### {i}. `{x['reaction_id']}`: {x['equation']}")
        add("")
        add(f"- **Graph:** {x['graph_status']}, component `{x['component_id']}`"
            + (f", quarantine reason `{x['quarantine_reason']}`" if x['quarantine_reason'] else "")
            + f"; duplicate merges: {x['duplicate_merges']}; hypothesis: {x['hypothesis']}")
        enz = "; ".join(f"{e['name']} (`{e['attribution']}`)" for e in x["enzymes"]) or "none"
        add(f"- **Enzyme(s):** {enz}. **Gene:** {x['gene']['value'] or 'none'}"
            + (f" (`{x['gene']['attribution']}`)" if x['gene']['value'] else ""))
        add(f"- **Species:** `{x['species_state']}`; value {x['species_value'] or '(empty)'}"
            + (f" (`{x['species_attribution']}`, cites {', '.join(x['species_evidence_ids'])})" if x['species_value'] else "")
            + f". Reason: \"{x['species_reason']}\"")
        add(f"- **Source paper:** `{x['paper_id']}`, \"{x['paper_title'].rstrip('.')}\". Claim(s) {', '.join(x['claim_ids'])} from `{x['model_call_id']}`")
        for sp in x["spans"]:
            extra = f"; identical text to {', '.join(sp['same_text_as'])}" if sp["same_text_as"] else ""
            add(f"- **Span `{sp['evidence_id']}`** ({sp['section']}, `{sp['locator']}`, chars {sp['char_start']}-{sp['char_end']}{extra})"
                + (": verbatim excerpt, full text in Appendix A:" if sp["excerpt"] else ""))
            for sent in sp["excerpt"]:
                add(f"  > {sent}")
        nb = "; ".join(f"{n['kind']} via '{n['via']}' with `{n['reaction_id']}`" for n in x["graph_neighbours"]) or "none"
        add(f"- **Seed connection:** {x['seed_connection']}: {x['seed_how']}. Graph: {nb}")
        add(f"- **Decision:** `{x['decision_status']}` (`{x['decision_id']}`, `{x['evidence_level']}`): \"{x['decision_reason']}\"")
        add("")
    pep = r["pepper_span"]
    add("**Why the 9 pepper reactions are `in_vitro_or_cell_free`.** All cite the one introduction span `ev_f7c56801adf6792d`. "
        "The model cited *Capsicum annuum* for 8 of them, from:")
    add(f"> {pep['species_sentence'][0]}")
    add("")
    add("The in-vitro label comes from this sentence about other work, which is test 2 in `classify_species` and runs before the "
        "species comparison:")
    add(f"> {pep['in_vitro_trigger_sentence'][0]}")
    add("")
    add("The passage itself is a literature summary of the pathway, largely Arabidopsis work, and not an in-vitro experiment. "
        "Holding every one back is the right outcome. The honest label for the 8 with a cited species is `non_target_species` "
        "(*Capsicum annuum*). The one with no species is closer to `species_not_stated`.")
    add("")

    # ------------------------------------------------------------ 3
    add("## 3. Every included paper")
    add("")
    add("Title organisms are the frozen `Acquirer._other_organisms_in_title`, replayed read-only on the recorded title. "
        "'Text used' is what extraction read.")
    add("")
    add("| # | paper id | title | source | how included | matched term / recorded reason | title organisms (non-target) | text used | passages / call / claims |")
    add("|---|---|---|---|---|---|---|---|---|")
    for i, p in enumerate(inc, 1):
        how = p["route"]
        reason = p["inclusion_reason"]
        if p["production_exclusion"]:
            reason += " -- " + p["production_exclusion"].replace("the production screen excluded it: ", "production screen: ")
        text = p["text_used"]
        if p["fetched_text_names_this_title"] is False:
            text += f" **of a different article** (\"{p['fetched_text_header'][:90]}...\")"
        add(f"| {i} | `{p['paper_id']}` | {cell(p['title'])} | {p['source']} | {how} | {cell(reason)} | "
            f"{cell(', '.join(p['title_other_organisms']) or 'none')} | {cell(text)} | "
            f"{p['passages']} / {p['model_call'] or '-'} / {p['claims_extracted']} |")
    add("")
    r2 = [p for p in inc if p["route"] == "mining_rescue route 2"][0]
    add(f"**Route 2** (`{r2['paper_id']}`): matched species form `{r2['matched_species_form']}` and term `{r2['matched_term']}` (a seed). "
        "The sentence, re-found by splitting title + abstract with the frozen `_SENTENCE`:")
    add(f"> {r2['route2_sentence']}")
    add("")
    add("The production screen had called it `off_topic` (score 3.0). The seed enzyme `AOS` can never be a route-2 term: it is 3 "
        "characters, under `_MIN_TERM_CHARS = 5`. So round 1 had one usable rescue term, `12-OPDA`.")
    add("")
    add("**Route 1** (5 papers, all round 2, all abstract-only NCBI records):")
    for p in inc:
        if p["route"] != "mining_rescue route 1":
            continue
        fh = f" The abstract contains the foreign-host word \"{p['foreign_host_in_abstract']}\"; the guard reads the title only." if p["foreign_host_in_abstract"] else ""
        add(f"- `{p['paper_id']}` score {p['screening_score']}: {p['title']} Claims: {p['claims_extracted']}; processed round {p['processed_round']}.{fh}")
    add("")
    add("`PMC:PMC5104190` is a false rescue. It is about grape (*Vitis amurensis*) PAT1, and its abstract says \"ectopic expression of "
        "VaPAT1 enhances cold tolerance in Arabidopsis\". The title word 'grape' is unknown to both organism detectors and to "
        "`_EXTRA_HOSTS`, the documented ceiling. It was never processed: retrieval returned no passage for it (A7).")
    add("")
    add("**Production inclusions worth noting.** `PMC:PMC5382222` is a wheat transcriptome paper. The title detector finds 'wheat' in it, "
        "but that check applies only to route 1; the production screen passed it (score 2.75). `PMC:PMC13364333` is the Marchantia "
        "review pilot-001 read. It still has evidence type `unknown` (F6).")
    add("")
    add("### NCBI identity evidence (A1)")
    add("")
    add(f"NCBI supplied {nc['discovered']} of {v('papers_discovered')} discovered papers ({nc['with_pmcid']} with a PMC id) and "
        f"{nc['included']} of {v('papers_included')} included. The parser reads the PMCID and DOI from any `ArticleIdList` in the "
        "record, and the last one wins (`rag/acquire.py:536-543`). A synthetic record run through the frozen `_parse_pubmed_article` "
        f"(own ids {json.dumps(r['ncbi_identity_proof']['synthetic_record_own_ids'])}, one cited reference "
        f"{json.dumps(r['ncbi_identity_proof']['cited_reference_ids'])}) returns `{json.dumps(r['ncbi_identity_proof']['parser_returned'])}`.")
    add("")
    add("| evidence | NCBI paper id | what the record says | what the id actually is |")
    add("|---|---|---|---|")
    for p in inc:
        if p["fetched_text_names_this_title"] is False:
            add(f"| full text fetched for the recorded PMCID | `{p['paper_id']}` | \"{cell(p['title'])}\" ({p['year']}) | "
                f"\"{cell(p['fetched_text_header'][:110])}...\" (header of the fetched full text) |")
    for d in dup:
        e = [x for x in d["records"] if x["source"] == "europepmc"]
        n = [x for x in d["records"] if x["source"] == "ncbi"]
        if e and n:
            add(f"| same article also found via EuropePMC | `{n[0]['paper_id']}` (included: {n[0]['included']}) | \"{cell(d['title'][:100])}\" | "
                f"EuropePMC gives it `{e[0]['paper_id']}` (included: {e[0]['included']}) |")
    p411 = next(p for p in inc if p["paper_id"] == "PMC:PMC411572")
    add(f"| DOI inconsistent with the record | `PMC:PMC411572` | \"{cell(p411['title'])}\" (year {p411['year']}) | URL `{p411['url_recorded']}`: PNAS volume 76, i.e. 1979 |")
    add("")
    add("Consequences in this run:")
    add("- The only primary reaction cites the wrong paper id.")
    add("- Two extraction calls read another article's full text under the included paper's title (`PAPER:` header). The NAD+ paper's "
        "one claim came from its own PubMed abstract (`ev_49c948bc587a2016`, null offsets because that abstract is not a substring of "
        "the fetched text). Its other 7 passages were the cotton-bollworm paper.")
    add("- The NAD+ article was screened twice. The EuropePMC copy was excluded (score 3.0, `omics_only`); the NCBI copy was included "
        "(4.5).")
    add(f"- {v('papers_discovered')} 'discovered' papers are {nc['unique_articles_discovered']} distinct articles.")
    add("")
    add("### Recall check: excluded titles that name the pathway")
    add("")
    add("| paper id | title | source | score | exclusion | title names Arabidopsis |")
    add("|---|---|---|---|---|---|")
    for p in r["excluded_titles_naming_pathway"]:
        add(f"| `{p['paper_id']}` | {cell(p['title'])} | {p['source']} | {p['score']} | {cell(p['reason'][:90])} | {p['names_arabidopsis']} |")
    add("")
    add("Most of these are rightly out: wrong organism, transgenic hosts, reviews, signalling, or transcriptomics. A curator would "
        "want three for *A. thaliana* jasmonate biosynthesis:")
    add("")
    add("- `PMC:PMC4844422` 'Activity Regulation by Heteromerization of Arabidopsis Allene Oxide Cyclase Family Members': AOC "
        "enzymology, `off_topic` at 0.25. Route 1 never takes `off_topic`, and route 2 had no matching run term: 'allene oxide "
        "cyclase' is neither a seed nor a `target_species_supported` reaction name.")
    add("- The CYP20-3 / OPDA-flux paper (2.25), a documented borderline case.")
    add("- OPR3, `PMID:31834385` (`off_topic` at 1.5). Its title does not name the species, so route 2 cannot reach it.")
    add("")
    add("The AtLOX2 transcriptomics paper is borderline.")
    add("")

    # ------------------------------------------------------------ 4
    vf = r["verbatim_fallback"]
    add("## 4. `verbatim_cited_span` enzyme / gene values (bio watch item N-1)")
    add("")
    add(f"**None.** No extracted claim (of 16) and no stored claim (of 11) has a field attributed `verbatim_cited_span` "
        f"(`field_attribution`, both files). There is no row pairing to check. The raw lineage shows why. Every filled field in the 3 "
        f"productive completions came with its own list: {json.dumps(vf['raw_field_citation_lists'])}. The fallback only fires when a "
        "list is missing.")
    add("")
    add(f"Participants are the exception. Prompt 1.1.0 asks every substrate and product for its own `evidence_ids`, but "
        f"{vf['raw_participant_citations'].get('uncited', 0)} of "
        f"{vf['raw_participant_citations'].get('uncited', 0) + vf['raw_participant_citations'].get('cited', 0)} raw participants "
        "have none; only the NAD+ claim's two do. `build_claims` never drops a participant, and the claim-level `evidence_ids` still "
        "carry the span, so nothing was lost. `evidence_policy.md` nonetheless describes participants as 'each participant with its "
        "own evidence ids' (A15).")
    add("")

    # ------------------------------------------------------------ 5
    add("## 5. Empty completions")
    add("")
    add("Input sizes are exact: the frozen `extract_claims` rebuilt each call's system and user messages from `evidence_spans.jsonl` "
        "into a capturing stub. The shrink ladder truncates the user string to 100% / 60% / 35%, with the tight schema on rung 3 "
        "(`modelclient.py:50, 271-280`). `input_tokens` in `model_calls.jsonl` is the provider's total over every request of the call. "
        "The per-request token figure splits that total in proportion to characters sent; for single-request calls it is exact.")
    add("")
    add("| call | round | paper | passages | user chars (rung 1) | est. input tok / request (rung 1) | rungs: fraction -> outcome (requests) | provider requests | input tok | output tok | output tok / request | response chars | claims | latency s | cost USD |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in r["model_calls"]:
        rungs = "; ".join(f"{g['shrink_fraction']} {g['schema']} -> {g['finish_reason']}/{g['parse_outcome']} ({g['provider_requests']})" for g in x["rungs"])
        add(f"| `{x['call_id']}` | {x['round']} | `{x['paper_id']}` | {x['passages']} | {x['user_chars_rung1']} | "
            f"{x['est_input_tokens_per_request_rung1']} | {cell(rungs)} | {x['provider_requests']} | {x['input_tokens']} | "
            f"{x['output_tokens']} | {x['output_tokens_per_request']} | {x['response_chars']} | {x['claims']} | {x['latency_s']} | {x['cost_usd']} |")
    add("")
    add("**What the raw lineage shows.** For each failed rung, `rounds/round_001/model_raw.jsonl` holds a row like:")
    add("")
    add("```json")
    add('{"attempt": 1, "call_id": "mc_001_extract_claims_0002", "call_status": "empty_completion", "error": "empty completion (finish_reason=length)", '
        '"finish_reason": "length", "model": "deepseek/deepseek-v4-flash", "paper_id": "PMC:PMC13364333", "parse_outcome": "empty_completion", '
        '"purpose": "extract_claims", "raw_text": "", "response_hash": "", "round_index": 1}')
    add("```")
    add("")
    add("There are 7 such rows: rungs 1-3 of `_0002` and `_0003`, and rung 1 of `_0006`. Every one has `raw_text: \"\"`. Not one "
        "character of JSON came back, so this is not truncated JSON. `_0006` rung 2 returned `{ \"claims\": [] }` (16 chars) and the "
        "four round-2 abstract calls returned `{\"claims\": []}`.")
    add("")
    c2, c3, c6 = calls["mc_001_extract_claims_0002"], calls["mc_001_extract_claims_0003"], calls["mc_002_extract_claims_0006"]
    add("**Classification: output-budget exhaustion. The reasoning spent `max_tokens` before any content.**")
    add(f"- `_0002` and `_0003` each billed exactly {c2['output_tokens']} and {c3['output_tokens']} output tokens over 9 requests: "
        f"4,000 per request, the configured `max_tokens`. Each ended `length` with an empty `content`. A request that had begun emitting "
        "JSON would have returned a truncated, non-empty string (`truncated_completion`), and none did. The model config sets "
        "`reasoning: \"\"`, so no reasoning budget or effort is passed. Only visible `content` is read (`chat_detailed`).")
    add(f"- `_0006`: 3 x 4,000 on rung 1, then {c6['output_tokens'] - 12000} output tokens for 16 characters on rung 2. Successful calls "
        "show the same reasoning overhead. `_0008` spent 940 output tokens and `_0010` 611 to say `{\"claims\": []}` (14 chars). "
        "`_0001` spent 2,661 on 774 chars.")
    add("- There is no provider error and no content filter: every empty `finish_reason` is `length`. The one `error` belongs to "
        "`_0009` and was retried successfully.")
    add("")
    add("**Is it concentrated on large inputs?** Partly, but input size is not the cause.")
    add(f"- The two calls that failed outright are the two largest inputs: 20 passages each, about {c2['est_input_tokens_per_request_rung1']:,} "
        f"and {c3['est_input_tokens_per_request_rung1']:,} tokens per request.")
    add(f"- Their 35% rungs, about {c2['rungs'][2]['est_input_tokens_per_request']:,} and {c3['rungs'][2]['est_input_tokens_per_request']:,} "
        f"tokens, still failed. `_0004` succeeded at {calls['mc_001_extract_claims_0004']['est_input_tokens_per_request_rung1']:,} tokens "
        "(16 passages, 14 claims), and `_0005` at 8,705.")
    add(f"- `_0006` failed at about {c6['est_input_tokens_per_request_rung1']:,} tokens.")
    add("- The ladder shrinks the *input*, but the budget went on reasoning and required output. Rung 3's tight schema does not cap "
        "the reasoning either. Truncating the user string keeps its first passages, which are the dense ones.")
    add("")
    add(f"**Shrink ladder.** Three calls climbed it: `_0002` and `_0003` (all 3 rungs, all empty) and `_0006` (rung 2 succeeded). "
        f"**No call produced a claim after shrinking.** `_0006`'s rung-2 answer was `{{ \"claims\": [] }}` for a wheat transcriptome "
        "paper.")
    add("")
    p1 = ec["pilot001_same_paper_call"]
    add(f"**Pilot-001 comparison (n = 1, and the stack is non-deterministic at temperature 0).** Pilot-001 sent the **same 20 evidence "
        f"ids** of PMC13364333 (set-equal: {ec['pilot001_same_paper_same_passages']}) under prompt `{p1['prompt_version']}`. It got "
        f"`{p1['finish_reason']}` in one request: {p1['input_tokens']:,} in, {p1['output_tokens']:,} out, {p1['response_chars']:,} chars, "
        "21 claims. That run was already at 91% of the 4,000-token cap. Prompt 1.1.0 asks for a citation list per field, which adds "
        "output and bookkeeping for a 20-reaction review. Every one of 9 requests then ran out.")
    add("")
    add(f"**Cost of the failure.** `_0002` + `_0003`: ${ec['failed_cost_usd']} ({100 * ec['failed_cost_usd'] / h['estimated_cost_usd']:.0f}% "
        f"of the run's estimate) and {ec['failed_latency_s']} s of provider latency ({100 * ec['failed_latency_s'] / h['elapsed_seconds']:.0f}% "
        "of wall clock). `_0002` alone took 890 s. The two most pathway-dense papers of round 1 contributed nothing.")
    add("")

    # ------------------------------------------------------------ 6
    add("## 6. Search")
    add("")
    pr = s["papers_discovered_by_round_and_source"]
    add("| round | queries | new papers discovered (by source) | included | processed |")
    add("|---|---|---|---|---|")
    for n in ("1", "2", "3"):
        nq = sum(1 for x in q if str(x["round"]) == n)
        ninc = sum(1 for p in inc if str(next((y["round"] for y in q if p["paper_id"] in y["new_included"]), "")) == n)
        nproc = sum(1 for p in inc if str(p["processed_round"]) == n)
        add(f"| {n} | {nq} | {sum(pr[n].values())} {json.dumps(pr[n])} | {ninc} | {nproc} |")
    add("")
    add(f"`queries.jsonl` has {len(q)} rows and {len(q) - sum(v_ - 1 for v_ in s['duplicate_query_ids'].values())} distinct query ids "
        f"(A10). `cache_hit` is true on {s['cache_hit_true']} rows (A13).")
    add("")
    add("**Per frontier item** (attempt 1 = the item's own terms AND the species; attempt 2 = plus the topic; 'wider window' = attempt 1's "
        "text again over 2 x 25 when the topic is already a term):")
    add("")
    add("| round | kind | target | attempts | windows | results | new discovered | new included | outcome |")
    add("|---|---|---|---|---|---|---|---|---|")
    for it in s["items"]:
        add(f"| {it['round']} | {it['kind']} | {cell(it['target'])} | {', '.join(it['attempts'])} | {it['windows']} | {it['results']} | "
            f"{it['new_discovered']} | {cell(', '.join(it['new_included']) or '-')} | {cell(it['outcome'])} |")
    add("")
    add(f"Round-2 exhaustions: {json.dumps(ex2)}. Round 2 had {s['round2_frontier']} items and {s['round2_items_asked']} were asked. "
        f"The {len(s['round2_items_never_asked'])} never asked are: {cell('; '.join(f'{k}: {t}' for k, t in s['round2_items_never_asked']))}. "
        "Three of them are the enzyme items for AOS, AOC and OPR3.")
    add("")
    add("**Per query** (window = what the controller asked for: `min(25, remaining)`, or `min(50, remaining)` for a wider window; "
        "'filled' = results reached the window, so the true hit count is unknown):")
    add("")
    add("| # | round | kind | target | attempt | window | results | filled | new discovered (by source) | new included | cumulative | query text |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in q:
        att = f"{x['attempt']} (wider)" if x["deeper"] else str(x["attempt"])
        add(f"| {x['n']} | {x['round']} | {x['kind']} | {cell(x['target'][:60])} | {att} | {x['window']} | {x['result_count']} | "
            f"{'yes' if x['window_filled'] else 'no'} | {x['new_discovered']} {json.dumps(x['new_by_source']) if x['new_by_source'] else ''} | "
            f"{cell(', '.join(x['new_included']) or '-')} | {x['cumulative_discovered']} | {cell(x['query_text'])} |")
    add("")
    add("**What stopped the run.** The frontier did not run out: 19 items were open at the stop, 15 of them never asked. The binding "
        "limit was `max_papers_discovered`.")
    add("- Round 1's 3 seed queries filled their 25-result windows and brought in 66 papers.")
    add(f"- Round 2's first 15 queries took the count to 139. Query {q149['n']} (executed {q149['executed_at'][11:19]}, "
        f"{q149['cumulative_discovered']} papers) left one slot.")
    add(f"- Every later query ran with a window of 1 ({s['queries_with_window_1']} queries in 61 s). Each returned the single top hit, "
        "already seen, until query 58 found paper 150 at "
        f"{cap_q['executed_at'][11:19]}, {s['cap_reached_seconds_after_start']:.0f} s after the start.")
    add("- `check_papers_discovered` then recorded the stop. Round 3 started anyway, broke on its first item and closed as an "
        "unproductive round (A5).")
    add("")
    add("So yes, 150 is now the binding stop. In effect it was reached at query 18, about 1,974 s in. Most of the elapsed time before "
        "that was the two failed extraction calls, not search.")
    add("")
    add("**What the new search did right (item 3).** AND formulation: 2 of the 3 seed items gained a second formulation, and the "
        "topic seed a wider window. NCBI is reached: 83 of 150 papers, 8 of 11 included. Every query records its attempt. Round 2 "
        "discovered 84 new papers and included 6. Pilot-001 found 9 new and included 0. The bridge queries are unanswerable by "
        "construction here: they AND two synonyms of the same compound, so 2 of 3 returned zero (A9).")
    add("")

    # ------------------------------------------------------------ 7
    add("## 7. Side by side with pilot-001")
    add("")
    add("Both columns are computed by the same function from each run directory (`pilot/001/run/` for pilot-001).")
    add("")
    add("| | pilot-001 | pilot-002 |")
    add("|---|---|---|")
    for k in r["side_by_side"]["pilot-002"]:
        add(f"| {k.strip()} | {cell(r['side_by_side']['pilot-001'][k])} | {cell(r['side_by_side']['pilot-002'][k])} |")
    add(f"| empty rungs / 'giving up' log lines / provider requests | 0 / 0 / 1 | {ec['rungs_empty']} / {ec['log_giving_up_lines']} / {ec['provider_requests']} |")
    add(f"| distinct articles among discovered | 21 | {nc['unique_articles_discovered']} |")
    add("| NCBI papers discovered | 0 (never reached, F2) | 83 |")
    add("")

    # ------------------------------------------------------------ 8
    add("## 8. Anomalies")
    add("")
    add("### Pilot-001 findings F1-F10: do they still reproduce?")
    add("")
    add("| id | pilot-001 finding | pilot-002 |")
    add("|---|---|---|")
    for row in F_TABLE:
        add(f"| {row[0]} | {cell(row[1])} | {cell(row[2])} |")
    add("")
    add("### New anomalies")
    add("")
    add("| id | severity | finding | evidence | code / doc |")
    add("|---|---|---|---|---|")
    for row in anomalies(r):
        add("| " + " | ".join(cell(x) for x in row) + " |")
    add("")

    # ------------------------------------------------------------ appendix
    add("## Appendix A: cited evidence spans, verbatim")
    add("")
    add("Exact `text` from `evidence_spans.jsonl` for every span a reaction cites.")
    add("")
    seen = []
    for x in rx:
        for sp in x["spans"]:
            if sp["evidence_id"] in seen:
                continue
            seen.append(sp["evidence_id"])
            add(f"### `{sp['evidence_id']}`: `{x['paper_id']}`, {sp['section']} `{sp['locator']}`, chars {sp['char_start']}-{sp['char_end']}")
            add("")
            add("```text")
            add(r["span_texts"][sp["evidence_id"]])
            add("```")
            add("")
    add("## Appendix B: files")
    add("")
    add("| file | what |")
    add("|---|---|")
    add("| `build_audit.py` | computes every number and writes both outputs |")
    add("| `md_text.py` | this document's narrative; reads the computed result |")
    add("| `pilot_002_audit.json` | the machine-readable twin (all tables above, uncut) |")
    add("| `explore.py`, `peek.py`, `peek2.py`, `fix_pointers.py` | scratch readers and a one-off text patch used while writing; not needed to rebuild |")
    add("| `g11/*.json` | `bounded_run.py` reports for every script run; all show 0 survivors |")
    add("")
    add("Rebuild (from `C:/t/mining-pilot-runs2/audit`):")
    add("")
    add("```bash")
    add("PYTHONPATH=C:/t/mining-pilot2/src PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 \\")
    add("  <venv>/python.exe C:/t/mining-pilot2/docs/pwml_recovery_sprint/evidence/bounded_run.py --timeout 300 \\")
    add("  --label audit-build --json g11/build_audit.g11.json -- <venv>/python.exe build_audit.py")
    add("```")


F_TABLE = [
    ("F1", "every enzyme/gene/species dropped as uncited (59/59)",
     "**Resolved.** 0 drops; 53 fields kept, all `model_citation`; the raw lineage shows every filled field with its own list. The fallback was never needed."),
    ("F2", "round 2 found nothing new: OR'd query, top-10 cap with NCBI never reached, one-shot exhaustion",
     "**Mostly resolved, one new failure mode.** AND queries, 25-result windows, NCBI reached (83 papers), 2 attempts plus a wider window; round 2 found 84 new papers, 6 included. But the discovery budget cut windows to 1, so 21 items were exhausted on starved windows (A6). NCBI brought the identity defect (A1)."),
    ("F3", "species labels by fall-through / wrong reasons",
     "**Reproduces in a new form.** 9 pepper reactions `in_vitro_or_cell_free` from an incidental sentence, although for 8 of them the model cited *Capsicum annuum*; the primary's kinetic evidence passes the in-vitro test (A3)."),
    ("F4", "frontier cap silently cut 62 items",
     "**Not exercised.** The round-2 frontier was 44 and the round-3 frontier 19, both under the cap of 60. No cut."),
    ("F5", "screen missed AtLOX2 enzymology, 12-OPDA metabolites, DAD1",
     "**Partly resolved.** The rescue re-admitted 6 (route 1: 5, route 2: 1), and route 2 produced the only primary reaction. Still out: Arabidopsis AOC heteromerization `PMC:PMC4844422` (`off_topic` 0.25), the CYP20-3 / OPDA-flux paper (2.25, documented borderline) and OPR3 `PMID:31834385` (`off_topic` 1.5). One route-1 false rescue: grape (A8)."),
    ("F6", "review labelled `unknown`; PMID and DOI empty",
     "**Reproduces.** All 11 included are `unknown`, including the PMC13364333 review; every included record has `doi: \"\"`, and NCBI records carry a wrong PMCID and URL (A1)."),
    ("F7", "report.md misstates claims extracted and primary-graph connections/entities",
     "**Reproduces, plus two more.** 'Claims extracted: 11' (16); 'Primary graph: 1 reactions, 11 chemical connections, 20 entities' (the primary has 0 and 2); 'Model calls: 15' (10 calls, 30 requests); 'generated but never asked' lists 4 items that were asked (A12)."),
    ("F8", "cache_hit never set; screened-out papers missing from round provenance; retries semantics; relevance 0.0; abstract span with null offsets / HTML",
     "**Reproduces.** `cache_hit` false on 58/58; round metrics `papers_discovered` 5 and 6 (usage 66 and 84), with `papers_considered` and `new_paper_ids` included-only; `retries` is the max inner attempt count; all 3 candidate components have relevance 0.0; `ev_49c948bc587a2016` has null offsets; titles keep HTML entities (`&lt;i&gt;`)."),
    ("F9", "mining CLI does not load .env",
     "**Still present; not exercised.** The launcher was used again. `flags.mining_enabled` returns before `ensure_dotenv_loaded()` whenever the CLI flag supplies an override (`flags.py:55-62`), and nothing else in `t2pw.mining` loads .env."),
    ("F10", "cli.md shows the flag after the subcommand",
     "**Mostly fixed.** `cli.md:4, 11, 20` are right; the `preflight` example at `cli.md:109` still puts `--enable-pathway-mining` after the subcommand."),
]


def anomalies(r):
    ec = r["empty_completion_summary"]
    s = r["search"]
    return [
        ("A1", "high", "NCBI records take their PMCID and DOI from the last cited reference, not the article",
         f"Frozen parser on a synthetic record returns PMC411572, the reference's id. In this run: 6 articles discovered twice, and each NCBI copy's PMCID differs from EuropePMC's. Of the 8 included NCBI papers, 2 full-text fetches returned text, and both were another article, which extraction then read under the included title (the other 6 returned no text). The only primary reaction cites `PMC:PMC411572` / `10.1073/pnas.76.9.4350` (PNAS 1979) for a 2021 abstract. 150 discovered = 144 articles. The NAD+ article was excluded via EuropePMC and included via NCBI. Class: `product_contract_violation` (provenance). Shared production code, latent until NCBI was reached",
         "`src/t2pw/rag/acquire.py:536-543` (`findall('.//ArticleIdList/ArticleId')`); `mining/acquisition.py:528-` `Acquirer.acquire` (dedup by `paper_id` only)"),
        ("A2", "high", "Extraction of dense passages fails: reasoning exhausts max_tokens, and the ladder cannot fix it",
         f"2 calls, 18 requests, each exactly 4,000 output tokens with empty content; 35% rungs about 8.4k / 6.6k tokens failed while 15k succeeded elsewhere. ${ec['failed_cost_usd']} and {ec['failed_latency_s']} s for 0 claims. Same passages succeeded under prompt 1.0.0 in pilot-001. The inner client re-sends identical requests 3 times per rung: 30 provider requests against a budgeted 'llm_calls' of 15",
         "`mining/config.py:148, 158` (`max_tokens` 4000, `reasoning` \"\"); `modelclient.py:50, 271-280`; `llm/client.py:682-756`"),
        ("A3", "medium", "Species states from keyword tests contradict the cited evidence",
         "Pepper introduction -> `in_vitro_or_cell_free` for 9 reactions because of 'in vitro studies confirmed ...' about AtLOX2 phosphomutants, although for 8 of them the model cited *Capsicum annuum*, which would be `non_target_species`. The primary's abstract (Vmax/Km, 'in cascade with 12-OPDA pathway enzymes') is purified-enzyme work that no marker catches. Outcome safe; labels wrong; the primary's state is a policy question",
         "`mining/species.py:108-115, 276-281`; `species_policy.md` state table"),
        ("A4", "medium", "An abstract-only paper is indexed twice, so one passage counts as two spans",
         "`ev_b58e2cb64e2f1d87` (abstract#0) and `ev_ba6fb9f5cdb8ce58` (body#0) have identical text. The same happens for every abstract-only paper: 2 identical passages each for all 5 abstract-only papers processed (1 in round 1, 4 in round 2). The primary's reason says 'explicitly supported by 2 evidence span(s)'. The model is also shown the same text twice",
         "`mining/retrieval.py:170-183` (candidate built with `abstract` = the record abstract and `full_text` = the same abstract)"),
        ("A5", "medium", "A phantom round after the stop reason is recorded",
         "`max_papers_discovered_reached` was recorded inside round 2, right after query 58. `check_before_round` checks neither discovered papers nor an existing `stop_reason`, so round 3 started with 19 items, broke on the first and closed as a completed, unproductive round: `rounds_completed=3`, unproductive streak 2. Had discovery not been recorded first, this empty round would itself have stopped the run as `converged_no_useful_additions`, naming the wrong cause. The controller docstring says 'stop at the first limit reached'",
         "`mining/budget.py:193-235`; `mining/controller.py:319-331, 383`"),
        ("A6", "medium", "The discovery budget silently starves and exhausts frontier items",
         f"Windows are `min(25, remaining)`. After 149 papers, {s['queries_with_window_1']} queries asked for 1 result and 21 items were marked exhausted after two 1-result windows, the same look as a real search that found nothing. `limits_and_stopping.md` promises exhaustion only 'after every attempt is tried without a new paper', which is literally true",
         "`mining/controller.py:403-408, 443-446`"),
        ("A7", "medium", "Paper-scoped retrieval is a store-wide top-60 filtered to one paper",
         "`PMC:PMC5104190` got 0 passages: included, never processed (11 included / 10 processed). Passage counts per paper (2-20) depend on what else is in the store, not on `max_passages_per_paper=20`",
         "`mining/retrieval.py:227-238` (`store.query(top_k=max(k*3, k))` then filter by `paper_ids`)"),
        ("A8", "medium", "Route-1 false rescue: target species only an ectopic-expression host",
         "`PMC:PMC5104190` (grape PAT1): the abstract says 'ectopic expression of VaPAT1 enhances cold tolerance in Arabidopsis'; 'grape' is unknown to the title detectors; the foreign-host guard reads the title only. Inside the documented ceiling (`evidence_policy.md`, `_EXTRA_HOSTS`); no effect because it was never processed",
         "`mining/acquisition.py:62-70, 426-448, 487`"),
        ("A9", "medium", "Lexical entity split disconnects the primary reaction and makes bridges unanswerable",
         "'13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT)' and '(13S)-hydroperoxy-octadecatrienoic acid (13-HPOT)' are two entities, as are 'α-linolenic acid (LeA)' and 'α-linolenic acid'. The primary has 0 connections. The bridge query ANDs the two synonyms (0 results, exhausted at once), and two upstream items ask the identical question. Documented as a v1 limitation (`known_limitations.md`); the AND bridge formulation is what makes it a guaranteed 0",
         "`mining/normalize.py` `comparison_key`; `mining/queries.py:84-108` (bridge requires both sides)"),
        ("A10", "low", "Query ids collide across frontier items",
         f"`query_id = hash(text, round)`: {len(s['duplicate_query_ids'])} ids appear twice in `queries.jsonl`, with different items and different result counts (e.g. `qr_faf2adeaf3e7810c`: 25 results for one item, 1 for the other). `discovered_via_query_id` alone is ambiguous",
         "`mining/queries.py:184`; `mining/ids.py:82-83`"),
        ("A11", "low", "A nested trailing parenthetical is not stripped from a query term",
         "'(12S,13S)-epoxy-hydroperoxy-octadecatrienoic acid ((12S,13S)-EOT)' went verbatim into 4 queries. `limits_and_stopping.md`: 'A trailing parenthetical note ... is dropped from a term'",
         "`mining/queries.py:77` (`\\s+\\([^()]*\\)$`)"),
        ("A12", "medium", "report.md misstates numbers (F7 plus two)",
         "'Claims extracted: **11**' (16 extracted; 11 is the stored count). 'Primary graph: **1** reactions, **11** chemical connections, **20** entities' (whole graph; the primary has 0 connections and 2 entities). 'Model calls: **15**' (10 logical calls, 15 rungs, 30 requests). 'These questions were generated but never asked' lists the 3 seeds (asked 3 times each) and OPC-8 (asked once)",
         "`mining/report.py:149, 159-160, 275, 299, 417`"),
        ("A13", "low", "Observability gaps in the lineage and round files",
         "`model_raw.jsonl` carries no shrink fraction or chars sent; `model_calls.jsonl` `attempt_log` is overwritten per rung (3 of `_0002`'s 9 requests survive); `retries` is a max, not a count; `round.json` `input_frontier` is serialised after the round (round-1 seeds show `attempts: 1`); `cache_hit` false on 58/58; round metrics `papers_discovered` counts included papers only",
         "`mining/modelclient.py:291-300, 371-374`; `mining/controller.py:552-566`"),
        ("A14", "low", "Evidence type and identifiers still missing",
         "All 11 included papers `unknown`, including a review; `doi` empty on all 11 (only `url` carries it)",
         "`mining/acquisition.py` `classify_evidence_type` / `record_from_candidate`"),
        ("A15", "low", "Participants are uncited, despite prompt rule 1 and `evidence_policy.md`",
         "30 of 32 raw substrate/product entries have no `evidence_ids`; `build_claims` never drops a participant",
         "`mining/extraction.py:46-53, 237-259`; `evidence_policy.md:83`"),
        ("A16", "low", "Documentation drift",
         "`artifacts.md` gives `PROMPT_VERSION` as `mining-extract/1.0.0` (code: 1.1.0); `cli.md:109` flag order (F10); `ids.py` docstring and `resume_and_cancellation.md` say no id uses a counter, but `call_id` uses a per-process ordinal (a resume would reuse `mc_..._0001`; not exercised)",
         "`docs/pathway_mining/artifacts.md`; `mining/ids.py:1-6, 94-95`; `mining/modelclient.py:242-243`"),
        ("A17", "info", "One completion repeated its own claims, and the same step became two nodes",
         "`_0004` emitted 14 claims for 9 identities: 5 in-call duplicates, merged correctly. 'α-linolenic acid -> 13-HPOT' from the same sentence became two candidate nodes (`rx_b30980df6dd1f29b4504` with no species, `rx_c1d050d4171b110ad2d4` Capsicum) because species is part of identity",
         "`mining/extraction.py` identity; `mining/claims.py`"),
    ]

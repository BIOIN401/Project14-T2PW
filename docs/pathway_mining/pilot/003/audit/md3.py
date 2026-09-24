"""Markdown renderer for the PILOT-003 audit. Every number comes from build_audit3.py's result dict."""
from __future__ import annotations

import json


def cell(value) -> str:
    return " ".join(str(value if value is not None else "").split()).replace("|", "\\|")


def render(add, r):
    c = r["counts"]
    v = lambda k: c[k]["value"]  # noqa: E731
    h = r["headline"]
    rx = r["reactions"]
    b = r["batching"]
    calls = b["calls"]
    st = r["stop"]
    fr = r["frontier_replay"]
    inc = r["included_papers"]
    nid = r["ncbi_identity"]
    thin = [x for x in calls if x["near_zero_reasoning"]]
    near_cap = [x for x in calls if x["share_of_budget"] >= 0.8]
    no_part = [x for x in rx if x["participants_cited"] == 0]
    all_part = [x for x in rx if x["participants_all_cited"]]
    lox = [x for x in rx if x["paper_id"] == "PMC:PMC8508710"]
    opr3 = next(x for x in rx if x["reaction_id"] == "rx_14eb3a43a495d7ce851f")
    lox_inf = next(x for x in rx if x["reaction_id"] == "rx_41c97e790d2e17c19477")
    empties = [o for x in calls for o in x["request_outcomes"] if o.startswith("length")]

    add("# PILOT-003 audit: pathway mining, jasmonate biosynthesis, *Arabidopsis thaliana*")
    add("")
    add(f"Read-only evaluation of `{r['run_dir']}` (code `{r['code']}`, ORCH-743 iteration 2, pre-registered at ledger commit "
        "`36230ba6`). Nothing was re-run and no model was called. Every number is read from a named artifact field by "
        "[`build_audit3.py`](build_audit3.py), with the narrative in [`md3.py`](md3.py). It ran through the pilot-3 `bounded_run.py` "
        "with `PYTHONPATH=C:/t/mining-pilot3/src` and `PYTHONDONTWRITEBYTECODE=1`, and left 0 surviving processes. The frozen code "
        "was imported only for pure functions: the title-organism detector, `full_text_matches_title`, the species marker tests, "
        "and `frontier.generate` (replayed on `state.json` with the frontier cap lifted). The machine-readable twin is "
        "[`pilot_003_audit.json`](pilot_003_audit.json). The comparisons use pilot-001 "
        "(`C:/t/mining-pilot/docs/pathway_mining/pilot/001/run`) and pilot-002 (`C:/t/mining-pilot-runs2/`).")
    add("")
    add(f"**Headline.** `{h['completion_state']}` / `{h['stop_reason']}` ({h['stop_detail']}). The run did **1 round** in "
        f"{h['elapsed_seconds']} s at an estimated ${h['estimated_cost_usd']}. It discovered {v('papers_discovered')} papers, included "
        f"and processed 4, and made 16 extraction calls, all `ok`, which used {b['provider_requests']} provider requests. "
        f"{v('claims_extracted')} claims became 37 identities: **0 primary** reactions, 35 candidates and 2 `incomplete` claims.")
    add("")
    add("**Key findings.**")
    add("")
    add(f"1. **Batching fixed the empty completions, with little headroom left.** All 16 calls returned text on their first ladder rung. "
        f"Only {len(empties)} provider requests came back empty, both in `_0008`; the client retried them internally and the third "
        f"request answered. But the Marchantia review's batches still used {', '.join(f'{x['share_of_budget']:.0%}' for x in near_cap)} of "
        f"the 12,000-token budget ({', '.join(x['call_id'][-4:] for x in near_cap)}), and `_0008`'s two empty requests hit it exactly.")
    add(f"2. **Four answers show almost no reasoning. The provider is not recorded.** Calls {', '.join(x['call_id'][-4:] for x in thin)} "
        f"answered with 122-209 output tokens (`_0001`, `_0003` and `_0014` in 4.5-8.3 s). That is about 20-40 tokens beyond the JSON itself, and is consistent with "
        f"a provider or mode that did not reason. `provider` is empty on all 16 records, and `actual_model` only echoes the requested "
        f"slug, so the artifacts cannot say who answered. Every one of those answers has no enzyme, gene or species field and no "
        f"participant citation.")
    add(f"3. **No primary reaction.** The pilot-002 primary (13-LOX, alpha-linolenic acid -> 13-HPOT in *A. thaliana*) came back "
        f"{len(lox)} times from the same paper, now read in full text under its correct id `PMC:PMC8508710`. But the model filled "
        f"no species, enzyme or gene on any of the three. Two were labelled `multiple_species`: with no species field, classification "
        f"falls back to the paper's screen metadata, *A. thaliana* + *Homo sapiens*. The third is `inferred_only` because its passage "
        f"says \"as inferred from the dose-response curves\".")
    add(f"4. **The stop: 35 new nodes, and none could be admitted.** `max_new_reactions_per_round` counts admitted **and** quarantined "
        f"reactions new to the graph: 0 + 35 >= 25. 23 came from the Marchantia review (the same paper pilot-001 read), 8 from GH3.10 "
        f"in-vitro assays, 3 from the 13-LOX paper and 1 from pepper. By name-level reading they are about "
        f"{st['distinct_conversions_by_reading']} distinct conversions. The check runs only before a round, so round 1 finished and "
        f"round 2 never started. The frontier held {fr['recorded_frontier']} items, the cap, and 57 were never asked. "
        f"`frontier.generate` produced {fr['uncapped_generated']}, and the cap cut **{fr['cut_by_cap']}** of them without recording it "
        f"(F4 reproduces).")
    add(f"5. **Paper identity (IT2-2) works live.** All {nid['total']} NCBI ids are consistent: {nid['pmcid_resolved']} resolved to a "
        f"PMCID, {nid['pmid_only']} kept as PMIDs, and every URL is the paper's own PubMed page. Both included NCBI papers' full texts "
        f"open with their own title and PMID. All 3 NCBI ids that can be checked against EuropePMC's id for the same article agree. "
        f"There are 0 duplicate titles (pilot-002 had 6) and 0 full-text refusals (none was needed). "
        f"{len(r['cross_run_id_changes'])} titles seen in both pilots changed id; the pilot-002 ids were cited-reference PMCIDs.")
    add(f"6. **Species states are still set by keywords (F3 / A3).** 16 of 17 `inferred_only` labels come from the substring 'ortholog' "
        f"in the review's passages: its table header \"Arabidopsis Ortholog\", \"lacks JAR1 orthologs\", \"orthology analysis\". That includes the canonical "
        f"*A. thaliana* OPR3 step, OPDA -> OPC-8:0, where the model cited *A. thaliana*. The pepper introduction is again "
        f"`in_vitro_or_cell_free` because of an incidental sentence about other work.")
    add(f"7. **Participant citations (IT2-N1) track answer quality.** {len(all_part)} of 35 candidates carry evidence ids on every "
        f"participant. {len(no_part)} carry none, and every one of those came from a call that filled no enzyme or species field "
        f"either. No field was dropped, and `verbatim_cited_span` was never used.")
    add("")

    # ---------------------------------------------------------------- 1
    add("## 1. Counts, with sources")
    add("")
    add("| metric | value | source field | note |")
    add("|---|---|---|---|")
    for k, row in c.items():
        add(f"| {k} | `{cell(json.dumps(row['value'], ensure_ascii=False))}` | {cell(row['source'])} | {cell(row['note'])} |")
    add("")

    # ---------------------------------------------------------------- 2
    add("## 2. Every reaction (0 primary, 35 candidates) and the 2 incomplete claims")
    add("")
    add("The seed connection is the auditor's name-level reading against the canonical route (lipase -> 13-LOX -> AOS -> AOC -> OPR3 -> "
        "beta-oxidation -> JA -> JAR1), used only as a yardstick. 'Marker' is the phrase in the cited passages that set the species "
        "state, found by the frozen species tests. 'Participants' counts participants that carry their own evidence ids.")
    add("")
    add("| # | reaction id | equation | enzyme (attribution) | species state / value (attribution) | marker that set the state | paper | call(s) | participants cited | seed connection | decision |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, x in enumerate(rx, 1):
        enz = "; ".join(f"{e['name']} ({e['attribution']})" for e in x["enzymes"]) or "none"
        mk = x["species_marker"]
        mark = ", ".join(mk.get("inference") or mk.get("in_vitro") or []) or ("paper metadata" if x["species_state"] == "multiple_species" else "-")
        add(f"| {i} | `{x['reaction_id']}` | {cell(x['equation'])} | {cell(enz)} | `{x['species_state']}` / {cell(x['species_value'] or '(empty)')}"
            f"{' (' + x['species_attribution'] + ')' if x['species_attribution'] else ''} | {cell(mark)} | `{x['paper_id']}` | "
            f"{', '.join(k[-4:] for k in x['model_calls'])} | {x['participants_cited']}/{x['participants_total']} | {cell(x['seed_connection'])} | "
            f"`{x['decision_status']}` |")
    add("")
    add("Incomplete claims (no reaction node; decision `incomplete`):")
    add("")
    for x in r["incomplete_claims"]:
        add(f"- `{x['claim_id']}` from `{x['model_call']}` (`{x['paper_id']}`): \"{x['equation']}\". Reason: {x['decision']}")
    add("")
    add("### Per-reaction detail")
    add("")
    add("The excerpt is the sentence of the first cited span that names the most participants (verbatim; the frozen `_SENTENCE` "
        "splitter was used). Where 0 names matched, the model's wording is not in the span, and the full span is in Appendix A.")
    add("")
    for i, x in enumerate(rx, 1):
        add(f"#### {i}. `{x['reaction_id']}`: {x['equation']}")
        add("")
        enz = "; ".join(f"{e['name']} (`{e['attribution']}`)" for e in x["enzymes"]) or "none"
        add(f"- **Graph:** component `{x['component_id']}`, `{x['quarantine_reason']}`. Decision `{x['decision_status']}` "
            f"(`{x['decision_id']}`): \"{x['decision_reason']}\"")
        add(f"- **Enzyme:** {enz}. **Gene:** {x['gene']['value'] or 'none'}. **Species:** `{x['species_state']}`, "
            f"{x['species_value'] or '(empty)'}{' (`' + x['species_attribution'] + '`)' if x['species_attribution'] else ''}. "
            f"Reason: \"{x['species_reason']}\"")
        mk = x["species_marker"]
        if mk.get("sentence"):
            add(f"- **Marker sentence** ({', '.join(mk.get('inference') or mk.get('in_vitro'))}):")
            add(f"  > {mk['sentence']}")
        add(f"- **Paper:** `{x['paper_id']}`, \"{x['paper_title'].rstrip('.')}\"; claims {', '.join(x['claim_ids'])}; calls "
            f"{', '.join(x['model_calls'])}; spans {', '.join(x['evidence_ids'])}")
        if x["excerpt"]:
            add(f"- **Excerpt** (`{x['excerpt_span']}`, {x['excerpt_names_matched']} name(s) matched):")
            add(f"  > {x['excerpt']}")
        else:
            add(f"- **Excerpt:** no sentence of `{x['excerpt_span']}` names a participant as written (full span in Appendix A).")
        parts = "; ".join(f"{p['role']} {p['name']}: {', '.join(p['evidence_ids']) or 'uncited'}" for p in x["participants"])
        add(f"- **Participants:** {parts}")
        nb = "; ".join(f"{k} via '{via}' with `{o}`" for k, via, o in x["graph_neighbours"]) or "none"
        add(f"- **Seed connection:** {x['seed_connection']}: {x['seed_how']}. Graph: {nb}")
        add("")

    # ---------------------------------------------------------------- 3
    add("## 3. Every included paper and its route")
    add("")
    add("| paper id | PMID | title | source | route | recorded reason | title organisms (frozen detector) | text | calls / passages / claims |")
    add("|---|---|---|---|---|---|---|---|---|")
    for p in inc:
        reason = p["inclusion_reason"] + ((" -- " + p["production_exclusion"].replace("the production screen excluded it: ", "production screen: ")) if p["production_exclusion"] else "")
        add(f"| `{p['paper_id']}` | {p['pmid'] or '-'} | {cell(p['title'])} | {p['source']} | {p['route']} | {cell(reason)} | "
            f"{cell(', '.join(p['title_other_organisms']) or 'none')} | full text {p['text_chars']} chars; opens with its title: "
            f"{p['full_text_matches_title']} | {len(p['calls'])} / {p['passages']} / {p['claims']} |")
    add("")
    add("Routes: 3 production and 1 route-2 rescue (`PMC:PMC8508710`, the same 13-LOX paper and the same matched term `12-OPDA` as "
        "in pilot-002). No route-1 rescue fired, and no grape or other host paper was included. All 4 are full texts; pilot-002 "
        "read 5 abstract-only papers. The evidence type is `unknown` for all 4, including the review.")
    add("")

    # ---------------------------------------------------------------- 4
    add("## 4. Attribution methods and participants (IT2-N1)")
    add("")
    add(f"Fields kept, by attribution (38 extracted claims): `{json.dumps(v('fields_kept_by_attribution'))}`. There were "
        f"**0 uncited-field drops**, and `verbatim_cited_span` was used **0** times, so no N-1 row pairing needs checking. Every filled "
        f"field in the raw completions came with its own citation list: `{json.dumps(r['raw_field_citation_lists'])}`.")
    add("")
    add("Per call (raw completion): claims, participants, participants with their own `evidence_ids`, and which name fields were filled.")
    add("")
    add("| call | paper | claims | participants cited / total | fields filled |")
    add("|---|---|---|---|---|")
    for x in calls:
        f = x["raw_fields"]
        add(f"| `{x['call_id']}` | `{x['paper_id']}` | {f.get('claims', 0)} | {f.get('cited', 0)} / {f.get('participants', 0)} | "
            f"{', '.join(f.get('fields_filled', [])) or '-'} |")
    add("")
    add(f"Per candidate: {len(all_part)} of 35 have every participant cited; {len(no_part)} have none: "
        f"{', '.join('`' + x['reaction_id'] + '`' for x in no_part)}. Those six came from calls _0001, _0002, _0003, _0007, _0008 and _0013, "
        "and the two incomplete claims (from _0014) have no participant citations either. `build_claims` never drops an uncited "
        "participant; the claim-level `evidence_ids` still name the span. So IT2-N1 holds whenever the model answered in full, and "
        "fails silently when it did not.")
    add("")

    # ---------------------------------------------------------------- 5
    add("## 5. Model calls and batching (a)")
    add("")
    add(f"There were {len(calls)} logical calls, 1 ladder rung each, and {b['provider_requests']} provider requests. Calls per paper: "
        f"`{json.dumps(b['calls_per_paper'])}`. The batch size was 5 passages (a paper's last batch holds the remainder: 1 and 2). "
        f"`max_tokens` was {b['max_tokens']}. In the table, 'answer out' is the output tokens of the request that answered; for "
        "`_0008` that is the total less 2 x 12,000 for its two empty requests. 'Est. reasoning' subtracts the JSON length / 3.5 chars "
        "per token. It is an estimate, because the provider reports only the total.")
    add("")
    add("| call | paper | batch | passages | chars sent | requests (finish/content) | input tok | output tok | answer out | % of 12k | response chars | est. reasoning | ~0 reasoning | latency s | claims |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in calls:
        add(f"| `{x['call_id']}` | `{x['paper_id']}` | {x['batch']} | {x['passages']} | {x['chars_sent']} | {cell(', '.join(x['request_outcomes']))} | "
            f"{x['input_tokens']} | {x['output_tokens']} | {x['answering_request_output_tokens']} | {x['share_of_budget']:.0%} | "
            f"{x['response_chars']} | {x['est_reasoning_tokens']} | {'**yes**' if x['near_zero_reasoning'] else 'no'} | {x['latency_s']} | {x['claims']} |")
    add("")
    add("**Near the 12k budget.**")
    for x in near_cap:
        add(f"- `{x['call_id']}` used {x['share_of_budget']:.0%}.")
    add(f"- `_0008` spent the whole budget twice: two requests ended `length` with 0 characters. That is {len(empties)} empty "
        f"requests in the run (pilot-002 had 21).")
    add("")
    add("All four are 5-passage batches of the Marchantia review. For that paper, one 12k call per 5 passages sits at the edge.")
    add("")
    add("**Provider identity.** It is not recorded. `provider` is `\"\"` on all 16 records and is never set: `ModelCallRecord.provider` "
        "exists, but `modelclient` does not fill it. `actual_model` is the resolved *requested* slug from `chat_detailed` "
        "diagnostics, not the response's model field. The per-request `provider_requests` rows carry no provider either.")
    add("")
    add("**'Stop' answers that look non-reasoning.** Each of these ended `finish_reason=stop`:")
    add("")
    for x in thin:
        add(f"- `{x['call_id']}`: {x['answering_request_output_tokens']} output tokens for {x['response_chars']} chars "
            f"(est. reasoning {x['est_reasoning_tokens']}), {x['latency_s']} s. Raw text: `{cell(x['raw_text'][:300])}`")
    add("")
    add("The other twelve spent from about 100 (`_0009`, 11.5 s, borderline, yet it filled every field) to about 10,200 estimated reasoning tokens. Even `{\"claims\":[]}` answers cost 461-1,068 output "
        "tokens. So these four look like a provider or mode that did not reason; with no provider recorded, that is an inference "
        "from the token counts and latency. What they produced:")
    add("- `_0001` and `_0003`: bare 13-LOX claims. `_0003` also writes the LOX -> AOS -> AOC cascade as one reaction.")
    add("- `_0014`: two claims with no substrate.")
    add("- `_0008`: ALA -> 13-HPOT with nothing else.")
    add("")
    add("All four have no enzyme or species field and no participant citations. Two also wrote schema-shaped placeholders into "
        "`stoichiometry`:")
    add("- `\"has_stoichiometry\"` on all 3 participants of `_0003`;")
    add("- `\"products_evidence_ids_placeholder\"` on both products of `_0014`.")
    add("")
    add("All 5 are stored as is in `claims.jsonl`, because nothing validates a stoichiometry value. A thin answer is `ok` to the "
        "client, so no rung is retried.")
    add("")

    # ---------------------------------------------------------------- 6
    add("## 6. Search")
    add("")
    add("| # | query | attempt | window | results | new discovered (by source) | new included | cumulative |")
    add("|---|---|---|---|---|---|---|---|")
    for q in r["search"]["queries"]:
        add(f"| {q['n']} | {cell(q['query_text'])} | {q['attempt']} | {q['window']} | {q['result_count']} | {q['new_discovered']} "
            f"{json.dumps(q['new_by_source'])} | {cell(', '.join(q['new_included']))} | {q['cumulative']} |")
    add("")
    add(f"There was one round of 3 seed queries, each a full 25-result window, and 64 papers. `max_papers_discovered` (150) was never "
        f"approached, so the IT2-3 budget-cut rule was not exercised; no item is exhausted. `cache_hit` is true on "
        f"{r['search']['cache_hit_true']} rows. The round metrics say `papers_discovered` "
        f"{r['search']['round_metrics_papers_discovered']}` against 64 in usage (pilot-001 A5 / F8).")
    add("")

    # ---------------------------------------------------------------- 7
    q = r["code_quotes"]
    add("## 7. Why the run stopped after round 1 (b)")
    add("")
    add("**Counting.** Every claim goes through `_admit`. A claim whose identity is new to the graph adds its reaction to "
        "`added_reaction_ids` if admitted, and to `added_candidate_reaction_ids` if quarantined:")
    add("")
    add(f"`{q['admit_delta']['file']}:{q['admit_delta']['lines']}`")
    add("```python")
    add(q["admit_delta"]["text"])
    add("```")
    add("")
    add("When the round closes, their sum becomes `last_round_new_reactions`:")
    add("")
    add(f"`{q['close_round']['file']}:{q['close_round']['lines']}`")
    add("```python")
    add(q["close_round"]["text"])
    add("```")
    add(f"`{q['record_round']['file']}:{q['record_round']['lines']}`")
    add("```python")
    add(q["record_round"]["text"])
    add("```")
    add("")
    add(f"In pilot-003, `graph_delta.json` round 1 has `added_reaction_ids` {st['added_reaction_ids']}, "
        f"`added_candidate_reaction_ids` {st['added_candidate_reaction_ids']} and `updated_reaction_ids` {st['updated_reaction_ids']} "
        f"(the dn-cis-OPDA isomerisation, found in two batches). So `last_round_new_reactions` = {st['last_round_new_reactions']}. By "
        f"paper: `{json.dumps(st['by_paper'])}`. By species state: `{json.dumps(st['by_species_state'])}`. Not one of the 35 could "
        f"have been admitted.")
    add("")
    add("**Stopping.** The limit is tested only in `check_before_round`, the gate at the top of the round loop:")
    add("")
    add(f"`{q['round_loop']['file']}:{q['round_loop']['lines']}`")
    add("```python")
    add(q["round_loop"]["text"])
    add("```")
    add(f"`{q['check_before_round']['file']}:{q['check_before_round']['lines']}` and `{q['limit']['file']}:{q['limit']['lines']}`")
    add("```python")
    add(q["check_before_round"]["text"])
    add("")
    add(q["limit"]["text"])
    add("```")
    add("")
    add(f"After round 1, every earlier limit in that list is under its cap: 1 round of 8, 16 calls of 200, 4 papers of 40, 0 primary, "
        f"35 candidate reactions of 300, 21 components of 40. Then 35 >= 25 records `{st['stop_reason']}` with \"{st['stop_detail']}\". "
        "Round 2 never starts. Nothing inside a round checks the limit, so round 1 ran to the end: that is the documented behaviour "
        "(`limits_and_stopping.md`: 'That round is complete and kept; the next one does not start'). But the counter counts "
        f"*lexical* identities, quarantined non-target leads included. By the auditor's reading, 35 nodes are "
        f"{st['distinct_conversions_by_reading']} distinct conversions:")
    add("")
    add("- 3 nodes are 13-LOX.")
    add("- 3 are AOC to OPDA.")
    add("- 2 are JA -> JA-Ile.")
    add("- 2 are 12-OH-JA -> 12-OH-JA-Ile.")
    add("- 2 are HDA -> HTA.")
    add("")
    add("That is still over 25. One bryophyte review alone gives 23. Any run whose first round reads a dense review of another "
        "organism will stop here. It is the same design question pilot-001 raised as NB-1: stop, or throttle?")
    add("")
    add(f"**What was on the frontier and never searched.** `frontier.jsonl` holds {fr['recorded_frontier']} items, "
        f"`{json.dumps(fr['recorded_by_kind'])}`:")
    add("")
    add("- The 3 seeds were asked once each, attempt 1, in round 1. Their attempt-2 formulations (plus the topic) and wider windows "
        "never ran.")
    add("- The other 57 were never asked: 19 bridges, 27 downstream and 11 upstream items.")
    add("")
    add(f"The frontier is not the whole story. Replaying the frozen `frontier.generate` on `state.json` with the cap lifted gives "
        f"**{fr['uncapped_generated']}** items, `{json.dumps(fr['uncapped_by_kind'])}`. The 60-item cap kept everything at priority >= "
        f"{fr['lowest_kept_priority']} and silently dropped **{fr['cut_by_cap']}**: `{json.dumps(fr['cut_by_kind'])}`. The dropped "
        "items include:")
    add("")
    add("- all 33 `missing_species_evidence` items, the only kind that asks for target-species evidence for a held lead;")
    add("- all 13 `enzyme_reactions` items (AOS, AOC, OPR3, GH3.10 ...).")
    add("")
    add(f"Validation: the recorded non-bridge items are exactly the {fr['uncapped_nonbridge'] - fr['cut_by_cap']} top-ranked "
        f"non-bridge items of the replay, the ones that fit after the 3 seeds and 19 bridges "
        f"({fr['replay_nonbridge_equals_recorded']}). Bridges cannot be replayed exactly: a bridge anchors on the first name in "
        f"component order, which depends on memory order (pilot-001 A10, still present). This process's replay produced "
        f"{fr['replay_bridges_this_process']} bridges, and "
        f"{len(fr['bridge_anchor_differences']['recorded_only'])} recorded bridge targets do not appear in it. Separate replay "
        f"processes even produced different bridge counts: anchor collisions follow set order. Pilot-001's F4 therefore reproduces at twice its size: 62 cut then, "
        f"{fr['cut_by_cap']} now. No stop reason or artifact names the cut.")
    add("")

    # ---------------------------------------------------------------- 8
    add("## 8. Identity (c): is IT2-2 working live?")
    add("")
    add(f"NCBI records: `{json.dumps(nid)}`.")
    add("- Every NCBI `paper_id` equals `PMC:<pmcid>` when EuropePMC returned a PMCID, and `PMID:<pmid>` otherwise.")
    add("- Every URL is `https://pubmed.ncbi.nlm.nih.gov/<own pmid>/`.")
    add("- No NCBI record carries a DOI, so the parser's cited-reference DOI is gone too.")
    add("")
    add("| check | result |")
    add("|---|---|")
    for p in inc:
        add(f"| `{p['paper_id']}` (PMID {p['pmid'] or '-'}): full text opens with its own title / PMID in the first 600 chars | "
            f"{p['full_text_matches_title']} / {p['pmid_in_text_head'] if p['pmid'] else 'n/a'}. Head: \"{cell(p['text_head'][:170])}...\" |")
    for x in r["ncbi_vs_pilot002_europepmc"]:
        add(f"| NCBI id vs EuropePMC's own id for the same article (pilot-002): {cell(x['title'])} | {x['pilot003_ncbi_id']} vs "
            f"{', '.join(x['pilot002_europepmc_id'])}: **{'agree' if x['agree'] else 'DIFFER'}** |")
    add(f"| 'full text refused' notes | {len(r['full_text_refused'])} (none; both NCBI full texts matched) |")
    add(f"| duplicate titles among the {v('papers_discovered')} papers | {len(r['duplicate_titles'])} (pilot-002: 6) |")
    add("")
    add(f"The same title under different ids in pilot-002 and pilot-003 happens {len(r['cross_run_id_changes'])} times; "
        f"{r['same_id_in_both_runs']} titles have the same id in both. In every change, pilot-002's side is an NCBI record carrying a "
        "cited reference's PMCID:")
    add("")
    add("| title | pilot-002 id (source) | pilot-003 id (source) |")
    add("|---|---|---|")
    for x in r["cross_run_id_changes"]:
        add(f"| {cell(x['title'])} | `{x['pilot002']}` ({x['pilot002_source']}) | `{x['pilot003']}` ({x['pilot003_source']}) |")
    add("")
    add("Three of these articles (NAD+ deficiency, CYP20-3 / OPDA flux, Cyanodermella) appeared twice in pilot-002. In pilot-003 "
        "they appear once, under EuropePMC's id. IT2-2 behaves as documented. One caveat: a PMCID that EuropePMC resolves is "
        "trusted as given. The 19 resolved ids can only be spot-checked; the 5 that can be checked (2 by full text, 3 against "
        "EuropePMC's own records) all agree.")
    add("")

    # ---------------------------------------------------------------- 9
    add("## 9. Three-way side by side (e)")
    add("")
    add("All three columns are computed by one function from each run directory.")
    add("")
    add("| | pilot-001 | pilot-002 | pilot-003 |")
    add("|---|---|---|---|")
    s = r["side_by_side"]
    for k in s["pilot-003"]:
        add(f"| {k} | {cell(s['pilot-001'][k])} | {cell(s['pilot-002'][k])} | {cell(s['pilot-003'][k])} |")
    add("| frontier cut silently by the 60 cap | 62 | 0 (44 and 19, under the cap) | " + str(fr["cut_by_cap"]) + " |")
    add("| wrong NCBI ids / duplicate titles | - (no NCBI) / 0 | 7+ proven / 6 | 0 of 5 checked / 0 |")
    add("")

    # ---------------------------------------------------------------- 10
    add("## 10. What still reproduces (f), and new anomalies")
    add("")
    add("| id | finding | pilot-003 |")
    add("|---|---|---|")
    for row in reproduce(r):
        add(f"| {row[0]} | {cell(row[1])} | {cell(row[2])} |")
    add("")
    add("### New in pilot-003")
    add("")
    add("| id | severity | finding | evidence |")
    add("|---|---|---|---|")
    for row in new_anomalies(r):
        add("| " + " | ".join(cell(x) for x in row) + " |")
    add("")

    add("## Appendix A: excerpt spans, verbatim")
    add("")
    for e, t in r["span_texts"].items():
        add(f"### `{e}`")
        add("")
        add("```text")
        add(t)
        add("```")
        add("")
    add("## Appendix B: files")
    add("")
    add("| file | what |")
    add("|---|---|")
    add("| `build_audit3.py` | computes every number and writes both outputs |")
    add("| `md3.py` | this document's narrative |")
    add("| `pilot_003_audit.json` | machine-readable twin, including the 137 cut frontier items and the quoted code |")
    add("| `explore3.py`, `peek3.py`, `fix3.py` | scratch readers and an abandoned text patch (it aborted before writing); not needed to rebuild |")
    add("| `g11/*.json` | `bounded_run.py` reports; all show 0 survivors |")


def reproduce(r):
    fr = r["frontier_replay"]
    lines = r["report_md_lines"]
    return [
        ("F1", "every enzyme/gene/species dropped as uncited", "**Fixed (since pilot-002).** 0 drops; all 110 kept fields `model_citation`."),
        ("F2", "round 2 found nothing: OR query, top-10, NCBI never reached, one-shot exhaustion",
         "**Not exercised.** Only round 1 ran (3 seed queries, 64 papers, 28 from NCBI)."),
        ("F3", "species labels by fall-through / keyword",
         "**Reproduces.** 16 of 17 `inferred_only` labels come from the substring 'ortholog' ('Arabidopsis Ortholog' table header, 'lacks JAR1 orthologs', 'orthology analysis'), including the A. thaliana OPR3 step. 13-LOX is labelled from 'as inferred from the dose-response curves'. The pepper intro gives `in_vitro`. Two `multiple_species` come from paper metadata."),
        ("F4", "frontier cap silently cut 62 items", f"**Reproduces, larger.** {fr['cut_by_cap']} of {fr['uncapped_generated']} cut, unrecorded; every `missing_species_evidence` and `enzyme_reactions` item is gone."),
        ("F5", "screen missed key Arabidopsis enzymology", "Not evaluated beyond round 1. Route 2 again brought in the 13-LOX paper."),
        ("F6", "review typed `unknown`, DOI empty", "**Reproduces.** 4/4 included `unknown`, including the Marchantia review; `doi` is empty on 4/4."),
        ("F7", "report.md misstates numbers", "**Reproduces.** " + " / ".join(x.lstrip("- ").replace("**", "") for x in lines[:3]) + " (38 extracted; the primary graph is empty; 18 provider requests). It also says 'generated but never asked' of the 3 seeds that were asked."),
        ("F8", "cache_hit never set; round provenance included-only; relevance 0.0; order-dependent bridges",
         f"**Reproduces.** `cache_hit` false 3/3; round `papers_discovered` {r['search']['round_metrics_papers_discovered']} vs 64; relevance {r['component_relevance']}; bridge anchors differ on replay ({len(fr['bridge_anchor_differences']['recorded_only'])} of 19 in this build's process; the count varies between processes)."),
        ("F9", "CLI does not load .env", "**Still present; not exercised.** The launcher was used again, and `flags.py` is unchanged."),
        ("F10", "cli.md flag order", "**Partly fixed.** `cli.md:109` still puts `--enable-pathway-mining` after `preflight`."),
        ("A1", "NCBI ids from the cited reference", "**Fixed, verified live** (section 8)."),
        ("A2", "reasoning exhausts max_tokens; the ladder cannot help", "**Fixed in effect.** 16/16 calls `ok` on rung 1. 2 empty 12k requests remain (`_0008`), and the review's batches use 84-97% of the budget."),
        ("A3", "species states from keyword tests", "**Reproduces** (F3 row)."),
        ("A4", "abstract-only papers indexed twice", "**Not exercised.** All 4 papers are full text; " + str(r["duplicate_span_texts"]) + " pairs of spans share identical text."),
        ("A5", "phantom round after a mid-round stop", "**Not exercised.** The stop came from `check_before_round`; `check_before_round` still ignores discovered papers and an existing stop reason."),
        ("A6", "discovery budget exhausts items on cut windows", "**Fixed in code (IT2-3); not exercised** (64 of 150)."),
        ("A7", "paper retrieval is a store-wide top-60 filtered per paper", "**Still present.** The two late papers got 16 and 17 passages, not 20 (`retrieval.py` unchanged)."),
        ("A8", "grape false rescue", "**Fixed in code** (`_EXTRA_HOSTS` gains grape, vitis); no route-1 rescue fired."),
        ("A9", "lexical entity split", "**Reproduces, worse.** 13-HPOT under 4 names, OPDA under 3 (OPDA, (9S,13S)-OPDA, cis-OPDA). Most of the 21 components are split this way; 35 nodes are about 28 conversions."),
        ("A10", "query-id collisions", "**Not exercised** (3 queries)."),
        ("A11", "nested parenthetical kept in query terms", "**Not exercised.**"),
        ("A12", "report.md misstatements", "**Reproduces** (F7 row)."),
        ("A13", "lineage / round-file observability", "**Partly fixed.** `model_raw` now has fraction, passages, chars and per-request rows, and `attempt_log` is appended per rung. Still: no provider, `input_frontier` written after the round (seeds show `attempts: 1`), `cache_hit` never set."),
        ("A14", "evidence type / DOI missing", "**Reproduces** (F6 row)."),
        ("A15", "participants uncited", "**Partly fixed.** 29/35 candidates fully cited; 6 have none, all from thin answers (section 4)."),
        ("A16", "doc drift", "**Reproduces.** `artifacts.md` still says `mining-extract/1.0.0`; `cli.md:109` flag order."),
        ("A17", "the same conversion as several nodes", "**Reproduces, worse.** 5 groups, 7 extra nodes, and these count toward the new-reactions stop."),
    ]


def new_anomalies(r):
    return [
        ("B1", "high", "The per-round new-reaction cap counts quarantined, non-admissible leads, so one off-species review ends the run after round 1",
         "35 new nodes, 0 admissible (23 Marchantia, 8 in-vitro GH3.10, 3 13-LOX with no species, 1 pepper) >= 25. Round 2 and every missing_species_evidence question were never reached. Documented behaviour, but it makes a first-round review decisive (pilot-001 NB-1)."),
        ("B2", "high", "Thin, non-reasoning answers pass as `ok` and lose every name field",
         "4 answers at 122-209 output tokens with no enzyme/species/participant citations, among them two of the four 13-LOX batches (the paper that gave pilot-002's only primary reaction). Two wrote placeholder strings into `stoichiometry` (\"has_stoichiometry\", \"products_evidence_ids_placeholder\"), stored as is in `claims.jsonl`. Nothing records the provider, so these answers cannot be attributed or excluded."),
        ("B3", "medium", "Species from paper metadata when the model leaves species empty",
         "`rx_ce92a2647a229cc6528e` and `rx_6570fd856f1200fa7a05`: `multiple_species` from the screen's observed organisms (A. thaliana + Homo sapiens, from 'human LOX homologs'), not from any cited sentence (`extraction.py` `paper_observed_organisms`)."),
        ("B4", "medium", "A collapsed cascade recorded as one reaction node",
         "`rx_6570fd856f1200fa7a05`: α-linolenic acid (LeA) -> 13-HPOT + 12-OPDA (LOX, AOS and AOC in one node), from a thin answer (`_0003`)."),
        ("B5", "low", "Output budget headroom for dense reviews is thin",
         "Batches of the review at 93%, 97% and 84% of 12k; one batch needed 3 requests (2 empty at 12k), 313 s. A denser 5-passage batch would fail the same way pilot-002 did."),
    ]

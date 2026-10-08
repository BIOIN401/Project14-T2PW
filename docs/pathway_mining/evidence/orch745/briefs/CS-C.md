# ORCH-745 change set C — run control (B1/F-7, A5, F-3, B2, F9)

Worktree `C:/t/m745-c`, branch `orch745/c-run-control` (from `f9556288`), scratch
`C:/t/bt/m745-c/`, your name `m745-c`. Follow `C:/t/bt/orch745/COMMON_RULES.md` exactly.

Read first: `docs/pathway_mining/limits_and_stopping.md`, `architecture.md`, `artifacts.md`;
`src/t2pw/mining/budget.py`, `controller.py` (the run loop ~:300-340 and `_run_round`, esp.
~:540-575 and `_admit`), `frontier.py`, `modelclient.py`, `cli.py`, `__main__.py`,
`artifacts.py` (`write_round`), `models.py` (`ModelCallRecord`, `FrontierItem`, `GraphDelta`),
`tests/helpers_mining.py` (offline fixtures: `ScriptedModel`, `ScriptedPaperSource`,
`build_controller`, `run_fixture`). Live records for fixtures: pilot-003 on
`experiment/pathway-mining-pilot-003` under `docs/pathway_mining/pilot/003/` (read with `git show`):
`run/rounds/round_001/model_raw.jsonl`, `run/model_calls.jsonl`, `run/state.json`,
`audit/PILOT-003-AUDIT.md` sections 5 and 7.

## B1 / F-7 — the per-round cap stops the run on junk  (decision already made; implement it)

`controller.py` calls `ledger.record_round(new_reactions=len(delta.added_reaction_ids) +
len(delta.added_candidate_reaction_ids))`, and `check_before_round` stops the run when that is
`>= max_new_reactions_per_round` (25). In pilot-003 one off-species review produced 23 of 35
quarantined, never-admissible candidates and ended the run after round 1 with 0 admitted.
**Decision: `max_new_reactions_per_round` counts ONLY reactions newly admitted to the target graph
(`delta.added_reaction_ids`: decisions `accepted`/`conflicting`). Quarantined candidates and leads
do not count. It stays a stop condition.** The discovery, processed, call, cost and wall-clock caps
still bound the run. Update `limits_and_stopping.md` (and any report/summary wording that says
otherwise). G9: an offline run whose round 1 adds ≥ 25 quarantined candidates and 0 admitted does
NOT stop with `max_new_reactions_per_round_reached` (at base it does); a round adding ≥ cap
admitted reactions still stops it.

## A5 — phantom round

`check_before_round` checks neither the papers-discovered budget nor a stop reason already
recorded mid-round, so pilot-002 started a round 3 that could not search. Make it return the
existing stop reason if one is set, and stop when `papers_discovered >= max_papers_discovered`.
G9: an offline run that exhausts the discovery budget in round N never starts round N+1
(`usage.rounds`, round directories, stop reason `max_papers_discovered_reached`).

## F-3 — the frontier cap starves the species search  (decision already made; implement it)

`frontier.generate` sorts by priority and truncates to `budgets.max_frontier_size` (60), and
`merge_frontier` truncates again. In pilot-003 it produced 194 items and silently cut 137,
including EVERY `missing_species_evidence` item (FRONTIER_SPECIES), so target-species evidence was
never searched for. **Decision: reserve at least 25 % of the frontier slots (ceil(0.25 × cap)) for
`missing_species_evidence` items whenever that many exist, ordered by the candidate's on-route
relevance; fill the rest by the existing priority order. Record every item the cap cuts — count,
kinds, and the items (id, kind, target) — in the round artifacts.** Apply it in both places the
cap is applied (`generate` and `merge_frontier`), with one shared helper. Define "on-route
relevance" simply and deterministically and document it in `limits_and_stopping.md` /
`architecture.md`; a reasonable version: higher when the lead names the run's target species, when
a participant / enzyme matches a seed (config `starting_inputs` metabolites/enzymes) or a
participant of a target-supported reaction, and when its component touches the primary component;
stable tie-break. Surface the cut in the round record (e.g. `frontier_cut` → its own
`rounds/round_NNN/frontier_cut.jsonl` via `artifacts.write_round`, plus `metrics.frontier_cut` with
count and per-kind counts) and keep `artifacts.md` in step. G9: `frontier.generate` over a state
with > 60 generated items where the species items sort last keeps 0 species items at base and
≥ 15 at your tip (cap 60), and the cut list is recorded.

## B2 — thin, reasoning-free answers accepted as `ok`

Calls `_0001`, `_0003`, `_0014` of pilot-003 answered with 122–209 output tokens in 4–8 s:
alphabetical keys, no enzyme/gene/species field, no participant `evidence_ids`, and placeholder
values such as `"stoichiometry": "has_stoichiometry"` and
`"stoichiometry": "products_evidence_ids_placeholder"` (see `model_raw.jsonl` `raw_text`). This is
most likely an OpenRouter provider route that does no reasoning; `llm/client.py` (frozen, do NOT
edit) neither pins the provider nor reports it. Mitigate inside mining:
1. In `modelclient.py`, an answer-quality check applied after a payload parses and validates. For
   the claims schema: thin = at least one claim AND no participant in any claim carries
   `evidence_ids` AND no claim carries any `*_evidence_ids` field list — OR any value is a schema
   placeholder echo (`has_*`, `*_placeholder`, `...`, an `ev_...` literal). An empty `{"claims": []}`
   is a legitimate answer and is NOT thin. Keep the predicate generic enough to live in
   `modelclient.py` (or define it next to the schema and pass it in — `extraction.extract_claims`
   may pass it as a new keyword argument; that one call site is the only `extraction.py` line you
   may change, because another change set is editing `build_claims` in parallel).
2. A thin answer is retried ONCE (a fresh request; on this stack a re-send can route to a different
   provider). The retry is a counted call (ledger). If the retry is not thin, use it. If it is thin
   again, keep it but mark the record (e.g. `answer_quality: "thin_after_retry"`) so it is visible in
   `model_calls.jsonl` and the lineage; never upgrade a thin answer silently.
3. Record what the client actually returns about the provider and usage for every request:
   `ModelCallRecord.provider` is empty on every record today. Fill it from whatever
   `diagnostics`/`attempt_log` carry; if the client reports nothing, record an explicit
   `"not_reported_by_llm_client"` instead of `""`, and keep per-request output tokens so thin answers
   are measurable. Document the production follow-up (`llm/client.py`: provider routing unpinned,
   reasoning setting never sent, `response_hash` empty) in `known_limitations.md`.
G9: a `MiningModelClient` with a fake `chat_fn` that returns pilot-003's `_0003` raw text first and a
properly cited answer second: at base the call returns the thin payload after 1 request; at your
tip it returns the cited payload after 2, with the thin attempt recorded.

## F9 — the mining CLI never loads `.env`

`python -m t2pw.mining ... preflight|run` reads `os.getenv` before anything has loaded the project
`.env`, so live preflight says `LLM_PROVIDER` is unset; every pilot used an outside launcher. Call
`t2pw.config.ensure_dotenv_loaded()` in the CLI before preflight/run/resume (it never overrides
variables already in the environment). Also fix `docs/pathway_mining/cli.md`: the
`--enable-pathway-mining` flag goes BEFORE the subcommand (`cli.md:109` shows it after `run`), and
`artifacts.md` still says prompt 1.0.0. G9: with `t2pw.config.ensure_dotenv_loaded` replaced by a
spy (monkeypatch), running the CLI's preflight path calls it before the preflight reads the
environment (at base it is never called).

## Out of scope for you
Do not touch `species.py`, `extraction.py` (except the one call-site keyword above), `claims.py`,
`admission.py`, `acquisition.py`, `retrieval.py` — other change sets own them in parallel.

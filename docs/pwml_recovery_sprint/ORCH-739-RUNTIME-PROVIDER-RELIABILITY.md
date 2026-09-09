# `ORCH-739` — runtime / provider delivery reliability. The final check before `RAG v2`.

**2026-09-09.** Read-only census of the archived legs, plus **58 live micro-probe calls** that send
their own short prompts and no paper text. **No production code was modified**: `git diff 24dd4342
HEAD -- src/` is empty. `main` untouched. Protected `streamlit_app.py` unchanged at
`sha256:47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632`. No paper was re-run, no
cohort launched, no worktree pruned, no cache committed.

Tools (both new, both read-only against the repository):
[`evidence/orch739_delivery_census.py`](evidence/orch739_delivery_census.py) ·
[`evidence/orch739_budget_model_probe.py`](evidence/orch739_budget_model_probe.py).
Reports `evidence/g11/ORCH-739/01`–`06`, every one `FINAL SURVIVING COUNT : 0` /
`cleanup : success`; **6 artifacts, 0 non-compliant** under `g11_evidence.py check --task ORCH-739`.
Tool payloads are under `evidence/orch739/` — see that directory's
[`README`](evidence/g11/ORCH-739/README.md) for why they are filed apart from the cleanup reports.

---

# EXECUTIVE SUMMARY

> ## The answer to the charter's primary question is YES — but not where anyone was looking.
>
> Provider delivery is **not** failing at Stage 1. `ORCH-733` was right about that and nothing here
> overturns it. Delivery is failing on the **small auxiliary calls**, and it consumes
> **44.9 % of all leg wall-clock time** in the recent cohorts.
>
> ## The root cause is now measured rather than hypothesised, and it is one parameter.
>
> `deepseek/deepseek-v4-flash` spends its **entire completion budget on reasoning tokens** and
> returns zero content. Reproduced **5 of 5** at `max_tokens=300`. With
> `reasoning: {enabled: false}` as the only change: **0 of 5**, three times faster and a third the
> tokens.
>
> ## Configuration cannot reach it. That is why this ends in a card and not an `.env` line.

**The unit that matters is the leg, and at the leg level the split is stark.**

| | legs that produced a PWML | legs that produced none |
|---|---:|---:|
| legs | 13 | 14 |
| median empty completions | **11** | **26** |
| median share of runtime lost to non-productive calls | **29.3 %** | **50.2 %** |
| worst | 58 | **90** |

**Both wall-clock timeouts in the recent corpus are delivery-dominated**: `PMC9200736` burned 90
empty completions and 57 % of its hour; `PMC11961743` burned 71 and 46 %, having already been at
58 % the day before. Neither is a Stage-1 failure — both got through extraction.

**This is stated as `contributed materially`, not `caused`.** A leg that loses half its budget to
calls returning nothing is closer to the wall than one that does not, and both timeouts sit at the
top of the waste table. That is as far as the evidence goes and no further.

---

# 1. The census

`runs_smoke/2026-09-07_2323` (`ORCH-732`) · `runs_smoke/2026-09-08_1528` (`ORCH-734`) ·
`runs_validation/2026-09-07_1929` (`ORCH-730`) · `runs_validation/c120/2026-09-08_1240` ·
`runs_validation/c121/2026-09-09_0028`. **27 legs, 1,384 traced model calls, 13.63 h of leg wall
clock.** Discovery is `rglob`-based with the nested `c120`/`c121` families included — the
fixed-depth trap `ORCH-733` § 8 registered.

| class | calls | share |
|---|---:|---:|
| **normal** (usable completion) | 351 | 25.4 % |
| **empty** (zero content) | **716** | **51.7 %** |
| **degenerate** (1–200 chars) | 267 | 19.3 % |
| **truncated** (>200 chars, `finish_reason=length`) | 50 | 3.6 % |
| **provider/API failure** (timeout, HTTP, rate-limit) | **0** | 0 % |
| **malformed** | not decidable from the trace — see below | |

**There is not one provider exception in the whole recent corpus.** Nothing timed out at the HTTP
layer, nothing was rate-limited, nothing returned a bad status. The provider answers `200` every
time and sends nothing. Any framing of this as "the API is flaky" is wrong.

**`malformed` is deliberately not claimed.** `LEG_TRACE.jsonl` records what crossed the provider
boundary, not what the parser later made of it. Downstream JSON-validity outcomes are
`ORCH-733` § 1's measurement and are not re-derived here.

## The measurement conditions, so the numbers are falsifiable

* **object** — `model_attempt` rows, which are **client-level calls**, not ladder attempts. One
  ladder attempt can be three rows. `attempts_issued: 1` and three traced calls are consistent.
* **duration** — the gap to the previous traced event. An **upper bound**: non-LLM work between two
  events is charged to the later call. For an unbroken run of retries carrying one `request_hash`
  the bound is tight, which is the case that dominates the waste figure.
* **leg runtime** — `wall time` from `RESULT.txt`, falling back to the last traced elapsed.

---

# 2. Runtime reliability table

`waste` is the upper-bound wall clock attributed to empty and degenerate completions.
`chain` is the longest unbroken run of them.

| cohort | paper | runtime s | calls | normal | empty | degen | trunc | waste s | waste % | chain | Stage 1 | status | PWML |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| ORCH-732 | `PMC10031235` | 3,242 | 34 | 10 | 14 | 8 | 2 | 1,463 | 45 % | 7 | 2/2 | fail | — |
| ORCH-732 | `PMC11487621` | 2,708 | 97 | 19 | 54 | 19 | 5 | 1,465 | 54 % | 23 | 2/3 | fail | — |
| ORCH-732 | `PMC12051542` | 2,167 | 39 | 14 | 20 | 4 | 1 | 795 | 37 % | 16 | 2/2 | pass | **YES** |
| ORCH-732 | `PMC4725005` | 1,472 | 30 | 10 | 11 | 8 | 1 | 401 | 27 % | 9 | 1/1 | pass | **YES** |
| ORCH-732 | `PMC7615680` | 553 | 3 | 1 | 0 | 2 | 0 | 537 | 97 % | 2 | 0/2 | fail | — |
| ORCH-732 | `PMC9544450` | 1,338 | 32 | 12 | 15 | 2 | 3 | 400 | 30 % | 11 | 1/1 | pass | **YES** |
| ORCH-734 | `PMC10055903` | 1,628 | 29 | 8 | 15 | 6 | 0 | 602 | 37 % | 12 | 1/1 | pass | **YES** |
| ORCH-734 | `PMC10269868` | 1,082 | 20 | 11 | 2 | 5 | 2 | 50 | **5 %** | 3 | 1/1 | pass | **YES** |
| ORCH-734 | `PMC11016064` | 1,319 | 28 | 12 | 6 | 8 | 2 | 250 | 19 % | 10 | 2/2 | pass | **YES** |
| ORCH-734 | `PMC11961743` | 2,358 | 102 | 9 | 59 | 33 | 1 | 1,498 | 64 % | 37 | 2/2 | fail | — |
| ORCH-734 | `PMC13184244` | 1,821 | 73 | 15 | 46 | 12 | 0 | 1,063 | 58 % | 23 | 1/1 | fail | — |
| ORCH-734 | `PMC4471609` | 2,552 | 44 | 14 | 26 | 2 | 2 | 855 | 34 % | 9 | 1/1 | fail | — |
| ORCH-734 | `PMC6112128` | 1,801 | 89 | 13 | 58 | 18 | 0 | 1,229 | 68 % | 19 | 1/1 | pass | **YES** |
| ORCH-734 | `PMC7402084` | 1,742 | 34 | 12 | 11 | 8 | 3 | 511 | 29 % | 9 | 3/3 | pass | **YES** |
| ORCH-734 | `PMC7910490` | 1,552 | 31 | 11 | 16 | 3 | 1 | 681 | 44 % | 10 | 2/2 | scope | — |
| ORCH-734 | `PMC8211424` | 838 | 24 | 11 | 3 | 7 | 3 | 172 | 20 % | 5 | 1/1 | pass | **YES** |
| ORCH-734 | `PMC9200736` | **3,601** | 151 | 29 | **90** | 24 | 8 | 2,057 | **57 %** | 36 | 2/2 | **timeout** | — |
| ORCH-734 | `PMC9544450` | 1,338 | 29 | 19 | 6 | 2 | 2 | 197 | **15 %** | 4 | 1/1 | pass | **YES** |
| ORCH-730 | `PMC12071552` | 2,013 | 110 | 19 | 70 | 19 | 2 | 1,325 | 66 % | 35 | 1/1 | fail | — |
| ORCH-730 | `PMC7232280` | 2,710 | 98 | 16 | 54 | 27 | 1 | 1,928 | 71 % | 21 | 1/2 | pass | **YES** |
| ORCH-730 | `PMC8510960` | 667 | 6 | 3 | 1 | 1 | 1 | 290 | 43 % | 1 | 2/4 | fail | — |
| C-120 | `PMC10031235` | 1,838 | 36 | 15 | 8 | 9 | 4 | 530 | 29 % | 12 | 1/2 | pass | **YES** |
| C-120 | `PMC11487621` | 443 | 4 | 1 | 3 | 0 | 0 | 430 | 97 % | 3 | 0/3 | fail | — |
| C-120 | `PMC9544450` | 1,408 | 38 | 20 | 15 | 2 | 1 | 514 | 36 % | 7 | 1/1 | fail | — |
| C-121 | `PMC11961743` | **3,600** | 113 | 9 | **71** | 30 | 3 | 1,663 | 46 % | **61** | 3/5 | **timeout** | — |
| C-121 | `PMC4471609` | 1,771 | 49 | 17 | 26 | 5 | 1 | 638 | 36 % | 21 | 1/1 | fail | — |
| C-121 | `PMC9544450` | 1,507 | 41 | 21 | 16 | 3 | 1 | 488 | 32 % | 9 | 1/1 | pass | **YES** |

**23 of 27 legs lose more than a quarter of their runtime to calls that returned nothing. Nine lose
more than half.** The single cleanest leg, `PMC10269868` at 5 %, produced a PWML in 18 minutes.

---

# 3. Where the empties actually are — and it is not Stage 1

| stage | `max_tokens` | calls | normal | **empty** | **empty %** |
|---|---:|---:|---:|---:|---:|
| `chat` — `map_ids` alias + `stoich` classifier | **300** | 469 | 3 | **374** | **79.7 %** |
| `rag_prose_extraction` | 1,500 | 454 | 61 | 247 | **54.4 %** |
| `gap resolver` | 450–900 | 147 | 60 | 46 | 31.3 % |
| `audit loop` | — | 91 | 38 | 27 | 29.7 % |
| Stage 2 inference | 16,000 | 82 | 63 | 14 | 17.1 % |
| **Stage 1 extraction** | **16,000** | **49** | **37** | **8** | **16.3 %** |
| `preprocessor` | 12,000 | 84 | 83 | **0** | **0.0 %** |

**The empty rate tracks the completion budget, and Stage 1 is among the healthiest stages in the
pipeline.** The preprocessor, on 12,000 tokens, did not return a single empty completion in 84
calls. The 300-token call sites returned nothing four times in five.

`chat` is the default label the client applies when a caller passes no `stage_name`
(`client.py:648`). Two call sites produce it, both asking for a small JSON object:

* `mapping/map_ids.py:403` — protein alias resolution, `max_tokens=300`, `OPENROUTER_GAP_MODEL`
* `stoich/classifier.py:133` — participant classification, `max_tokens=300`, **no `model_env_var`**,
  so it resolves to the global `OPENROUTER_MODEL`

They cannot be separated in the archives, because neither passes a stage name. That is registered
below as **`F-198`**.

---

# 4. THE ROOT CAUSE, reproduced live

58 calls, own prompts, no paper text, no pipeline. Two difficulty profiles: `easy` is a one-token
classification; `hard` is an ambiguous non-model-organism alias question of the shape
`map_ids.py` actually asks.

## 4.1 The production shape reproduces on demand

| model | budget | request | **empty** |
|---|---:|---|---:|
| `deepseek/deepseek-v4-flash` | 300 | production, verbatim | **5 / 5** |
| `deepseek/deepseek-v4-flash` | 1,500 | production, verbatim | **3 / 5** |
| `google/gemini-3.8-flash` | 300 | production, verbatim | 0 / 5 |
| `google/gemini-3.8-flash` | 1,500 | production, verbatim | 0 / 5 |

Every deepseek empty carries `content_chars=0`, `finish_reason=length`, and
`reasoning_tokens` **equal to the entire budget** — 294–301 of 300, and 1,499–1,500 of 1,500.

**That is `F-193`'s signature exactly**, and it is now explained rather than observed: the model
spends the whole completion allowance on reasoning tokens, which are billed against `max_tokens`
and never appear in `message.content`. `ORCH-731` A4 named this mechanism as one of two candidates
and could not separate them from the archives. **It is now separated: this is the one.**

**The easy profile does not reproduce it** — 0 of 18 empties at every budget. The failure needs a
question worth reasoning about, which is why it concentrates on the hardest small calls.

## 4.2 The remedy, and the remedy that does not work

Same model, same prompt, same budget, same temperature. One request parameter differs.

| control | 300 tokens | 1,500 tokens |
|---|---:|---:|
| none (production today) | **5/5 empty** | **3/5 empty** |
| `reasoning: {max_tokens: budget/4}` | **5/5 empty** | **3/5 empty** |
| **`reasoning: {enabled: false}`** | **0/5 empty** | **0/5 empty** |

**The cap is silently ignored.** Asked for 75 reasoning tokens, the backends returned 291, 299, 305
and 300. A fix built on `reasoning.max_tokens` would have looked principled and done nothing —
which is the entire reason it was tested before being proposed.

**Disabling reasoning is not merely non-empty, it is better on every delivery axis**: 84–199
completion tokens instead of 300 or 1,500, 1.3–8.4 s instead of 4–47 s, and 237–549 characters of
content where there were zero. It held across **ten different backends**.

## 4.3 What the probe does NOT establish

**Nothing about biological quality.** A model that delivers bytes is not thereby a better
biologist, and disabling a reasoning model's reasoning is a real quality risk on calls where
reasoning currently *succeeds*. `PRODUCT_CONTRACT.md` outranks every number above.

**The argument is narrower than "turn reasoning off".** At `max_tokens=300` the current
configuration returns nothing four times in five. There is no quality there to lose — the change
converts a near-certain zero into an answer. At 16,000 tokens Stage 1 is 16.3 % empty and reasoning
is plainly completing; **no case is made for touching it, and none should be inferred.**

## 4.4 `D-097` is not overturned, and this is a different class

`D-097` and `ORCH-731` retired the "raise `max_tokens`" reflex for Stage-1 **truncation** at 16,000,
where the budget was measured as non-binding — 16,000 tokens bought 38,462 characters while the
failing leg stopped at 9,501. **That ruling stands and is not in tension with this one.** This is a
different class: **zero** content at **300**, where the budget is binding by construction and
measured to the token. **Raising `max_tokens` is not proposed here either** — those budgets are
literals in frozen production source, and the fix below does not touch them.

---

# 5. The fallback route — live, correct, and irrelevant to this problem

| question | answer |
|---|---|
| is `OPENROUTER_EXTRACTION_FALLBACK_MODEL` present in the runtime? | **Yes.** `.env` line 1 of the ORCH-734 block: `google/gemini-3.8-flash`. `.env` is gitignored |
| has rung 3 issued since activation? | **No. Zero times.** All 1,384 traced calls across all 27 legs — including all 12 `ORCH-734` legs and all 3 `C-121` legs — carry `model: deepseek/deepseek-v4-flash`. Not one call to any other model has ever been issued by the pipeline |
| why not on the high-empty chains? | **Because rung 3 is Stage-1 only, and the high-empty chains are not at Stage 1.** 621 of the 716 empties are in `chat` and `rag_prose_extraction`, neither of which has any alternate-model route at all |

## Which conditions reach rung 3 — read at the frozen SHA

`pipeline.py:4060-4075` requires **all** of: a ladder present · `saw_empty_payload` · a materially
different request (`extraction_ladder.py:574-599`: a new `request_hash` **and** either a different
model or a narrower scope) · remaining budget · and the call being Stage-1 extraction.

`saw_empty_payload` is set **only** by a payload that is *present but degenerate* — a localized
repair that closed braces around nothing, or a prefix salvage that yielded an empty object.

| charter question | answer |
|---|---|
| does a **completely empty completion** qualify? | **NO.** Nothing parses, nothing salvages, `JSONDecodeError` is raised with no payload, the flag stays `False`, and the ladder terminates at the normal rung. `ORCH-733` § 3, re-verified |
| does **repeated empty + `length`** qualify? | **NO.** That is the same class — `content_chars=0`. It is also the exact shape § 4 reproduces 5/5, so the dominant failure mode is precisely the one that cannot reach the escalation |
| does **tiny + `stop`** qualify? | **YES**, when the tiny reply parses or salvages to a degenerate payload. This is the `PMC7615680` class |
| does **truncated JSON** qualify? | **CONDITIONALLY.** Only if repair or salvage yields something degenerate. If salvage recovers a usable partial payload the function returns it and never reaches rung 3 |

**The fallback is correctly configured and correctly gated. It is simply pointed at a stage that is
not bleeding, and gated on a shape the bleeding does not take.** Nothing here asks for it to be
changed or removed.

---

# 6. Provider and backend — now partly answered, and the seam re-confirmed

**Empty-response frequency by model, from production evidence: not computable.** Every one of the
1,384 archived calls was answered by `deepseek/deepseek-v4-flash`. There is no second model in the
production record to compare against. The § 4 comparison is a probe, not a cohort, and is scoped
accordingly.

**Backend identity is still not persisted in production** — `client.py` reads no `provider`
attribute and `CompletionDiagnostics` has no such field, exactly as `ORCH-733` § 4 recorded. The
seam it named (`client.py:685` and the mirror at `:899`) is unchanged and remains the right one.

**But the probe proves the field is there for the taking.** `resp.provider` came back populated on
every call, and the routing spread is wide:

* `deepseek/deepseek-v4-flash` was served by **fourteen** distinct backends across 39 calls —
  Alibaba, AtlasCloud, Azure, Baidu, DeepInfra, DigitalOcean, GMICloud, Mancer 2, NextBit, Parasail,
  Phala, SiliconFlow, StreamLake, Venice.
* `google/gemini-3.8-flash` was served only by Google / Google AI Studio, across 19 calls.

**Is one backend disproportionately to blame? No — and the evidence rules it out rather than merely
failing to find it.** Baidu returned one empty and one good answer on the same prompt and budget.
The empty result appeared on Mancer 2, Venice, GMICloud, Alibaba, StreamLake, NextBit and Baidu
alike. When reasoning was disabled, **ten different backends all delivered.** The behaviour tracks
the **model**, not the route.

**Therefore Option D (provider pinning) is rejected on evidence, not merely on cost.** It would also
require the same `extra_body` plumbing the fix needs, so it is not the "configuration already
permits this cleanly" case § 8 reserves it for.

---

# 7. Options, against the charter's own order

| option | verdict |
|---|---|
| **A — keep current primary + fallback** | **REJECTED.** Delivery failure is not a small tail at leg level. 44.9 % of wall clock, 23 of 27 legs over 25 %, and both timeouts delivery-dominated |
| **B — change primary extraction model** | **REJECTED, and it would have been the wrong fix.** Stage-1 extraction is 16.3 % empty and among the healthiest stages. Swapping the primary would change every biological judgment in the pipeline to fix a problem that lives in the auxiliary calls |
| **C — change fallback model** | **REJECTED as ineffective.** The fallback has never issued a single call and covers only a stage that is not bleeding. Changing it changes nothing measurable |
| **D — provider pinning / routing** | **REJECTED on evidence.** § 6 — the same backend produces both outcomes, and every backend delivers once reasoning is off. It is a model property, not a routing property |
| **E — production change** | **THE ONLY OPTION THE EVIDENCE SUPPORTS** |

## Why configuration genuinely cannot reach this — the part that decides the charter

Three independent reasons, each sufficient:

1. **No environment variable exposes a reasoning control.** `client.py` builds its request from
   `model`, `messages`, `temperature`, `max_tokens`, `timeout` and an optional `response_format`.
   There is **no `extra_body`, no `reasoning` parameter and no provider block** anywhere in the
   file. The remedy § 4.2 proves is unreachable from `.env` by construction.
2. **The reasoning cap does not work**, so even an exposed knob of that shape would be inert.
3. **Every model-swap lever also changes biology.** `OPENROUTER_GAP_MODEL` governs the gap
   resolver's candidate *selection*; `OPENROUTER_RAG_EXTRACT_MODEL` governs reaction extraction from
   literature prose; and `stoich/classifier.py` has **no per-stage variable at all**, so reaching it
   means moving the global `OPENROUTER_MODEL` and with it Stage 1 and Stage 2.
   **There is no purely-delivery scoping available by configuration.** Any `.env` fix to this would
   silently change which model makes biological judgments — which is the change this sprint has
   spent months refusing to make by accident.

---

# 8. `F-198` and `F-199` — registered

| id | finding | disposition |
|---|---|---|
| **`F-199`** | **Reasoning-token budget exhaustion is the root cause of `F-193`.** The primary model spends the entire `max_tokens` allowance on reasoning and returns zero content with `finish_reason=length`. Reproduced 5/5 at 300 tokens and 3/5 at 1,500; **0/5 at both** with `reasoning: {enabled: false}`. Costs 44.9 % of leg wall clock and contributed materially to both timeouts. Unreachable from configuration | **CHARTER CANDIDATE — see § 10** |
| **`F-198`** | **`chat_with_tools` emits no `LEG_TRACE` rows.** `_publish_attempt` is called only from `CompletionDiagnostics.note` (`client.py:412`), and `chat_with_tools` (`:837`) never constructs a `CompletionDiagnostics`. **Every `curator` and gap-resolver-synthesis call is invisible to every call census**, including this one — `PMC9200736`'s stderr shows curator empties that appear nowhere in its trace. **All call counts in this document are floors** | REGISTERED, not chartered. Observability only; no production behaviour is affected |

## `F-195` gains a third instance, and it is older than the other two

`runs_validation/2026-09-07_1929/papers/PMC7232280` (`ORCH-730`, *N. crassa*, Moco biosynthesis,
45,054 B) carries **two dangling `compound-location-id` references — 30 and 34 — both from
`transport-compound-visualization` elements**. Byte-identical shape to `PMC6112128` and
`PMC11016064`. `F-195` is now **3 papers**, and the oldest predates the run that found it.

**Not fixed, per § 15 of the charter.** The live PathWhiz import still decides its priority.
Its file is in the import set, labelled, to be imported **last**.

---

# 9. The identity failure — recorded, and it does NOT meet the repeat bar

§ 12 asks for the exact mechanism and warns against chartering on appearances. **The appearances say
it repeats. The mechanism says it does not**, and the mechanism wins.

Both legs die at the same gate with the same error string and the same lineage reason:
`post_normalization` hard gate → `Protein 'X' is missing a UniProt or DrugBank identifier` →
`identifier resolution ended novel (implausible_name_match)` → `failed_check: species_mismatch`.
At the string level that is a clean repeat across a fungus and a plant. **It is not the same defect.**

| leg | entity | best candidate found | why refused | is the refusal right? |
|---|---|---|---|---|
| `PMC4471609` *A. japonicus* | `DmaW` | `Q50EL0` **tryptophan dimethylallyltransferase**, *Aspergillus fumigatus*, score 0.56 | `species_mismatch` | **Arguably not.** Same genus, and the correct enzyme function for this pathway |
| `PMC13184244` *N. tabacum* | `UGT1` | `Q9HAW9` UDP-glucuronosyltransferase 1A8, ***Homo sapiens***, score 0.73 | `species_mismatch` | **Yes, plainly.** A human enzyme is not a tobacco one |
| `PMC13184244` *N. tabacum* | `MATE1` | `Q96FL8` multidrug and toxin extrusion protein 1, ***Homo sapiens***, score 0.73 | `species_mismatch` | **Yes, plainly** |
| `PMC13184244` *N. tabacum* | `β-GD1` | none — 0 candidates | `no_match` | different mechanism entirely; note the Greek β |

**The nicotine paper is the identity gate working correctly.** Admitting a human UGT as a tobacco
enzyme would have been exactly the fabricated identity the gate exists to refuse, and it would have
produced a PWML the product contract forbids. **Counting that leg as evidence of an identity defect
would have chartered a card to break a safeguard.**

**The cycloclavine paper is a genuinely different question** — whether a congeneric orthologue of
the right function may stand in for an organism whose own proteome is not in UniProt. That is a
**biological policy decision for the product owner**, not a code defect, and relaxing it is exactly
the "weaken a biological gate to increase PWML production" that merge rule 6 forbids without an
explicit product ruling.

> **Recommendation: do NOT charter an identity card. `n = 1` on the only shape that is even
> arguable.** Mechanism recorded, per § 12. If a second *congeneric-orthologue* refusal appears on
> an independent normal paper, that is the trigger to revisit — not another `species_mismatch`
> string match.

---

# 10. DECISION

## `ONE FINAL RUNTIME CARD REQUIRED`

Against § 16 C's three conditions, all of which must hold:

| condition | verdict |
|---|---|
| a repeated runtime mechanism kills multiple reasonable papers | ✅ Two legs hit the wall-clock wall delivery-dominated (90 and 71 empties; 57 % and 46 % waste); `PMC11961743` did it on two independent runs. 23 of 27 legs lose over a quarter of their budget |
| configuration cannot solve it | ✅ § 7 — no env var exposes it, the cap is ignored, and every model-swap lever changes biology |
| root cause is deterministic and narrow | ✅ § 4 — 5/5 → 0/5 across 30 live calls with **one request parameter** as the only difference, holding across ten backends |

**This is not a card chartered because model calls are sometimes ugly.** § 16 warns against exactly
that, and the charter was nearly satisfied without one: the empty-completion rate was already known,
already registered as `F-193`, and `ORCH-733` had correctly ruled it a small tail *at Stage 1*. What
changes the answer is that the leg-level cost is 44.9 % of wall clock, the cause is now a single
measured parameter rather than a hypothesis, and the remedy was **tested and one candidate failed**
before either was proposed.

## The card, scoped

**Boundary:** `src/t2pw/llm/client.py`, the two request builders only (`chat_detailed` ~`:660`,
`chat_with_tools` ~`:877`). **Nothing else.**

**Change:** pass an OpenRouter `reasoning` control on the request, **applied only to calls whose
`max_tokens` is too small for reasoning to complete** — the 300- and 1,500-token auxiliary calls
where the current configuration returns nothing four times in five. Stage 1 and Stage 2 at 16,000
tokens are **explicitly out of scope**; they are among the healthiest stages and no evidence here
supports touching them.

**Must not:** change any `max_tokens` literal · change any model selection or env var · touch
`reaction_support.py`, `strict_quarantine.py`, any admission gate, threshold or biological rule ·
alter the extraction ladder or its rungs · repair biology after the canonical graph is frozen.

**G9 proof is available and behavioural**, which is the gate this sprint fails most often: the base
tree returns `content_chars=0` on 5 of 5 calls at `max_tokens=300`; the tip returns content on 5 of
5. `evidence/g11/ORCH-739/04` and `06` are that pair already, and
`orch739_budget_model_probe.py` reproduces it on demand.

**The quality risk must be measured, not assumed.** Reasoning is being disabled on calls that are
today returning nothing, so there is no quality there to lose — but that claim needs a small cohort
after the patch, not before, and the card should carry it as an obligation.

## What this decision is NOT

**It does not reopen `C-119`, `C-120` or `C-121`.** No finding here touches them and none is
claimed. `PMC4471609` failing on identity after `C-121` fixed `F-192` is § 11's expected case, not a
regression.

**It does not delay the PathWhiz import.** § 11's inventory is ready now and the import is the
product owner's step; it does not depend on this card.

**It does not block `RAG v2` indefinitely.** The card is one file, two functions, one parameter,
with its proof already on disk.

---

# 11. PathWhiz import inventory and handoff

**Thirteen files, all committed and hash-pinned** in
[`pathwhiz_review/IMPORT-SET/`](pathwhiz_review/IMPORT-SET/) with
[`SHA256SUMS.txt`](pathwhiz_review/IMPORT-SET/SHA256SUMS.txt), verified `OK` by `sha256sum -c`.
`ORCH-735` committed eleven; **this task added the two that were still untracked on one disk** —
the `F-187` exposure. **Copies only; not one byte was modified, and every copy was `cmp`-verified
byte-identical to its source run tree, which is preserved unchanged.**

## Import in this order

**1 — clean `ORCH-734` outputs.** The primary set.

| file | pathway / organism | bytes | expect | inspect after import |
|---|---|---:|---|---|
| `ORCH734_PMC10055903.pwml` | sialic acid, *H. sapiens* | 55,011 | READY | UDP-GlcNAc → … → CMP-Neu5Ac chain intact; GNE/MNK present |
| `ORCH734_PMC10269868.pwml` | ADP-heptose, *H. pylori* | 48,436 | READY | GmhA/HldE/GmhB placed; the regulator CsrA is present and is **not** a pathway enzyme |
| `ORCH734_PMC7402084.pwml` | cyanogenic glucoside, *P. lunatus* | 40,294 | READY | CYP79D71 → CYP83E46/47 → UGT85K31 order |
| `ORCH734_PMC8211424.pwml` | corticosteroid, *H. sapiens* | 22,859 | READY | **thin by design — 2 reactions.** Confirm it renders rather than judging completeness |

**2 — `C-120` PSAT output.**

| file | pathway / organism | bytes | expect | inspect after import |
|---|---|---:|---|---|
| `C120_PMC10031235_PSAT.pwml` | PSAT, *H. sapiens* | 48,401 | READY | the `PSAT` → `PSAT1` family-stem resolution `C-120` was chartered for |

**3 — successful `C-121` and control outputs.** `PMC9544450` appears **three times on purpose** —
one paper, three draws, 4/5/6 reactions. Import them together; the spread *is* the finding.

| file | pathway / organism | bytes | expect | inspect after import |
|---|---|---:|---|---|
| `C121_PMC9544450.pwml` | menaquinone, *E. coli* | 48,826 | READY | **the `C-121` live-validation control.** 5 reactions, `cytoplasmic state` survived the sweep |
| `ORCH734_PMC9544450.pwml` | menaquinone, *E. coli* | 49,436 | READY | 6 reactions — same paper, different draw |
| `ORCH732_PMC9544450.pwml` | menaquinone, *E. coli* | 39,686 | READY | 4 reactions — the third draw |
| `ORCH732_PMC12051542.pwml` | — | 80,518 | READY | largest file in the set; check the canvas is legible at 8 reactions |
| `ORCH732_PMC4725005.pwml` | — | 54,459 | READY | |

**4 — the `F-195` files, LAST and separately.** § 15's rule was written before the test.

| file | pathway / organism | bytes | expect | dangling refs |
|---|---|---:|---|---|
| `ORCH734_PMC6112128.F195.pwml` | tubercidin, *S. tubercidicus* | 78,492 | **FAIL** | `compound-location-id` 55 |
| `ORCH734_PMC11016064.F195.pwml` | carnitine, *H. sapiens* | 107,951 | **FAIL** | 66, 77 |
| `ORCH730_PMC7232280.F195.pwml` | Moco, *N. crassa* | 45,054 | **FAIL** | 30, 34 — **the third instance, found by this task** |

**What their outcome decides, and nothing else decides it:**

* **PathWhiz accepts and renders them** → `F-195` is a limitation of our structural checker.
  Document as a tolerated defect and close it with no code.
* **PathWhiz rejects, breaks, or renders them wrongly** → `F-195` is a genuine final reliability
  blocker and may justify **one** narrow referential-integrity card.

**No pre-emptive fix either way, and do not repair a file to make an import succeed.** If an import
fails, the file is the evidence.

> **`IMPORT READY` is not `IMPORT PASS`.** Ten of the thirteen satisfy every property the importer
> checks that is decidable from the bytes. Pressing Import needs the product owner's account, and
> that step is the one this whole phase is waiting on.

---

# 12. `RAG v2` handoff — unchanged, and now with one more inherited defect

[`RAG-V2-HANDOFF-REQUIREMENTS.md`](RAG-V2-HANDOFF-REQUIREMENTS.md) stands as written. Its § 7
inherited-defect table gains two rows: **`F-198`** (the tool-calling path is invisible to every call
census, so a mode-C run's cost cannot be measured from `LEG_TRACE` alone) and **`F-199`**, which
matters more than it looks — **mode C issues LLM calls per frontier node**, so a 45 % wall-clock
tax on auxiliary calls compounds directly into the "unbounded mine is an unbounded bill" risk that
document already names. **Fixing `F-199` before `RAG v2` is worth more to that phase than to this
one.**

---

# 13. Process and provenance

* Six bounded jobs, all foreground or wrapper-owned, **all `FINAL SURVIVING COUNT : 0` /
  `cleanup : success`**. Heavy lock free before and after. Pre-existing Streamlit and isort
  processes **reported, never killed**. No `taskkill /IM`, no `pkill`.
* **58 live calls total**, all micro-probes with their own prompts. **No paper was re-run**, no
  cohort launched, no pipeline leg executed.
* `main` untouched — local `7531692`, remote `03f1af5`, unchanged.
* `git diff 24dd4342 HEAD -- src/` **empty**. Production remains frozen.
* Run directories, caches, worktrees and PWML files **all preserved**. No cache committed.

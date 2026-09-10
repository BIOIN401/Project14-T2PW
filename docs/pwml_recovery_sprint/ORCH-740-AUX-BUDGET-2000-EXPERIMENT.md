# `ORCH-740` — auxiliary budget `300 → 2000`, reasoning ENABLED. The experiment, and its result.

**2026-09-09.** A bounded call-level experiment against the one production call site the
`ORCH-739` census implicated. **No production code was changed** — `git diff 24dd4342 HEAD --
src/` is empty, and no experimental diff was ever written, so none had to be reverted. `main`
untouched. Protected `streamlit_app.py` unchanged at `47e4fafa…`. No paper re-run, no cohort
launched, no worktree pruned, no cache committed.

**40 live calls**, own prompts verified byte-equal to production's, reasoning left **enabled**
throughout. Tool: [`evidence/orch740_aux_budget_probe.py`](evidence/orch740_aux_budget_probe.py).
Reports `evidence/g11/ORCH-740/01`–`02`, both `FINAL SURVIVING COUNT : 0` / `cleanup : success`.

---

# DECISION

## `REJECT 2000`

**And the hypothesis was half right, which is why this needs saying carefully.**

> **`300` IS too small for a reasoning model. That part of the hypothesis is confirmed, not
> refuted.** At 300 tokens a reasoning backend returns nothing **93 %** of the time. At 2000 it
> returns something **53 %** of the time, and it produced three useful alias sets that 300 could
> not have produced at any draw.
>
> **`2000` is still the wrong fix, because the arithmetic goes the wrong way.** Each production
> call costs **2.33×** more and each *successful* answer costs **1.77×** more. The motivating
> problem was that empty retries consume 44.9 % of leg wall clock. A change that increases
> wall clock per answer does not solve that problem, it relocates it.
>
> **The budget is not the binding variable. The backend's reasoning mode is.** When a
> non-reasoning backend answered, the empty rate was **0 % at BOTH budgets** and the answer
> arrived in 0.3–6.2 s. The budget only changes the odds inside the reasoning subset, and buys
> those odds at four times the latency.

`2000 + reasoning enabled` **does not solve the measured mechanism.**

---

# 1. What was actually tested, and the population correction that had to come first

## 1.1 `ORCH-739` mis-attributed the affected population. Corrected here.

`ORCH-739` § 3 named the untagged `chat` bucket as *"`map_ids` alias + `stoich` classifier"*.
**The `stoich` half is wrong and no budget there should be touched.**

| call site | why it is NOT in the measured population |
|---|---|
| `stoich/classifier.py:136` (300 tok) | reached only from `stoich/agent.py:420` → `run_stoich_agent` → `streamlit_app.py:4277`, which fires only when the `use_stoich_agent` checkbox is set. **The batch driver never sets it**, so this code did not execute in any measured leg |
| `stoich/agent.py:568`, `:596` (300 tok) | call `_client.chat.completions.create` **directly**, bypassing `t2pw.llm.client` — no retry loop, no trace row, so they cannot be in a bucket derived from `LEG_TRACE` |
| `extraction/extract.py:18` (1200 tok) | `run_demo()`, a hard-coded glutathione string behind two `__main__` guards |
| `map_ids.py:3267` / `gap_resolver.py:298` (450 tok) | `infer_entity_species` passes `stage_name="gap resolver"`, so it is measured — in the **`gap resolver`** bucket at 31.3 %, not in `chat` |

Every other `chat`/`chat_detailed` caller passes a `stage_name`. **The `chat` bucket is
`mapping/map_ids.py:406`, `_ai_protein_synonym_lookup`, alone** — the protein alias lookup that
runs when UniProt fails to match a protein by its primary name. That single call site is the
whole affected population, and it is the only thing this experiment touched.

**A search-and-replace of `300` would have changed three call sites that never ran and one that
is not in the bucket.** § 2 of the charter forbade exactly that, and it was right to.

## 1.2 The prompt is production's, and the probe refuses to run if it drifts

The probe rebuilds `_ai_protein_synonym_lookup`'s prompt and, before issuing a single call,
re-reads `map_ids.py` and refuses unless every prompt line is present in the source. It printed
`prompt drift check: PASS -- 10 prompt lines all present in map_ids.py` on both rounds.

Cases are five proteins whose identity resolution **actually failed** in committed runs: `DmaW`
and `EasF` (*A. japonicus*, `PMC4471609`), `UGT1` and `MATE1` (*N. tabacum*, `PMC13184244`),
`SdPCS` (*S. divaricata*, `PMC11961743`).

## 1.3 One variable

Same model (`deepseek/deepseek-v4-flash` via `OPENROUTER_GAP_MODEL`), same provider routing, same
prompt, same `response_format`, same temperature, same retry policy, **reasoning untouched and
enabled**. Only `max_tokens` differs. Two rounds, r1 at 1 repeat and r2 at 3, pooled to 20 calls
per arm.

---

# 2. Result

## 2.1 The raw comparison

| budget | calls | empty | empty % | usable | **productive** (`aliases > 0`) | malformed | median s | total s | median completion tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **300** | 20 | 13 | **65 %** | 7 | **2** | **0** | **4.3** | **88.5** | 300 |
| **2000** | 20 | 7 | **35 %** | 11 | **4** | **2** | 17.6 | 290.6 | 1,110 |

**`usable` overstates the gain and must not be quoted alone.** It applies production's own
predicate (`json.loads`, then a list under `aliases`), and `{"aliases": []}` satisfies it. That is
a *valid* production answer meaning "I know nothing", and it resolves no identity. Six of the
eleven `usable` answers at 2000 are exactly that, 15 characters long. **`productive` is the column
that matters, and it moved from 2 to 4 of 20.**

## 2.2 The comparison that decides it — retry-adjusted, per production call

A production alias lookup is one `chat()` invocation, and the client retries an empty HTTP 200
*inside* its own loop up to `LLM_MAX_RETRIES=3`. So the honest unit is the whole invocation.

| budget | mean s / attempt | E[attempts] | **E[s] per call** | **P(call yields nothing)** | **s per successful answer** |
|---:|---:|---:|---:|---:|---:|
| **300** | 4.43 | 2.07 | **9.18** | **27.5 %** | **12.65** |
| **2000** | 14.53 | 1.47 | **21.39** | **4.3 %** | **22.35** |

**2000 buys a large reliability gain and pays more than it is worth in time.** Outright failure
falls from 27.5 % to 4.3 %, which is real and would matter if time were free. It is not: the same
call costs **2.33×** more, and every answer that does arrive costs **1.77×** more.

## 2.3 The confound, and it is larger than the effect under test

| budget | subset | n | empty | median s |
|---:|---|---:|---:|---:|
| 300 | backend **did not** reason (`reasoning_tokens == 0`) | 6 | **0 (0 %)** | **1.4** |
| 300 | backend **did** reason | 14 | 13 (**93 %**) | 4.5 |
| 2000 | backend **did not** reason | 5 | **0 (0 %)** | **0.8** |
| 2000 | backend **did** reason | 15 | 7 (**47 %**) | 20.3 |

**Eleven of forty calls were served by a backend running the model in non-reasoning mode, and not
one of them returned empty — at either budget, in under two seconds.** OpenRouter routed this one
model name across DigitalOcean, DeepInfra, AtlasCloud, Venice, NextBit, GMICloud, SiliconFlow,
Baidu, Phala, StreamLake and Novita, and which one answers is not chosen by us and is not recorded
by production.

The productive answers make the point sharpest:

| budget | entity | aliases | reasoning tokens | seconds | backend |
|---:|---|---:|---:|---:|---|
| 300 | `DmaW` | 3 | **0** | 6.2 | DigitalOcean |
| 300 | `MATE1` | 3 | **0** | 1.8 | DeepInfra |
| 2000 | `MATE1` | 3 | **0** | **1.8** | DeepInfra |
| 2000 | `DmaW` | 6 | 1,114 | 12.1 | Baidu |
| 2000 | `UGT1` | 4 | 1,823 | 20.4 | GMICloud |
| 2000 | `SdPCS` | 5 | 1,185 | 23.6 | StreamLake |

**Both productive answers at 300 came from backends that did not reason.** Three of the four at
2000 came from backends that did, and each took 12–24 s to deliver what a non-reasoning backend
delivered in under two.

**This is the genuine, honest gain of 2000, and it should not be dismissed:** those three answers
are outcomes 300 cannot produce at any draw, because at 300 the reasoning consumes the entire
budget before a single content token is emitted. `DmaW` returning 6 aliases is exactly the
identity help that leg needed. The finding is not that 2000 does nothing. It is that 2000 pays
four times the latency for something a different reasoning setting delivers immediately.

## 2.4 Malformed output increased, and the failure mode is instructive

Zero malformed at 300; **two at 2000**, both from Venice, both with `finish_reason=stop`:

```
{": []}? Wait, the JSON must be valid. The instruction says ": []}
{"": []}
```

**The model's reasoning leaked into the content.** Production would parse the first as valid JSON
with no `aliases` key and silently treat it as no aliases, so this degrades quietly rather than
raising. A larger budget gives the model more room to finish its answer and also more room to
narrate itself into the payload.

---

# 3. Against the charter's own criteria

## § 9 — success requires all four

| criterion | verdict |
|---|---|
| empty responses fall substantially | ✅ 65 % → 35 %, and 93 % → 47 % within the reasoning subset |
| **total wall-clock does not materially worsen** | ❌ **2.33× per call, 1.77× per successful answer, 3.3× total probe time** |
| paper-level retries decrease | ✅ E[attempts] 2.07 → 1.47; outright failure 27.5 % → 4.3 % |
| no biological or schema safeguard regresses | ✅ **trivially — no code was changed.** But malformed output rose 0 → 2, which is a mild negative signal in the same direction |

**Two of four hold. The wall-clock criterion is the one the whole exercise exists to satisfy, and
it fails.**

## § 10 — reject if any hold

| criterion | met? |
|---|---|
| the model burns 2000 reasoning tokens and still emits empties | ✅ **7 of 20 calls consumed exactly 2,000 reasoning tokens and returned zero content** |
| call latency increases substantially | ✅ **4.1× median, 3.3× mean** |
| total paper runtime worsens | **projected, not measured** — see § 4 |
| malformed output increases | ✅ **0 → 2** |
| no meaningful reduction in retry burden | ❌ not met; there **is** a real reduction |

**Three of five reject conditions are met outright.** § 10's instruction is explicit: *"Do NOT
keep 2000 merely because it is larger."*

---

# 4. Why no paper-level validation was run

§ 8 conditions the 3–5 paper cohort on the call-level result being **"clearly better."** It is
not. It is better on reliability and worse on time, with the two roughly cancelling and the
wall-clock side losing on the retry-adjusted arithmetic.

Running it would have required writing the production diff § 14 then tells me to discard, and
2–3 hours of leg time to measure a change the call-level data already says costs 2.33× per call.
**No production diff was ever written, so there is nothing to revert** — which is the cleanest
possible form of § 14's instruction.

**What that leaves unmeasured, stated plainly:** whether fewer failed lookups would shorten the
caller's re-ask loop enough to offset the per-call cost. § 5 below shows why that loop is the
bigger lever, and why it is not this card's variable.

---

# 5. `F-200` — the amplification, which is a larger lever than the budget

The 469 `chat` calls in the census come from **40 distinct request hashes.**

| paper | `chat` calls | distinct requests | calls per request |
|---|---:|---:|---:|
| `PMC13184244` | 34 | 2 | **17.0** |
| `PMC6112128` | 41 | 2 | **20.5** |
| `PMC9200736` (timed out) | 78 | 4 | **19.5** |
| `PMC12071552` | 71 | 6 | 11.8 |
| `PMC11961743` (ORCH-734) | 69 | 6 | 11.5 |
| `PMC11961743` (C-121) | 77 | 10 | 7.7 |
| `PMC11487621` | 55 | 6 | 9.2 |
| `PMC7232280` | 44 | 4 | 11.0 |
| **total** | **469** | **40** | **11.7** |

`LLM_MAX_RETRIES` is **3**, so the client can account for at most ~2.07 attempts per invocation.
**The remaining factor of roughly 5.6 is the caller re-asking the same question.** `PMC9200736`
spent 78 calls on four questions before hitting the wall.

**Forty questions should not cost 469 calls at any budget.** Raising the budget multiplies the
cost of each call in that loop; it does not shorten the loop. **REGISTERED, NOT CHARTERED** —
§ 12 restricts this card to one variable, and the re-ask loop is a different one.

---

# 6. `F-201` — the stoich agent's calls are untraced, and an empty answer silently becomes "uncertain"

`stoich/agent.py:568` and `:596` call `_client.chat.completions.create` directly at
`max_tokens=300`, bypassing `t2pw.llm.client` entirely. They therefore get **no retry loop, no
empty-completion handling and no trace row.** The reply is consumed as:

```python
av = _parse_json_from_text(audit_response.choices[0].message.content or "").get("verdict", "uncertain")
```

**An empty completion becomes `"uncertain"`**, indistinguishable from the model genuinely being
unsure. Given the 65 % empty rate measured at this budget on this model, a run with
`use_stoich_agent` enabled would have its biochemical audit silently degrade to no-opinion on most
compounds, with nothing anywhere recording that the provider never answered.

**Latent, not active:** the path is behind a Streamlit checkbox the batch driver never sets, so no
measured leg executed it. **REGISTERED, NOT CHARTERED**, and explicitly out of this card's scope
under § 12. It is noted because anyone acting on the "raise the 300s" instruction would have
edited these two literals and changed a code path that has never run.

---

# 7. What the next experiment would be, and it is NOT implemented here

§ 10 names it: **reasoning-disabled auxiliary calls.** `ORCH-739` § 4.2 already measured that arm
on a different prompt — 0 of 5 empty at 300 and at 1,500, at a third the tokens and a fraction of
the latency — and § 2.3 above independently corroborates it from this call site's own data: every
non-reasoning draw here answered, at either budget, in under two seconds.

**This card does not implement it, propose it as its decision, or treat this experiment as its
justification.** `F-199` remains the standing charter candidate and is unchanged by this result;
if anything this experiment strengthens it, because it shows the budget lever cannot substitute
for the reasoning lever.

**One option this experiment does NOT rule out** and which nobody has measured: an intermediate
budget. The productive answers from reasoning backends used 1,114–1,823 completion tokens, so
something near 2,000 is what a reasoning draw needs *when it works* — the cost is intrinsic, not
an artifact of the ceiling. That is why a middle value is unlikely to rescue the arithmetic, and
why it is noted rather than recommended.

---

# 8. Process and provenance

* Two bounded jobs, both `FINAL SURVIVING COUNT : 0` / `cleanup : success`. Heavy lock free before
  and after. Pre-existing Streamlit and isort processes reported, never killed.
* **40 live calls**, all micro-probes with production's own prompt and no paper text. No pipeline
  leg, no Stage-1 extraction, no Streamlit run.
* **Stage 1 and Stage 2 untouched.** No `max_tokens` literal anywhere in `src/` was modified;
  `git diff 24dd4342 HEAD -- src/` is empty.
* **The probe's own prompt-drift guard refused to run twice before any call was issued**, because
  its first two versions were too strict about the seam where the source splits one prompt line
  across two string literals. Both refusals were **before** the OpenAI client was even constructed,
  so neither cost a call. Those two reports were **overwritten by the successful rerun at the same
  `--json` path**, so only the passing pair is on disk and both read `exit 0`. The refusals are
  recorded here rather than in an artifact. The guard is the reason this probe cannot silently
  measure a prompt production no longer sends, and it earned its place by firing.
* `main` untouched. Run directories, caches, worktrees and PWML files all preserved.

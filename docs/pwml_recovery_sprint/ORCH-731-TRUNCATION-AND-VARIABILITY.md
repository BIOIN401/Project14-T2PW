# ORCH-731 — the two ORCH-730 failures, diagnosed to root cause.

**Read-only diagnosis, 2026-09-07.** No production edit. No pipeline leg re-run. Production
still frozen at `c7a0663e`. Evidence: `evidence/orch731_provider_limit_probe.py` / `.log` /
`orch731_provider_probe.json` · `evidence/g11/ORCH-731/02-provider-probe.json`.

The one thing archives could not answer needed **three direct chat-completion calls** with the
probe's own prompts — no paper text, no leg, no cache write. Everything else is archive work.

---

# PART A — the Stage-1 truncation (`PMC8510960`)

## A1. What model/provider was actually called

`deepseek/deepseek-v4-flash`, every call, all nine roles. **But the backend behind it varies
per call.** The probe's three calls were routed by OpenRouter to **three different providers**:

| probe call | provider |
|---|---|
| A | `AtlasCloud` |
| B | `GMICloud` |
| C | `GMICloud` |
| (first run of A, minutes earlier) | `Alibaba` |

**The archived traces record `model` but never `provider`.** So for `PMC8510960` the backend is
**unrecoverable**. That is an observability gap, and it matters — see A6.

## A2. What exact `max_tokens` reached the API request

**16000, unmodified.** The path is `streamlit_app.py:5467` `max_tokens=int(extract_tokens)` →
`pipeline.run_stage_one_with_chunking(max_tokens=…)` → `client.py:657` `"max_tokens": max_tokens`
into `chat.completions.create(**kwargs)`. **The client applies no clamp, no per-chunk division
and no scaling.**

## A3. What the provider reported

**Per-call usage is recorded NOWHERE.** `client._record_usage` (`client.py:118-124`) only
increments a module-level cumulative counter that is never persisted;
`orch716_openrouter_usage.log` is account-level. So `PMC8510960`'s own token counts **cannot be
recovered** — a second observability gap.

Measured live instead:

| probe | requested | `completion_tokens` | `reasoning_tokens` | `prompt_tokens` | finish | content chars |
|---|---:|---:|---:|---:|---|---:|
| **A** over-long | **16000** | **16000** | 337 | 52 | `length` | **38,462** |
| **B** trivial | 16000 | 33 | 27 | 18 | `stop` | 12 |
| **C** over-long | **256** | **256** | 189 | 52 | `length` | 170 |

## A4. Where was the response actually cut?

**Our own `max_tokens` is honoured exactly** — 16000 → 16000, 256 → 256, to the token. There is
no provider cap below ours and no transport limit in play.

**Therefore `PMC8510960` was NOT cut at our configured budget.** Its Stage-1 call stopped at
**9,501 content characters**, while probe A shows 16000 tokens buys **38,462 characters** on the
same setting. 9,501 chars is roughly **3,500 tokens — about 22 % of the budget we paid for.**

Two mechanisms can produce that, and the archives cannot separate them:

1. **Reasoning tokens.** They are billed against `max_tokens` and **never appear in
   `message.content`**. Probe C is the proof in miniature: 256 completion tokens of which
   **189 were reasoning**, leaving 170 characters of answer. At Stage-1 scale, a reasoning-heavy
   draw could burn ~12,500 tokens and leave ~3,500 for content — exactly what was observed.
2. **Backend variation.** The same model name is served by AtlasCloud / GMICloud / Alibaba, and
   which one answers is not recorded.

**It was not a JSON/parser boundary and not a transport limit:** `finish_reason` is `length`
from the provider, and the parser saw a genuinely incomplete object.

> **This retires the "raise the Stage-1 budget" reflex.** The budget was not the binding
> constraint. `D-097`'s prohibition on blaming token budget stands, and now has a mechanism
> behind it rather than only caution.

## A5. Every `finish_reason=length` in the corpus — 2,675 attempts across 4 runs

**922 are `length` (34.5 %) — but 842 of those (91 %) returned ZERO characters** with
`status=empty`.

| | count | what it is |
|---|---:|---|
| `length` + **empty**, 0 chars | **842** | the provider consumed the budget and returned nothing. Already known to the client as "empty-but-successful" (`client.py:695`) and retried as a transient |
| `length` + `ok`, content present | **80** | genuine truncation |

**Genuine truncations by stage** (all one model, so no model clustering is possible):

| n | stage | content chars min / p50 / max |
|---:|---|---|
| 33 | audit loop | 126 / 2,685 / 14,686 |
| 19 | `rag_prose_extraction` | 40 / 167 / 1,180 |
| 18 | gap resolver | 1 / 339 / 2,763 |
| 4 | chat | 11 / 15 / 410 |
| **2** | **Stage 1 extraction** | **9,501 / 10,198 / 10,895** |
| 2 | Stage 2 inference | 5,060 / 30,360 / **55,660** |
| 1 | inference json repair | 30,664 |
| 1 | preprocessor | 318 |

**Genuine Stage-1 extraction truncation has happened exactly twice in the whole corpus:**
`PMC8510960/strict` (this run, 9,501) and `PMC13017326/strict` (the pilot, 10,895 — that leg
timed out). It is a **rare** event, not a systematic budget shortfall.

And the clustering settles A4 further: **Stage 2 reached 55,660 characters on the same nominal
16000 budget.** A stage that can emit 55.7k chars is not being capped at 9.5k by the budget.

## A6. Is there a larger-budget retry, and why didn't it engage?

**Yes — and it is wired to the wrong stage.**

`pipeline.py:567` `_default_retry_tokens`:

```python
if _looks_truncated_json_failure(attempts):          # :578 -- matches "unterminated string",
    return min(24000, max(max_tokens + 800,          #         "expecting value", "unexpected end", "eof"
                          int(max_tokens * 1.5)))    # 16000 -> 24000
return max(200, int(max_tokens * 0.6))               # otherwise SHRINK to 9600
```

A truncation-aware escalation that would have taken 16000 → **24000** exists and does exactly
the right thing. **But `_default_retry_tokens`, `retry_max_tokens` and
`_looks_truncated_json_failure` appear only at `pipeline.py:428, 511-513, 567, 573, 578, 674,
712` — every one of them inside the Stage-2 inference chunk loop.**

**`run_stage_one_with_chunking` (`:2630`) and the Stage-1 extraction path have no equivalent.**
`PMC8510960` failed at Stage 1, so the escalation was never reachable.

Instead Stage 1 did this:

```
stage1_extraction     finish=length  9,501 chars -> invalid_json
stage1_json_repair    finish=stop       84 chars -> semantic_guard_failed   <- refused to fabricate
stage1_extraction     finish=length        0 chars -> empty (the 842-case)
stage1_extraction     finish=stop    1,267 chars -> valid_json_zero_processes
stage1_extraction     finish=stop    1,159 chars -> valid_json_zero_processes
ladder termination
```

**The repair guard behaved correctly** — it declined to invent the missing remainder. The leg
failed in the safe direction. What it never got was a retry with more room.

**Registered as an observation, NOT chartered.** Whether to port the Stage-2 escalation to
Stage 1 is a product decision, and on this evidence its expected value is low: genuine Stage-1
truncation is 2 events in 2,675 attempts, and A4 shows the budget was not the binding limit
anyway.

---

# PART B — `PMC12071552` variability: pilot PASS vs ORCH-730 FAIL

## B1. The divergence is upstream, at Stage 1, and it is total

| | pilot (PASS) | ORCH-730 (FAIL) |
|---|---|---|
| **Stage-1 reactions** | **2** — `DltA-catalyzed D-Alanine adenylation`, `…D-Serine adenylation` | **0** |
| Stage-1 transports | 2 | 2 |
| Stage-1 proteins | `DltA, DltC, DltB, DltD` | `DltA, DltC, DltB, DltD` — **identical** |
| merged payload | 4 reactions | 4 reactions |
| **final canonical** | **2 reactions** (the DltA adenylations) | 4 reactions (`DltA transfers D-Ala to DltC`, `…D-Ser…`, `D-Ala transfer from DltC-D-Ala to WTA`, `…D-Ser…`) |
| core_accepted | ≥1 | **0** |

**The same four proteins were extracted both times.** What changed is *which reactions* Stage 1
produced — and therefore **which proteins became essential participants**.

- Pilot's surviving reactions are **DltA**-centric. DltA carries an identifier → they survive.
- ORCH-730's reactions are **DltC**- and **DltB**-centric. Neither carries an identifier → all
  six processes quarantine.

## B2. The six quarantined entities

| # | process | reason | essential participant |
|---|---|---|---|
| 0 | `DltA transfers D-Ala to DltC` | `protein_missing_external_identity` | `DltC-D-Ala` |
| 1 | `DltA transfers D-Ser to DltC` | `protein_missing_external_identity` | `DltC-D-Ser` |
| 2 | `D-Ala transfer from DltC-D-Ala to WTA` | `undeclared_entity_in_outputs` | `DltC` |
| 3 | `D-Ser transfer from DltC-D-Ser to WTA` | `undeclared_entity_in_outputs` | `DltC` |
| 4 | `D-Ala transport to cell wall…` | `protein_missing_external_identity` | `DltB` |
| 5 | `D-Ser transport to cell wall…` | `protein_missing_external_identity` | `DltB` |

## B3. Per-entity: query, candidates, selection, score, margin, species verdict

**The queries are byte-identical across both runs**, e.g.
`(protein_name:"DltA" OR gene:"DltA") AND organism_name:"Staphylococcus aureus (MRSA N315)"`
then the unrestricted retry.

| entity | judged candidate | score | margin | species verdict | outcome |
|---|---|---:|---:|---|---|
| `DltA` **pilot** | **P10515** *Homo sapiens* (gene alias `DLTA`) | 0.56 | −1.0 | **mismatch** | rejected |
| `DltA` **ORCH-730** | **P0C397** *S. aureus* | 0.56 | −1.0 `no_competing_candidate` | **ok** | **verified_real_protein** |
| `DltB` | P39580 *B. subtilis* | 0.56 | −1.0 | mismatch | rejected |
| `DltC` (pilot) | P39579 *B. subtilis* | 0.56 | −1.0 | mismatch | rejected |
| `DltC-D-Ser` | P0C426 *S. aureus* | 0.56 | −1.0 | ok | **`implausible_name_match`** |
| `DltD` | Q2FZW3 *S. aureus* **strain NCTC 8325** | 0.56 | −1.0 | **mismatch** | rejected |

## B4. Did the API results differ? The cache? The ordering?

**No, no, and no.**

- **API results: identical.** Same accessions, same order, same scores in both runs —
  `P10515, Q81G39, P0C397, Q9X2N4, P39581…` for DltA; `P39579, Q5M4V3, Q2FZW4, P0C426…` for DltC.
- **Candidate ordering: identical.**
- **Cache: the snapshots differ in bytes** (pilot `589d6ce3…` 4.8 MB; ORCH-730 `f912e1d1…`
  5.5 MB, 1,693 vs 1,473 protein entries) **but the Dlt data does not.** The pilot snapshot holds
  **no** `dlt` key at all; ORCH-730 holds 8. Cached or live, **the returned candidate sets agree**.

## B5. Did the correct identities exist in both runs?

> ## YES. Every correct *S. aureus* accession was present in the candidate pool, in both runs, and was not selected.

```
DltA -> P0C397  D-alanine--D-alanyl carrier protein ligase   Staphylococcus aureus   [position 3]
DltC -> Q2FZW4 / P0C426  D-alanyl carrier protein            Staphylococcus aureus   [positions 3, 4]
DltD -> Q2FZW3  Protein DltD                                 S. aureus NCTC 8325     [position 1]
```

**Why they were not selected — two independent seams, both keyed on the strain parenthetical:**

1. **The local DB lookup dies outright.** The cache records, for all four proteins:
   `db::dlta::staphylococcus aureus mrsa n315 → status: unmapped, reason:
   species_not_found:Staphylococcus aureus (MRSA N315)`. **The organism string with its strain
   parenthetical matches no PathBank species**, so the DB path contributes nothing.
2. **The API path then falls back without an identifier.** All four cache entries read
   `status: mapped, reason: best_effort_fallback, chosen_rule: best_effort_fallback,
   confidence 0.56` — and **`uniprot: null`, `accession: null`.** It keeps a `resolved_name` and
   ships **no ID**. For DltA that `resolved_name` is
   *"Dihydrolipoyllysine-residue acetyltransferase…"* — **the human P10515**, i.e. candidate #1.

**And every candidate scores exactly 0.56.** With a flat score across all eight, selection is
decided by **API result ordering alone**, and nothing prefers the requested organism. The
species comparator then rejects what ordering picked — correctly, but too late to rescue the
right row that was sitting at position 3.

`protein_missing_external_identity` follows directly from `accession: null`.

## B6. Upstream or downstream stochasticity?

**Both, and they must not be conflated.**

- **The mapping layer is deterministic and stable.** Identical queries, identical candidate
  pools, identical flat 0.56 scores, identical `best_effort_fallback` with no accession, in both
  runs. `DltB`, `DltC`, `DltD` fail the same way every time. **This is `F-185`, and it is not
  random.**
- **Stage-1 extraction is stochastic**, and that is what moved the outcome: it decides *which
  reactions exist*, hence *which proteins are essential*, hence *whether the deterministic
  identity failures are load-bearing*.
- **One genuine downstream difference:** `DltA` ended `species_mismatch` against human P10515 in
  the pilot but `verified_real_protein` against P0C397 in ORCH-730 — **from an identical
  candidate pool**. With all scores tied at 0.56 this is order/tie-break sensitivity, not new
  information.

## B7. Why `core_accepted` went to 0

```
Stage-1 draw produced no DltA-centric reactions
  -> the surviving reactions are DltC- and DltB-centric
     -> DltC and DltB carry accession: null   (best_effort_fallback, DB species_not_found)
        -> all 6 processes quarantined: 4x protein_missing_external_identity
                                        2x undeclared_entity_in_outputs
           -> core_accepted 0, coverage 0.000 < 0.500
              -> "no viable requested-pathway core"
                 -> no payload to gate -> final_gate_report_missing -> fail (correctly, closed)
```

**The pilot passed because its draw happened to depend on the one Dlt protein that resolves.**
Not because identity resolution worked — it failed on `DltB`, `DltC` and `DltD` in the pilot
too. **The pilot's success was one reaction-selection away from this failure the whole time.**

---

# What this changes

- **`F-185` is deterministic; its *impact* is stochastic.** The identity layer fails the same
  way every run. Whether that failure is fatal depends on which reactions Stage 1 happens to
  emit. This is a sharper statement than `ORCH-725` could make and it belongs in the manuscript.
- **The strain parenthetical is now confirmed at TWO seams**, not one: the **PathBank DB species
  lookup** (`species_not_found`) and the **species comparator**. `ORCH-725` Fix 3 addressed the
  API query shape; this shows the DB key fails first, before any API result is judged.
- **Flat 0.56 scoring makes organism-matching a coin flip.** The correct row is in the pool at
  position 3-4 with the same score as the wrong one at position 1.
- **Two observability gaps worth closing before any future measurement**: per-call token usage
  is never persisted, and the **provider/backend behind each call is never recorded** even though
  OpenRouter demonstrably routes the same model to different backends within minutes.
- **Nothing here is a `C-119` defect**, and nothing here changes the merged code.

**No production edits. No new engineering wave. These are registered observations for the
product owner.**

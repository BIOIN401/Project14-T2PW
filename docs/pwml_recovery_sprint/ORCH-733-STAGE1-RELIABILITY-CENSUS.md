# ORCH-733 — Stage-1 delivery reliability. A read-only census.

**2026-09-08.** No production code modified, no paper re-run, no LLM draw, no network call, no
model or budget changed. `main` untouched. Production remains frozen at `045447c8` and
`git diff 045447c8 HEAD -- src/` is **empty**. Protected `streamlit_app.py` unchanged at
`sha256:47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632`.

Tools: `evidence/stage1_census.py`. Reports: `evidence/g11/ORCH-733/01`–`03`, every one
`FINAL SURVIVING COUNT : 0` / `cleanup : success`.

---

# EXECUTIVE SUMMARY

> ## Terminal Stage-1 **delivery** failure is a small stochastic tail: **8 legs in 336 (2.4 %)**, 6 papers, across 14 months of archived runs.
>
> ## But the census found something better than a fix. The escalation that would answer it **already exists in production, is already tested, and has NEVER been issued — because one environment variable is unset.**

**Stage-1 terminal failures are dominated by content, not delivery.** Of **49** terminal Stage-1
failures, only **8** are the provider failing to deliver usable output. The other **41** are legs
where the provider delivered parseable JSON and the pipeline or a guard rejected it. A
retry/alternate-route policy cannot help those, and counting them together would have manufactured
a case for one.

**The headline mechanism finding.** `run_stage_one_with_chunking`'s **rung 3 — "a materially
different strategy (narrower section-based extraction or an alternate model)"** — was **reached 6
times in the whole corpus and issued 0 times.** Every refusal carries the same cause:

```
skip_cause : strategy_not_materially_different
detail     : "same model as an earlier attempt and no narrower scope: a prompt tweak is not a rung"
```

`OPENROUTER_EXTRACTION_FALLBACK_MODEL` is **not set** in `.env` and not in the environment, so
`alternate_model_env_var()` returns `""`, rung 3 falls back to the **same** model, and
`materially_differs` correctly refuses to call that an escalation.

**Is Stage-1 delivery reliability material to PWML yield?** **Marginally, and it is not the next
bottleneck.** 2.4 % of legs, and the remedy for most of them is a configuration change with a
**zero-line production diff**.

---

# 1. Census

**Discovery by shape, reconciled before interpretation.** 336 leg directories matched
`…/papers/<paper>/<mode>` at any depth; all five artifact counts derived independently by a second
traversal matched exactly (`RESULT.txt` 334, `stage1_payload.json` 230, `LEG_TRACE.jsonl` 72,
`extraction_boundary_report.json` 74, `final_mapped.json` 144). The census **refuses to report** if
reconciliation fails. Two legs carry no `RESULT.txt` and are excluded from leg-level rates.

| | |
|---|---:|
| leg directories discovered | **336** |
| with `RESULT.txt` | 334 |
| reaching or past Stage 1 | 332 |
| **terminal Stage-1 failures** | **49** |
| distinct papers affected | 22 |

## Mechanism table

| mechanism | Stage-1 attempts in those legs | terminal legs | papers |
|---|---:|---:|---:|
| **A** empty completion, `finish_reason=length` | 3 | **1** | 1 |
| **B** tiny/degenerate completion, `finish_reason=stop` | 9 | **4** | 4 |
| **C** genuine truncated JSON, `finish_reason=length` | 6 | **2** | 2 |
| **D** malformed JSON, not truncated | 3 | **1** | 1 |
| **E** semantic guard after recovery | — | **0** | 0 |
| **F** provider/API exception | — | **0** | 0 |
| **— provider-DELIVERY subtotal —** | | **8** | **6** |
| **H** valid JSON, required container missing | 4 | 19 | 12 |
| **I** valid extraction, zero processes | 30 | 14 | 3 |
| **J** extraction skipped by a guard | 0 | 7 | 4 |
| **T** leg wall-clock timeout (recorded at stage1, not a delivery failure) | 9 | 1 | 1 |
| **G** other/unclassified | — | **0** | 0 |

Every leg is classified; nothing is parked in "other". The three named exemplars land where the
charter expects: `PMC11487621` → **A** (0 chars ×3, `length`), `PMC7615680` → **B** (2 chars ×2,
`stop`), `PMC8510960` → **C** (9,501 chars, `length`). `PMC12444477/strict` is a second **C** at
40,537 and 60,802 chars.

**The tiny threshold is 200 characters, derived not invented:** observed degenerate completions
measure 2 chars (`PMC7615680`) and 84–1,267 chars (`PMC8510960`'s repair rungs), while the two
genuine truncations carry 9,501 and 10,895. 200 sits above every observed degenerate completion and
two orders of magnitude below any real extraction.

## Attempt level — and why it must not be read as the leg-level story

`LEG_TRACE.jsonl` exists for **72 of 336** legs. Every attempt number below is scoped to those.

| | all stages | **Stage 1 only** |
|---|---:|---:|
| attempts | 2,988 | **116** |
| `finish_reason=stop` | 1,903 | 103 |
| `finish_reason=length` | 1,081 | 12 |
| empty content | 983 | **9** |
| empty **and** `length` | 982 | **9** |
| sizeable **and** `length` (genuine truncation shape) | 74 | **3** |
| tiny content (1–200 chars) | 820 | 10 |

Stage-1 content-size buckets: `>10 000` **30** · `2 001–10 000` **54** · `201–2 000` 13 ·
`1–200` 10 · `0` 9.

> **The empty-completion pathology is overwhelmingly NOT a Stage-1 phenomenon.** It is 982 of 2,988
> attempts corpus-wide (**33 %**) but only 9 of 116 Stage-1 attempts (**7.8 %**). Quoting the
> corpus-wide rate as if it were Stage 1's would overstate the problem more than fourfold.

---

# 2. Current retry behaviour

Read at the frozen production SHA; nothing changed.

| question | answer |
|---|---|
| attempts for an empty completion | **Two layers.** The client retries "HTTP 200 with no text" *inside* its own loop as a transient (`LLM_MAX_RETRIES=3`); the ladder counts that whole thing as **one** attempt. `PMC11487621` shows it exactly: 3 model calls, `attempts_issued: 1`. |
| what triggers a retry | a degenerate-but-parseable payload (`_payload_is_structurally_empty`), a JSON decode error, or an empty-but-successful completion at the client layer |
| treated as transient | empty-but-successful HTTP 200; operation timeouts are recorded and re-raised |
| what causes terminal stop | `identical_prompt_same_model` → `identical_empty_response`; attempt cap 3; `budget_exhausted`; `strategy_not_materially_different` |
| **truncated-JSON larger-token retry at Stage 1** | **NO.** `_default_retry_tokens`, `retry_max_tokens` and `_looks_truncated_json_failure` live only in the Stage-2 inference loop (`pipeline.py:428, 511-513, 567, 578, 674, 712`). ORCH-731 A6, re-verified here. |
| **alternate provider/model route at Stage 1** | **YES — rung 3 exists** (`pipeline.py:4064-4130`) and is *never issued*. See § 3. |
| retry a tiny-but-`stop` completion | yes, via the normal rung, until the identical-prompt rule stops it |
| retry semantic-guard failures | **no, correctly** — the guard declining to fabricate a missing remainder is a refusal, not a transient |
| difference from Stage 2 | Stage 2 has truncation-aware token escalation (16 000 → 24 000); Stage 1 has an alternate-model rung instead. **Neither stage has both.** |

---

# 3. The finding: an escalation that exists and has never run

```
rung 3 reached : 6 legs   (PMC7615680, PMC8510960, PMC12444477 ×2, PMC12856317, PMC12326985)
rung 3 issued  : 0 legs
refusal        : strategy_not_materially_different, all six, identical wording
```

`alternate_model_env_var()` (`extraction_ladder.py:181-203`) returns `""` when
`OPENROUTER_EXTRACTION_FALLBACK_MODEL` is unset **or** resolves to the same model as the primary —
a deliberate C-round-1 correction so two variable names pointing at one model cannot masquerade as
an alternate. It is unset here. So `rung3_env` becomes the primary model, `materially_differs`
sees the same model and no narrower scope, and the rung is refused. **The guard is behaving
exactly as designed; there is simply nothing to escalate to.**

**Class A cannot reach rung 3 at all.** Rung 3 is gated on `saw_empty_payload`, which is set only
by a payload that is *present but degenerate* — a repair that closed braces around nothing, or a
prefix-salvage that yielded an empty object (`pipeline.py:4020-4054`). A **wholly empty** completion
raises `JSONDecodeError` with nothing to salvage, so the flag stays `False` and the ladder
terminates at the normal rung. `PMC11487621`'s report confirms it: `attempts_issued: 1`,
`attempts_remaining: 2`, one `skipped_step` (`rung: normal`), and **no rung-3 entry at all**.

---

# 4. Observability gaps — confirmed, and the seam named

| question | answer |
|---|---|
| is the backend provider available in the SDK response? | **Not captured.** Nothing in `client.py` reads a `provider` attribute; `CompletionDiagnostics` (`client.py:345-368`) has no such field. ORCH-731 A1 confirmed: OpenRouter routed one model to AtlasCloud, GMICloud and Alibaba within minutes, and the archives cannot say which answered. |
| is token usage available? | **Partially read, never persisted.** `_record_usage` (`client.py:118-124`) takes `prompt_tokens` and `completion_tokens` into a **module-level cumulative** dict. **`reasoning_tokens` is not read at all** — the very field ORCH-731 A4 showed can consume most of a budget. |
| does existing code discard it? | Yes. `get_token_stats()` is a process-wide snapshot; no per-call record reaches any artifact. |
| **exact seam for future persistence** | `client.py:685` (and the mirror at `:899`), where `_record_usage(resp)` is called with `resp` still in scope, immediately beside the `CompletionDiagnostics` construction whose `to_dict()` already flows into `LEG_TRACE.jsonl` `model_attempt` rows. Adding `provider` and a per-call usage triple there would reach every existing consumer with no new plumbing. |

**Diagnosis only. Nothing was implemented.**

---

# 5. Candidate fixes, ranked

### 1 — Configure the alternate extraction model. **Zero production diff.**

Set `OPENROUTER_EXTRACTION_FALLBACK_MODEL` to a genuinely different model. Rung 3 then becomes
materially different on the model comparison alone and is issued instead of refused.

* historical legs where it would have been **attempted**: **6** · distinct papers **6**
* covers classes **B, C, D** — every delivery class except A
* implementation size: **one line of `.env`**, no `src/` change, no card, no review
* risk of hiding semantic failures: **none** — rung 3 is reached only after the pipeline has
  already drawn nothing usable twice, and its output goes through the same parse and degeneracy
  checks
* risk of repeated expensive calls: **bounded** — attempt cap 3, `budget_conditional=True`, so it
  is refused outright when the leg deadline is short

> **It cannot be claimed this recovers 6 legs.** It gives each one *one more draw from a different
> model where today it gets none*. Whether that draw succeeds is unmeasured, and no PWML-rate
> improvement is claimed.

### 2 — Let a wholly empty completion reach rung 3. **Small code change. n = 1.**

Set `saw_empty_payload` when the completion is empty, so class A escalates like class B does. One
condition in `pipeline.py`. **Historical impact: 1 leg, 1 paper.** Below the bar on its own; it
becomes worthwhile only as a rider if fix 1 is adopted and a second class-A instance appears.

### 3 — Port Stage-2's truncation-aware token escalation to Stage 1. **NOT recommended.**

Covers class C: **2 legs, 2 papers**. ORCH-731 A4 established that the budget was never the binding
constraint — 16 000 tokens bought 38 462 characters on the same setting while `PMC8510960` stopped
at 9 501, and Stage 2 reached 55 660 characters on the same nominal budget. **Per § 9 of the
charter, "increase max_tokens" is not a valid default, and this census does not overturn that.**

---

# 6. Decision

## MIXED — one subclass is worth acting on, and it is not a production card

**Act on classes B / C / D by configuration, not code:** set
`OPENROUTER_EXTRACTION_FALLBACK_MODEL` to a different model so the already-implemented,
already-reviewed rung 3 stops being refused. **No `src/` change, no unfreeze, no card, no G9
proof required** — production stays frozen at `045447c8`.

**Do not charter class A** (1 leg) or the Stage-1 token escalation (2 legs, and the mechanism is
refuted).

**Do not treat terminal Stage-1 failure as the next bottleneck.** At 8 delivery failures in 336
legs it is a small stochastic tail. The larger Stage-1 population — **41 legs** where the provider
delivered fine and the content was rejected (`H` 19, `I` 14, `J` 7) — is a different question
entirely, about extraction *quality* and guard policy, and it is where the next measurement should
go if PWML yield is the goal.

---

# 7. `F-192` — population unchanged

Not fixed, per instruction. The broader shape-based discovery used here finds **336** legs against
the F-192 census's 144 (that census scanned only legs carrying `final_mapped.json`). Re-checked
against the new discovery: the blocked population is **still 1 leg** (`PMC9544450`,
`runs_validation/c120/2026-09-08_1240`) and the auto-state-removal population is unchanged.
**The new method does not change F-192's counts.** It stays registered and monitored.

---

# 8. Tooling hygiene — fixed-depth run discovery

Measured, not asserted: a fixed-depth `*/papers/*/*` glob finds **333** legs where shape-based
discovery finds **336**. It misses exactly three, all in `runs_validation/c120/2026-09-08_1240` —
**the only nested run family in the repository, created today.**

| file | line | can miss nested families? | produced a currently cited metric? |
|---|---|---|---|
| `evidence/allowlist_generator.py` | 100 | **yes** — `glob("*/papers/*/*/final_mapped.json")` | not affected — see below |
| `evidence/c051c_provenance_probe.py` | 214 | **yes** — same pattern | not affected |
| `evidence/c056b_s0_measure.py` | 69 | **yes** — same pattern | not affected |
| `evidence/c057_lineage_equivalence.py` | 129 | **yes** — keys runs off `leg.parts[-4]`, which shifts under nesting | not affected |
| `evidence/c080_release_flip_probe.py` | 62-67 | **yes** — `os.listdir(base)` then `join(base, run, "papers")` | not affected |

**No currently published metric can have been affected**, because every one of them was computed
before `runs_validation/c120/` existed and it is the only nested family. The defect is latent, not
active.

**Registered, not refactored**, per instruction. The standing note for anyone re-running these
tools: they will silently undercount by 3 legs today, and by more if another nested family is
created. The cheap fix, when someone owns these files, is `rglob("papers/*/*")` plus a
reconciliation against an independent count — which is what `stage1_census.py` does and what the
F-192 census failed to do.

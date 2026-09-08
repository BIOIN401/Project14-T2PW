# ORCH-732 — EXPECTED-WORKING-PWML-SMOKE. Frozen configuration, recorded BEFORE execution.

**2026-09-07.** A practical yield check authorized by the product owner. **Not a benchmark,
not a publication milestone, not a re-score of anything.** No production code is modified.
`main` untouched. Production still frozen at `c7a0663e`.

> ## This is a NEW dataset with its own identity.
> `runs_smoke/2026-09-07_2323` is not `runs_verify/2026-09-06_1425` (the ORCH-724 unseen
> pilot) and not `runs_validation/` (the ORCH-730 post-`C-119` validation). **Its numbers may
> never be merged into either denominator, or quoted as though they were part of one.** The
> pilot and the validation are read-only and are neither superseded nor rescored by anything
> here.

---

# 1. The question

> **Given reasonable pathway papers, does the current product actually write PWML files?**

That is the whole question. It is a product-yield question, not a correctness benchmark.

## What counts as success

**A PWML on disk.** `release_ready` is **NOT** required. `pathway.review_required.pwml`
counts as a usable output when a canonical pathway exists, meaningful connected reactions
exist, the F-179 support floor passes and the file serializes.

**Primary metric:** PWML generation rate on reasonable pathway papers.
**Secondary:** reaction count, obvious completeness, and the reason for each refusal.

Conservative refusal is **not** scored as success here.

---

# 2. Cohort — six papers, strict mode, one run each

| # | paper | requested pathway | organism | why it qualifies |
|---|---|---|---|---|
| 1 | `PMC9544450` | menaquinone biosynthesis | *Escherichia coli* | MenD chemistry stated explicitly: alpha-ketoglutarate + isochorismate to SEPHCHC, then elimination of pyruvate to SHCHC. Named enzyme, named substrates and products. |
| 2 | `PMC4725005` | thiamine biosynthesis | *Escherichia coli* | PRA synthesis routed through IlvA and TrpD. Named enzymes, ordinary cofactor pathway, bounded scope. |
| 3 | `PMC11487621` | coenzyme A biosynthesis | *Bacillus subtilis* | States it elucidated the **complete** pantothenate and CoA pathways, with the enzymes and the transporter identified. Multiple connected reactions in a preferred organism. |
| 4 | `PMC12051542` | riboflavin biosynthesis | *Aquifex aeolicus* | ARAPDP to ARAPD dephosphorylation, reconstituted one-pot with the other four pathway enzymes. Fully annotated pathway, one missing step closed. |
| 5 | `PMC10031235` | L-serine biosynthesis | *Homo sapiens* | The phosphorylated pathway, three steps off glycolysis; PSAT substrates and products named exactly. |
| 6 | `PMC7615680` | coenzyme Q biosynthesis | *Homo sapiens* | The COQ metabolon, COQ3 through COQ9, reconstructed in vitro. Multiple connected reactions. |

**Six is inside the authorized 5–8 band.**

## No overlap, and no control

All six were checked mechanically against the **39** paper ids already used by development,
the gold set, the ORCH-724 pilot and ORCH-730. **Overlap: none.** No known control is
included, because the product owner did not ask for one — the default in the charter is a
fully unseen cohort.

## Why these and not the ones acquisition chose

The first staging pass used **search** lines and let the eligibility screen pick. It produced
a cohort that fails the product owner's own selection rules, which is itself worth recording:

| staged by the screen | verdict |
|---|---|
| `PMC12657337` vitamin K2 production in *E. coli* | **already used** — in the exclusion set |
| `PMC12704756` BioE initiates bacterial biotin synthesis | rejected — *Elizabethkingia* + *Chryseobacterium*, rare pathogens, multi-organism |
| `PMC13361465` biosensor-driven evolution for L-tryptophan | rejected — strain engineering, not pathway mechanism |
| `PMC13181364` PABA analogues as DHPS/DHFR inhibitors | rejected — medicinal chemistry, not a pathway paper |
| `PMC4504715` rewiring of purine metabolism in glioma stem cells | rejected — omics association paper |
| peptidoglycan, bile acid | under-delivered 0/1, stopped at `candidate_ceiling` |

**The eligibility screen does not select for "mechanistic pathway paper."** It screens for
pathway *terms*, so an inhibitor paper and an omics paper score well. The cohort above is
therefore **pinned with scope**, curated from Europe PMC by abstract review against the
stated criteria. `PMC8303990` was also rejected on review — two lactobacilli plus synthetic
operons and knockouts, failing both the multi-organism and the constructs rules.

Scope strings are chosen to match each paper's own language, to avoid the Stage-0
`scope_conflict` that closed a pilot leg. **Scope is mandatory:** a bare pinned id stages a
scopeless, invalid run in which every downstream reader sees an empty requested pathway.

## Staging outcome

`requested 6, examined 6, eligible 6, ineligible 0, no_full_text 0, accepted 6` — zero
skipped, full text 74k–80k characters each.

---

# 3. Token limits — CHOSEN AND STATED, not inherited

| | value | provenance |
|---|---|---|
| **Stage 1 (extraction) `max_tokens`** | **16000** | `OPENROUTER_EXTRACTION_MAX_TOKENS` |
| **Stage 2 (inference) `max_tokens`** | **16000** | `OPENROUTER_INFERENCE_MAX_TOKENS` |
| preprocessor `max_tokens` | 12000 | `OPENROUTER_PREPROCESSOR_MAX_TOKENS` |
| global `LLM_MAX_TOKENS` | 16000 | `.env` |

Stated explicitly because **`F-186`** exists: three different numbers are reachable for these
two settings — the uncommitted helper's own default of 64000, the pre-`F-186` committed
literals of 24000/20000, and the `.env` values of 16000/16000 — so silence here would be a
defect.

**16000 / 16000 is chosen to hold the configuration identical to the ORCH-724 pilot and the
ORCH-730 validation**, so tonight's yield is comparable to the only two datasets that measure
the same thing. Raising the budget would change extraction and confound a yield question with
a budget question.

**`ORCH-731` A4 retired the "raise the Stage-1 budget" reflex on evidence:** our `max_tokens`
is honoured to the token, Stage 2 reached 55,660 characters on the same nominal 16000, and
genuine Stage-1 truncation is **2 events in 2,675 attempts**. The budget was never the binding
constraint. `D-097`'s prohibition on blaming token budget stands.

**Consequence, stated plainly:** these legs carry the same `F-186` handicap. Extraction is, if
anything, **understated**. That is acceptable for a yield check and must not be reported as
the system's extraction ceiling.

---

# 4. Frozen state

## Code

| item | value |
|---|---|
| branch | `sprint/pwml-recovery` |
| repo HEAD at execution | `28cf10a871c6a91f2d47095e0aeec3c0607d6ff1` |
| local / `origin` / `ls-remote` | **all three identical** — verified before staging |
| **frozen production** | **`c7a0663eef43a1329f6e5c5acfb1f6dc3f2e6591`** (`C-119` merge) |
| `streamlit_app.py` CRLF sha256 | `47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632` |
| `streamlit_app.py` content sha256 (LF) | `251122389a2d29e80c157ee139837d06c6f82b7ad6e215d144ecc41b436933bf` |
| `streamlit_app.py` size | 361,433 bytes |
| `streamlit_app.py` git state | **`M` — modified, never staged, never committed** (`D-097`) |

**Byte-for-byte identical to the ORCH-730 freeze.** The reproducible identity of this run is
**(SHA + the `streamlit_app.py` content hash), not the SHA alone** — the `D-097` rule. The
reconstruction bundle is `evidence/repro/ORCH-724/`.

| module | sha256 (16) | status |
|---|---|---|
| `batch/driver.py` | `a1a6c582f4ea3fc1` | `C-119` seams 1–2 — unchanged |
| `pipeline/release_status.py` | `4d57b5078d1d7c53` | `C-119` seam 3 — unchanged |
| `pipeline/reaction_support.py` | `117e8e21a1541f37` | **F-179, untouched** |
| `pipeline/stage_contracts.py` | `66a6cd16a2e5ae43` | unchanged |
| `rag/admission.py` | `e45b2b1ebb27777c` | `C-118` template — unchanged |

All five match ORCH-730 exactly.

## Model / provider

```
LLM_PROVIDER    = openrouter
BASE_URL        = https://openrouter.ai/api/v1
model (all 9 roles, identical)
                = deepseek/deepseek-v4-flash
LLM_TEMPERATURE = 0
LLM_MAX_RETRIES = 3
```

**Temperature 0 is not determinism.** The standing sprint trap: identical legs give materially
different Stage-1 draws at temperature 0. `ORCH-731` Part B is the sharpest case on record —
the same paper, the same four proteins extracted, and a PWML on one run and `core_accepted 0`
on the next, purely from which reactions the draw happened to emit.

**The backend behind `deepseek/deepseek-v4-flash` is not recorded.** OpenRouter demonstrably
routed the same model to AtlasCloud, GMICloud and Alibaba within minutes (`ORCH-731` A1). Per-
call token usage is not persisted either. **Both remain open observability gaps** and neither
is closed tonight — no new engineering wave was opened for this run.

## Prompt hashes

| sha256 (16) | file |
|---|---|
| `9787fe422fdade3e` | `src/t2pw/curation/prompt_payload_keys.py` |
| `913b8f1379df3a19` | `src/t2pw/tools/pathwhiz_converter/prompts.py` |

The Stage-0/1/2 prompts are built inline in `streamlit_app.py`, pinned by its two hashes above.

## Retrieval / index

```
RAG_ENABLED            = true
RAG_VECTOR_BACKEND     = memory
RAG_INDEX_DIR          = data/rag_index
RAG_EMBEDDING_MODEL    = text-embedding-nomic-embed-text-v1.5
RAG_EMBEDDING_BASE_URL = http://127.0.0.1:1234/v1   (local LM Studio)
RAG_ACQUIRE_MAX_PAPERS = 20
RAG_SELECT_MAX_PAPERS  = 8
RAG_RETRIEVE_TOP_K     = 8
RAG_EXTRACT_REACTIONS  = true
```

**519 files** under `data/rag_index` at this freeze, against **475** at the ORCH-730 freeze.
The acquire cache grew through ORCH-730's own legs. Recorded rather than reset: resetting it
would change retrieval behaviour relative to both prior datasets.

**LM Studio verified reachable before execution** — `text-embedding-nomic-embed-text-v1.5` is
loaded and served at `127.0.0.1:1234`.

## Secrets

`*_API_KEY`, `LOCAL_API_KEY` and `RAG_EMBEDDING_API_KEY` are present in `.env` and are
**excluded by allowlist** — only keys on an explicit safe list are printed above, so a secret
added later cannot leak into this record by default.

---

# 5. Execution

```
bounded_run.py --timeout <wall> --label smoke-6legs --heavy-lock ORCH-732
               --json evidence/g11/ORCH-732/03-smoke-run.json
  -- python -u scripts/batch_run.py
       --topics topics_expected_working_smoke.txt
       --modes strict --out runs_smoke --timeout 3600 --deadline 8
```

Output tree `runs_smoke/` is **new**, so `runs/`, `runs_verify/` and `runs_validation/` are
untouched. Bounded wrapper, one heavy job, heavy lock held as `ORCH-732`, no detached
processes. Per-leg ceiling 3600 s; the pilot's observed range was 20–50 minutes per leg.

**Strict mode only.** Research is diagnostic by design (`acceptance.py:125`) and cannot produce
a PWML, so it would burn budget for no deliverable.

## NO FAVOURABLE-DRAW RETRIES

Each leg runs **exactly once**. A leg is re-run only for an objectively invalid infrastructure
execution — the ORCH-724 rule, unchanged. **If a leg fails, the blocker is diagnosed and
reported, and the leg is NOT re-run to seek a better draw.** A failure is a result.

## Pre-existing processes

Two `python.exe` processes were running before this job: PIDs 252336 and 323592, both the VS
Code isort language server, user-owned. They are **reported and never killed**. Cleanup targets
only PIDs this job created. `taskkill /IM python.exe` and `pkill python` remain forbidden.

---

# 6. What will be reported per leg

paper · organism · pipeline completed · final canonical reaction count · **F-179 verdict** ·
final live Stage-3 / pre-export verdict · **`C-119` superseded-report disposition** · release
state · **PWML yes/no** · exact path and size · `review_required` vs `release_ready` · one-line
blocker if no PWML.

If several PWMLs are produced they are inventoried for PathWhiz manual review. If yield stays
poor, the blockers are aggregated **by mechanism** and the single dominant one identified —
**no code change is proposed for five unrelated edge cases**, and a benchmark failure does not
by itself justify one.

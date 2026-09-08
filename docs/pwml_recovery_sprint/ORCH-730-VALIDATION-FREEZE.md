# ORCH-730 — post-`C-119` PWML validation. FROZEN CONFIGURATION, recorded before execution.

**Authorized by the product owner, 2026-09-07, as `F-187` option 2.** This is a
**separately-labelled post-fix validation**, **NOT** a re-run or re-score of `ORCH-724`.

> ## The `ORCH-724` unseen pilot stands unchanged.
> `runs_verify/2026-09-06_1425` is read-only and is **not** superseded, amended or rescored by
> anything here. Its numbers — 20 legs, 1 PWML, 72 reactions, `F-185`, `F-186` — remain the
> pilot's observations. **These three legs are a new dataset with its own identity** and may
> never be merged into the pilot's denominators or quoted as if they were part of it.

---

# 1. Scope — exactly three strict legs, no research, no retries

| paper | requested pathway | organism | why |
|---|---|---|---|
| `PMC7232280` | molybdenum cofactor biosynthesis | *Neurospora crassa* | recovered by `C-119`; must now serialize |
| `PMC8510960` | terpenoid indole alkaloid biosynthesis | *Catharanthus roseus* | recovered by `C-119`; must now serialize |
| `PMC12071552` | wall teichoic acid D-alanylation | *Staphylococcus aureus* | **known positive control** — shipped a PWML in the pilot |

**Strict mode only.** Research is diagnostic by design (`acceptance.py:125`) and cannot
produce a PWML; running it would burn budget for no deliverable.

**NO FAVOURABLE-DRAW RETRIES.** A leg is re-run only for an objectively invalid
infrastructure execution — the `ORCH-724` rule, unchanged. **If a leg fails, the new blocker
is diagnosed and reported, and the leg is NOT re-run to seek a better draw.** A failure here is
a result, not a problem to be re-rolled away.

---

# 2. The token limits — CHOSEN AND STATED, not inherited

| | value | provenance |
|---|---|---|
| **Stage 1 (extraction) `max_tokens`** | **16000** | `OPENROUTER_EXTRACTION_MAX_TOKENS` |
| **Stage 2 (inference) `max_tokens`** | **16000** | `OPENROUTER_INFERENCE_MAX_TOKENS` |
| preprocessor `max_tokens` | 12000 | `OPENROUTER_PREPROCESSOR_MAX_TOKENS` |
| global `LLM_MAX_TOKENS` | 16000 | `.env` |

**These were chosen deliberately, and the reasoning is on the record because `F-186` exists.**

The working-tree `streamlit_app.py` carries the `D-097`-pinned, uncommitted `_bounded_env_int`
helper. Its own committed default is **64000**, bounded `[500, 128000]`; the `.env` supplies
**16000/16000**; the pre-`F-186` literals in the committed file were **24000/20000**. Three
different numbers are reachable, so silence here would be a defect.

**16000 / 16000 is chosen because this validation asks whether `C-119` serializes these
payloads — not whether a larger budget extracts more.** The pilot ran at 16000/16000, so this
is the budget under which the payloads `C-119`'s regressions were derived from were produced.
Raising it would change the extraction and confound the two questions: a different pathway
would come out, and a pass would no longer be attributable to `C-119`.

**Consequence, stated plainly:** these legs carry the same `F-186` handicap the pilot did.
Extraction is, if anything, **understated**. That is acceptable — and correct — for a
serialization validation, and it must not be reported as a measurement of the system's
extraction ceiling.

---

# 3. Frozen state

## Code

| item | value |
|---|---|
| repo HEAD at execution | `d4cd940738eff888c5ec744110ce7733572383e4` |
| **frozen production** | **`c7a0663eef43a1329f6e5c5acfb1f6dc3f2e6591`** (`C-119` merge) |
| `streamlit_app.py` CRLF sha256 | `47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632` |
| `streamlit_app.py` content sha256 (LF-normalized) | `251122389a2d29e80c157ee139837d06c6f82b7ad6e215d144ecc41b436933bf` |
| `streamlit_app.py` size | 361,433 bytes |
| `streamlit_app.py` git state | **`M` — modified, never staged, never committed** (`D-097`) |
| `batch/driver.py` | `a1a6c582f4ea3fc1…` (`C-119` seams 1–2) |
| `pipeline/release_status.py` | `4d57b5078d1d7c53…` (`C-119` seam 3) |
| `pipeline/reaction_support.py` | `117e8e21a1541f37…` — **F-179, untouched** |
| `pipeline/stage_contracts.py` | `66a6cd16a2e5ae43…` |
| `rag/admission.py` | `e45b2b1ebb27777c…` (`C-118` template) |

**The reproducible identity of this run is (SHA + the `streamlit_app.py` content hash), not the
SHA alone** — the `D-097` rule, and it applies here exactly as it applied to the pilot. The
reconstruction bundle is `evidence/repro/ORCH-724/`.

## Model / provider

```
LLM_PROVIDER   = openrouter
BASE_URL       = https://openrouter.ai/api/v1
model (all 9 roles, identical)
               = deepseek/deepseek-v4-flash
LLM_TEMPERATURE = 0
LLM_MAX_RETRIES = 3
```

Identical to the pilot. **Temperature 0 is not determinism** — the sprint's standing trap:
identical legs give materially different Stage-1 draws at temperature 0.

## Prompt hashes

| sha256 (16) | file |
|---|---|
| `9787fe422fdade3e` | `src/t2pw/curation/prompt_payload_keys.py` |
| `913b8f1379df3a19` | `src/t2pw/tools/pathwhiz_converter/prompts.py` |

The Stage-0/1/2 prompts are constructed inline in `streamlit_app.py`, whose two hashes above
pin them.

## Retrieval / index

```
RAG_ENABLED            = true
RAG_VECTOR_BACKEND     = memory
RAG_INDEX_DIR          = data/rag_index   (475 files present at freeze)
RAG_EMBEDDING_MODEL    = text-embedding-nomic-embed-text-v1.5
RAG_EMBEDDING_BASE_URL = http://127.0.0.1:1234/v1   (local LM Studio)
RAG_ACQUIRE_MAX_PAPERS = 20
RAG_SELECT_MAX_PAPERS  = 8
RAG_RETRIEVE_TOP_K     = 8
RAG_EXTRACT_REACTIONS  = true
```

## Secrets

`*_API_KEY`, `LOCAL_API_KEY` and `RAG_EMBEDDING_API_KEY` are present in `.env` and are
**excluded by allowlist** — the dump above prints only keys on an explicit safe list, so a new
secret added later cannot leak into this record by default.

---

# 4. What will be reported per leg

pipeline completion · final canonical reaction count · **F-179 verdict** · final live
Stage-3 / pre-export verdict · **`C-119` superseded-report disposition** · release state ·
whether a PWML was written · exact path and size · `review_required` vs `release_ready` ·
any new blocker.

---

# 5. Process

Output directory `runs_validation/` — **a new tree**, so `runs/` and `runs_verify/` are
untouched. Bounded wrapper, one heavy job, no detached processes. Expected duration ~1–2.5 h
for three strict legs at the pilot's observed 20–50 min each.

*No production code is modified by this task. `main` untouched. The pilot is read-only.*

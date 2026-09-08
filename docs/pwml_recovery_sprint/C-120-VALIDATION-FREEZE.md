# C-120 — POST-MERGE PWML VALIDATION. Frozen configuration, recorded BEFORE execution.

**2026-09-08.** A separately labelled validation of the `C-120` identity-resolution merge,
required by the card § 12. **Not a benchmark. Not a re-score of anything.** No production code
is modified by this run. `main` untouched.

> ## A NEW dataset with its own identity.
> `runs_validation/c120/<stamp>`. **Not** the ORCH-724 pilot (`runs_verify/2026-09-06_1425`),
> **not** the ORCH-730 validation, **not** the ORCH-732 smoke (`runs_smoke/2026-09-07_2323`).
> All three remain read-only. **None of their denominators may absorb these numbers**, and
> nothing here rescores or supersedes any of them.

---

# 1. The question

> **Do the two identity failures `C-120` was chartered from now produce PWML — and does a
> paper that already worked still work?**

Deterministic proof that the two blocking predicates clear was completed **before** this run
(§ 4 below). This run answers the only thing the archives cannot: whether a whole leg reaches
a file on disk.

## What counts as success

**A PWML on disk.** `release_ready` is **NOT** required and is not sought.
`pathway.review_required.pwml` is a success when a canonical pathway exists, meaningful
connected reactions exist, the `F-179` support floor passes and the file serializes.

**The control is the leg that matters most.** `PMC9544450` produced a PWML in ORCH-732 under a
configuration identical to this one except for the `C-120` diff. **If the control regresses,
that is the finding of this run**, and it outranks any recovery on the other two.

---

# 2. Cohort — three papers, strict mode, ONE run each

| # | paper | pathway | organism | role | ORCH-732 outcome |
|---|---|---|---|---|---|
| 1 | `PMC11487621` | coenzyme A biosynthesis | *Bacillus subtilis* | mechanism A | **NO PWML** — 11 reactions, export blocked on `species_missing_taxonomy` + `species_missing_classification` @ `/entities/species/1` (`B. subtilis`) |
| 2 | `PMC10031235` | L-serine biosynthesis | *Homo sapiens* | mechanism B | **NO PWML** — 6 reactions, `blocking_issues = 1`: `gate.protein_psat_is_missing_a_uniprot_or_drugbank_identifier` @ `/entities/proteins/1` |
| 3 | `PMC9544450` | menaquinone biosynthesis | *Escherichia coli* | **CONTROL** | **PWML, 39,686 B, `review_required`** |

`topics_c120_validation.txt`, pinned with scope. Scope is mandatory — a bare pinned id stages a
scopeless run in which every downstream reader sees an empty requested pathway.

## `PMC7615680` is deliberately EXCLUDED

Card § 13. It failed on a degenerate **2-character** Stage-1 completion with
`finish_reason=stop` — a complete, well-terminated, essentially empty response, twice. That is a
provider/LLM reliability class, **not** truncation and **not** a token-budget event. `C-120`
does not touch Stage 1 and this run must not confound the two.

---

# 3. Frozen state

## Code

| item | value |
|---|---|
| branch | `sprint/pwml-recovery` |
| integration tip at freeze | `4d78c295d98746e1d4315736c0cfcf49fc33473f` |
| local / `origin` / `ls-remote` | **all three identical** — verified before this file was written |
| **production SHA (the `C-120` merge)** | **`045447c86f1a0288ce87c444de4854b2eb2ef943`** |
| previous frozen production | `c7a0663e` (`C-119`) — superseded by the above |
| `streamlit_app.py` CRLF sha256 | `47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632` |
| `streamlit_app.py` content sha256 (LF) | `251122389a2d29e80c157ee139837d06c6f82b7ad6e215d144ecc41b436933bf` |
| `streamlit_app.py` size | 361,433 bytes |
| `streamlit_app.py` git state | **`M` — modified, never staged, never committed** (`D-097`) |

**Both `streamlit_app.py` hashes are byte-for-byte identical to the ORCH-730 and ORCH-732
freezes.** The reproducible identity of this run is **(SHA + the content hash), not the SHA
alone** — the `D-097` rule.

| module | sha256 (16) | status |
|---|---|---|
| `mapping/map_ids.py` | `f21066b2097c1b17` | **MOVED by `C-120`** (was `3c9e4402401fb205`) |
| `pwml/ir.py` | `41a8d68295077cf3` | **MOVED by `C-120`** (was `2ac2d9cbceb445df`) |
| `batch/driver.py` | `a1a6c582f4ea3fc1` | `C-119` seams 1-2 — **unchanged** |
| `pipeline/release_status.py` | `4d57b5078d1d7c53` | `C-119` seam 3 — **unchanged** |
| `pipeline/reaction_support.py` | `117e8e21a1541f37` | **`F-179`, untouched** |
| `pipeline/stage_contracts.py` | `66a6cd16a2e5ae43` | unchanged |
| `rag/admission.py` | `e45b2b1ebb27777c` | `C-118` template — unchanged |

The five unchanged modules match the ORCH-730 and ORCH-732 freezes exactly. **Exactly two
modules moved, and they are the two `C-120` owns.**

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
different Stage-1 draws at temperature 0. `ORCH-731` Part B is the sharpest case on record — the
same paper, the same four proteins, a PWML on one run and `core_accepted 0` on the next, purely
from which reactions the draw emitted. **A single leg here can therefore neither confirm nor
refute `C-120` on its own**, and the report will say so.

The backend behind `deepseek/deepseek-v4-flash` is not recorded and per-call token usage is not
persisted. Both remain open observability gaps; neither is closed here.

## Token limits — CHOSEN AND STATED, not inherited

| | value | provenance |
|---|---|---|
| **Stage 1 (extraction) `max_tokens`** | **16000** | `OPENROUTER_EXTRACTION_MAX_TOKENS` |
| **Stage 2 (inference) `max_tokens`** | **16000** | `OPENROUTER_INFERENCE_MAX_TOKENS` |
| preprocessor `max_tokens` | 12000 | `OPENROUTER_PREPROCESSOR_MAX_TOKENS` |
| global `LLM_MAX_TOKENS` | 16000 | `.env` |

Stated explicitly because **`F-186`** exists: three different numbers are reachable for these
two settings. **16000/16000 holds the configuration identical to ORCH-724, ORCH-730 and
ORCH-732**, so the control leg is comparable to its own prior result. Raising the budget would
confound an identity question with a budget question. `D-097`'s prohibition on blaming token
budget stands.

## Prompt hashes

| sha256 (16) | file |
|---|---|
| `9787fe422fdade3e` | `src/t2pw/curation/prompt_payload_keys.py` |
| `913b8f1379df3a19` | `src/t2pw/tools/pathwhiz_converter/prompts.py` |

Both unchanged from ORCH-732. The Stage-0/1/2 prompts are built inline in `streamlit_app.py`,
pinned by its two hashes above.

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

**530 files** under `data/rag_index` at this freeze, against **519** at the ORCH-732 freeze and
**475** at ORCH-730. The acquire cache grew through ORCH-732's own legs. **Recorded rather than
reset** — resetting it would change retrieval behaviour relative to every prior dataset.

## Species taxonomy lookup — NEW, and load-bearing for this run

```
T2PW_SPECIES_NCBI = unset -> defaults to enabled (map_ids.py:9118)
```

**Mechanism A's recovery of `PMC11487621` depends entirely on a reachable NCBI E-utils
service.** In that payload both donor rows are rank-qualified (`Bacillus subtilis 168` @ `1423`
and `Bacillus subtilis (strain 168)` @ `224308`), so the offline tier is not eligible and the
abbreviation resolves only through tier 2's lookup of the expanded binomial. This dependence
**pre-dates `C-120`** — those two donors already disagreed before the merge — but it is stated
here because a network failure would present as an identity failure and must not be read as one.

## Secrets

`*_API_KEY`, `LOCAL_API_KEY` and `RAG_EMBEDDING_API_KEY` are present in `.env` and excluded by
allowlist — only keys on an explicit safe list are printed above, so a secret added later cannot
leak into this record by default.

---

# 4. Deterministic proof completed BEFORE this run — card § 9

`evidence/c120_archived_replay.py`, cleanup report `evidence/g11/C-120/38-archived-replay-v2.json`.
No LLM, no live network, no guard bypassed, no run directory written.

**`PMC11487621`.** Replaying `backfill_species_taxonomy` over the archived species section with
a stub serving the answer *this same run already recorded*:

```
esearch "B. subtilis[Scientific Name]"        -> miss   (pre-existing attempt, runs FIRST)
esearch "B. subtilis"                         -> miss   (pre-existing attempt)
esearch "Bacillus subtilis[Scientific Name]"  -> 1423   (C-120: the EXPANDED binomial)
efetch  1423
```

`B. subtilis` -> `taxonomy_id 1423`, `Prokaryote`, `source: pathway_alias_expansion_ncbi`,
`donors: ["Bacillus subtilis (strain 168)", "Bacillus subtilis 168"]`. The organism variants
converge on the species-rank taxon and **neither donor's id was copied**. On the archived
payload the required-field gate goes from

```
species_missing_taxonomy        /entities/species/1
species_missing_classification  /entities/species/1
```

to **`[]`**. Display text is unchanged; `row["name"]` is still `B. subtilis`.

**`PMC10031235`.** `verify_real_protein_identity` on the archived candidate now returns
`verified: true`, `verified_real_protein`, `Q9Y617`, all six rungs `ok` with
`name: keep / gene_symbol_family_identity`. On the reconstructed pre-fallback payload
`protein_missing_external_identity @ /entities/proteins/1` **disappears**.

### The limitation, stated rather than buried

A full export replay is **not** possible from archived payloads alone, so the above proves the
**mapping and gate predicates**, not a written file. Two specific gaps:

1. The archived `final_mapped.json` is the payload *after* the Unknown fallback fired, so the
   pre-fallback state had to be reconstructed. Reconstructing it leaves four
   `reaction_enzyme_must_be_protein_complex` findings at `/processes/reactions/1` and `/2`
   which are **artifacts of the reconstruction** — a bare resolved protein standing where
   production would have built a real enzyme complex. They are **not** a prediction about the
   paper, and they are **not** evidence that it will export.
2. Whether a now-resolved `PSAT` wraps into a valid enzyme complex is exactly the
   `reaction_enzyme_must_be_protein_complex` class `ORCH-725` recorded against `PMC11405693`.
   **Only a live leg settles it.** That is this run.

---

# 5. Execution

```
bounded_run.py --timeout <wall> --label c120-validation --heavy-lock C-120
               --json evidence/g11/C-120/<seq>-c120-validation.json
  -- python -u scripts/batch_run.py
       --topics topics_c120_validation.txt
       --modes strict --out runs_validation/c120 --timeout 3600 --deadline 8
```

Output tree `runs_validation/c120` is **new**, so `runs/`, `runs_verify/`, `runs_smoke/` and the
existing `runs_validation/` contents are untouched. Bounded wrapper, one heavy job, heavy lock
held as `C-120`, no detached processes. Per-leg ceiling 3600 s; ORCH-732's observed range for
these three papers was 22-54 minutes.

**Strict mode only.** Research is diagnostic by design (`acceptance.py:125`) and cannot produce a
PWML.

## NO FAVOURABLE-DRAW RETRIES

Each leg runs **exactly once**. A leg is re-run only for an objectively invalid infrastructure
execution — the ORCH-724 rule, unchanged. **If a leg fails, the blocker is diagnosed and
reported, and the leg is NOT re-run to seek a better draw.** A failure is a result.

---

# 6. What will be reported per leg

Card § 12: Stage-1 result · final canonical reaction count · identity mappings (specifically:
does `B. subtilis` carry a taxonomy id, and does `PSAT` carry `Q9Y617` rather than `Unknown`) ·
final live gates · `F-179` verdict · **PWML yes/no** · exact output path · bytes · release
disposition · remaining blocker.

Plus, explicitly: **whether the control still produces a PWML**, and whether any leg reached
`release_ready` — which it must not do merely because an identity was recovered.

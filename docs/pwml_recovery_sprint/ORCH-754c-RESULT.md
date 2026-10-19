# ORCH-754c — production prompt promoted to the measured control `r2_01_control_v03`

**Status:** delivered on branch `orch754/c-prompt-v03`, base `f9892cdf`. Not merged, not
pushed. Needs independent review of the diff.
**Authority:** product-owner ruling 2026-10-19, to be recorded as **D-104** by the product
owner (not appended here; `DECISIONS.md` is product-owner only).
**No LLM call and no pipeline run** was made for this task (product owner forbade it).
Every bounded child ran with `T2PW_OFFLINE_CURATOR=1`.

**G9 label: DELIBERATE BEHAVIOUR CHANGE.** Not a correction, not new functionality.

---

## 1. The change

`src/t2pw/llm/prompts/pwml_system.txt` is now a byte copy of
`docs/evaluation_v2/prompt_variants_r2/r2_01_control_v03.txt`.

The diff is one hunk, **18 inserted lines after line 225, nothing removed**: a blank line
and the 17-line block beginning `ONE CLAIM, ONE RESOLUTION — finest wins.` It ends with
*"If you cannot tell whether two rows are one claim, keep the finer row and drop the
coarser."* Confirmed independently before editing with
`diff <(tr -d '\r' < pwml_system.txt) <(tr -d '\r' < r2_01_control_v03.txt)` → `225a226,243`.

| | base `f9892cdf` (variant `v02_working_tree_resolution`) | tip (variant `r2_01_control_v03` = `v03_resolution_precedence`) |
|---|---|---|
| working-tree bytes (CRLF) | 37,044 B, 562 lines, sha256 `0a1adddfe72655cf96d76263fbfc88d86d8e6a3a21d533ea85a58bca6192a2c9` | 38,443 B, 580 lines, sha256 `bc716123774c3d8bba1e69a4e11634d9c464adffd48d139016723f1fdd6d0ce3` |
| text the model receives (LF, `read_text`) | sha256 `2108aed08a899a02c09328e31b05afbc050d66085bf40220f6e1426c06c703b8` | sha256 `27f5e8732d77083313cab590dd6259ffe3828816647b0795699a0da6a68b8156` |
| git blob (LF) | = LF text hash above | = LF text hash above (verified `git show HEAD:… \| sha256sum`) |

No BOM in either. The file's line endings are not changed by this task: like every tracked
text file without a `.gitattributes` entry, it is LF in the object store and CRLF on disk via
the machine's `core.autocrlf=true` (`git ls-files --eol`: `i/lf w/crlf`, both files).

Only Stage 1 reads this file: `src/t2pw/pipeline/pipeline.py:280` (the extraction stage)
and `src/t2pw/extraction/extract.py:6` (the legacy `SYSTEM` constant, re-exported by
`src/extract.py`). Both resolve it through `t2pw.paths.PROMPTS_DIR`.

---

## 2. What `r2_01_control_v03` was measured to do

### 2.1 The exact text was measured, not a near copy

`scripts/prompt_ab.py:213` records `prompt_sha256` over the prompt text it sends. Over the
local run outputs under `C:/t/` (uncommitted; the docs that score them are committed), the
production tip's LF hash `27f5e873…` appears in **41 draws**:

| round | where recorded | draws with `27f5e873…` |
|---|---|---|
| round 1 (ORCH-747 § 2) | `C:/t/prompt-grid/final/index.json` | 1 |
| round 2 (§ 10) | `C:/t/prompt-grid/r2/` | 3 (one per paper) |
| round 3 (§ 11) | `C:/t/prompt-grid/r3/` | 6 |
| round 4 (§ 12) | `C:/t/prompt-grid/r4/` | 10 |
| ORCH-751 cohort | `C:/t/cohort/*/index.json` | 21 |

So production now sends exactly the text those draws sent. The base production hash
`2108aed0…` appears in exactly **one** recorded draw (round 1).

### 2.2 Scored results for this prompt (certified gold)

Every number below is copied from the cited section; matched reactions per draw.

| source | enterobactin PMC12096016 (5 gold) | menaquinone PMC12421875 (9 gold) |
|---|---|---|
| round 2, `ORCH-747-RESULT.md` § 10.2 | 4 (F1 0.889) | 9 (F1 1.000) |
| round 3, § 11.1 | 3, 5, 4 | 9, 7, 4 — mean F1 0.855 over both papers, worst draw 0.615 |
| round 4, § 12.1 | 5, 5, 4, 5, 5 → F1 0.978 | 4, 7, 4, 9, 7 → F1 0.796 — overall mean F1 0.887 |
| ORCH-751 cohort, `ORCH-751-RESULT.md` § 2.1 | 4, 4, 4 matched (claimed 4, 5, 5) | 2, 9, 7 |

Aggregated by me from those per-draw rows (not a figure any of those docs states):
enterobactin 52 of 60 gold-reaction matches over 12 draws (mean recall 0.867); menaquinone
78 of 108 over 12 draws (mean recall 0.722).

What these establish:

1. **No variant tested beat it.** Round 4's head-to-head against the best challenger
   `r2_13_model_stated_example`: mean F1 0.887 vs 0.916, difference 0.029 against a pooled
   SD of 0.135, i.e. 0.21 SD at 10 draws per arm (§ 12.1). ORCH-747 § 12.4 item 1
   recommends keeping `v03` / `r2_01_control_v03`.
2. **Evidence discipline.** Zero rows without a verbatim evidence quote across all 104
   ORCH-747 draws (§ 12.5) and all 21 ORCH-751 draws (§ 2, column "rows without
   evidence"). This holds for every variant, so it is a property of the prompt family and
   the model, not something this block bought.
3. **Relationship vocabulary.** 0 out-of-vocabulary relationship values on the round-1
   draw (§ 2). Introduced by v02 and inherited unchanged, so v02 already had it.

### 2.3 What the 18-line block itself was measured to do vs v02 (production at base)

**One draw, one paper, no certified gold.** Round 1, PMC319880 (araC autoregulation),
`ORCH-747-RESULT.md` § 2:

| | v02 (base production) | v03 (tip production) |
|---|---|---|
| interactions | 13 | 11 |
| restatement pairs (§ 2 heuristic) | 1 | **0** |
| DNA entities | 16 | 9 |
| bad vocabulary / reactions / rows with no evidence | 0 / 0 / 0 | 0 / 0 / 0 |

That is the whole of the direct v02-vs-v03 evidence. § 1 calls the margin "one row" and
§ 2 says "n=1".

---

## 3. What was NOT shown — read before relying on this promotion

1. **v03 was never shown better than v02.** Rounds 2–4 and ORCH-751 contain no v02 arm
   (`VARIANTS_R2.md` lines 6–10: all round-2 variants derive from `r2_01`, round 1's
   baseline is not re-run). The only comparison is § 2.3 above, n=1, on a paper whose gold
   is not certified, on row counts not F1. ORCH-747 § 12.4 item 4 sets the bar for any
   prompt claim at ≥3 draws per arm and 5 to separate near-neighbours; v03 vs v02 has 1
   draw per arm. **The justification for this change is that production now matches what
   was measured, not that it is measured to be better.**
2. **The block's effect on metabolic papers is unmeasured against v02.** `VARIANTS.md`
   § v03 predicts it "near-inert" there; no draw tested that prediction against v02.
3. **The graph and the PWML are unmeasured.** `VARIANTS.md` § v03 "Risk if wrong": keeping
   the finer row (a site) over the coarser one (a gene) may trade resolvable rows for
   unresolvable ones and *increase* orphan nodes in `draft_graph` (AUDIT.md D7). Every
   score above is on the Stage-1 payload. No pipeline leg, mapping or export ran under
   v03 as production.
4. **It does not fix the regulatory paper.** All 16 round-2 variants, the control included,
   scored recall 0 on PMC319880's single gold reaction (§ 10.3). ORCH-751's two regulatory
   conditions without gold (Ara operon, sour taste) produced 0 reactions in every draw.
   `D-open-2` (is a transcription step a reaction) is still open.
5. **It does not prevent bad single draws.** The control's own worst draws: 4 of 9
   menaquinone (round 3, round 4) and **2 of 9** (ORCH-751). The ±0.25 F1 single-draw noise
   floor (§ 11.1) applies to it in full.
6. **Precision is not perfect.** ORCH-751 enterobactin: claimed 4, 5, 5, matched 4, 4, 4,
   so two draws each carried one unmatched row. § 11.3 recorded 2 of 36 round-3 draws
   (all variants) below precision 1.000.
7. **Every measurement was at 64,000 output tokens, temperature 0.** ORCH-747 § 3 records
   that the primary checkout's `.env` sets `OPENROUTER_EXTRACTION_MAX_TOKENS=16000` (not
   re-read here; the worktree has no `.env`), at which round 1 failed
   9 of 10 variants with empty bodies (§ 3, `D-open-1`). Nothing here says how v03 behaves
   at 16,000. The prompt is 1,399 bytes longer, which costs input tokens, not output.

---

## 4. Pins, hashes and what moves

**No pin moved.** Searched `tests/`, `src/`, `scripts/`, `streamlit_app.py` and
`docs/pwml_recovery_sprint/evidence/` for `pwml_system`, the base sha256 (`0a1adddf…`, and
its LF form `2108aed0…`) and any prompt-hash/digest/fingerprint pin. Nothing hard-codes the
prompt's content or hash. The only test that reads it, `test_stage_one_ensemble.py:173`,
compares the sent system prompt to the file read at runtime, so it is self-updating.

**One pin added, deliberately.** `tests/test_orch754c_production_prompt_is_measured_control.py`
pins `MEASURED_TEXT_SHA256 = 27f5e873…` (the LF text hash). From now on, **any edit to the
production prompt fails this test** until the pin is moved with a documented old → new
delta under merge rule 4. That is intended: it stops an unmeasured prompt reaching
production silently. It is also friction on every future prompt change (§ 7, Q2).

Why the LF hash and not the dispatch's `bc716123…`: the CRLF bytes exist only because of
this machine's `core.autocrlf=true`, so a clone with `autocrlf=false` would fail a CRLF pin
with nothing wrong. The LF hash is the blob, is what `read_text` hands the model, and is
the value `prompt_ab.py` recorded for every measured draw (§ 2.1).

**What keys on the prompt and therefore moves (none of it is pinned):**

| component | effect |
|---|---|
| `client._hash(messages)` → LEG_TRACE `model_attempt.request_hash` (`src/t2pw/llm/client.py:649`) | every Stage-1 request hash differs from a pre-754c run. Cross-run request-hash comparisons across this commit will not match, by design |
| `_ladder_request_hash` (`pipeline.py:3497`), used by `extraction_ladder.prompt_already_sent` / `prompt_only_ever_returned_nothing` | value changes; logic compares within one leg only, so behaviour is unchanged |
| `repro.py` `prompt_hash` | is `request_hash` (`repro.py:52-54`), same as row 1 |
| `scripts/prompt_ab.py --variant NAME=git:<rev>` | a revision before this commit now yields v02, after it v03 |
| LLM response cache | none exists keyed on the prompt; `client.py` has no response cache |

**Stale line citations (comments only, not changed — outside the boundary):**

| citation | where | state |
|---|---|---|
| `pwml_system.txt:401` ("only create if at least one input AND one output") | `src/t2pw/pipeline/stage_one_boundary.py:215`, `:299`; `tests/test_stage_one_boundary_lineage.py:177` | **moved by this change**: now line **419** |
| `pwml_system.txt:116` (provenance values) | `stage_one_boundary.py:205`, `:299`; `test_stage_one_boundary_lineage.py:177` | **already stale at base**: the provenance list is line 121 at base and tip; 116 is the verb-form rule |
| `pwml_system.txt:291` (scope taxonomy) | `tests/test_scope_membership_filter_end_to_end.py:177` | **already stale at base**: base line 291 is `]`; the taxonomy line is 313 at base, 331 at tip |
| `:6`, `:15-16`, `:109` | `pipeline.py:2099`, test docstrings, `stage_one_boundary.py` | unchanged (before line 226) |

---

## 5. Gates

All runs through `bounded_run.py` → `pinned_pytest.py`, `--heavy-lock ORCH-754c`, fresh
`--basetemp` under `C:/t/bt-orch754c/`, `PYTHONPATH=<worktree>/src`,
`T2PW_OFFLINE_CURATOR=1`. Reports in `evidence/g11/ORCH-754c/`, pins in
`evidence/g11/pin/ORCH-754c/`, pytest output (which the JSON report does not carry) in
`evidence/orch754c/`.

| # | run | tree | report · pin · log | result |
|---|---|---|---|---|
| 01 | proof | base | `g11/ORCH-754c/01-proof-base.json` · `pin/ORCH-754c/01-proof-base.pin.json` · no log | exit 1, 0 survivors. Count line lost: my `tail -25` filter cut it. Superseded by 02 |
| 02 | proof | **base** (prompt sha `0a1adddf…`, HEAD `f9892cdf`) | `02-proof-base.json` · `02-proof-base.pin.json` · `orch754c/02-proof-base.log` | **2 failed** in 0.11 s, 0 survivors. Byte diff at index 18496; text hash `2108aed0…` ≠ `27f5e873…` |
| 03 | prompt readers (3 files) | base | `03-readers-base.json` · `.pin.json` · `orch754c/03-readers-base.log` | **51 passed**, 0 survivors |
| 04 | proof | **tip** | `04-proof-tip.json` · `.pin.json` · `orch754c/04-proof-tip.log` | **2 passed**, 0 survivors |
| 05 | prompt readers (3 files) | tip | `05-readers-tip.json` · `.pin.json` · `orch754c/05-readers-tip.log` | **51 passed**, 0 survivors — unchanged from base |
| 06 | SMOKE, 22 files (`TEST_MATRIX.md` § Commands) | tip | `06-smoke-tip.json` · `.pin.json` · `orch754c/06-smoke-tip.log` | **508 passed** in 41.10 s, 0 survivors |

Prompt readers: `tests/test_scope_membership_filter_end_to_end.py`,
`tests/test_stage_one_boundary_lineage.py`, `tests/test_stage_one_ensemble.py`.

**G9.** The proof was run at base with the prompt untouched at `f9892cdf` and the proof
test in its committed form (not edited after run 02), then the prompt was replaced and the
same test re-run. It fails behaviourally at base (byte mismatch and hash mismatch), not on
a missing symbol.

`g11_evidence.py check --task ORCH-754c`: 6 artifacts, 0 non-compliant; 6 of 6 invoke
pytest directly with a committed pin verdict, 0 refused, 0 foreign src. Every report
`final_surviving_count: 0`.

---

## 6. Findings

1. **The dispatch's sha256 `bc716123…` is a CRLF working-tree hash that depends on
   `core.autocrlf`.** The clone-independent identity of the measured prompt is
   `27f5e873…`, and it matches `prompt_sha256` in all 41 recorded draws. Future citations
   of a prompt identity should use the LF hash.
2. **Base production `2108aed0…` has exactly one recorded draw** (round 1). Production has
   been running a prompt that the scored evaluation essentially never exercised.
3. **Two line citations to this prompt were already wrong at base** (`:116`, `:291`); this
   change makes a third (`:401`) wrong. Line-number citations into a prompt rot on any
   insertion.

---

## 7. Open questions for the product owner

**Q1. Is a v02-vs-v03 head-to-head wanted after the fact?** § 3 item 1. Options:
(a) none — accept the promotion as "production = measured", which is what D-104 rules;
(b) 5 v02 draws on each certified paper (10 billed extractions at 64,000 output tokens)
against the 12+12 existing v03 draws. It would show whether the block costs metabolic
recall. Cost: 10 calls, and nothing here suggests the answer would change the decision.

**Q2. Keep the new prompt-hash pin?** § 4. (a) keep — every prompt edit must move it with
a stated delta and name its measurement; (b) keep only the byte-equality test against the
variant file — an edit then requires changing the variant file or the test, which is the
same friction with weaker intent; (c) drop both after merge. Recommendation: (a).

**Q3. Who fixes the stale line citations?** § 4 table. (a) leave them; (b) a comment-only
follow-up updating `:401`→`:419`, `:116`→`:121`, `:291`→`:331`; (c) replace line numbers
with quoted anchors so the next insertion does not break them. Touches
`stage_one_boundary.py` and two test files, outside this task's boundary.

**Q4. `D-open-1` (extraction budget 16,000 in `.env` vs 64,000 measured) still stands** and
now applies to the promoted prompt: its measured behaviour is a 64,000-token behaviour.

**Q5. D-104 needs its `DECISIONS.md` entry** written by the product owner; this task did
not append it.

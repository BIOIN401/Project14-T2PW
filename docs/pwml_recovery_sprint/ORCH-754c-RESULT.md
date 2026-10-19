# ORCH-754c — production prompt promoted to the measured control `r2_01_control_v03`

**Status:** delivered on branch `orch754/c-prompt-v03`, base `f9892cdf`. Not merged, not
pushed. Needs independent review of the diff.
**Authority:** product-owner ruling 2026-10-19, recorded as **D-104** in `DECISIONS.md` on
`sprint/pwml-recovery`, commit `6d584b3f` (`DECISIONS.md:7034`, "prompt v03 promoted").
The curator-prompt revert (§ 8) is the D-104 amendment the orchestrator dispatched as an
ORCH-754c follow-up.
**Scope:** two prompt files. `pwml_system.txt` promoted to v03 (§§ 1–7); and
`pathway_curator_system.txt` reverted to `c9265fb8` (§ 8).
**No LLM call and no pipeline run** was made for this task (product owner forbade it).
Every bounded child ran with `T2PW_OFFLINE_CURATOR=1`.

**G9 labels:** the `pwml_system.txt` promotion is a **DELIBERATE BEHAVIOUR CHANGE** (not a
correction, not new functionality); the curator revert (§ 8) is a **CORRECTION**.

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
production tip's LF hash `27f5e873…` appears in **42 recorded draws, 41 of which produced a
payload** (corrected from 41: the independent reviewer found `pw701720`, and I then
confirmed its one draw is a failure):

| round | where recorded | draws with `27f5e873…` |
|---|---|---|
| round 1 at 16,000 output tokens (ORCH-747 § 3, `40-grid-16k.json`) | `C:/t/prompt-grid/pw701720/index.json` | 1, **failed**: `PipelineFailure … stop reason: identical_empty_response`, no payload. 9 of 10 variants failed this way; only v02 succeeded |
| round 1 (ORCH-747 § 2) | `C:/t/prompt-grid/final/index.json` | 1 |
| round 2 (§ 10) | `C:/t/prompt-grid/r2/` | 3 (one per paper) |
| round 3 (§ 11) | `C:/t/prompt-grid/r3/` | 6 |
| round 4 (§ 12) | `C:/t/prompt-grid/r4/` | 10 |
| ORCH-751 cohort | `C:/t/cohort/*/index.json` | 21 |

So production now sends exactly the text those draws sent. The base production hash
`2108aed0…` appears in **two** recorded draws, both round 1: `final/` (64,000 tokens) and
`pw701720/` (16,000 tokens, the one variant that did not come back empty). That single
16,000-token success is n=1 and ORCH-747 § 3 attributes the failures to the budget, not to
the prompt; it is recorded, not read as evidence that v02 tolerates a small budget better.

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
7. **Every scored measurement was at 64,000 output tokens, temperature 0.** At 16,000,
   round 1 failed 9 of 10 variants with empty bodies, v03 included (§ 2.1). **`D-open-1`
   is closed:** the primary checkout's `.env` now sets
   `OPENROUTER_EXTRACTION_MAX_TOKENS=64000` (`.env:36`, read by a single-key grep; the
   worktree has no `.env`), so production runs at the budget v03 was measured at. The
   prompt is 1,399 bytes longer, which costs input tokens, not output.

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
| 07 | curator proof (§ 8) | **base** (curator sha `5e3c293f…` CRLF, `git diff --quiet f9892cdf` on the file) | `07-curator-proof-base.json` · `.pin.json` · `orch754c/07-curator-proof-base.log` | **2 failed** in 0.11 s, 0 survivors. `'/processes/interactions/-' in` the prompt; text hash `e6da963e…` ≠ `6462c35e…` |
| 08 | curator tests (4 files) | base | `08-curator-tests-base.json` · `.pin.json` · `orch754c/08-curator-tests-base.log` | **101 passed**, 0 survivors |
| 09 | both ORCH-754c proofs | **tip 2** | `09-proofs-tip2.json` · `.pin.json` · `orch754c/09-proofs-tip2.log` | **4 passed**, 0 survivors |
| 10 | curator tests (4 files) | tip 2 | `10-curator-tests-tip.json` · `.pin.json` · `orch754c/10-curator-tests-tip.log` | **101 passed**, 0 survivors — unchanged from base |
| 11 | SMOKE, 22 files | tip 2 | `11-smoke-tip2.json` · `.pin.json` · `orch754c/11-smoke-tip2.log` | **508 passed** in 40.10 s, 0 survivors |

Prompt readers: `tests/test_scope_membership_filter_end_to_end.py`,
`tests/test_stage_one_boundary_lineage.py`, `tests/test_stage_one_ensemble.py`.

**G9.** The proof was run at base with the prompt untouched at `f9892cdf` and the proof
test in its committed form (not edited after run 02), then the prompt was replaced and the
same test re-run. It fails behaviourally at base (byte mismatch and hash mismatch), not on
a missing symbol.

The curator proof (07 → 09) followed the same order: test written in its committed form,
run against the curator file at its `f9892cdf` content, then the file restored with
`git checkout c9265fb8 -- <file>` and the test re-run. It fails at base on the
instruction itself being present, not only on a hash.

`g11_evidence.py check --task ORCH-754c`: **11 artifacts, 0 non-compliant**; 11 of 11 invoke
pytest directly with a committed pin verdict, 0 refused, 0 violation codes, 0 foreign src,
0 label mismatches. Every report `final_surviving_count: 0`.

---

## 6. Findings

1. **The dispatch's sha256 `bc716123…` is a CRLF working-tree hash that depends on
   `core.autocrlf`.** The clone-independent identity of the measured prompt is
   `27f5e873…`, and it matches `prompt_sha256` in all 42 recorded draws. Future citations
   of a prompt identity should use the LF hash.
2. **Base production `2108aed0…` has two recorded draws, both round 1 on the unscored araC
   paper** (§ 2.1). Production has been running a prompt that the scored evaluation never
   exercised.
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

**Q4. ~~`D-open-1`~~ — closed.** `.env` is now 64,000 (§ 3 item 7). Nothing to decide.

**Q5. ~~D-104 entry~~ — done.** Recorded by the product owner at `6d584b3f`. This task did
not touch `DECISIONS.md`.

**Q6. Did any run since `70d05e03` use the live curator with the § 5 prompt?** § 8. If so,
its `audited_json` may hold curator-added interactions or nucleic_acids entities that no
decision authorized and no evidence check covered. Not checked here: it needs a scan of run
trees for curator patches at `/processes/interactions/-`, and runs with
`T2PW_OFFLINE_CURATOR=1` (every deterministic gate) never called the curator. Options:
(a) scan run trees from the window the § 5 text was live — committed in `70d05e03` at
2026-10-19 13:39 -0600, but present in the primary checkout's working tree from ORCH-747
(work dated 2026-10-15, `ORCH-747-RESULT.md` § 10) — and quarantine any
leg with such a patch; (b) accept the exposure as nil if no live-curator app run happened in
that window.

**Q7. Is a curator interaction-add capability wanted at all?** The revert removes it; it
does not decide against it. Doing it properly needs, at minimum: sending the curator the
evidence strings it is told to quote (`pathway_curator.py` `user_payload` sends none today),
an evidence-substring check on accepted adds, and actor-role protection for interactions
(`_ACTOR_ROLE_PATH_RE`, `apply_audit_patch.py:1419`, does not match them). That would be a
new card with its own D-xxx.

---

## 8. Follow-up: curator prompt reverted to `c9265fb8` (D-104 amendment)

Dispatched by the orchestrator after an independent review of `70d05e03`; the product
owner ruled **REVERT**, recorded as a D-104 amendment. **G9 label: CORRECTION** of
pre-existing, unauthorized behaviour.

### 8.1 The change

`src/t2pw/llm/prompts/pathway_curator_system.txt` is the `c9265fb8` text again
(`git checkout c9265fb8 -- <file>`; `git diff --quiet c9265fb8 -- <file>` holds). This undoes
the whole `70d05e03` diff to the file, **35 insertions and 1 deletion**, and nothing else:

- **§ 5 "Regulation stated in the payload but absent from interactions[]"**, which told
  the curator to propose `op=add at /processes/interactions/-` with a full interaction
  object, and to add a `nucleic_acids` entity first when a named element was missing;
- its **"Removal:"** paragraph (self-loop / biological_state-pointing interactions →
  warning instead of remove patch);
- the clause **"Interactions are the one exception, under § 5, …"** appended to "Do not add
  new reactions or transports." That rule is back to the bare sentence.

| | `f9892cdf` (ORCH-747 text) | tip (`c9265fb8` text) |
|---|---|---|
| lines / blob bytes (LF) | 129 / 9,797 | 95 / 7,482 |
| working-tree sha256 (CRLF) | `5e3c293f0caa2ef91d439dd6919bc3f9527a43a5b9a0f4f96431fd03350052ee` | `23b16b591be7f0d2153dc035a7b4e1fbcc7f1ef4c8ed2ea3294916da6d433298` |
| LF text sha256 (= blob) | `e6da963e76e654ae23708636a3e06053e8884d996ff0b40d4d776a809dc16821` | `6462c35ed8131f557f9c5cc1af7c75d7f10911529897a744bb33b57e20896531` |

Line endings handled as for `pwml_system.txt`: LF in the object store, CRLF on disk via
`core.autocrlf` (`git ls-files --eol`: `i/lf w/crlf`; 95 CRLF pairs in 95 lines after the
restore).

### 8.2 Why, verified rather than relayed

Two of the reviewer's points I checked directly in this tree:

1. **The curator cannot quote evidence it is never sent.** `pathway_curator.py:286-297`
   builds `user_payload` from `task`, `pathway_name`, `organism`, `entity_index`,
   `reactions` (name, inputs, outputs only), `transports`, `reaction_order_exists` and an
   optional `reaction_summary`. No evidence strings and no `interactions` list. § 5's
   "`evidence` MUST be a verbatim quote copied from the payload" was unsatisfiable.
2. **`_ACTOR_ROLE_PATH_RE`** is at `apply_audit_patch.py:1419` (the dispatch said 1420; it
   is the `re.compile(` line) and is matched at `:2314`.

The accept-at-0.75 policy and the pre-freeze landing in `audited_json` I took from the
review and did not re-trace.

### 8.3 Proof

`tests/test_orch754c_curator_prompt_reverted.py`, two tests:

- `test_curator_prompt_gives_no_instruction_to_add_interactions`: the prompt contains
  neither `/processes/interactions/-` nor `Interactions are the one exception`. This is the
  behavioural assertion: it fails on the instruction being there.
- `test_curator_prompt_is_the_c9265fb8_text`: LF text sha256 == `6462c35e…`.

At the `f9892cdf` content: **2 failed** (run 07). At tip: **2 passed** (run 09, together
with the two § 1 proofs, 4 passed).

### 8.4 What moves

- **No pin, test or hash** references this prompt. Searched `tests/`, `src/` and the
  evidence tree for `pathway_curator_system` and the three hashes above: the only
  reference is the loader, `pathway_curator.py:118`.
- **No test expects the curator to propose interactions.** `70d05e03` touched seven test
  files (`test_draft_graph_entity_kinds`, `test_eval_v2_paper_gold`,
  `test_eval_v2_pathbank_gold`, `test_pwml_interaction_export`,
  `test_pwml_interaction_type_spellings`, `test_stage_one_ensemble` new; 9 lines added to
  `test_research_mode_report`). None exercises `pathway_curator`: `grep -il curator` hits
  only `test_eval_v2_paper_gold.py:129` and `test_eval_v2_pathbank_gold.py:55`, both about
  PathBank's human curators, not the curator stage. The four files that mention
  `pathway_curator` (`test_c105_actor_role_evidence`, `test_curation_patch_transaction`,
  `test_curator_offline_mode`, `test_streamlit_stage2_orchestration`) never read the
  prompt text: **101 passed before and after** (runs 08, 10). So nothing fails by design,
  and no test outside the boundary needed listing.
- **Runtime:** the prompt is read once per process into `_CURATOR_SYSTEM_PROMPT`, and only
  when the live curator runs. With `T2PW_OFFLINE_CURATOR=1`, which every deterministic gate
  sets, the curator issues no call, so these gates could never have observed § 5 either
  way. The behavioural effect is on live app runs only.
- The `_ACTOR_ROLE_PATH_RE` gap and the 0.75 add threshold are **unchanged**. Without the
  § 5 instruction the curator has no prompt licence to add interactions, but nothing in
  code stops it from proposing one. See § 7 Q7.

### 8.5 Not settled

§ 7 **Q6** (whether any live-curator run consumed the § 5 text) and **Q7** (whether a
properly gated curator interaction-add is wanted).

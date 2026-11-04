# ORCH-803 RESULT: the paper's own abbreviation definitions unify an entity (deterministic part)

PO ruling, 2026-11-04 (verbatim): "use papers defintion and make sure it is unified and we dont make duplciates, asking llm can be kept as bacup in some scenario they use two intercangeabley but then it should be easy to tell that its the same entity right"

This lane is the deterministic part: the paper's explicit definition. The LLM backup is a later lane (section 7).

| | |
|---|---|
| Branch | `orch803/paper-abbrev-identity`, base `563d39db` |
| Code commit | `a79dba9e` (code + test); docs in a separate commit |
| Files | `src/t2pw/pipeline/paper_lexicon.py` (+51/-3), `src/t2pw/pipeline/stage_one_ensemble.py` (+27/-2), `src/t2pw/mapping/map_ids.py` (+15/-0), `tests/test_orch803_paper_abbrev_identity.py` (+158, new) |

## 1. Design (written before coding)

### 1.1 The three defects, re-measured on the base code

Probe: `C:/t/orch803/probe.py`, scratch (bounded run `C:/t/orch803/00-probe.json`). It ran base `abbreviation_pairs` / `PaperLexicon` on each ORCH-801 leg's `01_source_text.txt`.

1. **tb (PMC5098729), "thymidine (dThd)( Fig 4C".** Base gives no `dThd` pair. The lead's theory is confirmed. Schwartz-Hearst needs the short form's first character, `d`, to start a word of the long form. "thymidine" has no word-initial `d`, because the `d` of `dThd` is the implicit 2'-deoxy prefix. So `dThd` is never rewritten. Stage-1 draws 1 and 2 wrote `dTMP -> dThd` and draw 3 wrote `dTMP -> thymidine`. That gives two pooled reactions (`run-1cb98e97639e/stage1_ensemble_report.txt`, REACTIONS 5 and 9, in the orch801run worktree's `tmp/extraction_diagnostics`). The same paper defines `deoxyuridine (dUrd)`, and that one already pairs, because "deoxyuridine" starts with `d`.
2. **cdadc1 (PMC12184417), "deoxyuridine monophosphate (dUMP)".** The pair already exists on base, and the Stage-1 rewrite `dUMP -> deoxyuridine monophosphate` is correct (`run-144a9506cbfc/stage1_ensemble_report.txt`, lines 69-71). The defect is the name gate alone. `final_mapped.json` records `name_gate.verdict = reject` with `no_shared_meaningful_token`, `compared_names = [dUMP, dUMP, DUMP]`, and PathBank 1087 ids in `rejected_mapped_ids`. The gate never sees the paper.
3. **salicylic (PMC12408352), "trans -CA to benzoyl-CoA (BA-CoA)".** The theory is only half right. Base does NOT fail to pair `BA-CoA`. It pairs it to a **wrong** long form: `BA-CoA -> "been shown to convert trans -CA to benzoyl-CoA"`. Schwartz-Hearst matches `A` to the `A` of `-CA` and `B` to "been". The two halves of the cause:
   - Benzoyl has no `a`, because `BA` abbreviates *benzoic acid* (the paper also defines `benzoic acid (BA)`).
   - That wrong pair did no visible damage in leg 1534 only because of a second cause: the Stage-1 rewrite map is built from **reaction participants only**. In that leg `BA-CoA` occurs only as transport cargo, an entity row and an element location, so it was never looked up. Every draw wrote the reactions with `benzoyl-CoA` (`run-4b3daad489d8`, REACTION 2 key `cinnamoyl coa -> benzoyl coa`).

### 1.2 Changes, and why each sits where it does

| # | Where | What | Why there |
|---|---|---|---|
| A1 | `paper_lexicon.abbreviation_pairs` | New pattern `shared_suffix`: `LONG-TAIL (SF-TAIL)`. The word directly before the parenthesis ends in the same hyphenated tail (case-sensitive, at least 2 characters, contains a letter). The abbreviation head looks like an abbreviation: at least 2 capitals, or at most 3 characters with a capital. Its first letter is the long head's first letter. The long form is that one word. This pattern is tried **before** Schwartz-Hearst. | Schwartz-Hearst overshoots across words here and produces the wrong `BA-CoA` pair. Inside the word that shares the tail, the one-word long form is the only reading. The capital-letter rule stops `acetyl-CoA (acyl-CoA)`, where the parenthesis holds a different molecule's name, not an abbreviation. |
| A2 | `paper_lexicon.abbreviation_pairs` | New pattern `deoxy_prefix`, for `LONG (dXxx)` only. It applies when the short form is lowercase `d` followed by a capital, and Schwartz-Hearst fails on the whole short form. It retries Schwartz-Hearst, with all its validity limits, on the short form minus that `d`. | This is the nucleoside convention (dThd, dCyd, dGuo, dAdo, dTMP). It is a fallback only: when the full form already pairs (`deoxyuridine (dUrd)`), nothing changes. Schwartz-Hearst's own guard still applies: a long form that ends with the remainder is rejected, so `catalytically inactive Cas9 (dCas9)` gives no pair. |
| B | `stage_one_ensemble._rewrite_map` | The rewrite map also reads `/processes/transports` cargo, not only reaction inputs and outputs. | D-104 § 3(a) says participant names, and cargo is a transport's participant. Without this, case 3 keeps a `BA-CoA` node even with the right pair. The rename already reaches cargo, entities and locations (`_canonicalise`). Only the lookup table missed cargo-only names. |
| C | `stage_one_ensemble.run_ensemble` + `paper_defined_pair()`; `PaperLexicon.defines()` | Each pooled Stage-1 pass registers its `PaperLexicon` against the run's diagnostics recorder in a `WeakKeyDictionary`. `paper_defined_pair(a, b)` answers whether any dictionary of **this run** defines `a` and `b` as one pair: either direction, short form case-sensitive, long form by `strict_key`, ambiguous short forms excluded. | This uses the scoping `merge_on_mapped_identity` already relies on at Stage 6 (`_pooled_draws` reads `current_diagnostics()`). `activate()` starts a new recorder per run, so a previous paper's dictionary can never answer for this one. Nothing is written to disk and no payload key is added (the payload is a model prompt, so its schema is frozen). At N=1 nothing is registered and the gate behaves exactly as before. |
| D | `map_ids._name_gate_verdict` | One new rung, `paper_abbreviation_identity`, placed immediately before the final `reject`. For a non-protein kind only, it keeps the match when the paper of this run defines the entity's name and one of the candidate's display names as one abbreviation pair. | The record is the right one by the paper's own definition, and the gate is where the refusal happens. Placed last, the rung can only turn a `reject` into a `keep`, never the reverse. Every other rung, threshold and ordering is unchanged. `_refuse_class_term_identity` still runs after it. Proteins are excluded (open for the PO, section 8). |

What is deliberately NOT built:
- **No convention table** (dThd = thymidine and so on) in the gate. The audit suggested one, but it would admit identity on knowledge that is not the paper's definition, which the brief forbids.
- **No LLM.**
- **No change to the identity merge.** Case 1 collapses at Stage 1, by rule 1, once the pair exists. Case 2 needs no merge: there was one node, and it only lost its identity.

### 1.3 Stop condition check

Change D admits identity only when the paper of the same run defines the pair. No other evidence admits anything, so the brief's stop condition does not trigger.

## 2. Diff summary (`563d39db..a79dba9e`)

- **`paper_lexicon.py`**
  - Adds `_shared_suffix_long()`, the `PATTERN_SHARED_SUFFIX` and `PATTERN_DEOXY_PREFIX` labels, and the `_DEOXY_SHORT` regex.
  - `abbreviation_pairs()` tries `shared_suffix` first (LONG (SF) only), then Schwartz-Hearst unchanged, then the `deoxy_prefix` retry (LONG (SF) only, only after Schwartz-Hearst failed).
  - New `PaperLexicon.defines(a, b)`.
  - Docstring and `__all__` updated.
- **`stage_one_ensemble.py`**
  - `_rewrite_map` also walks `/processes/transports` `cargo`.
  - New `_RUN_LEXICONS`, a `WeakKeyDictionary` keyed by the diagnostics recorder, and `paper_defined_pair(a, b)`.
  - `run_ensemble` registers `PaperLexicon(paper_text)` after the merge. `merge_draws` itself is untouched.
- **`map_ids.py`**
  - One rung at the end of `_name_gate_verdict`, before the final reject. It is `kind != "protein"` only and records `reason = paper_abbreviation_identity` and `paper_pair = [short, long]` on the gate record.
  - No other rung, threshold or ordering changed.
- **New test file**, 17 tests.

Known ceiling: each pooled pass now builds its `PaperLexicon` twice, once inside `merge_draws` and once to register it. That costs one regex pass over the paper. Returning the object from `merge_draws` would change its return shape, so it was not done.

## 3. Proofs

| Proof | Command | Result (from the piped wrapper output) | G11 report |
|---|---|---|---|
| G9 base | focused file, base export `C:/t/orch803/base` (`git archive 563d39db src tests/conftest.py pytest.ini`) | `5 failed, 12 passed in 0.21s`. The 5 are exactly the `test_g9_*` tests and every failure is an assertion, not a missing symbol: `assert None == 'thymidine'`, `assert 2 == 1` (two reactions), `assert 'reject' == 'keep'`, `assert 'been shown t...o benzoyl-CoA' == 'benzoyl-CoA'`, `['ba coa', ...] == ['benzoyl coa', ...]` | `evidence/g11/ORCH-803/01-focused-base.json` + `-stdout.txt` |
| G9 tip | same file at `a79dba9e` | `17 passed in 0.24s` | `02-focused-tip.json` + `-stdout.txt` |
| SMOKE | the 22-file block in `TEST_MATRIX.md` § Commands | `511 passed in 62.97s` (unchanged count) | `06-smoke.json` + `-stdout.txt` |
| Affected, outside SMOKE | `test_stage_one_paper_dictionary.py test_stage_one_ensemble.py test_c120_identity.py test_orch763_nested_locations.py test_orch758b_empty_chunk.py test_orch759a_duplicate_protein_fold.py`: every test file that imports `merge_draws`/`run_ensemble`/`_name_gate_verdict` | `254 passed in 6.92s` | `07-affected.json` + `-stdout.txt` |

Every wrapper report shows `FINAL SURVIVING COUNT : 0`. The pytest counts come from the piped wrapper output saved as `NN-*-stdout.txt` beside each JSON, because the JSON has no stdout (`.log` is gitignored under `g11/`).

`tests/test_orch759a_fold_app_path.py` also references the ensemble, but it is a Streamlit `AppTest` file. It belongs to chunk D (split-process gate), which the brief excludes, so it was **not run**.

G9 labelling: the 5 `test_g9_*` tests are corrections of delivered behaviour, each on the paper's own sentence. The 12 `test_twin_*` tests are negative twins and pass on base and tip. The required twins are all present:
- `(Fig. 2)`, `(n = 3)`, `(Supplementary Table 2)`, `(PDB 1ABC)`;
- `thymidine kinase (TK)`, which never makes TK = thymidine, at the pair level and at the gate;
- two molecules written side by side: `thymidine (deoxycytidine)`, `BA-CoA (CA-CoA)`, `benzoyl-CoA (cinnamoyl-CoA)`, `L-Glu (D-Glu)`;
- the gate refuses at N=1 and in the next run after `activate()`;
- `deoxy_prefix` never overrides a full match (`deoxyuridine (dUrd)`), and `catalytically inactive Cas9 (dCas9)` pairs nothing.

The payload-level test (`test_g9_tb_payload_one_node_and_the_duplicate_reaction_merged`) pools `dTMP -> dThd` with `dTMP -> thymidine` (same enzyme) over the tb sentence:
- tip: one reaction and one `thymidine` compound;
- base: two reactions.

The collapse happens at the Stage-1 merge (D-104 rule 1), not the Stage-6 identity merge. Once the paper pairs `dThd` with thymidine, both draws write `thymidine` before pooling. That is the earlier, existing merge, and it needs no database identity.

## 4. No-loss scan

Script: `evidence/orch803/no_loss_scan.py`. It ran once with base `src` and once with tip `src` on `PYTHONPATH`, then `compare`.

Corpus:
- every committed `runs*/**/01_source_text.txt`;
- `evidence/orch754a_inputs/papers/*.txt`;
- the 3 ORCH-801 leg texts (uncommitted, read-only from the orch801run worktree).

That is 222 files and 34 distinct texts.

Gate records: every committed `name_gate` record with compared names and an identity, plus every row's `db_match.chosen`. The second source was added because the tb `dThd` row kept no `name_gate` record; it was re-added by an audit patch. That gives 681 records, each replayed through `_name_gate_verdict` after a pooled Stage 1 over that paper's text.

| | count | report |
|---|---|---|
| (a) pairs, base / tip | 880 / 881 | `03-scan-tip-stdout.txt`, `04-scan-base-stdout.txt` |
| (a) removed | **1, a correction** | `evidence/orch803/scan_compare.json`, `.log`; `05-scan-compare.json` |
| (a) added | 2, both correct | same |
| (b) records replayed | 681 base, 681 tip | same |
| (b) keep -> reject | **0** | same |
| (b) reject -> keep | 5, all correct | same |

(a) classification:
- REMOVED `BA-CoA -> "been shown to convert trans -CA to benzoyl-CoA"` (PMC12408352). **Incorrect pair, removed.** It is the base defect of case 3, replaced by the line below. Strictly this breaks the brief's "removed pairs must be 0". I read that rule as protecting correct pairs; this one was wrong, and any participant it touched would have been renamed to a sentence fragment. Flagged for the PO in section 8.
- ADDED `BA-CoA -> benzoyl-CoA` (PMC12408352): **correct** (`shared_suffix`; the paper's "benzoyl-CoA (BA-CoA)").
- ADDED `dThd -> thymidine` (PMC5098729): **correct** (`deoxy_prefix`; "thymidine (dThd)").

No other pair in the 34 texts moved. That includes `deoxyuridine (dUrd)`, `cinnamoyl-CoA (CA-CoA)` and `benzoylacetyl-CoA (BAc-CoA)`, which the new patterns reproduce identically.

(b) classification:
- `SEPHCHC` vs `2-succinyl-5-enolpyruvyl-6-hydroxy-3-cyclohexene-1-carboxylate`, `runs_verify/2026-08-21_2014/papers/PMC12312563/{strict,research}/final_mapped.json` (2 records). The paper says "2-succinyl-5-enolpyruvyl-6-hydroxy-3-cyclohexene-1-carboxylate (SEPHCHC) synthase (MenD)", and the identity is `kegg:C16519`, which is SEPHCHC. **Correct admission.** This is a fourth real case the change repairs, beyond the brief's three.
- `deoxyuridine monophosphate` vs `dUMP` (PathBank 1087, CHEBI:17622), cdadc1, as `name_gate` and as `db_match` (2 records). **Correct** (case 2).
- `dThd` vs `Thymidine` (PathBank 187), tb (`db_match`). **Correct.** In production this case is now fixed earlier, by the Stage-1 rewrite. The gate rung would also admit it if a chunk boundary separated the definition from the name.

**No incorrect added pair and no wrong admission**, so there was no stop.

Scan limits (what it does not prove):
- Replays use each paper's **full** text, while production builds one lexicon per Stage-1 chunk. Full text is a superset, so this over-approximates what the rung can admit, which is the conservative direction for a no-loss check.
- The candidate is rebuilt from the record's compared names. Protein records therefore replay with less fidelity, but the new rung skips proteins, and base and tip received identical input.
- Records with no `name_gate` and no `db_match.chosen` are not covered.

## 5. Dead ends, disproved theories, slips

1. **"BA-CoA fails Schwartz-Hearst"** (lead's theory): **disproved.** It does not fail. It pairs to a wrong long form (section 1.1). The wrong pair stayed invisible only because cargo-only names were never looked up. A fix that relaxed the letter alignment would have left the wrong pair winning. That is why `shared_suffix` goes first.
2. **"dThd fails Schwartz-Hearst because of the implicit deoxy `d`"**: **confirmed** by the probe and by tracing `_best_long_form`.
3. **"cdadc1 needs a parser fix"**: **wrong.** The pair `dUMP -> deoxyuridine monophosphate` already exists on base and the rewrite is right. Only the gate was blind to the paper.
4. **Audit's suggested fix, a nucleoside convention table in the gate** (dThd/Thd = thymidine, dCyd/dCtd, dNMP): **rejected.** That is identity from knowledge, not from the paper's definition, and the brief forbids it.
5. **First draft of the `(Fig. 2)`-style twins included `acetyl-CoA (acyl-CoA)`.** It fails on base and tip alike. **Base Schwartz-Hearst already pairs `acyl-CoA -> acetyl-CoA`**, because every letter of "acyl" is in order inside "acetyl". This is a pre-existing false pair that this lane does not touch: `shared_suffix` declines it, since `acyl` has no capital, but Schwartz-Hearst then matches it. It did not occur in the 34 scanned texts (no pair moved there). It was replaced by `thymidine (deoxycytidine)` and is recorded as an open item (section 8).
6. **First draft of the TK twin called `PaperLexicon.defines`**, which does not exist on base, so the twin errored there. A twin must pass on base, so it was rewritten to use `long_form` and the gate.
7. **Wrapper slips (process, not result):**
   - The draft `compare` (scratch, `C:/t/orch803/cmp_draft.json`) was run **without** the bounded wrapper, and so was one trivial `python3 -c "print(1)"`.
   - Both were short and foreground, and no process survived.
   - The official compare was re-run inside the wrapper (`05-scan-compare.json`). Draft runs whose reports stayed in scratch: `C:/t/orch803/00-probe.json`, `01-focused-draft.json`, `02-scan-tip-draft.json`, `03-scan-base-draft.json`.
8. **Heavy lock:** held by ORCH-804 on 3 attempts (exit before child start). It was retried after other work, under the 5-attempt limit.

## 6. What each case does now, and what is NOT proven

- **tb:** Stage 1 rewrites `dThd` to `thymidine` in every draw, so `dTMP -> dThd` and `dTMP -> thymidine` pool as one reaction and one node. `thymidine` then maps to PathBank 187 on a shared token. `dCtd` is unchanged (section 7).
- **cdadc1:** the gate keeps PathBank 1087 for `deoxyuridine monophosphate` (`paper_abbreviation_identity`).
- **salicylic:** `BA-CoA` pairs to `benzoyl-CoA`, and cargo-only uses are renamed, so the transport cargo, the entity row and the location become benzoyl-CoA.
  - This fixes identity only. Whether the export transport should exist at all (audit F6, a disfavoured prior hypothesis) is a separate question and is untouched.
  - The salicylic leg ran **chunked** (two dictionaries, 43 and 18 pairs, in `run-4b3daad489d8`). The Stage-1 rename only works inside the chunk that contains "benzoyl-CoA (BA-CoA)". Which chunk wrote the cargo is not recorded, so it is **not proven** that this leg would rename it. If the cargo came from the other chunk, the node stays `BA-CoA` with no identity. The gate rung cannot help there because the name has no database candidates (`no_db_candidates`). Cross-chunk Stage-1 dictionaries are an open item (section 8).
- **No pipeline leg was run** (by brief). End-to-end PWML effect on the three legs is inferred from the tests and the scan replay, not measured.

## 7. LLM backup: where it plugs in (not built)

- **Plug-in point:** `PaperLexicon.canonical` as called by `stage_one_ensemble._rewrite_map`. It applies to a participant name the paper never defines: `long_form(name)` is `None`, the name is not a currency, and it looks like an abbreviation.
- **Input the LLM would see:**
  1. the name;
  2. every paper sentence that mentions it (`name_keys.mentions` over `lexicon.text`);
  3. the closed candidate list of the other participant names the pooled draws wrote. For tb that is `deoxycytidine` beside `dCtd`.
- **Output:** one candidate or none. It is written as a rewrite with a new rule label (for example `llm_abbreviation`) and `review_required`, in the side file's `names_rewritten`, so a reader can see that the two names were judged the same entity and by what. That answers the PO's "easy to tell that its the same entity".
- **Why this placement:** it runs before pooling, so duplicates collapse at Stage 1 exactly as the deterministic rule does. It never touches the name gate, so no database identity is ever admitted on an LLM answer.
- **Real cases left for it:**
  - tb `dCtd`, which the paper never defines (ORCH-801 tb REACTIONS 3/8 `dctd -> deoxyuridine` vs `deoxycytidine -> deoxyuridine`, and 6/10 `dcmp -> dctd` vs `dcmp -> deoxycytidine`; `dCtd` is `no_db_candidates`);
  - any pair split across Stage-1 chunks (section 6).

## 8. Open for the PO

1. **A wrong pair was removed** (`BA-CoA -> "been shown to ..."`). The brief said removed pairs must be 0. I treated removing an incorrect pair as allowed under "never worsen". Options: (a) accept, at no cost; (b) require a pair-replacement rule to be ruled first, which blocks case 3.
2. **The gate rung excludes proteins.** "thymidine kinase (TK)" would let `TK` admit a "Thymidine kinase" record. That is the paper's definition, but protein abbreviations collide with gene symbols across species. Options: (a) keep compounds only (now); (b) extend to proteins only when the species rung also passes, which needs its own lane and red-team review.
3. **The rung only works when Stage 1 pooled (N>1).** At N=1 no dictionary is registered, so the gate behaves exactly as before. Options: (a) keep, which is consistent with D-104 being an N>1 feature; (b) register the paper at N=1 too, which changes N=1 output and needs its own no-loss scan.
4. **Stage-1 dictionaries are per chunk** (pre-existing). Options: (a) leave; (b) build one dictionary from the whole paper and pass it to every chunk's merge. That is a small change but moves every chunked run, so it needs a scan.
5. **Pre-existing false pair `acyl-CoA -> acetyl-CoA`** (section 5 item 5): plain Schwartz-Hearst accepts a lowercase ordinary word in parentheses as an abbreviation. It was not seen in the scanned corpus. Options: (a) leave; (b) require a short form to contain a capital or digit, which is a Schwartz-Hearst deviation that needs its own scan.
6. **Transport cargo now enters the rewrite map, so typo snapping (rule 2) also reaches cargo-only names**, under the same safety rules and flagged `typo_snapped`. The scan does not measure Stage-1 renames, because no committed draws are pooled here. Options: (a) accept as within D-104 § 3(a); (b) restrict cargo to rule 1 only, at a few lines' cost.

## 9. Review round 1 (reviewer verdict on `a79dba9e`: APPROVE WITH NOTES)

Code commit `2ee07a80`. Line counts: `map_ids.py` +8/-1, `paper_lexicon.py` +6/-2, test +19.

- **F1 (MEDIUM), the gate rung trusted any display name.** `_candidate_display_names` includes `short_name` and the BioCyc mnemonic. So a record whose `short_name` only *equals* the paper's abbreviation would have been admitted. Example: paper "phosphatidic acid (PA)", record "Hexadecanoate" with short_name "PA".
  - **Fix:** when the paper pair is matched through its SHORT form, only the candidate's primary `name` counts (the fallback name when there is no candidate row). Long-form matches are unchanged.
  - cdadc1 still passes, because PathBank 1087's primary name is "dUMP".
  - **Ceiling:** a record whose *primary* name is the abbreviation of a different molecule would still pass. That is the reviewer's specified bound.
- **F3 (LOW), the deoxy retry paired phrases:** "nucleotide pool (dNTP)" gave dNTP = "nucleotide pool", and "nucleoside triphosphates (dNTPs)" paired too.
  - **Fix:** the retry keeps only a **one-word** long form.
  - **Why this tightening:** the convention the pattern exists for is a nucleoside's single name, where the `d` is the unwritten 2'-deoxy (thymidine to dThd, cytidine to dCyd). A multi-word long form has no such word to carry the implicit prefix, and every multi-word `dN...` case seen is a phrase.
  - **Cost:** "thymidine monophosphate (dTMP)" no longer pairs through this pattern. No scanned paper writes it that way. "deoxythymidine monophosphate (dTMP)" still pairs through plain Schwartz-Hearst.
- **Tests:** three new negative twins.
  - `test_twin_short_name_alone_never_admits_the_papers_long_form`.
  - `test_twin_deoxy_retry_never_pairs_a_phrase`, run on both dNTP phrases.
  - These are **not** G9 corrections of base behaviour: base has neither the rung nor the retry, so they pass on base. To show they bite, they were run against the reviewed commit `a79dba9e`: `3 failed, 17 passed in 0.61s`, exactly the 3 new twins.

| Proof | Result (piped wrapper output) | G11 report |
|---|---|---|
| focused, tip `2ee07a80` | `20 passed in 0.45s` | `08-r1-focused-tip.json` + `-stdout.txt` |
| focused, reviewed `a79dba9e` (export `C:/t/orch803/prev`) | `3 failed, 17 passed in 0.61s` (the 3 new twins) | `09-r1-focused-a79dba9e.json` + `-stdout.txt` |
| no-loss scan, tip vs the unchanged base scan | `pairs removed 1, added 2; records base 681 tip 681; keep_to_reject 0, reject_to_keep 5, other 0`. Same pairs and the same 5 flips as section 4. `scan_tip.json` and `scan_compare.json` are byte-identical to round 0; only `scan_compare.log` changed (wrapper PIDs). | `10-r1-scan-tip.json`, `11-r1-scan-compare.json`, `evidence/orch803/scan_compare.log` |
| SMOKE | `511 passed in 62.02s (0:01:02)` | `12-r1-smoke.json` + `-stdout.txt` |

Every report shows `FINAL SURVIVING COUNT : 0`. Base was not re-run, because no new test is a G9 correction. The base G9 result in section 3 still stands: the 5 corrections are unchanged.

**Decision record (lead's note, recorded as instructed):**
- The two new pair patterns, `shared_suffix` and `deoxy_prefix`, sit in tension with D-110 AMENDMENT 1/2.
- They are covered by the PO's ORCH-801 § 4 option-A ruling.
- The lead handles the DECISIONS.md record with the PO. This lane did not edit DECISIONS.md.

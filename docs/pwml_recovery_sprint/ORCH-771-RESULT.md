# ORCH-771 — the PWML known-issues catalogue (D-114 § 5)

**Status:** DONE (docs only). Branch `orch771/known-issues`, worktree `.claude/worktrees/orch771`, base `4befcb33`
(sprint tip). No Python, no test, no pipeline run, no LLM call, no network. Nothing under `src/` or `tests/`
changed; `LEDGER.md`, `TEST_MATRIX.md`, `DECISIONS.md` and `CLAUDE.md` were not edited.

**Delivered:**
- `PWML_KNOWN_ISSUES.md`: 64 entries, `KI-001` to `KI-064`, in seven groups (A row-level export requirements, B
  whole-pathway blockers, C wrong content in a written PWML, D duplicates and merge quality, E extraction, F
  LLM/runtime/harness, G operations), plus the appendix **"Reason code → KI id"** (82 machine codes, 19 human
  messages), the contract ORCH-772 labels omissions with.
- `FINDINGS.md`: `F-204` to `F-218` (15 durable findings) and a cross-reference paragraph under `F-203` (its second
  instance, the Ara paraphrase refusal). Each cross-references its KI id.
- This record.

## 1. What I read

| source | count | notes |
|---|---|---|
| RESULT docs, ORCH-747 to ORCH-769 | **32** | 747, 748, 751, 752, 754, 754a, 754b, 754c, 754d, 755, 756, 757, 758, 758a, 758b, 759, 759a, 759b, 759c, 759d, 759e, 761, 762, 763, 764, 765, 766, 767, 768, 769, 769b, 769c: 31 on the sprint tip, plus ORCH-765 read from its unmerged branch at `9fa8f3be` (`git show`) |
| tasks with no RESULT doc | 3 | ORCH-749, ORCH-750 (LEDGER rows only, `LEDGER.md` lines ~10964-10966), ORCH-753 (interrupted by the 2026-10-16 crash, no record) |
| `DECISIONS.md` | D-104 to D-114 | 11 entries plus 7 amendments (D-104 AMENDMENTS 1-5, D-110 AMENDMENTS 1-2); D-062, D-094 and D-098 looked up where an entry depends on them |
| `LEDGER.md` | lines 10940-11155 | 2026-10-16 to 2026-10-25, including the last two rows (crash #2 and its correction) |
| evidence files | about 30 | gold accuracy (`orch756_`, `orch759_`, `orch759p_`, `orch767_gold_accuracy.txt`), trace and leg stats (ORCH-755/756/759/759p/767), `orch769_diag/769a`, `769b`, `769d` outputs, `orch758_*` diagnosis outputs, the ORCH-761 probes; every evidence path cited in the catalogue was checked to exist |
| `runs_papers/ORCH-767/MORNING_SUMMARY.md` | 1 | primary checkout (run scratch) |
| reviewer reports `evidence/g11/ORCH-7*/*rev-*` | 306 found, 2 opened | they are bounded-run lifecycle reports with no stdout (memory note *bounded_run.py JSON report has no stdout*); review verdicts were taken from each RESULT doc's orchestrator record, not from these files |
| `src/t2pw` at `4befcb33` | about 100 strings | every reason code in the catalogue was located with Grep and its enclosing function or constant resolved (§ 4) |

## 2. Status of the 64 entries

| status | count | ids |
|---|---|---|
| fixed | 18 | KI-002, 004, 006, 009, 011, 013, 017, 019, 020, 024, 027, 029, 040, 043, 045, 052, 054, 063 |
| fixed in part (a fix merged, a named part still open) | 4 | KI-028, 039, 057, 060 |
| open | 28 | KI-005, 007, 018, 023, 025, 030, 031, 032, 035, 036, 037, 038, 041, 042, 044, 047, 048, 049, 050, 051, 053, 055, 056, 058, 059, 061, 062, 064 |
| by design | 11 | KI-001, 003, 008, 010, 014, 015, 022, 026, 033, 034, 046 |
| product decision | 3 | KI-012 (D-113 §§ 1-2), KI-016 (D-114 § 2), KI-021 (D-113 § 3) |

Several "fixed" entries still carry a smaller open residual, stated in the entry (e.g. KI-004's audit guard, KI-020's
AI refinement, KI-013's pruner/gate disagreement). **Fixed but not measurable offline** (Stage 3 or Stage 6, ORCH-768
F-6): KI-002, KI-006, KI-009 (Phase-2 part), KI-013, KI-017. KI-041 has a product decision (D-110 AMENDMENTS 1-2)
but its lane was not merged, so it is counted open.

**Requested modes and where they are:** small budgets and empty `length` completions KI-054 (residual KI-055); the
3,600 s interaction cap and leg timeouts KI-028; `Ca++` KI-024 (class KI-023); empty reference-list chunk KI-027;
Stage-0 paraphrase `scope_conflict` KI-025; unreferenced species row KI-019; DMK/MK and the parenthetical bridge
KI-041 (shared locant KI-042); the Stage-1 recorder KI-045; the inferred alias stealing EntB's UniProt KI-017;
UniParc-only strain proteins KI-002; location and class-narrowing duplicates KI-043 and KI-029; the audit-patch
index defect KI-018; the invented wrapper KI-012; interaction-only pathways KI-020 and KI-021; withheld unsigned
interactions KI-008; the WNT PCP render crash KI-020; the late morning summary KI-063; the forced shutdown KI-061.

## 3. Findings made while cataloguing

1. **The interaction codes are composed, not literal.** A Grep for `protein_endpoint_missing_external_identity`
   finds nothing in `src/`: `ir.interaction_pwml_verdict` builds it as `"protein_endpoint_missing_" +
   "_and_".join(...)`. I first read the 0 hits as "not verifiable"; reading the function gives three exact values
   (`..._species`, `..._external_identity`, `..._species_and_external_identity`), all in the appendix.
2. **Driver `gate.<slug>` codes are not stable identifiers.** `batch/driver._slug_code` cuts a gate message at its
   first `:`, drops digits and truncates to 64 characters. A message with no colon keeps the entity names: the
   enterobactin stoichiometry error becomes `gate.generated_protein_complex_enterobactin_synthase_complex_componen`,
   and the Stage-3 identity error becomes `gate.protein_<name>_is_missing_a_uniprot_or_drugbank_identifier`.
   ORCH-772 must match those by message pattern; the appendix says so per row. `gate.unknown_transporter_reference`
   is derived by the same rule and was not seen in an artifact.
3. **`no_drawable_process` does not exist in the code.** It is the orchestrator's reading in D-113 § 3, not yet
   built. The appendix lists it so ORCH-773 can adopt the same string.
4. **D-062 and the driver disagree.** D-062 rules that a Stage-0 organism conflict preserves the pathway as
   `review_required` under the observed organism. `driver._finalize_scope_conflict` records `diagnostic_only` and its
   docstring says `review_required` "is NOT constructible at this seam ... escalated, not assumed". Sour taste got a
   PWML only because D-110 § 2 changed the request. KI-026; open question § 5 Q2.
5. **Two "codes" are harness labels, not product strings:** `refinement_review_crash` and `upstream_contract_report`
   live in `evidence/orch768_replay_corpus.py`. The first is catalogued (KI-020) with that caveat; the second labels
   legs whose upstream contract report blocked and is not a failure mode by itself (ORCH-768 § 8 F-8), so it is left
   out.
6. **The ORCH-767 process rule is not in `DECISIONS.md`.** "Every runner writes its own end-of-run summary" is
   recorded in `ORCH-767-RESULT.md` under D-112; D-112's text does not contain it. KI-063 says so.
7. **The machine crashes are three events, not two.** 2026-10-16 (ORCH-753 lost, ORCH-752's ledger row missing),
   2026-10-22 (ORCH-756 Ara), and 2026-10-25, which the LEDGER first called "crash #2" and then corrected to a
   forced shutdown. KI-061, KI-062.
8. **Where sources disagree, the later correction is used and the conflict is named:**
   - ORCH-755 § 3 counts (chlorophyll "31 single-draw", "3/3 draws on 14 of 14") are corrected by ORCH-758 § 6
     C1-C4; the catalogue uses the corrected numbers.
   - ORCH-758 D4's corroboration ("0 of 312 runs hold their summaries") is corrected by ORCH-759b § 8 F1 (186 leg
     directories hold detached-recorder summaries with `run_id ""`); KI-045 uses the correction.
   - Corpus pass rates differ by base: ORCH-769b reports 51 → 52/91 and ORCH-769c 51 → 52/91, each measured on
     `e3d5b65f`; LEDGER and `ORCH-769-RESULT.md` give the cumulative 53 → 54 → 56/91 on the merged sequence. Both are
     right; the catalogue quotes the cumulative figure.
   - LEDGER files the ORCH-754c merge under a 2026-10-20 row; the merge commit `ca852bb6` is dated 2026-10-19 14:38
     -0600. KI-052 uses the commit date.
   - ORCH-751 attributed the chlorophyll inflation to cofactors; ORCH-752 disproved it with the same payloads.
     KI-039 records the disproved theory.

## 4. How the reason codes were verified

Each code string in the appendix was searched in `src/t2pw` at `4befcb33` and its enclosing top-level function or
module constant read from the file. Strings that are not in `src/` say where they are instead: Streamlit's own
AppTest timeout message, the corpus harness labels, `no_drawable_process` (not built) and
`parenthetical_definition_bridge` (unmerged branch only). Line numbers quoted from RESULT docs are at those docs'
base SHAs; I re-resolved the function names at the tip but did not re-check every line number (§ 6).

## 5. Open questions nobody has decided (for the product owner)

**Q1. What counts in the D-114 coverage denominator?** D-114 § 1 counts "reactions + interactions + transports in the
final text record, surviving plus omitted". Two kinds of row are omitted by design and can never be drawn: unsigned
interactions (KI-008; WNT PCP draws 1 of 18) and duplicates of kept reactions (KI-014).
- (a) Count every row (the literal reading). Cost: a regulatory or signalling paper can never reach 75 %; WNT PCP sits
  near 6 % however good the extraction.
- (b) Leave rows PathWhiz cannot represent, and duplicates, out of the denominator and list them separately. Cost: a
  second number to explain; a lane could hide a real loss in the excluded class.
- (c) Print both figures. Cost: none in code beyond ORCH-772's text; the target needs naming against one of them.

**Q2. A Stage-0 organism conflict: `review_required` PWML (D-062) or `diagnostic_only` (driver)?** (Finding 4.)
- (a) Implement D-062: carry the run on under the observed organism and write a `review_required` PWML. Cost: a driver
  and release-status change; the run continues on a request known to be wrong.
- (b) Amend D-062 to what the code does. Cost: a correct paper with a wrong request yields nothing until re-requested.
- (c) Leave both; correct the request by hand, as D-110 § 2 did. Cost: the contract and the code keep disagreeing.

**Q3. When one actor of a reaction cannot be exported, drop the reaction or only the actor?** (KI-004, KI-005;
ORCH-769b § 9 item 1, still open.) This decides how ORCH-773 omits rows.
- (a) Drop the whole process (today). Cost: chlorophyll exports ALAD, CPO and PPO uncatalysed and loses eight
  regulators; heme loses its only reaction to a cofactor in the modifier slot.
- (b) Drop only the actor, keep the process, flag it. Cost: quarantine edits extracted content, which it was built not
  to do; needs its own corpus run.
- (c) Re-file compound cofactors upstream (Stage 3). Cost: a normalizer lane the corpus cannot measure (KI-058).

**Q4. Stage-0 paraphrase refusals** (KI-025): registered by D-110 § 1, not decided. Options: leave it (an outcome that
depends on the draw); loosen the match (a gate change, could admit a wrong paper); keep rewording requests by hand.

**Q5. Interactions-only pathways** (KI-021): confirm or overrule the orchestrator's reading `no_drawable_process`, and
decide whether the Stage-2 `no_reactions` decline should be aligned with it. Today a pathway that starts
interactions-only is declined and one that becomes interactions-only exports.

**Q6. Wrong compound identities in written PWMLs** (KI-030: menaquinone → Menadione, OSB, DMK). No gate catches them
and no lane is chartered. Options: a read-only diagnosis of the mapper-cache entries (about 1 h), or accept them as a
manual-review item in the text record.

**Q7. ORCH-754c Q6 was never done:** did any live-curator app run between 2026-10-15 and `ca852bb6` use the
ungated interaction-add prompt (KI-052)? Option: scan run trees for curator patches at `/processes/interactions/-` in
that window (read-only), or accept the exposure as nil.

## 6. What I could not verify

- **No PathWhiz import was run** by anyone. KI-034 to KI-037 rest on ORCH-761's static reading of the PathWhiz source.
- **KI-030's cause** (wrong compound ids) is not diagnosed; the entries cite the gold scorer only.
- **Line numbers** quoted from RESULT docs are at their own base SHAs; function names were re-checked at `4befcb33`,
  line numbers mostly were not.
- **Leg directories** under `runs_papers/` and the agent worktrees were not opened, except the morning summary; leg
  facts are as the RESULT docs and evidence files record them.
- **The 306 reviewer reports** were not opened one by one (they hold no verdict text).
- **Whether ORCH-773 has started:** branch `orch773/omit-per-row` exists at the tip with no commit of its own; not
  inspected further.
- **The cause of the 2026-10-16 and 2026-10-22 crashes** is not recorded anywhere.

## 7. Left out of the catalogue, deliberately

- **Pathway-mining work** (ORCH-748 pilot-004, ORCH-760): it writes no PWML; its failure modes live in
  `docs/pathway_mining/`.
- **Defects caught by review before a lane merged** (for example ORCH-754a B1-B3, ORCH-754b's text-versus-IR
  disagreement, ORCH-762 B1-B3, ORCH-763 S1, ORCH-765 rounds 0-2): they never shipped. The two that did ship in pushed
  code are catalogued: the chimeric merge (KI-040) and the curator prompt (KI-052).
- **Test-harness-only failures**: the evaluator import layering (ORCH-754d, fixed `ff3457fa`), AppTest files that
  stall a batched run, the GOLDEN coverage tripwire, the four primary-only DB-reachability tests (ORCH-754 § 5
  item 7). They affect gates, not PWMLs.
- **Open product questions that are not failures**: the `type="integer"` side effect (ORCH-747 D-open-1c), ORCH-747's
  interaction lines in the LLM-fed summary (D-104 AMENDMENT 2 § 3), the `.gitignore` of run scratch (ORCH-754 § 5
  item 1), and evidence quotes that elide the paper's letter labels (`evidence/orch756_gold_accuracy.txt`).

## 8. Gates

None apply: no code changed and nothing was run. The commit is docs only (`git commit -F`), and `git status
--porcelain` was checked afterwards.

## 9. LEDGER row

For the orchestrator to append (I did not edit `LEDGER.md`):

> **`ORCH-771` · 2026-10-25 — DONE (docs only), branch `orch771/known-issues`.** `PWML_KNOWN_ISSUES.md`
> catalogues 64 failure modes found from ORCH-747 to ORCH-769 (KI-001 to KI-064): 18 fixed, 4 fixed in part, 28
> open, 11 by design, 3 product decisions; five fixes are Stage-3/6 and not measurable offline. Its appendix maps 82
> machine reason codes and 19 human messages to KI ids, each code verified in `src/t2pw` at `4befcb33`; ORCH-772
> labels omissions from it. Driver `gate.<slug>` codes can embed entity names, so they are matched by message;
> `no_drawable_process` (D-113 § 3) is not in the code yet. `FINDINGS.md` F-204 to F-218 added. Open for the
> product owner: the coverage denominator for rows PathWhiz cannot draw (unsigned interactions, duplicates); D-062
> versus the driver on Stage-0 organism conflicts; whole-process versus actor-only omission. No Python run. Record:
> `ORCH-771-RESULT.md`.

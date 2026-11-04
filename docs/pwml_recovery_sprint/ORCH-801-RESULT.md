# ORCH-801 — three new papers, text-to-PWML accuracy check · 2026-11-04

**Requested by the PO:** check that everything is on `main`, then run the pipeline on 3 papers we have not
run before, to test text-to-PWML conversion and accuracy. No test suites.

**Code:** `53331c83`, which is `main`, `origin/main` and `sprint/pwml-recovery`. The legs ran from the pinned
worktree `.claude/worktrees/orch801run`.
**Setup:** strict mode, 3 draws, 10,800 s per leg, one leg at a time under the heavy lock (`orch756_paper_runs.sh`
via `evidence/orch801_night.sh`, a copy of `orch798_night.sh` with only the conditions changed).

## 1. Paper choice

All 14 conditions in `docs/evaluation_v2/paper_runs_orch755/topics/` had already been run. Three ORCH-753
gold-cohort source papers that had never been run were added as new topic files. They are metabolic and come from
three different organisms. None has a gold case: the cohort holds only source manifests.

| condition | paper | topic line |
|---|---|---|
| `salicylic_acid` | PMC12408352, Zhu et al. 2025 Nature | Salicylic acid biosynthesis from phenylalanine, *Oryza sativa* (first written as *Arabidopsis thaliana*, see § 3) |
| `cdadc1_gemcitabine` | PMC12184417, Rodriguez et al. 2025 PNAS | dCTP deamination and gemcitabine/decitabine metabolism by CDADC1, *Homo sapiens* |
| `tb_pyrimidine` | PMC5098729, Leija et al. 2016 PLoS Pathog | Pyrimidine salvage and deoxypyrimidine nucleotide biosynthesis, *Trypanosoma brucei* |

## 2. Results

All four legs left 0 surviving processes. `g11_evidence.py check --task ORCH-801` exits 0.

| leg | G11 report | outcome | PWML | coverage | release | min |
|---|---|---|---|---|---|---|
| salicylic_acid, attempt 1 | `g11/ORCH-801/01` | `scope_conflict`: organism | none | - | - | 32.0 |
| cdadc1_gemcitabine | `g11/ORCH-801/02` | PASS (warnings) | yes | 6/6 (100%) | `review_required` | 21.4 |
| tb_pyrimidine | `g11/ORCH-801/03` | PASS (warnings) | yes | 10/10 (100%) | `review_required` | 40.0 |
| salicylic_acid, attempt 2 | `g11/ORCH-801/04` | `scope_conflict`: pathway wording | none | - | - | 45.9 |

The per-leg review pack, `MORNING_SUMMARY.md` and the audit are in `runs_papers/ORCH-801/`, which is untracked like
every earlier `runs_papers/` run. The leg directories are in the `orch801run` worktree.

## 3. Findings

1. **Attempt 1 refused the salicylic acid paper because of an operator error.** I wrote *Arabidopsis* on the topic
   line, but the paper is about rice. Stage 0 read *Oryza sativa* and correctly refused. This is not a defect.
2. **Attempt 2 was a false refusal.** Stage 0 read "salicylic acid biosynthesis (PAL pathway) in *Oryza sativa*"
   and refused the requested "Salicylic acid biosynthesis from phenylalanine". These name the same pathway: PAL is
   phenylalanine ammonia-lyase, the first step from phenylalanine. It looks like the same class as `F-203` and the
   ORCH-734 scope defect, where a correct paper is lost over wording. It cost 46 min and produced no PWML. Not
   investigated in code.
3. **Accuracy audit** by `pwml-bio-auditor`, read-only; full report `runs_papers/ORCH-801/ACCURACY_AUDIT.md`:
   - cdadc1: 5 of 6 reactions correct and 1 partly wrong. 4 steps the paper reports are missing (DCTD, TYMS,
     dTMP→dTTP, CDA on dFdC). No identity errors.
   - tb: all 7 distinct reactions correct. 3 are duplicates on split nodes (dThd vs thymidine, dCtd vs
     deoxycytidine). Thymidylate synthase and uridine phosphorylase are missing.
4. **DUT was injected as a catalyst it is not** (cdadc1 R1, `product_contract_violation`). The lead checked the
   cause: `enzyme_cues.cue_near_name` (`src/t2pw/pipeline/enzyme_cues.py:81`) runs
   `re.finditer(re.escape(name), text, re.IGNORECASE)` with no word boundaries, so "DUT" matches inside "dUTP".
   Any short gene symbol that sits inside a compound name can do the same.
5. **Split nodes:** the name gate refuses the abbreviation equivalences dThd = thymidine and dUMP = deoxyuridine
   monophosphate, which duplicates reactions (auditor confidence: medium).
6. **Arabidopsis appears in the tb PWML's species list.** All 4 tb enzymes are backed by the PathBank "Unknown"
   placeholder, which is tagged *Arabidopsis*. This is the standing `placeholder_backed_proteins` policy question.

## 4. Not settled: for the PO

- **Fix finding 4?** A boundary-aware match, `(?<!\w)…(?!\w)` rather than `\b`, because names end in `+`. This is
  one lane: implementer, reviewer, corpus no-loss check. Its G9 proof is cdadc1 R1 losing DUT. Risk: a real
  injection that relied on substring matching would be lost, and the corpus no-loss check covers that.
- **Fix finding 2?** Either loosen the Stage-0 pathway-scope comparison so equivalent wording passes, or re-run with
  the topic worded as Stage 0 reads it. The first changes a gate and needs a ruling. The second only gets a PWML for
  this paper.
- **Fix finding 5?** It changes what the name gate admits, so it needs a ruling. Held under the no-regression
  instruction.
- **Promote these three papers to gold cases?** Without gold, the recall gaps in finding 3 cannot be scored.

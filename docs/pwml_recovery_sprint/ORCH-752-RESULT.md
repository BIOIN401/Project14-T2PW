# ORCH-752 — cofactor-blind dedup key for Stage-1 ensembling, and what it did not fix

**Status:** delivered, uncommitted (no git state was changed; another session owns the
checkout). Base `c9265fb8`. Boundary: `src/t2pw/pipeline/stage_one_ensemble.py`,
`tests/test_stage_one_ensemble.py`, evidence. Evidence `evidence/g11/ORCH-752/01`–`05`,
pins `evidence/g11/pin/ORCH-752/`.

**Headline:** the key is now cofactor-blind as directed, and that is correct in principle —
but on the chlorophyll payloads it changes **nothing: 41 → 41**. ORCH-751 § 2.2's diagnosis
("draw 2 listed cofactors the others omitted") is **disproved by its own payloads** (§ 2).
The inflation is name-form variance in *defining* participants, which no lexical key may
collapse without a product ruling (§ 5).

---

## 1. The change

### 1.1 Reaction key

`(frozenset of defining input keys, frozenset of defining output keys)`. A participant is a
cofactor when its dict carries `role: cofactor` (the evaluator's `gold.ROLES` value) or its
`strict_key` is in `CURRENCY_KEYS`:

```
atp adp amp · nad nadh nadp nadph · fad fadh2 · coa "coenzyme a" · h2o water ·
pi ppi phosphate diphosphate · h proton · co2 o2 ·
sam "s adenosyl l methionine" "s adenosylmethionine" · sah "s adenosyl l homocysteine" "s adenosylhomocysteine" ·
mg2 mg fe2 fe3 fe zn2 zn mn2 mn ca2 ca cu2 cu k na cl
```

Whole-name match on `strict_key` output; no stemming, no fuzzy matching. **A side holding
nothing but currency keeps every name**, so `ADP + Pi -> ATP` stays distinct from
`GDP + Pi -> GTP` and a chelatase's metal still counts when it is the only input. Two rows
that differ in any defining participant never collapse (pinned:
`test_a_defining_participant_always_separates`).

When a later row collapses onto a survivor, the survivor keeps its `evidence` and its own
participants verbatim and gains, per side, any cofactor the later row named that it did not
(`_donate_cofactors`). The donation is listed on the side-file row as `cofactors_added`;
the payload schema gains no key.

`pipeline.cofactor_policy._CURRENCY` was read and deliberately **not** reused: it is
"unwired by design", normalises with a different fold, files CoA/SAM/SAH under `_HUB` with
glutamate and 2-oxoglutarate, and has no metal ions. Cited in the module comment.

### 1.2 Interaction key — changed, on payload evidence

`(strict_key(entity_1), relationship type, strict_key(entity_2))` where the relationship is
`ir.PWML_INTERACTION_TYPES[_interaction_type_key(rel)]` if the exporter's exact-match table
knows the word, else its `strict_key`. Evidence: the chlorophyll draws pooled 23 interaction
keys, of which three pairs were the same edge in verb and noun form — `hy5 | activates | chlh`
(draw 2) vs `hy5 | activation | chlh` (draw 3), likewise `toc1 → chlh` and `scl27 → porc`.
The exporter already emits one `Activation` for both, so they are duplicates by the product's
own definition. **23 → 20.** `binds` and `interacts_with` stay apart: the table carries no
sign-free word (merge rule 8 reasoning in `ir.py`). This is a verb-form fold, not a cofactor
fold; it is one function and one line, separable if the orchestrator wants it reviewed apart.

Note: the evaluator (`bench/eval_v2`) does not score interactions at all, so ORCH-750's claim
that this key is "the key the evaluator uses" held only for reactions.

### 1.3 Transport key — unchanged

All three draws carried 0 transports. No evidence; not touched.

## 2. The measurement — `evidence/orch752_chlorophyll_key_inflation.{py,json,txt}`

Payloads copied from `C:/t/cohort/PW766996_PMC5130987/` to
`evidence/orch752_chlorophyll_payloads/` (sha256 in the JSON). BEFORE is ORCH-750's
`row_key` restated verbatim in the script; AFTER is the tip's `merge_draws`. Run bounded:
`g11/ORCH-752/03-measure.json`.

| bucket | rows per draw | pooled keys before | after |
|---|---|---|---|
| reactions | 16, 19, 18 | **41** | **41** |
| interactions | 0, 7, 16 | 23 | **20** |
| transports | 0, 0, 0 | 0 | 0 |

Reaction keys shared by pair, before → after: 1→2: 1→1 · 1→3: 11→11 · 2→3: 1→1.

**Why nothing collapsed.** The script checks, for every defining key, whether its cofactor
set differs between draws: **none does.** The six currency names written (`Fe2+`, `Mg2+`,
`ATP`, `S-adenosyl L-methionine` ×3 each; `NADPH` ×2; `NADP+` ×1) appear on the same
reactions in every draw that has them. The merge donated **no** cofactor.

**What the 30 unshared keys actually are** (full list in the `.txt`):

- **Draw 2 wrote `full name (ABBR)` where draws 1 and 3 wrote `ABBR`** — 17 of its 19 rows:
  `glu 1 semialdehyde gsa` vs `gsa`, `5 aminolevulinate ala` vs `ala`, `protoporphyrin ix`
  vs `proto ix`, `3 8 divinyl protochlorophyllide dv pchlide` vs `dv pchlide` (draw 3) vs
  `3 8 divinyl protochlorophyllide` (draw 1). This is why draw 2 shares 1 key with each of
  the others and that one key is `chlide a -> chlide b`, the only row all three abbreviated.
- **Synonyms:** `protoheme` (1) / `heme b` (3) / `protoheme heme b` (2).
- **Typos, one per draw:** `geranyllgeraniol` (1), `porphobilogen`, `hydroxymethybilane` (2),
  `bilverdin` (3).
- Draws 1 and 3 genuinely differ on 5 + 7 rows of the same chemistry at different
  abbreviation levels (`3 8 divinyl protochlorophyllide` vs `dv pchlide`).

Every one of these is a difference in a *defining* participant. The task's own rule — never
collapse rows that differ in a defining participant — forbids a lexical key from touching them.

## 3. Proofs and gates

| run | report | result |
|---|---|---|
| focused, **pre-change module** (G9 behavioural base) | `g11/ORCH-752/01-focused-pre.json` | **3 failed, 18 passed**, 0 survivors. The 3 are the new-behaviour tests; `test_a_defining_participant_always_separates` passes on both sides, as a preservation test should |
| focused, tip | `g11/ORCH-752/02-focused.json` · pin `pin/ORCH-752/02-focused.pin.json` | **21 passed**, 0 survivors |
| measurement | `g11/ORCH-752/03-measure.json` | exit 0, 0 survivors; § 2 |
| N=1 identity vs `c9265fb8` | `g11/ORCH-752/04-n1-identity.json` · verdict `evidence/orch752_n1_identity_verdict.json` | **`n1_behaviourally_identical: true`** — "N=1 IDENTICAL to base", 4/4 cases, 0 survivors |
| SMOKE (22 files) | `g11/ORCH-752/05-smoke.json` · pin `pin/ORCH-752/05-smoke.pin.json` | **508 passed in 62.34 s**, 0 survivors |

G9 labelling: this is a **correction** of ORCH-750's delivered behaviour. The module is
untracked at `c9265fb8`, so a base-SHA failure is impossible by construction; the proof is
the certified failure on the file as delivered (`01-focused-pre`), stated in the test file's
section header. Every run went through `bounded_run.py` + `pinned_pytest.py`,
`PYTHONPATH=<repo>/src`, fresh `--basetemp` under `C:/t/bt-orch752/`, no `--` after
`pinned_pytest.py`; every cleanup report shows `final_surviving_count: 0`.

## 4. What I believed and what disproved it

I took the task's premise — cofactor presence explains 41 — as given until the pre-change
key list showed `atp + mg2 + proto ix -> mg proto ix` carried its ATP and Mg2+ in *all three*
draws. The cofactor-blind key was then measured before the module was edited (41 → 41) and
again after (41). The theory is disproved; the fix was shipped anyway because the key was
wrong in principle, aligns with the evaluator, and costs nothing when it does not fire.

## 5. Open for the product owner

1. **Abbreviation bridging across draws** is the whole chlorophyll inflation. The evaluator
   refuses to credit a parenthetical's inside (`name_conflict`, `matcher.inner_key`); the
   ensemble cannot do so either without a ruling. Options: (a) accept and surface the
   union-to-mean ratio ORCH-751 § 3 item 4 asked for (the measurement JSON now computes it:
   2.321 for reactions); (b) a prompt-side spelling instruction — `src/t2pw/llm/prompts/` is
   outside this boundary; (c) rule that the outer part of `X (ABBR)` and `ABBR` are one name
   *for pooling only*, contradicting evaluator policy. (a) costs nothing; (c) changes what
   "distinct row" means.
2. **`induces` / `stimulates` / `activates` → one `Activation` edge** at pooling. Mirrors the
   exporter; a PO may prefer the stricter per-word key. One-line revert.
3. **Metal ions in `CURRENCY_KEYS`.** Kept per the directive; the all-currency-side rule
   protects a chelatase whose metal is its only input, and the product side differs anyway.
   Removing them is a list edit.
4. Bare `pi` is in the list although `cofactor_policy` dropped it (phosphatidylinositol). The
   failure direction here is "keeps a duplicate", never "loses a reaction", so it stays.

## 6. Not done

- LEDGER row and any `DECISIONS.md` entry: orchestrator / product owner.
- No commit, no branch, no `.staging` placeholder left (`g11_evidence.py check` run below).
- The N>1 path end to end on a live leg was not re-exercised; the measurement is offline on
  the three ORCH-751 draws, as ORCH-751 § 5 chose for independence.

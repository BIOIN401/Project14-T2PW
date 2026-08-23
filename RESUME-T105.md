# Resume point — T-105 is ready to run

Written 2026-08-22 when the session was paused for a machine restart. Everything below is
verified, not assumed.

## State at pause

| check | value |
|---|---|
| integration tip | `6188758` — pushed, `local = origin = git ls-remote` |
| merge in progress | none |
| staged files | 0 |
| heavy lock | **free** (stale `postmerge-val` lock removed after proving pid 375756 dead) |
| sprint-owned Python processes | **0** (only the two `ms-python.isort` IDE servers) |
| product-owner edit | intact, 35 ins / 2 del, `sha256:e50a248bb7189c22…` |
| SMOKE | **473**, measured after both merges |

**Both T-105 blocker corrections are merged.** D-063's precondition is satisfied.

- **C-072** (F-094) merged `d7f4f96` — a declared core with unmatched anchors is never
  `release_ready`. Approved, zero correction rounds.
- **C-073** (F-096) merged `6373ad1` — an accession claimed across incompatible entity kinds does
  not ship. Approved after one evidence-backed REJECT and one correction round.

Full records: `LEDGER.md` C-072 and C-073 entries; findings F-098 and F-099 in `FINDINGS.md`.

## What was interrupted, and how to resume it

The post-merge affected-paper validation was stopped mid-leg. **Nothing is corrupted.**

- Run dir: `runs_verify/2026-08-22_2017` — 2 of 4 legs finished
  (PMC12856317 strict + research), PMC12452463/strict was in flight, PMC12452463/research pending.
- To resume, **omit `--fresh`** so it continues that directory and skips the finished pairs:

```bash
PYTHONPATH="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/src" \
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \
  --label postmerge-validate-resume --timeout 6000 --heavy-lock postmerge-val \
  --json <fresh T-105 g11 slot> \
  -- .venv/Scripts/python.exe -u scripts/batch_run.py \
       --topics topics_postmerge_validate.txt --out runs_verify \
       --modes strict,research --timeout 1800 --deadline 2
```

- One **explained** leftover: `evidence/g11/T-105/.staging/09-postmerge-validate.json` is the
  reservation for the interrupted job. It was never promoted because the job did not finish. Leave
  it or supersede it with a fresh allocation; it is not an unexplained record.

## Honest status of the paper-level validation

**Neither card's refusal path has been exercised on a live leg.** Both have deterministic
replay evidence over committed artifacts instead. Do not write either up as "confirmed in
production" on the current evidence.

- **C-072** — attempt 1 (`runs_verify/2026-08-22_1821`) reached the contract outcome
  (`review_required`, `strict_acceptance_eligible=false`, no bare `pathway.pwml`) but via the
  SEMANTIC gate, so the new cap never fired. That draw would have behaved identically on the base
  SHA. The cap fires only when the semantic gate does not, so exercising it is stochastic.
- **C-073** — PMC12856317/research came back with `ALAS2` carrying only
  `uniprot`/`pathbank_protein_id`/`gene_name`, i.e. **no `drugbank:DB00114`, so no collision
  existed** and Pass B correctly did nothing. What the leg DID confirm is the property the
  rejected first version violated: `heme`, `Glycine`, `aminolevulinic acid`,
  `Pyridoxal 5'-phosphate` and `succinyl-CoA` all kept full accession sets. **Zero collateral,
  verified in production.**

This is the standing temperature-0 draw-variance trap, now observed on both cards.

## T-105 — all preconditions cleared

Verified this session: LM Studio up at `http://127.0.0.1:1234/v1` with
`text-embedding-nomic-embed-text-v1.5`; all nine `OPENROUTER_*_MODEL` slots on
`deepseek/deepseek-v4-flash`, live-pinged successfully; account $70.90 remaining;
gold set validated (10 cases, 4 strict-exportable); `--verify-topics topics_t104.txt` →
`verdict: OK`, 10/10 `[pinned_override]`, 0 search calls.

Commands are in `HANDOFF-CORRECTIONS-T105.md` §5. Preflight first (`--fresh --stage-only`), then
the real run **without `--fresh`**, both in background because the timeouts exceed the tool clock.

## Predict before scoring — the full prediction is in the LEDGER entries

**Priority 1 will be 7, unchanged from T-104. Do not record that as a C-073 failure.**
The source-support pass that would catch the `succinyl-CoA` hallucination is dormant, and the
conflict C-073 does fix is not counted by priority 1 at all (`semantic.py:908` appends the
finding without incrementing `false_real`).

**C-072 demotes three legs, not one** — PMC12452463/strict plus PMC12096016/research and
PMC12782028/research, all of which were `release_ready` at T-104 with unmatched anchors.
**It moves no acceptance rate:** strict PWML stays 0/4 (PMC12452463 is gold `partial_only` and was
never in that denominator) and research deliverable stays 4/8 (`acceptance.py:605` is a filename
test that never reads release status).

**Expected identical, not findings:** 6 × `scope_conflict`, 2 × TIMEOUT (PMC12444477),
2 × `no_reactions` (PMC13231680, a declared negative control the gold calls correct).

## The two rulings waiting on the product owner

1. **Arm C-073's Pass A.** Two lines — `pipeline.merge_additions` forwarding a `source_text`
   kwarg, and `source_text=text` at `streamlit_app.py:5606`. Blocked only because that file
   carries the uncommitted 35-line edit. No code problem. Until it lands, the gold's designated
   HALLUCINATION TEST stays unfixed. Note the ≈55 KB/leg payload growth and the C-060 A6
   narrowing, both disclosed in the C-073 LEDGER entry.
2. **F-099** — make the compound resolver consult `rejected_mapped_ids` before a name-keyed
   lookup. Touches a shared path used by every compound on every leg, so it needs its own card.
   Not reachable until Pass A is armed; card them together.

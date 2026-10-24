# ORCH-768: the frozen replay corpus (D-112 § 1). RESULT

**Lane:** `orch768/replay-corpus`, base `e3d5b65f` (= `main` = `sprint/pwml-recovery`). The implementer was a
general-purpose agent running unattended. Not merged, not pushed, not self-approved: an independent reviewer judges
the diff. **No `src/` change.** The code is offline only: no LLM call, no pipeline run, no live network.

| commit | what |
|---|---|
| `8d38256c` | harness `evidence/orch768_replay_corpus.py`, manifest `evidence/orch768_corpus_manifest.json`, acceptance test `tests/test_orch768_replay_corpus.py`, fixture `tests/fixtures/orch768/` |
| docs commit | this file, `evidence/orch768_corpus_baseline.{txt,json}`, `evidence/orch768/` (fidelity, worked examples, audit-loop scan), `evidence/g11/ORCH-768/01-80`, `evidence/g11/pin/ORCH-768/` |

**Outcome.** The corpus has **100 legs**: 37 from the five paper runs, whose code SHA is recorded (tier A), and 63
usable `runs_verify` legs (tier B). Of these, **91 are strict legs with an outcome** and **81 have a no-LLM
checkpoint**. Each of the 81 was replayed at the code that ran it, and **81/81 reproduce the recorded verdict**:
PWML yes/no, quarantine verdict and refusal reasons, release status, the final gate error by error, the
required-field gate codes, and the export error. Byte-level checks on the same runs:

* all **43** PWMLs written are **byte-identical** to the leg's own PWML;
* all **73** canonical payloads re-hash to the leg's final gate report;
* the **decision-input hash** matches on 81/81;
* there were **0** network attempts.

**Baseline at `main`** (`evidence/orch768_corpus_baseline.txt`): PWML **51/91** strict legs (**51/81** replayed).
That is **30 replayable failures** and **10 LLM-originating failures** (no checkpoint), with 2 aborted legs not
counted. Tier A is 8/28, tier B 43/63. Two runs under different `PYTHONHASHSEED` gave **byte-identical** JSON and
text.

---

## 1. What the code runs, and so where the checkpoint is

`streamlit_app.run_post_pipeline_sbml_artifacts` (`:3353`) runs these steps in order:

1. Stage-2 `map_payload` and `normalize_process_payload`. No LLM.
2. **The audit loop** (`run_audit`, `use_llm`) and **the gap resolver** (`run_gap_resolution`). LLM.
3. **The curator** (`run_pathway_curator`, temperature 0.2, unless `T2PW_OFFLINE_CURATOR`). LLM.
4. Stage-6 `map_payload(use_cache=False)`. Network (UniProt, PathBank DB), no LLM.
5. `run_enrichment`, then `run_prefreeze_resolution`.
6. `freeze_canonical_payload` (`:3245`): the quarantine, then the final pre-export gate.
7. SBML build and `run_sbml_overwatch` (LLM, SBML only; nothing it returns reaches the PWML).

The app then calls `initialize_refinement_review_state` (`:1871`: draft graph and PNG), and the batch driver clicks
"Generate PWML", which calls `run_pwml_export` (`:4612`). That function runs the Stage-8 revalidation, the
required-field gate, the pre-freeze seam, the IR, the writer and QA.

`batch/driver.py` writes the leg files:

* `stage1_payload.json`: session `stage_one`.
* `merged_payload.json`: session `final_payload`, the post-pipeline **input**, before steps 2 and 3.
* `final_mapped.json` (`_add_identity_artifacts`, `:1801`): the **canonical** payload when the boundary passed.
  When it refused, the file is `final_mapped_db` (post-Stage-6, **pre-enrichment**).

So **`final_mapped.json` is the latest checkpoint after which no LLM runs.** Its kind is settled by hash:

* **canonical**: the file hashes to the recorded final gate's `payload_sha256`. 73 legs.
* **pre_freeze**: it does not. 8 legs. The replay first runs `run_enrichment` on a pinned cache, then
  `run_prefreeze_resolution`.

`merged_payload.json` is **not** a no-LLM checkpoint. In 15 of the 18 tier-A legs with a checkpoint, the LLM audit
or gap loop changed the payload after it (`evidence/orch768/audit_loop_changes.{py,txt}`). The curator's effect is
not recorded in the leg at all. Stage 6 cannot be replayed offline either: it runs `use_cache=False`, and no leg
recorded its UniProt or PathBank answers. See § 9, item 1.

**What the replay runs.** It calls the production functions of the `--src` tree, never copies of them:

1. `freeze_canonical_payload`, which re-runs the quarantine. On a canonical payload this is idempotent: 73/73
   re-hash exactly.
2. The batch driver's own verdict. `gate_reports.gate_verdict`, `_blocking_reports` and
   `_superseded_contract_reports` run on an artifact set made of the leg's recorded top-level `*_contract_report`
   objects plus the replayed final gate report and canonical payload.
3. `initialize_refinement_review_state`.
4. `run_pwml_export`, with the app's session defaults (`"Generated Pathway"`, `Metabolic`, 3200×1400, `strict_db`)
   and the replayed quarantine decision.
5. The driver's `_frozen_release_record` and `_pwml_artifact_name`, for the file name or withholding.

The `streamlit` module is the established `MagicMock` stub (`tests/test_prefreeze_third_export_seam.py`).
`streamlit_app.PROJECT_ROOT` points at the per-leg scratch directory while a leg replays. That keeps the one
diagnostic writer keyed on it (`_write_reaction_preservation_report`) out of the measured tree.

## 2. Inventory and checkpoint per leg (`evidence/orch768_corpus_manifest.json`)

**Code SHA per leg.** Tier A takes the `head=` from the run's `night.log` start line (the latest start line at or
before the leg's stamp). It is equivalent by `src` tree id:

| run | head | `src` tree = | legs |
|---|---|---|---|
| ORCH-755 | `a2ed7284` | `a2ed7284` | all 14 |
| ORCH-756 | `8b074dda` | `a2ed7284` (identical `src`) | enterobactin 2020, menaquinone 2048, chlorophyll 2109 |
| ORCH-756 | `f392d447` | `f392d447` | ara 0947, ara 1018, sour 1035 |
| ORCH-759 | `82b4fc33`, `13f8a5e2` | `f392d447` (docs-only HEAD moves; `git diff` on `src` empty) | all 7 |
| ORCH-759p | `21155be0` | `21155be0` | enterobactin, menaquinone, sour_taste_v2 |
| ORCH-759p | `360a6ac5` | `360a6ac5` = `d6849c9e` = `e3d5b65f` (one tree, `9b1943a8`) | chlorophyll |
| ORCH-767 | `d6849c9e` | the same tree | all 6 |

**Tier B (`runs_verify`).** No SHA is recorded in those runs. The SHA is **the primary checkout's HEAD at
`plan.json` `created_at`, read from its reflog**; the state of the working tree at that time is not recorded. That
gives 12 trees, from `671a5ae3` to `cd73cdbf`. A tier-B leg is "usable" when it is strict and has
`final_mapped.json` and `quarantine_report.json`: 63 of the 111 strict `runs_verify` legs. The other 48 died before
Stage 6 or predate the persisted quarantine set (the 2026-08-04 runs).

**Tier A, every leg** (`cp` = checkpoint; `own` = verdict at its own tree; class and detail at `main`):

| leg | cp | recorded outcome | own | at `main` |
|---|---|---|---|---|
| 755 enterobactin 2201 | pre_freeze | refusal `unexportable_entity:1` | yes | replayable: same refusal |
| 755 menaquinone 2238 | none | timeout (3,600 s cap) | n/a | LLM-originating |
| 755 chlorophyll 2338 | none | timeout | n/a | LLM-originating |
| 755 ara_operon 0039 | none | timeout | n/a | LLM-originating |
| 755 sour_taste 0139 | canonical | Stage-8 revalidation `actor_schema_not_canonical` | yes | replayable: same |
| 755 wnt_pcp 0207 | none | timeout | n/a | LLM-originating |
| 755 wnt_ca 0307 | none | Stage 1 "Chunk 3 failed to produce valid JSON" | n/a | LLM-originating |
| 755 × 7 research | none | research mode | n/a | not in the PWML corpus |
| 756 enterobactin 2020 | canonical | PWML | yes, PWML identical | pass; gold final and PWML P 0.714 R 1.000 F1 0.833 |
| 756 menaquinone 2048 | canonical | PWML | yes, PWML identical | pass; gold P 0.889 R 0.778 F1 0.830 |
| 756 chlorophyll 2109 | none | timeout (5,400 s) | n/a | LLM-originating |
| 756 ara_operon 0947 | none | no manifest row | n/a | aborted, not counted |
| 756 ara_operon 1018 | none | Stage-0 `scope_conflict` | n/a | LLM-originating |
| 756 sour_taste 1035 | none | no manifest row | n/a | aborted, not counted |
| 759 enterobactin 1037 | canonical | final gate, degree-0 `enterobactin synthase` (E1) | yes | replayable: same; gold final P 1 R 1 |
| 759 menaquinone 1148 | canonical | required gate `species_missing_taxonomy`/`_classification` | yes | replayable: same; gold final P 0.875 R 0.778 |
| 759 chlorophyll 1231 | pre_freeze | refusal `entity_type_overlap:4` | yes | **moved**: the quarantine passes (759c); now required gate `reaction_enzyme_must_be_protein_complex` (CPO; Stage 6, see F-6) |
| 759 ara_operon_v2 1359 | none | `scope_conflict` | n/a | LLM-originating |
| 759 sour_taste 1421 | none | `scope_conflict` (organism) | n/a | LLM-originating |
| 759 wnt_pcp 1432 | canonical | PWML | yes, PWML identical | pass |
| 759 wnt_ca 1528 | none | timeout during extraction | n/a | LLM-originating |
| 759p enterobactin 2101 | canonical | PWML | yes, PWML identical | pass; gold P 1 R 1 F1 1 |
| 759p menaquinone 2154 | canonical | required gate, species | yes | replayable: same; gold final P 1 R 1 (9/9) |
| 759p sour_taste_v2 2233 | canonical | PWML | yes, PWML identical | pass |
| 759p chlorophyll 2301 | canonical | PWML | yes, PWML identical | pass |
| 767 enterobactin 2224 | canonical | final gate, wrapper component | yes | replayable: same; gold final P 1 R 1 (5/5) |
| 767 chlorophyll 2328 | canonical | final gate, 9 × unknown modifier | yes | replayable: same |
| 767 sour_taste_v2 0042 | canonical | PWML | yes, PWML identical | pass |
| 767 wnt_pcp 0131 | canonical | crash opening refinement review | yes | replayable: same |
| 767 wnt_ca 0314 | pre_freeze | refusal `unexportable_entity:1` | yes | replayable: same |
| 767 menaquinone 0503 | canonical | PWML | yes, PWML identical | pass; gold final and PWML P 0.909 R 1.000 F1 0.952 |

**Tier B** has a line per leg in `evidence/orch768_corpus_baseline.txt` and the fidelity files.

## 3. Fixtures and the no-network proof

* **The pinned inputs.** The manifest pins 587 input files by sha256: 27.9 MB of leg files and 7 distinct
  enrichment caches totalling 319 MB. The leg files stay where they are (untracked in the primary and in the
  `orch756` and `orch767` worktrees). The replay refuses any file whose hash moved (`input drifted`).
* **The cache for a pre_freeze leg** is the enrichment cache as it stood when the leg **ended**. That is the next
  leg's start snapshot: the runner snapshots `data/` at each leg start. For tier B it is the next run's snapshot.
  With the leg's own start snapshot, the ORCH-767 Wnt/Ca2+ replay made **39 refused network attempts** (cache
  misses) and produced a different enriched payload (F-3, G11 `08`). With the end-of-leg cache it makes 0.
* **The offline name index** is each tree's own `data/pathwhiz_id_db.json`. The blob is identical in all 17 trees
  used (`75d2e5f4`), and so is `reference/PW000001.pwml` (`1833bc4e`). Both are exported with the tree.
* **The gold full texts** are `C:/t/eval-fulltext/PMC12096016.txt` and `PMC12421875.txt`, pinned by sha256 in the
  manifest.
* **The no-network proof.** `Offline` replaces `socket.socket.connect`, `connect_ex`, `socket.create_connection`
  and `socket.getaddrinfo` **before `t2pw` is imported**: every call raises and is counted. `probe_offline` makes
  three real attempts at the start of every run: `urlopen` to UniProt, a loopback connect and a raw socket. All
  three must come back `refused` and be counted, or the run stops (`ORCH768_NETWORK_NOT_BLOCKED`, exit 3). Every
  run passed the probe; it is printed on line 2 of each `.txt`. Each leg records the refused attempts it made:
  **0 on all 81 legs, in every run.**
* **Committed test fixture.** Three ORCH-767 legs (sour taste, enterobactin, WNT PCP), gzipped with `mtime=0`,
  about 190 KB, under `tests/fixtures/orch768/`. The test proves each decompressed file has the corpus manifest's
  sha256.

## 4. Fidelity: every leg at the code that ran it (`evidence/orch768/fidelity-<sha>.{json,txt}`)

| tree | legs | as recorded | G11 |
|---|---|---|---|
| `a2ed7284` | 755 enterobactin and sour; 756 enterobactin and menaquinone | 4/4 | `60` |
| `f392d447` | 759 enterobactin, menaquinone, chlorophyll, wnt_pcp | 4/4 | `61` |
| `21155be0` | 759p enterobactin, menaquinone, sour_taste_v2 | 3/3 | `62` |
| `360a6ac5` | 759p chlorophyll; 767 × 6 | 7/7 | `63` |
| 12 tier-B trees (`671a5ae3` … `cd73cdbf`) | 63 legs | 63/63 | `64`–`75` |

Every tree is a `git archive <sha> src reference/PW000001.pwml data/pathwhiz_id_db.json` export under
`C:/t/bt-orch768/src-<sha>`. The harness refuses to run (`ORCH768_WRONG_TREE`) unless `t2pw` resolves to
`<src>/t2pw`.

Fidelity compares, per leg: the outcome class, the quarantine `ok`, the refusal reasons, the release status, the
final gate `ok` and **its full `[path, reason]` error list**, the required-gate `ok` and codes, the export error
and the PWML name. The recorded reports are read by the measured tree's own functions. **No leg failed to
reproduce.** Two byte-level limits are stated, not hidden:

* **pre_freeze legs (8)**: every verdict reproduces, but the payload the leg quarantined does not hash the same
  (0/8). The cause is the PathBank DB, which is unreachable at replay. On the one leg whose compounds needed no DB
  lookup (`runs_verify/2026-08-24_1203/PMC12856317`), setting the payload's DB marker back to
  `{"available": true}` reproduces the leg's `admitted_payload_hash` **exactly**. On the other 7, compound
  resolution reports `db_unavailable` (F-4).
* **The PathBank DB is also absent on canonical legs**, and their PWMLs are still byte-identical. The post-freeze
  seam's re-resolution does not change the written bytes on these 43 legs.

## 5. Baseline at current `main` (`evidence/orch768_corpus_baseline.{txt,json}`, G11 `58`; repeat `59`)

| | all | tier A | tier B |
|---|---|---|---|
| strict legs with an outcome | 91 | 28 | 63 |
| with a no-LLM checkpoint | 81 | 18 | 63 |
| **PWML written** | **51** (51/91; 51/81) | 8 | 43 |
| replayable failures | 30 | 10 | 20 |
| LLM-originating failures | 10 | 10 | 0 |
| aborted, not counted | 2 | 2 | 0 |
| outcome unchanged since the leg ran | 53/81 | 17/18 | 36/63 |
| network attempts | 0 | 0 | 0 |

**Determinism.** Seed 1 and seed 2 are separate processes (`PYTHONHASHSEED=1` and `=2`). Their JSON is
byte-identical (sha256 `0e17b69a…`), and so is the text (`5667f941…`). The JSON carries a sha256 for every file
each replay wrote. The only normalisation: the work directory's own path is replaced by `<work>` before hashing,
because `run_enrichment` records its `dump_path` (F-5).

**Gold at `main`** (eval-v2 `score_leg` through the ORCH-756 scorer's loaders, certified gold only):

| leg | final | PWML |
|---|---|---|
| 756 enterobactin | P 0.714 R 1.000 F1 0.833 | P 0.714 R 1.000 F1 0.833 |
| 756 menaquinone | P 0.889 R 0.778 F1 0.830 | P 0.889 R 0.778 F1 0.830 |
| 759 enterobactin | P 1.000 R 1.000 F1 1.000 | not written |
| 759 menaquinone | P 0.875 R 0.778 F1 0.824 | not written |
| 759p enterobactin | P 1.000 R 1.000 F1 1.000 | P 1.000 R 1.000 F1 1.000 |
| 759p menaquinone | P 1.000 R 1.000 F1 1.000 (9/9) | not written |
| 767 enterobactin | P 1.000 R 1.000 F1 1.000 | not written |
| 767 menaquinone | P 0.909 R 1.000 F1 0.952 | P 0.909 R 1.000 F1 0.952 |
| 9 × `runs_verify` enterobactin | listed in the `.txt` | |

These equal the recorded gold in `ORCH-767-RESULT.md` (767 menaquinone PWML 0.909/1.000/0.952; 756 enterobactin
0.714/1.000/0.833), because the replayed canonical payloads and PWMLs are the leg's, byte for byte.

## 6. Replayable or LLM-originating: each failure at `main`

"Replayable" means the failing step is inside the replayed tail. **A fix is measurable on the corpus only if it
lands in the tail**: the quarantine, the final gate, the driver verdict, refinement review, Stage-8 revalidation,
the required-field gate, the pre-freeze seam, the IR or the writer, and for pre_freeze legs also enrichment and
pre-freeze resolution. A fix in Stage 3 (the normalizer) or Stage 6 (mapping) is upstream of every checkpoint, so it
cannot move a corpus leg, even when the failure it addresses is replayable.

**The four ORCH-767 failures (D-112 § 3), all replayable:**

* **(a) enterobactin**: final gate. The error is `/entities/protein_complexes/0/components/0`: "Generated protein
  complex 'enterobactin synthase complex' component 'enterobactin synthase' is missing positive stoichiometry."
  The same shape appears on tier-B `2026-08-21_2057/PMC12452463`.
* **(b) chlorophyll**: final gate, 9 × "Unknown protein/modifier reference" (PIF1, GBP × 3, FAR1, CPO1 × 2,
  PPO1 × 2). The same class appears on tier-B `2026-08-22_2017/PMC12856317`.
* **(c) WNT PCP**: `initialize_refinement_review_state` raises "DraftGraph has no reaction nodes — nothing to
  render" (0 reactions, 18 interactions). The harness also records, **as information only**, what
  `run_pwml_export` returns when the review step is skipped: `ok`, no gate error. **The crash is the only
  blocker** on this leg.
* **(d) Wnt/Ca2+** (pre_freeze): the quarantine refuses with `unexportable_entity:1`, the protein "phosphorylated
  CAMKII" (`protein_missing_external_identity`). The release is `diagnostic_only`
  (`serialization_would_require_invention`), and the final gate report is missing.

**Other replayable failures (26):**

* 755 enterobactin: refusal.
* 755 sour: `actor_schema_not_canonical`.
* 759 enterobactin: degree 0.
* 759 and 759p menaquinone: species, in the required-field gate.
* 759 chlorophyll: `reaction_enzyme_must_be_protein_complex`, **cause upstream in Stage 6**.
* 20 tier-B legs:
  * `no_defensible_reaction_support` × 8;
  * `reaction_enzyme_must_be_protein_complex` × 3;
  * IR `protein_complex_missing_components` × 2;
  * `actor_schema_not_canonical` × 1;
  * final gate × 3;
  * quarantine refusal × 3.

**LLM-originating (10):** each ended in or before an LLM stage, so it has no checkpoint and needs a live run.

* timeouts during extraction: 755 menaquinone, chlorophyll, ara, wnt_pcp; 756 chlorophyll; 759 wnt_ca;
* Stage-0 `scope_conflict`: 756 ara, 759 ara_v2, 759 sour;
* Stage-1 invalid JSON: 755 wnt_ca.

## 7. How a fix lane uses the harness (D-112 § 2)

Use absolute paths throughout, because the wrapper's `--cwd` is scratch (F-7). Take `PY` to be the primary
checkout's `.venv/Scripts/python.exe` and `EV=<worktree>/docs/pwml_recovery_sprint/evidence`.

```bash
# base: a clean export of the lane's base (or reuse evidence/orch768_corpus_baseline.json if the base is e3d5b65f)
git archive <base> src reference/PW000001.pwml data/pathwhiz_id_db.json | tar -x -C C:/t/<lane>/base
P=$($PY $EV/g11/g11_evidence.py next --task ORCH-769 --label corpus-base)   # guard $P as usual
$PY $EV/bounded_run.py --label corpus-base --timeout 540 --cwd C:/t/<lane>/cwd --json "$P" -- \
  $PY -u $EV/orch768_replay_corpus.py run --src C:/t/<lane>/base/src --out C:/t/<lane>/out-base \
  --json C:/t/<lane>/base.json --txt C:/t/<lane>/base.txt
# tip: the same, with --src <lane worktree>/src
$PY $EV/orch768_replay_corpus.py compare C:/t/<lane>/base.json C:/t/<lane>/tip.json --target <leg id>
```

* **What `compare` checks.** It prints the pass rate before and after, the legs fixed, the **passing legs turned
  failing**, any **gold recall lost** (final or PWML level), and the still-failing legs whose failing step moved.
  It ends with `D-112 § 2: ACCEPT` (exit 0) or `REJECT` (exit 1). A whole-corpus run takes about 24 s.
* **Running on a worktree is safe.** A replay at a worktree's own `src` writes nothing into it: `git status` was
  clean after the acceptance test, which replays against this worktree's `src`.
* **Worked example 1** (`evidence/orch768/example-compare-f392d447-to-main.txt`). Between `f392d447` and `main`,
  the pass rate does not move (51/91). ORCH-759 chlorophyll moves from the quarantine refusal to the required-field
  gate: ORCH-759c, a quarantine fix, is measured. ORCH-759e's Stage-6 rewrap is not (F-6).
* **Worked example 2** (`evidence/orch768/example-compare-main-to-orch766-040d65b9.txt`). This measures the
  **pending, unreviewed** ORCH-766 tip `040d65b9`, which changes only `strict_quarantine.py`:
  * the pass rate goes from 51/91 to **53/91**;
  * ORCH-759 and ORCH-759p menaquinone are fixed, and 759p menaquinone's PWML gold is P 1 R 1 F1 1 (9/9);
  * no passing leg turns failing and no gold recall is lost: **ACCEPT** under D-112 § 2.

  This is a measurement for the orchestrator. It is not a review of that lane.

## 8. Findings

1. **F-1. `final_mapped.json` is the no-LLM checkpoint, in two kinds** (§ 1). Re-running the quarantine on a
   canonical payload is idempotent at the same code: 73/73 re-hash exactly.
2. **F-2. The replay is exact at the leg's own code.** 81/81 verdicts reproduce, 43/43 PWMLs are byte-identical,
   and 81/81 decision-input hashes match. That includes tier B, whose SHA comes from the reflog and is therefore
   corroborated by the result.
3. **F-3. The enrichment cache must be the end-of-leg one.** With the start snapshot, ORCH-767 Wnt/Ca2+ made 39
   refused lookups and gave a different enriched payload, while its verdict happened to survive (G11 `08`). With
   the next leg's start snapshot it made 0 (G11 `07`).
4. **F-4. Pre-freeze legs depend on the PathBank DB** for compound resolution. Without it, the quarantined payload
   differs in bytes but no verdict changed (8/8). On the one leg needing no DB lookup, the DB marker was the only
   difference.
5. **F-5. `run_enrichment` writes its own `dump_path` into `enrichment_report.json`.** That alone made two runs
   into different `--out` directories differ in 8 file hashes (G11 `38`/`39`); fixed by the `<work>`
   normalisation (G11 `40`/`41` onward).
6. **F-6. Stage-6 fixes are invisible to the corpus.** ORCH-759e (merged `360a6ac5`, Stage 6) changes no corpus
   verdict: between `21155be0` and `main` nothing is fixed, broken or moved
   (`evidence/orch768/example-compare-21155be0-to-main.txt`).
   ORCH-759 chlorophyll still fails `reaction_enzyme_must_be_protein_complex` at `main`, because its Stage-6 output
   is frozen from `f392d447`.
7. **F-7. A relative `--json` path fails under `--cwd`.** G11 `79` exited 1 after a complete replay because its
   `--json` was relative to the worktree while the wrapper ran in `C:/t/bt-orch768/cwd`. Rerun with absolute paths
   as `80`.
8. **F-8. 27 of 63 tier-B legs changed outcome since they ran.**
   * The C-119/C-126 supersession rule clears stale `audit_round` contract findings. Of the 15 legs filed
     `upstream_contract_report`, 11 now write a PWML and 4 fail later, at export.
   * 5 legs that wrote a PWML in August now fail the required-field gate on `no_defensible_reaction_support`.
   * Release changes: 3 `release_ready` → `review_required` and 2 `diagnostic_only` → `review_required`.

   These are code changes, not replay errors: at their own trees the same legs reproduce 63/63.

## 9. Failures and dead ends

* **Considered and rejected: `merged_payload.json` as the checkpoint.** It precedes the LLM audit, gap resolver and
  curator, and 15/18 tier-A legs were changed by them (§ 1).
* **Tested and disproved: "the DB marker is the only pre-freeze difference."** True on 1 leg, false on 7, where
  compound resolution itself needed the DB (F-4).
* **Wrong at first: comparing gate codes from a stdlib slug.** The first fidelity draft built recorded gate codes
  with a stdlib slug while the replay used the driver's `_issue_code`. That reported 2 false differences. Replaced
  by comparing the full `[path, reason]` lists read by the measured tree.
* **Wrong at first: the contract-blocked stage label.** Legs whose final gate passed but an upstream contract
  report blocked were labelled by the gate stage. They now get `upstream_contract_report`, on both the recorded
  and the replayed side.
* **Incident.** One command contained an empty `python -` heredoc (against this lane's rules). It hung, was moved
  to the background by the shell, and was stopped with TaskStop. It imported nothing and wrote nothing. Bash
  heredocs here also collapse `\\`, so the edit scripts were written with the Write tool instead.
* **G11 history.** All 80 reports carry 0 survivors (`g11_evidence.py check --task ORCH-768`: 80 artifacts, 0
  non-compliant). The committed outputs come from the final passes; the earlier reports stay on the record:

  | G11 | run |
  |---|---|
  | `01`–`21` | development |
  | `22`–`37`, `42`–`57` | earlier fidelity passes, superseded by the `<work>` normalisation and a label rename |
  | `38`–`41` | earlier baseline passes, superseded the same way |
  | `58`–`59` | baseline (committed) |
  | `60`–`75` | fidelity (committed) |
  | `76`, `77`, `80` | worked examples |
  | `78` | the acceptance test, 9 passed, pin `78-acceptance-test.pin.json` |
  | `79` | F-7 |

## 10. Gates

* **The new acceptance test** `tests/test_orch768_replay_corpus.py` (labelled NEW CAPABILITY, G9, no base failure
  claimed): **9 passed** (G11 `78`, pinned to this worktree's `src`). It checks:
  * the fixture is the corpus input, byte for byte;
  * the measured tree is this checkout;
  * the probe refuses and counts 3 attempts, and the legs make 0;
  * each of the 3 legs reproduces its recorded verdict;
  * the enterobactin errors match the recorded ones exactly;
  * the fidelity check bites on 3 channels;
  * two replays are identical.
* **SMOKE**: not required, because there is no `src/` change.
* **`g11_evidence.py check --task ORCH-768`**: exit 0.

## 11. Not settled (for the product owner)

1. **Fixes upstream of the checkpoint cannot be measured** (Stage 3, Stage 6; F-6). Options:
   * (a) Record UniProt, UniParc, NCBI and PathBank answers per leg in future runs (a recording cache in the leg
     directory). This is a code change in mapping and the runner; once it exists, Stage 6 becomes replayable.
   * (b) Measure upstream fixes only by live runs (ORCH-770).
   * (c) Per-defect reconstruction of the Stage-6 input, as `orch759e/stage6_replay.py` does. It is approximate and
     one defect at a time.
2. **Pre-freeze legs replay without the PathBank DB** (F-4). The options are a recorded DB-answer fixture, which
   needs one read-only live query session and so a PO decision under this lane's no-network rule, or accepting
   verdict-level fidelity for those 8 legs.
3. **Tier-B SHAs come from the reflog**, not from a record. They are corroborated by 63/63 fidelity. Should tier B
   count in the D-112 § 2 rule? The harness counts all 91, and `compare` treats every leg alike.
4. **Which number is "the corpus pass rate"?** 51/91 (strict legs with an outcome) or 51/81 (replayable). Both are
   reported. The LLM-originating 10 can only move in a live run.
5. **The corpus inputs are untracked** in the primary and in two agent worktrees (`orch756`, `orch767`). Deleting
   those worktrees loses 7 tier-A legs except for the 3 fixture legs. Options: copy the 587 files (27.9 MB plus
   319 MB of caches) to a durable location, or commit the non-cache inputs gzipped (several MB; `.git` is already
   158 MB).
6. **The acceptance test pins verdicts that ORCH-769 is meant to change** (enterobactin, WNT PCP). Those lanes must
   move `EXPECTED` with the delta (merge rule 4). The alternative is to pin against an archived `d6849c9e` tree
   inside the test, which is heavier.
7. **Research-mode legs (7) are inventoried but not replayed.** Should research mode get its own corpus?

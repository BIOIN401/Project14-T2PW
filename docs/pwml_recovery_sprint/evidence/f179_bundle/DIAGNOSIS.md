# F-179 … F-182 — diagnosis of `runs_verify/2026-09-02_2052`

**Evaluation-only. Nothing was re-run. Production was not patched.** The source run is
**untracked and byte-untouched**; this bundle exists so the diagnosis is auditable
without the run becoming tracked.

## How to reproduce

```bash
# 1. re-hash every cited artifact against the live run (exit 1 on any drift)
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/f179_bundle_build.py . --verify

# 2. rebuild the extracts from scratch
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/f179_bundle_build.py .

# 3. re-run the recurrence census over committed + preserved runs
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/rd093_shortcut_census.py . \
    --json <out>.json

# 4. re-classify any archived reaction row
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/rd092_1_reaction_lineage.py . --json <out>.json
```

`INVENTORY.json` carries a SHA-256, byte count and exact source path for all 32 cited
artifacts. `EXTRACTS.json` carries the minimum record behind each finding. Neither
contains a full payload, source text or cache.

## The run's own scorecard measures the wrong thing

`SUMMARY.txt` reports **"strict PWML: 3 pass / 4 fail"**. Scored against what gold
actually expects, the run is **9 of 10 correct**:

| Paper | Gold `expected_export` | Outcome | Verdict |
|---|---|---|---|
| PMC12096016 | `strict_exportable` | exported | correct |
| PMC12782028 | `strict_exportable` | exported | correct |
| PMC12312563 / PMC12421875 / PMC12657337 | organism trap | Stage-0 abort | correct |
| PMC12444477 / PMC12452463 / PMC12856317 / PMC13231680 | `partial_only` | refused | correct |
| **PMC12180156** | **`partial_only`, "nothing exportable"** | **exported** | **F-179** |

Only **2 of 10** gold cases are `strict_exportable`; the other 8 are `partial_only`, so
refusal is the correct outcome for most of the corpus. The three scope conflicts are
**deliberate organism traps** — `topics_t104.txt` says so in capitals and each case lists
*Bacillus subtilis* in `forbidden_organisms`. Correcting them would be input-side gate
weakening.

---

## F-179 — production defect / false-positive biological export

**`runs_verify/2026-09-02_2052/papers/PMC12180156/strict`**

The delivered canonical payload contains **one reaction: `glycine → heme`**. Heme
biosynthesis is an eight-step pathway; this collapses all of it into a single step.

- Gold `relevance_note`: *"Zero heme-biosynthesis reactions have both sides named
  anywhere in the file."*
- Gold `export_rationale`: *"With zero heme-biosynthesis reactions recoverable, nothing
  about heme biosynthesis is exportable."*
- Gold `supported_reactions`: **0**
- Runtime recorded `release_status: review_required`, **`semantic_evaluation: passed`**,
  `strict_gates_passed: true`, and wrote `pathway.review_required.pwml`.
- The row carries **no `provenance_lineage` of its own** and **no `rag_provenance`**. Its
  only recorded stage, `identifier_mapping`, is **inherited from its participants** —
  glycine and heme are grounded in ChEBI/KEGG/DrugBank, which is identity evidence, not
  evidence that any paper states the reaction.
- `rd092_1_reaction_lineage.py` classifies it **`indeterminate`** — *"no reaction-specific
  evidence recoverable"*. The evaluator does not vouch for it.

**No stage ever claimed a paper stated this reaction, and the semantic gate passed it.**

## F-180 — production tokenizer defect · secondary / deferred

**`…/PMC12452463/strict`** — `ferric iron (Fe3+)` is parsed as a composite `+` separator:

```
Composite entity 'ferric iron (Fe3+)' in /entities/compounds has no protein-like
left component; unsupported.
/entities/compounds/5/name has '+' token: ferric iron (Fe3+)
```

Ionic charge notation misread as a composite token — on the core metabolite of an
iron-acquisition paper. The leg's **refusal was correct** (gold: the route is chemically
broken with EntA absent) but the **reason is wrong**, so the correct outcome here is
reached by coincidence.

## F-181 — production referential-integrity defect · secondary / deferred

**`…/PMC12856317/strict`** — interactions reference entities that were never registered:

```
/processes/interactions/1/entity_2 unknown entity: HRM3
/processes/interactions/2/entity_2 unknown entity: HRM6
```

`HRM` = heme regulatory motif. Its **final Stage-3 gate actually passed** (`ok: true`);
this blocked the leg separately.

## F-182 — lifecycle / observability defect · NOT automatically a biological failure

**`…/PMC12444477/strict`** — `final_stage3_gate_report.json` is **absent**, so
post-pipeline validation fails with `final_gate_report_missing` at
`final_pre_export_gate_lifecycle`. The leg ran ~40 minutes through gap resolution and
ended `fail` at `post_pipeline`.

**This says nothing about the biology.** Gold independently expects no strict export
here (*"a complete ID-resolvable pathway cannot be built from this paper: it contains
zero UniProt, EC, b-number, ChEBI or KEGG identifiers"*), so the outcome is right — but
the recorded reason is a missing report, not a biological verdict.

> **A correction recorded deliberately.** An earlier reading of this leg claimed a
> "UniProt promotion defect" — that the resolver found accessions and failed to write
> them onto the row. **That was wrong.** `LpxA` carries `mapped_ids.uniprot = P0A722` and
> `protein_external_identity` returns it. The missing-identifier errors come from the
> **initial** post-normalization gate, which runs *before* mapping. Corpus-wide only
> **2 of 2,189** protein rows carry a verified-but-invisible accession.

---

## Recurrence census — the mechanism is NOT isolated

`rd093_shortcut_census.py`, read-only over committed **and** preserved runs, populations
never summed:

| criterion | committed (115 legs / 433 rows) | preserved-untracked (10 legs / 33 rows) | distinct papers |
|---|---|---|---|
| `exact_glycine_heme` | 6 | 1 | **1** |
| `only_identifier_mapping_own_lineage` | 0 | 0 | 0 |
| `only_identifier_mapping_inherited` | 0 | 0 | 0 |
| `no_paper_and_no_rag` | 198 | 1 | **9** |
| `precursor_terminal_shortcut` | 62 | 4 | **6** |

**Intersection — produces the pathway's terminal product AND carries no paper-stated and
no RAG-literature attribution: 28 rows, 6 distinct papers, 13 distinct runs.** Of those,
**8 were in legs that exported a PWML.**

Restricted to the **current release-naming regime** (legs carrying a release
classification), the exported instances are:

| run | paper | reaction | pwml | semantic |
|---|---|---|---|---|
| `2026-08-21_2239` | PMC12782028 | `4,4-dimethylcholesta-…-3β-ol → cholesterol` | review_required | **failed** |
| `2026-08-22_2147` | PMC12180156 | `iron → heme` | review_required | **passed** |
| `2026-08-22_2147` | PMC12180156 | `Glycine → heme` | review_required | **passed** |
| `2026-08-24_1428` | PMC12180156 | `Glycine → heme` | review_required | **passed** |
| `2026-09-02_2052` | PMC12180156 | `glycine → heme` | review_required | **passed** |

Three older legs additionally shipped a **bare `pathway.pwml`** — the name
`PRODUCT_CONTRACT` § 13 reserves for "ship it, no review needed":
`runs/2026-07-28_0919` PMC12444477 `lipid IV A → lipid A precursor`, and
`runs_verify/2026-08-04_1754` PMC12180156 `glycine → heme` and `iron → heme`. Those
predate the `review_required` naming, so they are historical rather than current
behaviour — but they are what the mechanism produced when the reserved name was
reachable.

### Classification

**The `glycine → heme` shortcut is confined to PMC12180156 — but it is not a draw
artifact: it reproduces across four separate runs over one month, in both modes, under
five different reaction names, and the semantic gate passed it every time.**

**The underlying mechanism recurs independently on a second paper and a second pathway**
(PMC12782028, cholesterol biosynthesis, 2026-08-21). That instance was caught
(`semantic: failed`), which is evidence the gate *can* catch it and did not for
PMC12180156.

> **Therefore this is presented as a CANDIDATE REPEATED PRODUCT-CONTRACT VIOLATION
> requiring a narrow product-owner unfreeze, not as an isolated known limitation.**
> Per the standing rule, a benchmark failure does not by itself justify a code change;
> this is classified `product_contract_violation` because gold's `export_rationale`
> states nothing is exportable and the pipeline exported a reaction no source states.

**The unseen cohort must not be consumed until this is ruled on.** A false-positive
biological export that passes the semantic gate would be scored as a success on unseen
papers.

## Caveats

- `runs_verify/2026-09-02_2052` is **untracked**; its four legs rest on a single-disk
  artifact. Every other run cited here is committed.
- `precursor_terminal_shortcut` is a **heuristic** over gold: gold lists only 0–3
  signatures per paper, so a legitimate terminal step absent from that short list is
  flagged. It is reported alone and in intersection with `no_paper_and_no_rag`
  precisely because the intersection is the defensible signal.
- The 62/198 raw counts include `research`-mode legs, which never export PWML.

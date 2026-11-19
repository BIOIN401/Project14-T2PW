# PWML import fixes (branch `pwml/import-fixes`, from main `a751f8f2`)

Goal: the PWML defects that make PathWhiz `t2pw/latest` (`63959982`) refuse parts of our pathways or draw them badly.
Not merged. Code commit `b0d47b68`. Evidence: `evidence/import_fixes/` (A/B JSONs, scans, every script used).
Working files (replay outputs, harness runs, logs): `C:/t/pwml-fix-work/`.

## 0. Not established (read first)

1. The PathWhiz harness has an empty database plus the vocabulary seed (as in P4/P5 and the showcase). Ontology
   messages and the same-state collapses they cause are harness artefacts or need a vocabulary decision; they are out of
   scope and counted separately.
2. The replay feeds **stored** payloads (some normalized before 2026-11-12). A fresh run on main already applies the
   D-W20b transport-side rule; the replay does not re-run the normalizer.
3. No live LLM run was made. Run B2 was replayed from its stored `final_mapped.json` (strict, prefilled), not re-run.
4. Duplicate compartment boxes were measured with the V7b metric and with a stricter per-box metric (section 1.2);
   no screenshot comparison was made for them.

## 1. Defects, root causes, fixes

### 1.1 Whole export refused by one transport (run B2) - FIXED

**Root cause.** `ir.build_pwml_ir` resolves a transport's `from/to_biological_state` by **state name** only
(`resolve_state`). B2's Stage-1 row wrote the **location** name `extracellular` for the state `extracellular state`.
The name matched no state; transport sides are resolved with `required=True`, so the miss is an **error** whatever
`strict_db` says, and `postpipeline.run_pwml_export` refuses the whole PWML on any blocking IR error. A second latent
defect: B2's payload carries `cytosolic state` twice (same name, same species and location); `state_by_name` kept the
**last** row, while the default state is the **first**, so references split over two identical states (two
"cytosolic" boxes in PathWhiz once the export works).

**Why `pwml_ir_validation.json` said ok and `pwml_ir_report.json` said not ok.** They are two different checks.
`pwml_ir_validation.json` is `validate_pwml_ir(ir)`: a structural check that every key the IR uses exists. It passes,
because `resolve_state` hands back the default state's key on a miss, so the IR is internally consistent.
`pwml_ir_report.json` is the builder's resolution report; the unresolved name is recorded there as an error. The export
fails if either has a blocking error (`blocking_pwml_ir_errors(ir_report) or ir_validation["errors"]`), and the failure
record counts `issues` from the validation only (0), which is why `RESULT.txt` printed "issue codes: (none reported)".
Not changed (artifact shapes are pinned by tests); recorded here.

**Fix (`src/t2pw/pwml/ir.py`).**
- `resolve_state` fallback: a name that is not a state name resolves to the one state whose payload location has
  that name, or to the state named `<name> state`. Two candidates are ambiguous and stay unresolved. Warning
  `biological_state_resolved_by_location`.
- A transport side that still names no state withholds **only that transport**: it stays in the payload (and so the
  text record and our SBML), is listed in the IR report `withheld_transports` with its reason, and gets a warning
  `transport_withheld_from_pwml`; it is not drawn in the default state (that would misplace it). A transporter whose
  state names nothing loses only the transporter (the D-114 Am. 1 s2 rule the quarantine already uses).
- An exact repeated state row (same name and same resolved components) no longer takes the name over; references
  stay on the first row (the default state). Its location is not counted twice for the fallback.

**Before / after.** B2 replay (`b2_base_replay.json`, `b2_after_replay.json`): base `ok=false`, "PWML IR validation
failed."; after `ok=true`, 11 reactions, both DHNA transports `cytoplasm -> Extracellular Space`, one cytosolic box
(labels: cytosolic, extracellular). 97-run replay: 0 `unresolved_biological_state` issues in either arm, so no stored
run changes through this path.

### 1.2 Duplicate compartment boxes - NOT A DEFECT ON MAIN (metric artefact); real remainders deferred

**How our PWML encodes compartments.** One `<biological-state>` per IR state (species, subcellular location, cell type,
tissue). Layout v2 draws one `membrane-visualization` box per state that holds a drawn node, labelled with the state
name (`layout.label_text`, without " state"; long labels are cut into stacked one-line `label-location`s). PathWhiz does
not group or draw boxes itself: it draws our membranes and labels verbatim. On import it binds a state by its exact
(species, tissue, cell type, location) tuple (`t2pw/04`), so two of our states with one tuple become one DB state while
we still draw two boxes.

**What "9 of 97" measured.** The V7b metric (`layout_metrics.duplicate_compartment_boxes`) counts repeated
**label-location texts**. On V7b's base (pre-L6 merge) several states really repeated; on main every hit is a label
**fragment** of a stacked label: idx 68/76 "IM" (from "cytoplasmic leaflet of / IM" and "periplasmic leaflet of / IM"),
idx 3/151 "Type III cell" (from "Type III cell / cytoplasm" and the box "Type III cell"), idx 76 "membrane". Joining
each box's parts (`scripts/boxes.py`): **0 repeated box labels in 97 runs**, base and after (`boxes97_base.json`).
The V7b metric is vision code; fixing it is left to that lane (open question).

**Real same-location pairs on main** (11 runs, `states_sharing_location` in `boxes97_base.json`), all deliberate:
- generic vs specific state (idx 119, 126, 127: "cytosolic state" beside "astrocyte cytosol"): different PathWhiz
  states; merging would assert the generic one is that cell type (L7 territory);
- eukaryote cytoplasm vs cytosol (idx 3): kept apart by CORRECTNESS-FIXES (L6 deferral);
- placeholder `__auto_state__` / `AutoState_*` beside a stated state (idx 10, 48, 139, 146, 150, 151);
- the bacterium case ("intracellular" and "cytoplasmic" boxes) is the **menaquinone showcase** (live run): L6 deferred
  the merge with `transport_ends_would_merge` because the row "DHNA export" goes `cytoplasmic state ->` an intracellular
  state its paper never named (`elements_with_states` right side empty). Merging would make that transport go from a
  state to itself. The fix is the transport's missing side (D-W21 c), not the box.
- Exact repeated rows (B2) are fixed by 1.1.

### 1.3 Importer messages on our PWML

| Message (files) | Root cause | Status |
|---|---|---|
| nucleic acid "Nucleotide type can't be blank / not included in the list" + cascades "Nucleic acid can't be blank", interaction "wrong length", "Element can't be blank", interaction visualization "can't be blank" (chlorophyll) | rows with no DNA/RNA type. 4 of the 7 are named "X promoter"; the writer's name rule (`nucleotide_type_for`) knew `gene`/`dna` but not `promoter`, and the paper reader (D-W20c) left them unresolved (cue inside the name; sentence conflict) | **Partial**: promoter, enhancer, locus, plasmid, allele in the name now mean DNA (the same nouns the paper reader takes as DNA cues). Chlorophyll 7 -> 3 untyped (`ALAD1/HEMB1`, `CHL27`, `LHCB1.1`: bare gene symbols, gene or transcript not stated). The 3 are deferred (D-W21 d: unknown stays unknown) |
| "Transport can't contain duplicate elements" + 3 cascades (coenzyme A, idx 128) | the payload lists TcyJKLMN twice on one transport (one stated, one inferred row); the IR wrote two identical `transport-transporter`s; PathWhiz `Transport#no_duplicates` refuses the whole transport | **Fixed**: same transporter in the same state written once (warning `duplicate_transporter_written_once`) |
| "Biological state must have unique biological states" -> "Transport elements is too short" -> transport / transporter visualization "can't be blank" (sour taste idx 10; lipid A x3) | sour taste: the stored payload's row "OTOP1-mediated H+ permeation" is `__auto_state__ -> __auto_state__` (normalized 2026-10-22, before D-W20b; the row declares its right side in `elements_with_states`). Same-key transports in the replay: idx 10, 78, 102, 138, all from/to equal in the payload. Lipid A: the two leaflet states are distinct in our PWML but their locations ("cytoplasmic leaflet of inner membrane", ...) fail to create in the empty DB (ontology), so both collapse to one DB state | **Deferred**. Same-state rows are an extraction content gap; PO D-W21 c: fix at the extraction step, and "a fix that loses anything is not adopted" (withholding them would drop the transport from the PWML). Lipid A is the ontology cascade (out of scope) |
| transport visualization "Transport can't be blank" | cascade of the two rows above | follows them |
| Ontology can't be blank / is invalid | empty harness DB / vocabulary | out of scope |

### 1.4 Unlabeled compartment box (optional) - DEFERRED

13 runs on main (one each), all a placeholder state (`__auto_state__`, location `cell`): `layout.label_text` returns
nothing for generated names on purpose. Labelling it is a wording choice (the location name "cell", or "compartment not
stated") for the PO; it moves box geometry in 13 runs. Not done.

## 2. Gates

All runs through `bounded_run.py` with the heavy lock (holder `pwml-fix`, `C:/t/vis-runs/queue.sh`, MINGB 4).

**Replay no-regression (97 stored runs that export, of 152; each in its own export mode, flag on, layout v2, no
network).** Base = `git archive a751f8f2` (src, data, reference); after = `b0d47b68`. Script `scripts/replay97.py`
(adapted from `C:/t/corr-assess/scripts/dump_replay.py`: same `run_pwml_export` call, mode and strict_db per run;
no network, no dump into the shared base folder). Base reproduces the showcase PWMLs byte for byte (8/8) and is
deterministic (two base runs identical). Comparison `scripts/ab_replay.py` -> `replay97_ab.json`:

| | base | after |
|---|---|---|
| exports | 97 | 97 |
| byte-identical | | 95 |
| reactions / compounds / proteins / complexes | 381 / 691 / 376 / 397 | 381 / 691 / 376 / 397 |
| transports / transport elements' (cargo, left loc, right loc) | 74 | 74, 0 lost |
| interactions / states / membranes / labels | 49 / 234 / 167 / 192 | same |
| transport-transporter rows | 34 | 33 (idx 128: the exact duplicate TcyJKLMN) |
| nucleic acids with a type | 6 | 10 (idx 7: CHLH, HEMA1, ALAD1/HEMB1, PORB promoter -> DNA) |
| names lost / reaction signatures lost / transport signatures lost | | **0 / 0 / 0** |

Showcase (8 PWMLs replayed from the stored runs with `show_replay.py`): 6 byte-identical, chlorophyll and coenzyme A
changed exactly as idx 7 and 128; 0 lost (`showcase_ab.json`). B2: base refused, after exports (1.1).

**PathWhiz A/B** (`pathwhiz_ab.json`, raw rows `pathwhiz_*_results.jsonl`). Harness copy of
`C:/t/pwi/t2pw/tools/pathwhiz_import` in `C:/t/pwml-fix-work/harness`: compose project `t2pw-fix`, ports 3337/3034,
clone of `t2pw/latest` `63959982`, MolDB pointed at a closed local port (`C:/t/pwpr/show/config/moldb.yml`), vocabulary
seed, clean database per file. Files: the 8 showcase PWMLs + V7b idx 7, 10, 14, 116, 126, 128, each arm. Stack taken
down after (`docker compose down`; volumes `t2pw-fix_bundle`, `t2pw-fix_mysql_data` kept).

| 14 files | base | after |
|---|---|---|
| imports successful | 14/14 | 14/14 |
| compounds / proteins / complexes / reactions | 192 / 156 / 156 / 154 | same |
| transports | 26 | **28** (coenzyme A 2 -> 3, twice) |
| transport transporters | 14 | 16 |
| nucleic acids | 12 | **20** (chlorophyll 5 -> 9, twice) |
| interactions / compound locations / states / labels | 39 / 288 / 58 / 62 | same |
| importer messages | **113** | **89** |

Messages that went down: nucleotide type blank 14 -> 6 and not in list 14 -> 6; "Transport can't contain duplicate
elements" 2 -> 0; transport transporter visualization 10 -> 6; transport visualization 7 -> 5. No created count went
down in any file. Extra (after only): B2's PWML imports with 0 messages, 11 reactions, 1 of its 2 identical transports.

**Suites on `b0d47b68`** (`C:/t/vis-runs/suite2.sh` in one hold; log `evidence/import_fixes/suite.log`): SMOKE 511
passed; golden seam 2 / 8 / 4 / 24 passed (unedited); equivalence run_pathway 8, app_vs_service 14, default_path 6;
vision 357 passed + 1 skipped; web 83 passed; BW_EXIT 0, 0 surviving processes.
**Focused** (`focused1.log`): `test_pwml_import_fixes.py` (9 new) with the nucleic, writer, IR, transport, autostate,
quarantine-omission, final-payload and layout files: 344 passed, 0 survivors.

## 3. Deferred, and why

- Same-state transports (4 replay runs, sour taste showcase): D-W21 c; withholding = loss; re-applying the D-W20b rule
  at export would change 1 stored row (idx 10) and still leave a cell -> cell transport. Fresh runs already apply D-W20b.
- 3 chlorophyll gene-symbol nucleic acids and RyhB (idx 84, 105): type not stated; D-W21 d (flag unknown).
- Box label for placeholder states (1.4); V7b duplicate-box metric (1.2).
- Identical transports after resolution: B2 now has two `cytoplasm -> extracellular` DHNA transports; PathWhiz imports
  one ("unique set of elements"). Same as CORRECTNESS-FIXES open item "write identical transports once" (L6).

## 4. Open questions for the PO / orchestrator

1. **Test edits.** `tests/test_final_payload_nucleic_type.py` (one test) and `tests/test_final_payload_nucleic_lookup.py`
   (one row) used "CHLH promoter" / "PORA promoter" as examples of a name the name rule does **not** settle. With the
   new rule they are settled (DNA), so the examples were swapped for "CHLH" / "PORA region"; each test keeps its intent.
   Not golden or equivalence tests. Needs the reviewer's ruling.
2. **Release cap for a withheld transport.** A transport withheld by the IR is a warning (as ORCH-747 withheld
   interactions are); release status does not read the IR report, so a strict run could stay `release_ready` with a
   transport missing from the PWML. Options: (a) leave as is (parity with interactions); (b) map
   `/processes/transports/N/*_biological_state` IR pointers to the row in `strict_quarantine._row_failures`, so the
   quarantine omits the row and caps at `review_required` (touches `src/t2pw/pipeline/`). B2 itself no longer needs it
   (the location fallback resolves it).
3. **Placeholder box label** (1.4): "cell", "compartment not stated", or none.
4. **Gene-symbol nucleic acids** (3 in chlorophyll): default to DNA ("gene"), keep unknown, or withhold.
5. **Same-state transports**: switch on the D-W21 c extraction repair (`T2PW_TRANSPORT_SIDES`) after its test, or accept.

## 5. Dead ends

- Location-based deduplication of compartments by word set ("inner membrane cytoplasmic leaflet" = "cytoplasmic
  leaflet of inner membrane") was considered for lipid A and dropped: "outer leaflet of inner membrane" and "inner
  leaflet of outer membrane" have the same word set and are different places.
- Re-applying the normalizer's declared-side rule (D-W20b) inside the IR for stale payloads: 1 row in 97 (idx 10) and
  the result is still a cell -> cell transport; not worth a second copy of the rule.
- The first brief number (duplicate boxes in 9 of 97) does not reproduce on main: the V7b run's IR predates the L6
  merge (V7b's own irdiff: idx 7 had 10 states there, 4 on main).

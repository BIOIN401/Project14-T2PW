# 07. Do generated PWML files import into PathWhiz?

Worktree C:/t/oa (85b5f9d4). Scripts and raw output: C:/t/ev-oa/import/ and docs/pwml_output/evidence/07/ (validators.json 113 KB, run_validators.py, relaxed.py). Labels: M measured here, L doc/log evidence, G guess.

## 1. NOT ESTABLISHED

1. **Whether any current generated PWML imports into PathWhiz and renders.** No import result is recorded anywhere in the repo for the current pipeline. The manual import that D-099 section 19 requires is still pending: "RELIABILITY PHASE COMPLETE cannot be declared until that import happens. No automated task in this sprint can perform it." (docs/pwml_recovery_sprint/HANDOFF.md:919-920). L
2. **No offline check in the repo is a PathWhiz importer.** Nothing was run against PathWhiz or its Ruby parser (tools/ruby/pwml_parser.rb is a source copy that needs the Rails models; no Ruby run was attempted). M
3. **Whether the hash-pinned IMPORT-SET files (docs/pwml_recovery_sprint/pathwhiz_review/IMPORT-SET/, 17 to 27 files over time) were ever uploaded.** MANIFEST and HANDOFF call them "the files the product owner imports by hand"; grep of LEDGER, HANDOFF, DECISIONS and FINDINGS found no outcome. L, absence of evidence only.
4. **Whether PathWhiz's strict Nokogiri parse (`config.strict.noblanks`, pwml_parser.rb:41-43) accepts our files.** Our files parse with strict lxml (M, section 4), which is not the same parser.
5. **Rendering quality** (layout, labels) cannot be judged offline.

## 2. Evidence that exists (all L)

- The only real upload evidence is July 2026, a Herbaspirillum pathway on a LOCAL smpdb instance. It "failed to upload ... with a 500": emitted names differed from canonical DB names, and the importer's own cleanup crashed (docs/pwml_import_fixes_2026-07-16.md:3-6; section 0 at :17-60). The fixes (canonical names via name_index, label-type) predate the sprint. `reference/pathway (39).pwml` ("Generated Pathway", PW000000) is a T2PW-written file and passes today's QA (M). Whether it finally imported is not stated.
- F-204 (FINDINGS.md ~10427-10446): "PathWhiz never refuses a pathway over a bad protein: it skips the element, cascades the loss, and reports success ... a 'successful' import proves nothing about completeness ... Not established: no PathWhiz import was run; this is a static reading of the Ruby". So even a green import is weak evidence.
- docs/pathwhiz_requirements.md (2026-07-10) is the ground truth for required fields, taken from the Ruby source. Ids are document-local (parser skips `id`); compounds match by `find_by` on ALL non-blank scalar fields, so one mismatched field silently creates a duplicate; `pwc-id`, `short-name`, `pwbs-id` must be absent for novel rows (writer.py:774-796).
- HANDOFF.md:911-917: the two F-195 files go last; "if PathWhiz accepts and renders them, F-195 ... closes with no code". F-195 is the dangling-reference checker class.
- ORCH-747 (Oct 2026) added interaction import rules to qa.py: PathWhiz stores only Activation/Inhibition, exactly one left and one right element, a visualization with Left and Right members. Basis is a static reading, not an upload.

## 3. Validators: what they check

| validator | file | checks | does NOT check |
|---|---|---|---|
| structure validator `validate_generated_tree` | src/t2pw/pwml/validate.py:351-440 | root tag; order of root/pv/pathway children; each section exists; each item has every field seen in the first 3 items of the reference section (default ref reference/PW000001.pwml, writer.py:3006); integer type attribute | values, id uniqueness, cross-references, any attribute but integer, anything absent from the 3 sampled items |
| `repair_tree` | validate.py:269-348 | not a check: ADDS missing fields and reorders | the writer repairs, then validates (writer.py:2949-2950), so the stored `pwml_validation_report.json` cannot fail. 61/61 stored reports say ok (M) |
| QA `run_pwml_qa` | src/t2pw/pwml/qa.py:9-160 | strict XML parse; spontaneous plus enzyme; duplicate complex in a reaction; identical transport states; interaction shape (ORCH-747); stoichiometry positive int; HMDB/ChEBI format; orphan compound-location | dangling protein/complex/state/reaction ids (only compound-location is checked); protein UniProt/DrugBank rule; species/compound match against PathWhiz |
| IR validators `validate_pwml_ir` (ir.py:3326), `validate_required_pwml_contract` (ir.py:2595) | pwml/ir.py | IR consistency, required fields | the written XML |
| quarantine gates | pipeline/strict_quarantine.py | biology and referential gates before the freeze | importer acceptance |
| tests | tests/test_pwml_writer.py, test_pwml_ir*.py, test_pwml_interaction_export.py | writer and IR behaviour | no test imports into PathWhiz; no XSD or DTD exists in the repo (M, file search) |

## 4. Share of existing outputs passing (M; read-only; one light Python job, offline, no LLM)

Corpus: 61 canonical runs with a .pwml (quarantine_ok=true; 62 have pwml_ir.json, one has an IR but no .pwml); 6 non-canonical runs with a .pwml; 7 reference or control PWMLs in C:/t/oa/reference/ (the brief said 5; there are 7: PW000001, PW012926, PW1, PW176888, PW687775, PW741182, pathway (39)).

| set | n | structure validator, default signature | structure validator, signature as the writer relaxes it | QA with today's qa.py |
|---|---:|---|---|---|
| canonical runs | 61 | 0 pass (all lack pwc-id, short-name, pwbs-id) | 61 pass | 36 pass |
| non-canonical runs | 6 | 0 | 6 pass | 0 pass |
| references | 7 | 0 (miss sections that PW000001's first 3 items had; 2 differ in root order) | not run | 1 pass |

Reading the table:
- The default-signature failures are an artefact, not a defect: the writer deliberately omits those three fields so Rails allocates ids (writer.py:774-796) and relaxes its signature in memory, so the stored report is green. Even the references fail the default signature (0/7). The validator is a self-consistency check, not an acceptance test. M
- **QA failures are a single class: interactions.** 25 of 61 canonical runs (70 of the 87 interaction rows in the 61 runs) carry `<interaction>` with `interaction-type` like `substrate_of` or `cofactor_of`, a `<name>` child, no left/right elements and no interaction-visualization. Example: runs_validation/2026-09-10_2110/papers/PMC9544450/strict/pathway.review_required.pwml lines 492-501. All failing runs are dated Aug-Sep 2026 by folder name (Aug 15 of 22 fail, Sep 10 of 33 fail; the 4 Oct runs and 2 undated runs all pass). The stored `pwml_qa.json` in these runs says ok (61/61) because the older qa.py wrote it. That the cause is the pre-ORCH-747 writer is G; n=4 Oct runs is too few for a rate.
- Reference QA failures: 5 of 7 references fail strict `etree.fromstring` with "Namespace prefix option on end_arrow is not defined" (PathWhiz exports use `option:` tags without declaring the namespace); `PW1.pwml` fails QA because it holds bare `<interaction><id/></interaction>` stubs that QA counts as full interactions (a QA false positive on a real PathBank file). Only `pathway (39).pwml` (T2PW-written, declares `xmlns:option`) passes. QA's strict parse would reject PathWhiz's own exports. M

### 4.1 Structural comparison with the references (M; 67 generated vs 7 references)

- Root children order is identical in all 67 generated files and matches the references.
- Constants in all 67: `pw-id` PW000000, `cached-name` "Generated Pathway", pathway `id` 1. References carry real PW ids and names. `species-id` is 1 (32 files), 3 (19), 12 (4) among the 55 sampled by first protein; not checked against the paper organism. Placeholder constants may matter for upload matching (G).
- Element paths: references use 272 distinct paths, generated files 335. 68 generated paths occur in no reference: `ec-numbers/ec-number` (45 files), `protein_complex-proteins/protein-complex-protein` (65; the references spell the item `protein_complex-protein`, but Ruby `underscore` maps both to the same class, so likely harmless, G), `reaction-enzyme/inhibitor` (3), and the interaction tree (interaction-type in 36, left/right elements in 5). 5 reference paths never appear in generated files: `biological-state/pwbs-id` (deliberate) and the `protein_complex-protein` spelling.
- Attributes: generated files have `compound/description nil="true"` in 67/67 and `type="integer"` on complex-protein id, protein-id and stoichiometry (65). Not known to be rejected.
- Ids are document-local with fixed bases in generated files (20000 compounds, 30000 proteins, 40000 complexes, 50000 reactions; protein id 9659 is the PathBank Unknown sentinel). Ruby ignores `id` for matching, so ranges are not a rejection reason (L, pathwhiz_requirements.md section 1.1).
- Known PathWhiz rejections from repo docs (L): a protein without UniProt/DrugBank id is skipped and cascades (F-204); a reaction with `spontaneous=false` and no enzyme is dropped (reaction.rb:90-96); the interaction shape rules; canonical-name mismatch causes unique-key failure (import_fixes section 0). Offline checks cover the interaction rule and spontaneous+enzyme, not the other three.

## 5. Cheapest safe owner test (not attempted)

1. Use a staging or local PathWhiz, never production. Upload 3 files: (a) `reference/pathway (39).pwml` as positive control; (b) one Oct 2026 canonical run that passes QA and has at least 3 reactions and one interaction; (c) one Aug-Sep run with the failing interaction shape (PMC9544450 above) as negative control. If (c) imports clean, the ORCH-747 interaction rules are too strict. If (b) fails, validity is not established.
2. Look at: the HTTP result and the importer error hash (`@result[:errors]`, pwml_parser.rb:36,149,308,524-713); then per pathway, element counts versus the PWML (compounds, proteins, complexes, reactions, interactions), reactions that lost enzymes (F-204 cascade), species shown, duplicate compounds created, and that the canvas renders.
3. Then the existing IMPORT-SET files per the D-099 rule, the F-195 files last.

## 6. WHAT TO DO (proposals; none changes behaviour by itself)

1. **QUESTION for the owner:** run the section 5 test. It is the only way to close D-099 section 19; nothing in this repo can substitute.
2. **Stronger offline validator**, a new read-only script outside the product path: count reconciliation plus the documented importer rules (UniProt/DrugBank on every protein; enzyme-less reaction must be spontaneous; every `*-id` reference resolves to the right section; species and compound field match against data/pathwhiz_id_db.json). Gain: predicts the F-204 cascade that the importer hides. Risk: none if read-only and not wired into the writer.
3. **qa.py false positives:** count only `<interaction>` inside `<interactions>` that have children; declare or tolerate the `option:` namespace. Risk: low, but qa.py is imported by writer and postpipeline and a loosened check could let a bad file through; add tests using PW1.pwml and `pathway (39).pwml`. Touching a locked gate needs a D-xxx.
4. **Re-run today's qa.py on old outputs before anyone imports them:** 25 of 61 canonical Aug-Sep files carry importer-hostile interactions and their stored qa reports are stale.
5. Ask whether the `pw-id`, `cached-name`, `species-id` placeholders matter for upload (G).

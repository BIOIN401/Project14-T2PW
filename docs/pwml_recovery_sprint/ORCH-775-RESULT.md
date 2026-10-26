# ORCH-775 — Enzyme-less reactions in the PWML: is the enzyme in the paper, and what may fill the slot

**Task.** Read-only investigation (pwml-bio-auditor) of review finding N1 of ORCH-773 (`evidence/orch773/REVIEW-r1.md` lines
137-155): reactions exported with no enzyme and `spontaneous=false`, which PathWhiz drops on import
(`smpdb/app/models/data/reactions/reaction.rb:90-96`, `check_spontaneous`). Answers the product owner question, verbatim:
*"well are the enzymes in the paper? we add the enzyme — remember … when an entity doesn't match in the db we can create it,
check the docs. if it's still not there and not working we can even use an unknown protein as a sentinel."*

**Scope and rules kept.** No product code, PathWhiz code or database was changed. Two read-only scripts ran under
`bounded_run.py` without `--heavy-lock` (ORCH-774 holds it), each with `FINAL SURVIVING COUNT : 0`. DB access was SELECT only
through `PathBankDbResolver.from_env`; no credential is printed in any log (checked: 0 matches). Checkout HEAD `f3c330a4`
(sprint tip after the ORCH-773 merge `bab79db6`). Nothing committed by this task; main commits.

---

## 0. Answers in one screen

1. **Is the enzyme in the paper?** For 6 of the 20 reactions, yes, for that exact step, and Stage 1 extracted it (all six on
   ORCH-767 chlorophyll, PMC5130987). For 3 more the row own evidence quote names the enzyme, but each row is a duplicate of a
   reaction the PWML already draws with that enzyme (PMC12376012). For the other 11 the paper names no catalyst for the step:
   5 are an iron-chelation binding event, 1 is an assay reporter the gold forbids (LDH), 5 are steps the paper states without
   any catalyst (or names only a class). **No paper calls any of the 20 spontaneous or non-enzymatic** (grep of all six texts).
2. **Where were the six lost?** Not at quarantine and not at the writer. The Stage-4 LLM audit deleted them: the post-normalization
   contract report fed it "Protein 'X' is missing a UniProt or DrugBank identifier" (effect `feed_audit`), and its accepted
   patch "Removed 14 proteins lacking UniProt/DrugBank identifiers and all references to them in reactions". Stage 6, where
   the existing PathBank Unknown fallback would have wrapped each one, then had no enzyme left to wrap. For three of them
   (ALAD, CPO, PPO) an enzyme-bearing twin row survived the audit, and the replay admission quarantines that twin
   (`non_protein_actor_in_enzymes` / `_in_modifiers`) while it admits the enzyme-less one (KI-004 records this cost).
3. **What the docs allow.** Name-only compounds: allowed (no identity rule for compounds). New proteins: only with UniProt or
   DrugBank (D-106 §1). Generated single-protein wrappers and the **PathBank Unknown-backed functional complex** (generated
   wrapper named after the paper enzyme, single component = PathBank protein 9659 "Unknown"): already built, already sanctioned
   (`strict_quarantine.py:42-46`, `:728-732`; D-070 §O-1a/O-1b; TRAP-3), already shipping in 24 of the 72 tip corpus PWMLs. Forbidden: any
   forged accession (`placeholder_claims_real_identity`), and inventing an enzyme the paper does not state (PRODUCT_CONTRACT
   §1 "hard limit"); gold sets `unknown_backed_proteins_acceptable=false` on 3 of the 4 papers here that have gold.
4. **PathWhiz/PathBank.** PathBank uses Unknown-backed enzymes at scale: record 9659 (Arabidopsis) catalyses 315 reactions in
   1,773 pathways; Unknown-component complexes catalyse 231,287 reactions in 32,529 pathways. A sentinel wrapper would import
   by static reading (name "Unknown" waives the UniProt format, the literal satisfies presence; the wrapper complex needs no id).
   No import was run.
5. **Recommended rule.** (i) keep the paper enzyme: stop the audit from deleting a paper-explicit enzyme whose only defect is a
   missing identity (Stage 4, pre-freeze); (ii) ground it at Stage 6 (PathBank holds 5 of the 6 chlorophyll enzymes in species
   4), else the existing Unknown-backed wrapper; (iii) a reaction that still has no enzyme and is not paper-stated spontaneous
   is omitted pre-freeze and listed (a new per-row predicate equal to PathWhiz `check_spontaneous`). **Never** the sentinel on
   a reaction whose paper names no enzyme. Recovers 6/20 as catalysed reactions, 3/20 are duplicates of drawn reactions,
   11/20 are omitted and listed. **A sentinel is compatible with D-106 only when it backs an enzyme the paper names**; on a
   paper-silent step it does not break D-106 but it breaks the PRODUCT_CONTRACT §1 hard limit.

---

## 1. Inputs and evidence status

| what | where | committed? |
|---|---|---|
| leg inputs (`stage1_payload.json`, `merged_payload.json`, `final_mapped.json`, quarantine and removal reports) | leg dirs named in `evidence/orch768_corpus_manifest.json` | manifest committed (`8d38256c`) with sha256 of every input; `runs_verify/2026-08-18..08-28` legs git-tracked; `runs_verify/2026-09-06_1425`, `2026-09-16_2152` and the ORCH-756/767 worktree legs are untracked but sha256-pinned by the manifest |
| paper text | `<leg>/papers/<PMC>/01_source_text.txt` | tracked for the `runs_verify/2026-08-*` legs; untracked (on disk) for the others |
| tip replay (canonical + PWML per leg) | `C:/t/rev773r1/out-tip/<slug>/` (ORCH-773 reviewer scratch) | **not committed**. Its committed G11 reports (`evidence/g11/ORCH-773/112`, `123`) carry no stdout, so the PWML reaction ids come from the uncommitted log `C:/t/rev773r1/logs/123-pwml-analysis2.log` lines 86-106. Every biological claim below is also checked on the pinned leg input, which the replay reproduces byte-identically base to tip (REVIEW-r1 C2) |
| this task | `evidence/orch775_noenzyme_trace.{py,txt}`, `evidence/orch775_pathbank_unknown_probe.{py,txt}`, wrapper reports `evidence/orch775_g11_01_noenzyme_trace.json`, `evidence/orch775_g11_02_pathbank_unknown_probe.json` | written, not committed (main commits; move the two JSON reports under `evidence/g11/ORCH-775/` if G11 rule 9 requires that path) |

`final_mapped.json` caveat: 9 of the 10 legs recorded `quarantine_ok: true` (canonical checkpoint); `runs_verify/2026-08-18_1328`
recorded `quarantine_ok: false` (checkpoint `pre_freeze`), so its `final_mapped.json` is the enriched pre-quarantine payload. That
still answers "where was the enzyme lost" (it was already absent before quarantine) but is not a canonical-identity statement.

---

## 2. Part 1 — the 20 reactions: is the enzyme in the paper, and where was it lost

Population: the tip PWMLs written by the ORCH-773 reviewer replay hold 20 reactions with `<reaction-enzymes/>` and
`spontaneous=false` (`123-pwml-analysis2.log` lines 86-106). PWML reaction id = 50000 + index in `canonical.json` reactions.
**Correction to the charge and to REVIEW-r1:** the 14 base-exported ones sit on **8** base-passing legs, not 7 (the reviewer own
log lines 48-63 list 8: ORCH-756 enterobactin, ORCH-767 chlorophyll, rv 08-21_2239, 08-22_2147, 08-24_1203, 08-27_1341,
08-28_1816, 09-16_2152). The trace finds 21 enzyme-less rows in the tip canonicals; the 21st, ORCH-767 `singlet oxygen
generation from Pchlide` (id 50020), is exported WITH two reaction-enzymes because the IR promotes its `activator` modifiers to
enzymes (see side finding S2), so it is not in the 20.

"Lost" columns come from `evidence/orch775_noenzyme_trace.txt` (the reaction traced through `stage1_payload.json` →
`merged_payload.json` → `final_mapped.json` of the leg, by name and by input/output set). Rows absent from `stage1_payload.json`
and present in `merged_payload.json` entered at **Stage 2A** (attribution by set difference: those rows carry neither a RAG
carrier — `rag_provenance`/`source_papers`/`rag_confidence` all empty — nor a `provenance_lineage` stamp, while every Stage-1 row
of the same legs carries one; see S3).

| # | leg (paper) · PWML id | reaction (evidence quote in the row) | paper names a catalyst for THIS step? (quote from `01_source_text.txt`) | where the enzyme was lost |
|---|---|---|---|---|
| 1 | ORCH-756 enterobactin (PMC12096016) · 50005 | LDH-catalyzed reduction of pyruvate to lactate | **Yes, but an assay reagent**: "porcine lactate dehydrogenase (LDH) … purchased from Sigma‐Aldrich"; "LDH‐catalyzed conversion of pyruvate to lactate is then monitored by loss of OD 340". Gold forbids LDH, NADH, NAD+ and lactate | never extracted as enzyme; row born at Stage 2A. The row itself must not exist (S1) |
| 2 | ORCH-767 chlorophyll (PMC5130987) · 50001 | GSAT reaction | **Yes**: "GSA is converted to ALA by GSA aminotransferase (GSAT), which catalyzes the intramolecular transfer of an amino group" | extracted at Stage 1 (`enzymes: [GSAT]`), kept in merged; **removed by the Stage-4 audit** (`final_mapped` lineage `audit_repair/audit_modified` ×2, `enzymes: []`). Cause upstream: Stage-2B UniProt lookup `no_match` (queries `GSAT` and a junk literature alias `toolkit`, `contract_reports.json` post_normalization) |
| 3 | ORCH-767 · 50002 | ALA dehydratase reaction | **Yes**: "condensed asymmetrically to form … PBG, which is catalyzed by ALA dehydratase (ALAD; also known as PBG synthase)" | Stage 1 `[ALAD]` → audit removed. Twin `ALAD (HEMB1) reaction` keeps `Delta-aminolevulinic acid dehydratase 1, chloroplastic` but is quarantined at replay admission (`quarantined_broken_reference: non_protein_actor_in_modifiers`, FAR1); the live leg had admitted it |
| 4 | ORCH-767 · 50003 | PBG deaminase reaction | **Yes**: "Four PBG molecules are then sequentially polymerized by PBG deaminase" | Stage 1 `[PBG deaminase]` → audit removed (protein row then `degree_zero_after_quarantine`, `removed_entity_report.json`) |
| 5 | ORCH-767 · 50005 | Urogen III decarboxylase reaction | **Yes**: "metabolized to coproporphyrinogen (Coprogen) III by Urogen III decarboxylase (UROD)" | Stage 1 `[UROD]` → audit removed |
| 6 | ORCH-767 · 50006 | Coprogen III oxidase reaction | **Yes**: "Coprogen III is oxidized by Coprogen III oxidase (CPO)" | Stage 1 `[CPO1]` → audit removed. Twin `Coprogen III to Protogen IX` (enzymes `Coproporphyrinogen-III oxidase 1, chloroplastic` + `CPO1`) quarantined at replay admission `non_protein_actor_in_enzymes` (CPO1) |
| 7 | ORCH-767 · 50007 | Protogen IX oxidase reaction | **Yes**: "further oxidized by Protogen IX oxidase (PPO)" | Stage 1 `[PPO1]` → audit removed. Twin `Protogen IX to Proto IX` (enzymes `PPO` — an Unknown-backed wrapper — + `PPO1`) quarantined `non_protein_actor_in_enzymes` (PPO1) |
| 8 | rv 08-18_1328 (PMC12452463) · 50002 | enterobactin iron binding | **No.** "Enterobactin is secreted by bacteria into the extracellular environment, where it binds ferric iron with high affinity." A binding event; no catalyst, no "spontaneous" | never extracted; Stage 2A |
| 9 | rv 08-21_2239 (PMC12452463) · 50002 | Enterobactin iron binding | **No** (same passage; the row quote "forming ferric-enterobactin" is a paraphrase, not verbatim) | never extracted; Stage 2A |
| 10 | rv 08-22_2147 (PMC12452463) · 50003 | Enterobactin binding to ferric iron | **No** (same passage) | never extracted; Stage 2A |
| 11 | rv 08-28_1816 (PMC12452463) · 50004 | Enterobactin binding Fe3+ | **No** (same passage) | never extracted; Stage 2A |
| 12 | rv 08-24_1203 (PMC12096016) · 50006 | Enterobactin chelation of ferric iron | **No.** "Following secretion, enterobactin chelates extracellular ferric iron, and the ferric‐enterobactin complex is then imported via TonB‐dependent uptake" — part of the transport mention the gold excludes | never extracted; Stage 2A |
| 13 | rv 08-27_1341 (PMC12782028) · 50003 | cholesterol oxidation to oxysterols | **No.** "Excess cholesterol can be oxidized to form oxysterols, which can activate a variety of signaling pathways" (background; product is a class) | never extracted; Stage 2A (later audit-modified, no enzyme involved) |
| 14 | rv 09-16_2152 (PMC13170813) · 50003 | hordatine B formation | **No for HB.** "hordatine B (HB) is a heterodimer of p CA and FA". The paper names laccase only for HA: "laccase mediates HA formation … HA formed in reactions catalyzed by HvLAC1/2" | never extracted; Stage 2A. Attaching HvLAC1/2 would be inference by analogy |
| 15 | rv 09-16_2152 · 50004 | hordatine C formation | **No for HC** ("Hordatines A and C … are homodimers of p CA and FA, respectively"). The row also has one input for a homodimer | never extracted; Stage 2A |
| 16 | rv 09-06_1425 (PMC12376012) · 50002 | Ceramide synthase reaction (dihydrosphingosine → **ceramide**) | Names the class: "Dihydrosphingosine is N‐acylated by the ceramide synthases. In mammals, there are six isoforms" | Stage 1 row already had `enzymes: []` (never extracted for this row). A twin `ceramide synthase reaction` (→ dihydroceramide, enzyme `ceramide synthase`) is drawn. This row product is wrong by the route the paper itself gives (the desaturase makes ceramide) |
| 17 | rv 09-06_1425 · 50005 | Sphingomyelin synthase reaction | **Yes** in its own quote: "At the medial Golgi, sphingomyelin is produced from ceramide by sphingomyelin synthase 1" | never extracted; Stage 2A. **Duplicate**: twin `sphingomyelin synthase 1 reaction` (ceramide + phosphorylcholine → SM, Unknown-backed `sphingomyelin synthase 1`) is drawn |
| 18 | rv 09-06_1425 · 50006 | Glucosylceramide synthase reaction | **No.** "Alternately, ceramide may be converted to glucosylceramide or galactosylceramide in the trans‐Golgi." No "glucosylceramide synthase" or UGCG anywhere in the text; the enzyme is only in the model-written name | never extracted; Stage 2A |
| 19 | rv 09-06_1425 · 50008 | Sphingomyelin hydrolysis | Class only: "sphingomyelin can be hydrolyzed by one of several sphingomyelinases to generate ceramide" | never extracted; Stage 2A. **Duplicate**: twin `sphingomyelinase reaction` with the identical quote and an Unknown-backed `sphingomyelinase` is drawn |
| 20 | rv 09-06_1425 · 50009 | Glucosylceramide hydrolysis | **No.** "Glycosphingolipids undergo hydrolysis to ceramide, principally in the lysosome." (figure legend: "The sphingomyelinases and glycosidases liberate ceramide", a class). The row also narrows "glycosphingolipids" to glucosylceramide | never extracted; Stage 2A |

**No enzyme was found carried as a modifier and never attached** on any of the 20: every one has `modifiers: []` at
`final_mapped.json` and in the canonical (trace). The opposite happens once (S2).

**Tally.** Paper names the catalyst for the step and Stage 1 extracted it: 6 (#2-#7). Quote names it but the row is a duplicate
of a drawn reaction: 3 (#16, #17, #19; #16 also has the wrong product). Paper silent, class only, or a binding event: 10 (#8-#15,
#18, #20). Assay reporter forbidden by gold: 1 (#1).

### Side findings

* **S1. The LDH row is forbidden content** (gold PMC12096016 `forbidden_identifiers`: "lactate dehydrogenase … A porcine
  coupled-assay reporter enzyme, not an E. coli pathway member"; "NADH … NAD+ … lactate … Coupled-assay reporter species from the
  LDH readout"). It is KI-032. Attaching LDH would make it import, which is the wrong direction; the row must be omitted.
* **S2. The IR exports `activator` and `inhibitor` modifiers as reaction enzymes.** `pwml/ir.py:2140-2145` appends every
  modifier whose role is in `{catalyst, activator, inhibitor, enzyme}` to the PWML `<reaction-enzymes>`. ORCH-767
  `singlet oxygen generation from Pchlide` has `enzymes: []` and modifiers `glutamyl-tRNA reductase` and `FC1 complex`, both
  `role: activator`, evidence "HEMA2 and FC1 are also responsive to singlet oxygen signaling" (inferred). The tip PWML reaction
  50020 lists both complexes (40015, 40016) as catalysts. The paper describes a light-driven photosensitisation ("With light
  exposure, Pchlide accumulated in the mutant generates singlet oxygen"), not catalysis. Outside this task; adjudicated in § 6.
* **S3. Stage-2A rows carry no provenance.** In all 10 legs every Stage-1 row has `provenance_lineage`, and every row added
  between `stage1_payload.json` and `merged_payload.json` has none and no RAG carrier (per leg, the count of
  `provenance_lineage` keys is equal in the two files). PRODUCT_CONTRACT § 3 requires the stage that introduced a row.
  Instrumentation needed: a lineage stamp at the Stage-2A merge (`stage: "stage2a_inference"`). Until it exists, Stage 2A is
  attributable only by set difference, as done here; that is a measured fact about these 10 legs, not a general proof.
* **S4. A junk literature alias reached the UniProt query**: `GSAT` was also searched as `toolkit`
  (`contract_reports.json` post_normalization, `mapping_meta.literature_aliases`). Not investigated further.

---

## 3. Part 2 — what the project already allows and forbids

### 3.1 Creating an entity when the database match fails

| kind | rule today | source |
|---|---|---|
| **compound** | **Allowed name-only.** Strict representability has no compound identity rule (falls through to `True`). PathWhiz creates a compound with only a unique name (`hmdb_id` `allow_blank`). | `strict_quarantine.py:790`; `smpdb/app/models/data/compound.rb:38-41` |
| **protein** | **Only with a UniProt or DrugBank id**, or as the PathBank Unknown sentinel. "A protein without a valid UniProt or DrugBank identity is not exported as if it had one, and no lane proposes downgrading the gate." | **D-106 § 1**; `strict_quarantine.py:742-751`; `entity_identity.py:62-84`; KI-001 (by design) |
| **protein complex (declared)** | Allowed with no id; a declared complex with no components is a warning. | `strict_quarantine.py:753-763`; ORCH-761 § 1.3 |
| **generated single-protein wrapper** | Built at Stage 6 for a reaction enzyme (PathWhiz attaches enzymes to complexes only): `generated: true`, `generation_reason: single_protein_pathwhiz_wrapper`, rule `novel_enzyme_single_component_complex`; its component must itself be representable. | `map_ids.py:7888-7940`, `:8148-8210`; `entity_identity.py:317-336`; `strict_quarantine.py:764-788`; KI-006 |
| **Unknown-backed functional complex** | **Already built and sanctioned.** `_apply_pathbank_unknown_enzyme_fallback` (Stage 6, run at `map_ids.py:11908-11922` when `allow_complex_wrapper_creation`) wraps a still-unresolved reaction enzyme in a generated complex that keeps the **functional enzyme name** and has one component, PathBank protein **9659 "Unknown"** (uniprot literal `Unknown`, species Arabidopsis 4 / taxon 3702), with `identity_status: placeholder`, `fallback_used: true`, `real_mapping_failure_reason`, `role_evidence`. Docstring: "Correctly formed Unknown-backed functional complexes are explicitly valid: a complex whose component is the PathBank Unknown sentinel is the sanctioned representation for an enzyme the databases do not carry". | `map_ids.py:8916-9561`; `strict_quarantine.py:42-46`, `:728-732`; `entity_identity.py:87-116` |
| **complex named in the source** | `complex_named_source_entity_wrapper` (normalizer) is a claim about an assembly; D-113 § 1 quarantines an invented complex-named wrapper when the paper catalyst remains. | `process_normalizer.relocate_complex_named_proteins`; **D-113 § 1-2**; KI-012 |

**Preconditions of the Unknown fallback** (why it did nothing for the 20): it rewrites an actor already present in
`reactions[].enzymes` (`map_ids.py:9328-9335`); the name must be usable, not "unknown", "hypothetical", "uncharacterized" and the
like (`:8963-8977`); the role must be a catalyst claim and the actor not a compound (`:9059-9124`, otherwise the claim is
quarantined: "Wrapping an unsupported claim in a functional complex would manufacture an enzyme the paper never stated, which is
worse than a missing identifier"); the protein must appear nowhere else in a non-enzyme role (`:8989-9051`). **It never adds an
enzyme to a reaction with `enzymes: []`.** On the six chlorophyll reactions it had nothing to wrap because the Stage-4 audit had
already removed the actor and its protein row.

### 3.2 The Unknown / placeholder / sentinel decisions

* **D-070 § O-1a (LOCKED):** "PathBank record **9659** is the `Unknown` protein, and *Arabidopsis thaliana* is that record own
  species … Under the current PathBank representation it is **not** a defect." Treat as sentinel rows; do not strip or rewrite
  their identity; "None of the 21 sets `placeholder_claims_real_identity` — so none of them is a forged identity".
* **D-070 § O-1b:** generated wrappers "are legitimate generated-wrapper biology. **TRAP-3 protection stands** … placeholder
  status still may not forge a real database identity."
* **D-070 "Still forbidden":** C-098c refused ("an export-time fallback that replaces a false Arabidopsis at mapping time with a
  false default species at export time is merge rule 8"); "No path may end at `writer.py` `default_species_id`".
* **D-071 / D-074 (LOCKED):** scoring tolerance only. PMC12444477 alone may tolerate "the exact PathBank `Unknown` sentinel,
  and nothing else", via the row-aware `is_pathbank_unknown_protein(row)` predicate; "Do not implement this by adding `Unknown`
  to a name-only allowlist." C-100 pins the consequence: `tests/test_c100_tolerance_scope.py:259`
  `test_5b_the_bare_pathbank_sentinel_name_is_not_a_tolerated_entity`.
* **PRODUCT_CONTRACT § 13 / MASTER_PLAN TRAP-3:** `placeholder_backed_proteins` is a **standing policy disagreement** ("the
  pipeline treats `Unknown`-backed export as legitimate biology preservation. No agent may "fix" it. Escalate."). This report
  does not reclassify it and proposes no change to it; it only notes that any wider sentinel use moves that count.
* **Gold, per case:** `unknown_backed_proteins_acceptable` is **false** for PMC12096016 ("Every protein described
  catalytically is a named, canonical E. coli gene product … Any unknown-backed protein is fabricated."), PMC12452463 ("An
  extractor must also not invent a backing entity to bridge the missing EntA step.") and PMC12782028 ("All four expected
  enzymes are well-characterised human proteins … so every one resolves deterministically."). PMC5130987, PMC12376012 and
  PMC13170813 have **no gold case** (`src/t2pw/bench/gold/pinned_v1.json`, `src/t2pw/bench/eval_v2/gold_v2_supported_reactions.json`).
* **Invented ids:** the writer emits `pwp-id` `PW_P{document id:06d}`, which collides with PathBank complexes
  (`PW_P040000` on complex 40000 and on 431773, 431796, 431802); KI-035, **open** (ORCH-761 § 9).
* **D-114 AMENDMENT 1 § 1 (LOCKED):** "Reaction with a failing enzyme: omit the whole reaction … Keeping it without its enzyme
  would either fail import (`spontaneous=false`) or falsely claim it is spontaneous. Either breaks D-114 § 2". No rule yet
  applies that reasoning to a reaction that has **no** enzyme at all (REVIEW-r1 N1: "no T2PW gate requires a reaction enzyme").
* **PRODUCT_CONTRACT § 1 hard limit:** "The system must **never invent** reactions, directionality, stoichiometry, identities,
  **enzymes**, locations or other biological content merely to guarantee a PWML file." § 2: "no assay reporters as pathway
  members".
* Spontaneity is not modelled: `process_normalizer.force_reactions_non_spontaneous` (`:99-109`) and `pwml/ir.py:2064-2065`
  force `spontaneous=false` on every reaction.

**What is allowed today, exactly.** Name-only compounds; declared complexes without ids; Stage-6 generated wrappers around a
protein that has a UniProt/DrugBank id; and, for a reaction enzyme the payload names but no database carries, an Unknown-backed
wrapper keeping that name. **What is forbidden today.** A protein row without UniProt/DrugBank exported as if real; a placeholder
carrying a real accession, a `matched` resolution, or no `fallback_used` (`malformed_placeholder:*`); any post-freeze or
export-time repair (merge rule 8, C-098c); an enzyme the source does not state (contract § 1); the bare word `Unknown` as a
name-only tolerance (D-074).

---

## 4. Part 3 — what PathWhiz and PathBank actually do

### 4.1 PathWhiz importer (static reading of `C:/Users/Angad/Desktop/SummerBIOIN/smpdb`, re-verified this task)

| element | can it be created on import without UniProt/DrugBank/HMDB? | source |
|---|---|---|
| PWML `<id>` | document-internal only, never a DB lookup ("we assume that all ids numbers are internal to the PWML document"). Matching is by external id for Compound (`hmdb_id`) and Species (`taxonomy_id`) only, else `klass.find_by(<all present leaf fields>)`, else `new(...).save` | `lib/pwml_parser.rb:5-7`, `:629-637`, `:648-690` |
| Compound | **yes**: name present and unique; `hmdb_id`, `pwc_id` `allow_blank` | `compound.rb:38-41` |
| Protein | **no**, except one name: `uniprot_id` presence unless `drugbank_id`, and vice versa (`:57-59`); the UniProt format check is skipped `unless p.name == "Unknown"` (`:52-55`). So a protein named exactly `Unknown` with `uniprot_id "Unknown"` saves; PathWhiz itself creates such rows (`Protein.find_or_create_by(name: "Unknown", uniprot_id: "Unknown", species: new_species)`, `:221-224`) | `protein_complexes/protein.rb:46-64`, `:215-226` |
| ProteinComplex | **yes, no id field at all**: name present and unique per species, species present, at least one component (`:45-47`); matched by species + name before save, and then **binds to that existing complex and its components** | `protein_complexes/protein_complex.rb:45-52`, `:313-334` |
| Reaction | at least 1 left and 1 right; `check_spontaneous`: 0 enzymes and `spontaneous == false` fails; enzymes and `spontaneous == true` fails. `spontaneous` is a nullable boolean with no default (`db/schema.rb:938`), and `nil == false` is false in Ruby, so **0 enzymes with `spontaneous` NULL passes** (static reading) | `reactions/reaction.rb:31-37`, `:90-96`; `db/schema.rb:934-938` |

ORCH-761 § 1.2 already recorded `protein.rb:57,59`; re-verified unchanged. A failing element is skipped silently and the upload
still reports success (`pathways_controller.rb:225-241`; KI-034), which is why the 20 reactions vanish without an error.

### 4.2 PathBank database (read-only; `evidence/orch775_pathbank_unknown_probe.txt`)

* **The sentinel record.** `proteins.id = 9659`: name `Unknown`, `uniprot_id` `Unknown`, no DrugBank, species 4 *Arabidopsis
  thaliana* (taxon 3702), created 2015-04-10. It is the only `Unknown` row in species 4.
* **Per-species Unknown rows.** 2,196 protein rows named `Unknown` across 2,144 species (2,195 with the literal `Unknown`
  accession; row 7725, *Mus musculus*, carries `Z00Z00`). Species of the papers here: human 7866, 10207, 27280; *E. coli*
  (species 3) 21 rows; K-12 (25) 11484; mouse 12 rows; Arabidopsis only 9659; **barley (151) none**.
* **Unknown complexes.** 1,037 complexes are **named** `Unknown`, one per species (e.g. 14277 Arabidopsis, 14972 human).
  326,336 complexes contain an Unknown component; **323,420 contain only Unknown components** — the same shape T2PW builds
  (functional name, Unknown component).
* **Use as a reaction enzyme.** Complexes with an Unknown component catalyse **231,287 reactions** (83,333 complexes) drawn in
  **32,529 pathways**. Record 9659 itself: 220 complexes, **315 reactions, 1,773 pathways**. Examples (PW via
  `super_pathway_visualizations.pw_id`): **PW000839** Glycolysis, *A. thaliana* — `Aldose 1-epimerase` (complex 3445, reaction
  3096), `Glucose-6-phosphate isomerase` (3446), `Alpha-enolase` (3458), all on 9659; **PW002518** Gluconeogenesis from
  L-Malic Acid, *A. thaliana* — `phosphoglycerate kinase` (4812), `Glyceraldehyde 3-phosphate dehydrogenase` (4813). Other
  species: **PW000794** Proline Metabolism, *E. coli* — a complex named `Unknown` (protein 7752); **PW000760** Biosynthesis of
  Siderophore Group Nonribosomal Peptides, *E. coli* — `ferric enterobactin ABC transporter` with Unknown 11484.
* **How PathBank records a reaction with no enzyme.** Of 916,213 reactions: with an enzyme, 603,022 `spontaneous` NULL and
  300,278 `false`; **with no enzyme, 12,228 NULL, 456 `true`, 229 `false`**. So PathBank mostly leaves an uncatalysed reaction
  NULL, not `true`; the 229 `false` rows predate or bypass the validator.
* **Do the papers' enzymes exist?** Arabidopsis (species 4) by `gene_name`: GSA2 Q42522 (protein 23653; no GSA1 row), HEMB1
  Q9SFH9 (161092), HEMC Q43316 (161093, 23649), HEME1 Q93ZB6 (161095), HEME2 O22886 (23645), CPX1 Q9LR75 (161096); **no PPO1/PPOX
  row**. Human (species 1): CERS1-6, SGMS1 Q86VZ5 (720), SGMS2, UGCG Q16739 (790), SMPD1-3, GBA P04062 (1612), GBA2, LDHA.

### 4.3 Would a T2PW PWML that references the sentinel import?

**By static reading, yes; not tested by an import.** The Unknown fallback writes the component protein as name `Unknown`,
`uniprot_id` `Unknown`, species Arabidopsis / taxon 3702 (`map_ids.py:9254-9271`). PathWhiz maps the species by taxonomy id
(`pwml_parser.rb:632-633`) to species 4, then `Protein.find_by(name, uniprot_id, species_id)` returns 9659 (the only such row in
species 4); had it missed, a new `Unknown` row still saves (`protein.rb:52-57`). The wrapper complex has no id requirement and one
resolved component, so it saves or binds by species + name; the reaction then has a ReactionEnzyme and passes
`check_spontaneous`. The corpus already ships this shape: 24 of the 72 tip `outputs/pathway.pwml` files contain a `<name>Unknown</name>` protein (e.g.
ORCH-756 menaquinone, ORCH-759p chlorophyll, rv 08-24_1428 PMC12444477). Two caveats, both static: a wrapper named like an
existing PathBank complex of the same species binds to THAT complex and its real components (ORCH-761 § 1.3); and "matched by
id" never happens — PathWhiz ignores PWML ids, so it is always a match by name + accession + species.

---

## 5. Part 4 — recommended rule, and what it recovers on the 20

### 5.1 The rule

1. **The paper names the enzyme for the step → it stays attached, and the fix sits where it was lost: Stage 4, pre-freeze.**
   In all six such cases Stage 1 extracted it; the audit removed it because the post-normalization contract listed the protein
   as missing a UniProt/DrugBank id. That defect is Stage-6 work (mapping, or the Unknown fallback), not a reason to delete a
   paper-explicit catalyst. Recommended shape: the audit patch applier refuses, and records, an accepted patch that removes a
   `paper_extraction`/`explicit` enzyme actor (or the protein row it names) when the only error cited against it is the missing
   external identity. Deterministic and testable offline with a fixture; its effect on a leg is measurable only live (KI-058).
   Enzymes that appear only in a row evidence quote (#16, #17, #19) are **not** attached: each row duplicates a reaction the PWML
   already draws with that enzyme, and a quote-reading rule is a lexical gate (not recommended without a PO ruling).
2. **Not in the database under the paper name → ground it at Stage 6 by the existing rules, else the existing Unknown-backed
   wrapper.** No new creation path is needed and none is possible: PathWhiz cannot create an identity-less protein other than
   one named `Unknown` (§ 4.1). PathBank species 4 already holds HEMB1 Q9SFH9, HEMC Q43316, CPX1 Q9LR75 and one isoform each of
   GSA (GSA2 Q42522) and UROD (HEME1 Q93ZB6, HEME2 O22886); the isoform choice for GSAT and UROD belongs to the identity verdict
   (D-003, D-106, D-108) and may end Unknown-backed. PPO has no PathBank row, and the leg already holds an Unknown-backed `PPO`.
3. **Still no enzyme → omit the reaction before the freeze and list it**, unless the paper states it is spontaneous or
   non-enzymatic (none of the 20 does). This is a new per-row predicate equal to PathWhiz `check_spontaneous`, so it relaxes
   nothing and makes the PWML import-true. **The sentinel is never used on a reaction whose paper names no enzyme**: that is an
   invented enzyme (PRODUCT_CONTRACT § 1) and, on PMC12096016, PMC12452463 and PMC12782028, fabricated content by the gold
   `unknown_backed_rationale`.

### 5.2 What it recovers

| outcome | n | rows |
|---|---:|---|
| imports **catalysed** (step 1 keeps the enzyme, step 2 grounds or wraps it) | 6 | #2 GSAT, #3 ALAD, #4 PBG deaminase, #5 UROD, #6 CPO, #7 PPO. #6 and #7 also need KI-004/KI-011 cleared (CPO1 and PPO1 are declared as genes too) |
| omitted and listed **as a duplicate** of a drawn reaction | 3 | #16, #17, #19 |
| omitted and listed (paper silent, class only, binding event, or forbidden) | 11 | #1 (also KI-032), #8-#12, #13, #14, #15, #18, #20 |

A sentinel would add nothing to the first row beyond what the existing fallback already does, and on the other 14 it would turn
an honest omission into a fabricated catalytic claim.

### 5.3 What each step costs, and what it amends

| step | cost | decision touched |
|---|---|---|
| 1 audit guard | one lane in `src/t2pw/curation/apply_audit_patch.py` (or the `feed_audit` wiring of the post-normalization report); offline fixture, live-only leg effect; keeps more identity-less proteins alive to Stage 6, so more Unknown-backed wrappers and a higher `placeholder_backed_proteins` count (a standing policy disagreement, recorded, not a defect) | none amended; consistent with D-106 § 2 ("the remedy is upstream") and merge rule 7 |
| 2 grounding / wrapper | no new code expected (ORCH-762 FX1-FX5, existing fallback); not measurable offline | none (D-106, D-108, D-070 as written) |
| 3 omit predicate | removes 20 reactions from 10 legs (14 on 8 base-passing legs); every leg still draws at least 2 reactions (canonical counts 6/7, 16/22, 3/4, 2/3, 3/4, 6/7, 3/4, 4/5, 9/14, 3/5), so no leg stops writing a PWML, but any `release_ready` leg among them becomes `review_required` (D-114 § 2) and reported coverage falls to the import-true value (rv 09-06_1425 PMC12376012: 15/17 = 88.2% claimed, 10/17 = 58.8% true) | **extends D-114 AMENDMENT 1 § 1** from "a failing enzyme" to "no enzyme"; needs a PO D-entry because it changes base-passing PWMLs (D-112 § 2) |

### 5.4 Is a sentinel compatible with D-106?

**Yes, when it backs an enzyme the paper names; that is today behaviour and needs no amendment.** The Unknown-backed wrapper
never gives the paper enzyme an identity it lacks: the identity it exports is PathBank record 9659, whose literal `Unknown`
accession PathWhiz accepts by name (`protein.rb:52-57`); D-070 § O-1a rules it is not a forged identity; any placeholder that
carries a real accession is refused (`placeholder_claims_real_identity`, `malformed_placeholder:*`); and ORCH-761 § 2 rates T2PW
sentinel admission **stricter** than PathWhiz. No gate is relaxed. **On a reaction whose paper names no enzyme**, the sentinel does
not break D-106 (no protein is passed off as identified) but it breaks the PRODUCT_CONTRACT § 1 hard limit and the gold on three
cases. That use would amend the contract, not D-106, and it is not recommended.

---

## 6. Adjudications

Format: FINDING · CLASS · GOLD CITATION · ARTIFACT EVIDENCE · AFFECTED FILES · EXPECTED CORRECTION · REGRESSION FIXTURE ·
CONFIDENCE. Affected files are from static inspection; the implementer confirms the seam.

### A1. Reactions with no enzyme are exported with `spontaneous=false` (all 20)

* **CLASS:** `product_contract_violation` — D-114 § 2 "everything that IS in the PWML still meets every requirement, so the file
  imports"; PathWhiz `reaction.rb:93-94`; the reasoning of D-114 AMENDMENT 1 § 1 applies a fortiori.
* **GOLD:** PMC12096016 `export_rationale` "Export must exclude MenD, LDH and the transport mentions." (#1, #12). PMC12452463
  `export_rationale` "route is chemically BROKEN: with EntA absent, nothing converts 2,3-dihydro-2,3-dihydroxybenzoate onward" and
  `unknown_backed_rationale` "An extractor must also not invent a backing entity to bridge the missing EntA step." (#8-#11; the
  standing position holds: these legs stay `review_required`, never strict success). PMC12782028 `relevance_note` "the
  chemistry is cited background in the Discussion; the paper own output is transcriptomic only" (#13). PMC5130987, PMC12376012,
  PMC13170813: **no gold case**; adjudicated against the contract only.
* **ARTIFACTS:** each leg `final_mapped.json` (sha256 in `evidence/orch768_corpus_manifest.json`), reaction rows quoted in
  `evidence/orch775_noenzyme_trace.txt`; tip PWML ids in `C:/t/rev773r1/logs/123-pwml-analysis2.log:86-106` (uncommitted).
* **AFFECTED FILES:** `src/t2pw/pwml/ir.py` `validate_required_pwml_contract` (reaction checks near `:3015-3150`: add the
  `check_spontaneous` rule as an error on the reaction pointer) so the ORCH-773 per-row pass
  (`src/t2pw/pipeline/strict_quarantine.py` `_row_failures` / `_apply_per_row_failures`, `:2726-3090`) omits it;
  `src/t2pw/app/reaction_evidence.py` reason text; a new KI in `PWML_KNOWN_ISSUES.md`.
* **EXPECTED:** a reaction with 0 enzymes whose `spontaneous` is not `true` is removed before the freeze, recorded with a reason
  and listed under OMITTED FROM THE PWML; the PWML is written `review_required`; coverage is import-true.
* **FIXTURE:** canonical payload, species *E. coli* (taxon 562); compounds A, B, C (names only); protein P (UniProt P0AEJ2,
  species 562) in generated wrapper `P complex`; reactions R1 A→B `enzymes: [P complex]`, R2 B→C `enzymes: []`, `modifiers: []`,
  `spontaneous: false`. Base: PWML has 2 reactions, R2 with `<reaction-enzymes/>` (fails behaviourally). Tip: 1 reaction, R2 in
  the omissions.
* **CONFIDENCE:** high. Would change my mind: a PathWhiz import that keeps such a reaction, or a PO ruling for the NULL option
  (Q1 b).

### A2. The Stage-4 audit deletes paper-explicit enzymes whose only defect is a missing identity (#2-#7)

* **CLASS:** `product_contract_violation` — PRODUCT_CONTRACT § 2 ("correct enzyme … relationships") and merge rule 7; D-106 § 2
  places the remedy upstream in the lookup, not in deletion.
* **GOLD:** none (PMC5130987 has no gold case).
* **ARTIFACTS:** ORCH-767 chlorophyll leg
  (`.claude/worktrees/orch767/runs_papers/ORCH-767/legs/chlorophyll-strict/2026-10-23_2328/papers/PMC5130987/strict/`):
  `stage1_payload.json` and `merged_payload.json` carry `GSAT`, `ALAD`, `PBG deaminase`, `UROD`, `CPO1`, `PPO1`;
  `final_mapped.json` rows carry `enzymes: []` with the lineage reason "an accepted audit patch (replace) operation produced this
  row content" (the artifact writes 'replace' and "this row's"); `audit_iteration_summary.json` `llm_repair_rationale` "Removed 14 proteins
  lacking UniProt/DrugBank identifiers and all references to them in reactions"; `contract_reports.json` post_normalization
  errors "Protein 'GSAT' is missing a UniProt or DrugBank identifier.", effect
  `feed_audit`.
* **AFFECTED FILES:** `src/t2pw/curation/apply_audit_patch.py` (acceptance guard), or the `feed_audit` routing of
  identity-only errors in the post-normalization contract report.
* **EXPECTED:** the six Stage-1 enzymes reach Stage 6 and are grounded (HEMB1, HEMC, CPX1; GSA and UROD isoform by verdict) or
  wrapped Unknown-backed; the six reactions import catalysed.
* **FIXTURE:** post-normalization payload with `GSAT reaction` (Glu 1-semialdehyde → 5-aminolevulinic acid, `enzymes: [GSAT]`,
  lineage `paper_extraction`/`explicit`), protein `GSAT` with no accession and species Arabidopsis (3702), and an audit patch
  [remove the `GSAT` protein row; replace the reaction with `enzymes: []`] above the acceptance threshold. Base: applied
  (enzymes `[]`). Tip: refused and recorded; the enzyme stays.
* **CONFIDENCE:** medium-high. The accepted patch ops are not kept in the leg dir (only the summary and the lineage stamp), so
  per-op attribution is inferred. Would change my mind: retained ops showing a different op emptied the enzymes.
  Instrumentation needed: persist accepted audit ops per round in the leg dir.

### A3. The enzyme-bearing twin is quarantined while the enzyme-less twin is admitted (#3, #6, #7)

* **CLASS:** `product_contract_violation` (PRODUCT_CONTRACT § 2), already recorded as the cost of ORCH-769b (KI-004: "ALAD, CPO
  and PPO export uncatalysed"). The fix is gated on a PO ruling (Q4).
* **GOLD:** none.
* **ARTIFACTS:** tip replay `C:/t/rev773r1/out-tip/ORCH-767_chlorophyll-strict_2026-10-23_2328/outputs/quarantine_report.json`
  (uncommitted): `ALAD (HEMB1) reaction` `quarantined_broken_reference` / `non_protein_actor_in_modifiers` (FAR1);
  `Coprogen III to Protogen IX` and `Protogen IX to Proto IX` / `non_protein_actor_in_enzymes` (CPO1, PPO1). The leg own live
  `quarantine_report.json` admitted all three (`core_accepted` / `auxiliary_accepted`).
* **AFFECTED FILES:** `src/t2pw/pipeline/strict_quarantine.py` `_actor_verdict` (`:1355`), admission path.
* **EXPECTED (if the PO extends AMENDMENT 1 to admission):** the twin keeps its valid enzyme, the bad actor is dropped and listed;
  the enzyme-less twin is then a duplicate and A1 omits it.
* **FIXTURE:** two reactions with the same inputs and outputs; Ra `enzymes: []`; Rb `enzymes: [PPO (Unknown-backed wrapper),
  PPO1]` with `PPO1` also a `nucleic_acids` row. Base: Ra exported enzyme-less, Rb quarantined. Expected: Rb exported with PPO,
  Ra omitted.
* **CONFIDENCE:** medium (live and replay disagree; a live run at tip is needed).

### A4. Activator and inhibitor modifiers are exported as reaction enzymes (S2)

* **CLASS:** `product_contract_violation` — PRODUCT_CONTRACT § 2 ("correct enzyme, modifier … relationships") and § 1 (never
  invent enzymes).
* **GOLD:** none (PMC5130987).
* **ARTIFACTS:** `src/t2pw/pwml/ir.py:2140-2145`; tip PWML reaction 50020 of ORCH-767 chlorophyll lists complexes 40015 and
  40016; the canonical row has `enzymes: []` and both modifiers `role: activator`.
* **AFFECTED FILES:** `src/t2pw/pwml/ir.py:2137-2146`.
* **EXPECTED:** only `catalyst`/`enzyme` roles become `<reaction-enzyme>`; regulation stays out (PathWhiz carries it as an
  Activation/Inhibition interaction).
* **FIXTURE:** reaction A→B, `enzymes: []`, `modifiers: [{entity: "X complex", entity_type: "protein_complex", role:
  "activator"}]`, X complex a valid wrapper. Base: X complex is a reaction enzyme. Expected: it is not.
* **CONFIDENCE:** high on the code; the blast radius over the corpus was not measured.

### A5. The LDH row (#1) is off-pathway content

* **CLASS:** `product_contract_violation` — PRODUCT_CONTRACT § 2 "no assay reporters as pathway members". Already KI-032 (open).
* **GOLD:** PMC12096016 `forbidden_identifiers` (LDH; NADH, NAD+, lactate) and `export_rationale` "Export must exclude MenD,
  LDH and the transport mentions."
* No new code proposed here: A1 would remove it, but for the wrong recorded reason; KI-032 owns the right one.

### A6. Stage-2A rows carry no lineage (S3)

* **CLASS:** `product_contract_violation` — PRODUCT_CONTRACT § 3 (the stage that introduced every added process).
* **EXPECTED / instrumentation:** stamp `provenance_lineage` at the Stage-2A merge. Fixture: a payload where Stage 2A adds one
  reaction; expected the row carries a Stage-2A lineage entry. Confidence high on the observation (10 of 10 legs).

---

## 7. Dead ends, corrections and theories ruled out

1. **"The enzyme sat in `modifiers` and was never attached."** Ruled out for all 20: `modifiers: []` on every row at
   `final_mapped.json` and in the canonical (trace). The reverse defect exists (S2).
2. **"ORCH-773 stripped the enzymes."** Ruled out (REVIEW-r1 C2 already showed it; this trace confirms every one of the 20 has
   `enzymes: []` in the leg own `final_mapped.json`, i.e. it was lost before quarantine).
3. **"The Unknown fallback should have caught the six chlorophyll enzymes."** It would have, had the actors reached Stage 6; the
   audit removed them first. The fallback only rewrites an actor already present (§ 3.1).
4. **Reviewer count.** REVIEW-r1 N1 and the charge say 14 reactions on 7 base-passing legs; the reviewer own log lists **8** legs.
5. **My trace label is off by one.** `orch775_noenzyme_trace.txt` prints `pwml id 50001+k`; the true id is `50000+k`
   (checked on ORCH-767 id 50020 = index 20, which carries 2 enzymes, and on all 20 against `123-pwml-analysis2.log`). Indices
   and names in the trace are right; the table in § 2 uses the corrected ids.
6. **Process incidents.** (a) The first heredoc of the PathBank probe failed on a single quote; the script was assembled from
   parts in `C:/t/bt-orch775/` (same content). (b) A `py_compile` check of the probe ran outside the wrapper and wrote one `.pyc`
   into `docs/pwml_recovery_sprint/evidence/__pycache__/` (gitignored); I deleted that single file. No other write outside the
   permitted paths. Both wrapper runs ended `FINAL SURVIVING COUNT : 0`; none of my PIDs (421484, 418032, 437000, 436952) was
   alive afterwards.

## 8. Open questions for the product owner

**Q1. A reaction with no enzyme: what does the PWML do?** (a) **omit and list** (recommended; extends D-114 AMENDMENT 1 § 1;
removes 20 reactions on 10 legs, coverage becomes import-true, affected `release_ready` legs become `review_required`); (b) write
`spontaneous` as **NULL** — PathWhiz accepts 0 enzymes with NULL by static reading (`reaction.rb:93`, `schema.rb:938`) and PathBank
holds 12,228 such reactions, so 11 paper-silent rows would import uncatalysed without an invented enzyme or a false
"spontaneous"; cost: amends the AMENDMENT 1 § 1 reasoning and `force_reactions_non_spontaneous` / `ir.py:2064`, needs a real
import test, and #1 (forbidden) and the 3 duplicates must still be omitted; (c) accept as a KI with a coverage caveat (REVIEW-r1
option a): coverage stays overstated (88.2% shown, 58.8% imported on PMC12376012).

**Q2. The sentinel on a reaction whose paper names no enzyme?** Recommended **no**. Yes would amend PRODUCT_CONTRACT § 1 and
contradict the gold `unknown_backed_proteins_acceptable=false` on PMC12096016, PMC12452463 and PMC12782028 (7 of the 20 rows sit
on those papers), which is a gold edit with its own D-entry and gold A/B. It would also raise `placeholder_backed_proteins`
(standing policy disagreement).

**Q3. May the audit stop deleting a paper-explicit enzyme whose only defect is a missing identity (A2)?** One lane; offline
fixture, live-only leg effect (KI-058). Recovers up to 6 of the 20.

**Q4. When one of two enzymes fails at admission, keep the reaction with the valid one (A3)?** Extends AMENDMENT 1 to the
admission path (the enzyme analogue of REVIEW-r1 N3). Literal AMENDMENT 1 § 1 omits the whole reaction; but when a valid enzyme
remains, the reaction imports, which is the reason § 1 gives.

**Q5. Activator/inhibitor modifiers as reaction enzymes (A4):** fix, accepting that it changes base-passing PWMLs not measured
here?

## 9. What is NOT settled

* **No PathWhiz import was run.** Every import statement is static reading of `smpdb` (§ 4.1, § 4.3).
* Stage-2A attribution is by set difference only (S3); the per-op audit attribution (A2) rests on the lineage stamp and the
  rationale text, not retained ops.
* GSAT (GSA1 vs GSA2) and UROD (HEME1 vs HEME2): the identity verdict decides; either may end Unknown-backed.
* #6 and #7 also depend on KI-004/KI-011 (CPO1 and PPO1 declared as genes as well as proteins).
* The ALAD/CPO/PPO twins were admitted live and quarantined on replay; the tip live behaviour needs a live run.
* The tip replay outputs are uncommitted scratch (`C:/t/rev773r1/`); a committed copy of the 20-row list would need main to
  commit a reviewer log.

## 10. Evidence index

| file | sha256 | what |
|---|---|---|
| `evidence/orch775_noenzyme_trace.py` | `2272252fb5308edf9ccbcf6e4977e500edfc9a7c9447f2bea318e85bdfd0e292` | trace of each enzyme-less tip reaction through the leg payloads |
| `evidence/orch775_noenzyme_trace.txt` | `775809684d8b38f17e6e2f63e2fbc7b5317ecbb9131463a8faa2ada54093d3bb` | its output, with the cleanup report (0.28 s, 0 survivors) |
| `evidence/orch775_g11_01_noenzyme_trace.json` | `315fd77c771fee9b2baf6aab267b250c1f09341915799c001292f9554404bc3e` | wrapper JSON report |
| `evidence/orch775_pathbank_unknown_probe.py` | `e93071a8df5c8e7c6cd6c569fca34b0c4ca33cd470afeaa08a91de985e1d4d85` | read-only PathBank probe (SELECT only, asserted) |
| `evidence/orch775_pathbank_unknown_probe.txt` | `b65c34d3336f8593fc3ce847aa695bed3655c46d2ab0b89867c513405299eb62` | its output, with the cleanup report (247 s, 0 survivors) |
| `evidence/orch775_g11_02_pathbank_unknown_probe.json` | `4a0d7c8622602b2450c4055b235105acc56423bd96ff3d49405d66070673ef4f` | wrapper JSON report |

Inputs read: `evidence/orch768_corpus_manifest.json`; `evidence/orch773/REVIEW-r1.md`; `C:/t/rev773r1/logs/113-noenzyme-probe.log`,
`123-pwml-analysis2.log`; `C:/t/rev773r1/out-tip/<slug>/`; the leg dirs in § 1; `src/t2pw/bench/gold/pinned_v1.json`;
`ORCH-761-RESULT.md`; `PWML_KNOWN_ISSUES.md` (KI-001, -004, -006, -007, -012, -014, -032, -034, -035); `DECISIONS.md` (D-070,
D-071, D-074, D-106, D-108, D-112, D-113, D-114 and AMENDMENTS 1-2); `PRODUCT_CONTRACT.md` §§ 1-3, 13, 15.

# ORCH-761 — What PathWhiz requires of a protein, and why T2PW did not supply it

**Read-only investigation, 2026-10-22** (`pwml-bio-auditor`). No product code, no PathWhiz code and no database
row was changed. T2PW `sprint/pwml-recovery` @ `b06b86b5` (`src/` byte-identical at `d1da8572`, checked with
`git diff --stat b06b86b5 HEAD -- src/`). PathWhiz source `SummerBIOIN/smpdb` @ `6e11df0e` plus one uncommitted
hunk in `lib/pwml_parser.rb` (the polymorphic `element-id` remap, lines 572-581, not on the protein path); every
citation below is to that working tree. Governing ruling: **D-106** (the identity requirement is not relaxed).

All database work used T2PW's own `PathBankDbResolver.from_env()._query`, SELECT only (each probe asserts it),
through the bounded wrapper, every cleanup report `FINAL SURVIVING COUNT : 0`. No probe output contains the DB
host, user or password (checked by a script that prints only True/False). UniProt REST calls were plain `urllib`.

**The leg directories are uncommitted run output.** Every leg record cited here is reproduced, with the sha256 of
the file it came from, in `evidence/orch761_leg_identity_extract.txt` (script `orch761_leg_identity_extract.py`).

## 0. Summary

1. **PathWhiz requires a UniProt id or a DrugBank id on every protein row it creates** (`protein.rb:57,59`),
   plus a name and a species (`:46-47`). Nothing else counts as identity. A complex carries no identity field; it
   needs at least one component whose protein resolved (`protein_complex.rb:47`). **But PathWhiz never refuses a
   pathway over it:** a protein that fails to save is skipped with an error the upload response discards
   (`pathways_controller.rb:236-240`), and the loss cascades — the complex loses its component, the reaction
   loses its enzyme and, because T2PW writes `<spontaneous>false</spontaneous>`, the reaction itself fails
   (`reaction.rb:90-96`). T2PW's strict gate is the only place this loss is visible. It is not stricter than
   PathWhiz on the rule; it is stricter only in refusing to hide the consequence.
2. **PathBank: 0 of 161,094 protein rows lack both ids**; 156,185 carry a real UniProt accession, 2,730 a
   DrugBank id, 2,195 the literal `Unknown`. There is **no Men enzyme in any *Lactococcus* species** in PathBank.
   The *E. coli* Ent enzymes are all there (species 3, K-12 accessions).
3. **UniProtKB holds no entry for any Men enzyme of *L. lactis* KF147** — taxon 684738 has 2 UniProtKB entries in
   total. KF147's Men sequences exist only in UniParc (`UPI…`), which PathWhiz's UniProt format check rejects. The
   same-subspecies reference strain IL1403 has UniProtKB records for MenF, MenD, MenH (`menX`), MenC (`yhdB`),
   MenE, MenB and MenG, and **exact-sequence** records for MenB (Q9CHK2), MenI (Q9CHK5, `yhdA`) and one of three
   KF147 MenA paralogs (Q9CF18, `yrbC`).
4. **T2PW missed every one of them because of the query, not the gate.** It searches UniProt with
   `organism_name:"<the paper's free-text strain>"` (0 hits), falls back to an organism-free query that returns
   *E. coli*, *B. subtilis* and human records (correctly refused, `species_mismatch`), and its PathBank tier stops
   at `species_not_found` because PathBank spells the strain "(strain KF147)". Taxonomy ids never enter a query.
   On the night the organism was read as "Lactococcus lactis", the free-text scope admitted two subspecies and the
   margin rule refused the MG1363/IL1403 tie.
5. **Enterobactin has no lookup failure.** 2026-10-20: identity admission withheld EntB's verified P0ADI4 because
   an inferred compound `holo-EntB` took the same accession through the literature alias "EntB". 3-draw: every
   protein has a UniProt; the failure is a degree-0 synthetic row (ORCH-759 E1, confirmed). The invalid
   `generation_reason` is informational and was not made by the ORCH-754b interaction-endpoint path.

## 1. PathWhiz's import contract, from source

### 1.1 How the importer treats any element

| step | what happens | file:line |
|---|---|---|
| ids | PWML `<id>`s are document-internal; they are hash keys, never DB lookups | `lib/pwml_parser.rb:5-7`, `:645-646` |
| attribute hash | every **non-empty leaf** child, sliced to the model's columns; empty elements and elements with children (`<ec-numbers>`) are omitted | `:425-434`, `:470-471`, `:615-617` |
| flat entities (species, compound, protein, nucleic acid, element collection) | match by unique external id (Compound `hmdb_id`, Species `taxonomy_id` only), else `klass.find_by(attribute_hash)`; reuse the row if found, else `new(...).save` | `:629-637`, `:669`, `:677-687` |
| nested entities (complex, bound, reaction, transport, interaction) | build children with `get_child_object` (an invalid child is dropped and an error recorded), then `klass.find_existing(object)`; reuse if found, else save | `:726-792`, `:920-955`, `:759-763` |
| a failed save | error recorded under `@result[:errors]`, element skipped, import continues | `:692-705`, `:775-777`, `:889-901` |
| an exception inside one element | rescued, element skipped | `:708-719`, `:779-790` |
| whole-pathway abort | only an exception reaching `parse`'s outer rescue, or the super-pathway visualization failing to save | `:383-397`, `:400-420` |
| what the uploader returns | `success` → `{pathway_id}` only; per-element errors are dropped | `app/controllers/pathways_controller.rb:225-241` |

**Ownership and defaults.** `IsOwned` validates `user` presence (`app/models/concerns/is_owned.rb:6-8`); T2PW
writes no `user-id`, and the column default `user_id = 11` satisfies it (`db/schema.rb:701,653,984`; live DB:
default 11, user 11 exists, `evidence/orch761_pathbank_probe4.txt`). **`species_id` defaults to 1 = *Homo
sapiens*** on proteins and complexes (`schema.rb:702,654`; DB confirms). So an *empty or absent* `<species-id>`
silently files the row under human; a *present but unmappable* one becomes `nil`, overrides the default and fails
presence (`pwml_parser.rb:479-482`).

### 1.2 Protein — `app/models/data/protein_complexes/protein.rb`

| requirement | rule | line |
|---|---|---|
| name | `presence` | `:46` |
| species | `presence` (see the default-1 trap above) | `:47` |
| UniProt | `presence` **unless `drugbank_id.present?`** | `:57` |
| DrugBank | `presence` **unless `uniprot_id.present?`**; format `\ADB\d{5}\z` | `:58-59` |
| UniProt format | `/\A([A-N,R-Z][0-9][A-Z][A-Z, 0-9][A-Z, 0-9][0-9])\|([O,P,Q][0-9][A-Z, 0-9][A-Z, 0-9][A-Z, 0-9][0-9])\z/`, `allow_nil`, **skipped when `name == "Unknown"`** | `:52-55` |
| HMDBP | unique, `\AHMDBP\d{5}\z`, `allow_nil` | `:60` |
| UniProt uniqueness | **none** — one accession may sit on several rows (comment) | `:49-50` |
| on create / UniProt change | `check_uniprot` fetches the record and **overwrites `name`, `gene_name`, `description`, `ec_numbers` and `species`** (species = `Species.find_by_taxonomy_id(<accession's taxon>)`, created if absent); any failure is swallowed | `:64`, `:171-203` |

Notes, all static reading:
- The format regex has an alternation-precedence bug: `\A` binds only to the first branch and `\z` only to the
  second, so any string that *starts* with an accession-shaped 6-character prefix passes (PathBank holds
  `A0A5V5QKG2_SALER`, `orch761_pathbank_probe.txt`). It is case-sensitive. A UniParc id (`UPI0001C051B0`) fails both branches.
- **There is no protein-specific external-id match.** `find_by_unique_external_id` covers Compound and Species
  only (`pwml_parser.rb:629-637`), so a protein is matched on *all* its present leaf fields. T2PW writes `name`,
  `uniprot-id`, `species-id` (`pwml/writer.py:1216-1260`). PathBank row 6224 is named "Isochorismatase", T2PW
  writes "EntB": no match, so PathWhiz **creates a new row**, and `check_uniprot` renames it "Enterobactin
  synthase component B" and files it under taxon 83333 (PathBank species 25 "Escherichia coli (strain K12)"), not
  under the PWML's species 3. PathWhiz records a protein under its **accession's** organism, whatever the PWML says.
- **A protein with no UniProt and no DrugBank:** the missing fields are simply absent from the hash, so it is
  matched on name + species (+ any other present field). If a row matches, it is reused with that row's identity;
  if not, `save` fails ("Uniprot can't be blank", "Drugbank can't be blank") and the protein is skipped. It is
  never created with a nil identity. PathBank has **no** row without one of the two (§ 3.1), so whatever row it
  binds to carries a UniProt, a DrugBank id or the literal `Unknown` — bound by name coincidence, not identity.
- **`Unknown`:** a row named exactly "Unknown" may carry `uniprot_id = "Unknown"` (presence met by the literal,
  format waived by name). PathWhiz itself creates such rows per species when mapping proteins across species
  (`protein.rb:222-226`). On any other name, `uniprot_id "Unknown"` fails the format check.
- The legacy URL `check_uniprot` uses (`lib/uniprot_blast.rb:61,91`) still answers: it redirects to
  `rest.uniprot.org` (`evidence/orch761_legacy_uniprot_probe.txt`; Python followed the redirect, Ruby's
  `open-uri` was not run).

### 1.3 Protein complex — `protein_complexes/protein_complex.rb`, `protein_complex_protein.rb`

| requirement | rule | line |
|---|---|---|
| name | `presence`, unique **per species** | `protein_complex.rb:45` |
| species | `presence` (default 1 trap applies) | `:46` |
| components | `protein_complex_proteins` length ≥ 1 | `:47` |
| element set | `unique_element_set` (refuses a duplicate of an existing complex), `no_duplicates`, `elements_changed` | `:50-52`, `:391-418` |
| component row | `protein_id` presence, unique per complex; `stoichiometry` integer, `allow_nil` | `protein_complex_protein.rb:10-12` |
| identity | **no identity field on the complex**; identity is transitive through each component's Protein | — |
| `pwp_id` | set only when blank | `protein_complex.rb:42`, `:383-389` |
| matching | `find_existing`: same species + name + signature, **else same species + name** — applied *before* save | `:313-334`; `pwml_parser.rb:759-763` |

Consequences: a PWML complex named like an existing complex of the same species binds to that row and **its**
components; the PWML's components are ignored (T2PW's "Isochorismate synthase" binds to PathBank 1143). A complex
whose every component protein failed has no child rows (`pwml_parser.rb:936-941`) and fails `:47` unless it
matched an existing complex by name first. No rule ties a component's species to the complex's species.

### 1.4 The other element types

| element | requirements | identity / matching |
|---|---|---|
| Compound `compound.rb:38-44` | name present + unique; `short_name` ≤ 7; `hmdb_id` unique, `\AHMDB\d{7}\z`; `pwc_id` unique, `\APW_C\d{6}\z` | `hmdb_id` first, then all fields (`pwml_parser.rb:629-631,669`); a new row whose name collides with an existing one fails uniqueness and is skipped (comment `:619-628`) |
| NucleicAcid `nucleic_acid.rb:31-35` | name present + unique; `nucleotide_type` ∈ RNA, DNA | all fields |
| ElementCollection `element_collection.rb:27-36` | name present + unique; `collection_type` ∈ Compound, ProteinComplex, NucleicAcid; `external_id_type` if `external_id` | all fields |
| Bound `bounds/bound.rb:11-18,301-313`; `bound_element.rb:15-18` | ≥ 2 elements, ≥ 1 macromolecule (ProteinComplex / NucleicAcid, or a collection of one), unique set; element present, stoichiometry > 0 | `find_existing` |
| Reaction `reactions/reaction.rb:24-37,90-96,343-351` | ≥ 1 left, ≥ 1 right; direction `<`/`>`/`<>`; unique signature; **`check_spontaneous`: 0 enzymes with `spontaneous == false` fails** | by signature |
| ReactionEnzyme `reaction_enzyme.rb:13-14` | `protein_complex` present, unique per reaction | — |
| ReactionElement `reaction_element.rb:10-13` | element present, stoichiometry > 0 | — |
| Interaction `interactions/interaction.rb:23-35`; `interaction_element.rb:10-13` | exactly 1 left, 1 right; `interaction_type` ∈ Activation, Inhibition; element present (polymorphic, no type whitelist; PathWhiz's own index treats protein endpoints as ProteinComplex, `interaction.rb:50`) | by signature |
| Transport `transports/transport.rb:26-33`; `transport_element.rb:12-19`; `transport_transporter.rb:14-16` | 1-3 elements, two states, direction consistent; transporter complex + state present | by signature |
| Species `biological_states/species.rb:13-15` | name present, case-insensitively unique; `taxonomy_id` present, unique, integer; `classification` ∈ Eukaryote, Prokaryote | **by `taxonomy_id` first** (`pwml_parser.rb:632-633`) |
| BiologicalState `biological_state.rb:29-36,219-223` | at least one of species / tissue / cell type / location; combination unique | all fields |
| SubcellularLocation / Tissue / CellType | name + ontology id unique, `GO:` / `BTO:` / `CL:` + 7 digits | all fields |
| SubPathway `sub_pathways/sub_pathway.rb:16-24` | pathway present; type ∈ SubPathway, Inhibitory…, Activating…; `reference_pathway_id` unless `altext` | never matched (`pwml_parser.rb:659-662,755-757`) |
| Pathway `pathways/pathway.rb:20-24` | name present, unique per species (the importer appends a timestamp on collision, `pwml_parser.rb:156-158`); `subject` whitelist; species present | — |
| Visualizations | ProteinLocation: protein, template, x, y, zindex (`visualization/protein_location.rb:13-18`); ProteinComplexVisualization: complex + pathway visualization (`protein_complex_visualization.rb:26-27`); SPV: `smpdb_id` `SMP\d{7}` unique, `pw_id` `\APW\d{6}\z` (`super_pathway_visualization.rb:34-37`) | — |

### 1.5 The direct answers

- **Is a UniProt id required?** For any protein row PathWhiz creates: UniProt *or* DrugBank (`protein.rb:57,59`).
- **Is DrugBank an alternative?** Yes, in the `DB\d{5}` format (`:58`). PathBank uses it on 2,730 rows, 9 of them
  prokaryotic. **Nothing else is accepted**: `hmdbp_id`, `gene_name`, `genbank_id`, `locus` are not alternatives.
- **Does the importer create proteins?** Yes, whenever the full-attribute `find_by` misses (`pwml_parser.rb:685-687`).
- **Species:** required on proteins and complexes; empty means human by default; on create, UniProt's organism
  for the accession replaces the PWML's.
- **One failing element** never aborts the pathway; it is skipped, its dependants fail in turn, and the upload
  reports success.

## 2. T2PW's strict gates against PathWhiz's rules

D-106 holds: nothing below recommends relaxing a gate. "Stricter in consequence" means T2PW refuses where
PathWhiz would skip the element silently and report success.

| T2PW gate (file:line) | PathWhiz rule it enforces (file:line) | verdict |
|---|---|---|
| `protein_missing_name` (`pwml/ir.py:2771-2777`) | Protein name presence (`protein.rb:46`) | equal |
| `protein_missing_species` (`ir.py:2778-2785`; `strict_quarantine.py:709-710`; predicate `entity_identity.py:297-314`) | species presence (`protein.rb:47`) | equal on the rule; **stricter than PathWhiz's effective behaviour**, which defaults an empty species to human (`schema.rb:702`) and replaces it with the accession's organism on create (`protein.rb:180-190`) |
| `protein_missing_external_identity` (`ir.py:2786-2793`; `strict_quarantine.py:711-715`; predicate `entity_identity.py:62-84`) | UniProt unless DrugBank, DrugBank unless UniProt (`protein.rb:57,59`) | equal on presence. **Looser on format**: T2PW checks no UniProt/DrugBank format (PathWhiz: `:52-55,58`, itself loose). **Looser on serialisation**: the predicate also accepts an id found only in `ids` or `mapping_meta`, containers the writer does not serialise (`writer.py:1243-1256` reads `mapped_ids.uniprot`, `uniprot`, `uniprot_id`) — such a row passes the gate and imports with no `uniprot-id` |
| Unknown-sentinel admission (`entity_identity.py:87-116`; `strict_quarantine.py:711-712`) | `Unknown` name waives format, literal satisfies presence (`protein.rb:55,57`) | stricter (T2PW also requires PathBank id 9659 and the fallback rule) |
| `protein_complex_missing_species` (`ir.py:2800-2807`; `strict_quarantine.py:718-719`) | complex species presence (`protein_complex.rb:46`) | equal (and guards the default-1 trap, `schema.rb:654`) |
| `protein_complex_missing_components` — error for a generated wrapper, warning for a declared complex (`ir.py:2808-2816`) | ≥ 1 component (`protein_complex.rb:47`), unless matched to an existing complex by species + name first (`pwml_parser.rb:759-763`) | equal for generated wrappers; **looser for a novel declared complex** (PathWhiz rejects it); equal for a declared complex that matches a PathBank complex by name |
| `component_protein_unresolved` (`ir.py:2836-2880`) | component `protein_id` presence (`protein_complex_protein.rb:11`); an invalid child is dropped (`pwml_parser.rb:936-941`) | equal; error vs warning split mirrors whether PathWhiz would be left with an empty complex |
| `generated_complex_component_missing_species` (`ir.py:2881-2889`) | transitive: component Protein species (`protein.rb:47`) | equal |
| `generated_complex_component_missing_external_identity` (`ir.py:2890-2898`) | transitive: component Protein identity (`protein.rb:57,59`) | equal for generated wrappers; not checked for declared complexes, where PathWhiz drops the component and may keep the complex — equal in outcome |
| `reaction_enzyme_must_be_protein_complex` (`ir.py:1906`) | ReactionEnzyme belongs to a complex (`reaction_enzyme.rb:13`) | equal |
| strict quarantine removes the processes of an unexportable protein (`strict_quarantine.py:706-719`) | protein fails → complex loses component → reaction with `spontaneous=false` and no enzyme fails (`reaction.rb:93-94`) → upload still succeeds (`pathways_controller.rb:236-240`) | **stricter in consequence**, same rule |
| `species_missing_taxonomy` / `species_missing_classification` for a species with no PathBank id (`ir.py:2906-2937`) | `taxonomy_id` present, unique, integer; classification whitelist (`species.rb:13-15`) | equal |
| `gate.protein_has_degree_after_normalization` (`pipeline/process_normalizer.py:5099-5125`) | none: a protein with no edges imports; ProteinLocation needs only protein, template and coordinates (`protein_location.rb:13-18`) | **no PathWhiz basis** |
| `generated_wrapper_reason_invalid` / `generation_reason: Literal[...]` (`pipeline/payload_models.py:242,727-735`; `schema.py:270`) | not a PathWhiz column (`schema.rb:649-657`); unknown tags are sliced away (`pwml_parser.rb:615-617`) | **no PathWhiz basis** (informational in these legs) |
| identity admission, cross-kind accession conflict (`mapping/map_ids.py:9940-9995`; `mapping/identity_admission.py:806-855`) | PathWhiz Compound has no UniProt column; Protein UniProt is not unique (`protein.rb:49-50`; PathBank: 5,533 accessions on more than one row) | **no PathWhiz basis** (a T2PW integrity rule, D-064) |
| identity verdict: species, name plausibility, score, margin (`map_ids.py:5541-5550`, `:5613-5670`) | none: PathWhiz trusts the accession and overwrites name and species from UniProt (`protein.rb:171-203`) | **no PathWhiz basis** (a T2PW fidelity rule, D-003) |

## 3. The databases

### 3.1 PathBank (read-only; `evidence/orch761_pathbank_probe{,2,3,4,5}.txt`, G11 reports `orch761_g11_pathbank_probe*.json`)

**Storage.** `proteins.uniprot_id` and `proteins.drugbank_id` are free `varchar` columns beside `species_id`
(`orch761_pathbank_probe.txt` § "proteins columns"). No constraint enforces either; the model does.

| population | rows | real UniProt | literal `Unknown` | DrugBank | **neither** |
|---|---|---|---|---|---|
| all proteins | 161,094 | 156,185 | 2,195 | 2,730 | **0** |
| proteins drawn in a pathway (`protein_locations`) | 110,982 | 107,352 | 928 | 2,713 | **0** |
| prokaryote species | 137,359 | 135,222 | 2,132 | 9 | **0** |

Non-accession strings besides `Unknown` are a handful of curation slips (`:O15111`, `A0A5V5QKG2_SALER`, trailing
spaces, a leading tab). 5,533 accessions are carried by more than one row (64,648 rows).

**Species.** *E. coli*: id 3 (taxon 562), 25 (83333, K-12), 255 (511145, MG1655) and ~30 strains.
*Lactococcus*: 1030 *L. lactis* (1358), 1204 IL1403 (272623), 1205 SK11, 1206 MG1363 (416870), 10688 subsp.
*lactis* (1360), 10727 subsp. *cremoris* (1359), **10733 "Lactococcus lactis subsp. lactis (strain KF147)"
(684738)**, 10734, 10779. T2PW's species SQL (`map_ids.py:1004-1015`) on the paper's string "Lactococcus lactis
subsp. lactis KF147" returns **no row**; on taxon `684738` it returns 10733 (`orch761_pathbank_probe.txt`).

**Men and Ent enzymes.**
- *E. coli* species 3 holds every Ent enzyme: EntA 6341 P15047, EntB 6224 P0ADI4, EntC 6238 P0AEJ2, EntD 6383
  P19925, EntE 6301 P10378, EntF 6312 P11454 (+ EntS 7382, FepA 7426, Fes 6331, TolC 7446). PathBank complex 3623
  "enterobactin synthase" (species 3) = EntB×2, EntD, EntE×2, EntF (`orch761_pathbank_probe.txt`).
- Men gene rows exist for many bacteria (menA 57, menB 65, menC 22, menD 38, menE 9, menF 19, menG 42, menH 23,
  menI 14, ubiE 46; `orch761_pathbank_probe2.txt`), **none in any *Lactococcus* species**: all 28 *Lactococcus* protein rows are
  nisin-operon, RNA-polymerase, Sec and `Unknown` rows (`orch761_pathbank_probe2.txt`). No *Lactococcus* menaquinone or
  enterobactin pathway exists; *Lactococcus* has two pathways in all (PW499224 nisin, PW766173 fermentation).

**How bacterial pathways are recorded** (PW accession via `super_pathway_visualizations.pw_id`, `orch761_pathbank_probe2.txt`,
`orch761_pathbank_probe3.txt`):
- Enterobactin Biosynthesis **PW002048** (pathway 1817, *E. coli* 562): EntA, B, C, E, F — the K-12 Swiss-Prot
  accessions on species-3 rows.
- Menaquinone biosynthesis **PW438177** (pathway 438675, species 255 MG1655): its proteins sit on species 3 (562)
  and species 25 (83333) rows, e.g. MenH P37355 and MenI P77781 on K-12 rows. Protein species ≠ pathway species.
- Menaquinone biosynthesis **PW729140** (*B. subtilis* 1423): unreviewed TrEMBL accessions on species 92 and
  10743 (subsp. *subtilis*), several with ordered-locus gene names (`B4067_2491`, `B4067_3616`).

So PathBank's curators record a bacterial protein with the UniProtKB record of the same gene from a related
strain or subspecies when the exact strain has none. That is a fact about PathBank, not a ruling for T2PW.

### 3.2 UniProt (REST, 2026-10-22; `evidence/orch761_uniprot_probe*.txt`, `orch761_uniparc*_probe.txt`, `orch761_uniprot_entries.txt`)

**Taxonomy.** 684738 KF147: **1 reviewed + 1 unreviewed UniProtKB entry in total** (DivIB D2BKM6, Axe
A0ABF7SX93); non-reference proteome UP000001886, 2,458 proteins, held in UniParc. 272623 IL1403 and 416870 MG1363
are reference proteomes. 1359 "Lactococcus lactis subsp. cremoris" now has rank *species* under *Lactococcus*
(MG1363's parent is "Lactococcus cremoris subsp. cremoris"), so `taxonomy_id:1358` no longer contains it.

| enzyme (gold) | KF147 sequence (UniParc) | identical UniProtKB record | IL1403 (subsp. *lactis*) record | MG1363 (*cremoris*) | `gene:<sym> AND taxonomy_id:1358` |
|---|---|---|---|---|---|
| MenF | UPI0001C051B0, 404 aa | none | Q9CHJ9 TrEMBL, `menF`, EC 5.4.4.2 | A2RM72 | 1 hit |
| MenD | UPI0001C051AF, 560 aa | none | **Q9CHK0 Swiss-Prot**, `menD` | A2RM73* | 1 hit |
| MenH | UPI0001C051AE (`menX`/`menH_2`), 271 aa | none | Q9CHK1 TrEMBL, `menX` syn `menH`, EC 4.2.99.20 | A2RM74 | **2 hits**: Q9CHK1 and P49016 (`menG` syn `menH`, the methyltransferase) |
| MenC | UPI0001C05177, 367 aa | none | Q9CHK4 TrEMBL, `yhdB` syn `menC` | A2RM77 | 1 hit (via synonym) |
| MenE | UPI0001C05178, 451 aa | none | **Q9CHK3 Swiss-Prot**, `menE` | A2RM76 | 1 hit |
| MenB | UPI00000C68C6, 280 aa | **Q9CHK2** | Q9CHK2 TrEMBL, `menB` | A2RM75 | 1 hit |
| MenI | UPI00000C68C3 (`yhdA`), 138 aa, "1,4-dihydroxy-2-naphthoyl-CoA thioesterase" | **Q9CHK5** | Q9CHK5 TrEMBL, `yhdA`, "Thioesterase domain-containing protein" | not found | **0** |
| MenA | three paralogs: `menA_3`/`yrbC` UPI00000C6ABB; `menA_1`/`ybiG`; `menA_2`/`yjhB` | **Q9CF18** for `menA_3` only | Q9CF18 TrEMBL, `yrbC`, "1,4-dihydroxy-2-naphthoate polyprenyltransferase" | A2RHR6 | **0** |
| MenG | UPI0001C055C9 (`menG`/`ubiE`), 252 aa | none | **P49016 Swiss-Prot**, `menG` syn `menH` | A2RJA5 | 1 hit |

Reviewed and unreviewed entries exist; gene names are searchable for MenF, D, C (synonym), E, B, G and H (with a
collision); MenI and MenA are not findable by `men*` symbol in *L. lactis*. *E. coli*: EntA-F are reviewed K-12
(83333) entries P15047, P0ADI4, P0AEJ2, P19925, P10378, P11454, gene-searchable, and first in T2PW's own query
shape (`organism_name:"Escherichia coli"`, 9-15 hits each).

## 4. Why each identity was not found or was lost

### 4.1 The lookup path, traced

1. The protein loop takes the **row's free-text organism** — `protein.organism`, then `species`, then
   `species_ref.name` — and passes only that string on (`map_ids.py:10233-10255`). The species row's taxonomy id
   (NCBI backfill found 684738 for this leg) is never passed. Species canonicalisation, which renamed the species
   row to "Lactococcus lactis", runs later and on species rows only (`ir.py:1129`, called from
   `pwml/prefreeze_resolution.py:1394`).
2. PathBank tier: `_map_protein_with_strategy` → `PathBankDbResolver.map_protein_row` (`map_ids.py:6627-6656`,
   `:2087`). With no PathBank id or UniProt on the row, it calls `_find_species_ids(organism)` (`:990-1049`), which
   matches name, common name or `taxonomy_id` **against the organism string**. No row is named "…subsp. lactis
   KF147", so it returns `species_not_found` (`:2119-2125`) and **skips the gene, exact-name and alias rungs**.
3. UniProt tier: `map_protein_uniprot` (`:4527-4830`). The plan is `(protein_name:"X" OR gene:"X") AND
   organism_name:"<organism>"` for each name variant, then the same without the organism (`:4558-4598`). Free text
   only; no `organism_id`/`taxonomy_id` query clause exists anywhere in `src/t2pw`. Candidates from the organism-free
   query lose 0.04 (`:4636-4638`). The EuropePMC and LLM alias tiers run only when nothing was found (`:4654-4784`).
   If no candidate clears the acceptance thresholds (`:4435-4478`), the top one is still returned as
   `best_effort_fallback` (`:4806-4822`).
4. Name-plausibility / identity verdict (`:5541-5670`): species must agree — exact taxon if both sides carry one,
   else the same binomial or a word-boundary strain qualification (`:5655-5658`); score ≥ 0.5, margin ≥ 0.1.
5. Results are cached under keys that embed the organism string (`:6595-6612`, `api-v8::<name>::<organism>…`).

### 4.2 Menaquinone (PMC12421875), per protein

ORCH-756 1-draw leg, organism string "Lactococcus lactis subsp. lactis KF147" (`orch761_leg_identity_extract.txt`
§ A; leg cache `.claude/worktrees/orch756/data/id_mapping_cache.json`, sha256 `de78cd56…`):

| enzyme | what UniProtKB / PathBank have (§ 3) | what T2PW queried | what came back and why it missed |
|---|---|---|---|
| MenF | IL1403 Q9CHJ9; PathBank none | `(protein_name:"MenF" OR gene:"MenF") AND organism_name:"Lactococcus lactis subsp. lactis KF147"`, then without organism | 0 hits with the strain string (live re-run of the exact query: 0). Organism-free top hit P38051 *E. coli* K-12, 0.56 → `species_mismatch`. PathBank tier: `species_not_found` |
| MenD | IL1403 Q9CHK0 (Swiss-Prot) | same shape | 0, then P17109 *E. coli* → `species_mismatch`. **The one Men protein that is also an interaction endpoint** (`DHNA inhibits MenD`, twice), so the Unknown-backed fallback skips it (`_has_non_enzyme_reference`, `map_ids.py:8363-8364`, `:8711-8714`); it stays bare (`bare_unresolved_proteins_remaining: 1`) and the strict quarantine removes its reaction and both interactions (`protein_missing_external_identity`), then the protein (`degree_zero_after_quarantine`). Release: `review_required`, `missing_anchors ["MenD"]` |
| MenH | IL1403 Q9CHK1 (`menX` syn `menH`) | same | 0, then P37355 *E. coli* → mismatch |
| MenC | IL1403 Q9CHK4 (`yhdB` syn `menC`) | same | 0, then A0A0P0ZBS7 *Geobacillus* → mismatch |
| MenE | IL1403 Q9CHK3 (Swiss-Prot) | same | 0, then P23971 *B. subtilis* → mismatch |
| MenB | IL1403 Q9CHK2, identical to KF147's sequence | same | 0, then P0ABU0 *E. coli* → mismatch |
| MenI | IL1403 Q9CHK5 (`yhdA`), identical sequence | same | 0, then P77781 *E. coli* → mismatch |
| MenA | IL1403 Q9CF18 (`yrbC`), one of three KF147 paralogs | same | 0, then Q8N8S7 *Homo sapiens* → mismatch |
| MenG | IL1403 P49016 (Swiss-Prot) | same | 0, then Q9VM90 *Drosophila* → mismatch |

The species-mismatch refusals are correct: every organism-free candidate is from the wrong organism. The miss is
upstream — the only query that could have found an *L. lactis* record used a strain string UniProt does not use
("(strain KF147)") and a strain UniProtKB holds 2 entries for.

ORCH-755 night, organism string "Lactococcus lactis" (`tmp/post_pipeline_3ded14a7…/gate_fail_report.json`,
extract § B): the free-text scope `organism_name:"Lactococcus lactis"` returns **both subspecies**, scored
identically, so MenF (A2RM72 MG1363 = Q9CHJ9 IL1403, 0.8/0.8), MenD (A2RM73/Q9CHK0), MenC, MenB failed
`ambiguous_insufficient_margin` with margin 0; MenE and MenG 0.05; MenH's top candidate was **P49016, the MenG
methyltransferase**, through its `menH` synonym — refused, correctly. MenI fell to *E. coli* P77781. **MenA was
"verified" as A2RHR6, the MG1363 (*cremoris*) protein**, because the strain-qualification string rule accepts
"Lactococcus lactis subsp. cremoris (strain MG1363)" for "Lactococcus lactis" (`map_ids.py:5655-5658`), although
current taxonomy places it outside *L. lactis* 1358 and KF147 is subsp. *lactis*.

### 4.3 Enterobactin (PMC12096016)

No Ent enzyme failed lookup in any leg: all six resolve from PathBank species 3 by `pathbank_protein_id` (extract
§ D; ORCH-759 `final_mapped.json` proteins 0-6).

**2026-10-20 night, EntB (ORCH-758 D1 confirmed)** — extract § C, `runs_papers/ORCH-755/…/strict/`:
1. Stage 2 wrote a protein-location row for `holo-EntB`; no such entity existed. The normaliser filed the dangling
   token as a **compound** (`process_normalizer.py:1162-1202` → `_ensure_compound` `:933`), confidence 0.8,
   provenance `inferred`.
2. The mapper's protein-like name set includes every `protein_locations` name (`map_ids.py:5145-5188`, `:5163-5165`),
   so the router calls the compound `known_protein_like` (`entity_identity.py:416-417`) and maps the
   compound-bucket row through the protein route (`map_ids.py:10802-10830`).
3. UniProt found nothing for "holo-EntB"; the EuropePMC literature tier (`:4654-4717`) supplied the alias **"EntB"**
   and returned P0ADI4 0.85 (tied with O157:H7 P0ADI5 0.85), shipped as `best_effort_fallback`.
4. Identity admission pass B saw P0ADI4 on a compound and on the protein EntB under different names and withheld it
   from **both** (`map_ids.py:9940-9995`, `identity_admission.py:806-855`). EntB's own verdict was
   `verified_real_protein` from PathBank 6224; its accession moved to `rejected_mapped_ids`.
5. EntB, a component of PathBank complex 1189, became unexportable; quarantine `ok: false`,
   `refusal_reasons ["unexportable_entity:1"]`, release `diagnostic_only`.

The identity EntB lost was never refuted. The penalty fell on the grounded claimant because one inferred name was
given two kinds.

## 5. The 3-draw enterobactin failure (`runs_papers/ORCH-759/legs/enterobactin-strict/2026-10-22_1037/…/strict/`)

This confirms ORCH-759 finding E1 from the artifacts (extract § D). **It is not an identity failure**: all 8
protein rows carry a UniProt and a PathBank id.

1. `merged_payload.json` names the enzyme `{"protein_complex": "enterobactin synthase complex"}` on reactions 4
   and 5 with no entity row; `stage1_payload.json` has no such string.
2. Normalisation added it as a protein, and `relocate_complex_named_proteins` (`process_normalizer.py:5140-5241`)
   built a generated complex `enterobactin synthase complex` with `generation_reason:
   "complex_named_source_entity_wrapper"` (`:5205`) and renamed the protein to the stripped stem **`enterobactin
   synthase`**, a row no paper text names as a protein.
3. The initial gate failed on exactly that synthetic row: missing species and missing UniProt on protein 7 and on
   the wrapper's component (`initial_stage3_gate_report.json`, 5 errors).
4. The audit accepted 2 patches whose rationale ties "enterobactin synthase" to EntF P11454
   (`audit_iteration_summary.json`); the complex row's lineage records `audit_repair / audit_modified`.
5. The Stage-6 remap's `_reconcile_components_against_local_proteins` (`map_ids.py:6985-7078`) indexes local
   proteins by UniProt with `setdefault`, first row wins (`~:7027`): EntF (protein 4) wins over `enterobactin
   synthase` (protein 7), so the wrapper's component is renamed **EntF**.
6. Protein 7 is now referenced by nothing. The final pre-export gate exempts complex components by name only
   (`process_normalizer.py:5092-5098`), so "enterobactin synthase" is not exempt: `Protein has degree 0 after
   normalization: enterobactin synthase` (`:5099-5125`; `final_stage3_gate_report.json`, `ok: false`).

**The `generation_reason` findings** (`post_enrichment_runtime_schema_report`, `generated_wrapper_reason_invalid`
at `/entities/protein_complexes/1/generation_reason`) are the runtime schema allowing only
`single_protein_pathwhiz_wrapper` (`payload_models.py:242`, `:727-735`; `schema.py:270`) while the normaliser emits
`complex_named_source_entity_wrapper`. They are informational (RESULT.txt: "not blocking") and not the cause.

**Was the wrapper made by ORCH-754b?** No. `complex_named_source_entity_wrapper` is emitted only at
`process_normalizer.py:5205`. The ORCH-754b interaction-endpoint path (`map_ids.py:7976-8088`) goes through
`_resolve_complex_name` (`:7618-7724`), whose generated rows say `single_protein_pathwhiz_wrapper` (`:7652`); the
three rows in this payload that carry that reason (EntD, EntE and EntF complex) pass the schema.

**PathWhiz's view of this payload** (static): two protein rows with P11454 would both import — UniProt is not
unique (`protein.rb:49-50`) — and the degree-0 row would simply be an undrawn protein. The degree gate has no
PathWhiz basis (§ 2); ORCH-759 already classified the block against PRODUCT_CONTRACT § 1. This settles one item
ORCH-759 listed as not established: "whether a PWML with two rows sharing P11454 imports" — by the model
validations, yes.

PathBank holds complex **3623 "enterobactin synthase"** in species 3 (EntB×2, EntD, EntE×2, EntF; `orch761_pathbank_probe.txt`, `orch761_pathbank_probe4.txt`).
The mapper saw it as the only candidate for "enterobactin synthase complex" at 0.7567 and left the row
`ambiguous` (`ambiguous_complex_name_species`).

## 6. Adjudications

### 761-A — the *L. lactis* Men lookup never queries the records that exist
- **CLASS:** `product_contract_violation` (PRODUCT_CONTRACT § 1: "an entity groundable through an available
  database" may not end a run without the entity; § 8 runs the species/name/score/margin checks unchanged — the
  defect is the candidate set, not the checks). Which *rank* of record may stand for a KF147 protein is a separate
  product decision, **Q1** in § 8; this finding stands under either answer, because the query also fails for the
  species-level organism (ORCH-755 night).
- **GOLD CITATION:** `src/t2pw/bench/gold/pinned_v1.json`, PMC12421875 — `relevance_note`: "Enumerates the complete
  classical menaquinone pathway and pins every Men enzyme to a substrate-product pair … ORGANISM TRAP: this is a
  Lactococcus lactis study"; `export_rationale`: "Every exported reaction must carry organism Lactococcus lactis …
  Reconciled with D-062 … preserves the pathway for review under the OBSERVED organism and is never a strict
  export"; `unknown_backed_rationale`: "All nine catalysts are named canonical Men proteins … a correct extraction
  resolves by name plus organism rather than inventing accessions"; `actual_organism: "Lactococcus lactis"`;
  `forbidden_identifiers` lists `KF147` as a strain designation. The correct terminal outcome is a
  `review_required` PWML (`expected_export: "partial_only"`), not strict success.
- **ARTIFACT EVIDENCE:** extract § A (`initial_stage3_gate_report.json` `/entities/proteins/0-8/mapping_meta`
  `queries_tried`, `identity_verdict`; leg cache `db::<men*>::lactococcus lactis subsp lactis kf147` =
  `species_not_found`); § B (ORCH-755 margin ties); live: `orch761_uniprot_probe.txt` (exact T2PW query → 0),
  `orch761_pathbank_probe2.txt` (no *Lactococcus* Men rows), `orch761_pathbank_probe.txt` (species SQL on the strain
  string → 0 rows).
- **AFFECTED FILES:** `src/t2pw/mapping/map_ids.py` — `map_protein_uniprot` query plan (`:4558-4598`),
  `_map_protein_with_strategy` (`:6562-6700`, incl. cache keys `:6595-6612`), the protein loop's organism choice
  (`:10233-10255`), `PathBankDbResolver._find_species_ids` (`:990-1049`) and its short-circuit (`:2119-2125`).
- **EXPECTED CORRECTION:** the UniProt query is scoped by taxonomy (strain `organism_id`, then subspecies/species
  `taxonomy_id` lineage) instead of free text; the PathBank species lookup accepts the payload's taxonomy id; the
  verdict is unchanged. With Q1 answered "species/subspecies record", MenF Q9CHJ9, MenD Q9CHK0, MenC Q9CHK4, MenE
  Q9CHK3, MenB Q9CHK2, MenG P49016 resolve; MenH needs 761-A2 (FX4); MenI and MenA need FX5 or stay unresolved.
  With Q1 answered "strain-exact only", every Men enzyme stays unresolved — correctly, with a reason that says the
  strain has no UniProtKB entry rather than "species mismatch against E. coli".
- **REGRESSION FIXTURE:** payload `entities.proteins: [{name: "MenF", organism: "Lactococcus lactis subsp. lactis
  KF147"}]`, `entities.species: [{name: "Lactococcus lactis subsp. lactis KF147", taxonomy_id: "684738"}]`; a stub
  HTTP client that answers `organism_name:"…KF147"` with 0 results and `gene:menF AND taxonomy_id:1358` (or
  `:1360`) with the Q9CHJ9 record from `orch761_uniprot_probe.json`; a stub resolver whose species table holds 10733
  "(strain KF147)". Base: P38051 `species_mismatch`, PathBank `species_not_found`. Fixed: the taxonomy query is
  issued and the PathBank species resolves to 10733.
- **CONFIDENCE:** high on mechanism (cache records, live re-run, code). **Would change my mind:** a UniProt answer
  that differs by date (entries move), or a PO ruling that a strain-named paper's proteins must be identified only
  at strain rank (then the expected outcome is "unresolved, honestly reasoned", and only the query/PathBank parts
  remain a fix).

### 761-A2 — gene-symbol synonym collision (MenH)
- **CLASS:** `product_contract_violation` once 761-A is fixed (a real *L. lactis* SHCHC synthase, Q9CHK1, ties with
  the MenG methyltransferase P49016 through the shared synonym `menH`; the margin rule refuses — correctly — but the
  record is groundable). Not observable before 761-A is fixed except as ORCH-755's refusal.
- **GOLD CITATION:** PMC12421875 `expected_enzymes`: MenH, role "SEPHCHC -> SHCHC".
- **ARTIFACT EVIDENCE:** extract § B (`MenH` top candidate P49016, margin 0.05); `orch761_uniprot_entries.txt`.
- **AFFECTED FILES:** `map_ids.py` `_extract_uniprot_candidates` (`:4312-4412`; primary symbols already captured
  `:4352-4360`) and the identity verdict.
- **EXPECTED CORRECTION:** a synonym-only symbol hit loses to a candidate whose function (EC 4.2.99.20, or the
  reaction's substrate→product) matches; primary symbol alone is not enough here (Q9CHK1's primary symbol is `menX`).
- **REGRESSION FIXTURE:** candidates P49016 (`menG`, syn `menH`, EC 2.1.1.163) and Q9CHK1 (`menX`, syn `menH`,
  EC 4.2.99.20) for query "MenH" on reaction SEPHCHC → SHCHC; expected Q9CHK1.
- **CONFIDENCE:** medium (static). **Would change my mind:** evidence that T2PW's verdict already demotes synonym hits.

### 761-B — an inferred compound strips EntB's verified accession (ORCH-758 D1)
- **CLASS:** `product_contract_violation` (PRODUCT_CONTRACT § 8: `rejected` is "the only case where identifiers
  may be stripped" — EntB's verdict was `verified_real_protein`; § 1: an entity groundable through an available
  database).
- **GOLD CITATION:** `pinned_v1.json` PMC12096016 — `relevance_note`: "Primary research on E. coli enterobactin
  enzymes themselves …"; `export_rationale`: "A fully connected metabolite chain … with a named enzyme per step …
  Export must exclude MenD, LDH and the transport mentions"; `expected_export: "strict_exportable"`;
  `unknown_backed_rationale`: "Every protein described catalytically is a named, canonical E. coli gene product";
  EntB is expected enzyme R2. D-064 (LOCKED): "Holo-EntB and apo-EntB may share the underlying parent accession
  while remaining distinct pathway states."
- **ARTIFACT EVIDENCE:** extract § C — `final_mapped.json` `/entities/compounds/16/mapping_meta` (`route
  protein`, `known_protein_like`, `literature_alias "EntB"`, `best_effort_fallback`, admission withheld P0ADI4);
  `/entities/proteins/1/mapping_meta/rejected_mapped_ids`; `quarantine_report.json` `unexportable_entity:1`.
- **AFFECTED FILES:** `src/t2pw/pipeline/process_normalizer.py` `_rewrite_token`/`_ensure_compound` (`:1162-1202`,
  `:933`); `src/t2pw/mapping/map_ids.py` compound loop protein reroute (`:10802-10830`) and
  `_collect_protein_like_names` (`:5145-5188`).
- **EXPECTED CORRECTION:** one kind decision per name. A token the mapper will treat as protein-like is filed as a
  protein (a state of EntB, same-kind sharing per D-064, no conflict), or a compound-bucket row is never resolved
  through the protein route (a PathWhiz Compound has no UniProt column at all). EntB keeps P0ADI4; the 4-reaction
  core exports `review_required` or better.
- **REGRESSION FIXTURE:** `entities.proteins: [{name: "EntB", pathbank_protein_id: 6224, mapped_ids: {uniprot:
  "P0ADI4"}, species: "Escherichia coli"}]`, `element_locations.protein_locations: [{protein: "holo-EntB",
  biological_state: "cytoplasm"}]`, no `holo-EntB` entity; a stub UniProt client whose literature tier maps
  "holo-EntB" to alias "EntB" → P0ADI4. Base: EntB loses P0ADI4 to `accession_claimed_across_incompatible_entity_kinds`.
  Fixed: EntB ships P0ADI4.
- **CONFIDENCE:** high. **Would change my mind:** a product ruling that an inferred location token must stay a
  compound even when the mapper calls it protein-like (then the remaining lever is admission pass B, Q2).
- **Bio-auditor answer to ORCH-758 § 9 item 2.** Option (a) there is FX6 here and needs no policy change. Option
  (b) (an `inferred` compound claimant loses to a `verified` protein) is also biologically right — `holo-EntB` is a
  state of EntB, not a metabolite (D-064) — but it changes the admission rule and stays a product decision (Q2).

### 761-C — 3-draw degree-0 synthetic row (ORCH-759 E1)
- **CLASS:** `product_contract_violation` — concurring with ORCH-759 (PRODUCT_CONTRACT § 1: "an irrelevant
  degree-zero entity"). **Not an identity failure.** Its entity-row policy question is already routed to the
  product owner (ORCH-759 § 4 item 2); nothing here pre-empts it.
- **GOLD CITATION:** PMC12096016 as above; `src/t2pw/bench/eval_v2/gold_v2_supported_reactions.json` R5 enzyme
  `{"name": "EntF", "aliases": ["enterobactin synthase", "entF"]}` — the gold itself reads "enterobactin synthase"
  as EntF, which is what the audit wrote.
- **ARTIFACT EVIDENCE / FILES / FIXTURE:** § 5; ORCH-759-RESULT § 4 item 2 (fixture: rename the component back or
  drop the duplicate row → 0 errors).
- **CONFIDENCE:** high on the chain.

### 761-D — `generation_reason` values outside the runtime-schema Literal
Not a biological finding and no class: informational, no PathWhiz basis (§ 2), not the cause of any failure.

### 761-E — why MenD, of nine, was lost rather than placeholder-wrapped
No new class. It is downstream of 761-A: MenD alone is an interaction endpoint, so the Unknown-backed fallback is
skipped (`map_ids.py:8711-8714`) and D-106's refusal applies. The placeholder asymmetry sits inside the standing
`placeholder_backed_proteins` `policy_disagreement`; no fix is proposed for it. Fixing 761-A gives MenD a real
identity and removes the question for this paper.

## 7. Fixes that keep the identity requirement

None of these touches a gate. Each changes what the lookup finds or what may overwrite a found identity. The
recovery counts are **hypotheses until a leg runs**; they come from the live UniProt/PathBank answers in § 3.

| # | fix | T2PW file / function | recovers | needs a PO ruling? |
|---|---|---|---|---|
| FX1 | **Query UniProt by taxonomy, not free text.** Strain `organism_id:<taxid>` first; if the strain has no entries (KF147: 2), climb to subspecies then species `taxonomy_id:<taxid>` (lineage). Keep the verdict unchanged and keep `requested_taxonomy_id` unstamped (`uniprot_evidence.py:161-166` explains why). Bump the API cache key (`api-v8` → next), or the cached KF147 entries (E. coli candidates) replay | `map_ids.py` `map_protein_uniprot` `add_query_plan` (`:4558-4598`); caller `_map_protein_with_strategy` (`:6562`) and the protein loop (`:10233-10255`) to carry the taxid; cache keys `:6595-6612` | MenF Q9CHJ9, MenD Q9CHK0, MenC Q9CHK4, MenE Q9CHK3, MenB Q9CHK2, MenG P49016 (6/9) — and the ORCH-755 *cremoris* tie disappears because 1359 is outside 1358 | yes, Q1 decides which rank may stand for KF147 |
| FX2 | **Feed the lookup the resolved species, not the row's free-text organism.** The species row already carries a taxonomy id before mapping ends (NCBI backfill); use it and its canonical name for the query and the verdict | `map_ids.py:10233-10255`; species hydration order relative to `prefreeze_resolution.py:1394` | makes FX1 reachable; no protein on its own | Q1 (same ruling) |
| FX3 | **PathBank species by taxonomy id** (and PathBank's "(strain X)" spelling) so `species_not_found` stops skipping the gene / exact-name / alias rungs | `PathBankDbResolver._find_species_ids` (`:990-1049`), short-circuit in `map_protein_row` (`:2119-2125`) | 0 Men today (PathBank has no *Lactococcus* Men rows); maps KF147 to PathBank species 10733 and helps every strain-named organism | no |
| FX4 | **Break gene-synonym collisions by function**: a synonym-only symbol hit loses to a candidate whose EC / reaction (SEPHCHC → SHCHC, EC 4.2.99.20) matches | `_extract_uniprot_candidates` (`:4312-4412`) or the identity verdict | MenH → Q9CHK1 (not P49016, the MenG methyltransferase) | no |
| FX5 | **Exact-sequence route for strains absent from UniProtKB** (new capability, labelled new acceptance test under G9): UniParc `taxonomy_id:<strain> AND gene:<sym>` → UPI → the **active UniProtKB record of the identical sequence** | UniProt tier of `map_ids.py` / `mapping/uniprot_evidence.py` | MenB Q9CHK2, MenI Q9CHK5 — the *same polypeptide* as KF147's; MenA only if a paralog may be chosen (KF147 has three `menA`-annotated genes; the paper gives no locus tag) | yes, Q1(a) |
| FX6 | **One kind per name** so an inferred row cannot take a grounded protein's accession (761-B) | `process_normalizer.py` `_rewrite_token` → `_ensure_compound` (`:1162-1202`, `:933`); `map_ids.py` compound reroute (`:10802-10830`) | EntB P0ADI4 on the ORCH-755 shape | no (D-064 already rules holo/apo states share the accession within one kind) |
| FX7 | **Do not orphan a row whose identity a component already carries** (761-C): among local rows sharing the component's accession, keep the row with the component's own name; or exempt by identity, not name, at the final gate | `_reconcile_components_against_local_proteins` (`map_ids.py:6985-7078`, `setdefault` `~:7027`); `process_normalizer.py:5092-5098` | the ORCH-759 3-draw `review_required` PWML | ORCH-759's pending entity-row ruling |

**Explicitly not recommended:** matching "enterobactin synthase complex" to PathBank complex 3623 by stripping
" complex". The row is Stage-2-inferred and absent from the paper, and 3623's composition (EntB×2, EntD, EntE×2,
EntF) is not the gold's R5 enzyme (EntF). That would add biology to make a gate pass. Also not recommended: any
change to the verdict's thresholds, to the species rung, or to the Unknown-placeholder policy.

**Behaviour change to prove under G9:** FX1 removes an identity T2PW currently calls verified — ORCH-755's MenA
→ A2RHR6 (MG1363, *cremoris*), admitted by the string rule (`map_ids.py:5655-5658`). A lineage-scoped query no
longer returns it. That is the intended direction, but it is a moved outcome and needs its own base-failing proof.

## 8. Not settled — for the product owner

**Q1. For a strain-specific paper whose strain has no UniProtKB entries, what may stand as the protein's UniProt
identity?** KF147 has 2 UniProtKB entries; its Men proteins exist only in UniParc, which PathWhiz rejects.

| option | what it admits for PMC12421875 | cost / consequence |
|---|---|---|
| (a) exact-sequence only (UniParc identity → the active UniProtKB record of the same sequence) | MenB Q9CHK2, MenI Q9CHK5 (+ MenA Q9CF18 if one of three paralogs may be named) | needs FX5 (new capability); 6 enzymes stay without identity; the PathWhiz row is filed under IL1403 (`check_uniprot`), although the polypeptide is KF147's |
| (b) same-subspecies record of the same gene (IL1403, subsp. *lactis*) | (a) + MenF, MenD, MenC, MenE, MenG, MenH (with FX4) | an ortholog record, not the strain's own sequence; matches PathBank's own curation (§ 3.1: MG1655/K-12 and *B. subtilis* pathways use related-strain records); PathWhiz files each protein under IL1403 (`protein.rb:180-190`) |
| (c) same-species record, any subspecies | (b) plus *cremoris* records | UniProt's taxonomy now ranks 1359 *cremoris* as a species outside 1358; this is what the string rule already does today (ORCH-755 MenA → A2RHR6) |
| (d) strain-exact only | nothing | the MenD loss and the 8 placeholders persist for this paper; every Men protein ends unresolved, honestly reasoned |

The gold (`actual_organism: "Lactococcus lactis"`, `KF147` a forbidden strain identifier, "resolves by name plus
organism") reads as (b) or (c); D-106 says "organism and strain-aware lookup". The two need reconciling by the
product owner, not by a lane.

**Q2. (only if FX6 is declined)** In identity admission pass B, may a cross-kind conflict withhold the accession
from the claimant whose accession came from a literature alias / `best_effort_fallback`, and leave it on the
claimant verified from PathBank? That changes the D-064 implementation; options and cost are ORCH-758 § 9 item 2.

## 9. Observed, not adjudicated (outside this charter)

- **T2PW writes fabricated PathWhiz complex ids.** `writer.py:1311` emits `pwp-id` `PW_P{document id:06d}`;
  PathWhiz keeps a non-blank `pwp_id` (`protein_complex.rb:383-389`) and has no unique index on it. The DB already
  holds `PW_P040000` on PathBank complex 40000 *and* on complexes 431773 "Hexokinase complex", 431796, 431802;
  `PW_P040001-3` likewise (`evidence/orch761_pathbank_probe5.txt`). The colliding rows' naming matches T2PW wrappers;
  their provenance was not verified.
- **`EntCB complex` is in the ORCH-756 1-draw `pathway.review_required.pwml`** (`.claude/worktrees/orch756/runs_papers/
  ORCH-756/legs/enterobactin-strict/2026-10-21_2020/…/strict/`, sha256 `5245ce32…`; complex 40000, reaction-enzyme 21).
  The gold lists `EntCBAE` / `EntCB` / `EntC-EntB` under PMC12096016 `forbidden_identifiers` ("Hypothesised
  multi-enzyme assembly and pairwise complex names; each denotes several proteins"). Needs its own adjudication.
- **The writer sets every complex's `species-id` to the pathway default** (`writer.py:1310`), not the complex row's
  species; irrelevant for single-species papers, not checked further.

## 10. What I could not verify

- No PathWhiz import was run. The cascade (protein → complex → reaction), `check_uniprot`'s renaming and
  re-speciation, and the "upload reports success" behaviour are static readings of the Ruby.
- Ruby `open-uri` against the legacy UniProt URL was not run; Python follows its redirect.
- That the PathBank database I queried is the one PathWhiz imports into: it contains T2PW-shaped complexes (§ 9),
  which is consistent, not proof.
- Sequence identity is UniParc membership (one UPI = one identical sequence); not checked residue by residue.
- Which KF147 `menA` paralog the paper's MenA is; the paper and the gold give no locus tag.
- Whether the audit wrote the name `EntF` or only the accession (ORCH-759 left this open; not resolved here).
- The duplicate-`pwp_id` census query timed out (`orch761_pathbank_probe5.txt`); only the six ids queried are confirmed.
- UniProt and PathBank answers are as of 2026-10-22; entries can move.
- Every recovery count in § 7 is a hypothesis until a leg runs with the fix.
- All leg artifacts are uncommitted; the extract pins each by sha256.

## 11. Evidence (all under `docs/pwml_recovery_sprint/evidence/`)

| file | content |
|---|---|
| `orch761_leg_identity_extract.py` / `.txt` | identity records of the four legs, each source pinned by sha256 |
| `orch761_pathbank_probe.py` … `probe5.py` / `.txt`; `orch761_g11_pathbank_probe*.json` | read-only PathBank queries and their bounded-run reports (0 survivors each) |
| `orch761_uniprot_probe.py` / `.txt` / `.json`; `orch761_uniprot_probe2.*` | UniProtKB searches: T2PW's exact query shape, taxonomy-scoped queries, taxon statistics |
| `orch761_uniparc_probe.*`, `orch761_uniparc_xref_probe.*` | KF147 sequences in UniParc and the UniProtKB records sharing them |
| `orch761_uniprot_entries.*` | names, genes, EC, organism of every accession cited |
| `orch761_legacy_uniprot_probe.*` | the URL PathWhiz's `check_uniprot` uses |

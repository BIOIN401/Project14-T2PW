# PF-746-01 — rag/eligibility.py: the organism lexicon cannot see most plant hosts

Status: REGISTERED, not dispatched. Production is frozen (D-102); dispatch needs a product-owner decision.

## Defect

`observe_metadata` (`src/t2pw/rag/eligibility.py:1468-1500` at `b4bb0b74`) can observe an organism in only three ways:

- the organism is an entry in `_ORGANISM_ENTRIES` (`:463-550`);
- it is a binomial or bare genus whose genus is in `_KNOWN_GENERA` (`:563-565`). That table is built from the same entries (`if e.genus`) and is scanned by `_binomials` (`:1448-1465`);
- it is the *requested* organism, when that organism is missing from the lexicon (`:1487-1489`).

`_ORGANISM_ENTRIES` holds 45 entries: 30 bacteria, 2 fungi, 9 animals, 2 viruses and **2 plants**. *Lactobacillus plantarum* (`:490-494`) is a bacterium. The two plants:

```
510:    _entry("Arabidopsis thaliana", "arabidopsis thaliana", "a thaliana"),
511:    _entry("Quercus robur", "quercus robur", "q robur"),
```

The `Quercus robur` entry has no comment saying why it is there. Its only test references are the cholesterol false-positive fixture comments for PMC12113831 ("Quercus robur keratinocyte proteomics", `tests/helpers_eligibility.py:112`, `tests/test_paper_eligibility.py:994`). *Arabidopsis* and *Quercus* are the only plant genera the binomial scan knows.

**What this does to a paper about any other plant host:**

- **It names no lexicon organism.** `observed_organisms` is `[]` and `compare_organism` (`:1523-1538`) returns `unknown`. The organism veto (`:2002-2009`) and the `incompatible_organism` penalty (`:1746-1751`) never fire.
- **It also names *Arabidopsis*** (for example as a gene source or a citation). Only *Arabidopsis* is observed, and the verdict is `match`.

`eligibility.py` is the same blob (`505ee9b1`) at the frozen SHA `5f6a7aa3`, at `b4bb0b74`, at mining `82fc69a7`, at species `686ae134`, and at the pilot code SHAs `ecf6b9a0` (pilot-002) and `f9556288` (pilot-003).

## Observed impact

**Observed in the pilots** (target *A. thaliana*, topic jasmonate biosynthesis). Source: `docs/pathway_mining/pilot/00{2,3}/run/papers.jsonl` on `experiment/pathway-mining-pilot-002` and `-003`.

- **PMC:PMC5382222** (pilot-002, a wheat transcriptome paper). Its title names wheat, and its abstract names wheat and *Triticum aestivum*. The production screen returned `eligible` (score 2.75) with `observed_organisms: []` (`PILOT-002-AUDIT.md:260`).
- **PMC:PMC13364333** (pilot-002 and pilot-003, a review with *Marchantia polymorpha* in its title). `eligible`, `observed_organisms: []`.
- **PMC:PMC12787732** (pilot-002 and pilot-003, pepper CaLOXs). `eligible`, `observed_organisms: []`. Its title and abstract say only "pepper": they never say *Capsicum* or *annuum*.
- **PMC:PMC5104190** (pilot-002, grape PAT1). The observed organisms were `["Arabidopsis thaliana", "Arabidopsis arabidopsis"]`. The screen did not see grape, nor the abstract's "Vitis amurensis".
  - Mining route 1 admitted it. The audit records this as a false rescue: "'grape' is unknown to both organism detectors" (`PILOT-002-AUDIT.md:258`, A8 at `:569`).
  - It was never processed (A7, `:568`), so no claim came from it.

**Measured dependence, on non-plant organisms.** The species branch's `known_limitations.md:87-100` (`686ae134`) reports a red-team run on a 358-document real corpus, re-measured at 6454ef98: 29 sentences were credited with the screen empty; with the screen, 27 were held back and 2 credited. The organism list it gives (E. coli, Listeria, Pseudomonas, Fusarium, mouse, zebrafish) is for the earlier fd7c737c measurement (34 of 37 held). *Fusarium* has no entry or genus in the production lexicon, so how the screen supplied it is unverified.

**Inferred, not observed:**

- **Mis-credit in mining.** Take a plant-host paper whose loss-of-function sentence names no organism, and whose passage names no organism from mining's `_ORGANISM_WORDS`. That sentence would be credited to the target *A. thaliana*, because the screen supplied no other organism (ceiling N4, `species_policy.md:411-418` at `686ae134`). No pilot record shows this happening.
  - Pilot-002's only primary came from an *A. thaliana* 13-LOX paper. Pilot-003 had 0 primaries.
  - In pilot-003, 21 of the *Marchantia* review's 23 reactions carry a model-cited species and 2 carry none (`PILOT-003-AUDIT.md:44-78`).
- **Production RAG admission.** When a span names no organism, admission falls back to the paper-level observed organisms carried from the screen (`src/t2pw/rag/synthesize.py:1752-1761`). The span-level reader `organisms_in_span` uses the same lexicon. So the same blind spot would reach production admission on plant topics. No production run on a plant pathway was examined.

**N-11 is a different defect and is not covered by this card.** The handoff (`:77`) reports COW1, RAT5 and MAN7 colliding with "cow", "rat" and "man". `_normalize` keeps "rat5" as one token, so `\brat\b` does not match it (`eligibility.py:176-196`), and the lexicon has no "cow" or "man" alias. The collision comes from mining's title tokenizer `re.findall(r"[A-Za-z]+", title)` (`acquisition.py:711` at `82fc69a7`) feeding species.py `_COMMON_NAMES` ("man" `:77`, "rat" `:81`, "cow" `:100`).

## Current mitigation in mining

At `82fc69a7`:

- **Extra host list.** `_EXTRA_HOSTS = ("wheat", "triticum", "grape", "vitis")` (`src/t2pw/mining/acquisition.py:65-72`) is used only by the route-1 title guard `_other_organisms_in_title` (`:695-717`; route-1 condition `:761-763`).
  - That guard also runs `observe_metadata` on the title (counting `mismatch` only, `:709-710`), and it reads species.py `_COMMON_NAMES` (`species.py:74-105`, which includes tomato, rice, maize, corn and tobacco).
  - Its comment names what still gets through (`:69-71`): "barley, soybean, 'Nicotiana tabacum' written as a binomial".
- **Foreign-host guard.** `_FOREIGN_HOST` (`:63-64`, applied at `:756`) turns away titles with "transgenic", "heterologous", "ectopic" or "overexpress".
- **No mitigation for the species gate's paper-level input.** The path is:
  1. `acquisition.py:916-923` copies the screen's `observed_organisms` onto the record;
  2. `controller.py:503` passes them on as `paper_observed_organisms`;
  3. `extraction.py:398-402` uses them when the claim cites no species of its own.
- **On the species branch** (`686ae134`), a sentence that names another organism earns no credit (`species.py:938-940`). For a sentence that does not name the target, `:941` withholds credit only when the screen (`paper_other`, `:919`) or the passage (`span_other`, `:975`) names another organism.
  - The passage check uses `organisms_named` (`:691-700`), built on mining's own `_ORGANISM_WORDS` (`:585-597`), which lists more plants than the screen.
  - For a passage that names no organism, the screen is the only remaining check.

## Proposed change

Add plant entries to `_ORGANISM_ENTRIES` in `src/t2pw/rag/eligibility.py`. No logic change.

**The product owner fixes the list.** Proposed from the evidence:

- the N-9 hosts: *Zea mays*, *Hordeum vulgare*, *Glycine max*, *Vitis vinifera*, *Nicotiana tabacum*;
- the pilot misses: *Triticum aestivum*, *Marchantia polymorpha*, *Capsicum annuum*. The binomial alone does not catch PMC12787732, which says only "pepper";
- the remaining plants in mining's common-name table: *Oryza sativa*, *Solanum lycopersicum*.

**Alias rules:**

- **Spellings.** Give the canonical binomial plus the abbreviated binomial ("z mays").
  - Exception: no "g max". The normalized centrifugation notation "100,000 × g max" ("100 000 g max") would match it.
- **Genera that are ordinary words.** A genus that is also a common word or chemical name gets `genus=""`, so it stays out of `_KNOWN_GENERA` (`:564`). Otherwise `_binomials` (`:1457-1464`) would read "glycine betaine" as a *Glycine* species, and a bare "glycine" (the amino acid) as the genus *Glycine*. Either reading gives `mismatch` against *A. thaliana*.
  - Cost: a `genus=""` entry compares by exact equality only (`:448-449`, `:1406-1407`). See Risks.
- **Common names need product-owner approval, one at a time** (maize, barley, soybean, grapevine, tobacco, rice, wheat, tomato, pepper).
  - They collide with media, food and exposure words: corn steep liquor, soybean meal, rice bran, tobacco smoke.
  - A false observation can turn `unknown` into `mismatch`, which is an organism veto with `needs_manual_review=False` (`:2002-2009`).

**Out of scope:** a taxonomy-backed source and a per-paper principal organism (N4). `known_limitations.md:184-186` (`686ae134`) says "No finite genus list closes this" about mining's `_ORGANISM_WORDS`; the same applies here. This card narrows the gap; it does not close it.

## Ownership

- `src/t2pw/rag/eligibility.py :: _ORGANISM_ENTRIES`: the tuple literal at `:463-550`, with new entries added next to `:510-511`.
- One new file: `tests/test_eligibility_plant_hosts.py`.

Nothing else may change:

- not `_binomials`, `_NOT_SPECIES`, `_taxon`, `_compare_one`, `observe_metadata` or `_ANIMAL_HOST_ORGANISMS`;
- not `rag/admission.py`;
- nothing under `src/t2pw/mining/`.

## Acceptance

Unless noted, the scope is `RequestedScope(requested_pathway="jasmonate biosynthesis", requested_organism="Arabidopsis thaliana")`.

1. **Single plant.** For every added entry, `observe_metadata(title, "", scope)` on a title naming only that plant returns the entry's canonical name, and `compare_organism` returns `mismatch`. This holds for the binomial form, and for the abbreviated form of every entry that has one (*Glycine max* has none).
2. **Mixed title.** A title naming *Arabidopsis thaliana* and an added plant returns both in `observed_organisms`, and `compare_organism` stays `match`. The title must contain no other "Arabidopsis <word>" pseudo-binomial.
3. **Full screen.** `screen_paper` with explicit `thresholds=EligibilityThresholds()`, an unpinned scope, and a title and abstract that name only an added plant returns `outcome == "ineligible_organism"`. The explicit thresholds are needed because the default is `thresholds_from_config()`, which reads the environment (`:1847`).
4. **Preserved behaviour.**
   - Every existing entry is observed exactly as before.
   - An abstract with "glycine betaine" and no organism observes nothing.
   - An abstract with a bare "glycine" and no organism observes nothing.
   - "100,000 x g max" observes nothing.
   - "RAT5 mutant" observes nothing.
5. **`genus=""` delta, documented.** `apply_stage0_observation` with requested "Glycine max" and a Stage-0 organism "Glycine max cv. Williams 82" returns no conflict at `b4bb0b74`. After the change it is either still conflict-free, or the move to a conflict is recorded as an exact delta for the product owner to accept (merge rule 4). This is inferred from reading the code and was not run.
6. **Existing tests.** These pass, or move by an exact documented delta (merge rule 4):
   - `tests/test_paper_eligibility.py` and `tests/test_paper_eligibility_corrections.py`, including the real-corpus sets `JUNK_2122`, `GENUINE_2122` and `CHOLESTEROL_FALSE_POSITIVES_2122` (`tests/helpers_eligibility.py:90-117`);
   - `tests/test_c077_stage0_conflict_disposition.py`;
   - `tests/test_semantic_release_gating.py` (organism verdict table `:494-506`);
   - SMOKE at 508.
7. **Boundary.** The diff touches only the lines listed under Ownership.

## G9 obligation

This corrects pre-existing observable behaviour, so the proof must fail behaviourally at `b4bb0b74`.

**Test:** parametrize over the added canonical names, with the requested organism set to *Arabidopsis thaliana*. For each, assert that the canonical name is in `observed_organisms` and that `compare_organism == "mismatch"`.

- **At `b4bb0b74` it fails:** `observed_organisms == []` and the verdict is `unknown`. No added name hits an existing alias or an existing genus. The requested organism is in the lexicon, so `:1487` adds nothing.
- **The requested organism must differ from the plant.** At `:1487-1489`, a requested organism that is missing from the lexicon is already found through its generated aliases. Requesting "Zea mays" would pass at base and prove nothing.

Two more cases:

- **Mixed title:** at base it returns `["Arabidopsis thaliana"]` only.
- **`screen_paper` returning `ineligible_organism`** (with explicit thresholds): impossible at base for such a paper, because the veto at `:2002` requires `mismatch`.

**Base arm.** The drafter reported a read-only in-memory probe of the `b4bb0b74` source. It covered titles naming *Zea mays*, *Glycine max*, *Nicotiana tabacum*, *Triticum aestivum*, *Marchantia polymorpha*, *Oryza sativa*, *Solanum lycopersicum*, maize, barley, soybean, grapevine, grape, tobacco, rice, tomato and wheat, and every one returned `[]` and `unknown`.

- The checker did not re-run it. The result matches a reading of the code.
- It was not a test run and has no G11 report.
- The implementer must reproduce both arms through the bounded wrapper, with `--basetemp`.
- The acceptance-4 cases are preservation guards. They pass at both SHAs and are labelled as preservation, not as the G9 proof.

## Risks and interactions

**Production readers of the lexicon:**

- **The screen.** `screen_paper` / `screen_candidate`, called at `batch/fetch.py:650,788`, `app/streamlit_app.py:543` and `scripts/eligibility_dry_run.py:322`.
- **Stage-0 organism conflict check.** `apply_stage0_observation` (`:2248-2254`), called at `batch/driver.py:732` (imported `:101`). It uses `compare_organism` → `_compare_one` → `_taxon`, which reads the entries and `_KNOWN_GENERA`.
- **RAG admission (the larger blast radius).**
  - `rag/admission.py::organisms_in_span` (`:2212-2239`) iterates `_ORGANISM_ENTRIES` and calls `_binomials` directly. It is called at `admission.py:3059` and `synthesize.py:1757`, so span-level organism observation in the production admission gate changes too.
  - `admission.compare_organism` (`:2171-2209`) delegates to `eligibility._compare_one`.
- **Benchmark.** `bench/semantic_production.py::_check_requested_organism` (`:153`, import `:172`) goes through `admission.compare_organism`.

**Direction of change:**

- In the screen, observations are only added and any match wins (`:1535-1537`). A screen `match` cannot be lost, but `unknown` can become `mismatch`, giving more organism vetoes with no manual review.
- In comparisons that run through `_taxon` on free-text names (Stage-0, admission, bench), a `genus=""` entry makes that organism compare by equality only. That can turn a `match` into a `mismatch` (acceptance 5). This is inferred from the code and was not run.
- **Inferred consequence for the product owner.** On an *A. thaliana* run, PMC5382222 (it names *Triticum aestivum*) and PMC13364333 (it names *Marchantia polymorpha*, and its title and abstract do not name *Arabidopsis*) would become `ineligible_organism`. Mining never rescues an organism veto (`acquisition.py:723-724` at `82fc69a7`). PMC12787732 would not change unless "pepper" is approved. PMC5104190 stays `match`.
- The screen gate is tightened, not weakened, so merge rule 6 is not engaged. The risk is wrongful vetoes from false observations, which is why the alias rules above exist.

**Loose epithet rule.** Every new genus inherits the binomial scan's epithet rule (`[a-z]{3,}`, `:1458`). That rule already produces pseudo-binomials such as "Arabidopsis jasmonate" (pilot-002 PMC3000969) and "Arabidopsis arabidopsis" (PMC5104190).

**Tests:**

- `test_paper_eligibility*.py` are not in the SMOKE file list (`TEST_MATRIX.md:259-271`).
- SMOKE files that may reach the lexicon through admission or the batch driver must stay green: `test_rag_admission_production_path.py`, `test_rag_gap_admission.py`, `test_rag_provenance_gates.py`, `test_batch_driver.py`.

**Mining:**

- Route 1's title guard will see more title organisms, so there will be fewer route-1 rescues (fail-closed).
- `paper_observed_organisms` gains plant hosts, so `paper_other` (`species.py:919` at `686ae134`) holds back more sentences that name no organism. This is the intended effect.
- `_EXTRA_HOSTS` becomes partly redundant; leave it alone.
- After a production merge, re-run the mining suite and the screen fixture (`docs/pathway_mining/evidence/017-item4-screen-fixture` at `82fc69a7`).

**Not covered:** N-11 (mining's tokenizer), N4 (principal organism), and any plant not on the approved list.

## References

- `src/t2pw/rag/eligibility.py` @ `b4bb0b74`: `:15-19`, `:176-196`, `:448-449`, `:463-550`, `:563-565`, `:1393-1442`, `:1448-1500`, `:1523-1538`, `:1746-1751`, `:1847`, `:1994`, `:2002-2009`, `:2248-2254`
- `src/t2pw/rag/admission.py` @ `b4bb0b74`: `:2171-2209`, `:2212-2239`, `:3059`
- `src/t2pw/rag/synthesize.py` @ `b4bb0b74`: `:1752-1761`
- `src/t2pw/batch/fetch.py` @ `b4bb0b74`: `:650`, `:788`; `src/t2pw/batch/driver.py`: `:101`, `:732`; `src/t2pw/app/streamlit_app.py`: `:543`; `scripts/eligibility_dry_run.py`: `:322`; `src/t2pw/bench/semantic_production.py`: `:153`, `:172`
- `tests/helpers_eligibility.py` @ `b4bb0b74`: `:90-117`; `tests/test_paper_eligibility.py`: `:994`; `tests/test_semantic_release_gating.py`: `:494-506`
- `src/t2pw/mining/acquisition.py` @ `82fc69a7`: `:63-72`, `:695-717`, `:723-724`, `:756`, `:761-763`, `:916-923`
- `src/t2pw/mining/controller.py` @ `82fc69a7`: `:503`
- `src/t2pw/mining/extraction.py` @ `82fc69a7`: `:398-402`
- `src/t2pw/mining/species.py` @ `82fc69a7`: `:74-105`
- `src/t2pw/mining/species.py` @ `686ae134`: `:585-597`, `:691-700`, `:919`, `:938-941`, `:975`
- `docs/pathway_mining/known_limitations.md` @ `686ae134`: `:87-100`, `:176-186`
- `docs/pathway_mining/species_policy.md` @ `686ae134`: `:411-418` (ceiling N4)
- `docs/pathway_mining/pilot/002/audit/PILOT-002-AUDIT.md` (pilot-002 branch): `:16`, `:258`, `:260`, `:568`, `:569`
- `docs/pathway_mining/pilot/003/audit/PILOT-003-AUDIT.md` (pilot-003 branch): `:44-78`, `:506`
- `docs/pathway_mining/pilot/00{2,3}/run/papers.jsonl` (pilot branches)
- `docs/pwml_recovery_sprint/ORCH-745-RESULT.md`: `:143-144`, `:155`, `:175-176` (where this defect was registered)
- `C:/t/mining-handoff/MINING-OVERNIGHT-HANDOFF.md`: `:76` (N-9), `:77` (N-11), `:98` (production-problems list; it does not name `eligibility.py`)
- `docs/pwml_recovery_sprint/TEST_MATRIX.md`: `:259-271` (SMOKE files)
- `docs/pwml_recovery_sprint/DECISIONS.md`: D-102 (`:6911`)

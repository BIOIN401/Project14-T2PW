# PathWhiz import inventory — every PWML currently on disk, ready for you to import

**2026-09-10.** Eighteen files, all committed and hash-pinned under
[`pathwhiz_review/IMPORT-SET/`](pathwhiz_review/IMPORT-SET/), every one `cmp`-verified byte-identical
to its source run tree, which is preserved unchanged. `sha256sum -c` passes on all eighteen.
Verified by [`evidence/orch734_pathwhiz_import_check.py`](evidence/orch734_pathwhiz_import_check.py),
report `evidence/g11/C-123/02-import-inventory.json`, `FINAL SURVIVING COUNT : 0`.

| | |
|---|---:|
| **structurally clean — import these** | **17** |
| **`F-195` dangling transport reference — import last** | **3** |

> **UPDATED AGAIN after the `C-126` live validation, 2026-09-10.** Two more clean files, both
> `review_required`, both `IMPORT READY`, both with **zero** transports and so **not** `F-195`
> candidates. Twenty files now; `sha256sum -c` passes on all twenty.
>
> | file | pathway organism | rxn | cpd | prot | cplx | edges | bytes | sha256 |
> |---|---|---:|---:|---:|---:|---:|---:|---|
> | `C126_PMC13123502.pwml` | *Paris polyphylla* — steroidal saponin | 2 | 5 | 1 | 4 | 13 | 36,756 | `14fbe1b041f6d3ba` |
> | `C126_PMC9544450.pwml` | *Escherichia coli* — menaquinone | 4 | 8 | 2 | 2 | 12 | 40,689 | `dddbfb0eac22ffb5` |
>
> **`C126_PMC13123502` is the one to look at first.** It is the paper `F-147` had been killing: a
> leg whose final gate passed clean while a superseded pre-remap report failed it. Its four
> unresolved UGTs ship as Unknown-backed complexes, so it is also the clearest live specimen of the
> placeholder policy. **`C126_PMC9544450` is the `C-126` control** and is this paper's *fourth*
> independent draw — 4, 5, 6 and now 4 reactions — so the four together remain the best available
> demonstration of draw dependence.

> **UPDATED after the `C-124`/`C-125` live validation.** `C124_PMC13488460.pwml` joins the clean set:
> *Borrelia burgdorferi* mevalonate pathway, 19,818 B, 1 reaction, **IMPORT READY**, no transports.
> It is the paper that previously died on `species_missing_taxonomy`, and it is the first PWML the
> system has produced for an organism the local database does not contain. Its species carries
> taxon **139** from NCBI with the paper's own spelling preserved. Small, but it is the direct
> product of a fix and worth importing to confirm a created species renders.

**Nothing here was modified to make an import succeed.** If one fails, the file is the evidence.

---

# 1. The fifteen clean files

Ordered by reaction count, largest first. `T` = transport visualizations, `sha256` = first 16 hex.

| file | pathway organism | taxon | rxn | cpd | prot | cplx | edges | bytes | T | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ORCH732_PMC12051542.pwml` | *Aquifex aeolicus* | 63363 | 8 | 11 | 5 | 6 | 29 | 80,518 | 0 | `44cf838f1a4d6ced` |
| `C122_PMC13474689.pwml` | *Moringa oleifera* — glucomoringin | 3735 | 7 | 8 | 1 | 7 | 21 | 60,767 | 0 | `bf13e38b6c7b42c8` |
| `C122_PMC13084691.pwml` | *Astragalus membranaceus* — calycosin | 649199 | 6 | 11 | 1 | 3 | 24 | 67,552 | 0 | `18082b27c6b51dc8` |
| `ORCH734_PMC9544450.pwml` | *Escherichia coli* — menaquinone | 562 | 6 | 8 | 2 | 2 | 17 | 49,436 | 0 | `cb209a921cc224da` |
| `C121_PMC9544450.pwml` | *Escherichia coli* — menaquinone | 562 | 5 | 9 | 2 | 2 | 15 | 48,826 | 0 | `57efe6e2fdf3fa12` |
| `ORCH732_PMC4725005.pwml` | *Escherichia coli* | 562 | 5 | 9 | 4 | 4 | 16 | 54,459 | 0 | `6a54fbda584bc358` |
| `ORCH734_PMC10269868.pwml` | *Helicobacter pylori* — ADP-heptose | 210 | 5 | 5 | 5 | 4 | 16 | 48,436 | 0 | `3d1c0e3b3695da72` |
| `C122_PMC12707518.pwml` | *Streptococcus suis* — branched-chain amino acid | 1307 | 4 | 13 | 1 | 3 | 22 | 63,894 | 0 | `d236302798037745` |
| `ORCH732_PMC9544450.pwml` | *Escherichia coli* — menaquinone | 562 | 4 | 8 | 2 | 2 | 11 | 39,686 | 0 | `679c6f028fcd69da` |
| `ORCH734_PMC10055903.pwml` | *Homo sapiens* — sialic acid | 9606 | 4 | 9 | 4 | 4 | 18 | 55,011 | **2** | `8a06b4edc980f94d` |
| `C122_PMC12914822.pwml` | *Aspergillus aculeatus* — *ent*-acu-dioxomorpholine A | 5053 | 3 | 6 | 1 | 3 | 11 | 35,078 | 0 | `504624aac550614c` |
| `C120_PMC10031235_PSAT.pwml` | *Homo sapiens* — PSAT | 9606 | 3 | 8 | 2 | 2 | 15 | 48,401 | 0 | `6f17819ce7ed1f48` |
| `ORCH734_PMC7402084.pwml` | *Phaseolus lunatus* — cyanogenic glucoside | 3884 | 3 | 6 | 4 | 4 | 12 | 40,294 | 0 | `368f9e7ea0ff673d` |
| `ORCH734_PMC8211424.pwml` | *Homo sapiens* — corticosteroid | 9606 | 2 | 4 | 1 | 1 | 6 | 22,859 | 0 | `e893778e21718dea` |
| `C124_PMC13488460.pwml` | *Borrelia burgdorferi* — mevalonate | **139** | 1 | 4 | 1 | 1 | 5 | 19,818 | 0 | see `SHA256SUMS.txt` |

## Two things worth knowing before you start

**`ORCH734_PMC10055903` is the one clean file that carries transports** — two of them, and **zero**
dangling references. It is the control for `F-195`: if it imports and renders correctly while the
three files in § 2 do not, the defect is isolated to how those three were constructed rather than to
transports as a feature.

**`PMC9544450` appears three times on purpose** — one *E. coli* menaquinone paper on three separate
draws, at 4, 5 and 6 reactions. Import them together. The spread *is* the finding: this pipeline is
draw-dependent, and seeing the same paper produce three different pathways is more informative than
any single file.

**`C122_PMC12914822` is the only `release_ready` file.** Every other file here is
`review_required`. It was briefly committed under a `WRONG-SPECIES` filename; **that label was an
error and has been withdrawn** — see § 3. Import it normally.

---

# 2. The three `F-195` files — import these LAST

Each carries a `transport-compound-visualization` that names a `compound-location-id` the document
never declares. **They will probably be rejected**, and that is the point of importing them.

| file | pathway organism | rxn | bytes | transports | dangling | sha256 |
|---|---|---:|---:|---:|---:|---|
| `ORCH734_PMC11016064.F195.pwml` | *Homo sapiens* — carnitine | 10 | 107,951 | 2 | **2** | `0032d8e88ca8aa21` |
| `ORCH734_PMC6112128.F195.pwml` | *Streptomyces tubercidicus* — tubercidin | 8 | 78,492 | 2 | **1** | `66493362c8d2eb54` |
| `ORCH730_PMC7232280.F195.pwml` | *Neurospora crassa* — Moco | 4 | 45,054 | 2 | **2** | `f9b5b80b053df3a1` |

**Their outcome decides `F-195` and nothing else does:**

* **PathWhiz accepts and renders them** → `F-195` is a limitation of our structural checker.
  Document it as a tolerated defect and close it with no code.
* **PathWhiz rejects them, or renders them broken** → `F-195` is a genuine importability defect and
  may justify one narrow referential-integrity card.

**No pre-emptive fix either way, and do not repair a file to make an import succeed.**

Note the ratio: **3 of the 4 transport-carrying files dangle**, and the fourth
(`ORCH734_PMC10055903`) does not. Transports are necessary but not sufficient for the defect.

---

# 3. What to look at after each import

The structural checker already decided everything decidable from the bytes — XML parses, the
visualization envelope is present, the canvas is non-zero, elements are positioned, edges carry real
paths, compounds and proteins are declared and placed, every reaction has both sides and its compound
references resolve, and the species carries a numeric taxonomy id. **`IMPORT READY` is not
`IMPORT PASS`.** What only you can decide:

1. **Does the import succeed**, and does the pathway render?
2. **Do coordinates and layout generate**, or is the graph blank?
3. **Are the reactions visible**, with compounds and proteins/complexes on the canvas?
4. **Is the pathway biologically recognisable** as the one the paper describes?
5. **Does a spurious second species appear in the UI?** Ten of these files declare an extra
   *Arabidopsis thaliana* alongside the real organism. An audit of 34 legs established this is
   **sentinel-only** — it is attached exclusively to the PathBank `Unknown` placeholder and to
   wrapper complexes built around one, and **no resolved protein carries a false organism**
   (`evidence/c123_sentinel_audit.py`, 129 sentinel rows, 0 mislabelled). It should therefore be
   invisible or inert. **If it is visible in the UI or attaches to an enzyme, tell me** — that would
   make it a real data-integrity defect rather than the harmless placeholder the audit says it is.
6. **For the three `F-195` files only:** does the import fail, and if it renders, is the transport
   drawn correctly or dropped?

---

# 4. Provenance

Every file is a byte-for-byte copy of a production run's own output. No file was edited, repaired,
regenerated or selected for flattering content. Seven cohorts are represented — `ORCH-730`, `ORCH-732`,
`ORCH-734`, the `C-120`, `C-121` and `C-124`/`C-125` validations, and the `C-122` final smoke — and
their numbers are
**never merged into one denominator**; each cohort's yield belongs to that cohort alone.

The run trees that produced them are preserved unchanged and untracked
(`runs_smoke/`, `runs_validation/`, `runs_verify/`), which is the standing `F-187` exposure. These
seventeen are the committed, hash-pinned copies — they are on `origin`, so a disk failure no longer
loses them.

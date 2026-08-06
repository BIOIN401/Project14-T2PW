# Corrected exporter-identity probe — measured output, 2026-08-06

Task **H-005**, branch `agent/h05-probe-and-authority-corrections`, base
`6bd4c21a14b8180bb8b1357f878431a989b0a662`.
Producer: `probe_exporter_identity_mutation.py` (this directory), run through
`bounded_run.py`. Cleanup reports: `g11/H-005/01-probe-prefix.json` (the pre-fix probe,
reproducing the defect), `07-probe-final.json` and `08-selfcheck-final.json` (the
committed probe file, definitive). `02-probe-corrected.json` and `03-probe-selfcheck.json`
are earlier runs of the same fix taken before the docstring was trimmed; they produced
identical measurements and are kept on the record rather than discarded.
Interpreter `.venv/Scripts/python.exe` 3.13.6, Windows.
Leg: `runs_verify/2026-08-04_1647/papers/PMC12856317/strict`, committed.

## What was wrong with the previous number

The probe keyed **both sides on the raw `name`** (old `:78-79`). The exporter
renames `glycine → Glycine`, so `canonical_rows.get("Glycine")` returned `{}` and
every field on that compound was counted as added. **This is the historical record
of what was reported and is not deleted** — the pre-fix run is reproduced verbatim:

```
  Glycine
      identity fields changed : {'pathwhiz_id': (None, 78), 'db_id': (None, 78), 'db_status': (None, 'matched_offline_name_index'), 'chosen_rule': (None, 'legacy_pathwhiz_id_unverified'), 'pathbank_compound_id': (None, 78)}
      mapped_ids changed      : {'hmdb': (None, 'HMDB0000123'), 'cas': (None, '56-40-6'), 'chemspider': (None, '730'), 'kegg': (None, 'C00037'), 'pubchem': (None, '5257127'), 'chebi': (None, '15428'), 'biocyc': (None, 'GLY'), 'drugbank': (None, 'DB00145'), 'pathbank_compound_id': (None, '78')}
      *** 9 identifier(s) ADDED by the exporter: ['biocyc', 'cas', 'chebi', 'chemspider', 'drugbank', 'hmdb', 'kegg', 'pathbank_compound_id', 'pubchem']
...
  TOTAL identifiers added post-freeze: 10
```

All nine are in fact present in the canonical Glycine row's `mapped_ids`; only the
`chebi` **value** differs (`CHEBI:15428` → `15428`). The pairing bug also made the
probe **structurally unable to reach its own T-102 target of 0**, since the metric
grew with how faithfully the exporter recorded a rename.

## Corrected pairing

Rename lineage (`raw_name` → canonical `name`) as the primary pairing, **corroborated
by position**, with the two required to agree. `raw_name` alone would trust the code
under examination to declare its own renames; position alone cannot survive a
reordering or a length change. Every unmatched or ambiguous row is a named, loud
condition and is never folded into "added" or "removed"; when one fires the category
counts are suppressed and the probe exits `2`.

`--selfcheck` proves each condition fires (`08-selfcheck-final.json`): **7/7 passed**,
covering `AMBIGUOUS_CANONICAL_NAME`, `AMBIGUOUS_LINEAGE_KEY`, `UNMATCHED_IR_ROW`,
`UNMATCHED_CANONICAL_ROW`, `ROW_COUNT_MISMATCH`, `PAIRING_DISAGREEMENT`, plus the
negative case that identical rows report nothing.

## Corrected measurement — five categories, never summed

```
    4 canonical rows, 4 IR rows, 4 paired, 0 integrity problem(s)

  canonical 'heme'  ->  IR 'heme'
      2. mapped-ID changes            : 1
          - mapped_ids.pubchem ADDED '3334' [within-row re-projection of canonical pubchem_cid]
      4. prefix normalization         : 2   (mapped_ids.chebi and chebi_id, 'CHEBI:17627' -> '17627')
      5. identity materialization     : 4   (pathwhiz_id 1799, db_id 1799, db_status, chosen_rule)
  canonical 'aminolevulinic acid'  ->  IR 'aminolevulinic acid'
      4. prefix normalization         : 2   ('CHEBI:17549' -> '17549')
      5. identity materialization     : 4   (pathwhiz_id 894, db_id 894, db_status, chosen_rule)
  canonical 'glycine'  ->  IR 'Glycine'
      1. name changes                 : 1
          - 'glycine' -> 'Glycine' (raw_name='glycine'; no provenance record accompanies it)
      3. synthetic database rows      : 1
          - db_row {"id": 78, "name": "Glycine"} fabricated at export time
      4. prefix normalization         : 2   ('CHEBI:15428' -> '15428')
      5. identity materialization     : 4   (pathwhiz_id 78, db_id 78, db_status, chosen_rule)
  canonical 'succinyl-CoA'  ->  IR 'succinyl-CoA'
      4. prefix normalization         : 2   ('CHEBI:15380' -> '15380')
      5. identity materialization     : 4   (pathwhiz_id 808, db_id 808, db_status, chosen_rule)

  ---- FIVE CATEGORIES, REPORTED SEPARATELY, NEVER SUMMED ----
    1. name changes                 :   1 instance(s) across 1 of 4 compound rows
    2. mapped-ID changes            :   1 instance(s) across 1 of 4 compound rows
    3. synthetic database rows      :   1 instance(s) across 1 of 4 compound rows
    4. prefix normalization         :   8 instance(s) across 4 of 4 compound rows
    5. identity materialization     :  16 instance(s) across 4 of 4 compound rows
```

Also observed and deliberately **not** counted in the five: an exporter-internal
handle `key = cmp_1..cmp_4` (structural, not a biological claim), and `enrichment`
present on every canonical row and absent from every IR row.

## Before / after

| | old (name-paired) | corrected |
|---|---|---|
| identifiers "added post-freeze" | **10** | **1** — heme `mapped_ids.pubchem='3334'`, a within-row re-projection of that row's own `pubchem_cid` |
| Glycine identifiers "added" | 9 | **0** — all nine are in the canonical row |
| name changes | not reported | **1** |
| synthetic `db_row` | not reported | **1** |
| prefix normalization | folded into the identifier diff | **8** across 4/4 rows |
| identity materialization | reported, uncounted | **16** across 4/4 rows |
| unmatched / ambiguous rows | silently became "added" | loud, named, counts suppressed, exit 2 |

## The violation is unchanged

`PRODUCT_CONTRACT` § 5 is violated **by kind, not by count**: a post-freeze entity
rename with no provenance record, a fabricated `db_row`, `CHEBI:` stripping on 4 of
4 compounds and identity materialization on 4 of 4. Only the magnitude was
overstated. **T-102 acceptance is now per category: all five must be 0**, and no
pairing-integrity problem may be reported. That is the same requirement as "the
diff is empty", stated so no category can be traded against another.

Measurement B of the probe is unaffected by the pairing defect and is unchanged.

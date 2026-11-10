# Clean model build over the 152 canonical runs

Script: `scripts/build_all.py` (local files only, no network, no run file copied here).

| measure | value |
|---|---|
| runs | 152 |
| built | 152 / 152 (100.0%) |
| round trip `to_payload(build(p)) == p` exact | 152 / 152 |
| round-trip diffs listed | 0 |
| runs with title | 0 |
| runs with PMID | 152 |
| runs with PMCID | 150 |
| runs with DOI | 150 |
| runs with organism | 152 |
| runs with year | 0 |
| reactions with evidence quote retained | 621 / 621 |
| transports with evidence quote retained | 106 / 106 |
| interactions with evidence quote retained | 351 / 352 |
| interaction raw spellings (exact / normalised) | 48 / 39 |
| controlled kinds used | 13 |
| interactions unsigned / total | 235 / 352 |
| complexes with any Unknown member | 138 / 415 (in 43 runs) |
| complexes with only Unknown members | 138 |
| interactions with an undeclared endpoint | 15 |
| removed records (view) | 1044 |
| mapping records | 2959 |

## Interactions by kind and sign

| kind / sign | count |
|---|---|
| inhibition/negative | 91 |
| binding/unsigned | 87 |
| association/unsigned | 41 |
| modification/unsigned | 40 |
| activation/positive | 26 |
| modulation/unsigned | 15 |
| catalysis/unsigned | 14 |
| cofactor/unsigned | 11 |
| complex_formation/unsigned | 9 |
| degradation/unsigned | 9 |
| transport/unsigned | 5 |
| other_unsigned/unsigned | 3 |
| competition/unsigned | 1 |

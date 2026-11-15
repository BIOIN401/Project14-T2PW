# Final payload P3 measure (code HEAD 3e782d76)

Scripts: scripts/ (copies of P2 scripts6 with paths changed). Raw: replay/{off,on}, import/{controls,on}, import/analysis.json, import/msgs.txt.

## 1. Flag-off identity
152 runs replayed from stored final_mapped.json, flag OFF at 3e782d76 against P2 off6 (code e9c3e197).
`git diff bf20c050 e9c3e197 -- src` is empty (bf20c050 is docs-only on e9c3e197), so P2 off6 = bf20c050 flag-off; no extra worktree replay done.
Same 97 runs write PWML; **97 identical, 0 different** (sha256). 55 write none in both.

## 2. Flag-on replay (T2PW_FINAL_PAYLOAD=1)
97 of 152 write PWML (same set as P2 on6). 66 of 97 byte-identical to P2 on6 (31 differ: B1/B3/B2 changes).

## 3. Harness import (59 files = same idx list as P2 on6; importer with 3 patches; vocab seed as P2)
| | P2 on6 | P3 on |
|---|---:|---:|
| files success true | 59 | 59 |
| clean (0 messages) | 29 | **43** |
| messages | 211 | **147** |
| ontology (location/cell/tissue) | 110 (27 files) | 62 (12 files) |
| transport (all transport_*) | 37 | 21 (5 files) |
| nucleic-acid (nucleic_acid*, interaction_nucleic_acid_visualization) | 33 (1 file) | 33 (1 file) unchanged |
| left=right "Biological state must have unique biological states" | 6 (5 files) | 6 (5 files) unchanged |
| transport_transporter_visualization "Transport transporter can't be blank" | 19 (12 files) | **3 (3 files)** NOT fully gone |
| interaction | 28 | 28 |
| other | 33 | 33 |
| transports created / in file | 31 / 37 | 31 / 37 |
| transport_transporters / transport_transporter_visualizations created | 16 / 0 | 16 / 16 |
| interactions created / in file | 9 / 17 | 9 / 17 |

Remaining 3 tt-viz messages: runs 151, 10, 68. Likely cascade of the 6 left=right transports that fail (not traced).
Ontology split P3: subcellular_location 26, cell_type 22, tissue 14.
Other residuals unchanged: Pwc taken 1, compound_location 1, protein_complex 2 etc.

## 4. Controls (5; P2 ran only 2)
PW012926 success, 1 msg; PW741182 success, 0 msgs (clean); PW687775 3 msgs; PW176888 2 msgs; PW1 (stub) 176 msgs (81 in the original harness run). Clean 1/5. P2 controls6: PW687775 + PW176888 only, 8 msgs total, now 5 msgs.

## Process notes / unverified
- Import batches were started with a trailing `&` by mistake (rule violation); ran under bounded_run with heavy lock, all 7 batches exit 0, final_surviving_count 0 on the 6 on-batches I grepped (controls json not grepped).
- Comparison of 31 differing flag-on files not inspected for biological correctness.
- Not production PathWhiz; empty DB plus seed.
- Flag-on PWML kept in repo dir replay/ (12 MB total), P2 did the same. Docker project t2pw-pwi stopped. Nothing committed.

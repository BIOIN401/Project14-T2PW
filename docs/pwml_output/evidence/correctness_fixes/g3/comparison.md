# G3 live A/B (partial: 1 of 3 gold papers, strict only). Blocked by host memory < 4 GB for 2 h.

BASE 12c99a005fd7500fbe0f0cc22762052e8a92baac, AFTER c5734190, T2PW_EXTRACTION_DRAWS=3, one draw set per leg.

| PMC12421875 strict (menaquinone, L. lactis) | BASE | AFTER |
|---|---|---|
| export status | review_required (123,600 B) | review_required |
| wall | 1636 s | 1375 s |
| reactions in payload | 11 | 14 |
| gold R1-R9 matched (eval_v2) | 9/9, 0 missing | 9/9, 0 missing |
| TP rows / precision / F1 | 9 / 0.818 / 0.90 | 11 / 0.786 / 0.88 |
| extra rows | 1 unsupported, 1 partial | 1 unsupported, 1 partial, 1 paper_stated_summary |
| enzyme accuracy | 9/9 | 9/9 |
| substrate / product accuracy | 10/10, 9/9 | 10/10, 9/9 |
| species accuracy | 11/11 | 14/14 |
| proteins / complexes / compounds | 8 / 9 / 12 | 8 / 9 / 13 |
| transports / interactions | 2 / 1 | 2 / 1 |
| subcellular_locations | 5 (cytoplasm, cell membrane, intracellular, extracellular, cell) | 3 (cytoplasm, extracellular, cell) |
| biological states (PWML) / compound-location | 9 / 23 | 2 / 15 |
| quarantined_locked_reactions | 8 | 2 |
| model attempts (LEG_TRACE) | 42 | 51 (gap resolver 5 -> 11, rag_prose 18 -> 20, audit 3 -> 4) |

Token usage: not recorded in any run artifact (client token stats are in-process only); request counts above are the only usage figure.

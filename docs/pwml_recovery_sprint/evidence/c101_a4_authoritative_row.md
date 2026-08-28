# C-101 A4 — the authoritative PMC12444477 sentinel row, identified BEFORE any edit

Written at base SHA `d7cf4a4`, before the first production edit on
`card/C-101-o1-metric-split`. D-074 and AMENDMENT 1 § A4 make this a stop condition: if the
accepted C-100 evidence does not uniquely identify one row, C-101 stops. It does identify one
— but **not by the route the charter names**, and that discrepancy is recorded here rather
than smoothed over.

---

## 1. The row

| Field | Value |
|---|---|
| **exact artifact path** | `runs/2026-08-02_2130/papers/PMC12444477/strict/final_mapped.json` |
| **run identity** | `runs/2026-08-02_2130` — *the pinned run*, per LEDGER § "F-141 CLASSIFIED", "which run is 'the pinned run'" |
| **paper** | `PMC12444477` |
| **mode** | `strict` |
| **entity bucket** | `entities.proteins` |
| **complete row identity** | pointer `/entities/proteins/4` · `name = "Unknown"` · `pathbank_protein_id = 9659` · `uniprot = "Unknown"` · `mapping_meta.chosen_rule = "pathbank_unknown_protein_fallback"` · `species = "Arabidopsis thaliana"` (`species_id = 4`) · `generated = false` · `generation_reason = null` · `claims_real = ""` |
| **pointer file that records it** | `docs/pwml_recovery_sprint/evidence/orch710_pinned21.json`, entry 7 of `placeholders[]` |

### Why it satisfies the sentinel predicate

`entity_identity.is_pathbank_unknown_protein(row)` requires **all four**, and the recorded row
carries all four:

| Predicate clause | Recorded value |
|---|---|
| `pathbank_protein_id == 9659` | `9659` |
| `normalize(name) == normalize("Unknown")` | `"Unknown"` |
| `uniprot.casefold() == "unknown"` | `"Unknown"` |
| `chosen_rule == "pathbank_unknown_protein_fallback"` **or** `cross_species_placeholder is True` | `chosen_rule = "pathbank_unknown_protein_fallback"` |

`orch710_pinned21.json` independently records `"sentinel": true` on it, and
`claims_real: ""` — so it sets no `placeholder_claims_real_identity`, consistent with D-070's
"none of the 21 is a forged identity".

---

## 2. How it was made unique — and the respect in which A4's premise is off

**A4 says to take the row "used by C-100's accepted A/B". C-100's accepted A/B contains no
payload row at all.** REV-100 § "The A/B — zero movers, run twice by two parties" is a
**test-node** A/B:

| Node set | Base | Tip | Delta |
|---|---|---|---|
| SMOKE, 20 files | 473 passed | 473 passed | 0 |
| gold-readers, 22 files | 2 failed / 453 passed / 8 skipped | 2 failed / 453 passed / 8 skipped | 0 |

20 + 22 = the **42 files** of "zero movers on 42 files". That A/B identifies test nodes, not
sentinel rows.

The row-level evidence C-100 does carry names **more than one row**:

* **`03-base-probe` / `04-tip-probe`** (`evidence/g11/C-100/`, stdout preserved only in the
  C-100 session scratchpad logs) enumerate **three** PMC12444477 rows with `sentinel=True`:
  1. `runs/2026-08-02_2130/papers/PMC12444477/strict/final_mapped.json` → `proteins[4]`
  2. `runs_verify/2026-08-24_1428/papers/PMC12444477/strict/final_mapped.json` → `proteins[9]`
  3. `runs_verify/2026-08-25_1216/papers/PMC12444477/strict/final_mapped.json` → `proteins[7]`
* **REV-100 § REGISTRATION 1** says so in words: *"3 sentinel rows across archived legs"*.
* **REV-100's net-effect note** names **two** of those legs, not one:
  `runs/2026-08-02_2130/.../strict` **0 → 9** (7 core enzymes + `LpxH` + `Unknown`) and
  `runs_verify/2026-08-24_1428` **0 → 2**.

So on the charter's literal route the answer is three rows, or two legs — not one.

**The unique row comes from two other committed records, and it is unique under both:**

1. **`evidence/orch710_pinned21.json`** — LEDGER § F-141 CLASSIFIED names it *"the pointer
   file the 16/5 partition was computed against"*. It carries all 21 placeholder rows with
   `sentinel` flags. It holds exactly **five** sentinel rows (matching D-070's partition
   table), and exactly **one** of them is on PMC12444477 — `/entities/proteins/4` in
   `runs/2026-08-02_2130`. The other four are PMC12096016, PMC12180156, PMC12782028 and
   PMC12856317.
2. **LEDGER § F-141 CLASSIFIED, "which run is 'the pinned run'"** — *"It is
   `runs/2026-08-02_2130`, not `runs_verify/2026-08-24_1428`."* That correction exists
   precisely so the next reader does not pick the wrong tree. Of C-100's three sentinel rows,
   exactly one lies in the pinned run.

Both routes select the same single row. That is the identification above.

---

## 3. What this determination does and does not decide

**It fixes the A/B target and the demonstration row.** The other two archived sentinel rows are
used as **preservation controls**, per A4 — never as substitutes.

**It does not bias the behaviour.** The tolerance implemented under D-074 is *row-predicated*:
it evaluates `is_pathbank_unknown_protein(row)` plus the gold's declared sentinel identity on
whatever row it is handed. All three archived PMC12444477 sentinel rows carry identical values
on every predicate clause, so all three are tolerated identically and the choice of A/B target
cannot move an outcome. Had the choice been able to move an outcome, the ambiguity would have
been material and this card would have stopped.

**No live run was launched to settle any of this**, per A4 and D-074. Every fact above is read
from a committed artifact or from the preserved stdout of a C-100 bounded job.

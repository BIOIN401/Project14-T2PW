# F-161 — HELD FOR THE LEAD TO SEQUENCE (not yet written into `FINDINGS.md`)

**Status at the time C-113 reached this step:** `card/C-112-residual-sweep` is **NOT merged**
into `sprint/pwml-recovery` (`git merge-base --is-ancestor` says no), and its diff **does edit
`docs/pwml_recovery_sprint/FINDINGS.md`** (`+16 -…` on that file). C-113's charter § 4 says that
in exactly this situation the entry is to be **held for the Lead to sequence** rather than
written, so that two unmerged branches do not both rewrite the same file.

**So `FINDINGS.md` is deliberately UNTOUCHED by this branch.** The finished entry is below,
verbatim, ready to be appended after `F-160` once C-112 lands. Nothing else in C-113 depends on
it.

---

## F-161 — the gold-readers selection is not a superset of SMOKE, so a gold edit's mandated gate was structurally blind

- **Severity** HIGH for the sprint's own instruments · **Class: defect in the REVIEW INSTRUMENT**
  — the criteria were incomplete; **the reviewer is not at fault** · **Registered 2026-09-01 (C-113)**
- **Found by** the failed merge of F-150 half 1 at `b05a7281`, and measured by the Lead's A/B:
  `evidence/orch718_smoke22_postf150.log` / `orch718_smoke22_postrevert.log`, G11 `ORCH-718/04`
  and `/05`.

### The measurement

REV-F150's mandated gate for the gold edit was the **22-file gold-readers selection**. It ran the
four-step A/B honestly and got a **byte-identical `456 passed / 8 skipped / exit 0` in both arms**,
with a per-file delta of zero on all 22 files, and returned **VERIFIED — APPLY HALF 1**.

`tests/test_c102_coverage_denominator.py` **reads the gold** — it builds
`{case.paper_id: case for case in load_gold_set(pinned_gold_set_path()).cases}` at module scope —
and it **is in SMOKE**. It is **not in the gold-readers selection.** So the gate that was mandated
for a gold edit could not see the only two tests that edit actually moved:

```
WITH the gold edit     b05a7281   501 passed / 2 failed   exit 1
WITHOUT the gold edit  700c9434   503 passed / 0 failed   exit 0
```

Both failures were in that one file. **A real consequence sat exactly one selection away from the
gate written to find it,** and the arms agreed to the test because neither arm ran the file.

### Why this is a defect in the instrument, not in the reviewer

**The reviewer did precisely what its criteria asked, and did it correctly.** Its A/B was sound:
same tree, same interpreter, predictions written first, one failed measurement kept. The verdict
it returned is still correct — REV-F150 is not reopened, and the gold edit re-lands unchanged at
C-113 with the byte-identical blob `36f4b7b6…`. What failed is the **choice of population** the
criteria named as sufficient.

**This is the exact mirror of the standing lesson that SMOKE does not cover the gold readers.**
The sprint already knew one direction of that gap. The other direction is just as real:

> **Neither selection is a superset of the other. A gold edit needs BOTH.**

### The consequence, stated plainly

A gate that is *mandated* rather than *chosen* carries the authority of the process. When such a
gate is blind by construction, a green result from it is read as a licence to merge — and it was:
the merge went in, SMOKE caught it, and merge rule 10 required the merge not to stand. **Merge
rule 10 was the only thing between an instrument gap and a landed regression-shaped tip.**

### Disposition

**Effective immediately, for any change to `src/t2pw/bench/gold/pinned_v1.json`:** run **BOTH**
the 22-file gold-readers selection **and** SMOKE, and report both. Neither alone certifies a gold
edit.

**RAISED, NOT ANSWERED — for the Lead:** should the gold-readers selection be *extended* to
include every SMOKE file that reads the gold (starting with
`tests/test_c102_coverage_denominator.py`)? That is a **`TEST_MATRIX` change with its own cost** —
runtime, per-branch obligations, and a moved `456 / 8` baseline that every future A/B is measured
against. It is **not C-113's to make**, and C-113 wires no file into any selection or chunk.

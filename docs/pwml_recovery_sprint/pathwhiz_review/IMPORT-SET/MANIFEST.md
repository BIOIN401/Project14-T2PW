# PathWhiz import set — the files the product owner imports by hand

**2026-09-08, `ORCH-735`.** Eleven PWML files, copied byte-for-byte out of their run trees and
**committed**. That is the point of this directory: until now every one of them existed as a single
untracked copy on one disk — the `F-187` exposure `ORCH-734` § 7 flagged and asked to have fixed.
They are now on `origin` as well.

**Nothing here was modified to make an import succeed.** Copies only. Every file's sha256 is below
and matches its source. If an import fails, the file is the evidence; do not repair it.

Import readiness re-verified independently, not taken from the `ORCH-734` report:
`evidence/orch734_pathwhiz_import_check.py` over all eleven. **9 READY · 2 FAIL.** G11 cleanup
report `evidence/g11/ORCH-735/01-pwml-inventory.json`, `FINAL SURVIVING COUNT : 0`.

---

## The recommended import set — nine files

Import these. They pass every check the structural validator can decide from the bytes.

| file | source | bytes | cpd | prot | cplx | rxn | edges | sha256 (first 16) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `ORCH734_PMC10055903.pwml` | `runs_smoke/2026-09-08_1528` | 55,011 | 9 | 4 | 4 | 4 | 18 | `8a06b4edc980f94d` |
| `ORCH734_PMC10269868.pwml` | `runs_smoke/2026-09-08_1528` | 48,436 | 5 | 5 | 4 | 5 | 16 | `3d1c0e3b3695da72` |
| `ORCH734_PMC7402084.pwml` | `runs_smoke/2026-09-08_1528` | 40,294 | 6 | 4 | 4 | 3 | 12 | `368f9e7ea0ff673d` |
| `ORCH734_PMC8211424.pwml` | `runs_smoke/2026-09-08_1528` | 22,859 | 4 | 1 | 1 | 2 | 6 | `e893778e21718dea` |
| `ORCH734_PMC9544450.pwml` | `runs_smoke/2026-09-08_1528` | 49,436 | 8 | 2 | 2 | 6 | 17 | `cb209a921cc224da` |
| `ORCH732_PMC12051542.pwml` | `runs_smoke/2026-09-07_2323` | 80,518 | 11 | 5 | 6 | 8 | 29 | `44cf838f1a4d6ced` |
| `ORCH732_PMC4725005.pwml` | `runs_smoke/2026-09-07_2323` | 54,459 | 9 | 4 | 4 | 5 | 16 | `6a54fbda584bc358` |
| `ORCH732_PMC9544450.pwml` | `runs_smoke/2026-09-07_2323` | 39,686 | 8 | 2 | 2 | 4 | 11 | `679c6f028fcd69da` |
| `C120_PMC10031235_PSAT.pwml` | `runs_validation/c120/2026-09-08_1240` | 48,401 | 8 | 2 | 2 | 3 | 15 | `6f17819ce7ed1f48` |

**`PMC9544450` appears twice on purpose.** The same paper on two different draws — 6 reactions in
`ORCH-734`, 4 in `ORCH-732`. Keeping both is the honest record of a draw-dependent pipeline, and the
pair is worth importing together for exactly that reason.

## The two `F-195` files — import them LAST, and the result decides a card

| file | source | bytes | why it fails |
|---|---|---:|---|
| `ORCH734_PMC11016064.F195.pwml` | `runs_smoke/2026-09-08_1528` | 107,951 | `no_broken_references` — `transport-compound-visualization` names `compound-location-id` 66 and 77, neither declared |
| `ORCH734_PMC6112128.F195.pwml` | `runs_smoke/2026-09-08_1528` | 78,492 | `no_broken_references` — names `compound-location-id` 55; the document declares 88 to 101 |

**Both exported successfully and both count as `ORCH-734` successes.** `F-195` is an
import-validity defect, invisible to every pipeline gate, and it is **not** part of `C-121`.

Their outcome is what settles the question, and the rule was written before the test:

* **PathWhiz accepts and renders them** → `F-195` is a limitation of our structural checker.
  Document it and close it. No code.
* **PathWhiz rejects them, or renders them broken** → `F-195` is a genuine importability defect and
  may justify one separate narrow card.

**No pre-emptive `F-195` engineering either way.**

## What "IMPORT READY" does and does not mean

`IMPORT READY` is **not** `IMPORT PASS`. The checker decides what the bytes can decide — the XML
parses, the visualization envelope is present, the canvas is non-zero, elements are positioned,
edges carry real paths, compounds and proteins are declared and placed, every reaction has both
sides and its compound references resolve, and species carry a numeric taxonomy id.

It cannot decide what only the live importer can: whether Rails accepts the payload, whether
reference-database lookups resolve against the real instance, or whether the rendered pathway is
biologically legible to a curator. **That judgement needs the product owner's account and is the
one reliability check no automated task in this sprint has been able to perform.**

## Provenance note, so a later reader is not misled

Two byte counts in the `C-121` pre-dispatch measurement look like they disagree with this table and
do not. That replay injected its own pathway name and description, so
`ORCH734_PMC9544450` came out 49,464 bytes there against 49,436 here. **A 28-byte metadata
difference, not a content difference** — the reaction, compound, protein and edge counts are
identical. The replay outputs are deliberately **not** committed and must never enter this
directory: they are archived-payload replays, not production deliverables.

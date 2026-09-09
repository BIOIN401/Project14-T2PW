"""One-shot reconciliation of docs/change_log.md with the C-121 pin move.

Not a general tool: it exists so the edit is reviewable as a script rather than as
an opaque in-place sed, and it refuses unless each target text appears exactly once.
"""
import io

p = r"C:/t/c121/docs/change_log.md"
s = io.open(p, encoding="utf-8").read()

old1 = "9/28 required contract; 9/9 IR; 9/28 exportable; by leg 19/19/4, by row 27/27/4)."
new1 = "9/28 required contract; 9/9 IR; 9/28 exportable; by leg 19/19, by row 27/27)."
assert s.count(old1) == 1, s.count(old1)
s = s.replace(old1, new1, 1)

old2 = """The 19 failures are one class, and quarantine is the wrong place to fix it:
`species_missing_classification` (19 legs, 27 rows), `species_missing_taxonomy`
(19 legs, 27 rows), `no_biological_states` (4 legs, 4 rows). The gate wants a numeric"""
new2 = """The 19 failures are one class, and quarantine is the wrong place to fix it:
`species_missing_classification` (19 legs, 27 rows), `species_missing_taxonomy`
(19 legs, 27 rows). The gate wants a numeric"""
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, new2, 1)

old3 = """`test_stage_three_recovery_is_not_strict_exportability` pins both halves and will
fail if the gap closes upstream, so the claim cannot drift from the measurement."""
new3 = """`test_stage_three_recovery_is_not_strict_exportability` pins both halves and will
fail if the gap closes upstream, so the claim cannot drift from the measurement.

> **C-121 / F-192 removed a third code from this list, 2026-09-08.**
> `no_biological_states` was `(4 legs, 4 rows)` here. `quarantine_and_close` now
> re-establishes the compartment placeholder after the closure loop converges, so a
> payload that still carries exportable content no longer reaches the required-field
> gate with zero biological states. **No leg's verdict moved:** all four also carry
> `species_missing_*`, so `required contract` stays 9 / 28 and `fully exportable`
> stays 9 / 28 — one error code stopped being emitted, and nothing was exported that
> was not exported before. The species-metadata class is untouched at 19 legs / 27
> rows and remains the next pipeline defect."""
assert s.count(old3) == 1, s.count(old3)
s = s.replace(old3, new3, 1)

io.open(p, "w", encoding="utf-8", newline="").write(s)
print("OK")

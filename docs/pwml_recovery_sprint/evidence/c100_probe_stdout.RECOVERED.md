# Recovered: C-100's `03-base-probe` / `04-tip-probe` stdout

Recovered by **C-101** on 2026-08-28 from the C-100 session scratchpad
(`AppData/Local/Temp/claude/.../29355bf3-.../scratchpad/`), which is outside the repository
and one cleanup from deletion.

`bounded_run.py`'s JSON report carries **no child stdout** by design, so
`evidence/g11/C-100/03-base-probe.json` and `04-tip-probe.json` certify that the jobs ran
bounded and clean while preserving **nothing about what they found**. That is F-140's
class, and F-141's ruling already recorded it once: *"A certificate that a job was clean is
not a record of what the job found."* This is the second time in this wave the only record
of a bounded job's findings lived outside the repo.

**C-101 relied on these two logs**, so they are committed here rather than cited from a
temp directory. What they carry that nothing else does:

* the enumeration of **three** PMC12444477 `Unknown` rows with `sentinel=True`, across
  three archived legs — the fact behind C-101's A4 determination and behind REV-100
  § REGISTRATION 1's *"3 sentinel rows across archived legs"*;
* the base-vs-tip excusal tables, which show `'Unknown' -> True` at C-100's base and
  `'Unknown' -> False` at its tip.

The probe **source** (`probe_c100.py`) was already gone from that directory when C-101
looked; only the logs survived. Recorded so nobody assumes the source can still be found.

See `c101_a4_authoritative_row.md` for what was concluded from them, including the respect
in which C-101's charter mis-described C-100's accepted A/B.

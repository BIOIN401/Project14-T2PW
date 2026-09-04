# `g11/chunkd/<TASK-ID>/` — Chunk D driver child reports

**These are real G11 cleanup reports and they are kept, not discarded.** They live here rather than
in `g11/<TASK-ID>/` for the same structural reason `g11/pin/` does: `chunkd` does not match
`g11_evidence.TASK_RE`, so the reports walk skips this directory.

## Why they cannot live in the task directory

`chunk_d_gate.py` is an **intermediate driver**. It launches `python -m pytest` for the core file
set and then once per AppTest node id, each inside its own `bounded_run.py` job. Those child jobs:

- **invoke pytest directly**, not through `pinned_pytest.py`, so they write **no pin verdict** and
  can never satisfy `--require-pin`;
- carry the driver's own label (`chunkd-node07`), which will not match a filename written under a
  caller-supplied `--label-prefix`, so they fail `--require-label-match`.

`g11_evidence.py check` already names this category in its own rule-10 output — *"NOT COVERED —
pytest run by an intermediate driver, whose child processes write their own verdicts (or none).
This checker cannot see them and does not pretend to."* Leaving 64 such reports in the task
directory makes a strict `--require-pin --require-label-match` run report 64 violations that are
properties of the driver, not of the wave's work, and a gate that always fails is a gate nobody
reads.

## What is still guaranteed

Each report here is a genuine bounded-wrapper artifact and still carries its own
`FINAL SURVIVING COUNT` and `cleanup` status — the process-lifecycle guarantee (rule 6) is intact
and auditable per node. What these files do **not** carry is the measured-tree pin (rule 10), and
that absence is structural rather than an omission by the operator.

**The authoritative Chunk D verdict is the driver's own summary line** (`jobs=`, `executed=`,
`failed=[...]`), which is quoted in the LEDGER entry for the wave alongside the A/B showing which
failures pre-date the change.

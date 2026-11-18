# V7a evidence: not yet produced

The heavy-lock gate printed GATE_BLOCKED twice (free RAM 0.4 to 1.0 GB, floors 2 GB for tests and 4 GB for the proof).
Per the 2026-11-17 addendum no further retry was made. Nothing in this folder is a result.

Still to run (full commands in `docs/vision/results/V7a.json`, `commands_still_to_run`):

1. `tests/test_vision_checks.py` (focused, floor 2 GB).
2. `python -m t2pw.vision.checks.run_checks --runlist C:/t/ev-cm/runlist.json --idx ok_idx.txt --scopes paper,default`
   on the 97 runs with PWML (floor 4 GB), writing `totals97.json` here (totals plus up to five examples per failure class).

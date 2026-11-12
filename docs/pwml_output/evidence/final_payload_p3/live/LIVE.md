# Final payload phase 3: live runs per mode

Status: BLOCKED. No live run was started. Tree C:/t/fp, branch pwml/final-payload, HEAD 3e782d76.

## Blocker (credentials)

- The worktree C:/t/fp has no `.env`. `t2pw/config.py` and `t2pw/llm/client.py` load `PROJECT_ROOT/.env`, and `PROJECT_ROOT` is the worktree.
- `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` and the PathBank DB host variables are unset in the process environment (presence checked only; no values read).
- Copying the main checkout's `.env` into the worktree (the way E2E-1 did) was refused by a deny rule: "cp ... .env was blocked by a deny rule". Per RULES.txt this step stops; no workaround was tried.
- To unblock: the product owner places a `.env` in C:/t/fp (gitignored) or exports the variables, then runs the three commands below.

## Prepared, not run

Driver: `drive.py` (adapted from docs/webserver/evidence/E2E-1/drive.py; only change is NAME/MODE/PMID/PATHWAY/ORG arguments and output root C:/t/fpl/NAME). It goes JobStore -> Worker -> driver_bridge.run_pathway -> batch driver. Wrapper script: `run1.sh` (bounded_run.py, `--timeout 3000`, heavy lock `fp`, `T2PW_FINAL_PAYLOAD=1`, `PYTHONPATH=C:/t/fp/src`, `T2PW_OFFLINE_CURATOR` unset). Note the worker's per-attempt cap is 3600 s, so the 3000 s wrapper bound could kill a long run before a timeout retry; raise it if a run is slow.

Modes in the code (driver_bridge.py): export modes `strict` (PathWhiz mode) and `research`; name-only mode exists (pathway name and organism, no PMID, one fetched paper).

Input chosen: PMC12782028, PMID 41522513, "cholesterol biosynthesis", Homo sapiens. Earlier strict legs passed 7 times with 3 to 4 PWML reactions in 590 to 1136 s (runs_verify manifests). PMID 32185015 was rejected because every earlier run on it gives 0 reactions.

Commands, one at a time, each run in the background without `&`:

    bash C:/t/fp/docs/pwml_output/evidence/final_payload_p3/live/run1.sh pathwhiz strict 41522513 "cholesterol biosynthesis" "Homo sapiens"
    bash C:/t/fp/docs/pwml_output/evidence/final_payload_p3/live/run1.sh research research 41522513 "cholesterol biosynthesis" "Homo sapiens"
    bash C:/t/fp/docs/pwml_output/evidence/final_payload_p3/live/run1.sh nameonly strict - "cholesterol biosynthesis" "Homo sapiens"

Then per run: read `C:/t/fpl/<name>/report/job_result.json`, check the PWML for `final_payload_fill` provenance, title and PMID reference, import in the Docker harness (project t2pw-pwi, IMPORT-HARNESS.md), stop the project.

## Results

| mode | exit | wall | crash | PWML | final_payload_fill | title + PMID ref | import |
|---|---|---|---|---|---|---|---|
| PathWhiz (strict) | not run | | | | | | |
| research | not run | | | | | | |
| name-only | not run | | | | | | |

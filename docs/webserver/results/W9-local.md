# W9-local: T2PW web server running on the development PC (Windows)

Branch `deploy/local` (from `webserver/refactor` @ `893257b0`, which contains `main` `fd814b8e`).
Code commit `eb329451`. Operator: general-purpose agent for the orchestrator, 2026-11-19.
PO choice: "this PC first"; a real lab host comes later (W9 proper).

## What runs

| | |
|---|---|
| URL (this PC) | http://localhost:8800 |
| URL (LAN) | http://192.168.1.232:8800 (Wi-Fi, DHCP: the address can change; `T2PW_BASE_URL` in `start.sh` then needs updating) |
| Checkout | `C:/t/t2pw-server` (own worktree: a real run rewrites tracked cache and output files). `.env` copied in by the PO (gitignored) |
| Python | primary checkout's `.venv` (3.13.6), `PYTHONPATH=src;deploy/local`, `PYTHONUTF8=1` |
| Data dir | `C:/t/t2pw-server-data/`: `jobs.sqlite` (queue, rate limits, notifications), `work/item-N/attempt-M/` (job folders), `curator_queue/`, `logs/`, `rate_key_secret`, `pids` |
| Logs | `C:/t/t2pw-server-data/logs/uvicorn-<stamp>.log`, `worker-<stamp>.log`, and their `*.bounded.json` wrapper reports; per job `work/item-N/attempt-M/child.log` and `bounded_run.json` |

Start (Git bash, stays attached; from Claude Code use a background Bash with NO `&`):
`"C:/Program Files/Git/bin/bash.exe" C:/t/t2pw-server-data/start.sh`.
**Who owns it:** the server lives exactly as long as the shell that ran `start.sh`. Started from
a subagent's background Bash, it was killed when the subagent returned (14:20, about 2.5 h after
start): both wrappers died without writing their reports, no process survived, port freed, no
heavy lock left. Start it from a session that stays open: the PO in a Git bash window, or the
orchestrator's own background Bash.

Stop: `"C:/Program Files/Git/bin/bash.exe" C:/t/t2pw-server-data/stop.sh`. It kills only the two
PIDs `start.sh` recorded (with their trees). If that kill caught a job mid-run, it clears the heavy
lock only when the lock names `W9-local-job` and that holder's PID is gone.

## How it is wired (no change under `src/`)

`deploy/local/t2pw_deploy.py`:

* `create_app` (uvicorn `--factory`): the stock app with the deployment caps. It sets
  `limits.RATE_LIMIT_SUBMISSIONS` and `jobs.MAX_QUEUED_ITEMS` before building the app, so the
  limits the site shows are the ones it enforces.
* `run_pathway` (worker `--run-fn`): runs the real `t2pw.service.pathway:run_pathway` as its own
  process under `bounded_run.py --heavy-lock W9-local-job` (lock `C:/t/heavylock`). While the
  lock is held elsewhere it waits (exit 95, retry every 30 s). The inner timeout ends 120 s before
  the worker's deadline, so the wrapper releases the lock rather than a hard Job Object kill.
  It gives up as a timeout if less than 10 min is left after waiting.

`uvicorn` (0.0.0.0:8800) and the worker each run under `bounded_run.py` without the lock, 24 h
timeout. After 24 h both stop; run `start.sh` again.

## Limits (approved by the PO)

* 2 submissions per visitor (IP) per 24 h (stock default 5). Verified: a 3rd POST got 429,
  `Retry-After` 86086 s, "You have reached the limit of 2 submissions per day."
* Queue cap 5 open pathways (stock 20). Up to 5 pathways per submission (stock, unchanged).
* One job at a time (one worker); each job waits for the PC-wide heavy lock.
* Run limit 90 min per pathway, 2 retries for crash/LLM/network kinds (stock W4 policy).
* `T2PW_RATE_KEY_SECRET`: random 32 bytes in `C:/t/t2pw-server-data/rate_key_secret` (not in git).

## Checks

* Home page 200 on `localhost` and on `192.168.1.232`. Page says "2 submissions per day per visitor".
* PathBank DB via the app's own resolver (`PathBankDbResolver.from_env`, `.env` settings):
  connected, `pathways` = 537,623 rows. No setting printed.
* Firewall: Private profile on, Wi-Fi is Private. The socket is owned by the venv's base
  interpreter `C:\Python313\python.exe`, which already has an inbound Allow rule on Private, so
  the LAN should reach it. Not proven from a second machine (a request to the LAN address from
  this PC loops back). If a LAN device cannot connect, from an admin PowerShell:
  `New-NetFirewallRule -DisplayName "T2PW 8800" -Direction Inbound -Protocol TCP -LocalPort 8800 -Action Allow -Profile Private`
* Heavy-lock adapter: job 2 found the lock held by another session's `loc-research` job, waited,
  then ran. Every job wrapper: lock acquired and released, 0 surviving processes.

## Test jobs (real form POSTs, PMID 40400396 = PMC12096016, enterobactin, *Escherichia coli*)

| # | Job | Result | Wall time |
|---|---|---|---|
| 1 | `ff4b592d...` (item 1), localhost | Failed: "no open-access full text found for PMID 40400396" (finding 1) | 56 s |
| 2 | `b7ff128e...` (item 2), localhost | Same failure; first waited ~2 min for the heavy lock held by another session (`loc-research`) | 125 s |
| 3 | `87aab82d...` (item 3), via the LAN address, submitted when Europe PMC answered in < 5 s | **Review required**, one attempt. Strict gates passed, semantic evaluation passed. 5 reactions, 2 transports, 8 proteins, 9 compounds, 38 edges; 0 gate errors. PWML 101,295 bytes. All 23 download links 200 with content | **20.3 min** (submit to terminal 1216 s; pipeline 1200.6 s) |

Job 3 wrapper: exit 0, heavy lock acquired and released, 0 surviving processes. A real run
rewrites tracked files in `C:/t/t2pw-server` (`data/enrichment_cache.json`,
`data/id_mapping_cache.json`, `out/`, `outputs/`, `tmp/`), as expected. Never commit them from
this checkout: use pathspec commits only.

## Findings

1. **A slow Europe PMC fails every PMID job for good, with the wrong message.** Europe PMC's
   `fullTextXML` was answering in 16-28 s (and 503 in places) during this session. The fetch uses
   `HttpClient` with a 15 s timeout (`src/t2pw/mapping/map_ids.py`), so it returns "" and the
   job ends `fail` / `no_reactions` with "no open-access full text found ... please upload the
   PDF", which is final (content kind, no retry). The text is open access; the right outcome is a
   retryable network failure, or a longer timeout for full text. The fix is under `mapping/`
   (off-limits to this track): open question for the PO. Related to the R5 caller note in
   `service/inputs.py:resolve_pmids`.
2. The 3rd test went through the LAN address, which counts as a separate visitor for the rate
   limit (127.0.0.1 had used both slots on the two failed runs).

## Open W9 gaps (for the lab host)

* HTTPS and a reverse proxy: not done locally (plain HTTP on the LAN). With a proxy, set
  `T2PW_TRUSTED_PROXIES` and the proxy body cap.
* SMTP: unset. Every job's email is recorded `not_sent` permanently.
* `NCBI_EMAIL` (and `NCBI_API_KEY`): set, but only because the developer's `.env` was copied
  (checked as booleans through the app's own loader; values not read). The lab host needs a
  lab or service address of its own.
* No LLM spending cap and no global daily job cap: only the per-IP limit, the queue cap and one
  job at a time (about 28 runs/day at ~50 min each).
* No disk quota or cleanup for `work/` and uploads.
* Read-only server DB user: this PC uses the developer's `.env` credentials (OpenRouter key too).
* Linux bits not exercised: POSIX worker kill path, `systemctl stop`, systemd units.
* The site still shows the citation and lab-credit placeholders.

## Open questions

* **Q-W9L-1 (PO):** Europe PMC slowness (finding 1). (a) Map an empty full-text fetch after a
  timeout to a retryable network failure. (b) Raise the full-text timeout (e.g. 60 s). (c) Both.
  Either needs a change under `mapping/` or `service/inputs.py`, outside this track's file
  boundary.

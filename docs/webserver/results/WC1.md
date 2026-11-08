# WC1: carry-forward wording fixes from the W4b review

Branch `ws/WC1-carry-copy`, base `71cfe7d3`. Comments and user-facing copy only.

## Changes
- F3 `src/t2pw/web/jobs.py`: backlog comment now states typical 7-20 h and the worst case 20 x 90 min = 30 h.
- F4 `src/t2pw/web/retry.py`: `MIN_RETRY_BUDGET_S` comment now says the 15 min budget sits below the ~18 min median run, so a retry started on it often times out. Constant unchanged.
- F7 `src/t2pw/web/templates/help.html`: comma splice split into two sentences; "90-minute limit".
- "20 to 60 minutes" kept: `tests/test_web_app.py:121` pins it.

## Tests
13-file web+bridge list through `bounded_run.py --heavy-lock WC1`: see `evidence/WC1/web_output.txt`, exit 0, 0 survivors.

## Open questions
- Q-WC1-1: `MIN_RETRY_BUDGET_S` (15 min) is below the 18 min median, so a retry started at the floor will usually time out. Options: leave (a retry can still fail fast on a crash/network kind and succeed on a quick run), or raise to ~20 min (fewer wasted retries, but fewer retries overall). Not changed here.

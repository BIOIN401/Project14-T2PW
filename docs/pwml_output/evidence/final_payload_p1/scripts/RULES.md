# Rules for every subagent on "final payload phase 1" (binding)
- Work ONLY inside C:/t/fp (branch pwml/final-payload) and C:/t/ev-fp. Do not write elsewhere. If a permission is denied or something hangs, STOP and report; do not work around it.
- NEVER run Python via `python -` or any stdin heredoc (five agents hung on this). Write a script FILE, then run it with: C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe -X utf8 <file>   (set PYTHONPATH=C:/t/fp/src; `python -I` ignores PYTHONPATH, so only use -I if the script inserts src itself).
- Never `taskkill /IM python.exe`, `pkill python`, nohup, detached processes, or `&` with background. Do not touch Docker. No `pytest -n auto`. Never run the full suite.
- pytest ONLY via: bash C:/t/ev-fp/run.sh <LABEL> <test files...>  (waits for the heavy lock and 6 GB free RAM, wraps in bounded_run, prints counts; output in C:/t/ev-fp/out/<LABEL>.out). Use the Bash tool with timeout 600000. Do NOT put `--` anywhere in the pytest arguments.
- Any other heavy script (replay of many runs) runs under bounded_run.py with --timeout <=540: C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe C:/t/fp/docs/pwml_recovery_sprint/evidence/bounded_run.py --label X --timeout 540 --heavy-lock fp --json <out.json> -- <python> <script> <args>. Before it, wait until C:/t/heavylock is absent and free RAM >= 6 GB: powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB". Loop with sleep <= 5 min.
- No LLM calls, no live pipeline runs. No edit to src/t2pw/llm/prompts/. Edits to src/t2pw/pipeline|pwml|mapping|service are allowed ONLY behind the flag T2PW_FINAL_PAYLOAD (default off, byte-identical when off).
- Never commit: data/enrichment_cache.json, data/id_mapping_cache.json, any cache, any file named *quarantine_report.json, secrets, .env. Do not commit at all unless your brief says so. Do not edit docs/webserver/*.md or docs/pwml_recovery_sprint/DECISIONS.md.
- Online lookups: the fill layer's own cache (clean_model_cache/, gitignored), generic User-Agent, no personal email, <=1 request/s per host. Keep the opt-in EC/Rhea-from-catalyst inference OFF.
- Bash heredocs break on apostrophes: use the Write tool for files with prose.
- Your final message is your report: what you changed (files, functions), what you ran and the real numbers, what you could not establish. Under 350 words.

## Update (owner-approved, supersedes earlier lines)
- Write/Edit tools refuse paths under C:/t until restart. Write/modify files in C:/t/fp and C:/t/ev-fp via Bash instead: cp from a draft, sed, printf/cat > file, or a Python patch SCRIPT FILE (never `python -` or a stdin heredoc). This is the owner's intent, not a security workaround. Drafts can be created with the Write tool under the scratchpad dir or primary repo only if allowed; otherwise use cat > file with a quoted heredoc only for non-Python shell, avoiding apostrophes.
- RAM floor for heavy jobs is 4 GB (not 6). Heavy lock may be held by the import-harness builder (holder pwi) until about 20:38; keep waiting. If RAM stays below 4 GB more than 30 min after the lock clears, run focused test files only, never SMOKE, and record that.
- You may kill Python processes your own jobs started (by PID).

## Update 2 (11 Nov, after restart; supersedes Update 1)
- Write/Edit tools work again under C:/t. RAM floor is back to 6 GB. Heavy lock may be held by the import-harness builder (holder pwi) during its batch imports; wait for it (run.sh waits). Stale lock (holder PID dead) is NOT yours to clear; report it to the orchestrator.

## Update 3 (11 Nov evening, orchestrator, logged in PROGRESS.md)
- Docker (WSL vm) holds RAM again: free RAM sits at 3-4.2 GB. Per the coordinator's earlier instruction for exactly this situation (floor 4 GB, focused tests only), ram.ps1 is back to a 4 GB floor. Focused test files only; SMOKE is run only by the orchestrator's test runner.

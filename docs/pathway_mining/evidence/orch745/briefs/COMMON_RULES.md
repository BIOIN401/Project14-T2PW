# ORCH-745 implementer rules (apply to every change set)

You implement ONE change set for ORCH-745 ("make the experimental pathway-mining feature work"),
working ONLY in your own git worktree on your own branch. The product owner is asleep: never ask
questions, never wait for approval. If something in your brief is impossible, do the rest, and
report exactly what you could not do and why.

## Hard boundaries (violating any of these is a reject)
- Allowed to edit: `src/t2pw/mining/**`, `tests/test_mining_*.py`, `tests/helpers_mining.py`,
  `tests/fixtures/mining/**`, `docs/pathway_mining/**` — in YOUR worktree only.
- Forbidden: every other file. In particular `src/t2pw/rag/**`, `src/t2pw/llm/**`, `src/t2pw/app/**`,
  `src/t2pw/pipeline/**`, `docs/pwml_recovery_sprint/**` (TEST_MATRIX, DECISIONS, LEDGER).
  You may READ and CALL production code (e.g. `t2pw.rag.acquire` helpers) but never edit it.
- **Never weaken a biological gate.** Recombinant / cell-free evidence stays non-admissible, an
  ortholog never transfers species, uncited values stay dropped. Where in doubt, fail closed.
- No live network calls, no LLM calls, no API spend. Tests are offline and deterministic.

## Unattended-run traps (a permission prompt hangs you forever, silently)
- NO recursive deletes of any kind (`rm -r`, `rm -rf`, `Remove-Item -Recurse`, `shutil.rmtree`).
  Use a fresh, never-used directory name instead of cleaning one.
- Write ONLY inside your worktree and your scratch dir `C:/t/bt/<your-name>/`. Never `cd` into,
  copy into or write into any other worktree (including `C:/t/mining`, `C:/t/mining-rep-base`).
- Never `git push`, `git merge`, `git rebase`, `git stash`, `git reset --hard`, `git checkout <other branch>`,
  `git worktree ...`. Read other branches only with `git show <branch>:<path>`
  (e.g. the pilot records on `experiment/pathway-mining-pilot-002` / `-003`).
- Never `taskkill /IM`, `pkill`, or kill processes you did not start.

## Running tests (every pytest run, no exceptions)
Python: `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe` (call it PY).
From your worktree root W:

```
mkdir -p C:/t/bt/<name>/runs/<unique>
cd W && PYTHONPATH=W/src PY docs/pwml_recovery_sprint/evidence/bounded_run.py --label <l> --timeout 400 \
  --json C:/t/bt/<name>/runs/<unique>/report.json -- \
  PY -u docs/pwml_recovery_sprint/evidence/pinned_pytest.py \
  --pin-verdict C:/t/bt/<name>/runs/<unique>/pin.json -q -p no:cacheprovider \
  --basetemp=C:/t/bt/<name>/runs/<unique>/bt <test files or node ids>
```
- `PYTHONPATH=W/src` is required (else exit 98). `pinned_pytest.py` takes NO `--` separator.
- A fresh `<unique>` directory per run (create the parent first; never reuse or delete one).
- Give the Bash tool `timeout: 480000` for a 400 s wrapper. Never append `&`. Never `pytest -n`.
- Check the wrapper prints `FINAL SURVIVING COUNT : 0` (or the JSON `final_surviving_count: 0`).
- Do NOT run the production SMOKE suite; the orchestrator does. DO run the full mining suite
  (`tests/test_mining_*.py`, all files) before your final report and give the pass count.
- Baseline at `f9556288`: mining suite **281 passed**.

## G9 (proof obligation for every correction)
- For each defect you fix, add at least one test named `test_g9_745<cs>_<defect>_*` that FAILS
  BEHAVIOURALLY at the base SHA `f9556288` and passes at your tip: a wrong value / failed assertion
  through code paths that exist at base. An ImportError / AttributeError / NameError / missing
  symbol at base is NOT proof. Design these tests to call functions that exist at base
  (`classify_evidence_type`, `PassageIndex.add_paper/retrieve`, `build_claims`, `BudgetLedger`,
  `frontier.generate`, the CLI `main`, `MiningModelClient.call_json`, `MiningController`, ...).
- A genuinely new helper may additionally get tests labelled `test_new_acceptance_745<cs>_*`.
- You do NOT run the base arm (the orchestrator does, in its own base worktree). Write
  `C:/t/bt/<name>/g9_manifest.json`: a list of `{"node": "tests/<file>::<test>", "defect": "...",
  "expected_base_failure": "<what assertion fails at f9556288 and with what value>",
  "fixtures": ["tests/fixtures/mining/..."]}`. Each node must be runnable at base by copying in
  only its test file plus the listed fixture files (so keep G9 tests in a file that imports only
  what exists at base, or guard new imports inside the non-G9 tests).

## Commits
- Write the message to a file under `C:/t/bt/<name>/`, `git add <explicit paths only>`,
  `git commit -F <file>`, then `git log -1 --stat` to verify the commit exists (heredoc commits
  silently no-op here). One commit per defect id where practical.
- Subject style: `experiment: ORCH-745 <defect id> - <what>`.
- Every message ends with the line:
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`
- Never commit caches, `.env`, `data/`, run output, or anything outside the allowed paths.

## Docs
Update the affected `docs/pathway_mining/*.md` (policy docs especially) so they describe the
behaviour your change set actually ships. Keep the existing tone: plain statements, real names.

## Code style
Read the module you change first and match it: docstrings that explain WHY, `ponytail:` comments
for deliberate simplifications with a known ceiling, no speculative abstraction.

## Final report (your last message; it is all the orchestrator sees)
1. Branch, tip SHA, and every commit (sha + subject).
2. Files changed (`git diff --stat f9556288..HEAD`).
3. Per defect: what you changed, where (file:function), and the design decisions you made.
4. Tests added (names) and the full mining-suite count from the wrapper output, with the run's
   `final_surviving_count`.
5. G9 manifest path and its node list.
6. Anything you could not do, deviations from the brief, and concerns a reviewer should check.

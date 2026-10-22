# ORCH-756 — the 7 strict cohort legs at 1 draw with the D-105 budgets: PWMLs for hand comparison

**Status:** DONE (stopped by the product owner's reprioritization, 2026-10-22 10:35). **2 of 7 papers have a PWML
at 1 draw: enterobactin and menaquinone** (`pathway.review_required.pwml` each). Chlorophyll timed out (5,400 s,
phase 1); Ara operon stopped at Stage 0 `scope_conflict`; **sour taste, WNT PCP, Wnt/Ca2+ and the chlorophyll retry
were DEFERRED by the product owner, not failed** (ORCH-759, 3 draws, goes next; the 1-draw legs run only if the
product owner asks later).
Decision: **D-105 § 3(a)** (product owner, 2026-10-21). Sub-orchestrator record.

## 1. What runs, and from where

- The 7 strict conditions of ORCH-755 (topics `docs/evaluation_v2/paper_runs_orch755/topics/`): enterobactin,
  menaquinone, chlorophyll, Ara operon, sour taste, WNT PCP, Wnt/Ca2+.
- From a dedicated worktree `.claude/worktrees/orch756`, whose own `.env` is a copy of the primary's with
  **`T2PW_EXTRACTION_DRAWS=1`** (the only differing line, verified by `diff`). The primary `.env` (draws = 3) was never
  changed. Budgets: `LLM_MAX_TOKENS`, `OPENROUTER_PREPROCESSOR_MAX_TOKENS`, `OPENROUTER_INFERENCE_MAX_TOKENS` = 128000,
  `OPENROUTER_EXTRACTION_MAX_TOKENS` = 64000. The primary's caches (`data/enrichment_cache.json`,
  `data/id_mapping_cache.json`) and `data/rag_index/` were copied in, so lookups match the primary's environment.
- Runner `evidence/orch756_paper_runs.sh` (copy of ORCH-755's, with tree/task/modes/conditions/timeout as parameters,
  strict only, a `PAUSE` file the orchestrator uses to take the heavy lock between legs), review pack
  `evidence/orch756_review_pack.py` (copies every `*.pwml` and `research_pathway_*`). The worktree's own copies run,
  so the review pack reads the worktree's `tmp/extraction_diagnostics`. One `bounded_run.py` per leg with
  `--heavy-lock ORCH-756`; G11 reports in the primary `evidence/g11/ORCH-756/`; run log and REVIEW in the primary at
  `runs_papers/ORCH-756/` (legs stay in the worktree, `.claude/worktrees/orch756/runs_papers/ORCH-756/legs/`).

**Product owner's entry point: `runs_papers/ORCH-756/REVIEW/INDEX.txt`.**

## 2. Code under test — two phases

| phase | legs | code | pair / wrapper timeout |
|---|---|---|---|
| 1 | enterobactin, menaquinone, chlorophyll (reports `01`-`03`) | `main` @ `8b074dda` (pre-757: the hardcoded audit, RAG-prose, curator, gap and overwatch budgets still small) | 5,400 / 6,000 s |
| 2 | Ara operon, sour taste, WNT PCP, Wnt/Ca2+, then a chlorophyll retry (reports `05`+) | merged `main` @ `f392d447` (ORCH-757 budgets, 758a `Ca++`, 758b empty chunk) | **10,800 / 11,400 s** |

The worktree was moved to `f392d447` between the phases (untracked runner copies removed first so the checkout could
not collide; caches, `.env` and leg outputs untouched).

**Operating ruling (not a DECISIONS entry).** After chlorophyll timed out at 5,400 s with Stage-2 calls of 21-26 min,
the sub-orchestrator asked `main` (options: keep 5,400 s; 10,800 s; lower Stage 2 to 64k). `main` approved, 2026-10-21:
**10,800 s per leg** for the remaining legs, **one retry pass** of the timed-out legs at 1 draw on the merged code, and
10,800 s for ORCH-759; **Stage 2 is not lowered**. Recorded as an orchestrator operating ruling under D-105's intent:
the product owner said to remove the limits, and a timeout costs only wall time. The remaining phase-2 legs were run
at 10,800 s directly rather than at 5,400 s and then retried, which is the same ruling without a wasted 90-minute
attempt.

## 3. Results

Review: `runs_papers/ORCH-756/REVIEW/INDEX.txt`; run log `runs_papers/ORCH-756/night.log`; legs
`.claude/worktrees/orch756/runs_papers/ORCH-756/legs/`. Every report below shows `FINAL SURVIVING COUNT : 0`.

| paper | phase | G11 | minutes | outcome | PWML |
|---|---|---|---|---|---|
| enterobactin (PMC12096016) | 1 | `01` | 28.4 | PASS (with warnings), release `review_required`, 7 reactions, 7 interactions | **yes**, `pathway.review_required.pwml` |
| menaquinone (PMC12421875) | 1 | `02` | 20.6 | PASS (with warnings), release `review_required`, 9 reactions, 0 interactions | **yes**, `pathway.review_required.pwml` |
| chlorophyll (PMC5130987) | 1 | `03` | 90.1 | timeout at 5,400 s, killed in Stage 2 (see § 4) | no |
| Ara operon (PMC11983282) | 2 | `04` | — | **interrupted by the machine crash** (§ 5 item 3); run dir `2026-10-22_0947` kept | — |
| Ara operon | 2 | `05` | 16.3 | **Stage 0 `scope_conflict`**: Stage 0 read the pathway as "AraC autoregulation in araBAD operon", which the batch check rejects against the requested "Ara operon inactivation (autoregulation of AraC)" (`legs/ara_operon-strict/2026-10-22_1018/.../RESULT.txt`). In ORCH-755 the same paper passed Stage 0 (it timed out later), so the check's verdict depends on the Stage-0 draw | no |
| sour taste (PMC7059786) | 2 | `06` | ~3 | **stopped by the orchestrator** on the product owner's reprioritization (§ 5 item 5); deferred | — |
| WNT PCP, Wnt/Ca2+ (PMC10425814) | 2 | — | — | **deferred by the product owner**, never started | — |
| chlorophyll retry at 10,800 s | 2 | — | — | **deferred by the product owner**, never started | — |

Reports `04` and `06` were reserved and never written (crash; stop); their placeholders sit in the git-ignored
`.staging/`. `g11_evidence.py check --task ORCH-756` exits 0.

**The two PWMLs came from phase 1, i.e. pre-757 code.** In those legs the `.env`-driven stages no longer failed
(Stage 2: 0 empty-length of 6 attempts), but the hardcoded sites still did: RAG prose 24/69 empty-length, audit loop
14/15 (`evidence/orch756_leg_stats.txt`), the very sites ORCH-757 lifts. Against ORCH-755 (3 draws, small budgets):
enterobactin went from `final_gate_report_missing` to a PWML (D1's trigger, a Stage-2 `holo-EntB` row, did not recur),
menaquinone from a 3,600 s timeout to a PWML in 21 min.

## 4. Long calls and provider errors (asked for by `main`)

A trace records each model attempt when it ends and has no start time, so a call's duration is taken as the time since
the previous attempt ended (an upper bound). Script `evidence/orch756_leg_stats.py`.

| run | model attempts | calls >= 1,200 s | `finish_reason=error` |
|---|---|---|---|
| ORCH-755, 14 legs, small budgets (baseline, `evidence/orch756_leg_stats_orch755_baseline.txt`) | 627 | **0** | 2 (18 s, 11 s; rag prose, Stage 2) |
| ORCH-756, all legs incl. phase 2's Ara operon (`evidence/orch756_leg_stats.txt`) | 126 | **2** | **2** |

All four flagged calls are **chlorophyll Stage 2 inference** at the 128,000 budget:
1. attempt 1, `stop`, 48,898 chars, ~1,288 s (21 min);
2. the next call, attempt 1, provider **`finish_reason=error`, 0 chars, ~1,585 s** (26 min): the provider ended the
   call after 26 minutes with nothing;
3. its retry, attempt 2, **`finish_reason=error` with 8,522 chars, ~610 s**: a partial answer, then repaired by
   `inference json repair` (28 s).

Stage 2 alone took ~3,483 s of the leg's 5,400 s. **New risk for the product owner:** with the small budgets gone, a
reasoning model can spend 20-26 minutes on one call, and a provider error after that long costs the whole wait. At
small budgets (ORCH-755) no call reached 20 minutes and both provider errors came within 20 s.

## 5. Incidents

1. **Stray command (orchestrator).** Early on, a malformed command that piped into `python -` hung for 120 s and was
   moved to the background by the harness; it was stopped with `TaskStop`; a process listing showed no stranded
   Python process. No test or leg was affected.
2. **Runner stop for phase 2.** Stopping the paused runner's Bash task left its `bash.exe` (PID 976424, created by this
   job) alive in its `sleep` loop. A `Stop-Process` was refused by the permission classifier; `taskkill /PID 976424 /T
   /F` (that one PID, as CLAUDE.md's cleanup rule allows) succeeded; a process listing then showed none left. No leg
   was running (the heavy lock was free).
3. **Machine crash, 2026-10-22, during Ara operon (phase 2, report `04`).** The wrapper (root PID 1441772) died with the
   machine; report `04` was reserved but never written (its placeholder sits in the git-ignored `.staging/`;
   `g11_evidence.py check --task ORCH-756` exits 0). `main` cleared the stale heavy lock by the documented procedure
   (holder ORCH-756, PID 1441772 proven dead, two identical samples); holder sample
   `C:/t/bt-orch755/stale_holder_ORCH-756_2026-10-22.json`. The leg is **interrupted, not failed**; its partial run
   directory `legs/ara_operon-strict/2026-10-22_0947/` is kept and the leg was re-run with a fresh run directory
   (`2026-10-22_1018`, report `05`). The crash is recorded in `night.log`.
4. **About 11 h idle** between 2026-10-21 ~22:55 and 2026-10-22 09:46 in the orchestrator session, before phase 2
   started (most likely an unattended permission prompt). No job was running in that window.

5. **Stop for ORCH-759, 2026-10-22 10:35.** The product owner, through `main`, put the 3-draw run next: let Ara finish,
   then stop. Ara had already ended (16 min) and the runner had started sour taste (report `06`) 0-2 minutes before the
   message arrived. `bounded_run.py` cleans up gracefully only on `KeyboardInterrupt`, which cannot be delivered to
   it from this session, so the runner's tree was killed by exact PID (`taskkill /PID 26412 /T /F`: runner bash →
   wrapper 35772/39812 → batch children). The Job Object took the descendants: **0 survivors** (process listing). The
   lock was left in shape 1 (holder ORCH-756, `leg-sour-taste-strict`, dead PID 39812, nothing else in the directory)
   and was cleared by the documented procedure with two byte-identical samples,
   `C:/t/bt-orch756/stale_holder_ORCH-756_sour_2026-10-22_{a,b}.json`, then `rm holder.json` and `rmdir`. Recorded in
   `night.log`.

## 6. Not settled

1. **The Stage-0 scope check rejects a paraphrase** (Ara operon, § 3). "AraC autoregulation in araBAD operon" names the
   requested pathway in other words. Options: (a) leave it: the leg fails whenever Stage 0 paraphrases (cost: Ara has
   no PWML on that draw, and the outcome depends on the draw); (b) loosen the match (cost: a code lane, and a looser
   check could admit a genuinely wrong paper); (c) reword the topic line to the paper's own phrase (cost: none in code,
   but it tunes the input to pass). Not diagnosed further here. **Ruled 2026-10-22 by the product owner: (c), reword
   the request, the check stays as strict; the paraphrase rejection is registered as an open policy question.** See
   `ORCH-759-RESULT.md` § 2 and § 4.
2. **Long reasoning-model calls** (§ 4). Options: accept them (cost: wall time; a 10,800 s leg absorbs them); add a
   per-call timeout in the client (cost: a lane, and a cut-off call wastes its tokens); reduce Stage 2's budget (ruled
   out by `main` on 2026-10-21).
3. **The deferred 1-draw legs** (sour taste, WNT PCP, Wnt/Ca2+, chlorophyll retry) run only if the product owner asks.
   Each is up to 3 h at 10,800 s, from `.claude/worktrees/orch756` (at `f392d447`, `.env` draws = 1):
   `bash <wt>/docs/pwml_recovery_sprint/evidence/orch756_paper_runs.sh <wt> ORCH-756 strict "<conditions>" 10800`.

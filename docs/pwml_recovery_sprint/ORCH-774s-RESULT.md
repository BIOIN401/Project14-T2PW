# ORCH-774s — the morning summary lists omissions with KI ids and shows a "would import" coverage

**Status:** implemented and checked; **not merged, not self-approved**. Branch `orch774s/summary-import-check`, worktree
`.claude/worktrees/orch774s`, base **`1f223061`** (the ORCH-772b round-1 tip). Code commit **`b12086f9`**; a docs commit
follows. The work was time-boxed to 23:30 and finished at about 22:20.

**Scope:** `evidence/orch770_morning_summary.py` and its new test. No `src/` change. The overnight runner calls this script
from the PRIMARY checkout as its last step, so it takes effect once merged there.

## 1. What the summary now shows, per paper

1. **The omitted processes with KI ids.** A new section, "Omitted from the PWML, per paper (kind, name, known issue)". For each
   paper it gives counts by KI, then one line per omitted process (`- <kind> <name>: <KI>`), then one line per actor left out of
   a kept process (`- actor <actor> of <kind> <process>: <KI>`). The rows come from the `--src` tree's
   `reaction_evidence.build_report` (`omitted` rows that count, plus ORCH-773's `omitted_actors`), with the leg's recorded
   no-PWML cause (ORCH-772b). Nothing is re-implemented.
2. **"dropped on import"**: the number of reactions PathWhiz drops on import (`reaction.rb` `check_spontaneous`; ORCH-773 r1
   reviewer N1).
   - **How it is detected:** the leg's written PWML file (`pathway.pwml` or `pathway.review_required.pwml`) is parsed with
     ElementTree. A `<reaction>` counts when its `<spontaneous>` text is not `true` and its `<reaction-enzymes>` holds no
     `<reaction-enzyme>`; the writer emits that as `<reaction-enzymes/>`.
   - Shown as `-` when no PWML was written.
   - The reviewer's probe (`C:/t/rev773r1/scripts/noenzyme.py`) reads the canonical payload instead. It was read, not
     imported.
3. **"would import" coverage**: (drawn − dropped on import) ÷ the headline denominator (KI-008 left out, D-114 AMENDMENT 2).
   - It appears in the coverage cell after the headline and the full count.
   - It is only added when a PWML was written. The cell of a leg with no PWML is unchanged, as pinned by ORCH-772b.

## 2. Validation

- **The reviewer's corpus leg** `runs_verify/2026-09-06_1425` PMC12376012 (G11 `02`, `evidence/orch774s/validate.{py,out.txt}`):
  - Inputs: the replay output at the ORCH-772b round-1 tip, `C:/t/orch772b/out-r1tip1-27-…/…PMC12376012/pathway.review_required.pwml`,
    and the corpus record `C:/t/orch772b/r1tip1.json`.
  - Drawn 15, headline denominator 17, so **88.2 %**.
  - **5** reactions have no enzyme and `spontaneous=false` (5 × `<reaction-enzymes/>` out of 14 `<reaction>`).
  - **would import 10/17 = 58.8 %**, which equals the reviewer's 88.2 % and 10/17 = 59 %.
- **ORCH-767 legs** (their own directories): menaquinone and sour taste write a PWML and drop **0** reactions. Sour taste's 7 drawn
  processes are transports. The other four write no PWML.

## 3. Sample: the ORCH-767 legs

Generated with the tip script into scratch (`C:/t/orch774s/ms767`, never the primary's run directory), G11 `03`. Full file:
`evidence/orch774s/morning_summary_ORCH-767_sample.md`.

| paper | PWML | PWML coverage | dropped on import | release |
|---|---|---|---|---|
| enterobactin | no | 0/21 (0%); full count 0/21 (0%); no PWML: KI-012 | - | review_required |
| chlorophyll | no | 0/61 (0%); full count 0/61 (0%); no PWML: KI-004 | - | review_required |
| sour_taste_v2 | yes | 7/12 (58%); full count 7/12 (58%); would import 7/12 (58%) | 0 | review_required |
| wnt_pcp | no | 0/66 (0%); full count 0/66 (0%); no PWML: unclassified | - | review_required |
| wnt_ca | no | 0/48 (0%); full count 0/48 (0%); no PWML: KI-016 | - | diagnostic_only |
| menaquinone | yes | 14/14 (100%); full count 14/14 (100%); would import 14/14 (100%) | 0 | review_required |

Omitted from the PWML, counts by KI (the sample file lists every row):

| paper | left out | by KI |
|---|---|---|
| enterobactin | 21 processes, 0 actors | KI-012 17, KI-001 4 |
| chlorophyll | 61 processes, 0 actors | KI-004 32, KI-011 29 |
| sour_taste_v2 | 5 processes, 0 actors | KI-001 5 |
| wnt_pcp | 66 processes, 0 actors | KI-015 39, unclassified 18, KI-001 9 |
| wnt_ca | 48 processes, 0 actors | KI-016 48 |
| menaquinone | 0 processes, 0 actors | none |

The `unclassified` 18 on WNT PCP are the 18 payload processes of a leg with no PWML: its recorded cause, the refinement-crash
message, has no catalogue entry (ORCH-772b § 9 Q2). The ORCH-767 legs ran at `d6849c9e`, before ORCH-773, so no actor omissions
appear.

## 4. Tests and runs (bounded, heavy lock `ORCH-774s`, pinned; every report 0 survivors)

| run | G11 | result |
|---|---|---|
| `tests/test_orch774s_morning_summary.py` (new, 3) + `tests/test_orch772b_no_pwml_cause.py` | `01` | 12 passed |
| validation probe on the corpus leg and the ORCH-767 PWMLs | `02` | as § 2 |
| morning summary on the ORCH-767 legs | `03` | as § 3 |
| final, at `b12086f9`: 774s + 772b + 772 test files | `04` | **45 passed** |

**G9:** the new tests are labelled NEW ACCEPTANCE TESTS; the functions do not exist at base. `g11_evidence.py check --task
ORCH-774s`: 4 artifacts, 0 non-compliant.

**No PAUSE was used.** The heavy lock was free for every job, all before 22:20, so I never created
`runs_papers/ORCH-774/PAUSE`. ORCH-775 probes were not seen. Orphan sweep after the jobs: VS Code language servers only; nothing
killed.

## 5. Not settled

* "would import" removes only the reactions PathWhiz drops for a missing enzyme. Any other import-time drop is not counted, so the
  figure is an upper bound for what survives (KI-034: PathWhiz drops failing elements silently).
* The section lists every row. For a large no-PWML leg (WNT PCP, 66 rows) the summary grows by that many lines; counts by KI come
  first. If the morning summary must stay one page, the rows could move to an appendix file. Cost: one more output file.
* Q4 of ORCH-772b stands: the summary runs outside `bounded_run` in `orch770_night.sh`.

## 6. LEDGER row

> **`ORCH-774s` · 2026-10-25 — IMPLEMENTED, awaiting review, branch `orch774s/summary-import-check` @ `b12086f9` (+ docs).** The
> morning summary now adds, per paper:
> * omitted processes and actors with KI ids, from the report model, with counts by KI;
> * "dropped on import": reactions the PWML XML exports with no enzyme and `spontaneous=false`;
> * a "would import" coverage beside the headline and the full count.
>
> Validated on `runs_verify/2026-09-06_1425/PMC12376012`: 88.2 % headline, 5 dropped, would import 10/17 = 58.8 % (equals the
> ORCH-773 r1 reviewer). The ORCH-767 PWML legs drop 0. Script and test only; no `src/` change. Record: `ORCH-774s-RESULT.md`.

## 7. TEST_MATRIX delta

* **New focused selection:** `tests/test_orch774s_morning_summary.py` (3, under 1 s). Run it, with
  `tests/test_orch772b_no_pwml_cause.py`, for any change to `evidence/orch770_morning_summary.py`.
* No pin moved. SMOKE not affected (no `src/` change).

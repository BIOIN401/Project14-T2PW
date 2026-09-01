# REV-111 — independent review of C-111 (F-148 timeout observability)

**Reviewed SHA: `ae9c1570ea77e7fea4cd9dccc9803ea4a516bbd3`**, re-verified as the final tip of
`card/C-111-timeout-observability` at the time of this verdict. Base `45c4f73996bdf312284d204936df2218dbb076db`.
The card advanced from `d771323f` to `ae9c1570` mid-review; `git diff --stat d771323f ae9c1570 -- src/ tests/`
is **empty**, so the code under review is unchanged and every finding below holds at the final tip.

**Verdict: APPROVE WITH REGISTERED RESIDUALS.**

---

## 0. Static vs behavioural — kept separate on purpose (F-156)

A static read and a mutation-proved property are different epistemic objects.

**Verified ON THE DIFF (static):** B2 (retry constructs textually identical), B3 (no write to any
override), B4 (no repair of the finalization seam), B5 (the `stage="init"` path), B7/B8 (no network
or provider call in any added line), B11 (no prompt body crosses the seam), B14 boundary
(`pipeline/` and `bench/` tree hashes), the two deleted `runner.py` lines.

**Verified BEHAVIOURALLY (constructed and measured):** B1 (nine items read off disk after a real
force kill, at base and at tip), B6 (three mechanisms constructed, labels read off disk), B12
(seven credential shapes planted by me, caught; four benign strings, not caught), B13 (base/tip
contrast run as two real kills), B18 (five mutations, four caught, one survived and then explained
by measurement), and the fsync timing question.

---

## 1. The Lead's pre-verified claims — re-derived independently

| Claim | My result |
|---|---|
| `src/t2pw/pipeline/` byte-identical | **Confirmed.** Tree hash `fd3efcba643566039a041809cca7d5361e7b938c` at base and tip. `git diff --stat` empty. |
| `bench/acceptance.py` untouched | **Confirmed**, and the whole `bench/` diff is empty. |
| its hash `4bd893ac…` | **Confirmed, with a note.** `4bd893ac…` is the CRLF working-tree file; the LF blob is `d9f817e1dca20bf9e17813d366a88dcac275072cf323a43c115e1acf69e59653`. Both are identical either side of the card. Not a discrepancy — worth pinning which form is quoted. |
| F-158 verdict line byte-identical at `:718` | **Confirmed.** `verdict = "PASS" if status == _STATUS_PASS else "FAIL"` at line 718 in base and tip. |
| No assignment to `leg_timeout_override*` | **Confirmed — they are READS.** See B3. |
| Files changed | **Confirmed:** `leg_trace.py` (new, 733), `runner.py` (+190/−2), `client.py` (+47/−0), `test_c111_timeout_observability.py` (new, 728). Nothing else under `src/` or `tests/`. |

**No discrepancy with the Lead on any of these.**

---

## 2. B1–B15

**B1 — the nine items preserved durably past cleanup. PASS, behaviourally.**
Not accepted from the card's test. I ran my own: `rev111_reviewer_probe.py` arm A launches a real
subprocess through the real `runner.launch_child`, lets the real `_kill_tree` force-kill it, and
then reads the leg directory with a *separate reader process*. Nothing asserted touches an object
that was alive at kill time.

```
A.tip_arm_was_really_killed      OK
A.tip_arm_preserved_a_trace      OK ['LEG_TRACE.jsonl', 'partial_reactions.json']
A.tip_leg_yields_three_attempts  OK 3
A.tip_row_STILL_says_files_empty OK the row must NOT have been repaired
```

The card's own `test_new_c111_acceptance_nine_items_survive_a_hard_kill` is the same shape and is
honest: `summarize()` reads the filesystem, and `NINE_ITEMS` is asserted item by item off it.

**B2 — nothing about retry behaviour changed. PASS.**
`client.py` is **+47 / −0** — purely additive, zero deleted lines. Every retry construct is textually
identical at base and tip:

```
diff <(base: grep time.sleep|base_sleep|max_sleep|max_retries|for attempt in|empty_attempts)
     <(tip:  same) -> RETRY CONSTRUCTS BYTE-IDENTICAL base vs tip
```

The only insertion into a live path is the last statement of `CompletionDiagnostics.note`:

```python
        self.response_status = str(status)
        _publish_attempt(self, row)
```

placed **after** every state mutation, returning `None`, its return value used nowhere, and totally
guarded by two bare `except`s. My arm F read the knobs out of both trees in separate processes:
`DEFAULT_PAPER_TIMEOUT`, `_CHILD_GRACE`, `LEG_TIMEOUT_SECONDS`, `PARENT_CHILD_GRACE_SECONDS`,
`DEFAULT_FINALIZATION_RESERVE_SECONDS`, `child_deadline_seconds(1800,120)`, `SDK_MAX_RETRIES`,
`LLM_MAX_RETRIES` default — **identical, base and tip**.

**B3 — the leg ceiling and `leg_timeout_override_*` untouched. PASS.**
The only producers of those two fields in the tree are `pipeline/deadline.py:230-231`, in a directory
proved byte-identical. Every occurrence in the diff is a docstring, a probe *output* log, or a test
pinning that they are unchanged. The `leg_timeout_seconds=` occurrences the Lead flagged are all
**reads**:

```python
        reserve = leg_trace.finalization_reserve_record(
            elapsed_seconds=float(elapsed),
            leg_timeout_seconds=float(timeout),      # reads the parameter
            grace_seconds=_CHILD_GRACE,
        )
...
            "leg_timeout_seconds": round(float(timeout), 2),   # writes a DIAGNOSTIC dict
```

`finalization_reserve_record` takes `leg_timeout_seconds` as an input and returns a dict; it assigns
no module attribute and no config. **No write anywhere in the diff.**

**B4 — the finalization seam not "fixed". PASS.**
The probe proved a narrow defect and the card **registered** it rather than repairing it:
R-C111-1 (post-deadline finalization unbounded and unpriced), R-C111-2 (the outer kill is
unconditionally hard), R-C111-3 (`launch_child` accepts a `deadline=` the leg loop never passes).
`driver.py` and `deadline.py` are untouched. Correct sequencing per charter § 5.

**B5 — `stage=unknown` not made to guess. PASS.**
`_timeout_row` still emits `"stage": "unknown"`, with a comment and no inference. My arm C
constructed both paths and read the rows: outer kill → `stage == "unknown"`; in-process →
`stage == "input"`. On the `stage="init"` the Lead asked about — **it is not the outer-kill path.**
It sits in `run_single`, inside the *child*, on the `read_error` branch:

```python
    read_error = _text(paper.get("source_text_error"))
    if read_error:
        ...
        leg_trace.record_event("leg_end", status=_STATUS_ERROR, stage="init",
                               timeout_source=leg_trace.SOURCE_NONE)
```

An unreadable input file, before `runner = run_fn or driver.run_one` is even reached. No stage has
run. `init` is **true** there, not guessed. Not a B5 violation.

**B6 — the timeout source genuinely distinguishes the mechanisms. PASS, constructed.**
I built the first two end to end and read the labels off disk:

```
C.outer_labelled_outer_parent_kill    OK outer_parent_kill    (real subprocess, real _kill_tree)
C.inproc_labelled_in_process_deadline OK in_process_deadline  (real run_single, child stops itself)
C.the_two_are_DIFFERENT_strings       OK outer_parent_kill vs in_process_deadline
C.wrapper_inferred_from_shape_on_disk OK wrapper
```

**On the author's claim that `wrapper` is deliberately unreachable from the classifier — I judge it
honest, with one narrow imprecision worth registering.** Honest, because when the bounded wrapper
kills the whole tree no parent survives to classify anything; asking a process that was not there to
name the mechanism is exactly the guessing B5 forbids. The imprecision: `summarize` infers `wrapper`
from `trace_exists and not closed` with no terminal record — which is *also* the shape of a leg that
is **still running**, or whose parent died some other way. The docstring discloses precisely this
("inferred by a reader who finds a `LEG_TRACE.jsonl` with no `LEG_TERMINAL.json` beside it"), so it
is disclosed rather than overclaimed. **Registered as RES-3.**

**B7 — no T-107 leg rerun. PASS.**
No command in any of the card's eight G11 reports is a benchmark, a T-run or a batch leg: they are
`pytest`, `pinned_pytest.py`, `c102_goldreaders_split.py` and the offline probe. The probe imports
only stdlib plus `t2pw.batch.{driver,runner,fetch}` and `pipeline.deadline` — **`t2pw.llm.client` is
not imported.** `LIVE_RUN_TREES = ("runs", "runs_verify")` exists in the probe only to assert it
touches neither.

**B8 — the probe was offline and cheap. PASS.** Every added line scanned for
`requests|urllib|http[s]://|openai|\.create\(|runs/`: zero hits outside docstrings and comments.
Probe writes go to one `tempfile.mkdtemp`, removed at exit. Durations: probe 01/02, focused 6.8 s
and 52.5 s, SMOKE 45.3 s, gold readers 68.7 s.

**B9 — three readings separated by MEASUREMENT, hypotheses written first. PASS, verified from git.**

```
4fde91b3  2026-09-01 14:14:09  "write the three readings down BEFORE the probe exists"
          1 file changed: c111_three_readings_hypotheses.md  (115 insertions)   <- ONLY the hypotheses
9bf2b351  2026-09-01 14:32:21  "the offline probe, and which of the three readings it establishes"
          c111_deadline_probe.py first appears here
```

18 minutes apart, in that order, the hypotheses commit containing **nothing else**. The author's
claim is exactly right. The verdict rules in § 3 of the hypotheses file are decidable from probe
output alone and reference no number from the artifacts. Every one of the six measurements carries a
named positive and negative control, and the collector fails the run on `value is not True` — which
it demonstrably did on the first attempt.

**B10 — `1798.3 s` treated as a hint. PASS, and unusually well.** The hypotheses file fixes the rule
in advance: *"It is used here only to choose which measurements to take (M3 and M4), never as a
premise… If the probe contradicts the hint, the probe wins."* The result file then says M4 *"would
have refuted the mechanism just as readily."* And reading 1, the reading the hint pointed away from,
was **refuted by measurement** (`_run_app` refused the fourth interaction at 3.001 s of a 3.0 s
budget — a 0.001 s overrun reported as a magnitude, not a boolean, because `_Budget.slice` has a
`max(1.0, …)` floor). That is a lead tested, not a conclusion confirmed.

**B11 — no secrets, no prompt bodies; existing detectors reused. PASS.**
`CREDENTIAL_PATTERNS` is vendored from `g11_evidence.py`'s `CRED_PATTERNS` and pinned
pattern-for-pattern by `test_c111_credential_patterns_are_the_sprint_detectors_not_a_new_set`, which
loads the `docs/` module by path — so the two cannot drift. On the § 9 ambiguity: **the card did not
widen policy.** `model_attempt` records `request_hash` / `response_hash` / `content_chars` and a
clipped `reason`, and the event key set is closed and asserted:

```python
    assert set(event) == {
        "seq", "kind", "elapsed_seconds", "stage", "attempt", "status", "model",
        "reason", "finish_reason", "content_chars", "request_hash", "response_hash",
    }
```

The conflict is **reported, not resolved** — charter § 4's required behaviour, and the correct one.

**B12 — the secret scan proved to FAIL. PASS, on my own strings.**
A scanner nobody has seen fail is not a scanner, so I planted seven of my own, different from the
card's, and added a false-positive arm the card does not have:

```
B.detector_fires_on_openai_style_key          OK   B.detector_fires_on_github_token       OK
B.detector_fires_on_bearer_token              OK   B.detector_fires_on_google_api_key     OK
B.detector_fires_on_inline_secret_assignment  OK   B.detector_fires_on_aws_access_key_id  OK
B.detector_fires_on_credentialed_url          OK
B.no_false_positive_on 'task-1234567890123' / 'risk-assessment-of' /
                       'the extraction sta'  / 'sha256:deadbeefcaf'   all OK []
B.LEG_TRACE.jsonl_clean_by_scanner  OK []   B.LEG_TRACE.jsonl_no_verbatim_secret  OK []
B.LEG_TERMINAL.json_clean_by_scanner OK []  B.LEG_TERMINAL.json_no_verbatim_secret OK []
```

All three shapes REV-111 names explicitly — OpenAI key, bearer token, inline assignment — fire and
are redacted before disk. **My first run of this arm FAILED its own control** and is preserved:
`rev111_reviewer_probe.attempt1-malformed-akia.log`, G11 `02`. The cause was **mine, not the
detector's**: my AWS string had 14 characters after `AKIA` and the pattern requires 16. The failure
was informative in one direction — the malformed string reached disk verbatim, which is the detector
correctly *not* over-firing. Corrected and re-run as G11 `03`.

**Reported, not a finding:** `x-internal-credential 8fA2b9Qz71LmPd` matches nothing. That is the
sprint's detector set, unchanged by this card, and widening it is not C-111's job.

**B13 — G9 labels honest. PASS.** The card labels this **NEW capability with no fabricated base
failure**, which is the correct shape — and I checked it is *true*, not merely asserted, because
symbol absence proves nothing on its own. Arm A ran the **same synthetic kill twice**, once with the
base `src` on the child's path and once with the tip's:

```
A.base_has_NO_leg_trace          OK
A.base_arm_did_produce_a_payload OK ['partial_reactions.json']    <- work really happened
A.base_arm_preserved_NO_trace    OK ['partial_reactions.json']    <- and left no record of it
A.base_leg_yields_zero_evidence  OK   (tip reader, base artifacts: 0 calls, trace absent)
A.tip_leg_yields_three_attempts  OK 3
```

So a base failure **was** constructible, and the card declined to claim one anyway — the
conservative direction. Nothing in the diff is a corrected pre-existing behaviour mislabelled as new:
the F-158 `RESULT.txt` change adds two print lines and is covered by
`test_new_c111_acceptance_result_text_names_the_two_operational_fields`, also labelled new.
**No fabricated base failure exists. Not a reject.**

**B14 — gates. PASS, all four re-verified from the committed artifacts.**

| Gate | Result | Report |
|---|---|---|
| focused split | exit 0 | `03`, 0 survivors |
| focused combined | **316 passed**, exit 0 | `06`, 0 survivors |
| SMOKE | **503 passed / 22 files**, exit 0 | `07`, 0 survivors |
| gold readers | **456 passed / 0 failed / 8 skipped / 0 errors**, exit 0 | `08`, 0 survivors |

The 8 skips are all in `test_strict_failure_replay.py`, where they have always been (line 14 of the
split log; every other file reports `skipped=0`). Every pin verdict shows `refused: false`,
`violations: []`, `foreign_src_entries: []`, correct tree and venv interpreter. **I additionally ran
the card's test file in my own worktree at the final tip: 23 passed, exit 0.**

**B15 — job lifecycle. PASS, on all thirteen jobs.** Every C-111 report and every REV-111 report
shows `FINAL SURVIVING COUNT : 0` and `cleanup : success`, with the heavy lock acquired **and**
released under its own holder id. `pre-existing (reported, NEVER killed): 4` throughout — the two
`ms-python.isort` language servers among them, matched on command line, never PID, and never
targeted. No `taskkill /IM`, no `pytest -n auto`, no unchunked suite.

**On the gold-readers gate run TRACKED-BACKGROUND: I agree D-026 was satisfied.** Every condition is
evidenced in the artifact itself, not asserted: same unmodified `bounded_run.py`
(`wrapper build sha256:83d1395…`, identical to the digest in my own reports), task id and output path
recorded (`label c111-goldreaders`, `json report … 08-c111-goldreaders.json`, `json report written:
True`), one heavy job at a time enforced by the mutex rather than by protocol (`holder=C-111
acquired=True released=True`), cleanup executed (`53 descendants observed / 53 terminated`),
survivors verified by re-snapshot (`FINAL SURVIVING COUNT : 0`), canonical report inspected, nothing
detached. The card's reason is sound and I would go further: a foreground shell clock is what
stranded this very mutex earlier on this card, and running the third gate in the foreground would
have risked reproducing R-C111-4 *while reporting it*. **Nothing in that gate's measurement depends
on the foreground/background distinction** — which is the retrospective test D-026 itself applies.

---

## 3. Adversarial work — B16, B17, B18

**B16 — killed more than one way. PASS.** Wall-clock overrun (`_timeout_row` at 1800.47 s of 1800.0,
`left_seconds == -0.47`, reproducing `PMC12444477/research` exactly), exception mid-stage (real
non-zero returncode), and a hard kill with no chance to finalize. **The third is the one that
matters and it is real**: my own kill arm confirmed `timed_out is True` and the child had registered
an `atexit` hook and three signal handlers in the card's probe and still wrote no `finalized` marker.

**B17 — "no retries" now distinguishable from "no instrument". PASS.** Proved on disk: an
instrumented leg reports `total_model_calls == 3, retry_reasons == [], _trace_present is True`; an
uninstrumented one reports `0, [], False`. Same empty retry list, two different objects.
**Correction to my own criteria:** REV-111 B17 says `LLM_MAX_RETRIES=3`. The code default is **8**
(`client.py:484, 672, 865`), measured identical at base and tip. The card's docstring says 8 and is
right; **my criteria document is stale here**, and the error is in the direction that makes B17 more
important, not less.

**B18 — my own mutations. 4 of 5 caught; the fifth explained by measurement.**

| Mutation | Removes | Result |
|---|---|---|
| M1 | item 6, payload before cleanup | **caught** (2 red) |
| M2 | item 2, retry reasons | **caught** (2 red) |
| M3 | `flush()` + `os.fsync()` per event | **SURVIVED — 23 passed** |
| M4 | collapse outer kill into the in-process label | **caught** (2 red) |
| M5 | `_publish_attempt` at the LLM seam | **caught** (1 red) |
| N0 | neutral comment (negative control) | **stayed green**, as required |

Baseline green, final green, `git status --porcelain src/t2pw tests` clean, tracked `.pyc` **56 before
and 56 after**, every restore replaying the SAVED BYTES and proved by sha256 and CRLF count (D-084).
Bytecode purged scoped to `src/t2pw` and `tests` before every arm (F-160).

**M3's survival is not a missing guard, and I measured that rather than arguing it.** F-160's warning
is that a survived mutation reads as *"this guard has no test, delete it"*, so I ran a dedicated arm:
the same real force kill, 25 attempts, with and without the fsync.

```
ARM 1 UNMUTATED (flush + fsync present) : {'timed_out': True, 'calls': 25, 'events': 25}
ARM 2 MUTATED   (fsync REMOVED)         : {'timed_out': True, 'calls': 25, 'events': 25}
```

**The per-event open/close alone survives a force kill** — a process kill does not discard the OS
page cache, and `LegTrace.event` re-opens inside a `with` block per event, so the close already hands
every completed event to the OS. The fsync buys durability against a **machine** crash, a strictly
wider threat than the one the module documents. The suite cannot see it **by construction**, because
on the threatened failure mode there is nothing to see. Registered as RES-1, not a gap.

---

## 4. The Lead's two specific questions

**Q1 — the two deleted `runner.py` lines. Both benign; nothing was dropped.**

The diff deletes exactly two lines and I read both in place.

`-from t2pw.batch import driver, report` is **not** a relocation — it is an in-place edit to
`from t2pw.batch import driver, leg_trace, report`. `driver` and `report` both survive; one name was
added.

`-        target = paper_dir(run_dir, slug) / mode` **is** a genuine relocation, moved ~20 lines
earlier in `_run_batch`, from after the row-building block to immediately after `elapsed = clock() - began`:

```python
        target = paper_dir(run_dir, slug) / mode
        payload_before_cleanup = leg_trace.scan_payload(target)
```

I checked the three things that could hide a change in it and all three hold. (i) `paper_dir` is
`return Path(run_dir) / PAPERS_DIRNAME / slug` — pure, no `mkdir`, no side effect, so evaluating it
earlier does nothing. (ii) Nothing between the new and old positions rebinds `run_dir`, `slug` or
`mode`; they are the loop variable and its enclosing scope, and the intervening statements are
`parse_child_output`, `_identify`/`_relocate_files`/`_timeout_row`/`_crash_row` and a `log`. (iii)
`target` is bound **exactly once** in `_run_batch` at tip. The value is identical and the later
`result_path = target / RESULT_NAME` consumer is unchanged. **No behaviour was dropped with it.** The
relocation is load-bearing for the card's honesty: it lets `scan_payload` read the directory
*before* the parent writes `RESULT.txt` into it, which is what makes item 6 a reading rather than an
inference.

**Q2 — `_publish_attempt`'s synchronous I/O on the measured path. Acceptable, and I have the number.**

Your framing is right that this is the sharp question: an instrument that changes the thing it
measures is worthless, and a fsync per model attempt is a real timing side effect on exactly the path
this card exists to observe. I formed no view either until I measured it.

**Measured: +0.55 ms per model attempt** (200 attempts, with a no-trace control arm at 0.000143 s
total confirming the control really is a no-op, and `attempt_log` identical either way so the
in-memory behaviour is untouched).

Put against the quantities it could perturb:

| Against | Cost |
|---|---|
| PRODUCT_CONTRACT § 9's cap of **3** Stage-1 attempts | **+1.7 ms** |
| the 120 s finalization reserve | **0.0014 %** |
| the 1680 s child deadline | **0.0001 %** |
| a pathological runaway of **1000** attempts | +0.55 s, i.e. **0.46 %** of the reserve |

The F-148 overruns being diagnosed are **120 seconds**. The instrument's footprint is five orders of
magnitude below the effect it is built to observe, and it stays below it even under a retry runaway
an order of magnitude past the contract's own ceiling — which is the adversarial case, since retry
amplification is the hypothesis this instrument exists to test. It also **changes no retry logic**:
`_publish_attempt` is the last statement of `note()`, returns `None`, is consumed nowhere, and is
totally guarded.

**My assessment: acceptable, and I would not even register it — except that the fsync is unguarded
(M3) and, per the arm above, is not what delivers force-kill durability.** Those two facts belong
together, so I register them as one item, RES-1. This is not a B2 violation on its face and not one
underneath it either.

---

## 5. Registered residuals — none blocking

**RES-1 — the fsync is unguarded and buys a wider threat than the one documented.** M3 removes
`handle.flush()` + `os.fsync(handle.fileno())` and the suite stays green; measurement shows the
per-event open/close alone survives the force kill, so the fsync is machine-crash insurance the
suite cannot test by construction. It is also the entire source of the +0.55 ms/attempt timing cost.
**Keep it** — the cost is negligible and the insurance is real — but the module docstring currently
presents flush+fsync as *why* the design works, which overstates what it does on the documented
threat model. A one-line docstring correction, or a test that pins durability across a simulated
power loss, would close it. Not this card's obligation.

**RES-2 — a tautological assertion inside the test that pins B2.**
`tests/test_c111_timeout_observability.py:598`:

```python
    assert int(os.getenv("LLM_MAX_RETRIES", "8")) == int(os.getenv("LLM_MAX_RETRIES", "8"))
```

`x == x`. It asserts nothing. Every other line in `test_c111_changes_no_retry_or_ceiling_knob` is a
real value pin and the test's substance is unaffected — **and B2 is independently proved by the diff
being +47/−0 with byte-identical retry constructs, which I verified separately** — but a vacuous
assertion sitting in the test whose docstring says it pins B2 is exactly the shape F-156 exists to
catch. Cosmetic, registerable, worth fixing whenever this file is next touched.

**RES-3 — `summarize`'s `wrapper` inference is broader than its label.**
`elif trace_exists and not closed: timeout_source = SOURCE_WRAPPER` also matches a leg that is still
running, or a parent that died some other way. The docstring discloses the inference honestly, so
this is imprecision, not overclaim.

**RES-4 — `scan_payload` counts a child-written `RESULT.txt` as payload.** `INSTRUMENT_FILES` excludes
only `LEG_TRACE.jsonl` and `LEG_TERMINAL.json`. On the outer-kill path the child never writes
`RESULT.txt`, so the T-107 case is unaffected; on the crash path `existed` can be `True` on a
`RESULT.txt` alone. `_record_leg_terminal` already skips `RESULT_NAME` when building cleanup
decisions, so the effect is confined to one boolean. Narrow.

**Also noted, not a residual against this card:** REV-111 B17's `LLM_MAX_RETRIES=3` is stale; the
value is 8. And the card's registered R-C111-1…R-C111-4 are correctly registered rather than
repaired; R-C111-4 is a measured incident, not a hypothesis.

---

## 6. Two things I checked because their absence would have been a finding

**The failed first probe attempt is present, beside its correction.** G11
`01-deadline-probe.json` exits **1**, `CONTROLS_FAILED: M2.control_positive_parent_waits_full_ceiling=False`,
with `FINAL SURVIVING COUNT : 0` and `cleanup : success` — a job that was perfectly clean and
**measured nothing**, because the synthetic `BatchPaper` had no `slug`, `plan_pairs` skipped it and
the leg loop never ran. Only the control made that visible. `02` is the correction. Both committed.

**`payload_before_cleanup.existed = True` while the row still says `files: []` was explained, not
repaired.** Pinned in the card's own test and independently in my arm A:

```
A.tip_row_STILL_says_files_empty  OK   (row["files"] == [] and row["counts"] == {})
```

while the same leg's terminal record names `partial_extraction.json` / `partial_reactions.json` on
disk. The row is not repaired; the emptiness is explained. **That is instrumenting, not fixing** —
and it is the single clearest demonstration that this card kept its charter.

**The card corrected its own committed record in the open** (`f454f51c`): its strand report had
concluded the anonymous lock was not C-111's, and it appended the correction beside the original
rather than rewriting it, naming where the inference failed — *"a negative result about one job is
not a negative result about the card that owns it."* It refused to clear a lock it could not
attribute, which was correct whoever owned it. Not a criterion, but it is the behaviour this sprint
asks for and it is worth the record.

---

## 7. Verdict

**APPROVE WITH REGISTERED RESIDUALS** — RES-1 through RES-4, none blocking, none requiring a
correction round.

This card instruments and does not fix. I looked hard for the fix riding along inside it — the two
deleted lines, the `stage="init"`, the fsync on the measured path, the override reads, the retry
seam — and there is none. The one place it could most easily have cheated, repairing `files: []`
once it could see the payload on disk, it explicitly did not.

**Jobs:** five, all `FINAL SURVIVING COUNT : 0`, `cleanup : success`, lock acquired and released.

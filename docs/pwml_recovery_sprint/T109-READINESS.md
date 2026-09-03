# T-109 release-candidate readiness

**The separately recorded readiness decision `T108-READINESS.md` § 7.1 requires for any candidate
after T-108.** Built by the Lead Orchestrator, session `project14-t2pw-51` `[e2c249]`, task
`ORCH-720`, on **2026-09-02**, against the committed integration tip and nothing remembered.

---

## 0. Why the milestone is called T-109

**The product-owner ruling says "launch T-108 once."** T-108 has already been launched once: it ran
20/20 legs on `runs_verify/2026-09-01_1612`, is scored **`NOT ACCEPTED`**, and is immutable. The
**same ruling** says *"Do not rerun T-108"* and *"Do not redraw, rerun, or chase a favorable
outcome."* `T108-READINESS.md` § 7.1 says a further candidate needs a **new milestone identity and a
separately recorded readiness decision.**

> **`T-109` is the only reading under which all three of those hold at once.** The ruling's
> instruction is executed exactly — the release-candidate milestone is launched **once** — and
> T-108's record is left untouched.

**Nothing about T-108 is re-run, re-scored, reinterpreted or edited.** `T108-RESULT.md` stands.

**The instruments keep their `t108_` filenames** — `t108_stage_verify.py`, `t108_score.py`. They are
generic (`<repo> <run-dir>`), they are committed, and every prior report was produced by those exact
bytes. **Renaming them to match the milestone would break comparability with T-104 through T-108 for
a cosmetic gain**, which is F-163's standing reason for not touching measurement tooling casually.

---

## 1. What changed since T-108's `NO-GO`, and nothing else did

`T108-READINESS.md` § 7.3 made launches NO-GO on **one** ground: the acceptance instrument was
mid-correction, so *"a release candidate scored on an instrument that is mid-correction cannot be
interpreted."* The Priority-4/5 boundary was *"about to move by design."*

**`D-089` settles that the boundary does not move.** D-088 clause 10 controls for this release; the
INCOMPLETE-CORE CAP is unchanged; the reaction-level replacement is deferred to the RAG / LLM
evaluation phase as `R-D089-1`. **The instrument is stable and fully documented, and its one known
error has a known sign** (`FINDINGS.md` F-173).

**§ 7.3's blocker is therefore discharged on its own terms, not waived.**

---

## 2. The rebuilt table — every row re-derived at `0859fba9`, none carried forward

**`T108-READINESS.md` § 5.4 step 1 names rows 4, 5, 11, 16 and 17 as the ones that silently rot and
must never be carried forward. All five were re-measured today.** So were 2, 12, 13, 15 and 18,
because they were cheap and because a table that mixes measured and remembered rows is the failure
F-171 recorded.

| # | Condition | State | Evidence at this tip, measured today |
|---|---|---|---|
| 1 | F-155 merged and independently approved | **GREEN** | C-108 `2e2a294e`, REV-108 approved. Unchanged since |
| 2 | Mutation harness executable and green | **GREEN — re-measured** | `c107_mutation_attack.py`: **`MUTATIONS: 17  SURVIVORS: 0  []`**, exit 0. G11 `ORCH-720/05` |
| 3 | Census pin is in SMOKE | **GREEN** | `test_c102_coverage_denominator.py` is in the 22-file selection and passed inside the 503 |
| 4 | F-146 remains rejected | **GREEN — re-measured** | **`F146=REJECTED`**. G11 `ORCH-720/03` |
| 5 | 29-case battery at zero mismatches | **GREEN — re-measured** | **`battery=0/29`**, `C1=0 C2=0 C3=0 C4=0 C5=0 C6=0`. Unmoved across C-115, C-116 and C-117. G11 `ORCH-720/03` |
| 6 | Corpus movers understood in both directions | **GREEN** | 19 refused / 0 admitted, mover set stable. Closed by C-108 |
| 7 | Negative controls scored per the Q1 ruling | **GREEN** | C-110 merged; `PASS_NEGATIVE_CONTROL` implemented, default-deny |
| 8 | Q2/Q3 decision merged and reviewed | **GREEN** | Q3 ruled, no code change; Q2 half 1 merged (C-113, REV-113). Half 2 is an open product question and **explicitly not a launch blocker** — `T108-READINESS.md` § 5.3 |
| 9 | Every applied F-150 correction passed its independent A/B | **GREEN** | Applied at C-113; REV-113 re-derived it rather than accepting it |
| 10 | No absolute acceptance priority guaranteed to fail | **GREEN** | The absolutes are **Priorities 1-3** (`acceptance.py:1050`). **Priority 5 is not one of them** — see § 3, and `D-089` accepts its `0/2` explicitly |
| 11 | Deterministic SMOKE + gold-reader gates green | **GREEN — re-measured** | SMOKE **503 passed / exit 0 / survivors 0** (G11 `ORCH-720/01`, pinned via `pinned_pytest.py`); gold-readers **456 passed, 0 failed, 8 skipped, 0 errors, exit 0** across 22 split files (G11 `ORCH-720/02`) |
| 12 | `acceptance.py` hashes identically | **GREEN — re-measured, and the three forms disambiguated** | CRLF working-tree **sha256 `4bd893ac…`** · LF-content **sha256 `d9f817e1…`** · git **blob id `56aa593e…`**. All three measured today; `git status` clean. **The row has always quoted the first two; the third is added because two of them are sha256 and one is not, and they have been conflated before** |
| 13 | Integration pushed and remotely verified | **GREEN — re-measured** | `local = origin/ = git ls-remote` all three equal to `0859fba9`. `main` untouched: local `7531692`, remote `03f1af5` |
| 14 | Pinned 10-paper / 20-leg plan verifies offline | **GREEN, re-verified AT LAUNCH** | `topics_t104.txt` unmodified since `2673067f`; gold blob **`36f4b7b690b577f72882c3045ca6728d1ec8d9d1`**, `git status` clean. **Must still be re-proved inside T-109's own staged directory** — a launch step, not a pre-launch blocker |
| 15 | Configured provider and pinned model available | **GREEN — re-measured, and one trap recorded** | `LLM_PROVIDER=openrouter`, `LLM_TEMPERATURE=0`, `LLM_MAX_RETRIES=3`, all **nine** `OPENROUTER_*_MODEL` pinned to `deepseek/deepseek-v4-flash`, key present (73 chars, `sk-or-v1-` prefix). **Verified through the loader, never printed.** See § 4 |
| 16 | Heavy lock free | **GREEN — re-measured** | `C:/t/heavylock` absent before the claim; acquired and released cleanly by all five ORCH-720 jobs |
| 17 | Zero sprint-owned Python | **GREEN — re-measured** | Exactly two `python.exe`, both `ms-python.isort-2026.6.0/bundled/tool/lsp_server.py`. **Matched on COMMAND LINE, never on count or PID** — the count has been 2 and 3 at different points in this sprint |
| 18 | No peer owns an overlapping live job | **GREEN — re-measured** | `ListAgents`: 26 peers, **none live in this repository.** The one peer that ever held a stand-down here, `[2bdab1]`, is **offline**. Corroborated at the machine by 16 and 17 |
| 19 | Enough time to monitor or formally transfer the run | **GREEN** | See § 5. This is the row that was RED before T-108 and it is the row that depends on the operator rather than the tree |
| **20** | **The acceptance instrument is not mid-correction** | **GREEN — NEW ROW, and the one § 7.3 was waiting on** | `D-089`: the cap is unchanged, the Priority-4/5 boundary does not move, and the replacement is deferred to `R-D089-1`. The rule the run will be scored by is **fixed, written down, and its one known error has a known sign** (F-173) |

**Nineteen of twenty green, and row 14 green pending its in-directory re-proof at launch. Decision: `GO`.**

### 2.1 One red result that is deliberately NOT a row — Chunk D

**`chunk_d_gate.py` is RED in the primary checkout at this tip: `run-core 159/160` and `node15
0/1`.** It is recorded in full as **F-174** and it is **not** a readiness row, for three reasons
stated rather than assumed:

1. `TEST_MATRIX:244` — *"Chunk D is excluded from the smoke gate."* Merge gate 10 is SMOKE, and SMOKE
   is **503 / exit 0** here.
2. **It cannot be a code regression.** The last green Chunk D is the C-116 merge `175e1a6f`; the only
   commit since touched **three evidence artifacts and no `src/`, `tests/` or `scripts/`**.
3. **T-108 ran in this same primary checkout with this same `.env` and these same caches.** Whatever
   the second node reacts to was equally present for T-108, so it does not make T-109 less comparable
   to T-108.

**This is not a waiver.** F-174 registers node 2's precise lever as **OPEN**, and the honest statement
is that the gate has never been run outside a worktree and its `187/187` was a property of where it
ran.

---

## 3. The Priority-5 hard-gate check the ruling required BEFORE launch

The ruling directed: *"If authoritative contract text still makes Priority 5 an absolute hard gate,
quote the exact conflict before launching; do not silently rewrite an unrelated locked rule."*

**Checked. There is no conflict — the authoritative text says the opposite, in terms.**
`src/t2pw/bench/acceptance.py:1050-1052`, the docstring of `AcceptanceReport.priorities`:

> Priorities 1-3 are absolute: any non-zero count fails them, regardless of how good the rest of the
> run looks. **Priority 4 is a coverage judgement and priority 5 is a rate to maximise, so neither is
> a hard gate.**

**And nothing gates on them mechanically.** The only two consumers of `priorities()` in the tree are
`acceptance.py:1201`, which serialises the list into `acceptance_priorities`, and `render.py:61-66`,
which displays it. **No caller anywhere computes `all(entry["ok"] ...)`.** The `ok=None` refusal the
docstring describes is a property it reserves for a hypothetical caller, not a live gate.
`PRODUCT_CONTRACT` § 15 constrains **Priority 1** and the O-1 instrument and says nothing that makes
Priority 5 absolute.

**So the accepted limitation is non-blocking on the authoritative text, not on a Lead reading of it.
No locked rule was rewritten, and none needed to be.**

---

## 4. The `.env` trap that produced a false alarm today, written down so it does not produce another

**`grep -E "^OPENROUTER_API_KEY=" .env` finds NOTHING, and the only line it can match is a
commented-out one.** The live key is written with spaces:

```
12:#OPENROUTER_API_KEY=<REDACTED>
13:OPENROUTER_API_KEY = <REDACTED>
```

A shell check therefore reports the key **ABSENT** while `python-dotenv` — which tolerates
`KEY = value` — resolves it fine. **This fired during ORCH-720 and briefly looked like a hard
readiness failure.**

**The rule: verify configuration through the loader the program actually uses, or do not claim to
have verified it.** `evidence/t109_preflight_provider.py` is committed for exactly this, and it
prints names, booleans, lengths and a prefix test — **never a value**.

---

## 5. Row 19 — ownership, which is the only row that depends on the operator

**GREEN, and here is the whole basis rather than an assertion.**

| Requirement | State |
|---|---|
| A named owner for the entire run | **This session**, `project14-t2pw-51` `[e2c249]`, recorded in `T109-RUN-OWNERSHIP.md` before launch |
| Time to see it through | T-108 was **6.37 h** end to end at the same ceiling and the same corpus. This session's window is the operator's stated full day |
| A transfer path if needed | **None exists and none is claimed.** No live peer is authorized for sprint work — 26 peers, none live in this repository. **A transfer nobody has accepted is not a transfer** |
| The compliance rule | `TEST_MATRIX` § 0 rule 1 permits a tracked background job only where the orchestrator *"polls rather than launching duplicates"* and *"no detached or unowned job remains"*. The T-108 pattern — bounded wrapper in a tracked background task plus a persistent monitor, both owned by one session — satisfied it for 6.37 h and is the pattern used again |

**The honest residual risk, stated rather than smoothed:** a machine crash or session loss during the
run would leave it unowned. **One machine crash already happened during this wave.** The mitigations
are that the wrapper's Windows Job Object carries `KILL_ON_JOB_CLOSE`, so the children cannot outlive
it, and that `T109-RUN-OWNERSHIP.md` records the wrapper PID, task id, output path and lock token
**before** launch, so a successor can verify or adopt rather than guess. **T-108 accepted the
identical risk under an explicit ruling, and this ruling directs the launch and requires the
ownership.**

---

## 6. Launch protocol — `T108-READINESS.md` § 4, unchanged, with the identity substituted

1. Fresh **T-109** milestone identity.
2. The same ratified **10-paper / 20-leg** plan (`topics_t104.txt`, unmodified).
3. **Stage-only preflight.**
4. Verify the plan **and the gold** inside that exact staged directory.
5. Require **all pinned overrides and zero search calls**.
6. Continue the verified directory **without `--fresh`**.
7. Configured **pinned OpenRouter models**.
8. **One** run, through the bounded wrapper.
9. Background, with **explicit ownership**.
10. Wrapper timeout from measured T-104-T-108 durations, with cleanup headroom.
11. **Monitor the existing wrapper — never launch a duplicate.**
12. Score the **first valid official draw**, honestly.
13. Preserve **raw and contract-adjusted** results separately.
14. **Do not rerun it to improve stochastic composition.**

**Per-leg ceiling 3600 s with NO override**, verified as `leg_timeout_overridden: false` in the staged
directory **before** launch, not read out of the manifest after.

**There is no OpenRouter usage ceiling. Cost must not restrict justified work** — restated by the
ruling.

### 6.1 What T-109 must report, and the two limits that travel with it

20-leg completion and scorable denominator · every timeout and missing payload · Priority 1 raw
**and** accepted counts and composition · Priority 2 eligible denominator and its NOT-EVALUATED
population · Priority 3 referential-integrity failures · Priority 4/5 raw and accepted · negative
controls · every applied policy adjustment · whether the result is accepted · exact evidence paths ·
model and provider provenance · usage and cost where available.

**Reported SEPARATELY, by ruling: hard-gate acceptance versus diagnostic Priorities 4 and 5.** The run
is not called accepted if any actual hard gate fails.

**Two numbers must never be quoted bare:**

> **Priority 2 = N is a real number and it is not a measure of how much invented chemistry a run
> produced.** `supported_reactions_complete` is unset on all ten cases (D-087), so the
> unsupported-reaction verdict is evaluable only through `max_retained_reactions`, set on exactly two
> gold cases, both negative controls.

> **Priority 5's `0/2` is an accepted conservative limitation, not a pipeline capability
> measurement.** Half its strict denominator is known-misclassified in a known direction — **F-173**.

**If T-109 fails it is preserved as a failed official release candidate and triaged from committed
artifacts. It is NOT rerun.**

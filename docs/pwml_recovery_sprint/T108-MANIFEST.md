# T-108 — run manifest

**Written BEFORE launch.** Everything below was verified inside the exact staged run directory, not
recalled from a previous milestone and not read out of the result afterwards.

**T-108 is a NEW milestone identity.** It is not a re-run, re-score or re-reading of T-107.
**T-107's verdict is `NOT ACCEPTED` and is a fact about the artifacts it produced.** No T-108 result
may be reported as confirming or overturning it — a T-108 result is reported independently.

---

## 1. Identity

| Field | Value |
|---|---|
| Milestone | **T-108** |
| Run directory | `runs_verify/2026-09-01_1612` |
| Integration tip measured | `0bbac3fd863d3ff22d1172354c4b367d34a6d1bd` |
| Checkout | **primary** — `C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`. **Not a worktree** |
| Owner | `project14-t2pw-da` `[237fab]` — see `T108-RUN-OWNERSHIP.md` |
| Gold blob | `36f4b7b690b577f72882c3045ca6728d1ec8d9d1`, version `2026-08-01.1`, working tree **=** HEAD |
| Topics | `topics_t104.txt`, tracked and clean, `sha256:6f959692…` |
| Plan | 10 papers × 2 modes = **20 legs**, all `[pinned_override]`, **0 search calls**, `verdict: OK` |

## 2. The per-leg ceiling — 3600 s, NO override

**`leg_timeout_overridden: false` was verified in the staged tree before launch, not after.**
`_ceiling(3600.0).to_dict()` returns exactly
`{'leg_timeout_seconds': 3600.0, 'leg_timeout_default_seconds': 3600.0, 'leg_timeout_overridden': False}` —
**no `leg_timeout_override_reason` key is emitted at all**, because there is no override.

### Recorded rationale

> T-107 silently reduced the prior 3600-second ceiling to 1800 seconds without an override reason.
> Its slowest successful leg consumed 92.1% of that reduced ceiling, and PMC12096016 was lost to the
> clock rather than rejected biologically. T-108 restores the previously established 3600-second
> ceiling to prevent timeout policy from silently shrinking the scorable biological denominator.

### How the field is satisfied, stated exactly so nobody reads it as an omission

The launch order was *"do not leave `leg_timeout_override_reason` empty."* **At 3600 s that field is
not emitted at all, and that is the stronger outcome, not a loophole.** The failure mode the order
targets — T-107's `leg_timeout_overridden: true` beside an empty reason — is *unreachable* here:
`LegTimeout.to_dict()` adds the reason and source keys only when `overridden` is true. Manufacturing
an override to populate a reason field would require declaring 3600 ≠ 3600 and would re-create the
exact contract defect the ruling dissolves. **The honest way to satisfy a "record your reason" rule
is to stop needing an exception** — `T108-READINESS.md` § 2.1. The rationale is therefore recorded
here, in the run manifest, where PRODUCT_CONTRACT § 9 wants it.

### The measurement the ceiling rests on, with its limits

`evidence/orch718_leg_duration_census.py`, G11 `ORCH-718/02` — pooled finished legs, every tree:
`n=192 · median 927.9 s · p90 1609.0 s · max 3421.4 s`.

- A leg has demonstrably needed **3421.4 s**, so every ceiling below 3600 s is known-insufficient by
  direct measurement. 1800 s is **less than half** the observed requirement.
- p90 is **89.4% of an 1800 s ceiling** — a ceiling that times out by construction, not occasionally.
- 1800 s produced timeouts in **four separate run trees**. T-107's three were the fourth observation
  of a repeating pattern, not an anomaly.

**Two limits carried into the result, because a ceiling chosen on observed maxima is chosen on
censored data:** every timed-out leg is **censored** — it proves the work needed *more* than the
ceiling, never how much more, so the true requirement is **at least** 3421.4 s. 3600 s clears the
slowest *observed* finisher by **179 s (5%)**, which is thin. **A timeout at 3600 s is therefore not
automatically a defect, and must not be waved away either** — it is new information about the
requirement and belongs in the result report as such.

## 3. Wrapper and internal budgets

| Budget | Value | Basis |
|---|---|---|
| Per-leg ceiling | **3600 s**, no override | § 2 |
| Internal `--deadline` | **18 h** | stops *starting* new pairs; a deadline stop is a clean resumable partial, never a kill |
| Wrapper hard ceiling | **72000 s (20 h)** | `runner.py:102-104` — the authoritative calculation: *"10 papers x 2 modes x DEFAULT_PAPER_TIMEOUT is 20 hours in the worst case"* |

**Why the wrapper ceiling is 20 h and not 8 h.** Measured predecessors are T-104 **5.44 h**,
T-105 **4.85 h**, T-107 **5.63 h** (`20282 s`, from its own committed cleanup report) — but all three
ran at the **1800 s** leg ceiling. Restoring 3600 s raises the tail: T-107's three timed-out legs
alone can now consume up to 1800 s more each. The launch order sets a floor of eight hours *unless
the authoritative runner calculation requires more*, and the runner's own documented worst case is
**20 hours**. The two budgets are ordered deliberately so that **the wrapper is never the thing that
stops the run**: the last pair can start at 18 h, run its full 3600 s ceiling to 19 h, and still
leave an hour of wrapper headroom for finalization, scoring-artifact preservation and cleanup. A
wrapper kill is what destroyed T-107's retry telemetry (F-148); this ordering makes that outcome
reachable only after every internal stop has already fired.

## 4. Provider and model provenance

Verified without printing secrets — `evidence/g11/T-108/04-preflight-provider.json`:

| Field | Value |
|---|---|
| `LLM_PROVIDER` | **`openrouter`** |
| All nine `OPENROUTER_*_MODEL` slots | **`deepseek/deepseek-v4-flash`** |
| Fallback model | **none** — no `*FALLBACK*` key in `.env`; the string does not occur in `client.py` |
| `LLM_TEMPERATURE` | **`0`** |
| `LLM_MAX_RETRIES` | **`3`** (primary `.env`; the code default of 8 applies only where `.env` is absent) |
| Connectivity | `GET /api/v1/models` → **HTTP 200**, 419 models, pinned model **live** |
| Pricing (read-only) | prompt **$0.0717/M**, completion **$0.1434/M** |
| `.env` | **unmodified** — not one field edited for this run |

## 5. Child environment — and the documented exception

```
PYTHONPATH        = <primary>/src
PYTHONIOENCODING  = utf-8
```

**`T2PW_OFFLINE_CURATOR` is deliberately NOT set, and this is the exception to the sprint's usual
offline test setting.** It is recorded here rather than left to be rediscovered.

`TEST_MATRIX` § 0 and `HANDOFF` § 7 require `T2PW_OFFLINE_CURATOR=1` on sprint jobs, and
`T108-READINESS` § 3 records its absence as the measured root cause of BL-003. **That rule governs
deterministic test and gate jobs. It must not be applied to the live benchmark**, for a reason
verified in source rather than assumed:

- `pathway_curator.py` is *"one-shot LLM curation step run after the audit loop, before ID mapping"*,
  called on the production path at `streamlit_app.py:4232`, and it uses the ratified
  `OPENROUTER_CURATOR_MODEL` slot;
- setting the flag makes `run_pathway_curator` *"an explicit, deterministic no-op: **zero** model
  calls, output written byte-for-byte from input"*.

Setting it would therefore **disable a ratified stage of the pinned configuration** — the same class
of defect as a worktree silently falling back to `LLM_PROVIDER=local`, in different costume.

**The decisive evidence is T-107's own artifacts, not this reasoning.**
`runs_verify/2026-08-28_1816/papers/PMC12096016/strict/RESULT.txt:68` records

```
LLM returned an empty completion for curator (model deepseek/deepseek-v4-flash,
finish_reason=length, tools_sent=True) on attempt 1/3; retrying as a transient.
```

**T-107 ran the curator ONLINE.** T-108 matches it exactly, so comparability is preserved. The same
line independently confirms `LLM_MAX_RETRIES=3` is operative on the real run — the `.env` trap of
`T108-READINESS` § 3 settled by measurement rather than by reading.

`T2PW_SPECIES_LLM` is likewise **not set** (default `1`, enabled). D-058 rules that T-104 must not
inherit T-103's `T2PW_SPECIES_LLM=0`, and T-107 did not.

## 6. Pre-launch gate results — all re-derived at `0bbac3fd`, none carried forward

| Gate | Result | Evidence |
|---|---|---|
| `local = origin/ = git ls-remote` | **all three `0bbac3fd`** | direct |
| `main` untouched | local `7531692` / remote `03f1af5` | direct |
| merge in progress / staged | **none / none** | direct |
| Heavy lock | **absent** before launch | direct |
| Sprint-owned Python | **zero** — only the two `ms-python.isort` IDE processes, matched on command line | `Win32_Process` full command lines |
| `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d…` | direct |
| `acceptance.py` | `sha256:4bd893ac410d16d3…` **byte-identical before and after SMOKE** | direct |
| **SMOKE** (22 files) | **503 passed, exit 0**, survivors 0 | `g11/T-108/01` |
| **gold readers** (22 files) | **456 passed / 0 failed / 8 skipped / 0 errors, exit 0** | `g11/T-108/02` |
| **29-case battery** | **`battery=0/29  F146=REJECTED  C1..C6 all 0`** | `g11/T-108/03` |
| Provider preflight | **OK** | `g11/T-108/04` |
| Stage-only preflight | 20 pairs planned, **0 legs started** | `g11/T-108/05` |
| `--verify-plan` | **`verdict: OK` · 10 cases · 0 search calls · all `[pinned_override]`** | `g11/T-108/06` |
| Staged-tree verification | **`T108_STAGE_VERIFY: OK`** | `g11/T-108/07` |
| Whole-tree G11 | **5032 artifacts, 0 non-compliant** | direct, pre-launch |
| C-111 / C-112 / C-113 | **all ancestors** of `0bbac3fd` | `git merge-base --is-ancestor` |
| Caches / `topics_*` / stray `ValueError` | untouched, uncommitted | `git status` |
| `cache_snapshot/` | **absent** in the primary checkout, as the previous handoff records | direct |

## 7. Standing limits that must travel with the result

**Priority 2 = 1 is a real number and it is not a measure of how much invented chemistry a run
produced.** `supported_reactions_complete` is unset on **all ten** cases and
`max_retained_reactions` is set on exactly **two — both negative controls** — so Priority 2's
unsupported-reaction verdict can never be evaluated on a non-control paper. Any T-108 report quoting
Priority 2 carries this limit. See `DECISIONS.md` **D-087**.

**`LpxH` is UNVERIFIED on T-107** — both `PMC12444477` legs timed out with no payload. No T-107
result may be reported as confirming it, and the claim is not carried into T-108 unmeasured.

**T-108 is ONE-SHOT.** The first valid official draw is final. It is not re-run for a timeout,
stochastic composition, an unexpected count, a seven instead of a six, a failed acceptance priority,
or missing model-usage telemetry. **A Priority-1 result of 7 is `PASS_WITHIN_VARIANCE` and is not
re-drawn** (D-073).

**There is no spending, token, request or model-usage ceiling for T-108.** Cost does not restrict
justified work. Actual usage is read from the provider and recorded after the run.

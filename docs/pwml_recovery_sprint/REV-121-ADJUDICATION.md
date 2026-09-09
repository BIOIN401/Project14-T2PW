# `REV-121` adjudication — the blocking finding is a measurement artifact of the review's own probe

**2026-09-09. Lead Orchestrator and Integration Authority.** `REV-121` returned **APPROVE WITH
FINDINGS** on `card/C-121-autostate-lifecycle` @ `74fd6f7c`, with one finding marked as blocking
merge. I re-measured it rather than accepting it, and it goes the other way.

G11, both zero survivors, `cleanup : success`:

| job | report |
|---|---|
| call-site census, tip tree, 154 legs | `evidence/g11/ORCH-737/01-orch-callsite-tip.json` |
| call-site census, base tree, 154 legs | `evidence/g11/ORCH-737/02-orch-callsite-base.json` |
| mode isolation, tip | `evidence/g11/ORCH-737/03-mode-artifact.json` |
| mode isolation, base | `evidence/g11/ORCH-737/04-mode-artifact-base.json` |

---

# 1. The finding, and why it does not hold

`REV-121` FINDING 1 claimed the guard fires on **7** archived legs at the production call site
rather than 5, and therefore that the docstring of `autostate_restoration_required` states a false
measured fact. It named the two extra legs:

```
runs_verify/2026-08-22_2147/papers/PMC12452463/research
runs_verify/2026-08-24_1428/papers/PMC12782028/research
```

**Both are `/research/` legs, and the review's probe drove them through STRICT quarantine.**
`quarantine_and_close`'s own docstring says research mode *"runs every decision and applies none of
them"*. Pushing a research leg down the strict path deletes states production would have kept, and
the payload that reaches the guard is one production never builds.

I drove `quarantine_and_close` over all **154** archived legs in the base tree and the tip tree,
giving each leg **its own** export mode — `research` for a `/research/` path, `pathwhiz` for a
`/strict/` one — and compared the resulting payloads directly. Nothing is inferred from a predicate
call; the difference between the two trees *is* the answer.

| | |
|---|---:|
| legs compared | 154 |
| **legs whose post-quarantine payload MOVED** | **5** |
| legs with any reaction-set change | **0** |
| legs whose quarantine `ok` moved | **0** |

The five are the `D-099` § 5 population exactly. Then the pair in isolation, both modes, both trees:

| leg | mode | base | tip | moved |
|---|---|---|---|---|
| `PMC12452463/research` | **`research`** | 5 states incl. `__auto_state__` | the same 5 | **no** |
| `PMC12452463/research` | `pathwhiz` | `[]` | `['__auto_state__']` | yes |
| `PMC12782028/research` | **`research`** | `['__auto_state__']` | the same | **no** |
| `PMC12782028/research` | `pathwhiz` | `[]` | `['__auto_state__']` | yes |

**In the mode production runs them in, neither leg moves.** The number 5 is correct and the
docstring's figure stands. FINDING 1 is **NOT SUSTAINED** as a defect in the patch, and does not
block merge.

# 2. The finding's underlying concern IS legitimate, and is being acted on

`REV-121` was right that the method was imprecise, and right that it matters more because the claim
lives in `src/`. Both the implementer's census **and this orchestrator's own pre-dispatch census**
evaluated the predicate on the raw committed `final_mapped.json` instead of at the call site. That
happened to give the correct answer. It is not the same measurement, and a later reader running the
naive probe gets **7** and concludes the docstring lies.

So the docstring is being amended to state the **conditions** rather than only the number: measured
at the production call site, each leg in its own export mode, with the false-positive claim
qualified to legs that carry a committed gate report — 99 of the 154 do not. The two ways to get 7
are recorded in the code as the artifacts they are.

**This is the more valuable outcome than either the original claim or the challenge to it.** A bare
corpus count in a docstring is unfalsifiable by the next reader; a count with its measurement
conditions is checkable. The review earned that improvement even though its number was wrong.

# 3. On the standard applied to the review

`REV-121` did substantial independent work: it verified the base tree's fidelity before trusting
anything measured in it and found it to be a hybrid rather than a clean base checkout; it re-derived
the F-179 verdict over 154 legs; it measured release classification directly and found 0 of 154
moved; and it proved the change-log test still fails on a reappearing residual code by perturbing
the pin three ways. It also disclosed three of its own unwrapped diagnostic runs unprompted.

**The error was one of construction, not of care** — a single default argument in its own probe. It
is recorded here rather than passed over because the sprint's rule is that a measurement is checked
against the conditions it was taken under, and that rule binds a reviewer exactly as it binds an
implementer. `F-192`'s own census made the same class of error with a fixed-depth glob and
disclosed it; the C-120 report read a refusal as an overwrite. **This is the third instance of the
same failure mode: a correct-looking measurement taken on the wrong object.**

# 4. FINDING 2 — sustained, does not block, documented rather than fixed

The guard scans four `element_locations` buckets, matching the gate's own table.
`ensure_autostates` assigns a state to **two** of them. A payload whose only unassigned row is a
nucleic-acid or element-collection row would fire the guard, be mutated, and still fail the gate.

Measured as latent, not live: 3 archived legs carry `nucleic_acid_locations` rows and 7 carry
`element_collection_locations` rows, and **none has such a row missing a state.**

**Neither narrowing the guard nor widening `ensure_autostates` is authorized under `D-099`**, and
either would be a behaviour change requiring the whole verification run again. The asymmetry is
therefore being documented in the code as a known limitation, together with the specific hazard
`REV-121` identified: the test that locks the guard's bucket list to the gate's table would, if the
gate ever gained a fifth bucket, widen the guard without widening the assignment.

# 5. FINDING 3 — accepted

`evidence/g11/C-121/10-base-tree-export.json` is 252,865 bytes against the documented 64 KB cap,
because a base-tree export spawns one git subprocess per file and the wrapper records every one as
an observed descendant. `exit_reason: completed`, `final_surviving_count: 0`,
`cleanup_success: true`, no forced kill. **A documented cap meeting a per-file export loop is not a
survivor and not a laundered result.** Accepted, and recorded so a future whole-tree
non-compliance count is attributable.

# 6. FINDING 4 and the review's unconfirmed list — accepted as stated

`C:/t/c121base` is a hybrid tree: base `src` with the diff's tests and changed doc applied, no git,
no untracked run families. Correct for a G9 arm, wrong for SMOKE or anything reading `runs_verify`.
Reports `24`, `26` and `27` must not be read as clean-base measurements.

`REV-121` listed six things it could not confirm and labelled them unconfirmed rather than assuming
either way. That is the correct disposition. Of those, the serialized PWML bytes and the zero
tree-error claim were measured by this orchestrator pre-dispatch
(`C-121-PREDISPATCH-MEASUREMENT.md` § 6, `evidence/g11/C-121/05-replay-e2e.json`) and by the
implementer at `32-replay-e2e-final.json`; the review verified the necessary precondition
independently. **The "40 legs an unguarded re-run would perturb" comparand is a raw-artifact number
and its call-site equivalent has never been measured.** It is a counterfactual about a rejected
design, is load-bearing for nothing in the diff, and stays labelled as what it is.

# 7. Disposition

**The review's approval stands and its blocking finding is not sustained.** Merge proceeds after
the two text corrections in § 2 and § 4 land — no source-behaviour change, no test-expectation
change. `D-099` § 5 and § 6 are amended to record the call-site verification.

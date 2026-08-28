# RESUME — next session handoff

**Rewritten by the Lead Orchestrator, 2026-08-27, the O-1 ruling wave.** Everything below this
heading supersedes the PACK 11 record that used to be here; that content is in git history and in
`LEDGER.md`, which remains the single source of truth for task state.

> **⚠ Why this file is in the repo and not in a scratchpad.** A prior session wrote its handoff to a
> *session-local* scratchpad and the next session could not find it. **Keep this file in the repo and
> update it in place.** This wave proved the point twice over: the ORCH-710 probe scripts lived in a
> temp directory and were one cleanup away from being lost, taking the only record of what the O-1
> numbers measured with them.

---

## 1. Integration state

| | |
|---|---|
| Branch | `sprint/pwml-recovery` |
| Session start tip | `7bd8a86` |
| **Do not pin a tip SHA here** | the invariant is **local = `origin/` = `git ls-remote`**, verified after every push. Read it, do not recall it |
| Merges to `main` | **none, and none permitted** |
| Product-owner `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d…` — verified intact |
| Caches, `topics_*.txt` | uncommitted, untouched. `topics_wave_cohort.txt` belongs to a peer session |

## 2. What this wave settled

**O-1 is CLOSED** — `DECISIONS.md` **D-070**. The question was rejected as posed. The pinned 21 is
**16 generated functional wrappers + 5 PathBank `Unknown` sentinel rows**, overlap 0, and **none of
the 21 sets `placeholder_claims_real_identity`** — so none is a forged identity and no report may
call them one. TRAP-3 stands on the sixteen. The sentinels' *Arabidopsis* is a true fact about
PathBank record 9659 and is not a false mapping.

**D-071** rules PMC12444477: **scope the tolerance per entity, do not flip the Boolean.** Flipping
would newly penalise the seven entities the rationale legitimately excuses. The nine-versus-eight
prose mismatch is *resolved*, not picked: the Raetz pathway has nine steps, the ninth enzyme is
organism-dependent (`LpxH`/`LpxI`/`LpxG`), so the gold is right to list eight expected and file LpxH
under acceptable. **Neither list moves.**

**F-141** registers and classifies the 24 pinned / 82 corpus-wide withheld-identity population that
was hiding inside the O-1 metric. **All 24 are correct withholding and no card follows.** It is not,
and must never be reported as, `placeholder_backed_proteins`.

**C-098a and C-098b are both invalidated** and nothing is salvaged. **C-098c stays refused.**
**C-094 is not merged and not relabelled**; C-099 supersedes it for production purposes.

**T-107 is NO-GO** and no card in this wave could have changed that. See § 5.

## 3. Cards

| Card | Branch | State |
|---|---|---|
| **C-099** — preserve resolved species on Unknown-backed wrappers | `card/C-099-species-preservation` | **MERGED** `9e4a28a`. Gate 570 passed |
| **C-100** — per-entity tolerance scope for PMC12444477 | `card/C-100-tolerance-scope` | **MERGED** `8e5d549`. A/B zero movers on 42 files; gate 898 passed |
| **C-101** — the 16/5 metric split | *(not dispatched)* | chartered, and its dependency is now **satisfied** — C-100 is in. **This is the next card to dispatch.** |

**Two reds that are NOT ours and must not be absorbed as noise.**
`test_strict_failure_replay.py::{test_every_stored_strict_failure_replays_to_its_recorded_verdict,
test_recovered_cases_are_smaller_and_refused_cases_are_not_claimed}[only_unrelated_reactions_survive]`
fail on `f7dc223` **before either card existed**, and were confirmed unmoved three separate times. A
Glutathione payload: no gold case, no `unknown_backed` surface. The fixture records `recovers: false`
while `quarantine_and_close` returns `ok=True`. **Unowned — worth a card.**

**Registered with C-100, both needing the product owner rather than an agent.** Scoping makes the bare
PathBank `Unknown` sentinel a finding on PMC12444477, and D-070 § O-1a rules that sentinel is
PathBank's own legitimate representation — so the pipeline largely cannot clear the finding by doing
anything correct. Recorded as **F-132-class instrument tension**, not a defect; excusing it is an
eighth gold entry and needs **D-071 amended**. And the `lipoprotein` tolerance entry is **inert** —
the token appears in no protein row in any payload; it is kept with an empty quote because fabricating
a span for a token the paper does not contain is the one thing that field forbids.

Charters are `docs/pwml_recovery_sprint/prompts/C-099.md`, `C-100.md`, `C-101.md`.

Worktrees created this wave: `C:/t/c099`, `C:/t/c100`, `C:/t/c099base`, `C:/t/c100base`, plus
`C:/t/c099g9` (a hash-verified `f7dc223` export made by C-099's author because `pinned_pytest`
refuses a selection outside its expected tree). **Prune none of them**, nor the older ones a peer
session listed: `C:/t/c094`, `c094base`, `c095`, `c096`, `c097`, `c098`, `c098a`, `c098b`,
`rev095base`, `rev095m`, `rev096base`, `c092`, `c093`.

## 4. Baselines pinned this wave — use these to attribute movement

| Selection | Result | Where |
|---|---|---|
| SMOKE | **473 passed**, 47.3 s | `evidence/g11/ORCH-711/07` |
| `test_protein_export_policy.py` (in **neither** SMOKE nor Chunk D) | **63 passed** | `evidence/g11/ORCH-711/08` |
| whole-tree G11 | **4246 artifacts, 0 non-compliant** | `evidence/g11/ORCH-711/06` |

## 5. T-107 — NO-GO, and the one question that goes to the product owner

Gate condition 9 (*no absolute acceptance priority guaranteed to fail*) is **not met and is not
reachable by any engineering in this sprint**. `DECISION-BUNDLE-F132-PRIORITY1.md` § 9 already said
so: *"A does not clear it; only B does, because B is the acceptance."*

The bundle asks **two** things. The product owner ruled its **addendum** — O-1, now D-070 — and
PMC12444477, now D-071. **Asks A and B remain open:**

* **A** — reconcile the anchor set against `forbidden_identifiers`, so Priorities 1 and 4/5 stop
  scoring the same rows in opposite directions (**F-132**);
* **B** — accept or decline a **Priority-1 floor of 6** for T-107 purposes.

**Only B clears the gate.** Until it is answered T-107 cannot be scheduled, and that is the correct
outcome rather than a delay. Merging does not move it: C-092 and C-093 did not, and C-099 corrects a
false *species*, not a false *identifier*, so it cannot move Priority 1 in either direction.

## 6. Traps this wave paid for — in addition to the handoff's standing list

* **The pinned run is safe as a POPULATION and unsafe as a BEHAVIOUR.** `runs/2026-08-02_2130`
  predates D-003; `verification_status` and `unverified_identity_claim` are absent from every row.
  Counts taken from it stand; behavioural claims need re-measuring against a current run.
* **`runs/` and `runs_verify/` are both live.** "The pinned run" meant `runs/2026-08-02_2130`. The
  same criterion against `runs_verify/2026-08-24_1428` yields **5**, not 24. Nothing in the packet
  said which. **Always name the tree.**
* **A zero-hit grep on a document's vocabulary proves nothing.** `PRODUCT_CONTRACT` says
  `unverified_claim`; the code says `unverified_identity_claim`. Ask what the *code* calls it.
* **A two-sided obligation needs the side that must NOT fire tested.** § 8's carrier is correctly
  absent on `rejected` rows; a presence-only probe would have missed a carrier firing there.
* **`bounded_run.py`'s report carries no child stdout.** Commit the probe *and* its log, or the
  certificate proves a job was clean while preserving nothing about what it found.
* **`iter_reports` selects every non-dot file in a task folder** — a `README.md` inside
  `evidence/g11/<TASK>/` comes back `unexpected_artifact` and reddens the gate. Index files go in
  `docs/pwml_recovery_sprint/`.
* **Agent worktrees have no `.env` and no `.venv`.** The DB is hidden, so a green base leg can be
  green for the wrong reason. Make every agent *probe* and *state* which state its tree is in.

## 7. Peer sessions share this working tree

Two other Claude sessions were live during this wave and both were contacted and confirmed clean —
not pushing, no Python, not holding the lock. One was the previous Lead Orchestrator (author of the
ORCH-710 evidence), standing down; the other acted as independent adversarial reviewer.
**Session identities are not stable** — a peer was renamed mid-conversation. **Run `ListAgents` and
contact every peer before treating the branch, the lock or the worktrees as exclusively yours.** A
peer caught a branch moving under a session that believed it was untouched earlier in this sprint.

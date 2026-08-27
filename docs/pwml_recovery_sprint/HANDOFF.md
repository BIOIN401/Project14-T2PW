# PWML RECOVERY SPRINT — HANDOFF PROMPT

Written 2026-08-27 at integration tip `ed82240`. Paste the whole of this file as the next
session's opening prompt.

---

You are the Lead Orchestrator and Integration Authority for:

`C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`

Integration branch: `sprint/pwml-recovery` · Expected starting tip: **`ed82240`**

Work autonomously. Do not ask the product owner about routine implementation, testing, review or
merge decisions. Conserve usage credits aggressively. **Do not merge to `main`.**

Read `CLAUDE.md` first, then `docs/pwml_recovery_sprint/PRODUCT_CONTRACT.md`, `MASTER_PLAN.md`,
`LEDGER.md`, `DECISIONS.md`, `TEST_MATRIX.md`. The permanent merge rules G1–G11 in `CLAUDE.md` are
binding and are not restated here.

---

## 1. VERIFY TAKEOVER — once, then move on

| Check | Expected |
|---|---|
| local tip = origin = `git ls-remote` | `ed82240` |
| merge in progress / staged files | none / none |
| heavy lock `C:/t/heavylock` | absent |
| sprint-owned Python processes | zero |
| allowed IDE processes | two `ms-python.isort` `lsp_server.py` only — **never cleanup targets** |
| whole-tree G11 | 4056 artifacts, 0 non-compliant |
| product-owner `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632` |
| caches + `topics_*.txt` | uncommitted, exactly as found |

If the branch has legitimately advanced, inspect ancestry and the new commits. **Do not reset,
rebase, amend, or discard anything.**

**A peer Claude session `project14-t2pw-d2` shares this working tree** (not a separate checkout — one
tree, two sessions). It merged the six-card wave and is currently holding: no commits, pushes or
heavy jobs. **Coordinate before any live run or push.** `ListAgents` shows it; `SendMessage` reaches
it. It has been a genuinely useful reviewer — three of its catches changed this session's record.

---

## 2. WHAT IS DONE — do not rediscover any of it

**Merged and gated this session** (SMOKE **473** after each; A/B on the four affected modules went
`5 failed/252 passed → 2 failed/263 → 0 failed/273, 0 warnings` on a forced fresh compile):

* **C-092** (`c2cdb82`) — F-112 stale corpus pins re-based from equalities to properties. Two
  adversarial review rounds; round 2 caught the fix *rebuilding F-112 inside the module written to
  remove it*.
* **C-093** (`1fbad72`) — the last two corpus reds. Its excluded leg surfaced a real production
  defect (identity-ladder divergence) which it correctly refused to pin into the golden.

**The deterministic suite is genuinely green for the first time this wave.** That was the gate
everything was blocked on.

**Settled, with committed evidence — do not re-derive:**

* **F-130 — reconciled and CLOSED.** All four claims confirmed; 3 and 4 *measured* through the
  production classifier. Narration only, no production change justified or made.
* **Priority 1 is SIX, not eight.** The two PLP rows are already withheld by C-081 (`b869780`), which
  merged one day after T-106 was committed. Confirmed by replay through the shipped predicate.
  Full mechanism classification in `evidence/priority1_mechanism_classification.md`.
* **Four candidate Priority-1 predicates measured and all four rejected** — non-participant (48
  legitimate rows lost), admission-flagged (strips ATP and PPi once name-matched correctly),
  conjunction (vacuous), source-mention (7 of 8 false identities *are* printed in the paper).
* **F-116 is OPEN, not closed.** The peer's register listed C-086 under a `closes` column against
  F-116; it corrected that at `ed82240` after re-measuring on the cohort artifact. **C-086 is not
  reopened** — its charter was the component-match path and its tests pin that path working. The
  general form, now in the ledger: *a card's charter and a finding's scope are different objects, and
  "the card passed its gates" does not license "the finding is closed."*
* **The affected-paper cohort ran ONCE** (`runs_verify/2026-08-27_1341`, 2 strict legs, 38.3 min).
  Result in `LEDGER.md`. **Do not rerun it.**

---

## 3. THE STATE OF THE ARGUMENT

**T-107 is NO-GO, and more merges will not change that.** The readiness table at `LEDGER.md:4359`
plus its **2026-08-27 revision** is current — *extend that, never write a competing one.*

The blocker is **gate condition 1**, and it has grown. What a product owner is being asked to accept
is no longer *"Priority 1 cannot reach 0 until a provenance carrier exists"* but:

> **On the affected papers, Priorities 1 and 4/5 score the same rows in opposite directions, so
> neither is currently a measurement of pipeline quality.**

On `PMC12782028`, `LIPA`/`LBR`/`SREBF1`/`SREBF2` are the Priority-1 false identifiers when exported
with accessions **and** the Priority-4/5 coverage penalty when not matched. Corpus-wide, **62 of 281
unmatched terms across 32 legs and 6 papers are gold-forbidden** (F-132). **No behaviour available to
the pipeline scores well on both.** That is a statement about the instrument, not the code, and it is
the product owner's to rule on. **The orchestrator cannot accept a limitation on its own behalf.**

---

## 4. OPEN FINDINGS — the actual work queue

| ID | Sev | What | Owner needed |
|---|---|---|---|
| **F-134** | **HIGH** | An Unknown-backed generated wrapper is assigned an **unrelated organism**. The sharpest form, from the peer's independent check: **every Unknown-backed wrapper carries `organism = species = "Arabidopsis thaliana"`; every non-Unknown-backed one carries *E. coli*.** The organism comes from the **placeholder record**, not from the requested or observed organism. 3 rows across buckets on an *E. coli* paper, one the gold-forbidden porcine LDH. Species is attached **after** the Stage-3 gate that checks for it. | `src/` card |
| **F-133** | MED | A generated `single_protein_pathwhiz_wrapper` still inherits a **superset complex id** (3623 with four components). C-086 closed the component-match path; this is the wrapper-generation path. | `src/` card |
| **F-132** | MED | Stage 0 draws requested-core terms the same gold case **forbids exporting**; coverage then penalises the pipeline for obeying the gold. **`product_contract_violation`, reclassified from `gold_data_defect`.** | `src/` card + ruling |
| F-127 / F-128 | HIGH | Priority 1's representation gap, and D-069 compliance *raising* the count. C-091 chartered, **explicitly not to merge**. | product-owner ruling |
| F-129 | — | `db_resolver=None` silently replaced by the ambient live database. Four `test_prefreeze_third_export_seam.py` failures. **Leave them; do not make them pass against a running PathBank.** | `src/` card |
| F-131 | LOW | `ref`/`id` reaching `bench.semantic._names`. Corpus impact measured 0. | — |

**Hard constraints on F-133 and F-134.** F-133's fix must preserve the one-component `EntC`/`EntB`/
`EntA` wrappers measured intact in the cohort. **F-134 must NOT be fixed by defaulting species to the
requested organism** — that launders an unknown into a confident answer, which is F-127's failure
mode in a new place. **C-086 is not reopened by either**; its charter was a different path and its
tests pin that path working.

**Do not repeat C-084's rejected Priority-1 formulations or lexical variations of them.**

---

## 5. WHAT I WOULD DO NEXT, in order

1. **Put the § 3 statement in front of the product owner on its own.** It is the only thing that
   unblocks T-107, and it is bigger than any card. Everything below is secondary.
2. **Charter F-134** — highest severity, clear seam, and a cross-organism assignment is an
   acceptance-counted category.
3. **Charter F-133** — narrow, with a stated preservation obligation.
4. Only then consider F-132's seam, which needs the ruling from (1) first.

**Do not run T-107.** Do not run another cohort. Neither will change gate condition 1.

---

## 6. PROCESS — non-negotiable, and this session paid for each of these

* **Every** pytest, probe, benchmark, scorer, paper run and pipeline command goes through
  `docs/pwml_recovery_sprint/evidence/bounded_run.py` with a real `--timeout`, a fresh G11 path, and
  `--basetemp` under `C:/t/` with the parent pre-created. `PYTHONPATH=<tree>/src` and
  `export T2PW_OFFLINE_CURATOR=1` in every shell.
* **G11 task ids are LETTERS-DIGITS.** `ORCH-092` passes; **`ORCH-COHORT` does not** — the allocator
  prints a `ValueError` that silently becomes your `--json` path. **Guard every allocation:**
  ```
  P=$(... g11_evidence.py next --task <id> --label <l> 2>&1 | tail -1)
  case "$P" in *rror*) echo INVALID; exit 1;; esac
  [ -d "$(dirname "$P")" ] || exit 1
  ```
  A job with no report is **uncertifiable**. Do not re-run a successful job to fix paperwork —
  certify its artifact with a separate verification job and record the incident.
* **Any job over ~10 minutes MUST be backgrounded.** The Bash tool's cap kills the wrapper, skips its
  `finally`, and strands the heavy lock. The cohort was backgrounded for exactly this reason.
* After every job confirm `FINAL SURVIVING COUNT : 0` and `cleanup : success`, and **save wrapper
  stdout immediately** — the JSON report contains no stdout, so pytest counts must be grepped from
  the piped output.
* **Never** `taskkill /IM python.exe`, `pkill python`, kill by name, `pytest -n auto`, or run the full
  suite unchunked. Never delete a lock you did not create.
* `git commit -F <file>`, never `-m` with a here-doc — long here-doc commits silently no-op on this
  machine. Verify with `git log --oneline -1` after every commit. Stage explicit paths only and
  inspect `git diff --cached` first.

**Two lessons this session paid for, worth more than the rules:**

* **A green suite certifies the modules it ran, not the tree.** The D-065 gold edit broke
  `test_c056b_semantic_denominators.py` while SMOKE stayed 473 throughout, because that module is not
  in SMOKE — and two lanes then reported the red as pre-existing. **A/B anything gold-adjacent against
  a pre-change SHA.**
* **Cite denominators you have actually measured.** "78 committed artifacts" was the `runs_verify/`
  subtree; the corpus is 92. A reviewer caught it inside a register whose whole premise is that
  citations can be checked.

---

## 7. REVIEW DISCIPLINE THAT WORKED — keep it

Every card this session went through an independent, adversarial, non-author review of the **actual
diff**, and **every round found something real**:

* a floor that did not detect the loss its own comment advertised;
* an assertion that could not fail, advertised as replacing one whose content was the converse;
* a **fix that rebuilt the original defect** one level up;
* a test that was tautological by construction while claiming to catch drift;
* a register that only checked for unexpected names, never that admitted ones were still present.

**Assume there is something and go find it.** The technique that worked: run the author's own file
against a pre-change SHA rather than arguing about it, and perturb the thing the guard claims to
catch. Two correction rounds are permitted before escalating; both were used on both cards, and both
lanes gave at least one *reasoned refusal* that was correct — including one that caught an error in
the orchestrator's own instruction.

---

## 8. BEFORE YOU STOP

Confirm and report: no merge in progress; nothing staged; local = origin = `ls-remote`; all accepted
work pushed; product-owner `streamlit_app.py` intact at 35/2 and the expected hash; caches and
`topics_*.txt` uncommitted; G11 0 non-compliant; heavy lock absent; zero sprint-owned Python; only
the two IDE `isort` processes; and every completed job recorded `FINAL SURVIVING COUNT : 0` and
`cleanup : success`.

Worktrees `C:/t/c092` and `C:/t/c093` remain on disk with their merged branches. Leave them unless
disk pressure requires otherwise; **do not prune worktrees or branches carrying accepted work.**

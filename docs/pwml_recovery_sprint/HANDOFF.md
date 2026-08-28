# PWML RECOVERY SPRINT — HANDOFF PROMPT

Written 2026-08-27 at integration tip `e25247b`. Paste the whole of this file as the next session's
opening prompt. **It replaces the previous handoff, which was written at `ed82240`.**

---

You are the Lead Orchestrator and Integration Authority for:

`C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`

Integration branch: `sprint/pwml-recovery` · Expected starting tip: **`e25247b`**

Work autonomously. Do not ask the product owner about routine implementation, testing, review or
merge decisions. Conserve usage aggressively. **Do not merge to `main`.**

Read `CLAUDE.md` first, then `PRODUCT_CONTRACT.md`, `MASTER_PLAN.md`, `LEDGER.md`, `DECISIONS.md`,
`TEST_MATRIX.md`. The permanent merge rules **G1–G11** are binding and are not restated here.

---

## 1. VERIFY TAKEOVER — once, then move on

| Check | Expected |
|---|---|
| local tip = origin = `git ls-remote` | `e25247b` |
| merge in progress / staged | none / none |
| heavy lock `C:/t/heavylock` | absent |
| sprint-owned Python | zero |
| allowed IDE processes | two `ms-python.isort` `lsp_server.py` — **never cleanup targets** |
| whole-tree G11 | **4235 artifacts, 0 non-compliant** |
| product-owner `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632` |
| caches + `topics_*.txt` | uncommitted, as found |

**Re-read `git rev-parse` on every branch immediately before you act on it, including SHAs written in
this file.** A tip SHA is not a fact you can hold across messages — that cost this session a wasted
correction round when a working-tree read was mistaken for a committed state.

---

## 2. WHAT IS DONE — do not rediscover

**Merged, each with SMOKE 473 and zero survivors:**

* **C-096** (`cee45f1`) — F-129. `db_resolver=None` meant both *"unspecified, open the ambient
  PathBank"* and, unreachably, *"resolve nothing"*. `NO_DB_RESOLVER` adds the third state; `None` is
  untouched, because `PRODUCT_CONTRACT` § 8 forbids the exporter opening the connection.
* **C-095** (`13b5696`) — F-133, the remaining open path of **F-116**. A generated one-protein wrapper
  no longer inherits a superset complex id. Affected set re-run on the **combined** tree: **196
  passed** (182 + 14).
* **C-097** (`b35b6a2`) — F-131. `bench.semantic._names` stops reading the legacy `ref`/`id` tail.
  One executable line. REV-097 verified the zero-corpus-impact claim on a **larger** population than
  the card measured — 39,542 dicts at every depth, **0 legacy keys under `/processes`** — and proved
  the zero live with a two-process A/B that moves **6 → 100 orphans** under injection.

**Held, all correct, all blocked on one ruling — see § 3:**

| Branch | Tip | State |
|---|---|---|
| `card/C-094-f134` | `53eaf24` | REV-094 **APPROVE WITH CORRECTIONS**; blocked on O-1 |
| `card/C-098a-cap` | `8cfa33e` | inert cap arm; held with C-098b |
| `card/C-098b-gate` | `b589821` | gate arm; **not merged**, see § 3 |

Worktrees on disk, none to be pruned: `C:/t/c094`, `c094base`, `c095`, `c096`, `c097`, `c098`,
`c098a`, `c098b`, `rev095base`, `rev095m`, `rev096base`, plus the older `c092`/`c093`.
(`c097base` was removed after REV-097 — it carried no accepted work.)

**Settled — do not re-derive:** the F-132 corpus figures (62 of 281 unmatched terms, 32 legs, 6
papers); Priority 1 is **six**, not eight; the affected-paper cohort ran once at
`runs_verify/2026-08-27_1341` and **must not be rerun**; C-086 narrowed F-116 and is not reopened.

---

## 3. THE ONE THING THAT MATTERS — rule **O-1**, or nothing else moves

`DECISION-BUNDLE-F132-PRIORITY1.md` now carries **two** independent asks. The second is the blocker
for everything held above.

**O-1** (`DECISIONS.md:938`), verbatim: *`placeholder_backed_proteins`: gold-set error class, or
legitimate biology preservation? · Blocks: **any branch that touches protein export policy** · "a
genuine disagreement between two intentional designs, not a defect. **TRAP-3 forbids agents from
resolving it.**"*

**C-094 inverts a pinned product statement on that exact surface — by consequence, not in prose**,
which is worse, because nothing in the diff announces it.
`tests/test_protein_export_policy.py::test_strict_gates_accept_a_correctly_formed_unknown_backed_complex`
asserts today, and passes at `14121d5`, that an `Unknown`-backed complex passes **all three gates
including `validate_required_pwml_contract`**. C-094 makes it fail.

**Do not merge C-094, C-098a or C-098b until O-1 is ruled.** Four baseline moves are unauthorized and
none has been edited.

**A trap I fell into, recorded so you do not.** I first framed this as *"C-094 stops fabricating the
field that made unexportable entities look exportable."* That states one side of O-1 as fact. Under
the preservation reading the sentinel's species is **not** a fabrication — it is part of a coherent
*"this row is the PathBank Unknown record"* marker. What survives without a ruling is narrower: the
measured row is **internally contradictory** (`species: "Arabidopsis thaliana"` beside
`species_name: "Escherichia coli"`, `taxonomy_id: "562"` and an *E. coli* `species_ref` at confidence
1.0), and a **released** payload carried it — `runs_verify/2026-08-04_1754/papers/PMC12856317/strict`
shipped `pathway.pwml` with *Arabidopsis* on a **human** ALAS2 wrapper.

**Why the chain stopped at three cards.** `validate_required_pwml_contract` calls itself the
PWML-ready contract and raises `protein_complex_missing_species` as an **error**; and
`writer.py:1137-1165` ends its species chain at **`return default_species_id`**. A fourth card
punching through that gate would swap a false *Arabidopsis* at mapping time for a false default at
**export** time — merge rule 8, and the defect recreated one stage later. **C-098c is refused. Do not
charter it.**

---

## 4. OPEN FINDINGS

| ID | Sev | What |
|---|---|---|
| **F-135** | HIGH | The placeholder-species question. **= O-1.** Escalated, packet written |
| **F-132** | MED | Priorities 1 and 4/5 score the same rows in opposite directions. Packet written |
| F-127 / F-128 | HIGH | Priority 1's representation gap. C-091 **explicitly not to merge** |
| **F-136** | MED | A **third** ambient-dependent test — `test_streamlit_quarantine_boundary.py::test_research_mode_keeps_the_unmapped_candidate_and_does_not_block`. **F-129's class is narrowed, not closed.** Consequence: **Chunk D cannot go green in this environment at base or tip with the DB up** |
| **F-137** | MED | `NO_DB_RESOLVER` is absorbed by `_REVIEW_REQUIRED_REASONS` and demotes release status under a false `db_unavailable`. Outside C-096's boundary |
| **F-138** | LOW | `map_ids.py:6169` (C-086's function) carries the same false "the two seams cannot disagree" sentence C-095 removed. Comment-only |
| **F-139** | LOW | C-095's carve-out comment justifies only the legacy marker, not `generated: True` with no reason. Malformed-input only |
| **F-140** | LOW | A pin verdict records the tree and selection but **no hash of the source under test**, so a run cannot be attributed to a file state after the fact. Latent everywhere |

---

## 5. WHAT I WOULD DO NEXT

1. **Put O-1 in front of the product owner and stop.** Everything in § 2's held table waits on it.
2. **F-137**, **F-140** and the `writer.py` `default_species_id` seam are chartered-able **without**
   a ruling. They are the only genuine engineering left.
3. **Do not run T-107.** Do not run another cohort. Neither changes gate condition 1.

---

## 6. PROCESS — what this wave paid for

Everything in the previous handoff's § 6 still holds. **New, and each cost real time:**

* **Exported `PATHBANK_DB_*` cannot hide the database.** `src/t2pw/llm/client.py:22` calls
  `load_dotenv(dotenv_path=ENV_PATH, override=True)` and re-applies `.env` over your exported values
  for anything importing the LLM client. **Only physically renaming `.env` works**, with a
  `trap … EXIT`. A reviewer voided two of its own jobs discovering this.
* **An agent worktree may have no `.env` at all** — the opposite hazard. A card's four target tests
  were green at base in a worktree purely because `from_env()` returned `None`, which would have made
  the delta meaningless. **State which state your tree is in.**
* **A `git archive` base export has no `.git`**, so anything shelling out to `git ls-files`
  degenerates silently — four corpus tests skip. Use a **real git worktree** or an in-tree A/B with
  the restore verified by `git diff --stat`.
* **G11 task-id suffixes must be lowercase.** `C-098A` is rejected; `C-098a` passes.
* **Pin verdicts go in `evidence/g11/pin/<TASK>/`.** Putting them elsewhere left ten reports with
  dangling `--pin-verdict` pointers. Not re-run — `TEST_MATRIX` § 0 forbids re-running a green job to
  repair paperwork — but avoid it.
* **`cmd | tee log | head -N` truncates the log via SIGPIPE.** Redirect, then grep.
* **`git checkout -- <file>` in a restore trap silently un-does your own fix when base == HEAD.**
* **A gate-invisible baseline is a statement that can be inverted silently.**
  `tests/test_protein_export_policy.py` is in **neither SMOKE nor Chunk D**; three of its tests moved
  and only a reviewer selecting the file by hand caught it. **Second time this wave a gate-invisible
  file hid a real move.**

---

## 7. THE REVIEW RECORD — every round found something real

Six review rounds, six findings that changed the work. Keep the discipline.

* **REV-096** built a **mutation matrix** and *corrected both the author and the source comment*: arm
  order is irrelevant, the **`elif`** is the invariant. It also ran a 45-scenario differential proving
  `None` byte-identical, and **retracted its own method** on discovering the `load_dotenv` problem.
* **REV-094** found **three moved baselines nobody had run**, in a gate-invisible file — one of them
  the O-1 statement itself.
* **REV-095** found a guard that **fired on results conferring no identity**, recording a refusal of a
  complex the row never matched — `candidates[0]` of a ten-way *ambiguous* lookup. Its delta review
  then verified the fix by **tabulating all eleven returns** in the loop from source.
* **C-098b** measured a **second refusal point** and reported its § 7.1 target unmet rather than
  claiming success — after **correcting its own earlier measurement**, which I had already acted on.
* **C-095's author proved REV-095's unproven F2 against its own card**, and caught its own **vacuous
  fixture**, committing the vacuous run's report rather than replacing it.

**Two authors committed their own wrong measurement beside the right one. That is the behaviour to
want** — a quietly corrected probe leaves the record un-auditable, and in both cases the error was one
I had already built on.

**The pattern worth stating plainly:** *a refusal record is a claim, and a claim needs the same proof
as the behaviour it describes.* Four separate times this wave a guard was demonstrated against a case
that could not exercise it, or fired on one it should not have.

---

## 8. BEFORE YOU STOP

Confirm and report: no merge in progress; nothing staged; local = origin = `ls-remote`; all accepted
work pushed; product-owner `streamlit_app.py` intact at 35/2 and the expected hash; caches and
`topics_*.txt` uncommitted; G11 0 non-compliant; heavy lock absent; zero sprint-owned Python; only the
two IDE `isort` processes; and every completed job `FINAL SURVIVING COUNT : 0` / `cleanup : success`.

**T-107 is NO-GO.** Three of eight § 8 conditions unmet, and condition 1 cannot be cleared by any
engineering — only by the ruling in § 3. **No live paper leg, no cohort and no T-107 run happened this
wave, and none was needed.**

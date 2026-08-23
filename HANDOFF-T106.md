# Handoff — finish the correction wave, then run T-106

Written 2026-08-23. Integration tip **`f12115a`** on `sprint/pwml-recovery`, pushed,
`local = origin = git ls-remote`.

---

## 0. Verify once, then work

| check | expected |
|---|---|
| tip | `f12115a`, local = origin = `git ls-remote` |
| merge in progress / staged | none / 0 |
| heavy lock `C:\t\heavylock` | absent |
| sprint-owned Python processes | 0 (two `ms-python.isort` IDE servers — **never kill those**) |
| product-owner edit `src/t2pw/app/streamlit_app.py` | **35 ins / 2 del, uncommitted**, file `sha256:e50a248bb7189c22…` |
| SMOKE | **473** (measured six times this session; `CLAUDE.md`'s 465 is stale by the C-067 delta) |
| `TEST_MATRIX.md` | 578 lines, line 477 byte-identical (D-061) |

Expected untracked and correct: `topics_flip_strict.txt`, `topics_regression_research.txt`,
`topics_t100.txt`, `topics_verify_subset.txt`.

**Never** commit `data/enrichment_cache.json` or `data/id_mapping_cache.json`.
**Never** stage, stash or commit `streamlit_app.py` — see §4 for the one exception and how it was done.

---

## 1. What is done

**T-105 ran and is frozen as `MEASURED — NOT ACCEPTED`.** `runs_verify/2026-08-22_2147`, 4.85 h,
`COMPLETE (10/10 papers, 20/20 legs)`. **Do not rerun it, do not rescore it into its own record.**
Reports: `docs/pwml_recovery_sprint/evidence/t105_acceptance_report.{txt,json}`.

Four cards merged this session, each independently reviewed against the actual diff:

| card | merge | what it closed |
|---|---|---|
| C-072 | `d7f4f96` | F-094 — a declared core with unmatched anchors is never `release_ready` |
| C-073 | `6373ad1` | F-096 (part) — an accession claimed across incompatible entity kinds does not ship |
| C-074 | `861c796` | **F-100, F-101, F-103** — connected-pathway floor + an unstated request is unjudgeable |
| C-075 | `81b0bf9` | F-096 (rest) — source-support pass armed and enforced |
| wiring | `f12115a` | `source_text=text` at the one call site the app owns |

Findings registered: **F-098** through **F-105**. All in `FINDINGS.md`.

---

## 2. Do this next, in order

### 2a. Register F-106 — RAG provenance marks are self-inconsistent

Raised by the C-075 reviewer, not yet written up. Two corpus rows carry
`rag_provenance` with `source_type: "paper"`, a `source_title` that is a **pathway name**
("menaquinone biosynthesis", "lipid A modification (colistin resistance)"), and
`source_id: "seed_paper"`. Those three cannot all be right.

It matters because **C-075's entire route clause rests on trusting that mark**. It is a Stage-1 RAG
synthesizer data-quality question, not C-075's defect, and it does not change C-075's verdict —
but nobody has written it down. Rows: `PMC12312563/strict` `α-ketoglutarate`,
`PMC13278307/strict` `pmrCAB operon`.

### 2b. Dispatch C-076 — chartered, never dispatched

`docs/pwml_recovery_sprint/prompts/C-076.md` is complete and ready. Set its base SHA at dispatch.
It closes **F-102**: the acceptance scorer still flags the within-kind accession rule that C-073's
review **rejected** as contradicting D-035 clause 3c. The pipeline now follows D-035; the scorer
does not.

Two things the card turns on, both already measured:

- `EntE` / `enterobactin synthase` and `EntB` / `holo-EntB` are the same protein identity and must
  stop being conflicts.
- The gold entry lumps five aliases together, and **they are not the same case**: `apo-EntB`,
  `holo-EntB`, `apo-Fur`, `holo-Fur` are modification states of one polypeptide and may share the
  parent accession under the 2026-08-23 ruling; **`R196A` is a site-directed mutant — a different
  polypeptide — and stays forbidden.** Do not collapse that distinction.

**C-076 changes measured numbers.** Report the before/after delta for T-106's baseline. Do not
apply it retroactively to T-105.

### 2c. Run the affected-paper validation — REQUIRED before T-106

Nothing since the merges has been validated on a real paper. Minimum cohort:

* `PMC12856317` strict + research — must no longer emit an unjustified bare `pathway.pwml`
* `PMC13231680` strict + research — must remain the correct negative control
* `PMC12180156` research — `succinyl-CoA` must now be refused (**this is the first real test of
  the armed pass**)
* `PMC12782028` research

Use `--fresh` for a new run directory. Without it `batch_run.py` continues an existing directory
and **silently skips already-finished pairs** — that trap cost a leg earlier this session.

Confirm before T-106: unsupported entities acquire no real identifiers; supported aliases keep
theirs; EntE/full-name and holo/apo handled per the ruling; no biological gate weakened to raise
the strict rate.

### 2d. Then T-106

Fresh release candidate, same pinned 10-paper / 20-leg plan, `topics_t104.txt`, configured
`deepseek/deepseek-v4-flash`. **Do not reuse the T-105 identity.** Preflight
(`--fresh --stage-only`), then `--verify-plan` (expect `verdict: OK`, 10/10 `[pinned_override]`),
then the real run **without `--fresh`**, background, ~5 h.

---

## 3. Predict before you score

**A clean preflight funnel means nothing about scope conflicts** — Stage-0 reconciliation is
per-leg, so all three organism-trap papers pass eligibility and conflict later.

Expected unchanged, **not findings**: 6 × `scope_conflict` (D-062 ruled but never implemented —
charter preserved in this session's scratch and in the LEDGER), 2 × PMC12444477 TIMEOUT
(F-092, open and unowned).

**What T-105 taught about prediction:** priority 1 came in at exactly the predicted 7 — and the
match was coincidence. `succinyl-CoA`, `SREBF1/2`, `LIPA`, `LBR` all vanished by draw variance and
were replaced by `protoporphyrin IX`, `NADH`, `NAD+`, `holo-EntB`. **Predict compositions, not
counts.**

Also expect the negative control to land `review_required` rather than the gold's "empty pathway
plus a rejection reason". Merge rule 7 forbids dropping the payload, so `review_required` is the
closest permitted outcome and is what C-074 chartered. **Closing that gap is a product decision
about which rule yields — do not take it in a card.**

---

## 4. How the `streamlit_app.py` line was landed, in case it is needed again

The product owner's 35-line edit is uncommitted and must stay that way, which normally blocks
merging any branch that touches the file. The wiring was landed without stashing:

1. edit the file in the working tree;
2. `git diff -- <file> > full.patch`, then split it into hunks and keep only the new one;
3. `git apply --cached mine.patch` — stages **only** that hunk;
4. commit. The product-owner hunks stay unstaged in the working tree.

Verified after: `git diff --cached` showed exactly `1 insertion(+)`, and
`git diff` still showed `35 insertions(+), 2 deletions(-)`.

---

## 5. Traps this session paid for — do not pay again

- **`--fresh`**: omitting it silently skips finished pairs on a rerun.
- **Bash heredocs break on apostrophes**, even quoted. Write prose with the Write tool to
  scratch, then `cat >>`.
- **`bounded_run.py`'s G11 JSON has no stdout.** Grep pytest counts from the piped wrapper output
  or they are gone and the job must be re-run.
- **A job with thousands of short-lived children can never produce a compliant G11 report**
  (F-104 / F-090). A clean run can still fail `check`; that is not a lifecycle violation.
- **Single-run corpus figures are not safe.** Twice this session a measured "0 collateral" did not
  survive widening the corpus — C-073's first predicate (41 legitimate rows stripped over 53
  artifacts) and its Pass A as shipped (36 legitimate refusals over 70). **State sample size in
  every card and make the implementer re-derive.**
- **My charters kept asserting mechanism from reading rather than measurement** — three were
  corrected by implementers (C-073 §2 sample size, C-073 §4a wiring premise, C-074 §2 mechanism).
  Write cards that say what was measured and how, then require re-derivation.

---

## 6. Open, unowned, not blocking

**F-092** (HIGH) — identical wall-clock timeouts record different terminal reasons, never
`operation_timeout`; runner hard-codes "produced nothing". Re-confirmed on both PMC12444477 legs at
T-104 and T-105.
**F-099** — withholding a PathBank scalar is not durable to pre-freeze resolution
(`compound_resolution.py:503`). **Now reachable, because the pass is armed** — card it with F-105.
**F-105** — the source index rides into the interactive curator prompt at a second site
(`interactive_curator.py:164` is a blacklist). Not reachable on batch legs; must close before the
interactive app is used against a real paper.
**F-090 / F-104**, **F-086/087/088/089** — minor, recorded.
**D-062** — locked ruling, never implemented; charter and evidence in the LEDGER. Do not resolve it
by editing `topics_t104.txt`.

---

## 7. Process — non-negotiable

Every test, paper run, benchmark and probe goes through
`docs/pwml_recovery_sprint/evidence/bounded_run.py` with a real timeout, a fresh G11 allocation
(`evidence/g11/g11_evidence.py next --task <id> --label <label>` — the allocator lives in
`evidence/g11/`, not `evidence/`), a unique short `--basetemp` under `C:/t/`, and the heavy lock
where required. Any timeout over 10 minutes runs in background mode or the tool clock kills the
wrapper before its `finally`.

After every job confirm **`FINAL SURVIVING COUNT : 0`**, **`cleanup : success`**, lock released,
and no sprint-owned Python left. Never `taskkill /IM python.exe`, never `pkill python`, never
`pytest -n auto`, never the unchunked full suite (~16 GB). One heavy job at a time. A stale lock may
be removed only after reading `C:\t\heavylock\holder.json` and proving its PID dead.

Two implementation lanes maximum, plus a reviewer/measurement lane. Reviewers read the **actual
diff**, never the report. Up to two correction rounds automatically. Merge serially with `--no-ff`.
**Never merge to `main`.**

---

## 8. Leftover worktrees

`C:/t/c074` (`68107e0`), `C:/t/c074base` (detached `9cb491c`), `C:/t/c075` (`dc0ec25`) — all merged,
safe to remove with `git worktree remove` whenever convenient.

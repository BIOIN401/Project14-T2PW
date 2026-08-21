# T-101 and T-103 — authorization package

Prepared by the Lead Orchestrator, 2026-08-21, at integration `bffb62f`.
**Nothing in this document has been run. No live leg has been started and no money has been spent.**

Everything below is measured from source and configuration, with the command and the file cited. Where a
fact could not be established without an outward-facing call, it is marked **UNVERIFIED** rather than
estimated.

---

## 1. The headline — the cost question is not the money question

**Every OpenRouter model slot in `.env` is set to `openrouter/free`.** All nine:

```
OPENROUTER_MODEL=openrouter/free            OPENROUTER_PREPROCESSOR_MODEL=openrouter/free
OPENROUTER_EXTRACTION_MODEL=openrouter/free OPENROUTER_INFERENCE_MODEL=openrouter/free
OPENROUTER_AUDIT_MODEL=openrouter/free      OPENROUTER_CURATOR_MODEL=openrouter/free
OPENROUTER_GAP_MODEL=openrouter/free        OPENROUTER_OVERWATCH_MODEL=openrouter/free
OPENROUTER_FINAL_COMPLETENESS_MODEL=openrouter/free
```

with `LLM_PROVIDER=openrouter` (`.env:2`).

**So the marginal monetary cost of both milestones is approximately zero.** The binding constraints are
**free-tier rate limiting** and **wall clock**, not spend.

**Credential hygiene checked and clean:** `.env` is **not tracked** (`git ls-files --error-unmatch .env`
fails) and is covered twice by `.gitignore:1,3` (`.env.*` and `.env`). An `OPENROUTER_API_KEY` is present and
uncommented. **Its value has not been reproduced in any document, log, commit or message, and must not be.**

**UNVERIFIED, and deliberately so:** whether the key currently has usable credit and what its free-tier rate
limits are. One free, read-only `GET https://openrouter.ai/api/v1/key` would settle both. **I did not run
it** — it is an outward-facing call with the product owner's credential, it blocks nothing, and it is the
product owner's to authorize. Say the word and it takes seconds.

---

## 2. T-101 — M2

### Acceptance, verbatim (`TEST_MATRIX.md:478`)

> *no leg reports "produced nothing"; `identical_empty_response` recorded where two draws share a hash;
> **`budget_exhausted` distinct from failure***

### ⚠ Sequencing — the third clause is literally F-070

**Clause 3 is the C-064 fix.** Run T-101 **after C-064 merges**, or the clause is unassessable.

**Honest qualification:** F-070 binds T-103 and T-104 **by name** and does not name T-101, and T-101's legs
are ordinary pipeline legs that may never enter `run_rag_loop` — whose only production caller
(`streamlit_app.py:1437`) has existed only since C-055. **Two independent seams now emit the string
`budget_exhausted`**: `loop_policy.py:44` and `pipeline/deadline.py:84`, which F-070 measured as not feeding
each other. So clause 3 may already be assessable against the pipeline seam alone. **Sequencing after C-064
removes the ambiguity for free** and is the recommendation.

### Papers and scopes — one had to be recovered

| Paper | Scope | Source of the scope |
|---|---|---|
| PMC12444477 | `lipid A biosynthesis \| Escherichia coli` | `topics_regression_research.txt`, `topics_verify_subset.txt` |
| PMC12782028 | `cholesterol biosynthesis \| Homo sapiens` | `topics_flip_strict.txt` |
| PMC12312563 | `menaquinone biosynthesis \| Bacillus subtilis` | **recovered this session** — see below |

**PMC12312563's scope was recorded nowhere in any topics file, and `topics_verify_subset.txt:13` lists it as
a "Stage-0 scope abort".** That is what happens when it runs **scopeless** — the tree shows both outcomes:

* `runs/2026-07-27_1623/papers/PMC12312563__structures-of-listeria-monocytogenes-mend-in-th/` — **slugged, so
  scoped, and it succeeded**; its strict leg is in `GOLDEN`.
* `runs/2026-08-01_2000/papers/PMC12312563` and `runs/2026-08-02_2130/papers/PMC12312563` — bare id, no slug.
* `runs/INVALID-scopeless-2026-08-01_1724/papers/PMC12312563` — explicitly marked invalid.

The scope was recovered from that successful run's `00_PAPER.txt`:

```
organism   : Bacillus subtilis
topic      : menaquinone biosynthesis
query      : "menaquinone biosynthesis" AND "Bacillus subtilis"
```

**⚠ Flag for the product owner, not a blocker:** the paper's title is *"Structures of **Listeria
monocytogenes** MenD…"* while the recorded organism is *Bacillus subtilis*. That is how the paper was found
and how the committed, golden leg was produced. **Reproducing T-101 with the same scope is the right call for
comparability**, but the mismatch is real and someone should eventually decide whether it is a gold-data
issue.

### Leg count — a genuine ambiguity in the milestone row

`TEST_MATRIX.md:478` reads *"+ PMC12444477 ×2, PMC12782028, PMC12312563"*. The `×2` on the first paper means
both modes; the other two carry no multiplier.

* **Literal reading — 4 legs:** PMC12444477 strict+research, PMC12782028 **strict**, PMC12312563 **strict**.
  The single modes are inferrable: `topics_flip_strict.txt` is about *"PMC12782028/**strict** FAIL → PASS"*,
  and PMC12312563's `GOLDEN` entry is its **strict** leg.
* **Practical problem:** `scripts/batch_run.py --modes` is **per-run, not per-paper**. Three papers with
  `--modes strict,research` gives **6 legs**.

**Recommendation: run the 6-leg single invocation.** A superset satisfies the acceptance criteria, it is one
command instead of two, and the two extra research legs cost ~46 min at free-tier rates. Flag it in the run
record as a deliberate superset.

### Command

```bash
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \
  --label t101-m2-legs --timeout <wall+margin> --json <allocated g11 path> -- \
  .venv/Scripts/python.exe scripts/batch_run.py \
    --topics topics_t101.txt --out runs_verify \
    --modes strict,research --timeout 1800 --deadline 3 --fresh
```

`topics_t101.txt` **does not exist yet** and must be created and committed with the three scoped lines above.
Shape taken from T-100's committed invocation, recovered verbatim from
`evidence/g11/T-100/03-waveb-fresh-legs.json`.

### Wall clock

`TEST_MATRIX.md` says ~2 h. Measured single-leg times elsewhere in the sprint are **1308 s and 1511 s**
(~22–25 min). **6 legs ≈ 2.3 h serial.** The 4-leg literal reading would be ~1.5 h.

---

## 3. T-103 — M4

### Acceptance (`TEST_MATRIX.md:480`)

> *every RAG round re-entered normalization, mapping, gates, persistence, classification*

**Structural, not a release classification.** T-103 is **not** blocked by F-062.

### ⭐ The round multiplier is RESOLVED — it is 1×

`scratchpad/T-103-prep.md` (previous session) said *"Do not quote a T-103 cost until C-055 reports its round
cap."* C-055 is merged, so it can now be read:

* `streamlit_app.py:1273` — `rounds_allowed = int(max_rounds) if max_rounds is not None else rag_loop_max_rounds()`
* `streamlit_app.py:912-920` `rag_loop_max_rounds()` — reads env `RAG_LOOP_MAX_ROUNDS` (`:894`),
  `int(raw) if raw else 1`, `ValueError → 1`, `return max(1, value)`
* **`RAG_LOOP_MAX_ROUNDS` is not set in `.env`.**

**⇒ production `max_rounds = 1`. The multiplier is 1×, so T-103 ≈ 1.5 h — the same shape as T-100.**

This also independently confirms F-070's severity claim: `max_rounds=1` is the **default production path**,
not an edge case.

### Papers

Same two scoped topics as T-100, so the paper scoping matches every earlier committed leg for these ids:

```
PMC12452463 | enterobactin biosynthesis | Escherichia coli
PMC12096016 | enterobactin biosynthesis | Escherichia coli
```

2 papers × 2 modes = **4 legs, ~1.5 h**. `topics_t103.txt` does not exist yet and must be created.

### Two settings that are NOT defaults

1. **`T2PW_SPECIES_LLM=0` is MANDATORY** — PACK 9 RULING 3: *"T-103 SHALL run with `T2PW_SPECIES_LLM=0`.
   **T-104 must not inherit this.**"* Mapping must still run.
2. **`T2PW_OFFLINE_CURATOR=1` is RECOMMENDED.** It disables only the pathway-curator LLM call, not RAG
   retrieval or synthesis, so the loop and its five stages still run and the acceptance property is still
   observable. Cheaper, and removes one source of nondeterminism.
   **Against it:** the curator's accepted patches flow through mapping into `final_mapped_db`, so an
   offline-curator T-103 produces artifacts not directly comparable to T-100's. **T-103's acceptance does not
   require that comparability.** Flag the choice in the run record either way.

### Command

```bash
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \
  --label t103-rag-legs --timeout <wall+margin> --json <allocated g11 path> -- \
  env T2PW_SPECIES_LLM=0 T2PW_OFFLINE_CURATOR=1 \
  .venv/Scripts/python.exe scripts/batch_run.py \
    --topics topics_t103.txt --out runs_verify \
    --modes strict,research --timeout 1800 --deadline 3 --fresh
```

### One standing risk

PACK 9 RULING 4: **C-055 tightened a previously-unconditional gate (`validate_graph_delta`). A T-103 leg
merging fewer additions than a pre-C-055 leg is NOT a regression.** Do not let anyone score it as one.

---

## 4. Prerequisite checklist

| | T-101 | T-103 |
|---|---|---|
| Code prerequisite merged | **C-064 (recommended, see § 2)** | **C-055 ✔ merged** |
| Topics file exists | ✗ `topics_t101.txt` to create | ✗ `topics_t103.txt` to create |
| All scopes known | ✔ (one recovered this session) | ✔ |
| Heavy mutex free | ✔ at time of writing | ✔ |
| Marginal cost | ≈ $0, all-free-tier | ≈ $0, all-free-tier |
| Credits/rate limits usable | **UNVERIFIED** — one free GET settles it | **UNVERIFIED** — same |
| Wall clock | ~2.3 h (6 legs) or ~1.5 h (4 legs) | ~1.5 h |
| Blocked by F-062 | no | **no** |

---

## 5. What is actually being asked

**One authorization covering both**, to run **live legs against `openrouter/free`** — approximately 3.8 h of
wall clock combined, at approximately zero marginal cost, on the product owner's existing OpenRouter key.

**Not being asked, and explicitly out of scope here:** T-104 and T-105. Each is a separate ~7 h, 20-leg run;
T-104 is blocked behind F-062; and **they must never be collapsed into a single run** — T-105 is the second
release candidate and requires a triage pass between the two.

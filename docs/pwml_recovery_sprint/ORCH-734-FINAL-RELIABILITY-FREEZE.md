# ORCH-734 — FINAL EXPECTED-WORKING RELIABILITY SMOKE. Frozen configuration, recorded BEFORE execution.

**2026-09-08.** The last reliability-validation task before development moves to RAG v2 /
literature-expanded pathway reconstruction. Authorized by the product owner. **No production
code is modified.** `main` untouched. Production remains frozen at `045447c8`.

> ## A NEW dataset with its own identity.
> `runs_smoke/<this run>` is **not** `runs_smoke/2026-09-07_2323` (ORCH-732), **not**
> `runs_verify/2026-09-06_1425` (the ORCH-724 unseen pilot), **not** `runs_validation/`
> (ORCH-730), and **not** `runs_validation/c120/` (the C-120 post-merge validation).
> **Its numbers may never be merged into any of those denominators or quoted as though they
> were part of one.** All four remain read-only and none is superseded or rescored here.

---

# 1. The question

> **When the user gives the production system a reasonable biological pathway paper, does the
> system actually produce a usable PWML?**

## What counts as success

**A PWML on disk.** `release_ready` is **NOT** required. `pathway.review_required.pwml` counts
as successful PWML generation. Conservative refusal is **not** scored as success.

**Primary metric:** PWML files generated / reasonable pathway papers attempted.
**Also reported, and the two must not be conflated:**

* **meaningful-core yield** — how many papers produced a defensible canonical reaction core;
* **serialization yield** — how many of those cores became actual PWML files.

That split is the whole point: it separates *extraction* failure from *downstream PWML
generation* failure, which a single rate cannot do.

## What this is not

Not an adversarial benchmark, not a negative-control exercise, not a re-score of anything, and
**not a population-wide statistical rate.** Twelve papers is a *final expected-working
reliability smoke cohort* and will be called exactly that.

---

# 2. The one configuration change, and why it is not a card

**`OPENROUTER_EXTRACTION_FALLBACK_MODEL` is now set.** That is the entire change. It lives in
`.env`, which is gitignored, and `git diff 045447c8 HEAD -- src/` is **empty**.

`ORCH-733` § 3 established the finding this acts on: rung 3 of
`_run_json_stage` — § 9's *"a materially different strategy (narrower section-based extraction
or an alternate model)"* — was **reached 6 times in the whole archived corpus and issued 0
times**, every refusal carrying `strategy_not_materially_different`, because the variable was
unset, `alternate_model_env_var()` returned `""`, and the rung fell back to the model that had
already returned nothing twice. **The guard was behaving exactly as designed; there was nothing
to escalate to.** ORCH-733 § 6 ruled this a configuration change requiring no `src/` diff, no
unfreeze, no card and no G9 proof.

## The model chosen, and why

| | value |
|---|---|
| **primary Stage-1 extraction model** | `deepseek/deepseek-v4-flash` |
| **Stage-1 alternate (rung 3) model** | **`google/gemini-3.8-flash`** |

* **Materially different by resolution, not just by name.** Different vendor and family, so
  `alternate_model_env_var()`'s C-round-1 resolution comparison — which exists precisely so two
  variable names pointing at one model cannot masquerade as an alternate — returns the variable
  name rather than `""`.
* **Context.** 1,048,576 tokens. This cohort's full texts run **18k–124k characters**; the
  primary's context is the same order, so the alternate is never the narrower vessel.
* **Structured output.** Advertises `response_format`, which the extraction contract's
  `response_json=True` path requires.
* **Cost is bounded by the rung, not by the model.** Rung 3 is reached only after the pipeline
  has already drawn nothing usable twice — 6 legs in 336 historically — and is additionally
  capped by `MAX_TOTAL_MODEL_ATTEMPTS = 3` and by `budget_conditional=True`.

## What is explicitly NOT claimed

**No reliability improvement is claimed from this activation.** It gives a qualifying leg *one
more draw from a different model where today it gets none*. Whether that draw succeeds is
unmeasured. ORCH-733's own words: *"It cannot be claimed this recovers 6 legs."* Neither the
cohort below nor the liveness proof above is evidence that it does.

**Class A still cannot reach rung 3.** A wholly empty completion raises `JSONDecodeError` with
nothing to salvage, so `saw_empty_payload` stays `False` (ORCH-733 § 3). Configuration does not
change that and this task does not attempt to.

## Proof the route is live — `evidence/g11/ORCH-734/02-rung3-liveness-proof.json`

`evidence/orch734_rung3_probe.py`, six phases, all pass, run under the bounded wrapper,
`FINAL SURVIVING COUNT : 0`.

| phase | what it establishes |
|---|---|
| **A** BASE arm, variable unset | rung 3 **REFUSED** `strategy_not_materially_different`, **0** alternate-model calls — the archived corpus's exact behaviour, reproduced |
| **B** activated | primary invoked **twice** on `OPENROUTER_EXTRACTION_MODEL` → `deepseek/deepseek-v4-flash`; rung 3 **admitted and issued** on `OPENROUTER_EXTRACTION_FALLBACK_MODEL` → `google/gemini-3.8-flash`; payload recovered |
| **C** safeguard | fallback pointing at the **same** model is still refused |
| **D** safeguard | a **degenerate** rung-3 reply is not laundered into a success — the stage still raises |
| **E** safeguard | the § 9 ceiling of **three** total model attempts still binds |
| **F** **live** | one real call over the configured route: `model_answered = google/gemini-3.8-flash`, `finish_reason = stop`, valid JSON returned |

Phases A–E drive the **real** `_run_json_stage` over a stubbed provider boundary, so the ladder,
the rung admission, the degeneracy test and the attempt cap are the shipped ones. **Phase A is
the point:** a proof that only showed the tip passing could not distinguish *"the configuration
did something"* from *"this always worked."*

> **One defect found and corrected inside the probe itself, recorded because it would recur.**
> The first run reported phase A as a FAILURE. `t2pw.llm.client` calls `load_dotenv` at **import
> time**, so importing it lazily inside a phase re-populated the very variable phase A had just
> unset, and the base arm issued a fallback call. The fix is an eager import before any phase
> runs. **An environment-variable A/B in this repository must load the environment first and
> mutate it second**, or it measures `.env` rather than the code.

---

# 3. Token limits — CHOSEN AND STATED, not inherited

| | value | provenance |
|---|---|---|
| **Stage 1 (extraction) `max_tokens`** | **16000** | `OPENROUTER_EXTRACTION_MAX_TOKENS` |
| **Stage 2 (inference) `max_tokens`** | **16000** | `OPENROUTER_INFERENCE_MAX_TOKENS` |
| preprocessor `max_tokens` | 12000 | `OPENROUTER_PREPROCESSOR_MAX_TOKENS` |
| global `LLM_MAX_TOKENS` | 16000 | `.env` |

Stated explicitly because **`F-186`** exists: three different numbers are reachable for these two
settings — the uncommitted helper's own default of 64000, the committed literals of 24000/20000,
and the `.env` values of 16000/16000 — so silence here would be a defect.

**16000 / 16000 holds the configuration identical to the ORCH-724 pilot, the ORCH-730 validation
and ORCH-732.** Raising it would confound a yield question with a budget question, and
`ORCH-731` A4 already retired the "raise the Stage-1 budget" reflex on evidence: genuine Stage-1
truncation is **2 events in 2,675 attempts**, and Stage 2 reached 55,660 characters on the same
nominal 16000. `D-097`'s prohibition on blaming token budget stands.

**Consequence, stated plainly:** these legs carry the same `F-186` handicap. Extraction is, if
anything, **understated**, and this must not be reported as the system's extraction ceiling.

---

# 4. Frozen state

## Code

| item | value |
|---|---|
| branch | `sprint/pwml-recovery` |
| repo HEAD at freeze | `11f033afaf9d5ca26491d9343b3589f0d3d6ec35` |
| local / `origin` / `ls-remote` | **all three identical** — verified before staging |
| **frozen production** | **`045447c86f1a0288ce87c444de4854b2eb2ef943`** (`C-120` merge) |
| `git diff 045447c8 HEAD -- src/` | **empty** |
| `main` (local) | `75316922a7682892769a6bcb4e4731e628bdb4f8` — **not touched, and not to be** |
| `main` (`origin`, `ls-remote`) | `03f1af56702aa21ec518b46fbf42ca43dbc47e9a` — pre-existing divergence from local, recorded, **not reconciled by this task** |
| `streamlit_app.py` CRLF sha256 | `47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632` |
| `streamlit_app.py` content sha256 (LF) | `251122389a2d29e80c157ee139837d06c6f82b7ad6e215d144ecc41b436933bf` |
| `streamlit_app.py` size | 361,433 bytes |
| `streamlit_app.py` git state | **`M` — modified, never staged, never committed** (`D-097`) |
| interpreter | `.venv/Scripts/python.exe`, CPython 3.13.6 |

**Byte-for-byte identical to the ORCH-730, ORCH-732 and C-120 freezes.** The reproducible
identity of this run is **(SHA + the `streamlit_app.py` content hash), not the SHA alone** — the
`D-097` rule. Reconstruction bundle: `evidence/repro/ORCH-724/`.

| module | sha256 (16) | status |
|---|---|---|
| `batch/driver.py` | `a1a6c582f4ea3fc1` | `C-119` seams 1–2 — unchanged |
| `pipeline/release_status.py` | `4d57b5078d1d7c53` | `C-119` seam 3 — unchanged |
| `pipeline/reaction_support.py` | `117e8e21a1541f37` | **F-179, untouched** |
| `pipeline/stage_contracts.py` | `66a6cd16a2e5ae43` | unchanged |
| `rag/admission.py` | `e45b2b1ebb27777c` | `C-118` template — unchanged |
| `pipeline/extraction_ladder.py` | `1e1b796b124ff47f` | rung 3 — **unchanged; activated by configuration only** |
| `pwml/ir.py` | `41a8d68295077cf3` | `C-120` seam — unchanged |
| `mapping/map_ids.py` | `f21066b2097c1b17` | `C-120` seam — unchanged |

The first five match ORCH-730 and ORCH-732 exactly; the last three match the C-120 merge.

## Model / provider

```
LLM_PROVIDER    = openrouter
BASE_URL        = https://openrouter.ai/api/v1
model (all 9 roles, identical)
                = deepseek/deepseek-v4-flash
  preprocessor / extraction / inference / audit / curator / gap / overwatch /
  final_completeness / global OPENROUTER_MODEL
Stage-1 rung-3 alternate
                = google/gemini-3.8-flash        <-- NEW, and the only change
LLM_TEMPERATURE = 0
LLM_MAX_RETRIES = 3
```

**Stage 0, Stage 1 and Stage 2 are not separately routed** — every role resolves to the same
primary model. The alternate is consulted **only** by rung 3 of Stage-1 extraction; grep confirms
`OPENROUTER_EXTRACTION_FALLBACK_MODEL` is read in exactly one place,
`extraction_ladder.alternate_model_env_var`. **No primary path changes.**

**Temperature 0 is not determinism.** The standing sprint trap: identical legs give materially
different Stage-1 draws at temperature 0. `ORCH-731` Part B is the sharpest case on record — the
same paper, the same four proteins extracted, a PWML on one run and `core_accepted 0` on the
next, purely from which reactions the draw happened to emit. **A single-leg difference is not a
regression.**

**The backend behind either model id is not recorded, and per-call token usage is not
persisted.** OpenRouter demonstrably routed one model to three different backends within minutes
(`ORCH-731` A1). Both remain **open observability gaps**; ORCH-733 § 4 names the exact seam
(`client.py:685`) and neither is closed here — no engineering wave was opened for this run.

## Prompt hashes

| sha256 (16) | file |
|---|---|
| `9787fe422fdade3e` | `src/t2pw/curation/prompt_payload_keys.py` |
| `913b8f1379df3a19` | `src/t2pw/tools/pathwhiz_converter/prompts.py` |

The Stage-0/1/2 prompts are built inline in `streamlit_app.py`, pinned by its two hashes above.

## Retrieval / index

```
RAG_ENABLED            = true
RAG_VECTOR_BACKEND     = memory
RAG_INDEX_DIR          = data/rag_index
RAG_EMBEDDING_MODEL    = text-embedding-nomic-embed-text-v1.5
RAG_EMBEDDING_BASE_URL = http://127.0.0.1:1234/v1   (local LM Studio)
RAG_ACQUIRE_MAX_PAPERS = 20
RAG_SELECT_MAX_PAPERS  = 8
RAG_RETRIEVE_TOP_K     = 8
RAG_EXTRACT_REACTIONS  = true
```

**533 files** under `data/rag_index` at this freeze, against **519** at the ORCH-732 freeze and
**475** at ORCH-730. The acquire cache grew through ORCH-732's and C-120's own legs. **Recorded
rather than reset:** resetting it would change retrieval behaviour relative to every prior
dataset.

**LM Studio verified reachable before execution** — `text-embedding-nomic-embed-text-v1.5` is
loaded and served at `127.0.0.1:1234`.

## Secrets

`*_API_KEY`, `LOCAL_API_KEY`, `RAG_EMBEDDING_API_KEY` and the PathBank credentials are present in
`.env` and are **excluded by allowlist** — only keys on an explicit safe list appear above, so a
secret added later cannot leak into this record by default. **No API key is printed anywhere in
this task's artifacts.**

---

# 5. Cohort — twelve papers, strict mode, one run each

Manifest: **`topics_final_reliability_smoke.txt`**, committed with this freeze.

| # | paper | requested pathway | organism | class | full text |
|---|---|---|---|---|---:|
| 1 | `PMC9200736` | acarbose biosynthesis | *Actinoplanes* sp. SE50/110 | bacterial | 71,862 |
| 2 | `PMC10269868` | ADP-heptose biosynthesis | *Helicobacter pylori* | bacterial | 123,571 |
| 3 | `PMC6112128` | tubercidin biosynthesis | *Streptomyces tubercidicus* | bacterial | 39,767 |
| 4 | `PMC11016064` | carnitine biosynthesis | *Homo sapiens* | human | 96,017 |
| 5 | `PMC10055903` | sialic acid biosynthesis | *Homo sapiens* | human | 47,830 |
| 6 | `PMC8211424` | corticosteroid biosynthesis | *Homo sapiens* | human | 29,011 |
| 7 | `PMC13184244` | nicotine biosynthesis | *Nicotiana tabacum* | plant | 77,591 |
| 8 | `PMC11961743` | furochromone biosynthesis | *Saposhnikovia divaricata* | plant | 84,181 |
| 9 | `PMC7910490` | steroidal glycoalkaloid biosynthesis | *Solanum tuberosum* | plant | 66,887 |
| 10 | `PMC7402084` | cyanogenic glucoside biosynthesis | *Phaseolus lunatus* | plant | 62,856 |
| 11 | `PMC4471609` | cycloclavine biosynthesis | *Aspergillus japonicus* | fungal | 20,897 |
| 12 | `PMC9544450` | menaquinone biosynthesis | *Escherichia coli* | **CONTROL** | 18,045 |

**Twelve is the product owner's target, inside the authorized 10–15 band.** Organism mix
4 bacterial (incl. control) / 3 human / 4 plant / 1 fungal, against the requested ~3/3/3/3.

## Why each qualifies

Every paper was selected by **abstract review against the stated rules**, not by the eligibility
screen. ORCH-732 recorded why that matters: *the screen does not select for "mechanistic pathway
paper"* — it screens for pathway *terms*, so an inhibitor paper and an omics paper score well.

* **`PMC9200736`** — *"the complete biosynthetic pathway to acarbose"*; GDP-valienol from
  valienol 7-phosphate by three named cyclitol-modifying enzymes, the glycosyltransferase AcbI,
  final assembly by the pseudoglycosyltransferase AcbS. Named enzymes, named substrates and
  products, several connected reactions.
* **`PMC10269868`** — ADP-heptose biosynthesis in a single well-covered organism; HldE
  characterized as the central enzyme, with the heptose biosynthesis genes and their regulation.
* **`PMC6112128`** — the tubercidin pathway reconstituted in a heterologous host; TubE on
  phosphoribosylpyrophosphate + 7-carboxy-7-deazaguanine, TubD as an NADPH-dependent reductase,
  TubG as a Nudix hydrolase. Exactly the substrate/product/enzyme shape the contract wants.
* **`PMC11016064`** — the four-step carnitine pathway with the second enzyme (HTMLA) identified
  and experimentally validated as SHMT1/2. Named human genes, ordinary cofactor metabolism.
* **`PMC10055903`** — the sialic acid pathway monitored step by step: UDP-GlcNAc → ManNAc →
  ManNAc-6-P → Neu5Ac-9-P → Neu5Ac, with GNE and MNK named.
* **`PMC8211424`** — human corticosteroid biosynthesis with CYP21A2 and the progestogen
  substrates characterized regio-specifically.
* **`PMC13184244`** — *"the complete biosynthetic pathway to nicotine"*: a four-enzyme cascade
  reconstructed in vitro from nicotinic acid and N-methylpyrrolinium.
* **`PMC11961743`** — *"the complete biosynthetic pathway of major furochromones"*, via named
  prenyltransferase, peucenin cyclase, methyltransferase, hydroxylase and glycosyltransferases.
* **`PMC7910490`** — the potato solanidane branch, with DPS characterized as the
  2-oxoglutarate-dependent dioxygenase catalyzing the ring rearrangement via C-16 hydroxylation.
* **`PMC7402084`** — cyanogenic glucoside biosynthesis in *P. lunatus*, CYP79D71 identified as
  the first pathway enzyme producing oximes from valine and isoleucine.
* **`PMC4471609`** — discovery and **reconstitution** of the cycloclavine pathway from the
  chanoclavine-I intermediate. The smallest document in the cohort, included deliberately: a
  short, dense, fully reconstituted pathway is a realistic product input.

## The single control — and it is a sentinel, not a trap

**`PMC9544450`** is the one deliberately re-used paper. It produced a **39,686-byte**
`pathway.review_required.pwml` in ORCH-732 and then failed on **`F-192`**
(`no_biological_states`) in the C-120 post-merge validation. F-192's measured population is
**one leg in 336**, and the standing instruction is **not to fix it unless a second independent
instance appears**. This leg asks whether it does. It is an ordinary *E. coli* menaquinone paper
that has already produced a PWML once — it is not chosen for being pathological.

## No overlap, verified mechanically

The other eleven were checked against the **33** paper ids reachable from the gold set, every
`runs*/` directory and every `topics*.txt` in the repository — development gold, the ORCH-724
unseen pilot, ORCH-730, ORCH-732, and the C-119/C-120 validations. **Overlap: none.**

Topic exclusions carried forward from ORCH-732 and extended with this sprint's own used topics
are listed in the manifest header.

---

# 6. Execution

```
bounded_run.py --timeout <wall> --label final-reliability-12legs --heavy-lock ORCH-734
               --json evidence/g11/ORCH-734/<n>-final-smoke-run.json
  -- .venv/Scripts/python.exe -u scripts/batch_run.py
       --topics topics_final_reliability_smoke.txt
       --modes strict --out runs_smoke --timeout 3600 --deadline <hours>
```

**This is the ORCH-732 batch path, unchanged.** Same wrapper, same runner, same flags, same
output tree; only the manifest, the leg count and the deadline differ. The manual PWML CLI is
**not** substituted — `F-183` established it is not equivalent to the protected production export
path.

Bounded wrapper, one heavy job at a time, heavy lock held as `ORCH-734`, no detached processes,
no `nohup`, no untracked background jobs. Cleanup targets only PIDs this job created;
`taskkill /IM python.exe` and `pkill python` remain **forbidden**. Pre-existing processes are
**reported and never killed**.

**Strict mode only.** Research mode is diagnostic by design (`acceptance.py:125`) and cannot
produce a PWML, so it would burn budget for no deliverable.

## NO FAVOURABLE-DRAW RETRIES

Each leg runs **exactly once**. A leg is re-run only for an objectively invalid *infrastructure*
execution — the ORCH-724 rule, unchanged. **A leg is not re-run because its biological output is
disappointing. A failure is a result.** Existing internal retry and fallback logic — including
the newly live rung 3 — operates normally; that is *inside* the leg and is not a re-run.

---

# 7. What will be reported per leg

PMCID and title · organism · pathway · Stage-1 result · final canonical reaction count ·
surviving core reaction count · **F-179 verdict** · identity/quarantine result · final live gate ·
**`C-119` superseded-report disposition** · release disposition · **PWML yes/no** · exact path and
byte size · one-line failure reason.

Then the aggregate: papers attempted, meaningful cores, PWML files, PathWhiz import passes and
failures, and the **no-PWML causes aggregated by mechanism** across the ten-way classification
(Stage-1 provider/delivery · Stage-1 extraction/content · scope/guard refusal · identity
resolution · quarantine/core coverage · live semantic gate · F-179 · required-field/export gate ·
F-192/autostate · other).

The question that decides the outcome is **not** the rate. It is:

> **Is one deterministic mechanism repeatedly killing otherwise-good pathways?**

---

# 8. The stopping rule, written down before the numbers exist

**Reliability is COMPLETE** if a clear majority of these papers generate PWML, the files are
valid XML, representative files import into PathWhiz, and the failures are distributed across
isolated or stochastic mechanisms with no single deterministic mechanism repeatedly destroying
otherwise-valid cores. **100 % is not required. One weird paper does not reopen production.**

**One final narrow fix may be considered ONLY if** a mechanism repeats across multiple
independent papers, directly suppresses PWML generation, has a clear deterministic root cause,
and can be fixed narrowly without weakening F-179 or any other biological safeguard.

**Otherwise, stop.** Biological imperfections — a missing cofactor, one omitted reaction,
simplified stoichiometry, a duplicate enzyme, a minor disconnected auxiliary node — belong to
manual biological evaluation and are **not** reliability failures. And a benchmark failure does
not by itself justify a code change: it is classified first as `product_contract_violation`,
`gold_data_defect` or `policy_disagreement`, and only the first justifies code.

**Recording this before execution is deliberate.** A stopping rule written after the numbers are
known is not a stopping rule.

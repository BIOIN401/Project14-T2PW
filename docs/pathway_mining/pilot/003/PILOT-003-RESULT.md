# Pathway-mining pilot 003 — result

**Verdict (`pwml-bio-auditor`): NOT_PLAUSIBLE.** The A. thaliana target graph is empty. But the cause is no longer
the literature or paper identity: it is how the species gate reads passages, plus a run-stopping cap.
**Iteration 2's two fixes are confirmed live**, and this run has the best candidate layer of the three pilots.
No unsupported reaction entered the target graph.

## What ran

| | |
|---|---|
| pre-registration | `LEDGER.md` `ORCH-743`, commit `36230ba6`, before any model call |
| code | `experiment/pathway-mining` @ `f9556288` (iteration 2; code review `APPROVE_WITH_FINDINGS`, bio `SOUND_WITH_FINDINGS`) |
| config | identical to pilots 001 and 002. New defaults: 5-passage extraction batches, 12k `max_tokens`, 360 s request timeout |
| process | `bounded_run.py` (7,200 s bound), 1,116 s, exit 5, **0 survivors**. It used the `.env` launcher (F9 is still unfixed) |
| cost | est. **$0.0356** |

## Three pilots

| | pilot-001 | pilot-002 | pilot-003 |
|---|---|---|---|
| rounds / stop | 2 / frontier exhausted | 3 / 150 papers discovered | **1 / 35 new reactions ≥ 25** |
| discovered / included / processed | 21 / 1 / 1 | 150 / 11 / 10 | 64 / 4 / 4 |
| wrong NCBI ids | NCBI not reached | 7 of 8 | **0 of 28** |
| empty provider requests | 0 | 21 | 2 (both retried successfully) |
| **target (A. thaliana) reactions** | 0 | 1 (in vitro; wrong id) | 0 |
| correct A. thaliana-tied candidates | 0 | 1 | **8** |
| held back but admissible if read properly | 0 | 0 | 1 clear (JA → JA-Ile, GH3.10, in planta), 2 review-level (OPR3) |
| bio verdict | NOT_PLAUSIBLE | PLAUSIBLE_BUT_THIN | NOT_PLAUSIBLE |

## Why the target graph is empty

| id | finding | class | in ORCH-743 scope? |
|---|---|---|---|
| F-1 | The species gate tests its markers ("ortholog", "in vitro", "inferred from") against the whole multi-kilobyte span, not the claim's sentence. 26 of 35 labels are artefacts. It wrongly held back JA → JA-Ile (GH3.10; the same abstract reports knockout plants) and two OPR3 steps | contract violation (`species.py`) | **no**: the owner said `species.py` stays untouched |
| F-7 / B1 | `max_new_reactions_per_round` counts quarantined, non-admissible leads. One off-species review (23 of the 35) stopped the run after round 1 | **policy decision** (stop vs throttle, flagged at ORCH-742a) | no |
| F-3 / F4 | The frontier cap silently cut 137 items, including every `missing_species_evidence` query | `policy_disagreement` | no |
| B2 | Thin answers with almost no reasoning (a provider that did not reason?) are accepted as `ok`; no provider is recorded | needs `llm/` (production) | no |
| F-2 | A review is never graded `review_article_only` | contract violation (latent) | no |
| F-4, F-5, F-6 | bibliography/front-matter spans cited; participants and tissue not citation-gated; last-writer-wins species merge | contract violations (latent) | no |

**Ordering constraint from the bio-audit:** do not fix F-1 before F-2 and F-4. Otherwise review figure legends and
bibliography entries would enter the target graph graded as primary evidence.

Full detail: [`audit/PILOT-003-AUDIT.md`](audit/PILOT-003-AUDIT.md) and the bio-audit notes in `bio-audit/`.

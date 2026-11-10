# DECISIONS PROPOSED (orchestrator, 2026-11-10; the owner was away)

Not binding. The owner's files `docs/pwml_recovery_sprint/DECISIONS.md` and `docs/webserver/DECISIONS.md` were not edited. Rows marked QUESTION concern a locked decision.

| ID | What | Why | Alternatives | How to reverse |
|---|---|---|---|---|
| P-01 | Headline numbers use the 152 canonical runs (`quarantine_ok=true`); the other 40 reported separately | task instruction; canonical set from the duplication scan | all 192 | recompute from `evidence/` tables |
| P-02 | Docs-only branch, no code under `src/`, no LLM or pipeline runs | task rule | n/a | n/a |
| P-03 | Protein/enzyme failure headline = 168 of 435 (38.6%, final-output units, proteins plus complexes, `Unknown` excluded); the earlier 197/464 retained as a second definition | independent verifier got 88/308 on the proteins bucket only; reconciliation found four definitional differences; this best answers "what share of named protein entities lack an accession" | 197/464 (adds quarantine-only units); 88/308 (proteins bucket) | VERIFICATION.md lists all definitions |
| P-04 | Model/non-model organism gap reported with the caveat that it sits in complex wrappers (71.7% vs 26.0%), not in the proteins bucket (15.0% vs 19.7%) | reconciliation result | keep the first reading (73.8% vs 30.2%) | see 02 RECONCILIATION |
| P-05 | Spent 8 of 8 dispatches: 3 sonnet/high (funnel, mapping, payload), 5 sonnet/medium (accuracy+variance, import, KEGG licence, verifier, reconciler). No haiku, no opus | judgement-heavy analysis; budget cap | haiku for scans | n/a |
| P-06 | Verified three numbers independently; one of three failed first time and was reconciled by a definitional diff, not by trusting either agent | quality-check rule | accept the first | n/a |
| P-07 | Recommend NOT adopting sequence homology as an identity source; at most a flagged `inferred` suggestion outside id fields | bound 0 to about 10 units (5%) of failed units; paralogs reach 84%; D-106/D-108/D-117 | adopt with thresholds identity >= 50% and coverage >= 80% plus reciprocal best hit | owner ruling on 02 Q1, Q2 |
| P-08 | Recommend no code change for near-copy transports; do lineage instrumentation first | duplication assessment option 1 and 4; D-034/D-035/D-036; no-loss not provable | option 2 (extend D-104/D-109 compatibility to transports) | owner ruling |
| P-09 | KEGG: recommend option A (user-supplied KEGG ids only, ChEBI/UniProt default, no live KEGG calls, attribution lines) | 07b: live API use from a public server is HIGH risk; legal question unresolved | B paid service-provider licence; C skip KGML | owner ruling after asking Pathway Solutions |
| P-10 | Plan ordering: add-only and measurement items before logic changes; second-pass reaction recovery deferred | no-regression rule; Stage-2 rows 46% true positives vs 93.5% | promote early | owner ruling |
| P-11 | Treat PWML defaults (stoichiometry 1, direction ">") as byte-identical for PWML and `unstated` for other formats | does not change PWML; avoids inventing biology elsewhere | write defaults everywhere | owner ruling |
| P-12 | Clean model is pre-export, post-freeze, additive (a separate database-fill layer that never overwrites frozen values) | D-034/D-036 | change the payload before the freeze | owner ruling |
| P-13 | Evaluation protocol default: 10 papers x 5 runs per arm for LLM-affecting changes; frozen-payload replay with pinned `PYTHONHASHSEED` for deterministic changes | 05: 80% power for about 7 recall points | smaller budget, accept larger detectable gain | owner budget ruling |
| Q-01 | QUESTION: D-023 "no SBML implementation": still binding? | 06 Q1 | | |
| Q-02 | QUESTION: D-104 Amendment 1 (unsigned relations out of PWML): may other formats carry them? | 06 Q3; up to 35% of Stage-1 interactions | | |
| Q-03 | QUESTION: D-106/D-108 and D-117: flagged suggestions allowed? ORCH-739 revisit trigger met by count (44 `species_mismatch` units in 19 papers) | 02 | | |
| Q-04 | QUESTION: should compound identity stay PathBank-only (ORCH decision on `map_ids.py:7732` "novel" short-circuit)? | 02 Q3: up to 39% of unmapped compounds | | |
| Q-05 | QUESTION: D-087 bench edit to `wrong_identity` flag alias handling | 53% of 36 hand-checked flags were matcher artefacts | | |

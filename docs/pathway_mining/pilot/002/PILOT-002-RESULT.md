# Pathway-mining pilot 002 — result

**Verdict (`pwml-bio-auditor`): PLAUSIBLE_BUT_THIN.** This is an improvement on pilot-001 (NOT_PLAUSIBLE), and no
unsupported reaction entered the target graph. But the only admitted reaction rests on in-vitro evidence, and
its paper id was wrong.

## What ran

| | |
|---|---|
| pre-registration | `LEDGER.md` `ORCH-743`, commit `c52dd65c`, before any model call |
| code | `experiment/pathway-mining` @ `ecf6b9a0` (ORCH-743 iteration 1, approved after 2 repair rounds) |
| config | identical to pilot-001: A. thaliana, `jasmonate biosynthesis`, seeds `12-OPDA` + `AOS`, `deepseek/deepseek-v4-flash`, `max_tokens` 4000 |
| process | `bounded_run.py`, 2,040 s, exit 5 (`stopped_at_limit` / `max_papers_discovered_reached`), **0 survivors**. It used the `.env` launcher (F9 still unfixed) |
| cost | est. **$0.0589** |

## Counts

| | pilot-001 | pilot-002 |
|---|---|---|
| discovered / distinct / from NCBI | 21 / 21 / 0 | 150 / 144 / 83 |
| included (production / route 1 / route 2) / processed | 1 (1/0/0) / 1 | 11 (5/5/1) / 10 |
| claims / identities | 21 / 19 | 16 / 11 |
| **primary (A. thaliana)** / candidates | **0** / 19 | **1** / 10 |
| uncited-field drops | 59 | **0** (53 fields model-cited) |
| empty extraction calls | 0 | 2 calls (7 ladder rungs gave up) |

## The one admitted reaction

α-linolenic acid → 13(S)-HPOT by 13-LOX (LOX2/3/4/6); the species is model-cited.
- **Biology:** correct, and on the route (it makes AOS's substrate).
- **Evidence grade:** **recombinant** (At LOXs expressed in *E. coli*). `species_policy.md` calls that `in_vitro_or_cell_free`, which is not admissible. The in-vitro detector missed it (B-2).
- **Paper id:** recorded as `PMC411572` (Towbin 1979), a cited reference. The true article is PMC8508710 (A1 / B-1).

## Findings

| id | finding | fixed in iteration 2? |
|---|---|---|
| A1 / B-1 | the production PubMed parser (`rag/acquire.py:536`) takes a **cited reference's** PMCID/DOI. 7 NCBI ids were wrong, and 2 wrong full texts were read | **mitigated in mining** (IT2-2). The root fix in production needs its own card |
| A2 | prompt 1.1.0 plus reasoning exhausts `max_tokens` on large inputs: 78 % of cost, 58 % of time | **yes** (IT2-1, 5-passage batches) |
| A6 | windows cut by the discovery cap exhausted 21 items without searching them | **yes** (IT2-3) |
| A8 | false grape rescue | **yes** (IT2-4) |
| A13 | lineage overwritten per rung | **yes** (IT2-5) |
| B-2 | in-vitro evidence admitted as target-species; species labels are keyword artefacts (`species.py`) | **no**: out of scope (owner: `species.py` untouched) |
| A3–A5, A7, A9–A12, A14–A17 | see [`audit/PILOT-002-AUDIT.md`](audit/PILOT-002-AUDIT.md) | registered, not fixed |

## Files

- `run/`: the full run directory.
- `g11/`: the wrapper reports.
- `pilot-002.log`.
- `launch_mining.py`.
- `audit/`: the artifact audit and its scripts.
- `bio-audit/`: the biological audit's notes and fetches.
- `acquire_cache.sha256`: the cache, listed by hash and **not committed**.

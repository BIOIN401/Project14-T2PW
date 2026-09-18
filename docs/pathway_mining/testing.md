# Testing

**No test in this feature spends an API credit.** The model is a scripted
function, the literature service is a scripted source, embeddings run in the
offline lexical mode with a temp-directory cache, and the vector backend is
in-memory.

## Running the suite

```bash
# focused, fast (~4 s)
PYTHONPATH=<tree>/src python -m pytest -q --basetemp=<short-tmp> tests/test_mining_*.py
```

Under the sprint's bounded wrapper, as every test in this repository must run:

```bash
PYTHONPATH=<tree>/src python docs/pwml_recovery_sprint/evidence/bounded_run.py \
  --label mining-suite --timeout 600 --heavy-lock EXP-MINING \
  --json docs/pathway_mining/evidence/002-mining-suite.json -- \
  python -u docs/pwml_recovery_sprint/evidence/pinned_pytest.py \
    --pin-verdict docs/pathway_mining/evidence/002-mining-suite.pin.json \
    -q --basetemp=<short-tmp> tests/test_mining_*.py
```

`--basetemp` is mandatory and its **parent must already exist** — pytest does not
create intermediate directories, and a missing parent errors every test in setup,
which looks exactly like a large regression.

## The files

| File | Tests |
|---|---|
| `test_mining_feature_flag.py` | default-off, three activation paths, override in both directions, controller refusal, cheap import (asserted in a subprocess), no writes when disabled |
| `test_mining_species.py` | local normalization, genus≠binomial, all seven states, the ortholog barrier, every verdict carries a reason |
| `test_mining_graph.py` | the three connection kinds; enzyme / pathway-name / paper / vocabulary / cofactor **never** connect; multi-substrate representation; quarantine; bridge promotion; reconsideration; deterministic primary-component selection; JSON round trip |
| `test_mining_starting_inputs.py` | every accepted paper-reference form and refusal of the rest, composability, mandatory species, upload determinism, evidence-type labelling, config precedence |
| `test_mining_extraction.py` | per-field evidence attribution, uncited fields dropped, fabricated citations refused, schema validation, empty / truncated / parse failures, the shrink-and-retry ladder, explicit fallback, per-call pricing |
| `test_mining_admission.py` | the ordered gates, dedup preserving independent evidence, conflict retention, reconsideration chains, provenance-chain completeness |
| `test_mining_budget.py` | the closed stop-reason set, every limit, convergence, cancellation, cumulative-on-resume budgets, fresh wall clock |
| `test_mining_artifacts.py` | every required file, manifest completeness, no secrets anywhere, schema versioning, per-round self-containment, cancellation, resume, regeneration |
| `test_mining_cli.py` | gating, exit codes, usage errors, `status` / `report` / `list`, config-file overrides, one-shared-controller assertion |
| `test_mining_ui.py` | the entry hook's no-op behaviour, the main-app hook's shape, both view payloads, the round slider, DOT exports, the offline renderer, large-graph trimming |
| `test_mining_integration.py` | the eight required behaviours, end to end |

## The integration fixture

`tests/helpers_mining.py` holds a **synthetic**, jasmonate-flavoured corpus of
nine papers, built to exercise the policy rather than to be biochemically
authoritative. It deliberately contains:

| | |
|---|---|
| P1 + P2 | two paper-supported reactions that chemically connect (`α-linolenic acid → 13-HPOT → 12-OPDA`) |
| P3 | a relevant but **disconnected** reaction (`JA-Ile → 12OH-JA-Ile`) |
| P4 | a later paper (round 3) supplying a legitimate biochemical **bridge** |
| P5 | a **wrong-species** reaction (*Zea mays*) |
| P6 | an **ambiguous-species** claim: the target is named in the paper, nothing ties it to the reaction |
| P7 | a paper **conflicting** with P2 about the enzyme |
| P8 | **independent support** for P1's reaction from a second laboratory |
| P9 | a paper the model **cannot finish reading** (truncated completion) |

`ScriptedPaperSource` serves papers on a per-round schedule — the test advances
its round from the controller's own `round_start` progress event, which is how a
later paper can supply the bridge. `ScriptedModel` cites only evidence ids that
are actually present in the prompt, so the citation check in
`extraction.build_claims` is exercised rather than bypassed.

## What the integration test asserts

1. connected reactions enter the primary graph, via the shared metabolite;
2. the disconnected reaction starts in a candidate subgraph;
3. wrong-species evidence is retained but never enters the target-species graph
   — and the ambiguous-species claim is held back too;
4. the conflict is visible, both positions kept, the disputed attribute cleared
   on the node;
5. the round-3 bridge promotes the component, and the original discovery round
   survives;
6. every admitted reaction has an intact `reaction → claim → span → paper → id`
   chain, and independent papers are counted rather than collapsed;
7. every round directory is complete and the growth-by-round record matches;
8. no PWML, SBML or XML exists anywhere, both artifacts say `produces_pwml:
   false`, and no mining module references the exporter.

Plus: external validation is `not_evaluated` everywhere, the run stopped for a
named reason, the frontier asked new questions each round, and every evidence row
is quotable with a resolvable source identifier.

## The feature-disabled regression

The production regression gate, run on this branch with the feature off:

```
SMOKE (22 files)   508 passed in 44.45s
```

exactly the expected count. Report:
`docs/pathway_mining/evidence/001-smoke-feature-disabled.json`
(exit 0, 0 surviving processes, cleanup success, measured tree verified,
`violations: []`).

`T-107`, `T-108` and `T-109` were **not** rerun and are untouched.

## Live testing

None was performed. A tiny bounded live smoke test would be acceptable if
credentials, services, locks and repository policy allowed, but the fixture-based
integration is sufficient for verification and spends nothing. Evaluating output
*quality* against real literature is separate work needing explicit cost
authorization — see [`known_limitations.md`](known_limitations.md).

# `C-121` post-merge live validation — the seam fired in production, and it is provable from one leg

**2026-09-09, `ORCH-738`, `D-099` § 15.** Three strict legs, **one draw each, no favourable
reruns**, on the merged tree at `24dd4342`. Run: `runs_validation/c121/2026-09-09_0028`.
Cohort: `topics_c121_live_validation.txt`, three lines copied verbatim from `ORCH-734`'s topics
file and checked by `diff`, because a reworded scope string is a different experiment.

Wall clock 6,882 s. `FINAL SURVIVING COUNT : 0` · `cleanup : success` · heavy lock `ORCH-738`
acquired and released. Wrapper exit 1 is the batch runner reporting 2 of 3 legs failed — **not an
infrastructure failure**; `surviving` is `null`.

| job | report |
|---|---|
| the run | `evidence/g11/ORCH-738/01-c121-live-validation.json` |
| counterfactual attempt, base — **INVALID, see § 4** | `evidence/g11/ORCH-738/02-live-counterfactual-base.json` |
| counterfactual attempt, tip — **INVALID, see § 4** | `evidence/g11/ORCH-738/03-live-counterfactual-tip.json` |

---

# 1. The three legs

| | `PMC11961743` | `PMC4471609` | `PMC9544450` (control) |
|---|---|---|---|
| outcome | **TIMEOUT** at 3,600 s | **FAIL** at 1,771 s | **PASS** at 1,507 s |
| reached the seam | **no** | **yes** | **yes** |
| Stage 1 | reached, succeeded | reached, succeeded | reached, succeeded |
| reactions in final payload | none written | **6** | **5** |
| `__auto_state__` swept | — | **yes**, `state_unreferenced_after_quarantine` | **yes**, same reason |
| states in final payload | — | **`['__auto_state__']`** | `['cytoplasmic state']` |
| required-field gate | never reached | never reached | **PASS, zero errors** |
| PWML | none | none | **48,826 B, IMPORT READY** |
| blocker | provider budget exhaustion | 8 proteins with no UniProt/DrugBank id | none |

# 2. `PMC4471609` is the result. The seam fired, live, on a real draw

Its own artifacts contain a contradiction that only `C-121` explains:

* `removed_entity_report.json` — `__auto_state__` removed, reason
  `state_unreferenced_after_quarantine`. **The sweep fired, exactly as before.**
* `final_mapped.json` — `biological_states: ['__auto_state__']`. **The state is there anyway.**

**On the base code nothing can put it back.** The only assignment to `payload["biological_states"]`
anywhere in `strict_quarantine.py` is `_prune_biological_states:1978`, and it only ever *shrinks*
the list. After the closure loop the only thing that touches biological states is the one
`restore_autostates_if_required` call `C-121` added. So a payload whose auto-state the sweep removed
cannot regain it inside `quarantine_and_close` at base. It regained it here.

Same paper, yesterday and today:

| | `ORCH-734` | this run |
|---|---|---|
| sweep removed `__auto_state__` | yes | yes |
| states in final payload | **none** | **`__auto_state__`** |
| required-field gate | **refused, `no_biological_states`** | never reached |
| reactions | 4 | 6 |

**`F-192` is fixed in production, not only in replay.**

### The leg still failed, and on something this card is forbidden to touch

It died **earlier** than the required-field gate, at the stage-3 contract gate: `DmaW`, `EasF`,
`EasE`, `EasC`, `EasD`, `EasA`, `EasG` and `EasH` carry no UniProt or DrugBank identifier. That is
identity resolution, which `D-099` § 2 puts out of scope. **A fix that had made this leg pass would
have been the wrong fix.** `F-192` is repaired and a different pre-existing blocker now stands in
front of this paper.

# 3. The control does the job a control exists for

`PMC9544450` swept `__auto_state__` **and the guard stayed quiet**, because a real state
(`cytoplasmic state`) survived and no row was missing one. Gate passed with zero errors; 48,826 B,
`IMPORT READY`.

One leg therefore demonstrates all three of the safety properties live:

1. the sweep's removal policy is **unchanged** — it still removes the unused auto-state;
2. `__auto_state__` is **not immortal** — it stayed removed here;
3. a healthy leg is **unaffected** — clean gate, real PWML.

**It cannot be scored on byte equality** with `ORCH-734`'s 49,436 B. A new draw is a new extraction:
5 reactions here against 6 there. The claim is that it still produces a PWML, and it does.

# 4. A counterfactual I attempted, and am DISCARDING rather than reporting

I tried to prove the `PMC4471609` result by driving its `merged_payload.json` through
`normalize_process_payload` → `quarantine_and_close` in the base tree and the tip tree.
**The replay does not reconstruct the live leg**: it yields **0** surviving reactions and
`quarantine ok = False`, where the live leg had **6** and `True`. `merged_payload.json` is the
post-audit payload, and the live pipeline does more between there and quarantine — RAG admission,
gap resolution, identity mapping — which the replay does not perform.

**So it is not evidence and is not offered as any.** Both reports are committed and labelled
INVALID so the numbers in them are never mistaken for a measurement. The § 2 proof needs no replay:
it rests on the leg's own two artifacts plus the fact that base has no code path that re-adds a
state.

This is the same discipline that killed `REV-121`'s blocking finding — a measurement is only worth
what its conditions are worth, and a replay that does not reproduce the thing it replays is worth
nothing.

# 5. `PMC11961743` proves nothing, and that is reported rather than retried

It never reached the seam: **zero payload files**, `payload_before_cleanup.existed: false`. It spent
the full hour in model calls — **113 attempts, 71 empty, 70 of them `finish_reason=length`**, the
`F-193` degeneracy. It was not a Stage-1 failure; it got through extraction and inference into the
audit loop and gap resolver, then exhausted the budget on retries.

**`D-099` § 15 says not to re-run merely to exercise the fix, so it was not re-run.**

# 6. `F-193`'s rate is far worse than the record says, and it is now the top yield risk

`F-193` is registered as *"two observations across two runs is not a rate"*. There is now a rate,
measured over whole legs rather than Stage 1 alone:

| leg | model attempts | empty | last event |
|---|---:|---:|---:|
| `ORCH-734` `PMC11961743` | 102 | **58 %** | 2,298 s |
| `ORCH-734` `PMC4471609` | 44 | **59 %** | 2,348 s |
| `ORCH-734` `PMC9544450` | 29 | 21 % | 1,105 s |
| **this run** `PMC11961743` | **113** | **63 %** | **timed out** |

**The provider did not degrade today.** `PMC11961743` was already burning 58 % of its attempts on
empties yesterday and finished at 2,298 s of a 3,600 s budget — always one bad draw from the wall.
Nearly two thirds of all model calls on these papers return nothing.

**This is a bigger threat to PWML yield than `F-192` was**, and it is not a pipeline defect: the
extraction ladder behaves correctly, refusing to re-issue an identical prompt to the same model.
**REGISTERED, NOT CHARTERED.** It is a provider/model-selection question, not a code fix, and
nothing here authorizes one.

# 7. `F-196` reproduced exactly

The runner printed `1 NO DELIVERABLE` and `!! PASSED BUT PRODUCED NO DELIVERABLE !!` for the control
— **which wrote a 48,826-byte PWML**. A reader trusting the tally would score this run at zero
deliverables with one on disk. `F-196` stands, unfixed, and every count in this document comes from
files on disk rather than from the runner's summary.

# 8. What this does and does not establish

**Establishes:** the `C-121` seam executes in production, restores the placeholder after a real
sweep, is inert on a healthy leg, and preserves the sweep's removal policy. All on live draws.

**Does not establish:** that either `F-192` paper now yields a PWML. Neither did. One timed out
before the seam; the other cleared `F-192` and hit an unrelated identity blocker. **No PWML is
claimed for either**, and the § 6 provider failure rate means a repeat run would be a different
experiment, not a confirmation.

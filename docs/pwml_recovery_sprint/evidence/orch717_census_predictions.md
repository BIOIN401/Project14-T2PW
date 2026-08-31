# ORCH-717 — census predictions, recorded BEFORE the probe was run

Written by the Lead Orchestrator at tip `24c58c2`, before executing
`orch717_census_probe.py`. Sprint practice: predictions first, then measurement,
and the failed prediction stays beside the correction.

F-151 reports the breakage as `assert 72 == 62` on tests 10 and 13. That is the
assertion that fires FIRST. Both tests carry FURTHER derived pins after it that
have never executed against the grown corpus, because the census assert aborts
the test before reaching them. **Re-pinning 62 -> 72 alone may therefore not make
these tests green**, and a card that assumes it will is a card that will come
back red.

| # | Quantity | Current pin | Prediction |
|---|---|---|---|
| P1 | `len(paths)` — tracked `quarantine_report.json` | `>= 62` (line 375) | **72**, and the `>=` floor still holds |
| P2 | test 10 `legs` | `== 62` (line 397) | **72** |
| P3 | test 13 `checked` | `== 62` (line 511) | **72** |
| P4 | test 10 `withheld` | `== 92` (line 405) | **rises above 92.** T-107 committed ten legs including negative controls that carry forbidden terms. I do not name an exact value; I predict only the direction |
| P5 | test 13 `with_matched_forbidden` | `== 23` (line 515) | **rises above 23**, same reason |
| P6 | test 10 `set(affected_papers) - set(F132_PAPERS)` | `== {"PMC13231680"}` (line 404) | **AT RISK.** If any T-107 leg belongs to a gold paper outside F132_PAPERS and carries an excluded term, this set grows and the assert fails even after the census is re-pinned |
| P7 | test 10 `cleared` | `== []` (line 407) | **stays `[]`.** A leg clearing the threshold on the accepted ratio would be a substantive change, not a census change |

**P4, P5 and P6 are the ones worth having written down.** F-151's analysis, REV-104's
correction and the handoff all describe this as "re-pin to 72". If P6 fires, the
correct fix is larger than a census re-pin, and the card must say so rather than
quietly widening an assert to make a test green.

**No prediction here licenses relaxing a pin.** Every one of these is an
artifact-census or non-vacuity guard, not a measurement of pipeline quality
(F-151's "what this does NOT license"). Whatever the probe returns, the pins stay
`==` and move deliberately, with the delta attributed leg by leg.

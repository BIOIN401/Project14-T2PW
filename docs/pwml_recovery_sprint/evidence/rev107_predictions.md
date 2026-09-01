# REV-107 predictions, written from the diff BEFORE any measurement was run

Source: `git diff 33a99e7..9890770 -- src/t2pw/curation/apply_audit_patch.py` plus a read of
`_span_licenses_actor`, `_match_fold`, `_actor_role_family` at the tip. No author log, no test
file and no evidence artifact was read before writing this.

## B1 F-146
P1. The F-146 patch stays REJECTED at tip. Role `enzyme` -> family `catalysis`; the rationale span
    carries no catalysis stem cue (the bare schema noun "enzyme" is deliberately not a cue), so
    cue.search fails. RISK: the tip appends `_ENZYME_NOUN_RE_SRC` to the TRANSPORT family. Test a
    transport-container variant too.

## B2 reduction-of / redox
P2. `reduces|reducing|reduction of` is STILL in the catalysis set (read from the diff; not
    deleted). Closure is a new `_CATALYSIS_CONTRA_RE` = inhibition | (attenuation stem + <=40
    chars + one of activit|express|level|abundance|function). So:
      - "the reduction of NDM-1 activity by PSA"    -> refused  (contra fires)
      - "NADH-dependent reduction of the substrate" -> licensed (no object noun)
      - "ferrochelatase reduces ..."                -> licensed
    PREDICTED RESIDUAL: a genuine redox span whose nearby noun is one of the five would now falsely
    refuse. Also `block` is an attenuation stem, so "blocked ... activity" refuses (intended).

## B3 eleven near-synonyms
P3. Six added as BARE inhibition stems: blockade, impair, silenc, sequestr, ablat, interfer(e|i).
    Five are NOT (reduction, loss, depletion, disruption, quenching) and are reached only through
    `_ACTIVITY_ATTENUATION_SRC`. So all eleven should close in the "of X activity" frame, but only
    six close with no activity/level/expression/abundance/function object nearby.
    PREDICTION: `_ACTIVITY_ATTENUATION_SRC` requires the object AFTER the stem within 40 chars.
    "NDM-1 activity was reduced" puts the object BEFORE the stem -> predicted NOT closed. Real
    residual; must test.

## B4 29-case battery
P4. base = 1 refusal (cofactor case). tip = 0. Fail threshold is 2.

## B5 corpus, both directions
P5. Newly refused: not necessarily zero. The widened contra (impair, silenc, ablat, sequestr,
    blockade, interfer, plus the attenuation phrase), the anchored `mediat`, the 17 new stoplist
    words and the REMOVAL of the family-wide passive-with-agent cue are all narrowings.
    Newly admitted: cofactor rows (1e), transport -ase nouns (1d), the six new inhibition stems.

## B6 stoplist
P6. PASS predicted: `_NON_ENZYME_ASE_WORDS` grew by 17 ordinary English words; no enzyme allowlist.
    `_SHORT_ENZYME_NOUNS` unchanged.

## B7 plural bypass, generally
P7. Fix is `s?` inside the negative lookahead -> general, not an enumeration.
    RESIDUAL to check: `s?` covers only the `-s` plural; check every singular-only entry.

## B8 cofactor vs cofactor_as_protein
P8. Different gates. Predict no coupling; grep to confirm the F-100 gate does not read
    `_ROLE_CUE_RES`.

## B9 benchmark reachability
P9. Predict `src/t2pw/bench/` has no reference to `_span_licenses_actor`,
    `UNEVIDENCED_ACTOR_ROLE_REASON_PREFIX` or `apply_audit_patch`. Must grep.

## B10 cancelling pair
P10. `mediat` is anchored so it no longer matches inside "intermediate". `suppress` is UNCHANGED
     and still matches inside "suppressor". Predict the 140 KB span outcome is unchanged; measure.

## B11 oversized spans
P11. No `src/` change to span length. Predict PASS (registered only).

## B12 callers
P12. Re-derive. Author claims 7 sites / 6 modules vs C-105 recording 4.

## B13 fixture names under src/
P13. The `_span_licenses_actor` docstring ALREADY contained "PSA-mediated inhibition of NDM-1" at
     BASE. I must grep the ADDED lines only and report which names are NEW in this diff.

## B14 gold
P14. diff --stat shows no `src/t2pw/bench/gold/` path. Predict PASS.

## B15 T-107
P15. No `runs_verify/` in diff --stat. Predict PASS.

## Claim adjudication predictions
C1. 7 callers: plausible; import-vs-call distinction on pipeline.py. Re-derive.
C2. M6 survived: `_ANY_ROLE_CUE_RE` is built from `_ROLE_CUE_RES.values()` and the tip ADDS the
    cofactor pattern to `_ROLE_CUE_RES`. Deleting the map entry sends the role to "other" ->
    `_ANY_ROLE_CUE_RE` -> which NOW CONTAINS the cofactor vocabulary -> still licenses. Author's
    conclusion PREDICTED CORRECT. Also a finding: the tip widens the "other" fallback for EVERY
    unmapped role.
C3. `inhibitor` is a bare stem in the inhibition set at BASE, so a span saying "is an inhibitor
    target" already carries the contra at base. Predict the author is right.
C4. `reduction of` is the cue phrase; bare `reduction` is not in the catalysis set. In the bare
    frame only `reduction` supplies a cue -> only it admits. With "is mediated by" in the window,
    `mediat` supplies the cue for all eleven and the base contra contains none of them -> all
    eleven admit. Predict the AUTHOR is right and the Lead probe wrong in both directions.
C5. 1g not Stage 1: must measure.
C6a. "transporter" contains "transport", a bare stem in the transport family, at base and tip.
     Predict CONFIRMED, same class as F-146.
C6b. `_match_fold` maps `[^a-z0-9]+` -> " ", so no "." survives into the haystack. `[^.]` == `.`.
     CONFIRMED BY READING; confirm empirically.

## My own risk predictions, not on the Lead list
N1. `_ROLE_CUE_RES["cofactor"]` contains bare `requires`, `required for`, `depends on`,
    `dependent on`, `in the presence of`. Extremely common in a patch rationale. Predict the
    cofactor family self-licenses on a bare rationale -- same class as F-146, newly introduced by
    THIS card. MUST TEST.
N2. N1 leaks into `_ANY_ROLE_CUE_RE`, the fallback for every unmapped role.
N3. `block` appears both as attenuation stem and via `blocks|blocked|blocking`: redundant.
N4. The 1b passive route measures its contra window from `match.end()`, i.e. around the AGENT, not
    around the verb. A contra before the verb is not seen. Narrower contra coverage. MUST TEST.

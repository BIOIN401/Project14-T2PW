# ORCH-755 — overnight paper runs for hand comparison

**Status:** RUNNING (launched 2026-10-20 evening). Base: `main` = `sprint/pwml-recovery` @ `8b074dda`.
Requested by the product owner: run the papers in the background and have the text results and the
PWML files ready to check the next day. The id ORCH-755 comes from the 755-759 block the ORCH-760
session left free for this one.

## 1. What runs

Seven conditions, from the ORCH-751 cohort: enterobactin, menaquinone, chlorophyll, Ara operon, sour
taste, WNT PCP, Wnt/Ca2+. Each runs in strict and in research mode, for 14 legs. All strict legs go
first. Each leg is one `scripts/batch_run.py` pair through the real app, inside its own `bounded_run.py`
with the heavy lock, a 3,600 s pair timeout and a 4,200 s wrapper timeout. Configuration is D-104: 3
draws, 64,000 extraction tokens, prompt v03, live curator. Inputs and details are in
`docs/evaluation_v2/paper_runs_orch755/README.md`.

This is the **first end-to-end exercise of the 3-draw path** and of rule 3 (the Stage-6 database-ID
merge), which had been tested only offline and on synthetic payloads (ORCH-754-RESULT § 5).

## 2. Where the results land

`runs_papers/ORCH-755/REVIEW/INDEX.txt` is rewritten after every leg. Each `<condition>/<mode>/`
folder holds:
- `reaction_evidence.txt`: the app's deterministic Reaction Evidence export
- the PWML, if one was written
- the Stage-1 ensemble side files (support marks and merge flags)
- `RESULT.txt`

The batch driver does not copy the ensemble side files into a leg, because the app writes them under
`tmp/extraction_diagnostics/<unique>/`. The review script picks them by the leg's time window, and
every pick is listed in the index.

## 3. Results

To be filled when the night ends.

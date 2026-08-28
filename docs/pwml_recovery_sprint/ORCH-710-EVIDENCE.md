# ORCH-710 — the corrected O-1 decision evidence

Committed 2026-08-27. These are the bounded-wrapper lifecycle certificates for the five
measurement probes that produced the corrected O-1 decision packet, plus the certification
check that validates them.

The artifacts live in `evidence/g11/ORCH-710/`. They are **process certificates, not
measurements**: `bounded_run.py`'s JSON report records lifetime, ownership and cleanup, and
carries **no child stdout** (see `TEST_MATRIX` § 0). The measurements themselves are recorded
in `DECISIONS.md` D-070 and in `LEDGER.md`; these files prove those measurements were taken
under a bounded, owned, zero-survivor job at a known tree.

| # | Artifact | Probe | Exit | Survivors | Cleanup |
|---|---|---|---:|---:|---|
| 01 | `01-pinned21-reconstruct.json` | reconstructs the pinned 21 and partitions it | 0 | 0 | success |
| 02 | `02-losses-and-gold.json` | losses + gold tolerance — **superseded by 03** | **1** | 0 | success |
| 03 | `03-losses-entdef-gold.json` | the same measurement, re-run and completed | 0 | 0 | success |
| 04 | `04-stripped-identity-losses.json` | stripped/withheld candidate identities | 0 | 0 | success |
| 05 | `05-gold-tolerance-contradiction.json` | PMC12444477 Boolean vs. its rationale | 0 | 0 | success |
| 06 | `06-certify-check.json` | `g11_evidence.py check --task ORCH-710` | 0 | 0 | success |

All six report `FINAL SURVIVING COUNT : 0` and `cleanup : success`, all at
`repo_head 7bd8a86ca81e985555fb8d3656e665a4ea797437`, all against wrapper build
`sha256:83d139543d4c01b3…`, `wrapper vs HEAD: clean`.

Job 06 is the certification itself: **6 artifacts, 0 non-compliant (spec v1)**.

---

## The job-02 incident, preserved rather than concealed

**Job 02 exited 1 after its measurement was already complete.** The probe computed its result
and then died printing it: the Windows console codec is cp1252 and the output carried a
character outside it. The failure is in the *printing*, not the measurement, and it is
recorded here rather than quietly dropped because a non-zero exit in the evidence tree is
exactly the kind of thing a later reader is entitled to ask about.

**Job 03 supersedes it** — the same measurement, re-run with the printing fixed, exit 0.
Where the two disagree, or where anything is read from job 02 at all, **03 is controlling**.

Job 02 is **not deleted**. D-025 is explicit that genuine evidence is never removed to make a
record look clean, and a superseded or failing report stays committed beside the one that
replaced it. Both are certified; both are compliant; only one is authoritative.

## Why no README sits inside the task folder

`g11_evidence.py:iter_reports` selects **every** non-dot file in a task directory and
`check_many` calls anything that is not a valid report an `unexpected_artifact`. A `README.md`
dropped into `evidence/g11/ORCH-710/` would therefore turn the merge gate red. That is why
this index is a sprint document and not a file beside the artifacts it describes.

---

## The probes themselves are committed too (added with ORCH-711)

The table above certifies that six jobs ran bounded and left nothing behind. It does **not** record
what they measured, because `bounded_run.py`'s report carries no child stdout. The probe sources
lived in a session scratchpad outside the repository and survived by luck.

They are now in `evidence/`, beside the reports rather than inside the task folder:

| Probe | Source | Log |
|---|---|---|
| A | `orch710_probeA_pinned21.py` | `orch710_probeA_pinned21.log` |
| B | `orch710_probeB_losses_gold.py` | `orch710_probeB_losses_gold.log` |
| C | `orch710_probeC_losses_entdef_gold.py` | `orch710_probeC_losses_entdef_gold.log` |
| D | `orch710_probeD_stripped_identity.py` | `orch710_probeD_stripped_identity.log` |
| E | `orch710_probeE_gold_tolerance.py` | `orch710_probeE_gold_tolerance.log` |

`orch710_pinned21.json` is the pointer file the 16/5 partition was computed against.

**Probe D pins `PINNED = "runs/2026-08-02_2130"`.** That is the run the 24/82 figures name — **not**
`runs_verify/2026-08-24_1428`, the T-106 10-paper run, against which the same criterion yields 5.
Nothing in the O-1 packet said which tree it meant, and a session acting on the wrong one nearly
reported a certified measurement as unreproducible. See `LEDGER.md` section
"F-141 CLASSIFIED".

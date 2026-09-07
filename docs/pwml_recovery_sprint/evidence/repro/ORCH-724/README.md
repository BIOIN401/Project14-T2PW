# ORCH-724 — evaluation reproducibility bundle

**The unseen pilot is NOT reproducible from the production commit alone.** That claim was
recorded and is false. The correct statement is:

> **pilot state = production SHA `c4a97f6006cdfe2d3a30790072826d4f1eaa74de`
> + the patch `streamlit_app.pilot.patch` (sha256 `b5afa243acd3ce6a…`) applied to
> `src/t2pw/app/streamlit_app.py`.**

## Files

| file | what it is |
|---|---|
| `streamlit_app.pilot.patch` | the exact unified diff, committed → pilot-running. **1,942 bytes, sha256 `b5afa243acd3ce6a3d4b149f9d0f29eb00e0eeab4b659524a40c0d29f214321d`.** Carries `-text` in `.gitattributes` so checkout cannot CRLF-mangle it and invalidate that hash |
| `BUNDLE.json` | the full manifest — SHAs, model/provider config, token budgets, prompt hashes, retrieval config, evaluator versions, cohort manifest, all 20 leg paths, timestamps |
| `build_bundle.py` | regenerates `BUNDLE.json`. Read-only |
| `reconstruct_proof.py` | the reconstruction proof. Read-only w.r.t. the primary checkout |
| `reconstruct_proof.json` | its result — **`reconstruction_exact: true`** |

## The three hashes, and why there are three

| | sha256 | what it is |
|---|---|---|
| committed at the pilot SHA | `70299631b41762f7…` | the object-store blob (always LF) |
| pilot bytes, **LF** | `251122389a2d29e8…` | platform-independent **content identity** |
| pilot bytes, **CRLF** | `47e4fafa789d359d…` | **what this Windows machine actually executed** |

`core.autocrlf=true` is set globally and this path carries no `text` attribute, so the
object store holds LF and the working tree holds CRLF — `.gitattributes` documents exactly
this hazard. **7,269 of the 8,279-byte difference is line endings; only 1,010 bytes are
real content.** A record pinning only `47e4fafa…` would fail spuriously on Linux and would
overstate the size of the change by 8×.

## Reproduce it

```bash
git worktree add --detach /tmp/pilot c4a97f6006cdfe2d3a30790072826d4f1eaa74de
cd /tmp/pilot
git apply docs/pwml_recovery_sprint/evidence/repro/ORCH-724/streamlit_app.pilot.patch
# LF platforms:      sha256 == 251122389a2d29e8...
# Windows + autocrlf: sha256 == 47e4fafa789d359d...
```

Verified mechanically:

```
python docs/pwml_recovery_sprint/evidence/repro/ORCH-724/reconstruct_proof.py .
  → LF   MATCH: True
  → CRLF MATCH: True
  → RECONSTRUCTION EXACT: True
```

## ⚠ The patch is NOT UI-only. It changes pipeline execution configuration.

| | committed | pilot |
|---|---|---|
| Stage 1 (extraction) `max_tokens` | literal **24000** | `_bounded_env_int("OPENROUTER_EXTRACTION_MAX_TOKENS", 64000)` → **16000** |
| Stage 2 (inference) `max_tokens` | literal **20000** | `_bounded_env_int("OPENROUTER_INFERENCE_MAX_TOKENS", 64000)` → **16000** |

Consumed at `streamlit_app.py:5467` (`max_tokens=int(extract_tokens)`) and `:5588`
(`max_tokens=int(infer_tokens)`) — these are the Stage-1 and Stage-2 LLM calls.

**It reached the pilot** because `batch/driver.py` drives the app through `AppTest` and sets
only the export-mode radio, the input-mode radio, the source text area and the buttons. It
never sets the token number-inputs, so the **widget defaults are what executed** — verified
by grep over `src/` and `tests/`: nothing anywhere overrides them.

**Direction of the effect:** the pilot ran with *smaller* generation budgets than the
committed code would give (16000 vs 24000, 16000 vs 20000). If it mattered, it would tend to
**understate** extraction, not flatter it.

**Circumstantial, not proof:** two pilot legs failed with `failed to produce valid JSON`, a
known symptom of a generation budget cut mid-object. Establishing causation would require
re-running legs, which this task and the pilot charter both forbid. **It is recorded as a
hypothesis and must not be reported as a cause.**

## What this bundle deliberately did NOT do

- It did **not** commit the modified `streamlit_app.py`. The file is untouched at
  `47e4fafa…` and still uncommitted. Committing it would silently promote a user-owned
  working-tree change into production behaviour under a task that does not authorize it.
- It did **not** re-run any pilot leg, unseen paper, or biological evaluation.
- It did **not** alter run directories, gold, pilot outputs, protected state, or `main`.

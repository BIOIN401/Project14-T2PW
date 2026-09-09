# `ORCH-739` G11 evidence

Six bounded jobs, **6 artifacts, 0 non-compliant** under `g11_evidence.py check --task ORCH-739`.
Every one carries `FINAL SURVIVING COUNT : 0` and `cleanup : success`.

| report | what ran | live calls |
|---|---|---:|
| `01-delivery-census.json` | `orch739_delivery_census.py` — read-only leg census, 27 legs, 1,384 traced calls | 0 |
| `02-c121-import-check.json` | `orch734_pathwhiz_import_check.py` on the two previously untracked PWMLs | 0 |
| `03-budget-model-probe.json` | budget/model probe, `easy` profile, both models, 3 budgets | 18 |
| `04-budget-probe-hard.json` | budget/model probe, `hard` profile — **reproduces the production empty shape** | 20 |
| `05-reasoning-capped.json` | `reasoning: {max_tokens: budget/4}` — **the cap is ignored; no improvement** | 10 |
| `06-reasoning-disabled.json` | `reasoning: {enabled: false}` — **0 empty at both budgets** | 10 |

`04` and `06` are the behavioural pair a future `F-199` card should cite: the same model, prompt,
budget and temperature, with one request parameter as the only difference.

## Where the tool payloads live, and why they are not here

Each job's **data output** is under [`../../orch739/`](../../orch739/), matching the
`<seq>-<label>-data.json` names. Only bounded-wrapper cleanup reports belong in a G11 task
directory — `g11_evidence.py` validates every `.json` it finds here against the wrapper schema, and
a tool payload sitting alongside them is counted as a malformed report. `ORCH-734`'s directory
follows the same convention.

**`03`–`06` record a `--out` path under this directory**, because the payloads were written here
first and moved afterwards. `01` and `02` were re-run against the final path, so their recorded
commands are self-consistent; the four probe jobs were **not** re-run, because that would have meant
issuing 58 more live calls to correct a filing decision. The payloads are byte-unchanged from the
run that produced them.

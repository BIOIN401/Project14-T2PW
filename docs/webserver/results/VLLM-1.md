# VLLM-1: vLLM provider and per-run LLM overrides

Branch `pwml/vllm`, head `467c744f` (`1d745d7e` plus review fixes), base `main` `a6c07e16`. Ruling: D-W28 (e).

## What it adds
- A provider `vllm`, which is OpenAI-compatible. It is configured with `VLLM_BASE_URL` (default
  `http://127.0.0.1:8000/v1`), `VLLM_MODEL`, `VLLM_API_KEY` (default `EMPTY`) and `VLLM_MAX_TOKENS` (default
  16000, clamped to 500..128000).
- Per-run overrides `T2PW_LLM_PROVIDER` and `T2PW_LLM_MODEL`, which win over `.env`.
  - `src/t2pw/llm/client.py` loads `.env` with `override=True`, so six named variables are saved before
    dotenv runs and restored after it.
  - With all of them unset, behaviour is byte-identical to before.
- In vllm mode, a caller's `model_override` (for example `OPENROUTER_*_MODEL` from the completeness audit)
  is ignored. This fixes review finding F1.
- `scripts/vllm_tunnel.ps1` opens an SSH tunnel to the lab machine through the `lab-gpu` alias. It holds no
  credentials.
- `docs/setup.md` gains a vLLM section.

Files changed: `src/t2pw/llm/client.py`, `src/t2pw/service/defaults.py`, `docs/setup.md`,
`scripts/vllm_tunnel.ps1`, `tests/test_llm_vllm_provider.py`.

## Tests
All runs went through the bounded wrapper, with 0 survivors, on `467c744f`:
- focused: 188 passed (20 of them vllm provider tests);
- SMOKE: 511;
- equivalence: 8 (run_pathway), 14 (app vs service), 6 (default path), each file run alone.

## Review
`ws-reviewer` (opus/high):
- Round 1: CHANGES REQUESTED. F1 was blocking; F2, F4, F5 and F6 were notes.
- Round 2: APPROVE WITH NOTES.
  - N1: one test does not clear `LLM_PROVIDER` first. It is only fragile in an unusual shell.
  - N2: the 401 message shows the OpenRouter key mask in vllm mode. This is pre-existing and leaks nothing.

## Open questions
- In vllm mode, or when `T2PW_LLM_MODEL` is set, the Stage-1 ladder's alternate-model rung re-asks the same
  model. That costs time, not correctness. Fixing it needs a `pipeline/` change. `OPENROUTER_EXTRACTION_FALLBACK_MODEL`
  can only be unset by editing `.env`.

## Dispatch
| Step | Agent | Model / effort |
|---|---|---|
| Implement and fix | general-purpose | sonnet / high |
| Review x2 | ws-reviewer | opus / high |

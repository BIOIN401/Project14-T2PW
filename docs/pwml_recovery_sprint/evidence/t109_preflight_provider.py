"""T-109 provider preflight — prove the configured provider and the nine pinned
models resolve, WITHOUT printing a secret and WITHOUT issuing a billed call.

Why this exists as a committed probe rather than a shell one-liner
------------------------------------------------------------------
`T108-READINESS.md` row 15 is *"configured provider and pinned model available"*,
and the sprint has now twice been misled by reading `.env` with a regex instead of
through the loader the pipeline actually uses:

* the `.env` **worktree trap** (`TEST_MATRIX` § 0) -- `.env` is untracked, so a
  worktree silently gets `LLM_PROVIDER=local` while the primary checkout issues
  real billed calls. Anything that certifies the provider must say WHICH tree it
  read.
* **`OPENROUTER_API_KEY` is written `KEY = value`, with spaces around the `=`.**
  `grep -E "^OPENROUTER_API_KEY="` therefore finds **nothing**, and the only
  match left is the commented-out line above it, so a shell check reports the key
  ABSENT when `python-dotenv` resolves it fine. This probe was written after that
  false alarm fired during ORCH-720. **Read the file the way the program reads
  it, or do not claim to have read it.**

What is printed: names, booleans, lengths and a prefix test. **Never a value.**

Usage:  t109_preflight_provider.py <repo>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EXPECTED_PROVIDER = "openrouter"
EXPECTED_MODEL = "deepseek/deepseek-v4-flash"
EXPECTED_TEMPERATURE = "0"

#: Every model the pipeline resolves per stage. All nine must be pinned to the
#: same model, because a run whose stages disagree is not the pinned run.
MODEL_VARS = (
    "OPENROUTER_MODEL",
    "OPENROUTER_PREPROCESSOR_MODEL",
    "OPENROUTER_EXTRACTION_MODEL",
    "OPENROUTER_INFERENCE_MODEL",
    "OPENROUTER_AUDIT_MODEL",
    "OPENROUTER_CURATOR_MODEL",
    "OPENROUTER_GAP_MODEL",
    "OPENROUTER_OVERWATCH_MODEL",
    "OPENROUTER_FINAL_COMPLETENESS_MODEL",
)


def main() -> int:
    tree = Path(sys.argv[1]).resolve()
    env_path = tree / ".env"
    print(f"tree under measurement : {tree}")
    print(f".env read              : {env_path}  exists={env_path.is_file()}")
    if not env_path.is_file():
        print("VERDICT: FAIL -- no .env in the tree under measurement")
        return 1

    # Load into a private mapping, not os.environ, so this probe cannot leak
    # configuration into anything that runs after it in the same process tree.
    from dotenv import dotenv_values

    values = dotenv_values(env_path)

    failures = []

    provider = (values.get("LLM_PROVIDER") or "").split("#")[0].strip()
    print(f"LLM_PROVIDER           : {provider!r}")
    if provider != EXPECTED_PROVIDER:
        failures.append(f"LLM_PROVIDER is {provider!r}, expected {EXPECTED_PROVIDER!r}")

    temperature = (values.get("LLM_TEMPERATURE") or "").split("#")[0].strip()
    print(f"LLM_TEMPERATURE        : {temperature!r}")
    if temperature != EXPECTED_TEMPERATURE:
        failures.append(f"LLM_TEMPERATURE is {temperature!r}, expected {EXPECTED_TEMPERATURE!r}")

    retries = (values.get("LLM_MAX_RETRIES") or "").split("#")[0].strip()
    print(f"LLM_MAX_RETRIES        : {retries!r}  (ratified at 3; T108-READINESS section 3)")

    base_url = (values.get("OPENROUTER_BASE_URL") or "").split("#")[0].strip()
    print(f"OPENROUTER_BASE_URL    : {base_url!r}")
    if not base_url.startswith("https://"):
        failures.append(f"OPENROUTER_BASE_URL is not https: {base_url!r}")

    print()
    print("pinned models (all nine must be identical):")
    seen = set()
    for name in MODEL_VARS:
        model = (values.get(name) or "").split("#")[0].strip()
        ok = model == EXPECTED_MODEL
        print(f"  {name:<40} {model!r:<32} {'OK' if ok else 'MISMATCH'}")
        seen.add(model)
        if not ok:
            failures.append(f"{name} is {model!r}, expected {EXPECTED_MODEL!r}")
    print(f"  distinct model values across the nine  : {len(seen)}  (must be 1)")

    print()
    # The key: presence, length and prefix only. Never the value, and never a
    # slice of the value long enough to be useful.
    key = values.get("OPENROUTER_API_KEY")
    present = bool(key and key.strip())
    length = len(key.strip()) if present else 0
    prefix_ok = bool(present and key.strip().startswith("sk-or-v1-"))
    print(f"OPENROUTER_API_KEY     : present={present} length={length} prefix_ok={prefix_ok}")
    print("  (value NEVER printed. Note: this file writes it as 'KEY = value' with")
    print("   spaces, so a '^OPENROUTER_API_KEY=' grep finds only the commented line.)")
    if not present:
        failures.append("OPENROUTER_API_KEY is absent or empty")
    if not prefix_ok:
        failures.append("OPENROUTER_API_KEY does not carry the expected sk-or-v1- prefix")

    print()
    print("NO NETWORK CALL WAS MADE. This probe proves CONFIGURATION, not reachability;")
    print("reachability is proven by the first leg of the run itself and by nothing cheaper.")

    print()
    if failures:
        for line in failures:
            print(f"FAILURE: {line}")
        print(f"VERDICT: FAIL ({len(failures)} condition(s))")
        return 1
    print("VERDICT: PASS -- provider openrouter, nine models pinned identically, key present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

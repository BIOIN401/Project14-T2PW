"""D-086 observability probe: read actual OpenRouter account usage.

Orchestration tooling, not pipeline code. Nothing under ``src/`` imports it.

D-086 registered the gap: **the pipeline records no token usage anywhere**, so no run in
this sprint can be costed from its own artifacts. The charter for this wave asks that the
actual spend be read from the provider account instead, that measured values be
distinguished from estimates, and that any inability to retrieve exact usage be documented
rather than guessed at.

What this can and cannot answer
-------------------------------
OpenRouter's key endpoints report **cumulative account usage**, not per-run usage. There is
no run identifier in the pipeline's requests to filter on, because the pipeline sends none.
So this probe establishes an account-level total and its limit; attributing a slice of it to
T-107 specifically is **not** something the provider can be asked for here. That distinction
is the whole point of the exercise and is preserved in the output.

Credential handling
-------------------
The key is read from ``.env`` and used only as an ``Authorization`` header to
``openrouter.ai``, the provider it belongs to. It is **never printed, never written to the
report, and never sent anywhere else**. The output records the usage numbers only:
``label`` -- OpenRouter's own truncated rendering of the key -- and ``creator_user_id`` are
redacted by :func:`redact` on the response path, so the committed log carries no account
identifier either. ``.env`` is untracked; this file is committed, so it must stay free of
any value read from it.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

BASE = "https://openrouter.ai/api/v1"
ENDPOINTS = ("/auth/key", "/credits")
TIMEOUT_SECONDS = 30

#: Read verbatim: the sprint's ``.env`` carries the name with a trailing space
#: (``OPENROUTER_API_KEY =``), which a naive ``split("=")[0]`` lookup misses. Both
#: spellings are accepted rather than one being assumed.
KEY_NAMES = ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY ")


def read_env(path: str) -> Dict[str, str]:
    """``.env`` as a dict, keys and values stripped, ``#`` comments dropped."""

    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            out[name.strip()] = value.strip().strip('"').strip("'")
    return out


def find_key(env: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """``(key, name_it_was_found_under)``; ``(None, None)`` when absent."""

    for name in KEY_NAMES:
        value = env.get(name.strip()) or env.get(name)
        if value:
            return value, name.strip()
    return None, None


#: Fields the provider returns that identify the ACCOUNT rather than the spend.
#: ``label`` is OpenRouter's own truncated rendering of the key and ``creator_user_id``
#: names the account holder; neither is a usage figure and neither belongs in a committed
#: artifact. Redacted here rather than hand-edited out of the log afterwards, so the probe
#: and its output cannot drift apart.
REDACT_FIELDS = ("label", "creator_user_id")


def redact(body: Any) -> Any:
    """``body`` with every :data:`REDACT_FIELDS` key replaced, at any depth."""

    if isinstance(body, dict):
        return {
            k: ("<redacted>" if k in REDACT_FIELDS else redact(v))
            for k, v in body.items()
        }
    if isinstance(body, list):
        return [redact(item) for item in body]
    return body


def get(url: str, key: str) -> Dict[str, Any]:
    """One authenticated GET. Failures are reported, never raised past the caller."""

    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {"ok": True, "status": response.status, "body": redact(json.loads(body))}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        return {"ok": False, "status": exc.code, "error": "http_error", "detail": detail}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": "url_error", "detail": str(exc.reason)[:500]}
    except Exception as exc:  # noqa: BLE001 -- a probe must not die on an unexpected shape
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:500]}


def main(argv: Optional[list] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    root = args[0] if args else "."
    env_path = os.path.join(root, ".env")

    print("=" * 78)
    print("D-086 OBSERVABILITY PROBE -- actual OpenRouter account usage")
    print("=" * 78)
    print(f".env path        : {env_path}")
    print(f".env present     : {os.path.isfile(env_path)}")

    env = read_env(env_path)
    key, found_as = find_key(env)
    print(f"key variable     : {found_as or '(not found)'}")
    print(f"key present      : {bool(key)}")
    if key:
        # Length only. Never the value, never a prefix that could identify the account.
        print(f"key length       : {len(key)}")
    print(f"pinned model     : {env.get('OPENROUTER_MODEL', '(unset)')}")
    print(f"temperature      : {env.get('LLM_TEMPERATURE', '(unset)')}")
    print()

    if not key:
        print("RESULT: NOT RETRIEVABLE -- no OpenRouter key in .env.")
        print("Actual spend cannot be read. This is recorded as unretrieved, not estimated.")
        return 0

    for endpoint in ENDPOINTS:
        url = BASE + endpoint
        print("-" * 78)
        print(f"GET {url}")
        result = get(url, key)
        if result.get("ok"):
            print(f"  status : {result['status']}")
            print("  body   :")
            print(json.dumps(result["body"], indent=4, sort_keys=True))
        else:
            print(f"  FAILED : {result.get('error')} status={result.get('status')}")
            print(f"  detail : {result.get('detail')}")
    print("-" * 78)
    print()
    print("INTERPRETATION LIMIT, stated rather than left implicit:")
    print("  These figures are CUMULATIVE ACCOUNT TOTALS. The pipeline sends no run")
    print("  identifier, so the provider cannot attribute any part of them to T-107.")
    print("  Any per-run number derived from them would be an estimate, not a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

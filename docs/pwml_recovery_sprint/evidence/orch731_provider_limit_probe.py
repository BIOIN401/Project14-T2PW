"""ORCH-731: what does the provider actually do with max_tokens=16000?

READ-ONLY DIAGNOSTIC. Runs no pipeline leg, no Stage-1 extraction, no Streamlit app.
It makes a SMALL, FIXED number of direct chat-completion calls (<= 3) to answer three
questions the archived runs cannot, because per-call token usage is recorded NOWHERE:
``client._record_usage`` only increments an in-memory cumulative counter that is never
persisted, and ``orch716_openrouter_usage.log`` is account-level.

THE QUESTIONS
  Q2  What exact ``max_tokens`` reaches the API request?
  Q3  What does the provider report for prompt_tokens, completion_tokens, finish_reason,
      and any reasoning-token accounting?
  Q4  Was a truncated response cut at a provider maximum, a parser boundary, a transport
      limit, or our own client limit?

THE HYPOTHESIS UNDER TEST. Across 2,675 archived model attempts, 922 carry
``finish_reason=length`` and **842 of those returned ZERO content characters** with
``status=empty``. Meanwhile the only two genuine Stage-1 truncations in the whole corpus
both stopped near ~10k characters (9,501 and 10,895) despite a 16000-token request --
while Stage-2 reached 55,660 characters on the same nominal budget. A single mechanism
would explain all of it: the model emits REASONING tokens that are billed against
``max_tokens`` but never appear in ``message.content``. If so, ``finish_reason=length``
with empty content is the budget being consumed before any answer is produced, and it is
not a truncation of our output at all.

This probe does not modify any cache, run artifact or production file. It sends its own
prompts, not paper text.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional, Sequence


def dump(label: str, resp: Any, requested: int) -> Dict[str, Any]:
    choice = (resp.choices or [None])[0]
    msg = getattr(choice, "message", None)
    content = getattr(msg, "content", None) or ""
    reasoning = (
        getattr(msg, "reasoning", None)
        or getattr(msg, "reasoning_content", None)
        or ""
    )
    usage = getattr(resp, "usage", None)
    raw_usage: Dict[str, Any] = {}
    if usage is not None:
        for key in dir(usage):
            if key.startswith("_"):
                continue
            try:
                value = getattr(usage, key)
            except Exception:
                continue
            if callable(value):
                continue
            if isinstance(value, (int, float, str)) or value is None:
                raw_usage[key] = value
            else:
                try:
                    raw_usage[key] = json.loads(json.dumps(value, default=str))
                except Exception:
                    raw_usage[key] = str(value)[:300]
    record = {
        "label": label,
        "requested_max_tokens": requested,
        "finish_reason": getattr(choice, "finish_reason", None),
        "native_finish_reason": getattr(choice, "native_finish_reason", None),
        "content_chars": len(content),
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else None,
        "model_echoed": getattr(resp, "model", None),
        "provider": getattr(resp, "provider", None),
        "usage": raw_usage,
    }
    print("-" * 88)
    print(f"{label}   requested max_tokens={requested}")
    print(f"  finish_reason        : {record['finish_reason']!r}   native={record['native_finish_reason']!r}")
    print(f"  model echoed by API  : {record['model_echoed']!r}   provider={record['provider']!r}")
    print(f"  content chars        : {record['content_chars']}")
    print(f"  reasoning chars      : {record['reasoning_chars']}")
    print(f"  usage                : {json.dumps(raw_usage)}")
    ct = raw_usage.get("completion_tokens")
    if isinstance(ct, int):
        print(f"  completion_tokens vs requested : {ct} / {requested}"
              f"  ({'AT/OVER THE CAP' if ct >= requested - 8 else 'BELOW THE CAP'})")
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    root = args[0] if args else "."
    sys.path.insert(0, os.path.join(root, "src"))
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv(os.path.join(root, ".env"))
    from openai import OpenAI  # noqa: E402

    model = os.getenv("OPENROUTER_EXTRACTION_MODEL") or os.getenv("OPENROUTER_MODEL") or ""
    base = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    key = os.getenv("OPENROUTER_API_KEY") or ""
    requested = int(os.getenv("OPENROUTER_EXTRACTION_MAX_TOKENS") or "16000")

    print("=" * 88)
    print("ORCH-731 -- provider behaviour under the Stage-1 budget")
    print("=" * 88)
    print(f"model    : {model}")
    print(f"base_url : {base}")
    print(f"api key  : {'present' if key else 'ABSENT'} (never printed)")
    print(f"Stage-1 OPENROUTER_EXTRACTION_MAX_TOKENS = {requested}")
    print()
    if not key or not model:
        print("no credentials or model configured -- cannot probe")
        return 0

    client = OpenAI(base_url=base, api_key=key)
    records = []

    # 1. A request that CANNOT be satisfied inside the budget: forces finish_reason=length
    #    and shows exactly how the budget was spent.
    records.append(
        dump(
            "A. deliberately over-long output, full Stage-1 budget",
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content":
                           "Output a JSON object with a key 'items' whose value is an array of "
                           "4000 objects, each {\"i\": <index>, \"s\": \"filler text here\"}. "
                           "Output the JSON only. Do not stop early."}],
                temperature=0,
                max_tokens=requested,
                response_format={"type": "json_object"},
            ),
            requested,
        )
    )

    # 2. A trivially short answer at the same budget: isolates whether reasoning tokens
    #    are billed even when the answer is tiny.
    records.append(
        dump(
            "B. trivial output, same budget",
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content":
                           "Reply with exactly this JSON and nothing else: {\"ok\": true}"}],
                temperature=0,
                max_tokens=requested,
                response_format={"type": "json_object"},
            ),
            requested,
        )
    )

    # 3. The same over-long request at a DELIBERATELY TINY budget. If completion_tokens
    #    lands on the tiny cap, the cap is honoured and the limit is ours; if the earlier
    #    call stopped well below its own cap, the limit was the provider's.
    records.append(
        dump(
            "C. over-long output, max_tokens=256 (control)",
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content":
                           "Output a JSON object with a key 'items' whose value is an array of "
                           "4000 objects, each {\"i\": <index>, \"s\": \"filler text here\"}. "
                           "Output the JSON only. Do not stop early."}],
                temperature=0,
                max_tokens=256,
                response_format={"type": "json_object"},
            ),
            256,
        )
    )

    print()
    print("=" * 88)
    print("READING THE RESULT")
    print("=" * 88)
    a = records[0]
    ct = a["usage"].get("completion_tokens")
    if isinstance(ct, int):
        if ct >= a["requested_max_tokens"] - 8:
            print(f"  A consumed {ct} of {a['requested_max_tokens']} -- OUR OWN max_tokens is the")
            print("  binding limit. The response was cut by the value we sent.")
        else:
            print(f"  A stopped at {ct} of {a['requested_max_tokens']} -- our cap was NOT reached.")
            print("  Something provider-side ended it; our max_tokens is not the binding limit.")
    if a["reasoning_chars"]:
        print(f"  Reasoning content present ({a['reasoning_chars']} chars): reasoning tokens are")
        print("  billed against max_tokens but never appear in message.content -- which is how")
        print("  finish_reason=length can arrive with ZERO content characters.")
    else:
        print("  No reasoning content field was returned; the empty-completion mechanism is")
        print("  NOT explained by reasoning tokens on this evidence.")
    print()
    print("  NOTE: this probe uses its own prompts. It measures the PROVIDER, not the paper.")
    out = os.path.join(root, "docs/pwml_recovery_sprint/evidence/orch731_provider_probe.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"instrument": "orch731", "reran_no_leg": True, "records": records},
                  handle, indent=1, ensure_ascii=False)
    print(f"  report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

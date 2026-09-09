"""ORCH-739: does a small `max_tokens` budget produce empty completions, and is
that a property of the MODEL rather than of the provider being flaky?

LIVE DIAGNOSTIC, DELIBERATELY TINY. Runs no pipeline leg, no Stage-1 extraction,
no Streamlit app, and touches no run tree, cache or production file. It sends its
own short prompts -- never paper text -- and makes a small, fixed, argument-capped
number of direct chat-completion calls.

THE HYPOTHESIS UNDER TEST
-------------------------
`ORCH-733` measured Stage-1 delivery failure as a small tail (8 legs in 336) and
concluded delivery was not the bottleneck. That is correct AT STAGE 1. But the
ORCH-739 leg census finds the empty-completion rate is not uniform across stages;
it tracks the call's completion budget:

    call site                                    max_tokens   empty rate
    map_ids alias / stoich classifier ("chat")      300         79.7 %
    rag_prose_extraction                           1500         54.4 %
    gap resolver                                 450-900        31.3 %
    Stage 2 inference                             16000         17.1 %
    Stage 1 extraction                            16000         16.3 %
    preprocessor                                  12000          0.0 %

`ORCH-731` A4 established the mechanism that would explain this: reasoning tokens
are billed against `max_tokens` and never appear in `message.content`. If the model
reasons before answering, a small budget is consumed before a single content token
is emitted, and the provider correctly reports `finish_reason=length` with empty
content. That is not a provider outage and not a truncation of our answer. It is a
model whose output budget is being spent somewhere we do not read.

WHAT THIS PROBE DECIDES, AND WHAT IT DOES NOT
---------------------------------------------
DECIDES  : whether empty-at-small-budget reproduces on demand, and whether it is
           model-specific -- the same prompt and budget against the configured
           primary and the configured Stage-1 fallback.
DOES NOT : say anything about biological extraction quality. A model that delivers
           bytes is not thereby a better biologist. `PRODUCT_CONTRACT.md` outranks
           any delivery number here, and no extraction-quality claim is made or
           implied by this tool.

`D-097`'s prohibition on blaming token budget is NOT overturned by this probe and
is not in tension with it. That ruling addressed Stage-1 TRUNCATION at 16000, where
ORCH-731 measured the budget as demonstrably non-binding (16000 tokens bought 38,462
characters while the failing leg stopped at 9,501). This is a different class -- ZERO
content at 300 -- where the budget is binding by construction. Establishing that
distinction is the entire purpose of this file, and "raise max_tokens" is NOT
proposed by it: those budgets are literals in production source, which is frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Structured tasks of the SAME SHAPE as the failing production call sites:
# response_json, a small answer, zero paper text. `stoich/classifier.py` and
# `mapping/map_ids.py` both ask for a small JSON object at max_tokens=300.
#
# TWO difficulty profiles, because the first round of this probe found the easy
# one does NOT reproduce the production shape. `easy` fits its reasoning inside
# 300 tokens and always answers. `hard` is the realistic case: an ambiguous
# non-model-organism alias question of the kind `map_ids.py` actually asks, where
# the reasoning a model wants to do exceeds a 300-token completion budget.
PROFILES = {
    "easy": (
        "You are a precise classifier. Output JSON only.",
        'Classify this reaction participant. Return exactly '
        '{"class": "<substrate|product|cofactor>"} and nothing else. '
        "Participant: ATP, in the reaction 'glucose + ATP -> glucose-6-phosphate + ADP'.",
    ),
    "hard": (
        "You are a strict protein alias resolver. Output JSON only.",
        'Return exactly {"aliases": ["<name>", ...]} listing every synonym, gene '
        "symbol and systematic name under which the following enzyme is catalogued, "
        "and nothing else. Enzyme: DmaW, a dimethylallyltryptophan synthase acting on "
        "L-tryptophan and dimethylallyl diphosphate in the ergot alkaloid pathway of "
        "the fungus Aspergillus japonicus. Consider orthologue naming across Claviceps, "
        "Neotyphodium, Penicillium and Aspergillus, EC-number-derived names, and any "
        "systematic enzyme nomenclature. Weigh each candidate before answering.",
    ),
}


def load_dotenv(root: Path) -> None:
    """Read .env the way the app does. Never prints or logs a value."""
    path = root / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split("#")[0].strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def probe_once(
    client: Any, model: str, max_tokens: int, profile: str, reasoning_mode: str
) -> Dict[str, Any]:
    """One call. ``reasoning_mode`` selects the candidate remedy under test.

    ``none``     - today's production request, verbatim: no reasoning control.
    ``disabled`` - OpenRouter ``reasoning.enabled = false``.
    ``capped``   - OpenRouter ``reasoning.max_tokens`` held to a quarter of the
                   completion budget, so reasoning cannot consume all of it.

    The two non-``none`` modes are the ONLY thing that differs from the
    production request. Whether they change delivery is the question; whether
    they would change biology is NOT decided here and is not claimed.
    """
    system, user = PROFILES[profile]
    started = time.time()
    row: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "profile": profile,
        "reasoning_mode": reasoning_mode,
    }
    extra: Dict[str, Any] = {}
    if reasoning_mode == "disabled":
        extra["reasoning"] = {"enabled": False}
    elif reasoning_mode == "capped":
        extra["reasoning"] = {"max_tokens": max(32, max_tokens // 4)}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            timeout=120.0,
            **({"extra_body": extra} if extra else {}),
        )
    except Exception as exc:  # noqa: BLE001 - a provider error IS a result here
        row.update(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "seconds": round(time.time() - started, 2),
            }
        )
        return row

    choice = (resp.choices or [None])[0]
    msg = getattr(choice, "message", None)
    content = getattr(msg, "content", None) or ""
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
    usage = getattr(resp, "usage", None)

    def usage_get(*names: str) -> Any:
        for holder in (usage, getattr(usage, "completion_tokens_details", None)):
            if holder is None:
                continue
            for name in names:
                value = getattr(holder, name, None)
                if value is not None:
                    return value
        return None

    row.update(
        {
            "ok": True,
            "seconds": round(time.time() - started, 2),
            "finish_reason": getattr(choice, "finish_reason", None),
            "content_chars": len(content),
            "content_preview": content[:120],
            "reasoning_chars": len(reasoning),
            "prompt_tokens": usage_get("prompt_tokens"),
            "completion_tokens": usage_get("completion_tokens"),
            "reasoning_tokens": usage_get("reasoning_tokens"),
            # OpenRouter names the backend that actually served the request.
            # ORCH-733 section 4 recorded that production never reads this.
            "provider": getattr(resp, "provider", None),
            "served_model": getattr(resp, "model", None),
            "empty": len(content) == 0,
        }
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="300,1500,16000")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--profile", default="easy", choices=sorted(PROFILES))
    ap.add_argument("--reasoning-mode", default="none", choices=["none", "disabled", "capped"])
    ap.add_argument("--models", default="", help="comma-separated override; default is primary,fallback")
    ap.add_argument(
        "--max-calls",
        type=int,
        default=24,
        help="hard ceiling; the probe refuses to exceed it rather than trimming silently",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    load_dotenv(root)

    primary = (os.getenv("OPENROUTER_EXTRACTION_MODEL") or os.getenv("OPENROUTER_MODEL") or "").strip()
    fallback = (os.getenv("OPENROUTER_EXTRACTION_FALLBACK_MODEL") or "").strip()
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()

    print("=" * 78)
    print("ORCH-739 -- empty-at-small-budget, by model. LIVE, BOUNDED.")
    print("=" * 78)
    print(f"primary  : {primary or '(unset)'}")
    print(f"fallback : {fallback or '(unset)'}")
    print(f"base_url : {base_url}")
    print(f"api key  : {'PRESENT (never printed)' if api_key else 'ABSENT'}")

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              or [m for m in (primary, fallback) if m])
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    planned = len(models) * len(budgets) * args.repeats
    print(f"profile  : {args.profile}")
    print(f"reasoning: {args.reasoning_mode}")
    print(f"planned calls : {len(models)} models x {len(budgets)} budgets x {args.repeats} = {planned}")

    if not api_key or not models:
        print("\nno credentials or no model configured -- cannot probe")
        return 2
    if planned > args.max_calls:
        print(f"\nREFUSED: {planned} calls exceeds --max-calls {args.max_calls}")
        return 2

    from openai import OpenAI  # imported here so --help works without credentials

    client = OpenAI(api_key=api_key, base_url=base_url)

    rows: List[Dict[str, Any]] = []
    for model in models:
        for budget in budgets:
            for _ in range(args.repeats):
                row = probe_once(client, model, budget, args.profile, args.reasoning_mode)
                rows.append(row)
                mark = "EMPTY" if row.get("empty") else ("ERR" if not row.get("ok") else "ok   ")
                print(
                    f"  {model:<30} max_tokens={budget:<6} {mark}"
                    f"  content={row.get('content_chars', '-'):<5}"
                    f" finish={str(row.get('finish_reason')):<8}"
                    f" completion_tok={str(row.get('completion_tokens')):<6}"
                    f" reasoning_tok={str(row.get('reasoning_tokens')):<6}"
                    f" provider={str(row.get('provider'))[:18]:<18}"
                    f" {row.get('seconds')}s"
                )
                time.sleep(0.35)

    summary: Dict[str, Any] = {}
    for model in models:
        for budget in budgets:
            sel = [r for r in rows if r["model"] == model and r["max_tokens"] == budget]
            got = [r for r in sel if r.get("ok")]
            summary[f"{model}@{budget}"] = {
                "calls": len(sel),
                "errors": len(sel) - len(got),
                "empty": sum(1 for r in got if r.get("empty")),
                "median_content_chars": sorted(r["content_chars"] for r in got)[len(got) // 2]
                if got
                else None,
                "providers": sorted({str(r.get("provider")) for r in got}),
            }

    print("\n--- summary: empty / calls, by model and budget ---")
    for key, value in summary.items():
        print(f"  {key:<44} empty {value['empty']}/{value['calls']}   providers={value['providers']}")

    report = {
        "tool": "orch739_budget_model_probe.py",
        "primary": primary,
        "fallback": fallback,
        "profile": args.profile,
        "reasoning_mode": args.reasoning_mode,
        "prompt_shape": "response_json, small structured answer, no paper text",
        "summary": summary,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

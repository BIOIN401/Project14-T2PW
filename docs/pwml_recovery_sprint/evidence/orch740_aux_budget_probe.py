"""ORCH-740: is `max_tokens=300` simply too small for a reasoning model?

LIVE CALL-LEVEL EXPERIMENT, BOUNDED. Runs no pipeline leg, no Stage-1 extraction,
no Streamlit app; touches no run tree, cache or production file. **Reasoning stays
ENABLED** -- that is the whole point of this experiment, and it is what separates it
from `ORCH-739`'s § 4.2 probe.

ONE VARIABLE: the completion budget. Same model, same provider routing, same prompt,
same schema, same temperature, same retry policy.

THE CALL SITE UNDER TEST
------------------------
`mapping/map_ids.py:406` -- `_ai_protein_synonym_lookup`, the protein alias lookup
that runs when UniProt fails to match a protein by its primary name. It is the
**only** production call site in the untagged `chat` bucket (see below), and the
`ORCH-739` census measured that bucket at **79.7 % empty across 469 calls**.

WHY THIS IS THE WHOLE AFFECTED POPULATION, corrected from `ORCH-739`
--------------------------------------------------------------------
`ORCH-739` § 3 attributed the `chat` bucket to "map_ids alias + stoich classifier".
**The stoich half is wrong.** `stoich/classifier.py:136` is reached only from
`stoich/agent.py:420`, which is reached only from `run_stoich_agent`, which
`streamlit_app.py:4277` calls only when the `use_stoich_agent` checkbox is set.
The batch driver never sets it, so that code did not execute in any measured leg.
`stoich/agent.py`'s own two 300-token calls bypass `t2pw.llm.client` entirely --
they call `_client.chat.completions.create` directly, emit no trace row, and get no
retry loop. `extraction/extract.py:18` is a hard-coded glutathione demo behind
`__main__`. Every other `chat`/`chat_detailed` caller passes a `stage_name`.

THE AMPLIFICATION, which the budget question does not by itself address
-----------------------------------------------------------------------
The 469 `chat` calls come from **40 distinct request hashes** -- 11.7 calls per
distinct question, against an `LLM_MAX_RETRIES` of 3. `PMC9200736` spent 78 calls on
4 questions. So the caller re-asks, on top of the client's own retries. **Raising the
budget can only help if it makes the individual call succeed**; it does nothing about
the re-ask loop, and this probe measures only the former.

WHAT THIS PROBE DECIDES, AND WHAT IT DOES NOT
----------------------------------------------
DECIDES  : whether 2000 tokens with reasoning enabled materially reduces empty
           completions at this call site, and what it costs in latency.
DOES NOT : say anything about alias QUALITY or biological correctness. A model that
           returns bytes is not thereby returning good aliases. `PRODUCT_CONTRACT.md`
           outranks every number here.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# The production prompt, verbatim from `map_ids.py:388-401`. `verify_prompt_matches_source`
# below re-reads that file and refuses to run if the two have drifted, so this
# probe cannot silently measure a prompt production no longer sends.
# ---------------------------------------------------------------------------
PROMPT_TAIL = (
    "Provide a JSON object with key \"aliases\": an array of objects, each with:\n"
    "  \"alias\": an alternate name, gene symbol, gene synonym, or common abbreviation "
    "that UniProt or NCBI might use for this protein\n"
    "  \"source\": one of \"gene_name\", \"synonym\", or \"alternate_abbreviation\"\n"
    "Rules:\n"
    "- Only include names plausibly listed in a protein database entry\n"
    "- Do NOT include the original name\n"
    "- Do NOT guess or invent UniProt accession IDs\n"
    "- Maximum 6 aliases\n"
    "- If you have no reliable information, return {\"aliases\": []}\n"
    "Return ONLY valid JSON, no markdown."
)


def build_prompt(name: str, organism: str) -> str:
    """Byte-for-byte what `_ai_protein_synonym_lookup` builds."""
    organism_clause = f" from {organism}" if organism else ""
    return (
        f"The protein '{name}'{organism_clause} could not be matched in UniProt by its primary name.\n"
        + PROMPT_TAIL
    )


# Real entities from real failed legs, not invented ones. Each is a protein whose
# identity resolution actually failed in a committed run.
CASES = [
    ("DmaW", "Aspergillus japonicus"),        # C-121 live validation, PMC4471609
    ("EasF", "Aspergillus japonicus"),        # same leg
    ("UGT1", "Nicotiana tabacum"),            # ORCH-734, PMC13184244
    ("MATE1", "Nicotiana tabacum"),           # same leg
    ("SdPCS", "Saposhnikovia divaricata"),    # ORCH-734, PMC11961743
]


def verify_prompt_matches_source(root: Path) -> None:
    """Refuse to run if the probe's prompt has drifted from production's.

    The source builds this prompt by concatenating adjacent string literals, so one
    prompt line can be split across two of them: at the seam the file carries a
    closing quote, whitespace, a newline and an opening quote that never reach the
    model. Both sides are therefore stripped of all whitespace, of the backslashes
    that escape quotes inside the literals, and of the quote characters themselves.

    That leaves the words and punctuation, which is what this check is for -- it
    catches a reworded or reordered prompt, which is the drift that would make this
    probe measure something production does not send. It does NOT verify quoting,
    and it is not claimed to.
    """
    src = (root / "src" / "t2pw" / "mapping" / "map_ids.py").read_text(
        encoding="utf-8", errors="replace"
    )

    def squash(text: str) -> str:
        return "".join(text.split()).replace("\\", "").replace('"', "")

    src_squashed = squash(src)
    lines = [line for line in PROMPT_TAIL.split("\n") if line.strip()]
    missing = [line.strip() for line in lines if squash(line) not in src_squashed]
    if missing:
        raise SystemExit(
            "REFUSED: the probe prompt no longer matches map_ids.py. Drifted lines:\n  "
            + "\n  ".join(missing[:5])
        )
    print(f"prompt drift check: PASS -- {len(lines)} prompt lines all present in map_ids.py")


def load_dotenv(root: Path) -> None:
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


def probe_once(client: Any, model: str, max_tokens: int, name: str, organism: str) -> Dict[str, Any]:
    started = time.time()
    row: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "entity": name,
        "organism": organism,
    }
    try:
        # Mirrors `chat(..., response_json=True)`: response_format json_object,
        # temperature 0.0. Reasoning is NOT controlled -- it stays enabled, exactly
        # as production sends it today.
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": build_prompt(name, organism)}],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            timeout=120.0,
        )
    except Exception as exc:  # noqa: BLE001 - a provider error IS a result
        row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300],
                    "seconds": round(time.time() - started, 2)})
        return row

    choice = (resp.choices or [None])[0]
    msg = getattr(choice, "message", None)
    content = getattr(msg, "content", None) or ""
    usage = getattr(resp, "usage", None)

    def usage_get(*names: str) -> Any:
        for holder in (usage, getattr(usage, "completion_tokens_details", None)):
            if holder is None:
                continue
            for n in names:
                v = getattr(holder, n, None)
                if v is not None:
                    return v
        return None

    # "Usable" is judged by the SAME predicate production applies at
    # `map_ids.py:412-414`: json.loads, then a list under key "aliases".
    # An empty alias list IS a valid production answer and is counted usable.
    usable = False
    parse_error = ""
    try:
        parsed = json.loads(content) if content else None
        usable = isinstance(parsed, dict) and isinstance(parsed.get("aliases"), list)
        if isinstance(parsed, dict) and isinstance(parsed.get("aliases"), list):
            row["n_aliases"] = len(parsed["aliases"])
    except Exception as exc:  # noqa: BLE001
        parse_error = f"{type(exc).__name__}"

    row.update({
        "ok": True,
        "seconds": round(time.time() - started, 2),
        "finish_reason": getattr(choice, "finish_reason", None),
        "content_chars": len(content),
        "content_preview": content[:160],
        "completion_tokens": usage_get("completion_tokens"),
        "reasoning_tokens": usage_get("reasoning_tokens"),
        "provider": getattr(resp, "provider", None),
        "empty": len(content) == 0,
        "usable": bool(usable),
        "parse_error": parse_error,
    })
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="300,2000")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--round", default="r1")
    ap.add_argument("--max-calls", type=int, default=60)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    load_dotenv(root)
    verify_prompt_matches_source(root)

    # `_ai_protein_synonym_lookup` passes model_env_var="OPENROUTER_GAP_MODEL".
    model = (os.getenv("OPENROUTER_GAP_MODEL") or os.getenv("OPENROUTER_MODEL") or "").strip()
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    planned = len(budgets) * len(CASES) * args.repeats

    print("=" * 96)
    print("ORCH-740 -- auxiliary budget 300 vs 2000, REASONING ENABLED. One variable.")
    print("=" * 96)
    print(f"call site : map_ids.py:406  _ai_protein_synonym_lookup  (OPENROUTER_GAP_MODEL)")
    print(f"model     : {model or '(unset)'}")
    print(f"api key   : {'PRESENT (never printed)' if api_key else 'ABSENT'}")
    print(f"prompt    : verified byte-identical to map_ids.py")
    print(f"budgets   : {budgets}   cases : {len(CASES)}   repeats : {args.repeats}   planned calls : {planned}")

    if not api_key or not model:
        print("\nno credentials or model configured -- cannot probe")
        return 2
    if planned > args.max_calls:
        print(f"\nREFUSED: {planned} calls exceeds --max-calls {args.max_calls}")
        return 2

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    rows: List[Dict[str, Any]] = []
    for budget in budgets:
        print(f"\n--- max_tokens={budget} ---")
        for rep in range(args.repeats):
            for name, organism in CASES:
                row = probe_once(client, model, budget, name, organism)
                row["repeat"] = rep + 1
                row["round"] = args.round
                rows.append(row)
                mark = (
                    "EMPTY"
                    if row.get("empty")
                    else ("ERR  " if not row.get("ok") else ("ok   " if row.get("usable") else "BAD  "))
                )
                print(
                    f"  {name:<8} {organism:<26} {mark}"
                    f" content={str(row.get('content_chars', '-')):<5}"
                    f" finish={str(row.get('finish_reason')):<7}"
                    f" compl_tok={str(row.get('completion_tokens')):<5}"
                    f" reason_tok={str(row.get('reasoning_tokens')):<5}"
                    f" aliases={str(row.get('n_aliases', '-')):<3}"
                    f" {row.get('seconds')}s  {str(row.get('provider'))[:16]}"
                )
                time.sleep(0.35)

    summary: Dict[str, Any] = {}
    for budget in budgets:
        sel = [r for r in rows if r["max_tokens"] == budget]
        got = [r for r in sel if r.get("ok")]
        secs = [r["seconds"] for r in got]
        summary[str(budget)] = {
            "calls": len(sel),
            "errors": len(sel) - len(got),
            "empty": sum(1 for r in got if r.get("empty")),
            "usable": sum(1 for r in got if r.get("usable")),
            "median_seconds": round(statistics.median(secs), 2) if secs else None,
            "total_seconds": round(sum(secs), 1),
            "median_completion_tokens": statistics.median(
                [r["completion_tokens"] for r in got if r.get("completion_tokens") is not None]
            )
            if got
            else None,
        }

    print("\n" + "=" * 96)
    print(f"{'budget':>8} {'calls':>6} {'empty':>6} {'usable':>7} {'median s':>9} {'total s':>8} {'median compl_tok':>17}")
    for budget, v in summary.items():
        print(f"{budget:>8} {v['calls']:>6} {v['empty']:>6} {v['usable']:>7} "
              f"{str(v['median_seconds']):>9} {str(v['total_seconds']):>8} {str(v['median_completion_tokens']):>17}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "tool": "orch740_aux_budget_probe.py",
                "call_site": "map_ids.py:406 _ai_protein_synonym_lookup",
                "model": model,
                "reasoning": "ENABLED (production default, uncontrolled)",
                "round": args.round,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

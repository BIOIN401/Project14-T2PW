"""ORCH-743 IT2-1 diagnosis: replay pilot-002's _0002 (or _0003) rung-1 extraction input.

Usage (cwd C:/t/mining, PYTHONPATH=C:/t/mining/src, through bounded_run):
    python diag_it2.py dry  <outdir> <call_suffix>
    python diag_it2.py live <outdir> <call_suffix> <tag> <prompt:v110|v120> <max_tokens> [reasoning_json]
    python diag_it2.py mining <outdir> <call_suffix> <tag> <start> <length> <max_tokens>  # extract_claims on one passage batch

'live' sends exactly ONE provider request, built with the same kwargs chat_detailed
sends (model, messages, temperature, max_tokens, json_schema response_format, timeout)
through t2pw.llm.client's own configured OpenAI client, with SDK retries off. The only
optional addition is extra_body={"reasoning": ...}: the knob MiningModelConfig.reasoning
names but never sends. It reads usage.completion_tokens_details.reasoning_tokens, which
chat_detailed does not expose. 'mining' is one request through MiningModelClient
(max_attempts=1, LLM_MAX_RETRIES=1).
"""
from t2pw.config import ensure_dotenv_loaded

ensure_dotenv_loaded()

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

RUN = Path("C:/t/mining-pilot-runs2/pilot-002-jasmonate-ath")
PRICE_IN, PRICE_OUT = 0.10, 0.30  # the mining config's per-Mtok estimates (costs.json)

V120 = """You extract biochemical reactions from numbered passages of ONE scientific paper.

Rules:
1. Cite every field you fill in THAT field's own list: enzyme_evidence_ids,
   gene_evidence_ids, species_evidence_ids, compartment_evidence_ids,
   direction_evidence_ids, reversible_evidence_ids, and each substrate / product
   in its own evidence_ids. The claim-level evidence_ids never counts for another
   field. A filled field whose own list is missing or empty is discarded.
2. Fill only what the passages support; omit the rest. An empty field is correct.
3. Use the paper's own names. No database identifiers.
4. species: the organism the passages say the reaction occurs in; omit it if
   unstated. species_inferred=true if it rests on analogy or orthology.
5. hypothesis=true if the paper proposes or models the reaction.
6. One reaction per claim; split a multi-step sentence into steps.
7. Answer directly. Do not re-read every passage for every field.

Return JSON only:
{"claims": [{"substrates": [{"name": "...", "evidence_ids": ["ev_..."]}],
 "products": [{"name": "...", "evidence_ids": ["ev_..."]}],
 "enzyme": "...", "enzyme_evidence_ids": ["ev_..."],
 "species": "...", "species_evidence_ids": ["ev_..."],
 "evidence_ids": ["ev_..."]}]}
The other fields (gene, direction, compartment, reversible, species_inferred,
hypothesis) follow the same pattern. No reaction supported: {"claims": []}.
"""


def rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load(suffix):
    from t2pw.mining import extraction as ex
    from t2pw.mining.models import EvidenceSpan

    call = next(r for r in rows(RUN / "model_calls.jsonl") if r["call_id"].endswith(suffix))
    rnd = int(call["round_index"])
    by_id = {r["evidence_id"]: r for r in rows(RUN / f"rounds/round_{rnd:03d}/passages_retrieved.jsonl")}
    spans = [EvidenceSpan.from_dict(by_id[e]) for e in call["evidence_ids"]]
    paper = next(r for r in rows(RUN / "papers.jsonl") if r["paper_id"] == call["paper_id"])
    cfg = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    context = [f"TARGET SPECIES (declared by the run): {cfg['target_species']}",
               f"TOPIC OF INTEREST: {cfg['topic']}",
               "Do not assume the passages are about the target species. Report the species the "
               "passages actually state; leave it empty if they do not state one."]
    user = "\n".join(context) + "\n\n" + ex.render_passages(spans, paper_title=paper["title"])
    return ex, call, spans, paper, cfg, user


def main():
    mode, out, suffix = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    from t2pw.llm.client import _hash

    ex, call, spans, paper, cfg, user = load(suffix)
    # The record's request_hash is the LAST rung's (35%, tight schema): check that one.
    rung3 = user[: max(200, int(len(user) * 0.35))]
    info = {
        "call_id": call["call_id"], "paper_id": call["paper_id"], "spans": len(spans),
        "user_chars": len(user), "system_chars_v110": len(ex.SYSTEM_PROMPT), "system_chars_v120": len(V120),
        "recorded_request_hash": call["request_hash"],
        "rebuilt_rung3_hash": _hash([{"role": "system", "content": ex.SYSTEM_PROMPT},
                                     {"role": "user", "content": rung3}]),
    }
    info["identical_input"] = info["recorded_request_hash"] == info["rebuilt_rung3_hash"]
    print(json.dumps(info, indent=1))
    if mode == "dry":
        (out / f"dry_{suffix}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        return
    assert info["identical_input"], "rebuilt input differs from the pilot's; refusing to spend"

    tag = sys.argv[4]
    if mode == "mining":
        os.environ["LLM_MAX_RETRIES"] = "1"
        from t2pw.mining.budget import BudgetLedger
        from t2pw.mining.config import MiningBudgets, MiningModelConfig
        from t2pw.mining.modelclient import MiningModelClient

        start, length, max_tokens = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        batch = spans[start:start + length]
        model_cfg = MiningModelConfig.from_env(dict(cfg["model"], max_attempts=1,
                                                    request_timeout_seconds=450,
                                                    max_tokens=max_tokens, reasoning=None))
        client = MiningModelClient(model_cfg, BudgetLedger(MiningBudgets()))
        started = time.monotonic()
        info["batch"] = {"start": start, "length": len(batch), "evidence_ids": [s.evidence_id for s in batch]}
        result = ex.extract_claims(
            client=client, spans=batch, paper_id=call["paper_id"], paper_title=paper["title"],
            target_species=cfg["target_species"], target_species_normalized=cfg["target_species_normalized"],
            round_index=1, topic=cfg["topic"], paper_observed_organisms=paper.get("observed_organisms") or [],
        )
        rec = result.outcome.record if result.outcome else None
        summary = dict(info, tag=tag, route="mining extract_claims", prompt_version=model_cfg.prompt_version,
                       max_tokens=model_cfg.max_tokens, status=result.status, claims=len(result.claims),
                       dropped=result.dropped_fields, wall_s=round(time.monotonic() - started, 1),
                       record=rec.to_dict() if rec else None,
                       est_cost_usd=round(((rec.input_tokens if rec else 0) * PRICE_IN
                                           + (rec.output_tokens if rec else 0) * PRICE_OUT) / 1e6, 6),
                       attributions=[c.field_attribution for c in result.claims])
        raw = result.outcome.raw_attempts if result.outcome else []
        (out / f"{tag}_raw.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")
        (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: summary[k] for k in ("status", "claims", "max_tokens", "est_cost_usd", "wall_s")}),
              "dropped:", len(result.dropped_fields))
        return

    prompt, max_tokens = sys.argv[5], int(sys.argv[6])
    reasoning = json.loads(sys.argv[7]) if len(sys.argv) > 7 else None
    system = ex.SYSTEM_PROMPT if prompt == "v110" else V120
    batch_arg = os.environ.get("DIAG_BATCH", "")
    if batch_arg:
        # One passage batch, rendered exactly as extract_claims renders it.
        start, length = (int(x) for x in batch_arg.split(":"))
        spans = spans[start:start + length]
        context = [f"TARGET SPECIES (declared by the run): {cfg['target_species']}",
                   f"TOPIC OF INTEREST: {cfg['topic']}",
                   "Do not assume the passages are about the target species. Report the species the "
                   "passages actually state; leave it empty if they do not state one."]
        user = "\n".join(context) + "\n\n" + ex.render_passages(spans, paper_title=paper["title"])
        info["batch"] = {"start": start, "length": len(spans), "user_chars": len(user),
                         "evidence_ids": [s.evidence_id for s in spans]}
    from t2pw.llm.client import _client

    kwargs = {
        "model": cfg["model"]["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "timeout": 450,
        "response_format": {"type": "json_schema", "json_schema": ex.CLAIM_SCHEMA},
    }
    if reasoning is not None:
        kwargs["extra_body"] = {"reasoning": reasoning}
    started = time.monotonic()
    resp = _client.with_options(max_retries=0).chat.completions.create(**kwargs)
    wall = round(time.monotonic() - started, 1)
    choice = resp.choices[0]
    content = choice.message.content or ""
    extra = getattr(choice.message, "model_extra", None) or {}
    reasoning_text = str(extra.get("reasoning") or "")
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None
    claims, dropped = [], []
    parsed = None
    try:
        parsed = json.loads(content) if content.strip() else None
    except Exception as exc:  # noqa: BLE001
        parsed = None
        dropped.append(f"parse error: {exc}")
    if isinstance(parsed, dict):
        claims, dropped = ex.build_claims(parsed, spans=spans, paper_id=call["paper_id"],
                                          target_species_normalized=cfg["target_species_normalized"],
                                          round_index=1, model="diag")
    summary = dict(
        info, tag=tag, route="direct, chat_detailed kwargs", prompt=prompt, max_tokens=max_tokens,
        reasoning_param=reasoning, finish_reason=choice.finish_reason, content_chars=len(content),
        reasoning_chars=len(reasoning_text), prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens, reasoning_tokens=reasoning_tokens,
        est_cost_usd=round((usage.prompt_tokens * PRICE_IN + usage.completion_tokens * PRICE_OUT) / 1e6, 6),
        wall_s=wall, claims=len(claims), dropped=dropped,
        attributions=[c.field_attribution for c in claims],
        provider=str(getattr(resp, "provider", "") or (getattr(resp, "model_extra", {}) or {}).get("provider", "")),
    )
    (out / f"{tag}_raw.json").write_text(json.dumps({"content": content, "reasoning": reasoning_text},
                                                    indent=1, ensure_ascii=False), encoding="utf-8")
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("finish_reason", "content_chars", "completion_tokens",
                                              "reasoning_tokens", "claims", "est_cost_usd", "wall_s")}),
          "dropped:", len(dropped))


if __name__ == "__main__":
    main()

"""ORCH-743 item 2 diagnosis: replay pilot-001's round-1 extraction input.

Usage (from cwd C:/t/mining, PYTHONPATH=C:/t/mining/src, through bounded_run):
    python replay_extraction.py dry   <outdir>        # no model call; checks the request hash
    python replay_extraction.py live  <outdir> <tag> [max_tokens]  # ONE model call through extract_claims

The input is rebuilt from the READ-ONLY pilot record (P1): the same 20 spans in the
order the pilot sent them (model_calls.jsonl evidence_ids), the same paper title,
species and topic. 'dry' proves the rebuilt messages hash to the pilot's
request_hash before any money is spent.
"""
from t2pw.config import ensure_dotenv_loaded

ensure_dotenv_loaded()

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

P1 = Path("C:/t/mining-pilot/docs/pathway_mining/pilot/001/run")


def rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def main():
    mode, out = sys.argv[1], Path(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "dry"
    out.mkdir(parents=True, exist_ok=True)

    from t2pw.llm.client import _hash
    from t2pw.mining import extraction as ex
    from t2pw.mining.budget import BudgetLedger
    from t2pw.mining.config import MiningBudgets, MiningModelConfig
    from t2pw.mining.modelclient import MiningModelClient
    from t2pw.mining.models import EvidenceSpan

    call = rows(P1 / "model_calls.jsonl")[0]
    by_id = {r["evidence_id"]: r for r in rows(P1 / "rounds/round_001/passages_retrieved.jsonl")}
    spans = [EvidenceSpan.from_dict(by_id[e]) for e in call["evidence_ids"]]
    paper = next(r for r in rows(P1 / "papers.jsonl") if r["paper_id"] == call["paper_id"])
    cfg = json.loads((P1 / "config.json").read_text(encoding="utf-8"))

    # Rebuild the user message exactly as extract_claims does, to check the hash.
    context = [f"TARGET SPECIES (declared by the run): {cfg['target_species']}"]
    context.append(f"TOPIC OF INTEREST: {cfg['topic']}")
    context.append(
        "Do not assume the passages are about the target species. Report the species the "
        "passages actually state; leave it empty if they do not state one."
    )
    user = "\n".join(context) + "\n\n" + ex.render_passages(spans, paper_title=paper["title"])
    messages = [{"role": "system", "content": ex.SYSTEM_PROMPT}, {"role": "user", "content": user}]
    rebuilt = _hash(messages)
    info = {
        "tag": tag,
        "pilot_request_hash": call["request_hash"],
        "rebuilt_request_hash": rebuilt,
        "identical_input": rebuilt == call["request_hash"],
        "prompt_version": MiningModelConfig().prompt_version,
        "spans": len(spans),
        "paper_id": call["paper_id"],
    }
    print(json.dumps(info, indent=1))
    if mode != "live":
        (out / "dry.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        return

    # ONE request: no shrink ladder, no in-client retry loop.
    os.environ["LLM_MAX_RETRIES"] = "1"
    over = dict(cfg["model"], max_attempts=1, request_timeout_seconds=300)
    if len(sys.argv) > 4:
        over["max_tokens"] = int(sys.argv[4])  # output budget only; the prompt is unchanged
    model_cfg = MiningModelConfig.from_env(over)
    info["max_tokens"] = model_cfg.max_tokens
    client = MiningModelClient(model_cfg, BudgetLedger(MiningBudgets()))
    result = ex.extract_claims(
        client=client,
        spans=spans,
        paper_id=call["paper_id"],
        paper_title=paper["title"],
        target_species=cfg["target_species"],
        target_species_normalized=cfg["target_species_normalized"],
        round_index=1,
        topic=cfg["topic"],
        paper_observed_organisms=paper.get("observed_organisms") or [],
    )
    raw = result.outcome.raw_attempts if result.outcome else []
    (out / f"{tag}_raw.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")
    summary = dict(
        info,
        status=result.status,
        claims=len(result.claims),
        dropped=result.dropped_fields,
        record=result.outcome.record.to_dict() if result.outcome else None,
        claims_detail=[
            {
                "subject": c.subject,
                "object": c.object,
                "enzyme": c.enzyme,
                "gene": c.gene,
                "species_text": c.species_text,
                "species_state": c.species_state,
                "field_evidence": c.field_evidence,
            }
            for c in result.claims
        ],
    )
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "claims")}), "dropped:", len(result.dropped_fields))


if __name__ == "__main__":
    main()

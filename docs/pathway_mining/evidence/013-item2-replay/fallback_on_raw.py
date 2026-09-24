"""Offline: run build_claims over a persisted raw completion from the replay.

Usage: python fallback_on_raw.py <raw.json> <out.json>
A truncated completion is cut back to its last complete claim object (a
diagnostic reading only; the run itself never admits a truncated payload).
No model call, no network.
"""
import json
import sys
from pathlib import Path

from t2pw.mining.extraction import build_claims
from t2pw.mining.models import EvidenceSpan

P1 = Path("C:/t/mining-pilot/docs/pathway_mining/pilot/001/run")


def rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))[0]["raw_text"]
try:
    payload, cut = json.loads(raw), False
except ValueError:
    decoder, text = json.JSONDecoder(), raw
    start = text.index("[", text.index('"claims"')) + 1
    claims_ok = []
    while True:
        pos = text.find("{", start)
        if pos < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except ValueError:
            break
        claims_ok.append(obj)
        start = end
    payload, cut = {"claims": claims_ok}, True

call = rows(P1 / "model_calls.jsonl")[0]
by_id = {r["evidence_id"]: r for r in rows(P1 / "rounds/round_001/passages_retrieved.jsonl")}
spans = [EvidenceSpan.from_dict(by_id[e]) for e in call["evidence_ids"]]
claims, dropped = build_claims(
    payload, spans=spans, paper_id=call["paper_id"],
    target_species_normalized="arabidopsis thaliana", round_index=1, model="replay",
)
asserted = sum(1 for c in payload["claims"] for k in ("enzyme", "gene", "species") if str(c.get(k) or "").strip())
cited_keys = sum(1 for c in payload["claims"] for k in c if k.endswith("_evidence_ids"))
out = {
    "source": sys.argv[1],
    "truncated_and_cut_to_complete_claims": cut,
    "claims_parsed": len(payload["claims"]),
    "enzyme_gene_species_values_asserted": asserted,
    "per_field_evidence_id_keys_emitted": cited_keys,
    "kept_by_attribution": {
        how: sum(1 for c in claims for v in c.field_attribution.values() if v == how)
        for how in ("model_citation", "verbatim_cited_span")
    },
    "dropped": dropped,
    "claims": [
        {"subject": c.subject, "object": c.object, "enzyme": c.enzyme, "gene": c.gene,
         "species_text": c.species_text, "species_state": c.species_state,
         "field_evidence": c.field_evidence, "field_attribution": c.field_attribution}
        for c in claims
    ],
}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: out[k] for k in list(out)[:6]}, indent=1), "dropped:", len(dropped))

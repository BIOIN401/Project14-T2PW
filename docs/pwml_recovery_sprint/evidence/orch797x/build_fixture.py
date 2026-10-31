"""ORCH-797b: trim the recorded UniProt answers (orch797b_uniprot_responses.json) to the test fixture.

usage: python build_fixture.py <recording.json> <tests/fixtures/orch797b/uniprot.json>
Entries: the UniProt search-result JSON of each wanted accession, verbatim, from any recorded search answer.
TSV: the recorded family / gene_exact search answers (human, 9606) for the D-120 § 3 terms, verbatim.
"""
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

rec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
WANT = ["P05412", "P35561", "Q9UHQ1", "Q8WVD3", "O43318", "P49116", "Q8GZM7", "P93831"]
entries, tsv = {}, {}
for key in sorted(rec):
    hit = rec[key]
    body = hit.get("json") or {}
    for item in body.get("results") or [] if isinstance(body, dict) else []:
        acc = item.get("primaryAccession")
        if acc in WANT and acc not in entries:
            entries[acc] = item
    text = hit.get("text")
    query = (parse_qs(urlsplit(key).query).get("query") or [""])[0]
    if text and "organism_id:9606" in query and ("family:" in query or "gene_exact:" in query):
        tsv[query] = text
missing = [a for a in WANT if a not in entries]
print("entries:", sorted(entries), "missing:", missing)
print("tsv queries:", len(tsv))
for k in tsv:
    print(" ", k[:160], "|", tsv[k].count("\n") - 1, "rows")
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"entries": entries, "tsv": tsv}, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")

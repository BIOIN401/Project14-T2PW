"""ORCH-793: one-off read-only UniProt REST GETs, recorded so every test and replay runs offline.

usage: python record_uniprot.py <out.json>
Records, per accession, UniProt's own gene names / gene synonyms / protein names / organism for the
C4 rows (ORCH-789 § 2): PIF1, TAK1, c-Jun, Fur, Kir2.1, plus the rivals the legs' ladders saw.
"""
import json
import sys
import urllib.request

ACCESSIONS = ["Q8GZM7", "P93831", "O43318", "P49116", "P05412", "P0A9A9", "P0A9B0", "P0A9B1", "P35561"]
FIELDS = "accession,id,reviewed,protein_name,gene_names,organism_name,organism_id"
out = {}
for acc in ACCESSIONS:
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json?fields={FIELDS}"
    body = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:  # noqa: BLE001 -- recorded, not hidden
            print(acc, "attempt", attempt, "failed:", exc)
    if body is None:
        out[acc] = {"url": url, "error": "fetch_failed"}
        continue
    out[acc] = {"url": url, "json": body}
    genes = [(g.get("geneName") or {}).get("value") for g in body.get("genes") or []]
    syns = [s.get("value") for g in body.get("genes") or [] for s in g.get("synonyms") or []]
    desc = body.get("proteinDescription") or {}
    names = [(desc.get("recommendedName") or {}).get("fullName", {}).get("value")]
    names += [a.get("fullName", {}).get("value") for a in desc.get("alternativeNames") or []]
    shorts = [s.get("value") for n in [desc.get("recommendedName") or {}] + list(desc.get("alternativeNames") or [])
              for s in n.get("shortNames") or []]
    print(acc, body.get("entryType"), (body.get("organism") or {}).get("scientificName"),
          "| genes:", genes, "| synonyms:", syns, "| names:", names, "| short:", shorts)
with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as fh:
    json.dump(out, fh, indent=1, sort_keys=True, ensure_ascii=False)

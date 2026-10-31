"""ORCH-797b audit: every accession the tip ships that base did not (pattern: ORCH-793 23-new-ids-audit).

usage: python orch797b_new_ids.py <base replay log> <tip replay log> <recording.json>
Reads the ``shipped:`` line per leg of each orch797b_replay.py log, and describes each new accession from the
recorded UniProt answers (recommended name, genes, organism, reviewed). Offline.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def shipped(log):
    out, leg = {}, None
    for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"######## (\S+)", line)
        if m:
            leg = m.group(1)
        elif line.startswith("  shipped: ") and leg:
            out[leg] = json.loads(line[len("  shipped: "):])
    return out


def accs(value):
    return set(re.findall(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9](?:[A-Z][A-Z0-9]{2}[0-9])?)\b", value or ""))


rec = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
entries = {}
for hit in rec.values():
    body = hit.get("json")
    for item in (body or {}).get("results") or [] if isinstance(body, dict) else []:
        entries.setdefault(item.get("primaryAccession"), item)
    for line in (hit.get("text") or "").splitlines()[1:]:
        cols = line.split("\t")
        entries.setdefault(cols[0], {"tsv": cols})


def describe(acc):
    item = entries.get(acc) or {}
    if "tsv" in item:
        return f"(family/gene search) gene={item['tsv'][1]} taxon={item['tsv'][2]} families={item['tsv'][3] if len(item['tsv']) > 3 else ''}"
    desc = item.get("proteinDescription") or {}
    rec_name = ((desc.get("recommendedName") or {}).get("fullName") or {}).get("value")
    alts = [((a.get("fullName") or {}).get("value")) for a in desc.get("alternativeNames") or []]
    genes = [(g.get("geneName") or {}).get("value") for g in item.get("genes") or []]
    syn = [s.get("value") for g in item.get("genes") or [] for s in g.get("synonyms") or []]
    return (f"{rec_name!r} genes={genes} syn={syn} alt={alts[:4]} org={(item.get('organism') or {}).get('scientificName')} "
            f"{item.get('entryType', '')}") if item else "(not in the recording)"


base, tip = shipped(sys.argv[1]), shipped(sys.argv[2])
total = 0
for leg in tip:
    b = {k: accs(v) for k, v in (base.get(leg) or {}).items()}
    base_all = set().union(*b.values()) if b else set()
    print(f"\n## {leg}")
    for name, value in sorted(tip[leg].items()):
        new = sorted(accs(value) - b.get(name, set()))
        if not new:
            continue
        for acc in new:
            total += 1
            where = "also shipped elsewhere at base" if acc in base_all else "NEW to the leg"
            print(f"- {name} -> {acc} ({where}): {describe(acc)}")
    for name in sorted(set(b) - set(tip[leg])):
        if b[name]:
            print(f"- base-only row {name}: {sorted(b[name])}")
print(f"\nnew (row, accession) pairs at tip: {total}")

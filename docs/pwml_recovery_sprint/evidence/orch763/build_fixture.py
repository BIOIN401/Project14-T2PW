"""ORCH-763: rebuild the Stage-1 draws of an ORCH-759 leg from what the leg kept.

The per-draw payloads of a paper run are not kept anywhere (ORCH-758 section 8: only response
hashes are; ``tmp/stage1_raw_extraction.json`` is the POOLED payload and is overwritten by the
next leg). What survives is the pooled payload (``stage1_payload.json``) and the side file
(``stage1_ensemble_report.json``), which names, for every pooled row, the draw that wrote it
and every row merged into it (draw, name, its own quote, its entry quotes). This script puts
the two together. Offline, no model, no network.

Usage (through bounded_run.py):
    PYTHONPATH=<tree>/src python build_fixture.py <leg strict dir> <side file json> <out json>
        [--state-field "<state name>|<field>|<value>|<why>"]...

Reconstruction rules -- every one is recorded in the output under "reconstruction":
1. A pooled reaction row is VERBATIM from ``stage1_payload.json`` (minus the post-Stage-1 keys
   ``provenance_lineage``, ``locked_reaction_id``, ``scope_membership``), assigned to the draw
   that created it. Pooled rows and side-file records are matched per reaction key, k-th to
   k-th (both lists are in creation order), and the names must agree.
2. A row merged INTO a pooled row is rebuilt: its own name and quote from the side file; its
   participants and enzyme names from the survivor (the merge bridged their keys, so they
   are one reaction under the rules that ran); each entry quote at the pointer the side file
   gives (``/modifiers/0/evidence`` -> ``modifiers[0]``); a cofactor the side file lists as
   withheld from that row's quote is put back on its side.
3. A merged row's ``biological_state`` is NOT recorded anywhere. It is INFERRED, and listed:
   the survivor's label when the merged row's draw defines that label, else the label most
   of that draw's own pooled rows carry.
4. Each draw's ``biological_states`` are the side file's state rows of that draw, with the
   content of the pooled payload's first row of that name.
5. Names the side file lists as rewritten (``names_rewritten``: draw, from, to) are written
   back to the draw's own spelling in that draw's reaction participants.
6. ``--state-field`` patches a field the pooled payload lost (it carries no
   ``compartment_canonical``), each with its reason.
"""
import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from t2pw.pipeline.name_keys import strict_key
from t2pw.pipeline.participant_schema import PARTICIPANT_NAME_KEYS
from t2pw.pipeline.stage_one_ensemble import _label, row_key

R = "/processes/reactions"
POST_STAGE1 = ("provenance_lineage", "locked_reaction_id", "scope_membership")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def strip(row):
    return {k: v for k, v in row.items() if k not in POST_STAGE1}


def rename(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [rename(v, mapping) for v in value]
    if isinstance(value, dict):
        field = next((k for k in PARTICIPANT_NAME_KEYS if isinstance(value.get(k), str)), None)
        return {**value, field: mapping.get(value[field], value[field])} if field else value
    return value


ap = argparse.ArgumentParser()
ap.add_argument("leg")
ap.add_argument("side")
ap.add_argument("out")
ap.add_argument("--state-field", action="append", default=[])
a = ap.parse_args()
leg, side_path, out = Path(a.leg), Path(a.side), Path(a.out)
payload = json.loads((leg / "stage1_payload.json").read_text(encoding="utf-8"))
side = next(b for b in json.loads(side_path.read_text(encoding="utf-8")) if (b.get("buckets") or {}).get(R, {}).get("rows"))
n = side["draws_requested"]
records = side["buckets"][R]["rows"]
inferred, patches = [], []

# rule 1: match pooled rows to records, k-th to k-th per key
by_label = {}
for row in payload["processes"]["reactions"]:
    by_label.setdefault(_label(row_key(R, row)), []).append(strip(row))
taken = Counter()
pairs = []
for rec in records:
    row = by_label[rec["key"]][taken[rec["key"]]]
    taken[rec["key"]] += 1
    assert row["name"] == rec["name"], (row["name"], rec)
    assert not any(m.get("became_survivor") for m in rec.get("merges") or []), "survivor swaps are not rebuilt"
    pairs.append((rec, row))
assert sum(taken.values()) == len(payload["processes"]["reactions"]), "pooled rows left unmatched"

# rule 4: states per draw
first = {}
for s in payload.get("biological_states") or []:
    first.setdefault(strict_key(s.get("name")), s)
states = {d: [] for d in range(1, n + 1)}
for srow in side["buckets"].get("/biological_states", {}).get("rows") or []:
    content = {k: v for k, v in first[strict_key(srow["name"])].items() if k != "provenance_lineage"}
    for d in srow["draws"]:
        states[d].append(deepcopy(content))
for spec in a.state_field:
    name, field, value, why = spec.split("|")
    for d, rows in states.items():
        for s in rows:
            if s["name"] == name:
                s[field] = value
                patches.append({"draw": d, "state": name, "field": field, "value": value, "why": why})
defined = {d: {strict_key(s["name"]) for s in rows} for d, rows in states.items()}

# rules 2-3: the draws' reaction lists, in pooled order
own = {d: Counter(row.get("biological_state") for rec, row in pairs if rec["draws"][0] == d and row.get("biological_state"))
       for d in range(1, n + 1)}
draws = {d: [] for d in range(1, n + 1)}
for rec, row in pairs:
    draws[rec["draws"][0]].append(deepcopy(row))
    for m in rec.get("merges") or []:
        d = m["draw"]
        rebuilt = {k: deepcopy(row[k]) for k in ("inputs", "outputs") if k in row}
        rebuilt["name"], rebuilt["evidence"] = m["name"], m.get("evidence", "")
        enzymes = [e for slot in ("enzymes", "modifiers") for e in row.get(slot) or []]
        quotes = {}
        for q in m.get("entry_evidence") or []:
            parts = q["at"].split("/")
            if len(parts) == 4 and parts[1] in ("enzymes", "modifiers"):
                quotes[(parts[1], int(parts[2]))] = q["evidence"]
        slot = next(iter(quotes), ("enzymes" if row.get("enzymes") else "modifiers", 0))[0]
        for i, src in enumerate(enzymes):
            entry = {k: v for k, v in src.items() if k != "evidence"} if isinstance(src, dict) else {"protein": src}
            if (slot, i) in quotes:
                entry["evidence"] = quotes[(slot, i)]
            rebuilt.setdefault(slot, []).append(entry)
        for w in rec.get("cofactors_withheld") or []:
            if w.get("donor_evidence") == m.get("evidence"):
                rebuilt.setdefault(w["side"], []).append(w["cofactor"])
        label = row.get("biological_state")
        if label and strict_key(label) in defined[d]:
            why = "the survivor's label, defined in this draw"
        else:
            label = own[d].most_common(1)[0][0] if own[d] else None
            why = "the label most of this draw's own pooled rows carry"
        if label:
            rebuilt["biological_state"] = label
        inferred.append({"draw": d, "row": m["name"], "merged_into": rec["name"], "biological_state": label, "why": why})
        draws[d].append(rebuilt)

# rule 5: each draw's own spellings back
for r in side.get("names_rewritten") or []:
    for row in draws[r["draw"]]:
        for slot in ("inputs", "outputs"):
            if slot in row:
                row[slot] = rename(row[slot], {r["to"]: r["from"]})

result = {
    "source": {"stage1_payload": str(leg / "stage1_payload.json"), "stage1_payload_sha256": sha(leg / "stage1_payload.json"),
               "side_file": str(side_path), "side_file_sha256": sha(side_path),
               "side_file_paper_chars": side["paper_lexicon"].get("paper_chars"),
               "pooled_reactions": len(records)},
    "reconstruction": {"rules": (__doc__ or "").split("Reconstruction rules")[1].strip(),
                       "inferred_states": inferred, "state_field_patches": patches},
    "draws": [{"biological_states": states[d], "processes": {"reactions": draws[d]}} for d in range(1, n + 1)],
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out}: draws {[len(draws[d]) for d in range(1, n + 1)]} reactions (side file rows_per_draw "
      f"{side['buckets'][R]['rows_per_draw']}); {len(inferred)} inferred states; {len(patches)} patches")
for i in inferred:
    print(f"   inferred d{i['draw']} {i['row']!r} -> {i['merged_into']!r}: state {i['biological_state']!r} ({i['why']})")
for p in patches:
    print(f"   patch d{p['draw']} {p['state']!r}.{p['field']} = {p['value']!r} ({p['why']})")

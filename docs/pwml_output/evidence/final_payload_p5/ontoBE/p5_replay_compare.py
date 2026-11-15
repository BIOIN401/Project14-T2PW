"""Compare the flag-ON replay with the ontoBE code (label on) against the SAME replay on the commit just before it (label base: git worktree
at 010d2f39, i.e. with the P5 writer wildcard guard and nucleic work but without ontoBE) for 14 selected stored runs: the location, cell-type
and tissue rows each PWML writes. (The P4 replay is not used: it predates the P5 writer guard.) Output p5_replay_compare.md.
Inputs: C:/t/p5scratch/ontoBE/replay/on and /base (p5_replay.py output)."""
import json
from pathlib import Path

from lxml import etree

HERE = Path(__file__).parent
P5 = Path("C:/t/p5scratch/ontoBE/replay/on")
P4 = Path("C:/t/p5scratch/ontoBE/replay/base")
rec5 = {r["idx"]: r for r in map(json.loads, open(P5 / "results_0.jsonl", encoding="utf-8"))}
rec4 = {}
for f in P4.glob("results_*.jsonl"):
    for ln in open(f, encoding="utf-8"):
        r = json.loads(ln)
        rec4[r["idx"]] = r


def rows(path):
    if not path.exists():
        return None
    root = etree.parse(str(path)).getroot()
    out = {}
    for tag, p in (("location", ".//subcellular-locations/subcellular-location"), ("cell-type", ".//cell-types/cell-type"), ("tissue", ".//tissues/tissue")):
        out[tag] = sorted((n.findtext("name") or "", n.findtext("ontology-id") or "") for n in root.findall(p))
    return out


md = ["# ontoBE replay (flag ON, 14 stored runs) against the same replay on the parent commit 010d2f39", "",
      "`p5_replay.py` (adapted copy of the P4 replay): stored `final_mapped.json`, stage-0 context and `01_source_text.txt` (when the run folder has one) through `run_pwml_export`. "
      "Only the vocabulary rows each PWML writes are compared (name, ontology id); the rest of the PWML is not diffed here.", ""]
for idx in sorted(rec5):
    r5, r4 = rec5[idx], rec4.get(idx, {})
    a, b = rows(P4 / "pwml" / f"{idx}.pwml"), rows(P5 / "pwml" / f"{idx}.pwml")
    md.append(f"## run {idx} `{r5['dir'][-70:].replace(chr(92), '/')}`")
    md.append(f"- ontoBE: ok={r5['ok']} pwml_written={r5['pwml_written']} paper_text_found={r5.get('paper_text_found')} stage={r5.get('stage')!r}; base: ok={r4.get('ok')} pwml_written={r4.get('pwml_written')}")
    ldc = {k: v for k, v in (r5.get("layout_debug_counts") or {}).items() if k.startswith("vocabulary_") or "dropped_no_id" in k or "kept" in k}
    md.append(f"- P5 vocabulary/drop counters: {ldc}")
    ex = ((r5.get("ontology_report") or {}).get("extra")) or {}
    md.append(f"- P5 ontology extra: { {k: v for k, v in ex.items() if k != 'membrane_rows_by_organism' and v} }")
    for ent in r5.get("fill_entries") or []:
        if ent.get("field") in ("ontology_id", "cellosaurus_id"):
            extra = {k: ent[k] for k in ("matched_form", "via", "context_phrase", "context_source", "context_count") if k in ent}
            md.append(f"  - fill entry: {ent['kind']} `{ent['entity']}` -> {ent['field']} {ent['accession']} ({ent['source']}, {ent['match_type']}) {extra}")
    if a is None or b is None:
        md.append(f"- PWML: base {'present' if a else 'absent'}, ontoBE {'present' if b else 'absent'}")
    else:
        for tag in ("location", "cell-type", "tissue"):
            only4 = [x for x in a[tag] if x not in b[tag]]
            only5 = [x for x in b[tag] if x not in a[tag]]
            if only4 or only5:
                md.append(f"- {tag}: only in base {only4}; only in ontoBE {only5}")
        if all(a[t] == b[t] for t in a):
            md.append("- PWML vocabulary rows identical to base")
    md.append("")
(HERE / "p5_replay_compare.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

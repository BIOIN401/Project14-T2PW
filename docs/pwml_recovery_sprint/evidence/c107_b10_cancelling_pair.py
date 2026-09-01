"""C-107 / REV-107 B10: what happened to the cancelling pair on the 140 KB span.

REV-105 recorded, on one very long discussion section, `mediat` matching inside
*"intermediate"* and `suppress` matching inside *"suppressor mutations"* -- two
false positives producing the right outcome for the wrong reason. C-107 1f
anchors `mediat`, which fixes one half of a cancelling pair, so the outcome can
move. This locates every corpus span that carries BOTH artefacts, quotes the
neighbourhood of each, and prints the verdict.

It also answers 1g's other half: which stage writes the oversized spans.

Usage::  <python> c107_b10_cancelling_pair.py <repo-root>
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))

from t2pw.curation.apply_audit_patch import (  # noqa: E402
    _match_fold, apply_patch_with_policy,
)

CONT = ("enzymes", "modifiers", "modifiers_or_enzymes", "catalysts",
        "transporters", "cargo", "cargo_complex")
EVK = ("evidence", "evidence_quote", "source_evidence", "source_text")


def nm_of(r):
    if isinstance(r, str):
        return r.strip()
    if not isinstance(r, dict):
        return ""
    for f in ("entity", "protein", "protein_name", "protein_complex", "enzyme",
              "modifier", "name"):
        v = r.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def ev_of(r):
    if not isinstance(r, dict):
        return ""
    for k in EVK:
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def seam(cont, row, bucket):
    n = nm_of(row)
    proc = {"name": "p", "inputs": ["A"], "outputs": ["B"], "evidence": "chem only", cont: []}
    pl = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                       "proteins": [{"name": n}], "protein_complexes": [],
                       "nucleic_acids": []},
          "processes": {bucket: [proc]}}
    op = {"op": "add", "path": f"/processes/{bucket}/0/{cont}/-", "value": row,
          "confidence": 1.0}
    _r, rep = apply_patch_with_policy(pl, [op], stage="probe")
    return rep["summary"]["accepted_count"] == 1, (rep["rejected"][0]["reason"] if rep["rejected"] else "")


rows = []
seen = set()
files = sorted(glob.glob(str(REPO / "runs/**/final_mapped.json"), recursive=True) +
               glob.glob(str(REPO / "runs_verify/**/final_mapped.json"), recursive=True))
for f in files:
    try:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
    except Exception:
        continue
    procs = d.get("processes") or {}
    for bucket in ("reactions", "transports", "reaction_coupled_transports"):
        for rxn in (procs.get(bucket) or []):
            if not isinstance(rxn, dict):
                continue
            for cont in CONT:
                lst = rxn.get(cont)
                if not isinstance(lst, list):
                    continue
                for row in lst:
                    n, ev = nm_of(row), ev_of(row)
                    if not n or not ev:
                        continue
                    key = f"{cont}|{bucket}|{n}|{row.get('role') if isinstance(row, dict) else ''}|{ev}"
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append((f, bucket, cont, n, ev, row))

print("=" * 78)
print("B10. SPANS CARRYING BOTH HALVES OF THE CANCELLING PAIR")
print("=" * 78)
INTERMEDIATE = re.compile(r"[a-z]mediat")          # "mediat" preceded by a letter
SUPPRESSOR = re.compile(r"suppressor")
hits = 0
for f, bucket, cont, n, ev, row in rows:
    folded = _match_fold(ev)
    if not (INTERMEDIATE.search(folded) and SUPPRESSOR.search(folded)):
        continue
    hits += 1
    ok, reason = seam(cont, row, bucket)
    print(f"\n  [{cont}/{bucket}] actor={n!r}  span={len(ev)} chars")
    print(f"      file    : {Path(f).parent.parent.name}/{Path(f).parent.name}")
    print(f"      verdict : {'ACCEPTED' if ok else 'REFUSED'}")
    print(f"      reason  : {reason[:180]}")
    for label, pat in (("mediat-in-word", INTERMEDIATE), ("suppressor", SUPPRESSOR)):
        m = pat.search(folded)
        if m:
            s = max(0, m.start() - 90)
            print(f"      {label:14s}: ...{folded[s:m.end() + 90]}...")
    # is the actor named anywhere near either artefact?
    from t2pw.curation.apply_audit_patch import _identifying_match_tokens
    toks = _identifying_match_tokens(n) or [_match_fold(n)]
    for t in toks:
        occ = [m.start() for m in re.finditer(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", folded)]
        print(f"      token {t!r}: {len(occ)} occurrence(s) in the folded span")
print(f"\n  spans carrying both artefacts: {hits}")

print()
print("=" * 78)
print("B10 (wider). EVERY span where 'mediat' occurs ONLY inside a longer word")
print("=" * 78)
only_inside = 0
flips = []
for f, bucket, cont, n, ev, row in rows:
    folded = _match_fold(ev)
    if not re.search(r"(?<![a-z])mediat", folded) and INTERMEDIATE.search(folded):
        only_inside += 1
        ok, reason = seam(cont, row, bucket)
        flips.append((ok, cont, bucket, n, len(ev), reason))
print(f"  rows whose span carries 'mediat' ONLY inside a longer word: {only_inside}")
acc = sum(1 for x in flips if x[0])
print(f"  of those, ACCEPTED at this SHA: {acc}   REFUSED: {len(flips) - acc}")
for ok, cont, bucket, n, ln, reason in flips[:15]:
    print(f"      {'ACCEPT' if ok else 'REFUSE'}  [{cont}/{bucket}] {n!r} ({ln} chars)")

print()
print("=" * 78)
print("1g. WHICH STAGE PRODUCES THE OVERSIZED SPANS  (registered, NOT fixed)")
print("=" * 78)
over = [(len(ev), f, n, cont) for f, bucket, cont, n, ev, row in rows if len(ev) > 5000]
over.sort(reverse=True)
print(f"  rows with a span > 5,000 chars : {len(over)} of {len(rows)}")
print(f"  longest                        : {over[0][0] if over else 0}")
paper_dirs = sorted({str(Path(f).parent) for _l, f, _n, _c in over})
for d in paper_dirs:
    print(f"\n  {Path(d).parent.name}/{Path(d).name}")
    sibs = sorted(p.name for p in Path(d).iterdir() if p.is_file())
    print(f"      sibling artifacts: {', '.join(sibs[:24])}")
    # Which upstream artifact already carries a span this long?
    for cand in ("draft_graph.json", "final.json", "final_mapped.json",
                 "extraction.json", "sections.json", "audit_report.json",
                 "final.audited.json", "curated.json"):
        p = Path(d) / cand
        if not p.exists():
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        longest = 0
        try:
            obj = json.loads(txt)
        except Exception:
            obj = None

        def walk(o):
            global longest
            if isinstance(o, str):
                longest = max(longest, len(o))
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        if obj is not None:
            walk(obj)
        print(f"      {cand:24s} longest string = {longest}")

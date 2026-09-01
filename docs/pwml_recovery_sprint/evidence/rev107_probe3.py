"""REV-107 probe 3: claim 5 (1g stage attribution), the T1 correction, and the
five near-synonyms that remain open at the tip, stated as production verdicts.

Usage:  <python> rev107_probe3.py <data-root> <code-root>
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

DATA = Path(sys.argv[1]).resolve()
CODE = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(CODE / "src"))

import t2pw.curation.apply_audit_patch as M  # noqa: E402
from t2pw.curation.apply_audit_patch import (  # noqa: E402
    apply_patch_with_policy, _span_licenses_actor,
)
print("code loaded from:", M.__file__)

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


def actor_spans(doc):
    """Every actor-row evidence span in a payload-shaped document."""
    out = []
    procs = doc.get("processes") or {}
    for bucket in ("reactions", "transports", "reaction_coupled_transports"):
        for rxn in (procs.get(bucket) or []):
            if not isinstance(rxn, dict):
                continue
            for cont in CONT:
                rows = rxn.get(cont)
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    ev = ev_of(row)
                    if ev:
                        out.append((len(ev), cont, nm_of(row)))
    return out


def rowcount(doc):
    n = 0
    procs = doc.get("processes") or {}
    for bucket in ("reactions", "transports", "reaction_coupled_transports"):
        for rxn in (procs.get(bucket) or []):
            if not isinstance(rxn, dict):
                continue
            for cont in CONT:
                rows = rxn.get(cont)
                if isinstance(rows, list):
                    n += len(rows)
    return n


print()
print("=" * 90)
print("CLAIM 5 -- 1g attribution: which stage produces the oversized spans?")
print("=" * 90)

# The papers that carry a >5000-char actor span, from the final_mapped corpus.
affected = {}
for f in sorted(glob.glob(str(DATA / "runs/**/final_mapped.json"), recursive=True)
                + glob.glob(str(DATA / "runs_verify/**/final_mapped.json"), recursive=True)):
    try:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
    except Exception:
        continue
    spans = actor_spans(d)
    big = [s for s in spans if s[0] > 5000]
    if big:
        affected[f] = max(s[0] for s in big)

print(f"  final_mapped.json files carrying an actor span > 5000 chars: {len(affected)}")
for f, mx in sorted(affected.items(), key=lambda kv: -kv[1]):
    print(f"    max={mx:>7}  {f[len(str(DATA)):]}")

print()
print("  For each affected paper, the same measurement on the UPSTREAM artifacts:")
STAGES = ["stage1_payload.json", "merged_payload.json", "final_mapped.json",
          "stage2_payload.json", "normalized_payload.json"]
for f in sorted(affected, key=lambda k: -affected[k]):
    d = Path(f).parent
    print(f"\n    {str(d)[len(str(DATA)):]}")
    for st in STAGES:
        p = d / st
        if not p.exists():
            print(f"      {st:<26} (absent)")
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"      {st:<26} (unreadable: {exc})")
            continue
        spans = actor_spans(doc)
        mx = max((s[0] for s in spans), default=0)
        over = sum(1 for s in spans if s[0] > 5000)
        print(f"      {st:<26} actor rows={rowcount(doc):>4}  spans={len(spans):>4}"
              f"  max_span={mx:>7}  over5000={over}")

print()
print("  batch/driver.py: who writes merged_payload.json?")
drv = (CODE / "src/t2pw/batch/driver.py")
if drv.exists():
    lines = drv.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, ln in enumerate(lines, 1):
        if "merged_payload" in ln:
            print(f"    driver.py:{i}: {ln.strip()[:120]}")

print()
print("=" * 90)
print("T1 CORRECTION -- the earlier multi-token name 'LpxC hydrolase' contains an")
print("     enzyme noun, so the NAME itself supplied the cue. Re-run the >80-char")
print("     case with a name carrying no enzyme noun.")
print("=" * 90)


def run(container, value, evidence, bucket="reactions"):
    nm = value if isinstance(value, str) else value.get("entity")
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "A is converted in the gut", container: []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": nm}], "protein_complexes": [],
                            "nucleic_acids": []},
               "processes": {bucket: [proc]}}
    op = {"op": "add", "path": f"/processes/{bucket}/0/{container}/-",
          "value": value, "confidence": 1.0}
    if evidence is not None:
        op["evidence"] = evidence
    _r, rep = apply_patch_with_policy(payload, [op], stage="rev107")
    return rep["summary"]["accepted_count"] == 1


for nm in ("LpxC hydrolase", "MenD complex", "Qrx7 factor"):
    ev = nm + " " + ("x" * 100) + " catalyses the conversion"
    print(f"  name={nm!r:<20} cue >80 chars away -> "
          f"{'ACCEPT' if run('enzymes', nm, ev) else 'REFUSE'}   (want REFUSE)")
    ev2 = nm + " was quantified in the lysate"
    print(f"  name={nm!r:<20} no cue at all      -> "
          f"{'ACCEPT' if run('enzymes', nm, ev2) else 'REFUSE'}   (want REFUSE)")

print()
print("=" * 90)
print("THE FIVE NEAR-SYNONYMS THAT REMAIN OPEN, as PRODUCTION verdicts through")
print("apply_patch_with_policy -- an F-146-shaped promotion of an inhibited")
print("protein to catalyst, at the TIP.")
print("=" * 90)
OPEN = ["disruption", "reduction", "loss", "depletion", "quenching"]
CLOSED = ["blockade", "impairment", "silencing", "sequestration", "ablation",
          "interference"]
for w in OPEN + CLOSED:
    ev = f"the {w} of NDM-1 is mediated by PSA"
    got = run("enzymes", "NDM-1", ev)
    print(f"  {w:<14} {'ACCEPTED  <-- F-146 BY PARAPHRASE' if got else 'refused'}"
          f"   evidence={ev!r}")

print()
print("  and with the paper's own object noun, but >40 chars from the stem:")
for w in OPEN:
    ev = (f"the {w} of NDM-1 " + "x" * 45 + " enzymatic activity is mediated by PSA")
    got = run("enzymes", "NDM-1", ev)
    print(f"  {w:<14} {'ACCEPTED  <-- 40-char window evaded' if got else 'refused'}")

print()
print("  and with the object BEFORE the stem (ordinary English word order):")
for w in OPEN:
    ev = f"NDM-1 activity showed {w} in the PSA-mediated assay"
    got = run("enzymes", "NDM-1", ev)
    print(f"  {w:<14} {'ACCEPTED  <-- word order evades' if got else 'refused'}")

print()
print("REV107_PROBE3_DONE")

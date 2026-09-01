"""REV-107 probe 2. Runs at BASE and TIP; PYTHONPATH selects the tree.

Covers: the one-character-name trap in the inherited 29-case battery, B3's clean
residual frames, N4 isolated with padding, REV-105 finding-2 (claim 3), and the
B10 cancelling pair located in the REAL corpus.

Usage:  <python> rev107_probe2.py <data-root> <code-root>
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

DATA = Path(sys.argv[1]).resolve()
CODE = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(CODE / "src"))

import t2pw.curation.apply_audit_patch as M  # noqa: E402
from t2pw.curation.apply_audit_patch import (  # noqa: E402
    apply_patch_with_policy, _span_licenses_actor, _match_fold,
)

print("code loaded from:", M.__file__)
print()


def run(container, value, evidence, bucket="reactions", conf=1.0, name=None):
    nm = name
    if nm is None:
        nm = value if isinstance(value, str) else (
            value.get("entity") or value.get("protein") or value.get("protein_complex") or "")
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "A is converted in the gut", container: []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": nm}], "protein_complexes": [],
                            "nucleic_acids": []},
               "processes": {bucket: [proc]}}
    op = {"op": "add", "path": f"/processes/{bucket}/0/{container}/-",
          "value": value, "confidence": conf}
    if evidence is not None:
        op["evidence"] = evidence
    _r, rep = apply_patch_with_policy(payload, [op], stage="rev107")
    return rep["summary"]["accepted_count"] == 1


print("=" * 90)
print("T1 -- the one-character-name trap: REV-105's battery uses 'P' for 20 of 29")
print("     cases. Re-run every 'P' case with a MULTI-TOKEN name that exercises")
print("     _identifying_match_tokens rather than the whole-name fallback.")
print("=" * 90)
NAME = "LpxC hydrolase"          # two identifying tokens, neither one character
CASES_P = [
    ("catalysis span",       "enzymes", "{N} catalyses the conversion of A to B under physiological conditions", True, "reactions"),
    ("N-catalyzed",          "enzymes", "{N}-catalyzed conversion of A to B is the rate-limiting step", True, "reactions"),
    ("N-mediated",           "enzymes", "{N}-mediated hydrolysis of A yields B", True, "reactions"),
    ("is the enzyme resp.",  "enzymes", "{N} is the enzyme responsible for the decomposition of A into B", True, "reactions"),
    ("passive by N",         "enzymes", "A is converted to B by {N} in the intestine", True, "reactions"),
    ("catalysed by N",       "enzymes", "The reaction is catalysed by {N}", True, "reactions"),
    ("N converts A into B",  "enzymes", "The enzyme {N} converts A into B", True, "reactions"),
    ("N hydrolyses A",       "enzymes", "{N} hydrolyses A to give B", True, "reactions"),
    ("N acts on A",          "enzymes", "{N} acts on A to give B", True, "reactions"),
    ("N breaks down A",      "enzymes", "{N} breaks down A into B", True, "reactions"),
    ("N is an enzyme that",  "enzymes", "{N} is an enzyme that acts upon A", True, "reactions"),
    ("N shown to catalyse",  "enzymes", "{N} was shown to catalyse this step", True, "reactions"),
    ("cue >80 chars away",   "enzymes", "{N} " + ("x" * 100) + " catalyses the conversion", False, "reactions"),
    ("transport plain",      "transporters", "{N} transports A across the inner membrane", True, "transports"),
    ("transport importer",   "transporters", "{N} is the importer of A", True, "transports"),
    ("B2 inhibitor span",    "enzymes", "A significantly inhibited {N} enzyme activity", False, "reactions"),
    ("B3 structural rationale", "enzymes", "add {N} as an enzyme to resolve the structural inconsistency where an inhibitor is listed without a target enzyme", False, "reactions"),
    ("different protein",    "enzymes", "Q9Z catalyses the conversion of A to B", False, "reactions"),
]
bad = []
for label, cont, tmpl, want, bucket in CASES_P:
    ev = tmpl.replace("{N}", NAME)
    got = run(cont, NAME, ev, bucket=bucket)
    flag = "" if got == want else "  << MISMATCH"
    if got != want:
        bad.append(label)
    print(f"  {'ACCEPT' if got else 'REFUSE'} (want {'ACCEPT' if want else 'REFUSE'}){flag}  {label}")
print(f"\n  T1 MULTI-TOKEN-NAME MISMATCHES: {len(bad)} / {len(CASES_P)}   {bad}")

DICT_ROWS = [
    ("dict row own evidence", "enzymes", {"protein": NAME, "evidence": f"{NAME} catalyses the step"}, True),
    ("inhibitor evidenced",   "modifiers", {"entity": NAME, "role": "inhibitor", "evidence": f"{NAME} inhibits the reaction"}, True),
    ("activator evidenced",   "modifiers", {"entity": NAME, "role": "activator", "evidence": f"{NAME} activates the pathway"}, True),
    ("cofactor evidenced",    "modifiers", {"entity": NAME, "role": "cofactor", "evidence": f"{NAME} is a required cofactor for the step"}, True),
]
bad2 = []
for label, cont, val, want in DICT_ROWS:
    got = run(cont, val, None)
    if got != want:
        bad2.append(label)
    print(f"  {'ACCEPT' if got else 'REFUSE'} (want {'ACCEPT' if want else 'REFUSE'})"
          f"{'  << MISMATCH' if got != want else ''}  {label}")
print(f"  T1b DICT-ROW MISMATCHES: {len(bad2)}  {bad2}")


print()
print("=" * 90)
print("T2 -- B3 CLEAN residual frames. The earlier Frame C/D carried an")
print("     INDEPENDENT legitimate catalysis claim, so a True there was not")
print("     necessarily wrong. These frames carry NO independent claim.")
print("=" * 90)
NEAR = ["blockade", "impairment", "disruption", "reduction", "loss", "silencing",
        "sequestration", "depletion", "ablation", "interference", "quenching"]
FRAMES = {
    "E object AFTER stem (the fix's shape)":
        "the {w} of NDM-1 activity is mediated by PSA",
    "F object BEFORE stem, mediat cue":
        "NDM-1 activity showed {w} in the PSA-mediated assay",
    "G object BEFORE stem, catalys cue":
        "NDM-1 activity showed {w}, catalysed in the presence of PSA",
    "H object >40 chars after stem":
        "the {w} of NDM-1 " + ("x" * 45) + " enzymatic activity is mediated by PSA",
    "I no object noun at all, mediat cue":
        "the {w} of NDM-1 is mediated by PSA",
}
for fname, tmpl in FRAMES.items():
    print(f"\n  Frame {fname}")
    for w in NEAR:
        span = tmpl.replace("{w}", w)
        print(f"     {w:<14} -> {_span_licenses_actor(span, 'NDM-1', 'catalysis')}")


print()
print("=" * 90)
print("T3 -- N4: is the contra seen when it sits far BEFORE the passive verb?")
print("=" * 90)
PAD = "z " * 60   # 120 chars of filler, pushes the contra out of the agent window
for label, span in [
    ("contra 120 chars before the verb",
     "PSA inhibited NDM-1 " + PAD + " and the substrate is converted to product by NDM-1"),
    ("contra 120 chars after the agent",
     "the substrate is converted to product by NDM-1 " + PAD + " which PSA inhibited"),
    ("attenuation contra far before",
     "the reduction of NDM-1 activity was total " + PAD + " and A is converted to B by NDM-1"),
]:
    print(f"  {label:<38} -> {_span_licenses_actor(span, 'NDM-1', 'catalysis')}")


print()
print("=" * 90)
print("T4 -- claim 3: REV-105 finding-2's example, both halves")
print("=" * 90)
EX = "produced by decomposition, and NDM-1 is an inhibitor target"
print(f"  the quoted example: {EX!r}")
print(f"     -> {_span_licenses_actor(EX, 'NDM-1', 'catalysis')}   (False at BASE == the")
print("        example no longer demonstrates the finding)")
print(f"     does the base contra fire on 'inhibitor'? "
      f"{bool(M._ROLE_CUE_RES['inhibition'].search(_match_fold(EX)))}")
print("  the FINDING itself (passive agent is somebody else), on a clean span:")
CLEAN = "A is produced by decomposition of PSA, and NDM-1 was quantified in the same assay"
print(f"     {CLEAN!r}\n       -> {_span_licenses_actor(CLEAN, 'NDM-1', 'catalysis')}")


print()
print("=" * 90)
print("T5 -- B10: the cancelling pair, located in the REAL corpus")
print("=" * 90)
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


found = []
seen = set()
for f in sorted(glob.glob(str(DATA / "runs/**/final_mapped.json"), recursive=True)
                + glob.glob(str(DATA / "runs_verify/**/final_mapped.json"), recursive=True)):
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
                for row in (rxn.get(cont) or []) if isinstance(rxn.get(cont), list) else []:
                    n, ev = nm_of(row), ev_of(row)
                    if not n or not ev:
                        continue
                    role = str(row.get("role") if isinstance(row, dict) else "")
                    key = f"{cont}|{bucket}|{n}|{role}|{len(ev)}"
                    if key in seen:
                        continue
                    seen.add(key)
                    folded = _match_fold(ev)
                    if "suppressor" in folded and "intermediate" in folded:
                        fam = "transport" if cont in ("transporters", "cargo", "cargo_complex") else "catalysis"
                        found.append((len(ev), cont, bucket, n, role, f,
                                      _span_licenses_actor(ev, n, fam), fam))

found.sort(reverse=True)
print(f"  corpus rows whose span carries BOTH 'suppressor' and 'intermediate': {len(found)}")
for ln, cont, bucket, n, role, f, verdict, fam in found[:12]:
    print(f"    {ln:>7} chars  [{cont}/{bucket} role={role!r}] {n!r} family={fam}"
          f"  LICENSES={verdict}")
    print(f"        {f[len(str(DATA)):]}")

print()
print("  cue occurrences inside the largest such span:")
if found:
    ln, cont, bucket, n, role, f, verdict, fam = found[0]
    d = json.loads(Path(f).read_text(encoding="utf-8"))
    span = None
    for bkt in ("reactions", "transports", "reaction_coupled_transports"):
        for rxn in (d.get("processes") or {}).get(bkt) or []:
            for c in CONT:
                for row in (rxn.get(c) or []) if isinstance(rxn.get(c), list) else []:
                    if nm_of(row) == n and len(ev_of(row)) == ln:
                        span = ev_of(row)
    if span:
        h = _match_fold(span)
        for tok in ("intermediate", "suppressor", "mediat", "suppress", "catalys",
                    "hydroly", "inhibit"):
            print(f"     {tok:<14} occurrences in folded span: {len(re.findall(tok, h))}")

print()
print("REV107_PROBE2_DONE")

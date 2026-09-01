"""REV-107: W1/W2/W4 survived the suite. Are those anchors DEAD CODE, or merely
UNTESTED? Measure the behavioural difference each one makes, rather than reason
about it. D-084 restores via C-106 primitives with sha256 + CRLF proof.

Usage::  <python> rev107_anchor_exposure.py <worktree-root>
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from c102_mutation_attack import (  # noqa: E402
    apply_mutation, crlf_count, restore_saved_bytes, sha256_of,
)

PY = "c:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
ROOT = Path(sys.argv[1]).resolve()
GUARD = ROOT / "src/t2pw/curation/apply_audit_patch.py"

PROBE = r'''
import sys
sys.path.insert(0, r"{src}")
from t2pw.curation.apply_audit_patch import apply_patch_with_policy

def seam(container, value, evidence, bucket="reactions", name=None):
    nm = name or (value if isinstance(value, str) else value.get("entity"))
    proc = {{"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "A is converted in the gut", container: []}}
    payload = {{"entities": {{"compounds": [{{"name": "A"}}, {{"name": "B"}}],
                            "proteins": [{{"name": nm}}], "protein_complexes": [],
                            "nucleic_acids": []}},
               "processes": {{bucket: [proc]}}}}
    op = {{"op": "add", "path": "/processes/%s/0/%s/-" % (bucket, container),
          "value": value, "confidence": 1.0}}
    if evidence is not None:
        op["evidence"] = evidence
    _r, rep = apply_patch_with_policy(payload, [op], stage="anchor")
    return rep["summary"]["accepted_count"] == 1

# Legitimate catalysis spans whose prose contains a word that CONTAINS an
# attenuation word or one of the six inhibition inflections. All must ACCEPT.
CASES = [
    ("F1 reductase adjacent",   "P4X",  "the reductase P4X catalyses the conversion of A to B"),
    ("F1 silencer adjacent",    "P4X",  "the silencer complex P4X catalyses the conversion of A to B"),
    ("F1 blocker adjacent",     "P4X",  "the blocker protein P4X catalyses the conversion of A to B"),
    ("F1 interferon adjacent",  "IRF3", "interferon IRF3 catalyses the conversion of A to B"),
    ("F2 silencer after actor", "P4X",  "P4X activity was measured in the silencer assay and P4X catalyses the conversion"),
    ("F2 blocker after actor",  "P4X",  "P4X activity was quantified with the blocker control while P4X catalyses A to B"),
    ("F2 reductase after actor","P4X",  "P4X activity was compared with the reductase standard while P4X catalyses A to B"),
    ("INH photoablation",       "P4X",  "after photoablation the enzyme P4X catalyses the conversion of A to B"),
    ("INH counterinterference", "P4X",  "despite counterinterference the enzyme P4X catalyses the conversion of A to B"),
    ("INH microablation",       "P4X",  "following microablation the enzyme P4X catalyses the conversion of A to B"),
    ("INH nonimpairment",       "P4X",  "given nonimpairment the enzyme P4X catalyses the conversion of A to B"),
    ("mid-word oxidoreduce",    "P4X",  "the oxidoreduced cofactor and P4X catalyses the conversion of A to B"),
    ("CONTROL hydrolase",       "P4X",  "the hydrolase P4X catalyses the conversion of A to B"),
]
for label, nm, ev in CASES:
    print("%-28s %s" % (label, "ACCEPT" if seam("enzymes", nm, ev) else "REFUSE"))
'''

VARIANTS = [
    ("baseline", None, None),
    ("W1 left anchor off _ATTENUATION_WORD_SRC",
     '_ATTENUATION_WORD_SRC = (\n    r"(?<![a-z])(?:"\n',
     '_ATTENUATION_WORD_SRC = (\n    r"(?:"  # W1\n'),
    ("W2 right anchor off _ATTENUATION_WORD_SRC",
     '    r")(?![a-z])"\n)\n',
     '    r")"  # W2\n)\n'),
    ("W4 left anchor off the six inhibition additions",
     '        r"|(?<![a-z])(?:blockades?|impair(?:s|ed|ing|ment|ments)?"\n',
     '        r"|(?:blockades?|impair(?:s|ed|ing|ment|ments)?"  # W4\n'),
]

results = {}
for label, old, new in VARIANTS:
    saved = None
    before = GUARD.read_bytes()
    if old is not None:
        saved = apply_mutation(GUARD, old, new)
    try:
        r = subprocess.run([PY, "-c", PROBE.format(src=str(ROOT / "src"))],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=str(ROOT))
        out = r.stdout + r.stderr
    finally:
        if saved is not None:
            restore_saved_bytes(GUARD, saved)
    after = GUARD.read_bytes()
    assert sha256_of(after) == sha256_of(before), f"{label}: restore not byte-exact"
    assert crlf_count(after) == crlf_count(before), f"{label}: restore changed CRLF"
    results[label] = {ln.rsplit(None, 1)[0].strip(): ln.rsplit(None, 1)[1]
                      for ln in out.strip().split("\n") if ln.strip()}
    print(f"[{label}] restore proved: bytes={len(after)} crlf={crlf_count(after)} "
          f"sha={sha256_of(after)[:16]}")

base = results["baseline"]
print()
print("=" * 96)
print(f"{'case':<28}" + "".join(f"{k.split()[0]:>10}" for k in results))
print("=" * 96)
for case in base:
    row = "".join(f"{results[k].get(case, '?'):>10}" for k in results)
    diff = any(results[k].get(case) != base[case] for k in results)
    print(f"{case:<28}{row}{'   <<< ANCHOR IS LOAD-BEARING HERE' if diff else ''}")

print()
for label in results:
    if label == "baseline":
        continue
    n = sum(1 for c in base if results[label].get(c) != base[c])
    verdict = ("DEAD CODE on these cases -- no behavioural difference"
               if n == 0 else f"LOAD-BEARING but UNTESTED: changes {n} case(s)")
    print(f"  {label:<48} {verdict}")

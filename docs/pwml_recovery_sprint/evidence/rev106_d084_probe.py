"""REV-106: measure the OLD text-mode round trip on the real target, at BASE.

Adjudicates REV-104's `bytes=78077`. Uses a COPY -- the real tracked file is
never written by this probe.
"""
from __future__ import annotations
import hashlib, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
TARGET = ROOT / "src/t2pw/bench/acceptance.py"

def census(b: bytes):
    crlf = b.count(b"\r\n")
    return dict(bytes=len(b), crlf=crlf, bare_lf=b.count(b"\n") - crlf,
                sha=hashlib.sha256(b).hexdigest()[:16])

orig = TARGET.read_bytes()
o = census(orig)
print(f"on disk        : bytes={o['bytes']}  crlf={o['crlf']}  bare_lf={o['bare_lf']}  sha={o['sha']}")

tmp = Path(tempfile.mkdtemp()) / "acceptance.py"
shutil.copyfile(TARGET, tmp)
# The exact old harness loop, content held IDENTICAL (no substitution), so the
# whole delta is line endings.
text = tmp.read_text(encoding="utf-8")
tmp.write_text(text, encoding="utf-8", newline="")
h = census(tmp.read_bytes())
print(f"harness write  : bytes={h['bytes']}  crlf={h['crlf']}  bare_lf={h['bare_lf']}  sha={h['sha']}")
print()
print(f"REV-104 reported bytes=78077")
print(f"arithmetic 79745 - 1673 = {79745 - 1673}")
print(f"REV-106 measures bytes={h['bytes']}   delta from on-disk = {o['bytes'] - h['bytes']}  (== crlf count: {o['bytes'] - h['bytes'] == o['crlf']})")
print(f"VERDICT: REV-104's 78077 is {'CORRECT' if h['bytes'] == 78077 else 'WRONG'}; measured {h['bytes']}")

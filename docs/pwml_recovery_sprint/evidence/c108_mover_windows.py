"""C-108: print the exact +/-80 WINDOW behind each corpus mover.

The attribution instrument reports which pattern matched; this prints the text
it matched in, so a reviewer can read the mover rather than trust a label. It is
separate because a 140,000-character span cannot be printed and the window can.

Usage:  <python> c108_mover_windows.py <code-root> <flat-verdicts.json> <keys-file>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CODE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(CODE / "src"))

import t2pw.curation.apply_audit_patch as M  # noqa: E402

print("code loaded from:", M.__file__, file=sys.stderr)

CATALYST_CONT = ("enzymes", "catalysts", "modifiers_or_enzymes")
TRANSPORT_CONT = ("transporters", "cargo", "cargo_complex")
ROLE_FAMILY = {
    "catalyst": "catalysis", "enzyme": "catalysis", "activator": "activation",
    "inhibitor": "inhibition", "repressor": "inhibition",
    "transporter": "transport", "cofactor": "cofactor",
}


def family_of(cont, role):
    if cont in TRANSPORT_CONT:
        return "transport"
    if cont in CATALYST_CONT:
        return "catalysis"
    r = re.sub(r"[^a-z0-9]+", "", str(role or "").lower())
    return "catalysis" if not r else ROLE_FAMILY.get(r, "other")


flat = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
keys = [ln for ln in Path(sys.argv[3]).read_text(encoding="utf-8").splitlines() if ln.strip()]

seen = set()
for key in keys:
    cont, bucket, name, role, ev = key.split("|", 4)
    fam = family_of(cont, role)
    hay = M._match_fold(ev)
    needles = M._identifying_match_tokens(name) or [M._match_fold(name)]
    sig = (name, ev[:200])
    print("=" * 92)
    print("[%s/%s role=%r fam=%s] actor=%r  span %d chars"
          % (cont, bucket, role, fam, name, len(ev)))
    shown = 0
    for needle in needles:
        esc = re.escape(needle)
        for m in re.finditer(r"(?<![a-z0-9])" + esc + r"(?![a-z0-9])", hay):
            s = max(0, m.start() - M._ACTOR_CUE_WINDOW)
            e = min(len(hay), m.end() + M._ACTOR_CUE_WINDOW)
            window = hay[s:e]
            masker = getattr(M, "_mask_actor_name", None)
            cw = masker(window, name) if masker else window
            print("  needle=%r" % needle)
            print("    window       : %r" % window)
            if cw != window:
                print("    cue window   : %r" % cw)
            cue = M._ROLE_CUE_RES.get(fam, M._ANY_ROLE_CUE_RE)
            cm = cue.search(cw)
            print("    cue in window: %r" % (cm.group(0) if cm else None))
            shown += 1
            if shown >= 4:
                break
        if shown >= 4:
            break

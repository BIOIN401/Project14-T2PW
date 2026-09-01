"""REV-108: attribute the four merge-rule-6 movers to the guard that produced them.

Do NOT infer the firing guard from the final verdict. This calls the private
objects directly, at BOTH roots, and prints which of
  - _CATALYSIS_CONTRA_RE            (the window-level contra)
  - the actor-anchored frames F1/F2 (base) / F1..F4 (tip)
  - the positive cue
matched, for the exact folded window the seam builds.

Usage: <python> rev108_attribute.py <base-root> <tip-root>
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve()
TIP = Path(sys.argv[2]).resolve()

SPANS = [
    ("P4X", "P4X is a target of the inhibitor and catalyses the conversion of A to B"),
    ("P4X", "P4X was subject to inhibitors during the assay, yet catalyses A to B"),
    ("P4X", "P4X, whose inhibitor was characterised, catalyses the conversion of A to B"),
    ("P4X", "the repressor bound P4X and the catalysis of A to B stopped"),
    ("P4X", "the inhibitor of P4X was added before the assay"),
    ("P4X", "the inhibitor protein P4X catalyses the conversion of A to B"),
    ("P", "P channeled calcium into the cytosol"),
    ("P", "P channelled calcium into the cytosol"),
    ("P", "P channels calcium into the cytosol"),
]


def load(root):
    for name in [m for m in list(sys.modules) if m.startswith("t2pw")]:
        del sys.modules[name]
    sys.path.insert(0, str(root / "src"))
    mod = importlib.import_module("t2pw.curation.apply_audit_patch")
    assert str(root).lower().replace("\\", "/") in mod.__file__.lower().replace("\\", "/")
    return mod


def report(tag, mod):
    print("=" * 100)
    print("ROOT %s  (%s)" % (tag, mod.__file__))
    print("=" * 100)
    contra = mod._CATALYSIS_CONTRA_RE
    print("contra pattern len = %d" % len(contra.pattern))
    print("inhibition CUE len = %d" % len(mod._ROLE_CUE_RES["inhibition"].pattern))
    print("cue == contra object? %s" % (contra is mod._ROLE_CUE_RES["inhibition"]))
    for actor, span in SPANS:
        hay = mod._match_fold(span)
        print("\n  actor=%r" % actor)
        print("  folded=%r" % hay)
        cm = contra.search(hay)
        print("    _CATALYSIS_CONTRA_RE  : %s" % (repr(cm.group(0)) if cm else "NO MATCH"))
        cat = mod._ROLE_CUE_RES["catalysis"].search(hay)
        print("    catalysis CUE (raw)   : %s" % (repr(cat.group(0)) if cat else "NO MATCH"))
        tr = mod._ROLE_CUE_RES["transport"].search(hay)
        print("    transport CUE (raw)   : %s" % (repr(tr.group(0)) if tr else "NO MATCH"))
        masked = None
        if hasattr(mod, "_mask_actor_name"):
            masked = mod._mask_actor_name(hay, actor)
            print("    masked window         : %r" % masked)
        print("    seam catalysis verdict: %s" % mod._span_licenses_actor(span, actor, "catalysis"))
        print("    seam transport verdict: %s" % mod._span_licenses_actor(span, actor, "transport"))
        # actor-anchored attenuation frames, reconstructed exactly as the seam does
        needles = mod._identifying_match_tokens(actor) or [mod._match_fold(actor)]
        for needle in needles:
            escaped = re.escape(needle)
            if hasattr(mod, "_ATTENUATION_AGENT_NOUN_SRC"):
                f3 = re.compile(
                    mod._ATTENUATION_AGENT_NOUN_SRC + r"\s+" + mod._ATTENUATION_TARGET_HEAD_SRC
                    + r"(?:\s+" + mod._PASSIVE_AGENT_MODIFIERS_SRC + r"){0,4}\s+"
                    + escaped + r"(?![a-z0-9])")
                f4 = re.compile(
                    r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
                    r"(?:\s+" + mod._ATTENUATION_AGENT_ADJ_SRC + r"){0,"
                    + str(mod._ATTENUATION_AGENT_MAX_ADJ) + r"}\s+"
                    + mod._ATTENUATION_AGENT_NOUN_SRC + r"(?![a-z])")
                m3, m4 = f3.search(hay), f4.search(hay)
                print("    F3 target-directed    : %s" % (repr(m3.group(0)) if m3 else "NO MATCH"))
                print("    F4 head-final compound: %s" % (repr(m4.group(0)) if m4 else "NO MATCH"))


base = load(BASE)
report("BASE", base)
tip = load(TIP)
report("TIP", tip)

print()
print("=" * 100)
print("CUE BYTE-IDENTITY CHECK (member (d)'s central claim)")
print("=" * 100)
b = base._ROLE_CUE_RES["inhibition"].pattern
t = tip._ROLE_CUE_RES["inhibition"].pattern
print("base inhibition cue len=%d  tip len=%d  identical=%s" % (len(b), len(t), b == t))
ab = base._ANY_ROLE_CUE_RE.pattern
at = tip._ANY_ROLE_CUE_RE.pattern
print("_ANY_ROLE_CUE_RE base len=%d tip len=%d identical=%s" % (len(ab), len(at), ab == at))

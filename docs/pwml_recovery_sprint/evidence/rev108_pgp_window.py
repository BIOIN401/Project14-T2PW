"""REV-108: read the ONE newly-admitted corpus row from its own span.

B5 -- "a row admitted here is worth more scrutiny than ten refused". This does
not take the author's account of the Pgp row. It re-locates the actor in the
140k-character span, prints every window that licenses at the tip, and shows
exactly what the BASE contra matched inside that same window, so the question
"was the base refusal correct?" is answered from the text.

Usage: <python> rev108_pgp_window.py <base-root> <tip-root> <data-root>
"""
from __future__ import annotations

import glob
import importlib
import json
import re
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve()
TIP = Path(sys.argv[2]).resolve()
DATA = Path(sys.argv[3]).resolve()

ACTOR = "Pgp phosphatase complex"
CONT = ("enzymes", "modifiers", "modifiers_or_enzymes", "catalysts",
        "transporters", "cargo", "cargo_complex")
EVK = ("evidence", "evidence_quote", "source_evidence", "source_text")
NAMEK = ("entity", "protein", "protein_name", "protein_complex", "enzyme",
         "modifier", "name")


def load(root):
    for name in [m for m in list(sys.modules) if m.startswith("t2pw")]:
        del sys.modules[name]
    sys.path.insert(0, str(root / "src"))
    return importlib.import_module("t2pw.curation.apply_audit_patch")


def nm_of(r):
    if isinstance(r, str):
        return r.strip()
    if not isinstance(r, dict):
        return ""
    for f in NAMEK:
        v = r.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def find_span():
    """Re-derive the row's evidence span from the DATA root, not from a report."""
    for path in glob.glob(str(DATA / "**" / "*.json"), recursive=True):
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        stack = [doc]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for cont in CONT:
                    rows = node.get(cont)
                    if not isinstance(rows, list):
                        continue
                    for row in rows:
                        if nm_of(row) != ACTOR:
                            continue
                        for k in EVK:
                            v = (row.get(k) if isinstance(row, dict) else None) or node.get(k)
                            if isinstance(v, str) and v.strip():
                                return path, v
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    return None, None


path, span = find_span()
print("row source file :", path)
print("span length     :", len(span) if span else None)
if not span:
    sys.exit("could not re-derive the span from the DATA root")

base = load(BASE)
tip = load(TIP)

hay = tip._match_fold(span)
print("folded length   :", len(hay))
needles = tip._identifying_match_tokens(ACTOR)
print("needles         :", needles)
print("base verdict    :", base._span_licenses_actor(span, ACTOR, "catalysis"))
print("tip  verdict    :", tip._span_licenses_actor(span, ACTOR, "catalysis"))

W = tip._ACTOR_CUE_WINDOW
shown = 0
for needle in needles:
    escaped = re.escape(needle)
    for m in re.finditer(r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])", hay):
        s = max(0, m.start() - W)
        e = min(len(hay), m.end() + W)
        window = hay[s:e]
        masked = tip._mask_actor_name(window, ACTOR)
        cue = tip._ROLE_CUE_RES["catalysis"].search(masked)
        base_cue = base._ROLE_CUE_RES["catalysis"].search(window)
        base_contra = base._CATALYSIS_CONTRA_RE.search(window)
        tip_contra = tip._CATALYSIS_CONTRA_RE.search(window)
        licenses_at_tip = bool(cue) and not tip_contra
        if not (base_cue or cue):
            continue
        shown += 1
        print("\n" + "=" * 96)
        print("needle=%r  window offset %d..%d" % (needle, s, e))
        print("=" * 96)
        print("WINDOW (unmasked):")
        print("   ", repr(window))
        print("MASKED (what the tip cue search sees):")
        print("   ", repr(masked))
        print("  base catalysis cue  :", repr(base_cue.group(0)) if base_cue else "NO MATCH")
        print("  tip  catalysis cue  :", repr(cue.group(0)) if cue else "NO MATCH")
        print("  BASE contra match   :", repr(base_contra.group(0)) if base_contra else "NO MATCH")
        if base_contra:
            cs = max(0, base_contra.start() - 70)
            ce = min(len(window), base_contra.end() + 70)
            print("    base contra context:", repr(window[cs:ce]))
            print("    distance from needle match (chars in window):",
                  abs(base_contra.start() - (m.start() - s)))
        print("  TIP  contra match   :", repr(tip_contra.group(0)) if tip_contra else "NO MATCH")
        print("  -> licenses at tip  :", licenses_at_tip)
print("\nwindows printed:", shown)

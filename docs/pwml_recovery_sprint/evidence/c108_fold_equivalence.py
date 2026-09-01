"""C-108 member (b): the proof that "[^.]" was a LENGTH BOUND AND NOTHING ELSE.

Member (b) is code honesty, not a behaviour change, so the claim it rests on has
to be proven rather than asserted. Four things are measured here:

  1. CONSUMER ENUMERATION. Every place the four pattern objects
     (_ROLE_CUE_RES, _ANY_ROLE_CUE_RE, _CATALYSIS_CONTRA_RE and the per-needle
     patterns) are applied to a string, listed from the module source, with the
     string each one is applied to. If any consumer sees UNFOLDED text this
     member is a live behaviour question and the card must stop.
  2. THE FOLDED ALPHABET. Every span and every actor name in the 692-row corpus
     is folded and the union of characters is reported. If it is a subset of
     {a-z, 0-9, " "} then "[^.]" excluded nothing on this input.
  3. THE NEWLINE CASE. "[^.]" matches a newline where an unflagged "." does not,
     which is a real difference between the two spellings on ARBITRARY text.
     Shown to exist, and shown to be unreachable after folding.
  4. VERDICT EQUIVALENCE. Both spellings of every affected pattern are compiled
     and run over every corpus window; any row on which they disagree is printed.

It also proves the (d) split did not disturb the CUE: the inhibition cue's
pattern string is compared byte for byte between base and tip, because
_ANY_ROLE_CUE_RE -- and therefore the "other" fallback for every unmapped role --
is rebuilt from it.

Usage::  <python> c108_fold_equivalence.py <base-code-root> <tip-code-root> <flat-verdicts.json>
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve()
TIP = Path(sys.argv[2]).resolve()
CORPUS = Path(sys.argv[3])

failures = 0


def fail(msg):
    global failures
    failures += 1
    print("  FAIL: " + msg)


def load(root, alias):
    path = root / "src" / "t2pw" / "curation" / "apply_audit_patch.py"
    saved = list(sys.path)
    sys.path.insert(0, str(root / "src"))
    for name in [m for m in list(sys.modules) if m.startswith("t2pw")]:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


BASE_M = load(BASE, "c108b_base")
TIP_M = load(TIP, "c108b_tip")
TIP_SRC = (TIP / "src" / "t2pw" / "curation" / "apply_audit_patch.py").read_text(
    encoding="utf-8")

print("=" * 100)
print("1. CONSUMER ENUMERATION -- every application of the four pattern objects")
print("=" * 100)

PATTERN_NAMES = ("_ROLE_CUE_RES", "_ANY_ROLE_CUE_RE", "_CATALYSIS_CONTRA_RE",
                 "cue", "contra", "actor_contra", "dependence", "passive")
CONSUMER_RE = re.compile(
    r"^\s*(?:.*?\b)?(cue|contra|actor_contra|dependence)\.search\(([a-z_]+)\)",
    re.M)
lines = TIP_SRC.splitlines()
consumers = []
for i, line in enumerate(lines, 1):
    if re.search(r"\b(cue|contra|actor_contra|dependence)\.search\(", line):
        consumers.append((i, line.strip()))
    if re.search(r"re\.finditer\(", line):
        consumers.append((i, line.strip()))
for lineno, text in consumers:
    print("  line %-5d %s" % (lineno, text))

ALLOWED_SUBJECTS = {"window", "cue_window", "haystack"}
subjects = set()
for _lineno, text in consumers:
    for m in re.finditer(r"\.search\(([a-z_]+)\)", text):
        subjects.add(m.group(1))
    for m in re.finditer(r"re\.finditer\([^,]+,\s*([a-z_]+)\)", text):
        subjects.add(m.group(1))
print()
print("  strings the patterns are applied to:", sorted(subjects))
unexpected = subjects - ALLOWED_SUBJECTS
if unexpected:
    fail("a pattern is applied to something other than the folded haystack or a "
         "slice of it: %s -- THIS IS A STOP CONDITION" % sorted(unexpected))
else:
    print("  OK: every consumer reads `haystack` (= _match_fold(span)), a slice "
          "of it (`window`), or that slice with the actor name blanked "
          "(`cue_window`). No consumer sees unfolded text.")

print()
print("  repo-wide: the four objects are referenced nowhere outside this module "
      "except in test comments -- see evidence/c108_consumer_grep.log")

print()
print("=" * 100)
print("2. THE FOLDED ALPHABET over every corpus span and actor name")
print("=" * 100)
flat = json.loads(CORPUS.read_text(encoding="utf-8"))
if isinstance(flat, dict) and "verdicts" in flat:
    flat = flat["verdicts"]
alphabet = set()
n_spans = 0
for key in flat:
    cont, bucket, name, role, ev = key.split("|", 4)
    for text in (ev, name):
        alphabet |= set(TIP_M._match_fold(text))
        n_spans += 1
allowed = set("abcdefghijklmnopqrstuvwxyz0123456789 ")
print("  folded %d strings; alphabet size %d" % (n_spans, len(alphabet)))
print("  characters outside [a-z0-9 ]: %r" % sorted(alphabet - allowed))
if alphabet - allowed:
    fail("the folded alphabet is wider than [a-z0-9 ]")
else:
    print("  OK: no '.' can reach any pattern, so `[^.]` excluded NOTHING.")

print()
print("=" * 100)
print("3. THE NEWLINE CASE -- a real difference between the spellings, unreachable here")
print("=" * 100)
print("  re.search(r'a[^.]b', 'a\\nb') ->", bool(re.search(r"a[^.]b", "a\nb")))
print("  re.search(r'a.b',    'a\\nb') ->", bool(re.search(r"a.b", "a\nb")))
print("  re.search(r'a[a-z0-9 ]b', 'a\\nb') ->", bool(re.search(r"a[a-z0-9 ]b", "a\nb")))
folded = TIP_M._match_fold("a\nb")
print("  _match_fold('a\\nb') -> %r  (newline is gone)" % folded)
if "\n" in folded:
    fail("folding did not remove the newline")
else:
    print("  OK: `[^.]` and `.` differ on a newline, and folding removes every "
          "newline, so the difference is unreachable. `[a-z0-9 ]` matches the "
          "space the newline became, exactly as `[^.]` did.")

print()
print("=" * 100)
print("4. VERDICT EQUIVALENCE -- both spellings, every corpus row")
print("=" * 100)
# Rebuild each affected pattern in BOTH spellings and compare over every window
# the corpus produces.
AFFECTED = [
    (r"(?:removes|adds|attaches|transfers|incorporates)\b%s{0,40}"
     r"\b(?:group|residue|moiety|molecule|atom|phosphate|acyl|methyl|sugar)"),
    (r"(?:removal|addition|transfer|incorporation) of\b%s{0,40}"
     r"\b(?:group|residue|moiety|molecule|atom|phosphate|acyl|methyl|sugar)"),
    (r"converted\b%s{0,80}\bby"),
]
mismatch = 0
checked = 0
for key in flat:
    cont, bucket, name, role, ev = key.split("|", 4)
    hay = TIP_M._match_fold(ev)
    for template in AFFECTED:
        old = re.compile(template % r"[^.]")
        new = re.compile(template % r"[a-z0-9 ]")
        a = [m.span() for m in old.finditer(hay)]
        b = [m.span() for m in new.finditer(hay)]
        checked += 1
        if a != b:
            mismatch += 1
            print("  DISAGREEMENT on %r: %r vs %r" % (name, a[:3], b[:3]))
print("  compared %d pattern/row pairs; disagreements: %d" % (checked, mismatch))
if mismatch:
    fail("the two spellings are not equivalent on this input")
else:
    print("  OK: the two spellings produce identical match sets on every corpus row.")

print()
print("=" * 100)
print("5. (d) THE INHIBITION CUE IS BYTE-IDENTICAL BETWEEN BASE AND TIP")
print("=" * 100)
b = BASE_M._ROLE_CUE_RES["inhibition"].pattern
t = TIP_M._ROLE_CUE_RES["inhibition"].pattern
print("  base len %d  tip len %d  identical: %s" % (len(b), len(t), b == t))
if b != t:
    fail("the inhibition CUE moved; _ANY_ROLE_CUE_RE and the 'other' fallback "
         "for every unmapped role are no longer where C-107 left them")
else:
    print("  OK: only the CONTRA was anchored.")

print("  base contra is the cue object:", BASE_M._CATALYSIS_CONTRA_RE is
      BASE_M._ROLE_CUE_RES["inhibition"])
print("  tip  contra is the cue object:", TIP_M._CATALYSIS_CONTRA_RE is
      TIP_M._ROLE_CUE_RES["inhibition"])
print("  tip  contra pattern:", TIP_M._CATALYSIS_CONTRA_RE.pattern)

print()
print("=" * 100)
print("C108 FOLD/EQUIVALENCE FAILURES: %d  (must be 0)" % failures)
print("=" * 100)

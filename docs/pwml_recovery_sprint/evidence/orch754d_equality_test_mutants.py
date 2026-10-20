"""ORCH-754d evidence: does the equality test fire when the production copy drifts?

Each mutant is name_keys.py with one rule changed, loaded in memory from a temp file (the
source tree is never edited). The equality test module's ``N`` is pointed at the mutant and
its three test functions are called directly. Every mutant must make at least one fail,
except an EQUIVALENT mutant: one whose changed line no input can reach. Each of those is
listed with the fact that makes it unreachable, and that fact is checked here, not assumed.

Usage::

    PYTHONPATH=src python docs/pwml_recovery_sprint/evidence/orch754d_equality_test_mutants.py
"""
import importlib.util
import sys
import tempfile
import unicodedata
from pathlib import Path

W = Path(__file__).resolve().parents[3]
SRC = (W / "src/t2pw/pipeline/name_keys.py").read_text(encoding="utf-8")
NBSP, SUP_PLUS, SUP_MINUS = chr(0x00A0), chr(0x207A), chr(0x207B)
MUTANTS = {
    "pyrophosphate rule dropped": ('return key.replace("pyrophosphate", "diphosphate")', "return key"),
    "kappa dropped from _GREEK": ('"κ": "kappa", "Κ": "kappa",', ""),
    "unicode dash map dropped": ("if char in _DASHES:", "if False:"),
    "NBSP branch dropped": (f'elif char == "{NBSP}":', "elif False:"),
    "(abbrev.) note kept": ("|complex|abbrev\\.?)", "|complex)"),
    "-ic acid word boundary dropped": ('re.compile(r"ic acid\\b")', 're.compile(r"ic acid")'),
    "superscript plus not a charge": (f"[{SUP_PLUS}{SUP_MINUS}+]", f"[{SUP_MINUS}+]"),
    "text_key unpadded": ('return f" {strict_key(text)} "', "return strict_key(text)"),
    "mentions not token-bounded": ('f" {k} " in keyed_text', "k in keyed_text"),
    "casefold -> lower": ("text = text.casefold()\n", "text = text.lower()\n"),
}
#: Mutants no input can distinguish, and why: canonical_text applies NFKC before the line runs.
EQUIVALENT = {
    "NBSP branch dropped": ("NFKC maps U+00A0 to U+0020 before the branch is tested",
                            unicodedata.normalize("NFKC", NBSP) == " "),
    "superscript plus not a charge": ("NFKC maps U+207A to '+' before the charge regex runs",
                                      unicodedata.normalize("NFKC", SUP_PLUS) == "+"),
}

spec = importlib.util.spec_from_file_location("eqtest", W / "tests/test_name_keys_match_evaluation_matcher.py")
eqtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eqtest)
tests = [getattr(eqtest, n) for n in dir(eqtest) if n.startswith("test_")]

tmp = Path(tempfile.mkdtemp(prefix="orch754d_mut_"))
bad = []
for i, (label, (old, new)) in enumerate(MUTANTS.items()):
    assert SRC.count(old) == 1, (label, SRC.count(old))
    path = tmp / f"name_keys_mutant_{i}.py"
    path.write_text(SRC.replace(old, new), encoding="utf-8")
    ms = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(mod)
    eqtest.N = mod
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError:
            failed.append(t.__name__)
    if failed:
        print(f"{label:34s} KILLED by: {', '.join(failed)}")
        if label in EQUIVALENT:
            bad.append(f"{label}: declared equivalent but killed")
    elif label in EQUIVALENT and EQUIVALENT[label][1]:
        print(f"{label:34s} survived, EQUIVALENT (checked): {EQUIVALENT[label][0]}")
    else:
        print(f"{label:34s} SURVIVED")
        bad.append(label)

import t2pw.pipeline.name_keys as real  # noqa: E402
eqtest.N = real
for t in tests:
    t()
print("unmutated copy: all", len(tests), "tests pass")
print("VERDICT:", "EVERY NON-EQUIVALENT MUTANT KILLED" if not bad else f"PROBLEMS {bad}")
sys.exit(1 if bad else 0)

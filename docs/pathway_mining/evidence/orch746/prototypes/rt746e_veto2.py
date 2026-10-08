"""SCRATCH-ONLY prototype, variant 2: rt746e_veto plus a concessive veto (although / though /
despite / in spite of anywhere in the sentence), subtract-only like the rest of _NO_EFFECT."""
import re
import rt746e_veto  # noqa: F401  (applies the three vetoes first)
from t2pw.mining import species as S

S._NO_EFFECT = re.compile(S._NO_EFFECT.pattern + "|" + r"(?:\b(?:although|though|despite|in spite of)\b)")
print("PLUGIN ACTIVE: rt746e_veto2 (concessive veto added), module=%s" % S.__file__)

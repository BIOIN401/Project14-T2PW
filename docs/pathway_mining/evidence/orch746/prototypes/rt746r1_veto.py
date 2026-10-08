"""SCRATCH-ONLY prototype (never committed, never written into a worktree). Subtract-only: more
no-effect vetoes appended to species._NO_EFFECT, which grade_sentences uses only as a veto.
A. non-significance outside the shipped list; B. a delay stated by time points;
C. no difference: a wild-type / control genotype named in a sentence with no "than"."""
import os, re
from t2pw.mining import species as S

EXTRA = [
    # A
    r"\bsignificance\b", r"\btrend(?:s|ed)?\b", r"\btendenc", r"\(\s*ns\s*\)", r"\bn\.\s?s\b",
    r"\bp[- ]?values?\s*(?:of|=|>|\u2265|above|greater than)", r"\bno statistically\b",
    r"\bwithin the (?:normal )?range\b",
    # B
    r"\btill\b", r"\bearlier than\b", r"\b(?:for|during|within) the first\b", r"\blater\b",
    r"\bthereafter\b", r"\bonly (?:after|at|from|by)\b", r"\bdelay", r"\blag\b",
    r"\bbefore (?:the |one |two |three |four |five |six )",
    r"\b(?:but|;)[^.;]*?\b(?:present|detected|detectable|appeared|accumulated)\b",
    # C
    r"^(?!.*\bthan\b).*\b(?:wild[- ]?type|col(?:-0)?|wt|ws|ler|controls?|parental|parent)\b",
    r"\bany (?:of the )?genotypes?\b", r"\ball (?:the )?genotypes\b", r"\bthe same was true\b", r"\bnor\b",
]
if os.environ.get("RT_CC"):
    EXTRA += [r"^(?:while|whilst)\b", r"\bwhilst\b", r"\beven if\b", r"\bnotwithstanding\b", r"\bregardless of\b"]
S._NO_EFFECT = re.compile(S._NO_EFFECT.pattern + "|" + "|".join("(?:%s)" % p for p in EXTRA))
print("PLUGIN ACTIVE: rt746r1_veto (RT_CC=%s), module=%s" % (bool(os.environ.get("RT_CC")), S.__file__))

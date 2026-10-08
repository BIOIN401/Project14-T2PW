"""SCRATCH-ONLY prototype (never committed, never written into a worktree).
Subtract-only: three more no-effect vetoes in species._NO_EFFECT (a veto in grade_sentences only).
1. a non-significant change: not (statistically) significant, non-significant, insignificant, n.s.,
   P > 0.05 / P = 0.05-0.99, tended to;
2. a delay: until anywhere in the sentence; before / prior to followed by a number (a time);
3. no difference between the genotypes: the loss-of-function genotype coordinated by and / or with
   the wild type, Col-0 or WT, or alike."""
import re
from t2pw.mining import species as S

EXTRA = (
    r"\bnot (?:statistically )?significant\b", r"\bnot statistically\b", r"\b(?:non-?|in)significant",
    r"\bn\.\s?s\.", r"\bp\s*(?:>|\u2265)\s*0?\.\d", r"\bp\s*=\s*0?\.(?:0[5-9]|[1-9])", r"\btend(?:ed|s)? to\b",
    r"\buntil\b", r"\b(?:before|prior to)\s+\d",
    r"\b(?:mutants?|lines?|knock-?outs?|plants?)\s+(?:and|or)\s+(?:also\s+)?(?:in\s+)?(?:the\s+)?"
    r"(?:wild[- ]?type|col-0|wt)\b",
    r"\balike\b",
)
S._NO_EFFECT = re.compile(S._NO_EFFECT.pattern + "|" + "|".join("(?:%s)" % p for p in EXTRA))
print("PLUGIN ACTIVE: rt746e_veto, module=%s" % S.__file__)

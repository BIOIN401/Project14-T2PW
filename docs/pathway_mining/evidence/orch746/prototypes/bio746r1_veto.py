"""SCRATCH-ONLY prototype (rev746e-r1-bio; never committed, never written into a worktree).
Subtract-only: more entries in species._NO_EFFECT, which grade_sentences uses only as an in-planta veto.
A = the non-significance class written with the noun or a negated adjective ("did not reach (statistical)
    significance", "failed to reach significance", "there was no statistically significant difference",
    "not to a statistically significant extent");
B = the delay class written as a later appearance ("but accumulated at later time points", "became
    detectable only at 2 h", "reached ... by 2 h", "delayed");
E = a time course: two time points named in the sentence (an absence at one of them is a delay or a transient);
C = (optional) an absence restricted to an early window ("during the first 30 min", "earliest time points");
D = (optional) the shared-with-every-genotype form ("in all genotypes", "either genotype") and
    "within the range of biological variation"."""
import os, re
from t2pw.mining import species as S
T = r"\d+(?:\.\d+)?\s*(?:min|minutes?|h|hrs?|hours?)\b"
GROUPS = {
    "A": (r"\bsignificance\b", r"\btrends?\b", r"\btendenc", r"\b(?:not|no|without|never|nor)\b[^.;]{0,40}?\bsignificant\b"),
    "B": (r"\b(?:later|thereafter|subsequently|afterwards|onwards?)\b", r"\bonly (?:at|after|from|by|when|once)\b",
          r"\bbecame (?:detectable|measurable|apparent|visible)\b",
          r"\bby \d+(?:\.\d+)?\s*(?:min|minutes?|h|hrs?|hours?|d|days?)\b", r"\bdelay\w*"),
    "E": (r"\b" + T + r"[^;]*?\b" + T,),
    "C": (r"\b(?:first|earliest|early|initial)\b[^.;]{0,15}?\b(?:min|minutes|h|hours?|time points?)\b",),
    "D": (r"\b(?:all|each|either|any) (?:of the )?(?:genotypes?|lines)\b", r"\bbiological variation\b",
          r"\bwithin (?:the )?range of\b"),
}
ON = os.environ.get("BIOVETO", "ABECD")
EXTRA = [p for g in ON for p in GROUPS[g]]
S._NO_EFFECT = re.compile(S._NO_EFFECT.pattern + "|" + "|".join("(?:%s)" % p for p in EXTRA))
print("PLUGIN ACTIVE: bio746r1_veto groups=%s module=%s" % (ON, S.__file__))

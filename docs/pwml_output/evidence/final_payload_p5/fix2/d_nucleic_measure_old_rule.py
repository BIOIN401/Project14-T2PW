"""Counterfactual for R2-2: the same 152-run nucleic measure with the PREVIOUS protein-coding -> DNA rule (only a protein word in a
sentence naming the row blocks it; no text, or a row the text never names, passes). Everything else is the current code.

Usage: python d_nucleic_measure_old_rule.py OUT.json [CACHE_DIR]   (PYTHONPATH=C:/t/fp/src)
"""
import re
import runpy
import sys
from pathlib import Path

from t2pw.service import final_payload_nucleic as fn


def old_rule(text, row):
    from t2pw.pwml.writer import _PROTEIN_TOKENS
    pats = [fn._pattern(n) for n in fn._names(row)]
    for _, sent in fn._sentences(text if isinstance(text, str) else ""):
        if any(p.search(sent) for p in pats) and any(t in _PROTEIN_TOKENS for t in re.findall(r"[a-z0-9]+", sent.casefold())):
            return "protein_context"
    return None


fn._dna_context_code = old_rule
here = Path(__file__).resolve().parent
runpy.run_path(str(here.parent / "nucleicD" / "d_nucleic_measure.py"), run_name="__main__")

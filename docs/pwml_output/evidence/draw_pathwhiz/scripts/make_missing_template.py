"""Harness-only probe file: a copy of a PWML whose membrane-visualizations use a template id PathWhiz does not have.
usage: make_missing_template.py IN.pwml OUT.pwml [TEMPLATE_ID]"""
import re
import sys

tid = sys.argv[3] if len(sys.argv) > 3 else "99999"
s = open(sys.argv[1], encoding="utf-8").read()
i, j = s.index("<membrane-visualizations>"), s.index("</membrane-visualizations>")
block, n = re.subn(r'(<visualization-template-id type="integer">)\d+(</visualization-template-id>)', r"\g<1>%s\g<2>" % tid, s[i:j])
open(sys.argv[2], "w", encoding="utf-8", newline="").write(s[:i] + block + s[j:])
print("membrane template ids replaced:", n)
